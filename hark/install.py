"""`hark install` and `hark uninstall`: the Omarchy desktop integration.

install puts three things in place from the repo's omarchy/ folder:

  - the bar widget, copied to ~/.config/omarchy/plugins/hark/
  - the keys, as one marked block at the end of ~/.config/hypr/bindings.lua
    (the rest of the file is never rewritten)
  - the hark.service systemd user unit

and then, unless told not to, starts the unit, turns the widget on and reloads
Hyprland. Every file it is about to change is first copied to
~/.local/state/hark/backups/<time>/. Running it again changes nothing that is
already right. uninstall removes exactly those three things (and turns them off);
the backups stay.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "omarchy"
PLUGIN_ID = "hark"
UNIT = "hark.service"
BEGIN = "-- >>> hark keys, written by `hark install`; `hark uninstall` removes this block"
END = "-- <<< hark keys"
UNIT_MARK = "# Written by `hark install`"
# The hark path goes into a Lua string, a shell command and an ExecStart line.
SAFE_PATH = re.compile(r"/[A-Za-z0-9_./+-]+")

Runner = Callable[[list[str]], subprocess.CompletedProcess]


class InstallError(Exception):
    pass


def run_command(argv: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, 127, "", f"{argv[0]}: command not found")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, "", f"{argv[0]}: timed out")


@dataclass
class Paths:
    home: Path

    @property
    def bindings(self) -> Path:
        return self.home / ".config/hypr/bindings.lua"

    @property
    def plugin(self) -> Path:
        return self.home / ".config/omarchy/plugins" / PLUGIN_ID

    @property
    def unit(self) -> Path:
        return self.home / ".config/systemd/user" / UNIT

    @property
    def shell_json(self) -> Path:
        return self.home / ".config/omarchy/shell.json"

    @property
    def backups(self) -> Path:
        return self.home / ".local/state/hark/backups"


@dataclass
class Report:
    changed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class Backup:
    """One folder per run, made only when something is backed up. Kept outside the
    config folders: a copy of the plugin inside plugins/ would load as a plugin."""

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.dir: Path | None = None

    def save(self, path: Path) -> Path:
        if self.dir is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.dir = self.paths.backups / stamp
            n = 1
            while self.dir.exists():
                self.dir = self.paths.backups / f"{stamp}-{n}"
                n += 1
        try:
            rel = path.relative_to(self.paths.home)
        except ValueError:
            rel = Path(path.name)
        dest = self.dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, dest, symlinks=True)
        else:
            shutil.copy2(path, dest)
        return dest


def hark_binary() -> Path:
    """The hark entry point of the environment running this: the keys and the unit
    call it directly, which starts faster than `uv run`."""
    exe = Path(sys.prefix) / "bin" / "hark"
    if exe.exists():
        return exe
    found = shutil.which("hark")
    if found:
        return Path(found)
    raise InstallError("cannot find the hark command; run this as: uv run hark install")


def render(template: str, hark: Path) -> str:
    if not SAFE_PATH.fullmatch(str(hark)):
        raise InstallError(f"hark lives at {hark}, a path with characters the keys and the unit cannot hold; "
                           "move the checkout to a plain path")
    return template.replace("@HARK@", str(hark))


def bindings_block(hark: Path) -> str:
    return f"{BEGIN}\n{render((ASSETS / 'hypr/bindings.lua').read_text(), hark)}{END}\n"


def unit_text(hark: Path) -> str:
    return render((ASSETS / "systemd" / UNIT).read_text(), hark)


# --- files ---------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    """Replace path in one step, keeping its permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _find_block(text: str, path: Path) -> tuple[int, int] | None:
    """Character span of the marked block, end marker line included."""
    start = _line_at(text, BEGIN)
    if start is None:
        return None
    end = _line_at(text, END, start)
    if end is None:
        raise InstallError(f"{path} has the hark start marker but no end marker ({END}); fix it by hand")
    stop = text.find("\n", end)
    return start, len(text) if stop < 0 else stop + 1


def _line_at(text: str, line: str, start: int = 0) -> int | None:
    for m in re.finditer(re.escape(line) + r"[ \t]*(?:\n|$)", text[start:], re.MULTILINE):
        at = start + m.start()
        if at == 0 or text[at - 1] == "\n":
            return at
    return None


def install_bindings(paths: Paths, block: str, backup: Backup) -> bool:
    path = paths.bindings
    text = path.read_text() if path.exists() else ""
    span = _find_block(text, path)
    if span is not None:
        if text[span[0]:span[1]] == block:
            return False
        new = text[:span[0]] + block + text[span[1]:]
    else:
        new = text
        if new and not new.endswith("\n"):
            new += "\n"
        new += ("\n" if new else "") + block
    if path.exists():
        backup.save(path)
    _write(path, new)
    return True


def remove_bindings(paths: Paths, backup: Backup) -> bool:
    path = paths.bindings
    if not path.exists():
        return False
    text = path.read_text()
    span = _find_block(text, path)
    if span is None:
        return False
    before = text[:span[0]]
    if before.endswith("\n\n"):  # the blank line install put in front of the block
        before = before[:-1]
    backup.save(path)
    _write(path, before + text[span[1]:])
    return True


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def install_plugin(paths: Paths, backup: Backup) -> bool:
    src, dest = ASSETS / "plugin", paths.plugin
    if dest.is_dir() and _tree(dest) == _tree(src):
        return False
    if dest.exists() or dest.is_symlink():
        backup.save(dest)
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return True


def is_our_plugin(path: Path) -> bool:
    try:
        return json.loads((path / "manifest.json").read_text()).get("id") == PLUGIN_ID
    except (OSError, ValueError, AttributeError):
        return False


def install_unit(paths: Paths, text: str, backup: Backup) -> bool:
    path = paths.unit
    if path.exists() and path.read_text() == text:
        return False
    if path.exists():
        backup.save(path)
    _write(path, text)
    return True


def is_our_unit(path: Path) -> bool:
    try:
        return path.read_text().startswith(UNIT_MARK)
    except OSError:
        return False


def plugin_enabled(paths: Paths) -> bool:
    """The widget has a place in shell.json (any bar section, or plugins[])."""
    try:
        data = json.loads(paths.shell_json.read_text())
    except (OSError, ValueError):
        return False

    def walk(node: object) -> bool:
        if isinstance(node, dict):
            return node.get("id") == PLUGIN_ID or any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    return walk(data.get("bar", {}).get("layout", {})) or walk(data.get("plugins", []))


# --- install and uninstall -------------------------------------------------------


def install(home: Path, hark: Path, *, activate: bool = True, run: Runner = run_command) -> Report:
    paths, report, backup = Paths(home), Report(), Backup(Paths(home))
    block, unit = bindings_block(hark), unit_text(hark)  # both checked before anything is written
    if install_plugin(paths, backup):
        report.changed.append(f"bar widget copied to {paths.plugin}")
    unit_changed = install_unit(paths, unit, backup)
    if unit_changed:
        report.changed.append(f"service written to {paths.unit}")
    keys_changed = install_bindings(paths, block, backup)
    if keys_changed:
        report.changed.append(f"keys added to {paths.bindings}")
    if activate:
        _step(report, run, ["systemctl", "--user", "daemon-reload"])
        was_active = run(["systemctl", "--user", "is-active", "--quiet", UNIT]).returncode == 0
        _step(report, run, ["systemctl", "--user", "enable", "--now", UNIT])
        if was_active and unit_changed:
            _step(report, run, ["systemctl", "--user", "restart", UNIT])
        _step(report, run, ["omarchy-shell", "shell", "rescanPlugins"])
        if not plugin_enabled(paths):
            if paths.shell_json.exists():
                backup.save(paths.shell_json)
            if _step(report, run, ["omarchy", "plugin", "enable", PLUGIN_ID]):
                report.changed.append("bar widget turned on")
        if keys_changed:
            _reload_hyprland(report, run)
    if backup.dir is not None:
        report.changed.append(f"backups in {backup.dir}")
    return report


def uninstall(home: Path, *, activate: bool = True, run: Runner = run_command) -> Report:
    paths, report, backup = Paths(home), Report(), Backup(Paths(home))
    if paths.unit.exists():
        if is_our_unit(paths.unit):
            if activate:
                _step(report, run, ["systemctl", "--user", "disable", "--now", UNIT])
            paths.unit.unlink()
            report.changed.append(f"removed {paths.unit}")
            if activate:
                _step(report, run, ["systemctl", "--user", "daemon-reload"])
        else:
            report.problems.append(f"{paths.unit} was not written by hark install; left alone")
    if activate and plugin_enabled(paths):
        backup.save(paths.shell_json)
        if _step(report, run, ["omarchy", "plugin", "disable", PLUGIN_ID]):
            report.changed.append("bar widget turned off")
    if paths.plugin.exists():
        if is_our_plugin(paths.plugin) and not paths.plugin.is_symlink():
            shutil.rmtree(paths.plugin)
            report.changed.append(f"removed {paths.plugin}")
            if activate:
                _step(report, run, ["omarchy-shell", "shell", "rescanPlugins"])
        else:
            report.problems.append(f"{paths.plugin} is not the hark widget; left alone")
    if remove_bindings(paths, backup):
        report.changed.append(f"keys removed from {paths.bindings}")
        if activate:
            _reload_hyprland(report, run)
    if backup.dir is not None:
        report.changed.append(f"backups in {backup.dir}")
    return report


def _step(report: Report, run: Runner, argv: list[str]) -> bool:
    res = run(argv)
    if res.returncode != 0:
        why = (res.stderr or res.stdout or "").strip().splitlines()
        report.problems.append(f"`{' '.join(argv)}` failed: {why[-1] if why else f'exit {res.returncode}'}")
        return False
    return True


def _reload_hyprland(report: Report, run: Runner) -> None:
    if not _step(report, run, ["hyprctl", "reload"]):
        return
    res = run(["hyprctl", "configerrors"])
    errors = [line for line in (res.stdout or "").splitlines() if line.strip()]
    if res.returncode == 0 and errors:
        report.problems.append("Hyprland reports config errors: " + "; ".join(errors[:5]))


def main(command: str, *, activate: bool = True) -> int:
    try:
        if command == "install":
            report = install(Path.home(), hark_binary(), activate=activate)
        else:
            report = uninstall(Path.home(), activate=activate)
    except (InstallError, OSError) as exc:
        print(f"hark: {exc}", file=sys.stderr)
        return 1
    for line in report.changed or ["nothing to change"]:
        print(line)
    for line in report.problems:
        print(f"hark: {line}", file=sys.stderr)
    return 1 if report.problems else 0
