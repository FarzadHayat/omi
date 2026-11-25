"""Omarchy integration: the omarchy-* command allowlist and spoken app names.

The allowlist is generated at import from the "# omarchy:" header comments of the
scripts, so it follows whatever Omarchy version is installed. Anything that
installs, removes, updates, resets, touches power or runs an arbitrary command is
destructive and needs the user's yes.
"""

from __future__ import annotations

import configparser
import functools
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from hark.tools.base import NOT_CONFIRMED, Safety, Tool, ToolRegistry, ToolResult
from hark.tools.system import run

BIN = Path(os.environ.get("OMARCHY_PATH", "/usr/share/omarchy")) / "bin"

# Name parts that make a command destructive no matter what else it does.
_DESTRUCTIVE_PARTS = {
    "install", "reinstall", "remove", "uninstall", "update", "upgrade", "reset", "factory",
    "refresh", "setup", "provision", "migrate", "apply", "sudo", "pkg", "pkgs", "drive",
    "plymouth", "hibernation", "system", "power", "powerprofiles", "suspend", "reboot",
    "shutdown", "logout", "restart", "kill", "delete", "wipe", "dev", "agent", "plugin",
    "hw", "debug", "upload", "share", "sync", "channel", "branch", "snapshot", "firmware",
    "close", "all",
}
# Groups whose commands change only reversible things (look, focus, media, theme).
_REVERSIBLE_GROUPS = {
    "theme", "toggle", "launch", "audio", "brightness", "notification", "capture", "font",
    "hyprland", "menu", "webapp", "clipboard", "weather", "tui", "screenrecord", "show",
}
_READ_LAST = {"list", "current", "status", "version", "present", "stats", "show", "get", "info"}


@dataclass(frozen=True)
class Command:
    name: str  # without the "omarchy-" prefix, like "theme-set"
    summary: str
    args: str
    safety: Safety


def parse_header(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    try:
        with path.open(errors="replace") as fh:
            for i, line in enumerate(fh):
                if i > 20:
                    break
                if line.startswith("# omarchy:") and "=" in line:
                    key, _, value = line[10:].partition("=")
                    meta[key.strip()] = value.strip()
    except OSError:
        pass
    return meta


def classify(name: str, meta: dict[str, str]) -> Safety:
    parts = name.split("-")
    args = meta.get("args", "").lower()
    if meta.get("requires-sudo") == "true" or _DESTRUCTIVE_PARTS.intersection(parts):
        return "destructive"
    # Scripts that take a command to run (launch-or-focus evals it) can do anything.
    if "command" in args or "cmd" in args or "exec" in args:
        return "destructive"
    if parts[-1] in _READ_LAST or name == "version":
        return "read"
    if parts[0] in _REVERSIBLE_GROUPS:
        return "reversible"
    return "destructive"


def load_commands(bin_dir: Path = BIN) -> dict[str, Command]:
    commands: dict[str, Command] = {}
    if not bin_dir.is_dir():
        return commands
    for path in sorted(bin_dir.glob("omarchy-*")):
        meta = parse_header(path)
        if "summary" not in meta or meta.get("hidden") == "true":
            continue
        name = path.name.removeprefix("omarchy-")
        commands[name] = Command(name, meta["summary"], meta.get("args", ""), classify(name, meta))
    return commands


COMMANDS: dict[str, Command] = load_commands()


def resolve_command(command: str) -> Command | None:
    """Accepts 'theme set', 'theme-set', 'omarchy theme set' or 'omarchy-theme-set'."""
    key = re.sub(r"[\s_]+", "-", command.strip().lower())
    key = key.removeprefix("omarchy-")
    return COMMANDS.get(key)


def build_omarchy(command: str, args: list[str] | None = None) -> list[str]:
    cmd = resolve_command(command)
    if cmd is None:
        raise ValueError(f"omarchy {command} is not an allowed command")
    args = [str(a) for a in (args or [])]
    if any("\n" in a or "\0" in a for a in args):
        raise ValueError("arguments may not contain newlines")
    return [str(BIN / f"omarchy-{cmd.name}"), *args]


def command_safety(command: str) -> Safety:
    cmd = resolve_command(command)
    return cmd.safety if cmd else "destructive"


def confirm_question(command: str, args: list[str] | None = None) -> str:
    cmd = resolve_command(command)
    words = " ".join([(cmd.name if cmd else command).replace("-", " "), *(args or [])])
    return f"Run omarchy {words}?"


def make_tool(registry: ToolRegistry) -> Tool:
    """The omarchy tool. Its registry class is reversible; destructive commands ask
    the user here, per command, through the registry's ConfirmFn."""

    async def omarchy(command: str, args: list[str] | None = None) -> ToolResult:
        cmd = resolve_command(command)
        if cmd is None:
            return ToolResult(False, f"omarchy {command} is not an allowed command.")
        if cmd.safety == "destructive":
            if registry.confirm is None or not await registry.confirm(confirm_question(command, args)):
                return ToolResult(False, NOT_CONFIRMED)
        rc, out, err = await run(build_omarchy(command, args), timeout=10.0)
        text = (out or err).strip()
        if rc != 0:
            return ToolResult(False, f"That failed: {text.splitlines()[-1] if text else rc}")
        return ToolResult(True, text[:400] or "Done.", text)

    return Tool(
        "omarchy",
        "Run an Omarchy command, like command 'theme set' with args ['tokyo-night'].",
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "omarchy command, like 'theme set'"},
                "args": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["command"],
        },
        "reversible",
        omarchy,
    )


# --- spoken app names ---


@dataclass(frozen=True)
class App:
    name: str
    classes: tuple[str, ...]  # window classes that mean the app is already open
    launch: tuple[str, ...]  # argv that starts it


def _data_dirs() -> list[Path]:
    home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    rest = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    return [Path(d) / "applications" for d in [home, *rest.split(":")] if d]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.+ ]", " ", text.lower())).strip()


def _read_desktop(path: Path) -> dict[str, str] | None:
    parser = configparser.RawConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # keep key case
    try:
        parser.read(path, encoding="utf-8")
        entry = parser["Desktop Entry"]
    except (configparser.Error, KeyError, UnicodeDecodeError):
        return None
    if entry.get("Type", "Application") != "Application":
        return None
    if entry.get("NoDisplay") == "true" or entry.get("Hidden") == "true":
        return None
    return dict(entry)


def _desktop_app(desktop_id: str, entry: dict[str, str]) -> App:
    stem = desktop_id.removesuffix(".desktop")
    classes = {stem.lower(), stem.split(".")[-1].lower()}
    if entry.get("StartupWMClass"):
        classes.add(entry["StartupWMClass"].lower())
    exec_ = entry.get("Exec", "")
    if "omarchy-launch-webapp" in exec_:
        url = exec_.split("omarchy-launch-webapp", 1)[1].split()
        if url:
            host = urlparse(url[0].strip("\"'")).netloc.removeprefix("www.")
            classes = {host} if host else classes
    return App(entry.get("Name", stem), tuple(sorted(classes)), ("uwsm-app", "--", desktop_id))


# Last parts of reverse-DNS desktop ids that are not app names.
_GENERIC = {"app", "desktop", "service", "signed", "studio", "client", "launcher", "sms", "nonplasma"}


@functools.cache
def desktop_index() -> dict[str, App]:
    """Spoken key -> App for every visible installed .desktop file. User entries
    shadow system ones, as in the XDG spec."""
    index: dict[str, App] = {}
    seen: set[str] = set()
    for directory in _data_dirs():
        if not directory.is_dir():
            continue
        for path in directory.glob("*.desktop"):
            if path.name in seen:
                continue
            seen.add(path.name)
            entry = _read_desktop(path)
            if entry is None:
                continue
            app = _desktop_app(path.name, entry)
            stem = path.name.removesuffix(".desktop")
            keys = {_norm(entry.get("Name", "")), _norm(stem)}
            last = _norm(stem.split(".")[-1])
            if "." in stem and len(last) > 2 and last not in _GENERIC:
                keys.add(last)
            for key in keys:
                if key and key not in index:
                    index[key] = app
    return index


def _desktop_by_id(desktop_id: str) -> App | None:
    for directory in _data_dirs():
        path = directory / desktop_id
        if path.is_file():
            entry = _read_desktop(path) or {}
            return _desktop_app(desktop_id, entry)
    return None


@functools.cache
def _default_browser() -> App:
    try:
        out = subprocess.run(["omarchy-cmd-default-browser"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ""
    app = _desktop_by_id(out.strip()) if out.strip() else None
    return App("browser", app.classes if app else ("chromium",), ("omarchy-launch-browser",))


@functools.cache
def _default_terminal() -> App:
    classes: tuple[str, ...] = ("foot", "alacritty", "ghostty", "kitty")
    lists = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "xdg-terminals.list"
    try:
        for line in lists.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                app = _desktop_by_id(line)
                if app:
                    classes = app.classes
                break
    except OSError:
        pass
    return App("terminal", classes, ("omarchy-launch-terminal",))


def _editor() -> App:
    try:
        editor = (Path.home() / ".local/state/omarchy/defaults/editor").read_text().split()[0]
    except (OSError, IndexError):
        editor = "nvim"
    name = Path(editor).name
    return App("editor", (name, f"org.omarchy.{name}"), ("omarchy-launch-editor",))


_ALIASES = {
    "browser": _default_browser, "web browser": _default_browser, "internet": _default_browser,
    "terminal": _default_terminal, "console": _default_terminal,
    "files": lambda: App("files", ("org.gnome.nautilus", "nautilus"), ("omarchy-launch-nautilus",)),
    "file manager": lambda: App("files", ("org.gnome.nautilus", "nautilus"), ("omarchy-launch-nautilus",)),
    "spotify": lambda: App("spotify", ("spotify",), ("omarchy-launch-spotify",)),
    "editor": _editor, "text editor": _editor, "code editor": _editor,
}


def clean_app_name(spoken: str) -> str:
    text = _norm(spoken)
    text = re.sub(r"^(the|my|a|an) ", "", text)
    return re.sub(r" (app|application|window)$", "", text).strip()


def resolve_app(spoken: str) -> App | None:
    key = clean_app_name(spoken)
    if key in _ALIASES:
        return _ALIASES[key]()
    return desktop_index().get(key)


def app_names() -> set[str]:
    """Every spoken name resolve_app understands (for the router)."""
    return set(_ALIASES) | set(desktop_index())
