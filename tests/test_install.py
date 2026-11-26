import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from hark import cli, install
from hark.install import BEGIN, END, InstallError, Paths

OMI = Path("/opt/omi/.venv/bin/omi")
ORIGINAL = '-- my keys\no.bind("SUPER + B", "Browser", { launch = "chromium" })\n'
REPO = Path(__file__).resolve().parent.parent
QML_TEST = REPO / "omarchy" / "qml-test"
OMARCHY_SHELL = Path("/usr/share/omarchy/shell")


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    (h / ".config/hypr").mkdir(parents=True)
    (h / ".config/hypr/bindings.lua").write_text(ORIGINAL)
    return h


class FakeRun:
    """Records commands instead of running them; answers chosen ones with a failure or output."""

    def __init__(self, active: bool = False, fail: dict | None = None, out: dict | None = None):
        self.calls: list[list[str]] = []
        self.active = active
        self.fail = fail or {}
        self.out = out or {}

    def __call__(self, argv):
        self.calls.append(argv)
        key = " ".join(argv)
        if key in self.fail:
            return subprocess.CompletedProcess(argv, 1, "", self.fail[key])
        if argv[:3] == ["systemctl", "--user", "is-active"]:
            return subprocess.CompletedProcess(argv, 0 if self.active else 3, "", "")
        return subprocess.CompletedProcess(argv, 0, self.out.get(key, ""), "")

    def ran(self, *argv):
        return list(argv) in self.calls


def backups(home):
    root = Paths(home).backups
    return sorted(root.iterdir()) if root.exists() else []


def test_install_writes_widget_unit_and_one_keys_block(home):
    report = install.install(home, OMI, activate=False)
    paths = Paths(home)

    assert json.loads((paths.plugin / "manifest.json").read_text())["id"] == "omi"
    assert install._tree(paths.plugin) == install._tree(REPO / "omarchy/plugin")

    unit = paths.unit.read_text()
    assert unit.startswith(install.UNIT_MARK)
    assert f"ExecStart={OMI} daemon\n" in unit
    for line in ("Restart=on-failure", "PartOf=graphical-session.target", "After=graphical-session.target",
                 "WantedBy=graphical-session.target"):
        assert line in unit
    assert "@OMI@" not in unit

    keys = paths.bindings.read_text()
    assert keys.startswith(ORIGINAL + "\n" + BEGIN + "\n")
    assert keys.endswith(END + "\n")
    assert keys.count(BEGIN) == 1 and keys.count(END) == 1
    assert (f'o.bind_hold("SUPER + ALT + V", "Omi: talk (hold)", "{OMI} ptt start", '
            f'"Omi: send", "{OMI} ptt stop")') in keys
    assert f'o.bind("SUPER + ALT + ESCAPE", "Omi: stop", "{OMI} stop")' in keys
    assert 'o.bind("SUPER + ALT + O", "Omi panel", "omarchy-shell omi toggle")' in keys
    assert '"ESCAPE"' not in keys and "F10" not in keys
    assert "@OMI@" not in keys

    (stamp,) = backups(home)
    assert (stamp / ".config/hypr/bindings.lua").read_text() == ORIGINAL
    assert not (stamp / ".config/omarchy").exists()
    assert any("keys added" in c for c in report.changed)


def test_install_migrates_legacy_hark_install(home):
    paths = Paths(home)
    paths.legacy_unit.parent.mkdir(parents=True, exist_ok=True)
    paths.legacy_unit.write_text(install.LEGACY_UNIT_MARK + "\n[Service]\nExecStart=/old/hark daemon\n")
    paths.legacy_plugin.mkdir(parents=True, exist_ok=True)
    (paths.legacy_plugin / "manifest.json").write_text('{"id": "hark"}')
    paths.bindings.write_text(
        ORIGINAL + "\n" + install.LEGACY_BEGIN + '\no.bind("SUPER + ALT + H", "Hark", "x")\n' + install.LEGACY_END + "\n"
    )
    paths.shell_json.parent.mkdir(parents=True, exist_ok=True)
    paths.shell_json.write_text(json.dumps({"version": 1, "bar": {"layout": {"right": [{"id": "hark"}]}}}))

    run = FakeRun()
    report = install.install(home, OMI, run=run)
    assert report.problems == []
    assert not paths.legacy_unit.exists()
    assert not paths.legacy_plugin.exists()
    assert install.LEGACY_BEGIN not in paths.bindings.read_text()
    assert BEGIN in paths.bindings.read_text()
    assert run.ran("systemctl", "--user", "disable", "--now", "hark.service")
    assert run.ran("omarchy", "plugin", "disable", "hark")
    assert run.ran("omarchy", "plugin", "enable", "omi")


def test_install_twice_changes_nothing(home):
    install.install(home, OMI, activate=False)
    before = {p: p.read_bytes() for p in home.rglob("*") if p.is_file()}
    report = install.install(home, OMI, activate=False)
    after = {p: p.read_bytes() for p in home.rglob("*") if p.is_file()}
    assert report.changed == [] and report.problems == []
    assert before == after
    assert len(backups(home)) == 1


def test_install_replaces_its_block_in_place(home):
    install.install(home, OMI, activate=False)
    paths = Paths(home)
    paths.bindings.write_text(paths.bindings.read_text() + "-- added later by the user\n")
    moved = Path("/srv/omi/.venv/bin/omi")

    install.install(home, moved, activate=False)

    keys = paths.bindings.read_text()
    assert keys.count(BEGIN) == 1
    assert str(OMI) not in keys and f"{moved} ptt start" in keys
    assert keys.startswith(ORIGINAL) and keys.endswith(END + "\n-- added later by the user\n")
    assert f"ExecStart={moved} daemon" in paths.unit.read_text()
    newest = backups(home)[-1]
    assert str(OMI) in (newest / ".config/hypr/bindings.lua").read_text()
    assert f"ExecStart={OMI} daemon" in (newest / ".config/systemd/user/omi.service").read_text()


def test_install_replaces_a_changed_widget_and_backs_it_up(home):
    install.install(home, OMI, activate=False)
    paths = Paths(home)
    (paths.plugin / "Panel.qml").write_text("edited")
    (paths.plugin / "stray.txt").write_text("x")

    install.install(home, OMI, activate=False)

    assert install._tree(paths.plugin) == install._tree(REPO / "omarchy/plugin")
    saved = backups(home)[-1] / ".config/omarchy/plugins/omi"
    assert (saved / "Panel.qml").read_text() == "edited" and (saved / "stray.txt").exists()
    assert sorted(p.name for p in paths.plugin.parent.iterdir()) == ["omi"]


def test_uninstall_removes_exactly_what_install_added(home):
    install.install(home, OMI, activate=False)
    report = install.uninstall(home, activate=False)
    paths = Paths(home)

    assert paths.bindings.read_text() == ORIGINAL
    assert not paths.plugin.exists()
    assert not paths.unit.exists()
    assert report.problems == []
    assert len(backups(home)) == 2

    again = install.uninstall(home, activate=False)
    assert again.changed == [] and again.problems == []
    assert paths.bindings.read_text() == ORIGINAL


def test_install_without_a_keys_file_creates_one(tmp_path):
    home = tmp_path / "home"
    install.install(home, OMI, activate=False)
    keys = Paths(home).bindings.read_text()
    assert keys.startswith(BEGIN) and keys.count(BEGIN) == 1
    assert backups(home) == []
    install.uninstall(home, activate=False)
    assert Paths(home).bindings.read_text() == ""


def test_uninstall_leaves_things_it_did_not_write(home):
    paths = Paths(home)
    paths.unit.parent.mkdir(parents=True)
    paths.unit.write_text("[Service]\nExecStart=/usr/bin/something-else\n")
    paths.plugin.mkdir(parents=True)
    (paths.plugin / "manifest.json").write_text('{"id": "someone.else"}')

    report = install.uninstall(home, activate=False)

    assert paths.unit.exists() and (paths.plugin / "manifest.json").exists()
    assert paths.bindings.read_text() == ORIGINAL
    assert len(report.problems) == 2


def test_a_broken_block_is_left_for_the_user(home):
    paths = Paths(home)
    paths.bindings.write_text(ORIGINAL + BEGIN + "\no.bind('X')\n")
    with pytest.raises(InstallError, match="no end marker"):
        install.install(home, OMI, activate=False)
    with pytest.raises(InstallError):
        install.uninstall(home, activate=False)
    assert paths.bindings.read_text() == ORIGINAL + BEGIN + "\no.bind('X')\n"


def test_a_path_the_keys_cannot_hold_writes_nothing(home):
    with pytest.raises(InstallError, match="plain path"):
        install.install(home, Path('/home/me/my "omi"/.venv/bin/omi'), activate=False)
    assert not Paths(home).plugin.exists() and not Paths(home).unit.exists()
    assert Paths(home).bindings.read_text() == ORIGINAL


def test_install_turns_everything_on(home):
    paths = Paths(home)
    paths.shell_json.parent.mkdir(parents=True, exist_ok=True)
    shell = {"version": 1, "bar": {"layout": {"right": [{"id": "omarchy.audio"}]}}}
    paths.shell_json.write_text(json.dumps(shell))
    run = FakeRun()

    report = install.install(home, OMI, run=run)

    assert report.problems == []
    assert run.calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "is-active", "--quiet", "omi.service"],
        ["systemctl", "--user", "enable", "--now", "omi.service"],
        ["omarchy-shell", "shell", "rescanPlugins"],
        ["omarchy", "plugin", "enable", "omi"],
        ["hyprctl", "reload"],
        ["hyprctl", "configerrors"],
    ]
    saved = backups(home)[-1] / ".config/omarchy/shell.json"
    assert json.loads(saved.read_text()) == shell


def test_install_again_only_restarts_what_changed(home):
    paths = Paths(home)
    install.install(home, OMI, activate=False)
    paths.shell_json.parent.mkdir(parents=True, exist_ok=True)
    paths.shell_json.write_text(json.dumps({"version": 1, "bar": {"layout": {"right": [{"id": "omi"}]}}}))

    same = FakeRun(active=True)
    install.install(home, OMI, run=same)
    assert not same.ran("omarchy", "plugin", "enable", "omi")
    assert not same.ran("hyprctl", "reload")
    assert not same.ran("systemctl", "--user", "restart", "omi.service")

    moved = FakeRun(active=True)
    install.install(home, Path("/srv/omi/.venv/bin/omi"), run=moved)
    assert moved.ran("systemctl", "--user", "restart", "omi.service")
    assert moved.ran("hyprctl", "reload")


def test_install_reports_failures_and_config_errors(home):
    run = FakeRun(fail={"systemctl --user enable --now omi.service": "Failed to enable unit"},
                  out={"hyprctl configerrors": "bindings.lua:3: bad bind\n"})
    report = install.install(home, OMI, run=run)
    assert any("Failed to enable unit" in p for p in report.problems)
    assert any("bad bind" in p for p in report.problems)


def test_uninstall_turns_everything_off(home):
    paths = Paths(home)
    install.install(home, OMI, activate=False)
    paths.shell_json.parent.mkdir(parents=True, exist_ok=True)
    paths.shell_json.write_text(json.dumps({"version": 1, "bar": {"layout": {"tray": [{"id": "omi"}]}}}))
    run = FakeRun()

    report = install.uninstall(home, run=run)

    assert report.problems == []
    assert run.calls == [
        ["systemctl", "--user", "disable", "--now", "omi.service"],
        ["systemctl", "--user", "daemon-reload"],
        ["omarchy", "plugin", "disable", "omi"],
        ["omarchy-shell", "shell", "rescanPlugins"],
        ["hyprctl", "reload"],
        ["hyprctl", "configerrors"],
    ]


def test_cli_install_and_uninstall(home, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(install, "omi_binary", lambda: OMI)
    monkeypatch.setattr(install, "hark_binary", lambda: OMI)

    assert cli.main(["install", "--no-activate"]) == 0
    assert "keys added" in capsys.readouterr().out
    assert cli.main(["install", "--no-activate"]) == 0
    assert capsys.readouterr().out.strip() == "nothing to change"
    assert cli.main(["uninstall", "--no-activate"]) == 0
    assert Paths(home).bindings.read_text() == ORIGINAL


def test_omi_binary_is_the_running_environments():
    exe = install.omi_binary()
    assert exe.name in ("omi", "hark") and os.access(exe, os.X_OK)


@pytest.mark.skipif(shutil.which("omarchy") is None, reason="needs the omarchy command")
def test_widget_manifest_passes_omarchy_validation():
    res = subprocess.run(["omarchy", "plugin", "validate", str(REPO / "omarchy/plugin")],
                         capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr


@pytest.mark.skipif(shutil.which("quickshell") is None or not (OMARCHY_SHELL / "Ui").is_dir(),
                    reason="needs quickshell and the Omarchy shell kit")
def test_bar_widget_runs_against_a_fake_daemon(tmp_path):
    """Loads the real Omi widget + CompanionOverlay in a separate, offscreen quickshell
    with Omarchy's Commons and Ui, against fake_daemon.py, and checks what it shows and sends."""
    cfg = tmp_path / "cfg"
    shutil.copytree(OMARCHY_SHELL / "Commons", cfg / "Commons")
    shutil.copytree(OMARCHY_SHELL / "Ui", cfg / "Ui")
    shutil.copy(QML_TEST / "KeyboardPanel.qml", cfg / "Ui" / "KeyboardPanel.qml")
    shutil.copy(QML_TEST / "OverlayWindow.qml", cfg / "Ui" / "OverlayWindow.qml")
    shutil.copytree(REPO / "omarchy/plugin", cfg / "plugin")
    shutil.copy(QML_TEST / "shell.qml", cfg / "shell.qml")
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    sock_dir = Path(tempfile.mkdtemp(prefix="om", dir="/tmp"))  # socket paths must be short
    sock, log = sock_dir / "s.sock", tmp_path / "requests.jsonl"
    fake = subprocess.Popen(["python3", str(QML_TEST / "fake_daemon.py"), str(sock), str(log)],
                            stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(100):
            if sock.exists() or fake.poll() is not None:
                break
            time.sleep(0.05)
        env = {k: v for k, v in os.environ.items() if k not in ("WAYLAND_DISPLAY", "HYPRLAND_INSTANCE_SIGNATURE")}
        env.update(QT_QPA_PLATFORM="offscreen", XDG_RUNTIME_DIR=str(run_dir), OMI_SOCKET=str(sock),
                   OMARCHY_PATH="/usr/share/omarchy")
        res = subprocess.run(["quickshell", "-p", str(cfg)], env=env, capture_output=True, text=True, timeout=40)
        fake.wait(10)
    finally:
        if fake.poll() is None:
            fake.kill()
            fake.wait()
        shutil.rmtree(sock_dir, ignore_errors=True)
    output = res.stdout + res.stderr
    checks = [line[line.index("CHECK "):] for line in output.splitlines() if "CHECK " in line]
    assert "HARNESS DONE" in output, output[-3000:]
    assert checks and all(c.startswith("CHECK pass") for c in checks), "\n".join(checks)
    assert len(checks) == 16, "\n".join(checks)
    for bad in ("TypeError", "ReferenceError", "is not a type", "unavailable", "Cannot assign", "Unable to assign"):
        assert bad not in output, output[-3000:]
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    assert {"cmd": "confirm", "answer": True, "id": "q-1"} in requests
    assert {"cmd": "say", "text": "open the browser"} in requests
    assert {"cmd": "stop"} in requests
