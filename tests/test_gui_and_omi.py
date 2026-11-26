"""Tests for Omi desktop GUI tools, Chrome safety rules, mascot events, and Kokoro voice defaults."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hark import config, router
from hark.daemon import Daemon
from hark.session import Session
from hark.tools import dev, gui, safety
from hark.tools.base import ToolRegistry
from tests.fakes import CaptureOut, FakeBrain, FakeSTT, FakeTTS


def test_omi_defaults():
    cfg = config.Config()
    assert cfg.name == "Omi"
    assert cfg.tts.engine == "kokoro"
    assert cfg.tts.kokoro_voice == "af_heart"
    assert cfg.brain.claude.chrome is True
    assert cfg.memory.shared_dir == ""
    assert "Obsidian" not in cfg.memory.dir
    assert cfg.stt.hotwords == ["Omi", "Omarchy", "Hyprland"]


def test_dev_builders_and_safety():
    assert dev.build_herdr_list("w1") == ["herdr", "pane", "list", "--workspace", "w1"]
    assert dev.build_herdr_read("w1:p1", 500) == ["herdr", "pane", "read", "--source", "recent", "--lines", "200", "w1:p1"]
    assert dev.build_herdr_run("w1:p1", "git status") == ["herdr", "pane", "run", "w1:p1", "--", "git status"]
    assert dev.build_tmux_read("0:0.0", 25) == ["tmux", "capture-pane", "-p", "-S", "-25", "-t", "0:0.0"]
    assert dev.build_tmux_send("0:0.0", "ls") == [
        ["tmux", "send-keys", "-t", "0:0.0", "-l", "--", "ls"],
        ["tmux", "send-keys", "-t", "0:0.0", "Enter"],
    ]
    assert dev.build_t3_threads("/usr/bin/t3", "proj-1") == ["/usr/bin/t3", "thread", "list", "--json", "--project", "proj-1"]
    assert dev.build_t3_read("/usr/bin/t3", "t-1", 5) == ["/usr/bin/t3", "thread", "show", "--json", "--messages", "5", "t-1"]
    assert dev.build_t3_start("/usr/bin/t3", "proj-1", "fix tests") == [
        "/usr/bin/t3", "thread", "start", "--json", "--generate-title", "proj-1", "fix tests"
    ]
    assert dev.build_t3_start("/usr/bin/t3", "proj-1", "fix tests", "Tests") == [
        "/usr/bin/t3", "thread", "start", "--json", "--title", "Tests", "proj-1", "fix tests"
    ]
    assert dev.build_t3_send("/usr/bin/t3", "t-1", "go ahead") == ["/usr/bin/t3", "thread", "send", "--json", "t-1", "go ahead"]

    assert dev.terminal_command_safety({"command": "git status"}) == "reversible"
    assert dev.terminal_command_safety({"command": "ls -la"}) == "reversible"
    assert dev.terminal_command_safety({"command": "uv run pytest"}) == "destructive"
    assert dev.terminal_command_safety({"command": "rm -rf build"}) == "destructive"
    assert dev.terminal_command_safety({"command": "git push"}) == "destructive"
    with pytest.raises(ValueError, match="blocked"):
        dev.terminal_command_safety({"command": "cat .env"})
    assert safety.classify_claude_tool("mcp__hark__terminal_run", {"command": "git status"})[0] == "allow"
    assert safety.classify_claude_tool("mcp__hark__terminal_run", {"command": "rm -rf build"})[0] == "confirm"
    assert safety.classify_claude_tool("mcp__hark__terminal_run", {"command": "cat ~/.ssh/id_ed25519"})[0] == "deny"


@pytest.mark.asyncio
async def test_dev_tools_parse_herdr_and_t3_output(monkeypatch):
    monkeypatch.setattr(dev, "herdr_available", lambda: True)
    monkeypatch.setattr(dev, "find_t3_binary", lambda: "/fake/t3")

    async def fake_run(argv, timeout=3.0, stdin=None):
        if argv[:3] == ["herdr", "pane", "list"]:
            payload = {"result": {"panes": [
                {"pane_id": "w1:p1", "workspace_id": "w1", "cwd": "/tmp/app", "focused": True, "terminal_title_stripped": "zsh"}
            ]}}
            return 0, json.dumps(payload), ""
        if argv[:4] == ["/fake/t3", "thread", "list", "--json"]:
            payload = [{"threadId": "abc12345-6789", "title": "Fix bug", "status": "idle", "model": "sonnet"}]
            return 0, json.dumps(payload), ""
        return 1, "", "unexpected"

    monkeypatch.setattr(dev, "run", fake_run)
    r_term = await dev.terminal_list()
    assert r_term.ok and "w1:p1" in r_term.text and r_term.data["backend"] == "herdr"
    r_t3 = await dev.t3_threads()
    assert r_t3.ok and "Fix bug" in r_t3.text and r_t3.data["threads"][0]["id"] == "abc12345-6789"


def test_router_strips_omi_wake_words():
    assert router.route("hey omi turn the volume up").tool == "volume"
    assert router.route("omi what time is it").tool == "current_time"
    assert router.route("oh me switch to workspace three").args == {"n": 3}


def test_gui_command_builders(tmp_path):
    p = tmp_path / "screen.jpg"
    assert gui.build_capture_screen(p, 0.5) == ["grim", "-t", "jpeg", "-q", "82", "-s", "0.50", str(p)]
    assert gui.build_movecursor(120, 340) == ["dispatch movecursor 120 340"]
    assert gui.build_ydotool_click("left", 1) == ["ydotool", "click", "0xC0"]
    assert gui.build_ydotool_click("right", 2) == ["ydotool", "click", "--repeat", "2", "--next-delay", "60", "0xC1"]
    assert gui.build_ydotool_scroll(-4) == ["ydotool", "mousemove", "--wheel", "--", "0", "-4"]
    with pytest.raises(ValueError):
        gui.build_ydotool_click("banana")


def test_ui_action_safety_blocks_destructive_labels(monkeypatch):
    monkeypatch.setattr(gui, "focused_is_terminal", lambda: False)
    gui._LAST_ELEMENTS[0] = gui.UiElement(0, "App", "push button", "Settings", 10, 10, 80, 30, ["click"], [0])
    gui._LAST_ELEMENTS[1] = gui.UiElement(1, "App", "push button", "Delete Account", 10, 50, 80, 30, ["click"], [1])
    gui._LAST_ELEMENTS[2] = gui.UiElement(2, "App", "push button", "Place Order", 10, 90, 80, 30, ["click"], [2])
    assert gui.ui_action_safety({"index": 0, "action": "click"}) == "reversible"
    assert gui.ui_action_safety({"index": 1, "action": "click"}) == "destructive"
    assert gui.ui_action_safety({"index": 2, "action": "click"}) == "destructive"


def test_chrome_mcp_safety_classification():
    assert safety.classify_claude_tool("mcp__claude-in-chrome__read_page", {})[0] == "allow"
    assert safety.classify_claude_tool("mcp__claude-in-chrome__tabs_context_mcp", {})[0] == "allow"
    assert safety.classify_claude_tool("mcp__claude-in-chrome__navigate", {"url": "https://example.com"})[0] == "allow"
    assert safety.classify_claude_tool("mcp__claude-in-chrome__navigate", {"url": "javascript:alert(1)"})[0] == "deny"
    assert safety.classify_claude_tool("mcp__claude-in-chrome__computer", {"action": "left_click"})[0] == "allow"
    assert safety.classify_claude_tool("mcp__claude-in-chrome__javascript_tool", {"text": "1+1"})[0] == "confirm"
    assert safety.classify_claude_tool("mcp__claude-in-chrome__upload_image", {})[0] == "confirm"


@pytest.mark.asyncio
async def test_daemon_mascot_broadcasts_and_live_voice_set(tmp_path):
    import shutil
    import tempfile

    sock_dir = Path(tempfile.mkdtemp(prefix="om", dir="/tmp"))
    sock = sock_dir / "omi.sock"
    cfg = config.Config()
    from hark.events import TextDelta, TurnDone

    session = Session(
        stt=FakeSTT(),
        tts=FakeTTS(),
        brain=FakeBrain(lambda _: [(0.0, TextDelta("Hello there.")), (0.0, TurnDone())]),
        registry=ToolRegistry(),
        router=lambda _: None,
        out=CaptureOut(),
        state_dir=tmp_path,
    )
    switched: list[str] = []

    async def set_voice(v: str):
        switched.append(v)
        return session.tts

    daemon = Daemon(session, cfg, socket_path=sock, factories={"tts.voice": set_voice})
    await daemon.start()
    try:
        sub_r, sub_w = await asyncio.open_unix_connection(str(sock))
        sub_w.write(b'{"cmd":"subscribe"}\n')
        await sub_w.drain()
        assert json.loads(await sub_r.readline())["subscribed"] is True

        # Mascot point broadcasts to subscribers
        ok = await gui.notify_mascot({"action": "point", "x": 500, "y": 300, "label": "Here"}, socket_path=sock)
        assert ok is True
        ev = json.loads(await asyncio.wait_for(sub_r.readline(), 2.0))
        assert ev["event"] == "mascot" and ev["x"] == 500 and ev["label"] == "Here"

        # Live voice switch updates config
        c_r, c_w = await asyncio.open_unix_connection(str(sock))
        c_w.write(b'{"cmd":"set","key":"tts.voice","value":"bm_george"}\n')
        await c_w.drain()
        reply = json.loads(await asyncio.wait_for(c_r.readline(), 2.0))
        assert reply["ok"] is True
        assert switched == ["bm_george"]
        assert cfg.tts.kokoro_voice == "bm_george"

        c_w.close()
        sub_w.close()
    finally:
        await daemon.close()
        shutil.rmtree(sock_dir, ignore_errors=True)
