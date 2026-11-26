"""Tests for model-agnostic subscription CLI providers (opencode, codex, agy, gemini)
and live provider/model switching."""

from __future__ import annotations

import asyncio
import json
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import pytest

from hark import brain, config
from hark.brain import cli_providers as cp
from hark.daemon import Daemon
from hark.events import TextDelta, ToolEnd, ToolStart, TurnContext, TurnDone, WorkerDone, WorkerProgress
from hark.session import Session
from hark.tools.base import ToolRegistry
from tests.fakes import CaptureOut, FakeBrain, FakeSTT, FakeTTS


def test_build_cli_invocations():
    argv, env = cp.build_cli_invocation("opencode", "hi", model="opencode/deepseek-v4.1-flash-free", session_id="ses_1")
    assert argv[:4] == ["opencode", "run", "--format", "json"]
    assert "--model" in argv and "opencode/deepseek-v4.1-flash-free" in argv
    assert "--session" in argv and "ses_1" in argv and argv[-1] == "hi"
    mcp = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert mcp["mcp"]["omi"]["type"] == "local" and "hark.tools.mcp_server" in mcp["mcp"]["omi"]["command"]

    argv, env = cp.build_cli_invocation("codex", "hi", model="gpt-5-mini")
    assert argv[:4] == ["codex", "exec", "--json", "--skip-git-repo-check"]
    assert "-m" in argv and "gpt-5-mini" in argv and any("mcp_servers.omi.command=" in a for a in argv)
    argv2, _ = cp.build_cli_invocation("codex", "again", session_id="thr_9", use_mcp=False)
    assert argv2[:3] == ["codex", "exec", "resume"] and argv2[-2:] == ["thr_9", "again"]

    argv, _ = cp.build_cli_invocation("agy", "hi", model="gemini-4-neon", effort="low", session_id="conv_2")
    assert "--output-format" in argv and "stream-json" in argv
    assert "--effort" in argv and "low" in argv
    assert "--conversation" in argv and "conv_2" in argv
    assert "--settings" in argv and argv[-1] == "-p=hi"

    argv, _ = cp.build_cli_invocation("gemini", "hi", model="gemini-2.5-flash", session_id="g_3")
    assert argv == ["gemini", "--output-format", "stream-json", "-m", "gemini-2.5-flash", "--resume", "g_3", "-p", "hi"]


def test_parse_cli_events():
    # OpenCode
    ev = cp.parse_cli_event("opencode", {"type": "text", "sessionID": "ses_1", "part": {"text": "Hello"}})
    assert ev.session_id == "ses_1" and ev.text == "Hello"
    ev = cp.parse_cli_event("opencode", {"type": "tool_use", "part": {
        "callID": "c1", "tool": "mcp__hark__volume", "state": {"status": "running", "input": {"action": "up"}},
    }})
    assert ev.tool_start == ("c1", "volume", {"action": "up"})
    ev = cp.parse_cli_event("opencode", {"type": "tool_use", "part": {
        "callID": "c1", "tool": "mcp__hark__volume", "state": {"status": "completed", "output": "Done."},
    }})
    assert ev.tool_end == ("c1", "volume", True, "Done.")

    # Codex
    assert cp.parse_cli_event("codex", {"type": "thread.started", "thread_id": "t_1"}).session_id == "t_1"
    assert cp.parse_cli_event("codex", {"type": "item.completed", "item": {"type": "agent_message", "text": "Hi"}}).text == "Hi"
    assert cp.parse_cli_event("codex", {"type": "error", "message": "Reconnecting... 1/5"}).error is None
    assert cp.parse_cli_event("codex", {"type": "turn.failed", "error": {"message": "401"}}).error == "401"

    # Antigravity / Gemini
    ev = cp.parse_cli_event("agy", {
        "type": "stream_event", "conversation_id": "conv_9",
        "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hey"}},
    })
    assert ev.session_id == "conv_9" and ev.text == "Hey"
    ev = cp.parse_cli_event("gemini", {"type": "message", "session_id": "g_1", "role": "assistant", "content": "Yo"})
    assert ev.session_id == "g_1" and ev.text == "Yo"


def test_resolve_provider_and_factories(monkeypatch):
    assert cp.resolve_provider("openai") == "local"
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/opencode" if name == "opencode" else None)
    assert cp.resolve_provider("auto", role="talker") == "opencode"
    assert cp.resolve_provider("auto", role="worker") == "opencode"

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert cp.resolve_provider("auto", role="talker") == "local"
    assert cp.resolve_provider("auto", role="worker") == "off"

    reg = ToolRegistry()
    for name in ("opencode", "codex", "agy", "gemini"):
        cfg = config.Config()
        cfg.brain.talker = name
        cfg.brain.worker = name
        t = brain.create_talker(cfg, reg)
        w = brain.create_worker(cfg)
        assert isinstance(t, cp.CliBrain) and t.name == name
        assert isinstance(w, cp.CliWorker) and w.provider == name


def _make_fake_cli(tmp_path: Path, body: str) -> str:
    script = tmp_path / "fake_cli.py"
    script.write_text(f"#!{sys.executable}\n{body}\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


@pytest.mark.asyncio
async def test_cli_brain_streams_and_reuses_session(tmp_path):
    fake = _make_fake_cli(
        tmp_path,
        """
import json, sys
args = sys.argv[1:]
sid = args[args.index("--session") + 1] if "--session" in args else "ses_new"
print(json.dumps({"type": "step_start", "sessionID": sid}), flush=True)
print(json.dumps({"type": "tool_use", "part": {"callID": "t1", "tool": "volume", "state": {"status": "running", "input": {}}}}), flush=True)
print(json.dumps({"type": "tool_use", "part": {"callID": "t1", "tool": "volume", "state": {"status": "completed", "output": "ok"}}}), flush=True)
reply = "Second turn." if "--session" in args else "First **turn**."
print(json.dumps({"type": "text", "sessionID": sid, "part": {"text": reply}}), flush=True)
""",
    )
    cfg = config.Config()
    b = cp.CliBrain("opencode", cfg, use_mcp=False, binary=fake)
    await b.start()
    assert b.error is None

    events1 = [ev async for ev in b.turn("hello", TurnContext(session_id="s1"))]
    assert any(isinstance(e, ToolStart) and e.name == "volume" for e in events1)
    assert any(isinstance(e, ToolEnd) and e.ok for e in events1)
    text1 = "".join(e.text for e in events1 if isinstance(e, TextDelta))
    assert text1 == "First turn."  # SpeechFilter stripped markdown
    assert isinstance(events1[-1], TurnDone) and events1[-1].error is None

    events2 = [ev async for ev in b.turn("again", TurnContext(session_id="s1"))]
    text2 = "".join(e.text for e in events2 if isinstance(e, TextDelta))
    assert text2 == "Second turn."


@pytest.mark.asyncio
async def test_cli_worker_runs_job_and_narrates(tmp_path):
    fake = _make_fake_cli(
        tmp_path,
        """
import json
print(json.dumps({"type": "thread.started", "thread_id": "thr_1"}), flush=True)
print(json.dumps({"type": "item.started", "item": {"id": "c1", "type": "mcp_tool_call", "tool": "mcp__hark__terminal_read"}}), flush=True)
print(json.dumps({"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "All tests pass."}}), flush=True)
""",
    )
    cfg = config.Config()
    w = cp.CliWorker("codex", cfg, use_mcp=False, binary=fake)
    await w.start()
    try:
        jid = await w.submit("check terminal")
        got = []
        async for ev in w.events():
            got.append(ev)
            if isinstance(ev, WorkerDone):
                break
        assert any(isinstance(e, WorkerProgress) and "terminal" in e.text.lower() for e in got)
        done = got[-1]
        assert isinstance(done, WorkerDone) and done.job_id == jid and done.ok and done.summary == "All tests pass."
    finally:
        await w.close()


@pytest.mark.asyncio
async def test_daemon_live_provider_and_model_switch(tmp_path):
    sock_dir = Path(tempfile.mkdtemp(prefix="om", dir="/tmp"))
    sock = sock_dir / "omi.sock"
    cfg = config.Config()
    session = Session(
        stt=FakeSTT(),
        tts=FakeTTS(),
        brain=FakeBrain(lambda _: [(0.0, TextDelta("ok")), (0.0, TurnDone())]),
        registry=ToolRegistry(),
        router=lambda _: None,
        out=CaptureOut(),
        state_dir=tmp_path,
    )
    swapped: list[tuple[str, str]] = []

    async def set_talker(val: str):
        swapped.append(("talker", val))
        fb = FakeBrain(lambda _: [(0.0, TextDelta("ok")), (0.0, TurnDone())])
        fb.name = "opencode" if val == "auto" else val
        return fb

    async def set_model(val: str):
        swapped.append(("model", val))
        return session.brain

    daemon = Daemon(
        session, cfg, socket_path=sock,
        factories={"brain.talker": set_talker, "brain.model": set_model},
    )
    await daemon.start()
    try:
        r, w = await asyncio.open_unix_connection(str(sock))
        w.write(b'{"cmd":"set","key":"brain.talker","value":"auto"}\n')
        await w.drain()
        rep = json.loads(await asyncio.wait_for(r.readline(), 2.0))
        assert rep == {"ok": True, "key": "brain.talker", "value": "opencode"}
        assert cfg.brain.talker == "opencode"

        w.write(b'{"cmd":"set","key":"brain.model","value":"opencode/deepseek-v4.1-flash-free"}\n')
        await w.drain()
        rep2 = json.loads(await asyncio.wait_for(r.readline(), 2.0))
        assert rep2 == {"ok": True, "key": "brain.model", "value": "opencode/deepseek-v4.1-flash-free"}
        assert cfg.brain.opencode.model == "opencode/deepseek-v4.1-flash-free"
        assert daemon.status()["talker_model"] == "opencode/deepseek-v4.1-flash-free"
        w.close()
    finally:
        await daemon.close()
        shutil.rmtree(sock_dir, ignore_errors=True)
