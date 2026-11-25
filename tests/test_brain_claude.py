"""ClaudeSession and ClaudeBrain against a fake `claude` that speaks stream-json,
plus a few live turns against the real CLI (marked live)."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

import pytest

from hark.brain import claude_session
from hark.brain.claude import ClaudeBrain
from hark.config import Config
from hark.events import TextDelta, ToolEnd, ToolStart, TurnContext, TurnDone

FAKE_CLAUDE = r'''
import json, os, sys, threading, time
LOG = os.environ.get("FAKE_CLAUDE_LOG")
if LOG:
    with open(LOG, "a") as fh:
        fh.write(json.dumps({"argv": sys.argv[1:]}) + "\n")
lock = threading.Lock()
interrupted = threading.Event()

def out(obj):
    with lock:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

def delta(text):
    out({"type": "stream_event", "event": {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": text}}})

def result(text, error=False, subtype="success"):
    out({"type": "result", "subtype": subtype, "is_error": error, "result": text})

def tool_use(tid, name, args):
    out({"type": "assistant", "parent_tool_use_id": None, "message": {"role": "assistant",
         "content": [{"type": "tool_use", "id": tid, "name": name, "input": args}]}})

def tool_result(tid, text):
    out({"type": "user", "parent_tool_use_id": None, "message": {"role": "user",
         "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}})

def turn(content):
    out({"type": "system", "subtype": "init"})
    if "die" in content:
        os._exit(3)
    if "vanish" in content:  # stdout gone, process not reaped yet: the window that used to hang
        os.close(1)
        time.sleep(3)
        os._exit(3)
    if "slow" in content or "stubborn" in content:
        for _ in range(200):
            if interrupted.is_set() and "stubborn" not in content:
                out({"type": "user", "message": {"role": "user", "content": [
                    {"type": "text", "text": "[Request interrupted by user]"}]}})
                result("", True, "error_during_execution")
                return
            delta("word ")
            time.sleep(0.05)
        result("words")
        return
    if "tool" in content:
        for i in range(3):
            tool_use(f"t{i}", "mcp__hark__set_volume" if i == 0 else "Bash", {"percent": 30})
            tool_result(f"t{i}", [{"type": "text", "text": "Volume is 30 percent."}])
        delta("Done.")
        out({"type": "stream_event", "event": {"type": "message_stop"}})
        result("I did it. **All good**.")
        return
    for part in ["Hel", "lo **the", "re**."]:
        delta(part)
    out({"type": "stream_event", "event": {"type": "message_stop"}})
    out({"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": "Hello **there**."}]}})
    result("Hello there.")

for line in sys.stdin:
    msg = json.loads(line)
    if LOG:
        with open(LOG, "a") as fh:
            fh.write(json.dumps(msg) + "\n")
    if msg["type"] == "control_request":
        if msg["request"]["subtype"] == "interrupt":
            interrupted.set()
        out({"type": "control_response", "response": {"subtype": "success",
             "request_id": msg["request_id"], "response": {}}})
    elif msg["type"] == "user":
        interrupted.clear()
        threading.Thread(target=turn, args=(msg["message"]["content"],), daemon=True).start()
'''


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], Path]:
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE)
    log = tmp_path / "claude.log"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("HARK_STATE_DIR", str(tmp_path))
    return [sys.executable, str(script)], log


def log_lines(log: Path) -> list[dict]:
    return [json.loads(line) for line in log.read_text().splitlines()]


async def collect(brain, text: str, ctx: TurnContext | None = None) -> list:
    return [ev async for ev in brain.turn(text, ctx or TurnContext("s1"))]


async def test_chat_turn_streams_clean_text(fake_claude):
    cmd, log = fake_claude
    brain = ClaudeBrain(Config(), claude_cmd=cmd, use_mcp=False)
    await brain.start()
    try:
        events = await collect(brain, "hi")
    finally:
        await brain.close()
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello there."
    assert events[-1] == TurnDone()
    assert sum(isinstance(e, TurnDone) for e in events) == 1
    argv = log_lines(log)[0]["argv"]
    assert "--dangerously-skip-permissions" not in argv and "bypassPermissions" not in argv
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert argv[argv.index("--permission-mode") + 1] == "default"


def test_command_with_mcp_uses_approve_tool():
    session = claude_session.ClaudeSession("haiku", "x", use_mcp=True)
    cmd = session.command()
    assert cmd[cmd.index("--permission-prompt-tool") + 1] == "mcp__hark__approve"
    server = json.loads(cmd[cmd.index("--mcp-config") + 1])["mcpServers"]["hark"]
    assert server["args"][-2:] == ["-m", "hark.tools.mcp_server"]
    assert "--strict-mcp-config" in cmd and "--dangerously-skip-permissions" not in cmd


async def test_tool_events_strip_mcp_prefix(fake_claude):
    cmd, _ = fake_claude
    brain = ClaudeBrain(Config(), claude_cmd=cmd, use_mcp=False)
    await brain.start()
    try:
        events = await collect(brain, "use a tool")
    finally:
        await brain.close()
    starts = [e for e in events if isinstance(e, ToolStart)]
    ends = [e for e in events if isinstance(e, ToolEnd)]
    assert starts[0] == ToolStart("t0", "set_volume", {"percent": 30})
    assert ends[0] == ToolEnd("t0", "set_volume", True, "Volume is 30 percent.")
    assert events[-1] == TurnDone()


async def test_cancel_is_fast_and_process_is_reused(fake_claude):
    cmd, _ = fake_claude
    brain = ClaudeBrain(Config(), claude_cmd=cmd, use_mcp=False)
    await brain.start()
    try:
        pid = brain.session._proc.pid
        gen = brain.turn("slow please", TurnContext("s1"))
        assert isinstance(await anext(gen), TextDelta)
        t0 = time.perf_counter()
        await brain.cancel()
        rest = [ev async for ev in gen]
        latency = time.perf_counter() - t0
        assert rest[-1] == TurnDone(cancelled=True)
        assert latency < 0.1, latency
        events = await collect(brain, "hi again")
        assert events[-1] == TurnDone()
        assert brain.session._proc.pid == pid
    finally:
        await brain.close()


async def test_ignored_interrupt_restarts_process(fake_claude, monkeypatch):
    cmd, _ = fake_claude
    monkeypatch.setattr(claude_session, "INTERRUPT_GRACE_S", 0.3)
    brain = ClaudeBrain(Config(), claude_cmd=cmd, use_mcp=False)
    await brain.start()
    try:
        pid = brain.session._proc.pid
        gen = brain.turn("stubborn", TurnContext("s1"))
        await anext(gen)
        t0 = time.perf_counter()
        await brain.cancel()
        rest = [ev async for ev in gen]
        assert rest[-1] == TurnDone(cancelled=True)
        assert time.perf_counter() - t0 < 0.1
        events = await collect(brain, "hi")
        assert events[-1] == TurnDone()
        assert brain.session._proc.pid != pid
    finally:
        await brain.close()


async def test_dead_process_reports_error_then_restarts(fake_claude):
    cmd, _ = fake_claude
    brain = ClaudeBrain(Config(), claude_cmd=cmd, use_mcp=False)
    await brain.start()
    try:
        events = await collect(brain, "die now")
        assert events[-1].error and not events[-1].cancelled
        events = await collect(brain, "hi")
        assert events[-1] == TurnDone()
    finally:
        await brain.close()


async def test_memory_in_system_prompt_recall_and_history_in_message(fake_claude):
    cmd, log = fake_claude
    brain = ClaudeBrain(Config(), claude_cmd=cmd, use_mcp=False)
    await brain.start()
    ctx = TurnContext("s1", memory_context="The user likes tea.", recalled=["Cat is Miso."],
                      history=[{"role": "user", "content": "earlier q"},
                               {"role": "assistant", "content": "earlier a"}])
    try:
        await collect(brain, "hi", ctx)
        await collect(brain, "again", ctx)
    finally:
        await brain.close()
    lines = log_lines(log)
    argvs = [entry["argv"] for entry in lines if "argv" in entry]
    assert len(argvs) == 2  # restarted once for the new memory, not again
    prompt = argvs[1][argvs[1].index("--append-system-prompt") + 1]
    assert prompt.endswith("The user likes tea.")
    users = [entry["message"]["content"] for entry in lines if entry.get("type") == "user"]
    assert users[0].startswith("Earlier in this conversation:\nUser: earlier q\nYou: earlier a")
    assert users[0].endswith("hi\n\n(Notes from memory that may help: Cat is Miso.)")
    assert not users[1].startswith("Earlier")  # the live process already has the history


# --- live: the real claude CLI with haiku (4 calls here, 1 in test_worker) ---

LIVE_CMD = shutil.which("claude")


@pytest.mark.live
@pytest.mark.skipif(LIVE_CMD is None, reason="claude CLI not installed")
async def test_live_claude_chat_warm_and_interrupt(tmp_path, monkeypatch):
    monkeypatch.setenv("HARK_STATE_DIR", str(tmp_path))
    brain = ClaudeBrain(Config(), use_mcp=claude_session.hark_mcp_available())
    t0 = time.perf_counter()
    await brain.start()
    startup = time.perf_counter() - t0

    async def timed(text: str) -> tuple[float, str, TurnDone]:
        t = time.perf_counter()
        first, parts, last = None, [], None
        async for ev in brain.turn(text, TurnContext("live")):
            if isinstance(ev, TextDelta):
                first = first or time.perf_counter() - t
                parts.append(ev.text)
            last = ev
        return first, "".join(parts), last

    try:
        cold, text1, done1 = await timed("Say hello in five words.")
        warm, text2, done2 = await timed("Now say goodbye in five words.")
        gen = brain.turn("Count from one to fifty in words, comma separated.", TurnContext("live"))
        async for ev in gen:
            if isinstance(ev, TextDelta):
                break
        t = time.perf_counter()
        await brain.cancel()
        rest = [ev async for ev in gen]
        cancel_s = time.perf_counter() - t
        after, text3, done3 = await timed("Say ok.")
    finally:
        await brain.close()
    print(f"\nLIVE claude startup={startup*1000:.0f}ms cold_first_text={cold*1000:.0f}ms "
          f"warm_first_text={warm*1000:.0f}ms cancel={cancel_s*1000:.1f}ms "
          f"after_cancel_first_text={after*1000:.0f}ms texts={text1!r} {text2!r} {text3!r}")
    assert done1 == TurnDone() and done2 == TurnDone() and done3 == TurnDone()
    assert text1 and text2 and text3
    assert rest[-1] == TurnDone(cancelled=True) and cancel_s < 0.1


# --- fixes ---


async def test_start_with_memory_means_no_restart_on_turn_one(fake_claude):
    cmd, log = fake_claude
    brain = ClaudeBrain(Config(), claude_cmd=cmd, use_mcp=False)
    await brain.start("The user likes tea.")
    try:
        await collect(brain, "hi", TurnContext("s1", memory_context="The user likes tea."))
    finally:
        await brain.close()
    argvs = [entry["argv"] for entry in log_lines(log) if "argv" in entry]
    assert len(argvs) == 1
    assert argvs[0][argvs[0].index("--append-system-prompt") + 1].endswith("The user likes tea.")


async def test_turn_after_a_crash_never_hangs(fake_claude):
    """The CLI dying used to leave the next turn waiting forever or failing at random."""
    cmd, _ = fake_claude
    brain = ClaudeBrain(Config(), claude_cmd=cmd, use_mcp=False)
    await brain.start()
    try:
        for word in ("die", "vanish", "die", "vanish"):
            events = await asyncio.wait_for(collect(brain, f"{word} now"), 5)
            assert events[-1].error
            events = await asyncio.wait_for(collect(brain, "hi"), 5)
            assert events[-1] == TurnDone(), events[-1]
    finally:
        await brain.close()


async def test_close_while_starting_leaves_no_process(fake_claude):
    """A memory job cancelled while its brain is still starting (the user began talking,
    or the daemon is stopping) used to leave a claude process running for good."""
    cmd, _ = fake_claude
    brain = ClaudeBrain(Config(), claude_cmd=cmd, use_mcp=False)
    task = asyncio.create_task(brain.start())
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await brain.close()
    await asyncio.sleep(0.5)
    proc = brain.session._proc
    try:
        assert proc is None or proc.returncode is not None
    finally:
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()


# --- speed ---


def test_talker_runs_at_low_effort_and_other_sessions_keep_the_default(tmp_path, monkeypatch):
    # Haiku's adaptive thinking held back the first word on garbled transcripts.
    monkeypatch.setenv("HARK_STATE_DIR", str(tmp_path))
    cmd = ClaudeBrain(Config(), use_mcp=False).session.command()
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert "--effort" not in claude_session.ClaudeSession("haiku", "x", use_mcp=False).command()
