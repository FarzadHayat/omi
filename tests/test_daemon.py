import asyncio
import json
import os
import shutil
import signal
import socket
import sys
import stat
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hark import cli
from hark.config import Config
from hark.daemon import Daemon
from hark.events import TextDelta, TurnDone
from hark.session import Session
from tests.fakes import (
    CaptureOut,
    FakeAudioIn,
    FakeBrain,
    FakeRouter,
    FakeSTT,
    FakeTTS,
    FakeWavOut,
    FakeWorker,
    Route,
    make_registry,
)

FIXTURE = Path(__file__).parent / "fixtures" / "speech_a.wav"


@pytest.fixture
def sock_dir():
    # Unix socket paths are limited to about 108 bytes, so not tmp_path.
    d = Path(tempfile.mkdtemp(prefix="hk", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_daemon(sock_dir, brain=None, **kw):
    calls: list = []
    reg = make_registry(calls)
    out = CaptureOut(speed=50)
    session = Session(
        stt=FakeSTT("turn the volume up"),
        tts=FakeTTS(),
        brain=brain or FakeBrain(),
        registry=reg,
        router=FakeRouter({"close it": Route("close_window", {}, "Closed.")}),
        out=out,
        worker=kw.pop("worker", None),
        state_dir=sock_dir,
    )
    reg.confirm = session.confirm

    async def new_brain(value):
        if value == "bad":
            raise ValueError("no such brain")
        return FakeBrain(name=value)

    d = Daemon(
        session,
        Config(),
        audio_in=FakeAudioIn(),
        factories={"brain.talker": new_brain},
        wav_out=FakeWavOut,
        socket_path=sock_dir / "run" / "hark.sock",
        **kw,
    )
    d.out = out
    d.calls = calls
    return d


@pytest.fixture
async def daemon(sock_dir):
    d = make_daemon(sock_dir)
    await d.start()
    yield d
    await d.close()


class Client:
    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer

    @classmethod
    async def open(cls, d):
        r, w = await asyncio.open_unix_connection(str(d.socket_path), limit=1 << 24)
        return cls(r, w)

    async def send(self, obj):
        raw = obj if isinstance(obj, bytes) else (json.dumps(obj) + "\n").encode()
        self.writer.write(raw)
        await self.writer.drain()

    async def line(self, timeout=3.0):
        raw = await asyncio.wait_for(self.reader.readline(), timeout)
        return json.loads(raw)

    async def until_done(self, streaming=True):
        lines = []
        while True:
            msg = await self.line()
            lines.append(msg)
            if "event" not in msg and (msg.get("done") or not msg.get("ok") or not streaming):
                return lines

    async def close(self):
        self.writer.close()


async def call(d, req):
    c = await Client.open(d)
    await c.send(req)
    lines = await c.until_done(req.get("cmd") in ("say", "wav", "ptt_stop"))
    await c.close()
    return lines


async def test_socket_permissions(daemon):
    assert stat.S_IMODE(daemon.socket_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(daemon.socket_path.parent.stat().st_mode) == 0o700


async def test_say_streams_events_then_done(daemon):
    lines = await call(daemon, {"cmd": "say", "text": "hello"})
    events = [m["event"] for m in lines if "event" in m]
    assert events[0] == "state" and "transcript" in events and "timing" in events
    final = lines[-1]
    assert final["ok"] and final["done"] and final["reply"] == "Okay." and final["timing"]["path"] == "talker"
    assert daemon.out.writes


async def test_say_quiet_plays_nothing(daemon):
    final = (await call(daemon, {"cmd": "say", "text": "hello", "quiet": True}))[-1]
    assert final["reply"] == "Okay." and daemon.out.writes == []


async def test_wav_and_wav_out(daemon, tmp_path):
    out = tmp_path / "reply.wav"
    final = (await call(daemon, {"cmd": "wav", "path": str(FIXTURE), "out": str(out)}))[-1]
    assert final["ok"] and final["timing"]["stt_ms"] is not None
    data, sr = sf.read(out)
    assert sr == 16000 and len(data) > 0
    assert daemon.out.writes == []  # went to the file, not the speakers


async def test_wav_wrong_rate_and_missing(daemon, tmp_path):
    bad = tmp_path / "bad.wav"
    sf.write(bad, np.zeros(4410, np.float32), 44100)
    assert "16000" in (await call(daemon, {"cmd": "wav", "path": str(bad)}))[-1]["error"]
    assert not (await call(daemon, {"cmd": "wav", "path": str(tmp_path / "nope.wav")}))[-1]["ok"]


async def test_ptt(daemon):
    assert (await call(daemon, {"cmd": "ptt_stop"}))[-1] == {"ok": False, "error": "not listening"}
    c = await Client.open(daemon)
    await c.send({"cmd": "ptt_start"})
    assert await c.line() == {"ok": True}
    assert daemon.session.state.value == "listening"
    await c.send({"cmd": "ptt_stop", "quiet": True})
    lines = await c.until_done()
    assert any(m.get("role") == "user" and m["text"] == "turn the volume up" for m in lines)
    assert lines[-1]["done"]
    await c.close()


async def test_stop_from_other_connection(sock_dir):
    brain = FakeBrain([(0.0, TextDelta("Start. ")), (5.0, TextDelta("never")), (0.0, TurnDone())])
    d = make_daemon(sock_dir, brain)
    await d.start()
    try:
        c = await Client.open(d)
        await c.send({"cmd": "say", "text": "long one"})
        await asyncio.sleep(0.05)
        t = time.perf_counter()
        assert (await call(d, {"cmd": "stop"}))[-1] == {"ok": True}
        lines = await c.until_done()
        took = (time.perf_counter() - t) * 1000
        assert lines[-1]["cancelled"] and lines[-1]["reply"] == "Start."
        print(f"\nstop over the socket until the turn's final line: {took:.1f} ms")
        assert took < 100
        await c.close()
    finally:
        await d.close()


async def test_new_turn_cancels_running_one(sock_dir):
    brain = FakeBrain(lambda t: [(0.0, TextDelta("A. ")), (5.0 if t == "one" else 0.0, TextDelta("B.")), (0.0, TurnDone())])
    d = make_daemon(sock_dir, brain)
    await d.start()
    try:
        c1 = await Client.open(d)
        await c1.send({"cmd": "say", "text": "one", "quiet": True})
        await asyncio.sleep(0.05)
        second = (await call(d, {"cmd": "say", "text": "two", "quiet": True}))[-1]
        first = (await c1.until_done())[-1]
        assert first["cancelled"] and not second["cancelled"] and second["reply"] == "A. B."
        await c1.close()
    finally:
        await d.close()


async def test_confirm_command_flow(daemon):
    c = await Client.open(daemon)
    await c.send({"cmd": "say", "text": "close it", "quiet": True})
    while (msg := await c.line()).get("event") != "confirm_request":
        pass
    assert msg["question"] == "Close the window?" and msg["id"]
    assert (await call(daemon, {"cmd": "confirm", "answer": True, "id": msg["id"]}))[-1] == {"ok": True}
    final = (await c.until_done())[-1]
    assert final["reply"].endswith("Closed.") and daemon.calls == [("close_window", {})]
    await c.close()


async def test_confirm_nothing_pending_and_bad_answer(daemon):
    assert (await call(daemon, {"cmd": "confirm", "answer": True, "id": "x"}))[-1]["error"] == "nothing to confirm"
    assert not (await call(daemon, {"cmd": "confirm", "answer": "yes", "id": "x"}))[-1]["ok"]
    assert "id" in (await call(daemon, {"cmd": "confirm", "answer": True}))[-1]["error"]


async def test_ask_confirm(daemon):
    c = await Client.open(daemon)
    await c.send({"cmd": "ask_confirm", "question": "Delete it?", "quiet": True})
    await asyncio.sleep(0.05)
    assert daemon.session.state.value == "confirming"
    pending = (await call(daemon, {"cmd": "status"}))[-1]["confirm"]
    assert pending["question"] == "Delete it?"
    await call(daemon, {"cmd": "confirm", "answer": False, "id": pending["id"]})
    assert await c.line() == {"ok": True, "answer": False}
    # a spoken yes answers it too
    await c.send({"cmd": "ask_confirm", "question": "Delete it?"})
    await asyncio.sleep(0.05)
    ans = (await call(daemon, {"cmd": "say", "text": "yes, do it"}))[-1]
    assert ans["timing"]["path"] == "confirm"
    assert await c.line() == {"ok": True, "answer": True}
    await c.close()


async def test_status_and_set(daemon):
    await call(daemon, {"cmd": "say", "text": "hi", "quiet": True})
    st = (await call(daemon, {"cmd": "status"}))[-1]
    assert st["state"] == "idle" and st["talker"] == "local" and st["worker"] == "off"
    assert st["engines"] == {"stt": "moonshine", "tts": "piper"}
    assert st["session_id"] == daemon.session.session_id and st["timing"]["path"] == "talker"
    old = daemon.session.brain
    assert (await call(daemon, {"cmd": "set", "key": "brain.talker", "value": "claude"}))[-1]["ok"]
    assert daemon.session.brain.name == "claude"
    await asyncio.sleep(0.01)
    assert old.closed
    assert (await call(daemon, {"cmd": "status"}))[-1]["talker"] == "claude"
    assert "could not switch" in (await call(daemon, {"cmd": "set", "key": "brain.talker", "value": "bad"}))[-1]["error"]
    assert "cannot set" in (await call(daemon, {"cmd": "set", "key": "name", "value": "x"}))[-1]["error"]
    assert "live" in (await call(daemon, {"cmd": "set", "key": "tts.engine", "value": "kokoro"}))[-1]["error"]


async def test_subscribe_fan_out(daemon):
    subs = [await Client.open(daemon) for _ in range(2)]
    for s in subs:
        await s.send({"cmd": "subscribe"})
        assert (await s.line())["subscribed"]
    await call(daemon, {"cmd": "say", "text": "hello", "quiet": True})
    for s in subs:
        got = []
        while not any(m.get("event") == "timing" for m in got):
            got.append(await s.line())
        assert {"event": "transcript", "role": "user", "text": "hello", "final": True} in got
        await s.close()


async def test_bad_requests_keep_connection_usable(daemon):
    c = await Client.open(daemon)
    await c.send(b"{not json\n")
    assert await c.line() == {"ok": False, "error": "malformed JSON request"}
    await c.send(b"[1, 2]\n")
    assert (await c.line())["ok"] is False
    await c.send({"cmd": "dance"})
    assert await c.line() == {"ok": False, "error": "unknown command: dance"}
    await c.send({"cmd": "say", "text": ""})
    assert (await c.line())["ok"] is False
    await c.send({"cmd": "status"})
    assert (await c.line())["ok"] is True
    await c.close()


async def test_second_daemon_refuses_live_socket(daemon, sock_dir):
    other = make_daemon(sock_dir)
    with pytest.raises(RuntimeError, match="already running"):
        await other.start()
    await other.close()
    assert daemon.socket_path.exists()  # the failed one did not remove it


async def test_sigterm_shuts_down_cleanly(sock_dir):
    worker = FakeWorker()
    d = make_daemon(sock_dir, worker=worker)
    task = asyncio.create_task(d.run())
    while not d.socket_path.exists():
        await asyncio.sleep(0.01)
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(task, 2)
    assert d.session.brain.closed and worker.closed
    assert not d.socket_path.exists()


async def test_warmups_gate_turns(sock_dir):
    gate = asyncio.Event()
    d = make_daemon(sock_dir, warmups=[gate.wait])
    await d.start()
    try:
        assert (await call(d, {"cmd": "status"}))[-1]["ready"] is False
        c = await Client.open(d)
        await c.send({"cmd": "say", "text": "hi", "quiet": True})
        await asyncio.sleep(0.05)
        assert d.session.brain.turns == []
        gate.set()
        assert (await c.until_done())[-1]["reply"] == "Okay."
        await c.close()
    finally:
        await d.close()


# --- CLI against a daemon on fakes ---


async def run_cli(capsys, *argv):
    code = await asyncio.to_thread(cli.main, list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


async def test_cli_say_status_and_errors(daemon, capsys):
    sock = str(daemon.socket_path)
    code, out, _ = await run_cli(capsys, "--socket", sock, "say", "--quiet", "hello", "there")
    assert code == 0
    assert "you: hello there" in out and "hark: Okay." in out and "timing: path=talker" in out
    code, out, _ = await run_cli(capsys, "--socket", sock, "status")
    assert code == 0 and "state: idle" in out and "talker: local" in out
    code, out, _ = await run_cli(capsys, "--socket", sock, "--json", "say", "hi", "--quiet")
    lines = [json.loads(x) for x in out.splitlines()]
    assert lines[-1]["done"] is True
    code, _, err = await run_cli(capsys, "--socket", sock, "yes")
    assert code == 1 and "nothing to confirm" in err
    code, _, err = await run_cli(capsys, "--socket", sock, "set", "brain.talker", "bad")
    assert code == 1 and "could not switch" in err
    code, out, _ = await run_cli(capsys, "--socket", sock, "wav", str(FIXTURE), "--quiet")
    assert code == 0 and "you: turn the volume up" in out


async def test_cli_ptt_and_stop(daemon, capsys):
    sock = str(daemon.socket_path)
    assert (await run_cli(capsys, "--socket", sock, "ptt", "start"))[0] == 0
    code, out, _ = await run_cli(capsys, "--socket", sock, "--quiet", "ptt", "stop")
    assert code == 0 and "you: turn the volume up" in out
    assert (await run_cli(capsys, "--socket", sock, "stop"))[0] == 0


async def test_cli_events(daemon, capsys):
    task = asyncio.create_task(asyncio.to_thread(cli.main, ["--socket", str(daemon.socket_path), "events"]))
    await asyncio.sleep(0.1)
    await call(daemon, {"cmd": "say", "text": "hello", "quiet": True})
    await asyncio.sleep(0.05)
    await daemon.close()  # ends the stream
    assert await asyncio.wait_for(task, 2) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert lines[0] == {"ok": True, "subscribed": True}
    assert any(m.get("event") == "timing" for m in lines)


async def test_cli_daemon_not_running(sock_dir, capsys):
    code, _, err = await run_cli(capsys, "--socket", str(sock_dir / "none.sock"), "status")
    assert code == 2 and "not running" in err


async def test_tool_command_runs_daemon_tools(daemon):
    (reply,) = await call(daemon, {"cmd": "tool", "name": "set_volume", "args": {"level": 30}})
    assert reply == {"ok": True, "result": {"ok": True, "text": "Volume is 30 percent."}}
    assert daemon.calls == [("set_volume", {"level": 30})]
    (reply,) = await call(daemon, {"cmd": "tool", "name": "set_volume", "args": "x"})
    assert reply["ok"] is False


async def test_mcp_server_proxies_daemon_tools(daemon, monkeypatch):
    from hark.tools import mcp_server

    monkeypatch.setattr(mcp_server.config, "SOCKET_PATH", daemon.socket_path)
    tools = {t.name: t for t in mcp_server.daemon_tools()}
    assert {"remember", "recall", "forget", "delegate"} <= set(tools)
    res = await tools["remember"].fn(fact="I like tea.")  # the fake registry has no memory
    assert res.ok is False and "no tool called remember" in res.text


# --- fixes ---


async def test_confirm_from_a_process_the_daemon_started_is_refused(daemon, tmp_path):
    """Claude's shell is a child of the daemon: it must not be able to say yes."""
    c = await Client.open(daemon)
    await c.send({"cmd": "ask_confirm", "question": "Run rm -rf ~/Work?"})
    await asyncio.sleep(0.05)
    pending = (await call(daemon, {"cmd": "status"}))[-1]["confirm"]
    script = (
        "import json, socket, sys\n"
        "s = socket.socket(socket.AF_UNIX); s.connect(sys.argv[1])\n"
        "for req in sys.argv[2:]:\n"
        "    s.sendall((req + '\\n').encode())\n"
        "    print(s.makefile().readline().strip())\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, str(daemon.socket_path),
        json.dumps({"cmd": "confirm", "answer": True, "id": pending["id"]}),
        json.dumps({"cmd": "say", "text": "yes"}),
        stdout=asyncio.subprocess.PIPE)
    out, _ = await proc.communicate()
    confirm_reply, say_reply = [json.loads(x) for x in out.decode().splitlines()]
    assert confirm_reply == {"ok": False, "error": "only the user can answer a question"}
    assert say_reply["ok"] is False and "waiting" in say_reply["error"]
    assert daemon.session.pending_confirm is not None
    # the user's own client (here: this process) still answers it
    assert (await call(daemon, {"cmd": "confirm", "answer": False, "id": pending["id"]}))[-1] == {"ok": True}
    assert await c.line() == {"ok": True, "answer": False}
    await c.close()


async def test_stale_confirm_id_does_not_approve_a_newer_question(daemon):
    c = await Client.open(daemon)
    await c.send({"cmd": "ask_confirm", "question": "First?"})
    await asyncio.sleep(0.05)
    first = (await call(daemon, {"cmd": "status"}))[-1]["confirm"]["id"]
    await call(daemon, {"cmd": "confirm", "answer": False, "id": first})
    assert (await c.line())["answer"] is False
    await c.send({"cmd": "ask_confirm", "question": "Second?"})
    await asyncio.sleep(0.05)
    reply = (await call(daemon, {"cmd": "confirm", "answer": True, "id": first}))[-1]
    assert reply["ok"] is False and "no longer" in reply["error"]
    assert daemon.session.pending_confirm["question"] == "Second?"
    daemon.session.answer(False)
    await c.line()
    await c.close()


async def test_subscriber_that_stops_reading_never_blocks_turns(sock_dir, monkeypatch):
    from hark import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "OUTBOX_LIMIT", 50)
    brain = FakeBrain(lambda t: [(0.0, TextDelta("word " * 2000)), (0.0, TurnDone())])
    d = make_daemon(sock_dir, brain)
    await d.start()
    try:
        # A raw socket that never reads, like a suspended bar widget.
        stuck = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        stuck.connect(str(d.socket_path))
        stuck.sendall(b'{"cmd": "subscribe"}\n')
        await asyncio.sleep(0.05)
        for i in range(30):
            final = (await asyncio.wait_for(call(d, {"cmd": "say", "text": f"turn {i}", "quiet": True}), 2.0))[-1]
            assert final["done"]
        t = time.perf_counter()
        assert (await asyncio.wait_for(call(d, {"cmd": "stop"}), 1.0))[-1] == {"ok": True}
        assert time.perf_counter() - t < 0.5
        assert not d._subs  # the stuck subscriber was dropped
        stuck.close()
    finally:
        await d.close()


async def test_oversized_request_gets_an_error(daemon):
    c = await Client.open(daemon)
    await c.send({"cmd": "say", "text": "x" * (5 << 20)})
    reply = await c.line(timeout=5)
    assert reply["ok"] is False and "too large" in reply["error"]
    await c.close()


async def test_cli_reports_a_dropped_connection(sock_dir, capsys):
    async def hang_up(reader, writer):
        await reader.readline()
        writer.close()

    path = sock_dir / "x.sock"
    server = await asyncio.start_unix_server(hang_up, path=str(path))
    try:
        code, _, err = await run_cli(capsys, "--socket", str(path), "say", "hi")
        assert code == 1 and "without an answer" in err
    finally:
        server.close()


async def test_ptt_with_no_audio_is_an_error(sock_dir):
    d = make_daemon(sock_dir)
    d.audio_in.pcm = np.zeros(0, np.float32)
    d.audio_in.error = "no microphone device is available (no target node available)"
    await d.start()
    try:
        await call(d, {"cmd": "ptt_start"})
        final = (await call(d, {"cmd": "ptt_stop", "quiet": True}))[-1]
        assert final["ok"] is False and "no microphone" in final["error"]
        st = (await call(d, {"cmd": "status"}))[-1]
        assert "no microphone" in st["errors"]["microphone"] and st["state"] == "idle"
    finally:
        await d.close()


class StreamingSTT:
    """Fake streaming engine that records how audio arrived."""

    def __init__(self):
        self.calls: list[str] = []
        self.samples = 0

    def start(self):
        self.calls.append("start")
        self.samples = 0

    def feed(self, pcm):
        self.calls.append("feed")
        self.samples += len(pcm)
        return ""

    def finish(self):
        self.calls.append("finish")
        return f"heard {self.samples} samples"

    def transcribe(self, pcm):
        self.calls.append("transcribe")
        return "whole"


async def test_wav_is_fed_in_chunks_to_a_streaming_engine(daemon):
    stt = daemon.session.stt = StreamingSTT()
    final = (await call(daemon, {"cmd": "wav", "path": str(FIXTURE), "quiet": True}))[-1]
    n = len(sf.read(FIXTURE)[0])
    assert final["ok"] and "transcribe" not in stt.calls
    assert stt.calls[0] == "start" and stt.calls[-1] == "finish" and stt.calls.count("feed") > 1
    assert stt.samples == n and final["timing"]["stt_ms"] is not None


async def test_ptt_feeds_chunks_while_recording(sock_dir):
    d = make_daemon(sock_dir)
    stt = d.session.stt = StreamingSTT()
    chunks = asyncio.Queue()

    async def read():
        return await chunks.get()

    async def stop():
        chunks.put_nowait(None)
        return np.zeros(3 * 512, np.float32)

    d.audio_in.read, d.audio_in.stop = read, stop
    await d.start()
    try:
        await call(d, {"cmd": "ptt_start"})
        for _ in range(3):
            chunks.put_nowait(np.zeros(512, np.float32))
        await asyncio.sleep(0.05)
        assert stt.calls.count("feed") == 3  # decoded while the key is still down
        final = (await call(d, {"cmd": "ptt_stop", "quiet": True}))[-1]
        assert final["ok"] and stt.calls[-1] == "finish" and stt.samples == 3 * 512
    finally:
        await d.close()


async def test_shutdown_runs_the_memory_job_bounded(sock_dir, monkeypatch):
    from hark import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "SHUTDOWN_JOB_S", 0.3)
    d = make_daemon(sock_dir)
    ended = []

    async def on_end(sid, text):
        ended.append(text)

    d.session.on_session_end = on_end
    await d.start()
    await call(d, {"cmd": "say", "text": "hello", "quiet": True})
    await d.close()
    assert ended and "user: hello" in ended[0]

    d = make_daemon(sock_dir)

    async def slow(sid, text):
        await asyncio.sleep(10)

    d.session.on_session_end = slow
    await d.start()
    await call(d, {"cmd": "say", "text": "hello", "quiet": True})
    t = time.perf_counter()
    await d.close()
    assert time.perf_counter() - t < 1.0

    d = make_daemon(sock_dir)  # an empty session runs no job
    d.session.on_session_end = on_end
    await d.start()
    await d.close()
    assert len(ended) == 1


async def test_idle_job_waits_for_quiet_and_yields_to_turns(sock_dir, monkeypatch):
    from hark import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "IDLE_JOB_GRACE_S", 0.05)
    monkeypatch.setattr(daemon_mod, "IDLE_POLL_S", 0.01)
    runs = []

    async def job():
        runs.append("start")
        await asyncio.sleep(0.3)
        runs.append("done")

    async def until(pred):
        for _ in range(200):
            if pred():
                return
            await asyncio.sleep(0.01)
        raise AssertionError(runs)

    brain = FakeBrain([(0.2, TextDelta("ok")), (0.0, TurnDone())])
    d = make_daemon(sock_dir, brain, idle_jobs=[job])
    await d.start()
    try:
        await until(lambda: runs == ["start"])
        await call(d, {"cmd": "say", "text": "hi", "quiet": True})  # cancels the job
        assert runs == ["start"]
        await until(lambda: runs[-1] == "done")
        assert runs == ["start", "start", "done"]
    finally:
        await d.close()


async def test_failed_warmup_shows_in_status(sock_dir):
    async def broken():
        raise RuntimeError("llama-server exited during startup: model not found")

    d = make_daemon(sock_dir, warmups={"talker": broken})
    await d.start()
    try:
        await d.ready.wait()
        st = (await call(d, {"cmd": "status"}))[-1]
        assert "model not found" in st["errors"]["talker"] and st["confirm"] is None
        d.session.brain.error = None  # a later turn started it after all
        assert "talker" not in (await call(d, {"cmd": "status"}))[-1]["errors"]
    finally:
        await d.close()


async def test_cli_yes_answers_the_waiting_question_by_id(daemon, capsys):
    c = await Client.open(daemon)
    await c.send({"cmd": "ask_confirm", "question": "Delete the build folder?"})
    await asyncio.sleep(0.05)
    code, out, _ = await run_cli(capsys, "--socket", str(daemon.socket_path), "yes")
    assert code == 0 and "yes to: Delete the build folder?" in out
    assert await c.line() == {"ok": True, "answer": True}
    await c.close()
