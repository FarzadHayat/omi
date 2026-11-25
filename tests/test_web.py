"""The phone and tablet page: who gets in, audio framing both ways, and whole turns over
a WebSocket against a daemon built from fakes."""

import asyncio
import json
import os
import shutil
import socket
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from hark.config import Config, WebConfig
from hark.daemon import Daemon
from hark.events import TextDelta, TurnDone
from hark.protocol import WEB_AUDIO_HEADER
from hark.session import Session
from hark.web import server as web
from hark.web.websocket import connect
from tests.fakes import CaptureOut, FakeAudioIn, FakeBrain, FakeRouter, FakeSTT, FakeTTS, Route, make_registry

ME = "me@example.com"


@pytest.fixture
def sock_dir():
    d = Path(tempfile.mkdtemp(prefix="hk", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make(sock_dir, brain=None, peer_ok=lambda sock: True, origins=None):
    calls: list = []
    reg = make_registry(calls)
    out = CaptureOut(speed=50)
    session = Session(
        stt=FakeSTT("turn the volume up"),
        tts=FakeTTS(),
        brain=brain or FakeBrain([(0.0, TextDelta("Sure, all done.")), (0.0, TurnDone())]),
        registry=reg,
        router=FakeRouter({"close it": Route("close_window", {}, "Closed.")}),
        out=out,
        state_dir=sock_dir,
    )
    reg.confirm = session.confirm
    d = Daemon(session, Config(), audio_in=FakeAudioIn(), socket_path=sock_dir / "run" / "hark.sock")
    cfg = WebConfig(port=0, origins=origins or [])
    d.web = web.WebServer(d, cfg, logins={ME}, peer_ok=peer_ok)
    d.out, d.calls = out, calls
    return d


@pytest.fixture
async def daemon(sock_dir):
    d = make(sock_dir)
    await d.start()
    yield d
    await d.close()


async def http_get(port, path, headers=None):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    lines = [f"GET {path} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Connection: close"]
    lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
    await writer.drain()
    raw = await asyncio.wait_for(reader.read(), 3)
    writer.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b" ")[1])
    hdrs = dict(line.split(": ", 1) for line in head.decode().split("\r\n")[1:])
    return status, hdrs, body


def ws_headers(port, login=ME, origin=None):
    h = {"Origin": origin if origin is not None else f"https://127.0.0.1:{port}"}
    if login:
        h["Tailscale-User-Login"] = login
    return h


class Phone:
    """A test phone: collects text events and audio frames, acks played audio."""

    def __init__(self, ws):
        self.ws = ws
        self.events: list[dict] = []
        self.frames: list[tuple[float, bytes]] = []
        self._new = asyncio.Event()
        self._task = asyncio.create_task(self._read())
        self.ack = True

    @classmethod
    async def open(cls, d, **kw):
        port = d.web.port
        return cls(await connect(f"ws://127.0.0.1:{port}/ws", ws_headers(port, **kw)))

    async def _read(self):
        while (msg := await self.ws.recv()) is not None:
            if isinstance(msg, bytes):
                self.frames.append((time.perf_counter(), msg))
            else:
                ev = json.loads(msg)
                self.events.append(ev)
                if self.ack and ev.get("event") == "audio" and ev.get("phase") == "end":
                    await self.send({"cmd": "played", "seq": ev["seq"]})
            self._new.set()

    async def send(self, obj):
        await self.ws.send(json.dumps(obj) if isinstance(obj, dict) else obj)

    async def wait(self, pred, timeout=3.0):
        end = time.monotonic() + timeout
        while True:
            hit = next((e for e in self.events if pred(e)), None)
            if hit is not None:
                return hit
            if time.monotonic() > end:
                raise AssertionError(f"no matching event in {self.events}")
            self._new.clear()
            try:
                await asyncio.wait_for(self._new.wait(), max(0.01, end - time.monotonic()))
            except TimeoutError:
                pass

    def kinds(self, kind):
        return [e for e in self.events if e.get("event") == kind]

    async def close(self):
        await self.ws.close()
        self._task.cancel()


def mic_frames(seconds=0.4):
    pcm = (np.sin(np.arange(int(16000 * seconds)) / 8) * 8000).astype("<i2")
    return [pcm[i:i + 320].tobytes() for i in range(0, len(pcm), 320)]


# --- who gets in ---------------------------------------------------------------------


async def test_page_needs_an_allowed_tailscale_login(daemon):
    port = daemon.web.port
    status, _, _ = await http_get(port, "/")
    assert status == 403
    status, _, _ = await http_get(port, "/", {"Tailscale-User-Login": "someone@else.com"})
    assert status == 403
    status, hdrs, body = await http_get(port, "/", {"Tailscale-User-Login": ME})
    assert status == 200 and b"<title>Hark</title>" in body
    assert "default-src 'self'" in hdrs["Content-Security-Policy"]
    for path in ("/app.js", "/mic-worklet.js", "/sw.js", "/style.css", "/manifest.webmanifest", "/icon-192.png"):
        status, _, _ = await http_get(port, path, {"Tailscale-User-Login": ME})
        assert status == 200, path
    status, _, _ = await http_get(port, "/../config.py", {"Tailscale-User-Login": ME})
    assert status == 404


async def test_server_listens_on_loopback_only(daemon):
    assert daemon.web._server.sockets[0].getsockname()[0] == "127.0.0.1"


async def test_direct_connection_that_did_not_come_through_tailscaled_is_refused(sock_dir):
    d = make(sock_dir, peer_ok=None)  # the real check: the peer process must be root (tailscaled)
    await d.start()
    try:
        status, _, body = await http_get(d.web.port, "/", {"Tailscale-User-Login": ME})
        assert status == 403 and b"tailscale serve" in body
        with pytest.raises(ConnectionError, match="403"):
            await connect(f"ws://127.0.0.1:{d.web.port}/ws", ws_headers(d.web.port))
    finally:
        await d.close()


async def test_peer_uid_reads_the_owner_of_the_other_end():
    srv = socket.create_server(("127.0.0.1", 0))
    cli = socket.create_connection(srv.getsockname())
    conn, _ = srv.accept()
    try:
        assert web.peer_uid(conn) == os.getuid()
    finally:
        for s in (conn, cli, srv):
            s.close()


async def test_websocket_needs_login_and_the_page_origin(daemon):
    port = daemon.web.port
    url = f"ws://127.0.0.1:{port}/ws"
    with pytest.raises(ConnectionError, match="403"):
        await connect(url, ws_headers(port, login=None))
    with pytest.raises(ConnectionError, match="403"):
        await connect(url, ws_headers(port, login="intruder@example.com"))
    with pytest.raises(ConnectionError, match="403"):
        await connect(url, ws_headers(port, origin="https://evil.example"))
    with pytest.raises(ConnectionError, match="403"):
        await connect(url, ws_headers(port, origin=""))
    with pytest.raises(ConnectionError, match="403"):  # http origin: only https pages
        await connect(url, ws_headers(port, origin=f"http://127.0.0.1:{port}"))
    ws = await connect(url, ws_headers(port))
    assert json.loads(await ws.recv())["event"] == "hello"
    await ws.close()


async def test_configured_origin_is_accepted(sock_dir):
    d = make(sock_dir, origins=["https://hark.example"])
    await d.start()
    try:
        port = d.web.port
        ws = await connect(f"ws://127.0.0.1:{port}/ws", ws_headers(port, origin="https://hark.example"))
        await ws.close()
    finally:
        await d.close()


async def test_duplicate_login_headers_are_refused(daemon):
    reader, writer = await asyncio.open_connection("127.0.0.1", daemon.web.port)
    writer.write(f"GET / HTTP/1.1\r\nHost: x\r\nTailscale-User-Login: {ME}\r\nTailscale-User-Login: a@b.c\r\n\r\n".encode())
    raw = await asyncio.wait_for(reader.read(), 3)
    writer.close()
    assert raw.startswith(b"HTTP/1.1 400")


def test_owner_login_from_tailscale_status():
    status = {"Self": {"UserID": 42}, "User": {"42": {"LoginName": "owner@example.com"}}}
    assert web.owner_login(status) == "owner@example.com"
    assert web.owner_login({}) is None
    assert web.owner_login({"Self": {"UserID": 1}, "User": {}}) is None


async def test_no_allowed_login_shows_in_status(sock_dir, monkeypatch):
    d = make(sock_dir)
    d.web.logins = None

    async def nobody():
        return None

    monkeypatch.setattr(web, "tailnet_owner", nobody)
    await d.start()
    try:
        assert "web" in d.status()["errors"]
        status, _, _ = await http_get(d.web.port, "/", {"Tailscale-User-Login": ME})
        assert status == 403
    finally:
        await d.close()


async def test_busy_port_does_not_stop_the_daemon(sock_dir):
    busy = socket.create_server(("127.0.0.1", 0))
    d = make(sock_dir)
    d.web.port = busy.getsockname()[1]
    try:
        await d.start()
        assert "web" in d.status()["errors"]
        res = await d.session.run_turn("hello", quiet=True)
        assert res.reply
    finally:
        await d.close()
        busy.close()


# --- framing -------------------------------------------------------------------------


def test_mic_frames_decode_to_float():
    pcm = np.array([0, 16384, -32768, 32767], dtype="<i2")
    out = web.decode_mic(pcm.tobytes() + b"\x01")  # a stray odd byte is ignored
    assert out.dtype == np.float32
    assert np.allclose(out, [0.0, 0.5, -1.0, 32767 / 32768])


def test_reply_frames_carry_rate_and_turn():
    pcm = np.array([0.0, 0.5, -1.0, 2.0], dtype=np.float32)
    frame = web.encode_reply(pcm, 22050, 7)
    assert struct.unpack(WEB_AUDIO_HEADER, frame[:8]) == (22050, 7)
    assert len(frame) == 8 + 2 * len(pcm)
    rate, seq, back = web.decode_reply(frame)
    assert (rate, seq) == (22050, 7)
    assert np.allclose(back, [0.0, 0.5, -1.0, 1.0], atol=1e-4)  # clipped to [-1, 1]


def test_theme_css_reads_omarchy_colours_and_ignores_bad_values(tmp_path):
    p = tmp_path / "colors.toml"
    p.write_text('mode = "dark"\nbackground = "#1c1c1e"\naccent = "#0a84ff"\nred = "red; } body { x: y"\n')
    css = web.theme_css(p)
    assert "--bg: #1c1c1e;" in css and "--accent: #0a84ff;" in css and "--red" not in css and "body" not in css
    assert web.theme_css(tmp_path / "missing.toml") == ""


async def test_theme_css_is_served(sock_dir, tmp_path):
    d = make(sock_dir)
    (tmp_path / "colors.toml").write_text('accent = "#abcdef"\n')
    d.web.cfg.theme = str(tmp_path / "colors.toml")
    await d.start()
    try:
        status, hdrs, body = await http_get(d.web.port, "/theme.css", {"Tailscale-User-Login": ME})
        assert status == 200 and hdrs["Content-Type"].startswith("text/css") and b"#abcdef" in body
    finally:
        await d.close()


# --- turns ---------------------------------------------------------------------------


async def test_phone_voice_turn_plays_on_the_phone_only(daemon):
    phone = await Phone.open(daemon)
    await phone.wait(lambda e: e["event"] == "hello")
    await phone.send({"cmd": "ptt_start"})
    await phone.wait(lambda e: e.get("state") == "listening")
    for f in mic_frames():
        await phone.send(f)
    await phone.send({"cmd": "ptt_stop"})
    done = await phone.wait(lambda e: e["event"] == "done")
    assert done["ok"] and done["reply"] == "Sure, all done." and not done["cancelled"]
    texts = [(e["role"], e["text"]) for e in phone.kinds("transcript")]
    assert ("user", "turn the volume up") in texts and ("assistant", "Sure, all done.") in texts
    states = [e["state"] for e in phone.kinds("state")]
    assert states[:3] == ["listening", "idle", "thinking"] and "speaking" in states and states[-1] == "idle"
    assert phone.frames
    for _, frame in phone.frames:
        rate, seq, pcm = web.decode_reply(frame)
        assert rate == 16000 and seq == 1 and len(pcm)
    assert [e["phase"] for e in phone.kinds("audio")] == ["end"]
    assert daemon.out.writes == []  # nothing on the PC speakers
    assert daemon.session.stt.calls == 1
    await phone.close()


async def test_desktop_turn_is_not_played_on_the_phone(daemon):
    phone = await Phone.open(daemon)
    await phone.wait(lambda e: e["event"] == "hello")
    res = await daemon.session.run_turn("hello")
    assert res.reply == "Sure, all done."
    await phone.wait(lambda e: e.get("role") == "assistant")  # the phone sees the conversation
    assert phone.frames == [] and daemon.out.writes  # but the audio stays on the PC
    await phone.close()


async def test_typed_turn_and_drain_waits_for_the_phone(daemon):
    phone = await Phone.open(daemon)
    phone.ack = False
    await phone.wait(lambda e: e["event"] == "hello")
    t0 = time.perf_counter()
    await phone.send({"cmd": "say", "text": "hi"})
    end = await phone.wait(lambda e: e.get("event") == "audio" and e["phase"] == "end")
    await asyncio.sleep(0.3)
    assert not phone.kinds("done")  # still speaking: the phone has not finished playing
    await phone.send({"cmd": "played", "seq": end["seq"]})
    done = await phone.wait(lambda e: e["event"] == "done")
    assert done["ok"] and time.perf_counter() - t0 < web.DRAIN_SLACK_S + 0.5
    await phone.close()


async def test_stop_from_the_phone_cuts_the_reply(sock_dir):
    script = [(0.0, TextDelta("First part of a long answer, ")), (0.0, TextDelta("and more. ")),
              (5.0, TextDelta("never said")), (0.0, TurnDone())]
    d = make(sock_dir, brain=FakeBrain(script))
    await d.start()
    try:
        phone = await Phone.open(d)
        await phone.send({"cmd": "say", "text": "tell me a story"})
        await phone.wait(lambda e: e.get("state") == "speaking")
        await phone.send({"cmd": "stop"})
        await phone.wait(lambda e: e.get("event") == "audio" and e["phase"] == "stop")
        done = await phone.wait(lambda e: e["event"] == "done")
        assert done["cancelled"]
        await phone.close()
    finally:
        await d.close()


async def test_pressing_talk_while_it_speaks_interrupts(sock_dir):
    script = [(0.0, TextDelta("Here is a long answer, ")), (5.0, TextDelta("never said")), (0.0, TurnDone())]
    d = make(sock_dir, brain=FakeBrain(script))
    await d.start()
    try:
        phone = await Phone.open(d)
        await phone.send({"cmd": "say", "text": "tell me a story"})
        await phone.wait(lambda e: e.get("state") == "speaking")
        await phone.send({"cmd": "ptt_start"})
        done = await phone.wait(lambda e: e["event"] == "done")
        assert done["cancelled"]
        await phone.wait(lambda e: e.get("event") == "audio" and e["phase"] == "stop")
        assert d.session.state.value == "listening"
        await phone.send({"cmd": "stop"})
        await phone.wait(lambda e: e.get("state") == "idle")
        await phone.close()
    finally:
        await d.close()


async def test_confirm_from_the_phone_by_id(daemon):
    phone = await Phone.open(daemon)
    await phone.send({"cmd": "say", "text": "close it"})
    q = await phone.wait(lambda e: e["event"] == "confirm_request")
    await phone.send({"cmd": "confirm", "answer": True, "id": "wrong"})
    err = await phone.wait(lambda e: e["event"] == "error")
    assert "no longer" in err["text"]
    await phone.send({"cmd": "confirm", "answer": True, "id": q["id"]})
    done = await phone.wait(lambda e: e["event"] == "done")
    assert done["ok"] and daemon.calls == [("close_window", {})]
    assert daemon.out.writes == []  # the question and the ack were spoken on the phone
    await phone.close()


async def test_desktop_key_taking_over_the_utterance_leaves_the_phone_out(daemon):
    phone = await Phone.open(daemon)
    await phone.send({"cmd": "ptt_start"})
    await phone.wait(lambda e: e.get("state") == "listening")
    await daemon.session.begin_utterance()  # the desktop key, pressed meanwhile
    desktop_utt = daemon.session._utt
    for f in mic_frames(0.1):
        await phone.send(f)
    await phone.send({"cmd": "ptt_stop"})
    err = await phone.wait(lambda e: e["event"] == "error")
    assert err["text"] == "not listening" and daemon.session._utt is desktop_utt and desktop_utt.samples == 0
    await phone.close()


async def test_phone_talk_closes_the_desktop_microphone(daemon):
    daemon._listening = True  # the desktop push-to-talk key is held
    await daemon.audio_in.start()
    phone = await Phone.open(daemon)
    await phone.send({"cmd": "ptt_start"})
    await phone.wait(lambda e: e.get("state") == "listening")
    assert not daemon._listening and not daemon.audio_in.started
    await phone.send({"cmd": "stop"})
    await phone.close()


async def test_phone_hanging_up_mid_utterance_drops_it(daemon):
    phone = await Phone.open(daemon)
    await phone.send({"cmd": "ptt_start"})
    await phone.wait(lambda e: e.get("state") == "listening")
    await phone.close()
    for _ in range(50):
        if daemon.session.state.value == "idle":
            break
        await asyncio.sleep(0.02)
    assert daemon.session._utt is None and daemon.session.state.value == "idle"


async def test_bad_messages_keep_the_socket_usable(daemon):
    phone = await Phone.open(daemon)
    await phone.send("not json")
    await phone.wait(lambda e: e["event"] == "error" and "malformed" in e["text"])
    await phone.send({"cmd": "rm -rf"})
    await phone.wait(lambda e: e["event"] == "error" and "unknown command" in e["text"])
    await phone.send({"cmd": "ptt_stop"})
    await phone.wait(lambda e: e["event"] == "error" and e["text"] == "not listening")
    await phone.send({"cmd": "say", "text": "hi"})
    assert (await phone.wait(lambda e: e["event"] == "done"))["ok"]
    await phone.close()
