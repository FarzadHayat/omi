"""Omi guiding the look to talk setup out loud."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from omi import look_setup, look_tune
from omi.config import Config
from omi.daemon import Daemon
from omi.events import TurnDone
from omi.session import Session
from omi.tools.base import ToolRegistry
from omi.voice_id import Voiceprints
from tests.fakes import CaptureOut, FakeBrain, FakeSTT, FakeTTS


class _Tracker:
    mode = "quiet"

    def __init__(self, *a, **k) -> None:
        self.error = None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def wake(self) -> None: ...

    def lips_moving(self, now=None):
        return {"quiet": 0.015, "talk": 0.06, "away": None}[_Tracker.mode]

    def readings(self, since, until=None):
        return [(True, None, 0.8 if _Tracker.mode == "away" else 0.02)] * 10


class _Mic:
    def __init__(self, *a) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

    def between(self, since, until):
        return np.zeros(1600, np.float32)


class _Wake:
    """The hands-free listener: rests during the setup, comes back after."""

    def __init__(self) -> None:
        self.active = True
        self.looking_active = False
        self.stops = self.starts = 0

    async def start(self) -> None:
        self.active = True
        self.starts += 1

    async def stop(self) -> None:
        self.active = False
        self.stops += 1


@pytest.fixture
def daemon(tmp_path, monkeypatch):
    for name, value in (("QUIET_S", 0.1), ("TALK_S", 0.1), ("AWAY_S", 0.1), ("SETTLE_S", 0.0), ("TICK_S", 0.01)):
        monkeypatch.setattr(look_tune, name, value)
    monkeypatch.setattr(look_setup.look, "LookTracker", _Tracker)
    monkeypatch.setattr(look_tune, "Mic", _Mic)
    monkeypatch.setattr(look_tune, "mic_source", lambda s: "")
    _Tracker.mode = "quiet"
    session = Session(stt=FakeSTT("yes"), tts=FakeTTS(), brain=FakeBrain(lambda _: [(0.0, TurnDone())]),
                      registry=ToolRegistry(), router=lambda _: None, out=CaptureOut(), state_dir=tmp_path)
    d = Daemon(session, Config(), socket_path=tmp_path / "s.sock")
    d.voiceprints = Voiceprints(tmp_path / "v.json")
    d.wake = _Wake()
    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)
        if ev.get("event") == "progress" and "Step 2" in ev.get("text", ""):
            _Tracker.mode = "talk"
        elif ev.get("event") == "progress" and "Step 3" in ev.get("text", ""):
            _Tracker.mode = "away"

    d.broadcast = broadcast
    session.emit = broadcast
    d.events = events
    return d


async def _answer_all(session: Session, answers: list[bool], asked: list[str]) -> None:
    while answers:
        q = session.pending_confirm
        if q is not None:
            asked.append(q["question"])
            session.answer(answers.pop(0), q["id"])
        await asyncio.sleep(0.01)


async def test_omi_talks_you_through_it_and_saves_the_numbers(daemon, monkeypatch):
    monkeypatch.setattr(look_tune, "heard_you", lambda q, t: True)
    monkeypatch.setattr(look_setup.GuidedSetup, "_listen_for_answer", lambda self: asyncio.sleep(3600))
    saved_voice = []
    monkeypatch.setattr(look_setup.GuidedSetup, "_save_voice", lambda self, pcm: saved_voice.append(pcm) or asyncio.sleep(0))
    asked: list[str] = []
    answering = asyncio.create_task(_answer_all(daemon.session, [True, True], asked))
    assert daemon.look_setup.start() == "Starting the look to talk setup now."
    assert "already running" in daemon.look_setup.start()
    await asyncio.wait_for(daemon.look_setup.task, 10)
    answering.cancel()

    said = [e["text"] for e in daemon.events if e.get("event") == "transcript"]
    assert said[0].startswith("Let's tune look to talk")
    assert any("Step 2 of 3" in s and look_tune.SENTENCE in s for s in said)  # the sentence is on screen
    assert any("Look to talk is tuned" in s for s in said)
    cards = [e for e in daemon.events if e.get("event") == "tune"]
    talk = [c for c in cards if c.get("title", "").startswith("Step 2") and c["seconds"] > 0]
    assert talk and look_tune.SENTENCE in talk[0]["text"]  # the big card keeps the sentence up while you read
    assert talk[0]["seconds"] == look_tune.TALK_S
    assert any("only answer your voice" in c.get("text", "") for c in cards)  # questions show there too
    assert cards[-1] == {"event": "tune", "active": False}  # and it goes away at the end
    assert "only answer your voice" in asked[0] and "turn look to talk on" in asked[1]
    assert saved_voice and daemon.cfg.wake.look_to_talk is True
    assert daemon.wake.stops == 1 and daemon.wake.active  # listener rested, then came back
    w = daemon.cfg.wake
    assert 0.015 < w.lip_threshold < 0.06 and w.yaw_center == pytest.approx(0.02)


async def test_no_face_means_nothing_changes(daemon, monkeypatch):
    monkeypatch.setattr(look_tune, "FIND_FACE_S", 0.1)
    monkeypatch.setattr(_Tracker, "readings", lambda self, since, until=None: [])
    daemon.look_setup.start()
    await asyncio.wait_for(daemon.look_setup.task, 10)
    said = " ".join(e["text"] for e in daemon.events if e.get("event") == "transcript")
    assert "can't see your face" in said
    assert daemon.cfg.wake.lip_threshold == Config().wake.lip_threshold


async def test_a_silent_mic_skips_the_voice_question(daemon, monkeypatch):
    monkeypatch.setattr(look_setup.GuidedSetup, "_listen_for_answer", lambda self: asyncio.sleep(3600))
    asked: list[str] = []
    answering = asyncio.create_task(_answer_all(daemon.session, [False], asked))
    daemon.look_setup.start()
    await asyncio.wait_for(daemon.look_setup.task, 10)
    answering.cancel()
    said = " ".join(e["text"] for e in daemon.events if e.get("event") == "transcript")
    assert "couldn't hear you" in said
    assert asked == ["Should I turn look to talk on now?"]


async def test_stop_cancels_the_setup(daemon, monkeypatch):
    monkeypatch.setattr(look_tune, "QUIET_S", 30.0)
    daemon.look_setup.start()
    await asyncio.sleep(0.3)
    await daemon.look_setup.cancel()
    assert not daemon.look_setup.running
    cards = [e for e in daemon.events if e.get("event") == "tune"]
    assert cards[-2]["text"] == "Cancelled. Nothing was changed." and cards[-1]["active"] is False
    assert daemon.cfg.wake.lip_threshold == Config().wake.lip_threshold


async def test_a_spoken_yes_answers_the_question(daemon, monkeypatch):
    """Real voice detection on synthetic speech, a fake transcript of "yes"."""
    from omi.tts.kokoro import KokoroTTS

    try:
        cfg = Config().tts
        pcm24 = await asyncio.to_thread(lambda: np.concatenate(list(KokoroTTS(cfg).synth("Yes, please."))))
    except FileNotFoundError:
        pytest.skip("Kokoro voice model not installed")
    speech = np.interp(np.arange(0, len(pcm24), 1.5), np.arange(len(pcm24)), pcm24).astype(np.float32)
    audio = np.concatenate([np.zeros(8000, np.float32), speech, np.zeros(24000, np.float32)])
    t0 = [None]

    class _SpeakingMic(_Mic):
        def between(self, since, until):

            t0[0] = t0[0] or since
            a, b = int((since - t0[0]) * 16000), int((until - t0[0]) * 16000)
            return audio[max(0, a):max(0, b)]

    monkeypatch.setattr(look_tune, "Mic", _SpeakingMic)
    assert await asyncio.wait_for(daemon.look_setup._ask("Want me to only answer your voice?"), 10) is True


async def test_choosing_a_camera_by_voice(daemon, monkeypatch):
    monkeypatch.setattr(look_setup.look, "list_cameras", lambda: [{"id": "/dev/video0", "name": "Built-in"},
                                                                  {"id": "/dev/video2", "name": "Desk Cam"}])
    monkeypatch.setattr(daemon, "_persist", lambda *a: None)
    tool = look_setup.camera_tool(daemon)
    listed = await tool.fn()
    assert listed.ok and "Built-in, Desk Cam" in listed.text and "automatic" in listed.text
    picked = await tool.fn(camera="desk")
    assert picked.ok and "Desk Cam" in picked.text and daemon.cfg.wake.camera == "/dev/video2"
    assert not (await tool.fn(camera="phone")).ok and daemon.cfg.wake.camera == "/dev/video2"
    assert (await tool.fn(camera="auto")).ok and daemon.cfg.wake.camera == ""
