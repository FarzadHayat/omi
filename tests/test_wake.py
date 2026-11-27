"""Tests for local wake-word detection, BPE phrase encoding, WakeListener capture, and live daemon toggling."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omi import config
from omi.audio import wake
from omi.daemon import Daemon
from omi.events import State, TextDelta, TurnDone
from omi.session import Session
from omi.tools.base import ToolRegistry
from tests.fakes import CaptureOut, FakeBrain, FakeSTT, FakeTTS


def test_encode_phrase_and_keywords_text(tmp_path):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text(
        "<blk> 0\n▁HE 10\nY 11\n▁O 12\nM 13\nI 14\n▁OH 15\n▁ME 16\n▁COM 17\nP 18\nU 19\nT 20\nE 21\nR 22\n",
        encoding="utf-8",
    )
    vocab = wake.load_vocab(tokens)
    assert wake.encode_phrase("hey omi", vocab, "hey_omi") == "▁HE Y ▁O M I @hey_omi"
    text = wake.build_keywords_text(["omi", "hey omi", "computer"], tokens)
    lines = text.splitlines()
    assert "▁O M I @omi" in lines
    assert "▁HE Y ▁O M I @hey_omi" in lines
    assert "▁HE Y ▁OH ▁ME @hey_omi" in lines
    assert "▁COM P U T E R @computer" in lines


def test_wake_chime_pcm():
    from omi.audio import sfx

    for name in sfx.SOUNDS:
        pcm = sfx.render(name, 16000)
        assert pcm.dtype == np.float32
        assert 800 <= len(pcm) <= 6000
        assert 0.02 < float(np.max(np.abs(pcm))) <= 0.25
    assert len(wake.wake_chime_pcm(16000)) > 0
    assert len(wake.sleep_chime_pcm(16000)) > 0


@pytest.mark.asyncio
async def test_session_plays_heard_and_acting_sfx(tmp_path):
    from omi.events import ToolEnd, ToolStart

    played: list[str] = []
    session = Session(
        stt=FakeSTT("check my terminal"),
        tts=FakeTTS(),
        brain=FakeBrain(lambda _: [
            (0.0, ToolStart("1", "terminal_read", {})),
            (0.0, ToolEnd("1", "terminal_read", True, "ok")),
            (0.0, TextDelta("Looks clean.")),
            (0.0, TurnDone()),
        ]),
        registry=ToolRegistry(),
        router=lambda _: None,
        out=CaptureOut(),
        state_dir=tmp_path,
        play_sfx=played.append,
    )
    await session.start()
    try:
        res = await session.run_turn(pcm=np.ones(1600, dtype=np.float32))
        assert res.reply == "Looks clean."
        assert played == ["heard", "acting"]
        played.clear()
        await session.run_turn(pcm=np.ones(1600, dtype=np.float32), quiet=True)
        assert played == []
    finally:
        await session.close()


def test_is_goodbye_and_router_goodbye():
    from omi import router

    for yes in ("goodbye", "Bye!", "see ya", "See you later.", "thanks omi, that's all",
                "I'm done", "hang up", "stop listening", "go to sleep", "never mind"):
        assert wake.is_goodbye(yes), yes
    for no in ("what does goodbye mean in French", "tell me a story", "open the browser",
               "remind me to say goodbye to Sam at five", "hello there"):
        assert not wake.is_goodbye(no), no
    r = router.route("see ya")
    assert r is not None and r.tool == "goodbye"
    assert router.route("go to sleep") is None  # kept out of fast router per test_router


def test_smart_call_mode_decides_per_turn():
    fn = wake.should_continue_conversation
    # One-shot routed OS commands drop straight back to standby
    assert not fn("auto", user_text="turn the volume up", reply="Volume 60.", path="router", action_ms=15.0)
    assert not fn("auto", user_text="open spotify", reply="Opening Spotify.", path="router", action_ms=20.0)
    assert not fn("auto", user_text="what time is it", reply="It's 9:10 AM.", path="router", action_ms=1.0)
    # Short closed tool confirmations from the LLM also drop back
    assert not fn("auto", user_text="switch to gruvbox theme", reply="Switched to Gruvbox.", path="talker", action_ms=120.0)
    # Chat, greetings, explanations, and follow-up questions stay open like a phone call
    assert fn("auto", user_text="how are you", reply="Doing well! What can I help you with?", path="talker")
    assert fn("auto", user_text="explain wayland vs x11", reply="Wayland lets apps render straight to the compositor.", path="talker")
    assert fn("auto", user_text="what's failing in my terminal", reply="Pytest failed on test_wake. Want me to inspect it?", path="talker", action_ms=80.0)
    assert fn("auto", user_text="let's brainstorm app names", reply="Sure.", path="talker", action_ms=10.0)
    # Goodbyes always end even in continuous mode
    assert not fn("continuous", user_text="see ya", reply="See ya.", path="router")
    assert fn("continuous", user_text="turn the volume up", reply="Volume 60.", path="router", action_ms=10.0)
    assert not fn("single", user_text="how are you", reply="Good! You?", path="talker")


def test_wake_detector_real_model_hey_omi():
    cfg = config.WakeConfig()
    if not all((Path(cfg.model_dir) / f).is_file() for f in wake.MODEL_FILES):
        pytest.skip("KWS model not installed")
    from omi import tts

    eng = tts.create(config.TtsConfig(engine="kokoro", kokoro_voice="af_heart"))

    def synth_16k(text: str) -> np.ndarray:
        raw = np.concatenate(list(eng.synth(text)))
        n = int(round(len(raw) * 16000 / eng.sample_rate))
        return np.interp(np.linspace(0, len(raw) - 1, n), np.arange(len(raw)), raw).astype(np.float32)

    det = wake.WakeDetector(cfg)
    pos = np.concatenate([synth_16k("Hey Omi, what time is it?"), np.zeros(8000, dtype=np.float32)])
    hits = [det.feed(pos[i : i + 512]) for i in range(0, len(pos), 512)]
    assert any(h == "hey_omi" for h in hits), hits

    det.reset()
    neg = np.concatenate([synth_16k("Hello everyone, welcome back to the show."), np.zeros(8000, dtype=np.float32)])
    neg_hits = [det.feed(neg[i : i + 512]) for i in range(0, len(neg), 512)]
    assert not any(neg_hits), neg_hits


class _FakeStreamAudioIn:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = list(chunks)
        self._idx = 0
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def read(self) -> np.ndarray | None:
        if self._idx < len(self._chunks):
            c = self._chunks[self._idx]
            self._idx += 1
            await asyncio.sleep(0.005)
            return c
        await asyncio.sleep(0.05)
        return None

    async def stop(self) -> np.ndarray:
        self.stopped = True
        return np.zeros(0, np.float32)


class _ScriptedDetector:
    def __init__(self, trigger_at: int) -> None:
        self.trigger_at = trigger_at
        self.calls = 0
        self.resets = 0

    def feed(self, pcm: np.ndarray) -> str:
        self.calls += 1
        return "hey_omi" if self.calls == self.trigger_at else ""

    def reset(self) -> None:
        self.resets += 1


class _ScriptedVAD:
    def __init__(self, speak_chunks: int = 2, end_after: int = 4) -> None:
        self.speak_chunks = speak_chunks
        self.end_after = end_after
        self.count = 0
        self.speaking = False
        self.ended = False

    def reset(self) -> None:
        self.count = 0
        self.speaking = False
        self.ended = False

    def feed(self, chunk: np.ndarray) -> None:
        self.count += 1
        self.speaking = self.count <= self.speak_chunks
        if self.count >= self.end_after:
            self.ended = True


@pytest.mark.asyncio
async def test_wake_listener_captures_command_and_runs_turn(tmp_path):
    replies: list[str] = []
    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

    session = Session(
        stt=FakeSTT("what time is it"),
        tts=FakeTTS(),
        brain=FakeBrain(lambda text: [(0.0, TextDelta(f"Heard: {text}")), (0.0, TurnDone())]),
        registry=ToolRegistry(),
        router=lambda _: None,
        out=CaptureOut(),
        emit=broadcast,
        state_dir=tmp_path,
    )
    await session.start()
    chunks = [np.ones(512, dtype=np.float32) * 0.05 for _ in range(12)]
    audio = _FakeStreamAudioIn(chunks)
    chimes = []

    async def fake_chime() -> None:
        chimes.append(True)

    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, chime=True, conversational=False),
        broadcast=broadcast,
        audio_in_factory=lambda: audio,
        detector=_ScriptedDetector(trigger_at=2),
        vad_factory=lambda: _ScriptedVAD(speak_chunks=2, end_after=4),
        chime_fn=fake_chime,
    )
    await listener.start()
    try:
        for _ in range(80):
            if any(e.get("event") == "transcript" and e.get("role") == "assistant" for e in events):
                break
            await asyncio.sleep(0.02)
        assert any(e.get("event") == "wake" and e.get("phrase") == "hey_omi" for e in events)
        assert chimes == [True]
        assistant = [e["text"] for e in events if e.get("event") == "transcript" and e.get("role") == "assistant"]
        assert assistant == ["Heard: what time is it"]
    finally:
        await listener.stop()
        await session.close()


@pytest.mark.asyncio
async def test_wake_listener_cancels_when_user_says_nothing(tmp_path):
    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

    session = Session(
        stt=FakeSTT(""),
        tts=FakeTTS(),
        brain=FakeBrain(lambda _: [(0.0, TextDelta("nope")), (0.0, TurnDone())]),
        registry=ToolRegistry(),
        router=lambda _: None,
        out=CaptureOut(),
        emit=broadcast,
        state_dir=tmp_path,
    )
    await session.start()
    chunks = [np.zeros(512, dtype=np.float32) for _ in range(30)]
    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, chime=False, initial_timeout_s=0.05),
        broadcast=broadcast,
        audio_in_factory=lambda: _FakeStreamAudioIn(chunks),
        detector=_ScriptedDetector(trigger_at=1),
        vad_factory=lambda: _ScriptedVAD(speak_chunks=0, end_after=999),
    )
    await listener.start()
    try:
        for _ in range(60):
            if any(e.get("event") == "wake" for e in events) and session.state is State.IDLE and session._utt is None:
                break
            await asyncio.sleep(0.02)
        assert any(e.get("event") == "wake" for e in events)
        assert session.state is State.IDLE and session._utt is None
        assert not any(e.get("event") == "transcript" for e in events)
    finally:
        await listener.stop()
        await session.close()


class _FakeWakeControl:
    def __init__(self) -> None:
        self.active = False
        self.error: str | None = None
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1
        self.active = True

    async def stop(self) -> None:
        self.stops += 1
        self.active = False


@pytest.mark.asyncio
async def test_daemon_live_toggle_wake_enabled(tmp_path):
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
    fake_wake = _FakeWakeControl()
    daemon = Daemon(session, cfg, socket_path=sock, wake=fake_wake)
    await daemon.start()
    await daemon.ready.wait()
    try:
        assert fake_wake.active is True and fake_wake.starts == 1
        assert daemon.status()["wake"]["enabled"] is True

        r, w = await asyncio.open_unix_connection(str(sock))
        w.write(b'{"cmd":"set","key":"wake.enabled","value":"off"}\n')
        await w.drain()
        rep = json.loads(await asyncio.wait_for(r.readline(), 2.0))
        assert rep == {"ok": True, "key": "wake.enabled", "value": "off"}
        assert fake_wake.active is False and cfg.wake.enabled is False

        w.write(b'{"cmd":"set","key":"wake.enabled","value":"on"}\n')
        await w.drain()
        rep2 = json.loads(await asyncio.wait_for(r.readline(), 2.0))
        assert rep2 == {"ok": True, "key": "wake.enabled", "value": "on"}
        assert fake_wake.active is True and cfg.wake.enabled is True
        w.close()
    finally:
        await daemon.close()
        shutil.rmtree(sock_dir, ignore_errors=True)


class _MultiTurnSTT:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    def transcribe(self, pcm: np.ndarray) -> str:
        return self._texts.pop(0) if self._texts else ""


@pytest.mark.asyncio
async def test_conversational_mode_continues_until_goodbye(tmp_path):
    from omi import router

    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

    session = Session(
        stt=_MultiTurnSTT(["how are you", "see ya"]),
        tts=FakeTTS(),
        brain=FakeBrain(lambda text: [(0.0, TextDelta(f"Reply to {text}")), (0.0, TurnDone())]),
        registry=ToolRegistry(),
        router=router.route,
        out=CaptureOut(),
        emit=broadcast,
        state_dir=tmp_path,
    )
    await session.start()
    chunks = [np.ones(512, dtype=np.float32) * 0.05 for _ in range(300)]
    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, chime=False, conversational=True, followup_timeout_s=2.0),
        broadcast=broadcast,
        audio_in_factory=lambda: _FakeStreamAudioIn(chunks),
        detector=_ScriptedDetector(trigger_at=1),
        vad_factory=lambda: _ScriptedVAD(speak_chunks=2, end_after=3),
        post_speech_cooldown_s=0.02,
    )
    await listener.start()
    try:
        for _ in range(120):
            if any(e.get("event") == "wake" and e.get("conversing") is False for e in events):
                break
            await asyncio.sleep(0.02)
        replies = [e["text"] for e in events if e.get("event") == "transcript" and e.get("role") == "assistant"]
        assert replies == ["Reply to how are you", "See ya."]
        assert listener.conversing is False
        assert any(e.get("event") == "wake" and e.get("conversing") is False for e in events)
    finally:
        await listener.stop()
        await session.close()


class _OneTurnThenSilentVAD:
    def __init__(self) -> None:
        self.turn = 0
        self.count = 0
        self.speaking = False
        self.ended = False

    def reset(self) -> None:
        self.turn += 1
        self.count = 0
        self.speaking = False
        self.ended = False

    def feed(self, chunk: np.ndarray) -> None:
        self.count += 1
        if self.turn == 1:
            self.speaking = self.count <= 2
            if self.count >= 3:
                self.ended = True
        else:
            self.speaking = False


@pytest.mark.asyncio
async def test_conversational_mode_drops_on_prolonged_silence(tmp_path):
    events: list[dict] = []
    sleep_chimes: list[bool] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

    async def fake_sleep_chime() -> None:
        sleep_chimes.append(True)

    session = Session(
        stt=FakeSTT("hello"),
        tts=FakeTTS(),
        brain=FakeBrain(lambda _: [(0.0, TextDelta("Hi there.")), (0.0, TurnDone())]),
        registry=ToolRegistry(),
        router=lambda _: None,
        out=CaptureOut(),
        emit=broadcast,
        state_dir=tmp_path,
    )
    await session.start()
    chunks = [np.zeros(512, dtype=np.float32) for _ in range(60)]
    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, chime=True, conversational=True, followup_timeout_s=0.06),
        broadcast=broadcast,
        audio_in_factory=lambda: _FakeStreamAudioIn(chunks),
        detector=_ScriptedDetector(trigger_at=1),
        vad_factory=_OneTurnThenSilentVAD,
        chime_fn=lambda: asyncio.sleep(0),
        sleep_chime_fn=fake_sleep_chime,
        post_speech_cooldown_s=0.01,
    )
    await listener.start()
    try:
        for _ in range(120):
            if any(e.get("event") == "wake" and e.get("conversing") is False for e in events):
                break
            await asyncio.sleep(0.02)
        assert listener.conversing is False
        assert sleep_chimes == [True]
        assert session.state is State.IDLE
    finally:
        await listener.stop()
        await session.close()


class _AlwaysSpeakingVAD:
    speaking = False
    ended = False

    def reset(self) -> None:
        self.speaking = False

    def feed(self, chunk: np.ndarray) -> None:
        self.speaking = True


def _story_brain() -> FakeBrain:
    def script(text: str):
        if "story" in text:
            return [(0.0, TextDelta("Once upon a time, there was a tiny robot.")), (3.0, TextDelta("It liked tea.")),
                    (0.0, TurnDone())]
        return [(0.0, TextDelta(f"Reply to {text}")), (0.0, TurnDone())]
    return FakeBrain(script)


async def _talk_over(tmp_path, *, barge_in: bool) -> tuple[list[dict], FakeBrain]:
    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

    brain = _story_brain()
    session = Session(
        stt=_MultiTurnSTT(["tell me a story", "actually what's the weather"]),
        tts=FakeTTS(), brain=brain, registry=ToolRegistry(), router=lambda _: None,
        out=CaptureOut(), emit=broadcast, state_dir=tmp_path,
    )
    await session.start()
    chunks = [np.ones(512, dtype=np.float32) * 0.05 for _ in range(200)]
    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, chime=False, call_mode="continuous", followup_timeout_s=2.0),
        broadcast=broadcast,
        audio_in_factory=lambda: _FakeStreamAudioIn(chunks),
        detector=_ScriptedDetector(trigger_at=1),
        vad_factory=lambda: _ScriptedVAD(speak_chunks=2, end_after=3),
        barge_vad_factory=_AlwaysSpeakingVAD,
        barge_in=barge_in,
        post_speech_cooldown_s=0.02,
    )
    await listener.start()
    try:
        for _ in range(100):
            if any(e.get("event") == "transcript" and "weather" in e.get("text", "") and e["role"] == "assistant"
                   for e in events):
                break
            await asyncio.sleep(0.02)
    finally:
        await listener.stop()
        await session.close()
    return events, brain


@pytest.mark.asyncio
async def test_talking_over_omi_interrupts_it_and_runs_the_new_request(tmp_path):
    events, brain = await _talk_over(tmp_path, barge_in=True)
    assert [t for t, _ in brain.turns] == ["tell me a story", "actually what's the weather"]
    assert brain.cancels >= 1
    replies = [e["text"] for e in events if e.get("event") == "transcript" and e.get("role") == "assistant"]
    assert replies[-1] == "Reply to actually what's the weather"
    assert "It liked tea." not in " ".join(replies)  # the rest of the story was never said


@pytest.mark.asyncio
async def test_without_barge_in_omi_finishes_talking_first(tmp_path):
    events, brain = await _talk_over(tmp_path, barge_in=False)
    assert [t for t, _ in brain.turns] == ["tell me a story"]  # talking over it did nothing
