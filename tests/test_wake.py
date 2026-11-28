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


@pytest.mark.parametrize("text", ["", "Mm-hmm.", "Yeah, right.", "Okay.", "uh-huh", "Got it, cool."])
def test_acknowledgements_count_as_backchannel(text):
    assert wake.is_backchannel(text)


@pytest.mark.parametrize("text", ["Wait.", "Stop", "No, the other one.", "Yeah but what about tea?"])
def test_real_remarks_are_not_backchannel(text):
    assert not wake.is_backchannel(text)


class _DuckingOut(CaptureOut):
    def __init__(self) -> None:
        super().__init__()
        self.ducks: list[bool] = []

    async def duck(self, on: bool) -> None:
        self.ducks.append(on)


class _BurstVAD:
    """Hears the user for `n` chunks, once, starting when Omi begins to speak."""

    def __init__(self, session: Session, n: int) -> None:
        self.session, self.n, self.fed = session, n, 0
        self.speaking = self.ended = False

    def reset(self) -> None:
        self.speaking = False

    def feed(self, chunk: np.ndarray) -> None:
        if self.fed or self.session.state is State.SPEAKING:
            self.fed += 1
        self.speaking = 0 < self.fed <= self.n


async def _remark_over(tmp_path, texts: list[str]) -> tuple[list[str], FakeBrain, _DuckingOut]:
    """Ask for a story, then say something short (10 chunks) while Omi tells it."""
    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

    def replies() -> list[str]:
        return [e["text"] for e in events if e.get("event") == "transcript" and e.get("role") == "assistant"]

    brain, out = _story_brain(), _DuckingOut()
    session = Session(
        stt=_MultiTurnSTT(texts), tts=FakeTTS(), brain=brain, registry=ToolRegistry(),
        router=lambda _: None, out=out, emit=broadcast, state_dir=tmp_path,
    )
    await session.start()
    chunks = [np.ones(512, dtype=np.float32) * 0.05 for _ in range(900)]
    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, chime=False, call_mode="continuous", followup_timeout_s=2.0),
        broadcast=broadcast,
        audio_in_factory=lambda: _FakeStreamAudioIn(chunks),
        detector=_ScriptedDetector(trigger_at=1),
        vad_factory=lambda: _ScriptedVAD(speak_chunks=2, end_after=3),
        barge_vad_factory=lambda: _BurstVAD(session, 10),
        barge_in=True,
        post_speech_cooldown_s=0.02,
    )
    await listener.start()
    try:
        for _ in range(300):
            if len(brain.turns) == len(texts) - 1 and session.state is State.IDLE and replies():
                break
            await asyncio.sleep(0.02)
    finally:
        await listener.stop()
        await session.close()
    return replies(), brain, out


@pytest.mark.asyncio
async def test_an_acknowledgement_over_omi_lowers_its_voice_and_lets_it_finish(tmp_path):
    replies, brain, out = await _remark_over(tmp_path, ["tell me a story", "mm-hmm"])
    assert [t for t, _ in brain.turns] == ["tell me a story"]
    assert brain.cancels == 0
    assert "It liked tea." in " ".join(replies)
    assert out.ducks == [True, False]


@pytest.mark.asyncio
async def test_a_short_remark_over_omi_interrupts_it(tmp_path):
    replies, brain, out = await _remark_over(tmp_path, ["tell me a story", "wait", "wait, what about tea"])
    assert [t for t, _ in brain.turns] == ["tell me a story", "wait, what about tea"]
    assert "It liked tea." not in " ".join(replies)
    assert out.ducks[0] is True


@pytest.mark.asyncio
async def test_pre_speech_silence_respects_followup_timeout_over_max_utterance(tmp_path):
    """Waiting silently before speaking must use followup_timeout_s (not max_utterance_s)."""
    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

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
    chunks = [np.zeros(512, dtype=np.float32) for _ in range(80)]
    listener = wake.WakeListener(
        session,
        config.WakeConfig(
            enabled=True,
            chime=False,
            conversational=True,
            max_utterance_s=0.05,
            followup_timeout_s=0.0,  # never dismiss on silence alone
        ),
        broadcast=broadcast,
        audio_in_factory=lambda: _FakeStreamAudioIn(chunks),
        detector=_ScriptedDetector(trigger_at=1),
        vad_factory=_OneTurnThenSilentVAD,
        post_speech_cooldown_s=0.01,
    )
    await listener.start()
    try:
        for _ in range(60):
            if any(e.get("event") == "transcript" and e.get("role") == "assistant" for e in events):
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.15)  # well past max_utterance_s (0.05s)
        assert listener.conversing is True
        assert not any(e.get("event") == "wake" and e.get("conversing") is False for e in events)
    finally:
        await listener.stop()
        await session.close()


# ---- Look to talk -------------------------------------------------------------------

def _face(x: float, w: float, nose_dx: float, score: float = 0.9) -> list[float]:
    """A YuNet row: box, right eye, left eye, nose, mouth corners, score."""
    eye_y, eye_gap = 100.0, w * 0.4
    mid = x + w / 2
    return [x, 60.0, w, w * 1.2, mid - eye_gap / 2, eye_y, mid + eye_gap / 2, eye_y,
            mid + nose_dx * eye_gap, eye_y + 30, mid - 15, 150.0, mid + 15, 150.0, score]


def test_facing_from_faces_reads_head_turn_and_distance():
    from omi import look

    assert look.facing_from_faces(None, 640) is False
    assert look.facing_from_faces(np.array([_face(200, 120, 0.1)]), 640) is True  # facing the screen
    assert look.facing_from_faces(np.array([_face(200, 120, 0.6)]), 640) is False  # head turned aside
    assert look.facing_from_faces(np.array([_face(200, 30, 0.0)]), 640) is False  # across the room
    assert look.facing_from_faces(np.array([_face(200, 120, 0.0, score=0.4)]), 640) is False
    # The closest (largest) face decides, not someone in the background facing the screen.
    both = np.array([_face(20, 60, 0.0), _face(300, 140, 0.7)])
    assert look.facing_from_faces(both, 640) is False


def test_look_tracker_window():
    from omi.look import LookTracker

    t = [100.0]
    tracker = LookTracker(clock=lambda: t[0])
    assert tracker.looking() is False and tracker.facing_ratio(0.0) is None
    for i, facing in enumerate([False, False, True, True, True]):
        tracker.record(facing, at=99.2 + i * 0.2)
    assert tracker.looking() is True  # 3 of the last 4 frames in the last second
    assert tracker.facing_ratio(99.0, 99.5) == 0.0
    t[0] = 105.0
    assert tracker.looking() is False  # nothing recent: the camera went quiet


class _FakeAttention:
    def __init__(self, looking: bool = True, ratio: float | None = 1.0) -> None:
        self.is_looking = looking
        self.lips = True
        self.ratio = ratio
        self.error: str | None = None
        self.started = 0
        self.stopped = 0
        self.wakes = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def wake(self) -> None:
        self.wakes += 1

    def looking(self, now: float | None = None) -> bool:
        return self.is_looking

    def addressed(self, now: float | None = None) -> bool:
        return self.is_looking and self.lips

    def facing_ratio(self, since: float, until: float | None = None) -> float | None:
        return self.ratio


async def _look_turn(tmp_path, attention: _FakeAttention, *, wake_word: bool = False) -> tuple[list[dict], FakeBrain]:
    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

    brain = FakeBrain(lambda text: [(0.0, TextDelta(f"Heard: {text}")), (0.0, TurnDone())])
    session = Session(
        stt=FakeSTT("open my calendar"), tts=FakeTTS(), brain=brain, registry=ToolRegistry(),
        router=lambda _: None, out=CaptureOut(), emit=broadcast, state_dir=tmp_path,
    )
    await session.start()
    chunks = [np.ones(512, dtype=np.float32) * 0.05 for _ in range(40)]
    detector = _ScriptedDetector(trigger_at=10**6)  # nobody says the wake word
    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=wake_word, look_to_talk=True, chime=False, conversational=False),
        broadcast=broadcast,
        audio_in_factory=lambda: _FakeStreamAudioIn(chunks),
        detector=detector,
        vad_factory=lambda: _ScriptedVAD(speak_chunks=2, end_after=4),
        look_vad_factory=_AlwaysSpeakingVAD,
        attention_factory=lambda: attention,
        post_speech_cooldown_s=0.02,
    )
    await listener.start()
    try:
        assert attention.started == 1 and listener.looking_active
        for _ in range(60):
            if any(e.get("event") == "transcript" and e.get("role") == "assistant" for e in events):
                break
            await asyncio.sleep(0.02)
    finally:
        await listener.stop()
        await session.close()
    assert attention.stopped == 1
    assert attention.wakes > 0  # hearing a voice switched the camera on
    assert detector.calls == 0 or wake_word  # the wake word model idles while it is off
    return events, brain


@pytest.mark.asyncio
async def test_talking_while_facing_the_screen_reaches_omi_without_a_wake_word(tmp_path):
    events, brain = await _look_turn(tmp_path, _FakeAttention(looking=True))
    assert any(e.get("event") == "wake" and e.get("phrase") == "look" for e in events)
    assert [t for t, _ in brain.turns] == ["open my calendar"]


@pytest.mark.asyncio
async def test_talking_while_facing_away_is_ignored(tmp_path):
    events, brain = await _look_turn(tmp_path, _FakeAttention(looking=False))
    assert brain.turns == []
    assert not any(e.get("event") == "wake" for e in events)


@pytest.mark.asyncio
async def test_turning_away_mid_sentence_drops_the_request(tmp_path):
    events, brain = await _look_turn(tmp_path, _FakeAttention(looking=True, ratio=0.2))
    assert any(e.get("event") == "wake" and e.get("phrase") == "look" for e in events)
    assert brain.turns == []


class _FakeLookWake(_FakeWakeControl):
    def __init__(self) -> None:
        super().__init__()
        self.looking_active = False
        self.look_error: str | None = None
        self.cfg: config.WakeConfig | None = None

    async def start(self) -> None:
        await super().start()
        self.looking_active = bool(self.cfg and self.cfg.look_to_talk)

    async def stop(self) -> None:
        await super().stop()
        self.looking_active = False


@pytest.mark.asyncio
async def test_daemon_live_toggle_look_to_talk(tmp_path):
    sock_dir = Path(tempfile.mkdtemp(prefix="om", dir="/tmp"))
    sock = sock_dir / "omi.sock"
    cfg = config.Config()
    session = Session(
        stt=FakeSTT(), tts=FakeTTS(), brain=FakeBrain(lambda _: [(0.0, TextDelta("ok")), (0.0, TurnDone())]),
        registry=ToolRegistry(), router=lambda _: None, out=CaptureOut(), state_dir=tmp_path,
    )
    fake = _FakeLookWake()
    fake.cfg = cfg.wake
    daemon = Daemon(session, cfg, socket_path=sock, wake=fake)
    await daemon.start()
    await daemon.ready.wait()

    async def send(r, w, line: bytes) -> dict:
        w.write(line + b"\n")
        await w.drain()
        while True:
            msg = json.loads(await asyncio.wait_for(r.readline(), 2.0))
            if "event" not in msg:
                return msg

    try:
        assert daemon.status()["wake"]["look_to_talk"] is False
        r, w = await asyncio.open_unix_connection(str(sock))
        rep = await send(r, w, b'{"cmd":"set","key":"wake.look_to_talk","value":"on"}')
        assert rep == {"ok": True, "key": "wake.look_to_talk", "value": "on"}
        assert daemon.status()["wake"]["look_to_talk"] is True

        # Turning the wake word off keeps listening for look to talk.
        await send(r, w, b'{"cmd":"set","key":"wake.enabled","value":"off"}')
        assert fake.active is True and daemon.status()["wake"]["enabled"] is False

        await send(r, w, b'{"cmd":"set","key":"wake.look_to_talk","value":"off"}')
        assert fake.active is False and daemon.status()["wake"]["look_to_talk"] is False
        w.close()
    finally:
        await daemon.close()
        shutil.rmtree(sock_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_look_to_talk_turned_on_during_startup_opens_the_camera_once(tmp_path):
    session = Session(
        stt=FakeSTT(), tts=FakeTTS(), brain=FakeBrain(lambda _: [(0.0, TurnDone())]), registry=ToolRegistry(),
        router=lambda _: None, out=CaptureOut(), state_dir=tmp_path,
    )
    made: list[_FakeAttention] = []

    def make() -> _FakeAttention:
        made.append(_FakeAttention(looking=False))
        return made[-1]

    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, look_to_talk=True, chime=False),
        audio_in_factory=lambda: _FakeStreamAudioIn([np.zeros(512, np.float32)] * 50),
        detector=_ScriptedDetector(trigger_at=10**6),
        vad_factory=lambda: _ScriptedVAD(),
        look_vad_factory=_AlwaysSpeakingVAD,
        attention_factory=make,
    )
    try:
        await asyncio.gather(listener.start(), listener.start(), listener.sync_look())
        assert len(made) == 1 and made[0].started == 1
    finally:
        await listener.stop()
    assert made[0].stopped == 1


def test_list_cameras_skips_metadata_nodes_and_names_duplicates(tmp_path, monkeypatch):
    from omi import look

    sysfs, by_id = tmp_path / "sys", tmp_path / "by-id"
    sysfs.mkdir()
    by_id.mkdir()
    nodes = {"video0": "Integrated Camera: Integrated C", "video1": "Integrated Camera: Integrated C",
             "video2": "Lenovo 50 Monitor Camera : Leno", "video4": "Lenovo 50 Monitor Camera : Leno",
             "media0": "not a camera"}
    for node, name in nodes.items():
        (sysfs / node).mkdir()
        (sysfs / node / "name").write_text(name + "\n")
    (by_id / "usb-Lenovo-video-index0").symlink_to("/dev/video2")
    monkeypatch.setattr(look, "SYS_VIDEO", sysfs)
    monkeypatch.setattr(look, "DEV_BY_ID", by_id)
    monkeypatch.setattr(look, "_captures_video", lambda dev: dev.name != "video1")

    cams = look.list_cameras()
    assert cams == [
        {"id": "/dev/video0", "name": "Integrated Camera: Integrated C"},
        {"id": str(by_id / "usb-Lenovo-video-index0"), "name": "Lenovo 50 Monitor Camera"},
        {"id": "/dev/video4", "name": "Lenovo 50 Monitor Camera (2)"},
    ]
    assert look._camera_target("") == "/dev/video0"  # automatic: the first webcam
    assert look._camera_target("/dev/video4") == "/dev/video4"


@pytest.mark.asyncio
async def test_switching_camera_reopens_the_webcam(tmp_path):
    session = Session(
        stt=FakeSTT(), tts=FakeTTS(), brain=FakeBrain(lambda _: [(0.0, TurnDone())]), registry=ToolRegistry(),
        router=lambda _: None, out=CaptureOut(), state_dir=tmp_path,
    )
    cfg = config.WakeConfig(enabled=True, look_to_talk=True, chime=False)
    made: list[tuple[str, _FakeAttention]] = []

    def make() -> _FakeAttention:
        made.append((cfg.camera, _FakeAttention(looking=False)))
        return made[-1][1]

    listener = wake.WakeListener(
        session, cfg,
        audio_in_factory=lambda: _FakeStreamAudioIn([np.zeros(512, np.float32)] * 50),
        detector=_ScriptedDetector(trigger_at=10**6), vad_factory=lambda: _ScriptedVAD(),
        look_vad_factory=_AlwaysSpeakingVAD, attention_factory=make,
    )
    await listener.start()
    try:
        cfg.camera = "/dev/video2"
        await listener.reload_look()
        assert [cam for cam, _ in made] == ["", "/dev/video2"]
        assert made[0][1].stopped == 1 and made[1][1].started == 1
    finally:
        await listener.stop()


class _FakeCameraWake(_FakeLookWake):
    def __init__(self) -> None:
        super().__init__()
        self.reloads = 0

    async def reload_look(self) -> None:
        self.reloads += 1


@pytest.mark.asyncio
async def test_daemon_picks_a_camera(tmp_path, monkeypatch):
    from omi import look

    monkeypatch.setattr(look, "list_cameras", lambda: [{"id": "/dev/video0", "name": "Built-in"},
                                                       {"id": "/dev/video2", "name": "Desk Cam"}])
    cfg = config.Config()
    session = Session(
        stt=FakeSTT(), tts=FakeTTS(), brain=FakeBrain(lambda _: [(0.0, TurnDone())]),
        registry=ToolRegistry(), router=lambda _: None, out=CaptureOut(), state_dir=tmp_path,
    )
    fake = _FakeCameraWake()
    fake.cfg = cfg.wake
    daemon = Daemon(session, cfg, socket_path=tmp_path / "x.sock", wake=fake)
    assert [c["name"] for c in daemon.status()["cameras"]] == ["Built-in", "Desk Cam"]

    rep = await daemon._set("wake.camera", "/dev/video2")
    assert rep["value"] == "/dev/video2" and cfg.wake.camera == "/dev/video2" and fake.reloads == 1
    assert daemon.status()["wake"]["camera"] == "/dev/video2"
    rep = await daemon._set("wake.camera", "auto")
    assert rep["value"] == "auto" and cfg.wake.camera == ""
    from omi.daemon import RequestError

    with pytest.raises(RequestError, match="Desk Cam"):
        await daemon._set("wake.camera", "/dev/video9")


def test_camera_runs_only_after_a_voice_and_closes_after_quiet():
    from omi import look

    t = [100.0]
    tracker = look.LookTracker(clock=lambda: t[0])
    assert tracker.watching is False  # silence: camera off
    tracker.wake()
    assert tracker.watching is True
    t[0] += look.HOLD_S - 1
    assert tracker.watching is True
    tracker.wake()  # still talking: stays on
    t[0] += look.HOLD_S - 1
    assert tracker.watching is True
    t[0] += 2
    assert tracker.watching is False  # quiet long enough: camera released


def test_mic_users_ignores_omi_echo_cancel_monitors_and_paused_streams():
    from omi.audio.pipewire import parse_mic_users

    sources = "81\tsuperwhisper_muffle.monitor\n3388\tomi_echo_cancel_source\n8431\talsa_input.usb-cam\n"
    outs = [
        {"source": 8431, "corked": False, "properties": {"node.name": "echo-cancel-capture"}},
        {"source": 3388, "corked": False, "properties": {"application.name": "Omi", "node.name": "omi-capture"}},
        {"source": 81, "corked": False, "properties": {"application.name": "OBS"}},  # desktop audio
        {"source": 8431, "corked": False, "properties": {"application.name": "cava", "stream.monitor": "true"}},
        {"source": 8431, "corked": True, "properties": {"application.name": "Discord"}},  # muted, not live
    ]
    assert parse_mic_users(outs, sources) == []
    call = {"source": 8431, "corked": False, "properties": {"application.name": "Google Chrome"}}
    assert parse_mic_users([*outs, call, call], sources) == ["Google Chrome"]


def test_a_call_on_the_mic_pauses_look_to_talk_and_frees_the_camera():
    from omi import look

    t = [100.0]
    users: list[str] = []
    tracker = look.LookTracker(clock=lambda: t[0], mic_users=lambda: list(users))
    tracker.record(True, at=99.8)
    tracker.wake()
    tracker.poll_mic(force=True)
    assert tracker.watching and tracker.looking()
    users.append("Zoom")
    tracker.poll_mic()  # polled too recently: no new check yet
    assert tracker.paused_by == ""
    t[0] += look.CALL_POLL_S
    tracker.poll_mic()
    assert tracker.paused_by == "Zoom"
    assert not tracker.watching and not tracker.looking()  # camera released, no triggers
    tracker.wake()
    assert not tracker.watching  # talking on the call does not reopen it
    users.clear()
    t[0] += look.CALL_POLL_S
    tracker.poll_mic()
    assert tracker.paused_by == "" and tracker.watching


@pytest.mark.asyncio
async def test_a_call_resumed_after_a_restart_listens_without_the_wake_word(tmp_path):
    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

    session = Session(
        stt=FakeSTT("and the next one"),
        tts=FakeTTS(),
        brain=FakeBrain(lambda text: [(0.0, TextDelta(f"Heard: {text}")), (0.0, TurnDone())]),
        registry=ToolRegistry(),
        router=lambda _: None,
        out=CaptureOut(),
        emit=broadcast,
        state_dir=tmp_path,
    )
    await session.start()
    chunks = [np.ones(512, dtype=np.float32) * 0.05 for _ in range(40)]
    vad = _ScriptedVAD(speak_chunks=0, end_after=10**6)  # silent at first
    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, chime=False, conversational=True, followup_timeout_s=0.0),
        broadcast=broadcast,
        audio_in_factory=lambda: _FakeStreamAudioIn(chunks),
        detector=_ScriptedDetector(trigger_at=10**9),  # the wake word is never said
        vad_factory=lambda: vad,
        post_speech_cooldown_s=0.01,
    )
    listener.resume_call()
    await listener.start()
    try:
        for _ in range(100):
            if listener.waiting_for_speech:
                break
            await asyncio.sleep(0.01)
        assert listener.conversing and listener.waiting_for_speech
        assert session.state is State.LISTENING
        assert any(e.get("event") == "wake" and e.get("phrase") == "resume" for e in events)
        vad.speak_chunks, vad.end_after, vad.count = 2, 3, 0  # now the user talks
        for _ in range(100):
            if any(e.get("event") == "transcript" and e.get("role") == "assistant" for e in events):
                break
            await asyncio.sleep(0.02)
        assistant = [e["text"] for e in events if e.get("event") == "transcript" and e.get("role") == "assistant"]
        assert assistant == ["Heard: and the next one"]
    finally:
        await listener.stop()
        await session.close()


class _BrokenAttention(_FakeAttention):
    def start(self) -> None:
        raise RuntimeError("no camera")


@pytest.mark.asyncio
async def test_wake_word_still_works_when_look_to_talk_cannot_start(tmp_path):
    events: list[dict] = []

    async def broadcast(ev: dict) -> None:
        events.append(ev)

    brain = FakeBrain(lambda text: [(0.0, TextDelta(f"Heard: {text}")), (0.0, TurnDone())])
    session = Session(
        stt=FakeSTT("what time is it"), tts=FakeTTS(), brain=brain, registry=ToolRegistry(),
        router=lambda _: None, out=CaptureOut(), emit=broadcast, state_dir=tmp_path,
    )
    await session.start()
    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, look_to_talk=True, chime=False, conversational=False),
        broadcast=broadcast,
        audio_in_factory=lambda: _FakeStreamAudioIn([np.ones(512, np.float32) * 0.05 for _ in range(12)]),
        detector=_ScriptedDetector(trigger_at=2),
        vad_factory=lambda: _ScriptedVAD(speak_chunks=2, end_after=4),
        look_vad_factory=_AlwaysSpeakingVAD,
        attention_factory=_BrokenAttention,
    )
    await listener.start()
    try:
        assert listener.active and not listener.looking_active
        assert listener.look_error == "no camera"
        for _ in range(80):
            if brain.turns:
                break
            await asyncio.sleep(0.02)
        assert [t for t, _ in brain.turns] == ["what time is it"]  # "Hey Omi" still reached it
    finally:
        await listener.stop()
        await session.close()


def _mesh(gap: float) -> np.ndarray:
    pts = np.zeros((468, 3), np.float32)
    pts[61, :2], pts[291, :2] = (80, 130), (112, 130)  # mouth 32 px wide
    pts[13, :2], pts[14, :2] = (96, 130 - gap / 2), (96, 130 + gap / 2)
    return pts.reshape(1, -1)


def test_lip_openness_and_spread():
    from omi import look

    assert look.lip_openness(_mesh(0.0)) == 0.0
    assert abs(look.lip_openness(_mesh(8.0)) - 0.25) < 1e-6
    assert look.lip_spread([0.1, 0.1, 0.1]) is None  # too few frames to judge
    assert look.lip_spread([0.02, 0.03, 0.02, 0.03, 0.02]) < look.LIP_MOVE_SPREAD  # still
    assert look.lip_spread([0.02, 0.25, 0.05, 0.3, 0.1, 0.02]) >= look.LIP_MOVE_SPREAD  # talking


def _tracker_with(frames: list[tuple[bool, float | None]], *, lip_check: bool = True):
    from omi.look import LookTracker

    tracker = LookTracker(clock=lambda: 100.0, mic_users=lambda: [], lip_check=lip_check)
    for i, (facing, lips) in enumerate(frames):
        tracker.record(facing, at=99.1 + i * 0.1, lips=lips)
    return tracker


def test_addressed_needs_facing_and_moving_lips():
    talking = [(True, v) for v in (0.02, 0.2, 0.05, 0.3, 0.1, 0.25, 0.03, 0.2)]
    still = [(True, v) for v in (0.03, 0.04, 0.03, 0.04, 0.03, 0.04, 0.03, 0.04)]
    open_mouth = [(True, v) for v in (0.3, 0.31, 0.3, 0.32, 0.3, 0.31, 0.3, 0.31)]  # a yawn, a smile
    hidden = [(True, None)] * 8  # hand over the mouth
    away = [(False, None)] * 8
    assert _tracker_with(talking).addressed() is True
    assert _tracker_with(still).addressed() is False  # a video talking while you watch
    assert _tracker_with(open_mouth).addressed() is False
    assert _tracker_with(hidden).addressed() is False  # fall back to "Hey Omi"
    assert _tracker_with(away).addressed() is False
    assert _tracker_with(still, lip_check=False).addressed() is True  # check switched off


@pytest.mark.asyncio
async def test_facing_the_screen_with_still_lips_is_ignored(tmp_path):
    attention = _FakeAttention(looking=True)
    attention.lips = False  # a voice from the speakers while you watch silently
    events, brain = await _look_turn(tmp_path, attention)
    assert brain.turns == [] and not any(e.get("phrase") == "look" for e in events)


def test_face_mesh_model_reads_a_real_mouth_crop():
    """The pinned face mesh loads in OpenCV and answers in the expected shape."""
    from omi import look

    if not look.MESH_PATH.is_file():
        pytest.skip("face mesh not downloaded")
    import cv2

    mesh = cv2.dnn.readNetFromTFLite(str(look.MESH_PATH))
    frame = np.full((480, 640, 3), 128, np.uint8)
    face = np.array([260, 160, 120, 150, 290, 210, 350, 210, 320, 240, 295, 270, 345, 270, 0.9], np.float32)
    assert look._measure_lips(cv2, mesh, frame, face) is None  # a grey frame holds no face


@pytest.mark.asyncio
async def test_what_is_said_while_omi_starts_is_heard_once_it_is_ready(tmp_path):
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
    audio = _FakeStreamAudioIn([np.ones(512, dtype=np.float32) * 0.05 for _ in range(12)])
    listener = wake.WakeListener(
        session,
        config.WakeConfig(enabled=True, chime=False, conversational=False),
        broadcast=broadcast,
        audio_in_factory=lambda: audio,
        detector=_ScriptedDetector(trigger_at=2),
        vad_factory=lambda: _ScriptedVAD(speak_chunks=2, end_after=4),
    )
    await listener.listen_early()
    await asyncio.sleep(0.15)  # the models are still loading; the user already spoke
    assert audio._idx == 12 and events == []
    await listener.start()
    try:
        for _ in range(80):
            if any(e.get("event") == "transcript" and e.get("role") == "assistant" for e in events):
                break
            await asyncio.sleep(0.02)
        assistant = [e["text"] for e in events if e.get("event") == "transcript" and e.get("role") == "assistant"]
        assert assistant == ["Heard: what time is it"]
    finally:
        await listener.stop()
        await session.close()
    assert audio.stopped


@pytest.mark.asyncio
async def test_early_backlog_keeps_only_the_newest_audio():
    chunks = [np.full(1600, i, dtype=np.float32) for i in range(10)]
    backlog = wake.Backlog(_FakeStreamAudioIn(chunks), max_s=0.3)  # three 100 ms chunks
    await backlog.start()
    await asyncio.sleep(0.15)
    got = []
    while (c := await backlog.read()) is not None:
        got.append(int(c[0]))
    assert got == [7, 8, 9]
    await backlog.stop()


@pytest.mark.asyncio
async def test_someone_elses_voice_is_ignored_when_only_my_voice_is_on(tmp_path):
    for verdict, expect_turn in ((False, False), (None, True), (True, True)):
        events: list[dict] = []

        async def broadcast(ev: dict) -> None:
            events.append(ev)

        brain = FakeBrain(lambda text: [(0.0, TextDelta("ok")), (0.0, TurnDone())])
        session = Session(stt=FakeSTT("what time is it"), tts=FakeTTS(), brain=brain, registry=ToolRegistry(),
                          router=lambda _: None, out=CaptureOut(), emit=broadcast, state_dir=tmp_path)
        await session.start()
        heard: list[int] = []
        listener = wake.WakeListener(
            session, config.WakeConfig(enabled=True, chime=False, conversational=False), broadcast=broadcast,
            audio_in_factory=lambda: _FakeStreamAudioIn([np.ones(512, np.float32) * 0.05 for _ in range(12)]),
            detector=_ScriptedDetector(trigger_at=2), vad_factory=lambda: _ScriptedVAD(speak_chunks=2, end_after=4),
            voice_gate=lambda pcm, v=verdict: heard.append(len(pcm)) or v,
        )
        await listener.start()
        try:
            for _ in range(60):
                if brain.turns or (heard and not expect_turn and session._utt is None):
                    break
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.1)
        finally:
            await listener.stop()
            await session.close()
        assert heard and heard[0] > 0  # the check saw the request's audio
        assert bool(brain.turns) is expect_turn, verdict
