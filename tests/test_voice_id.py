"""Only my voice: voiceprints, the listener's check, and the setting."""

from __future__ import annotations

import stat

import numpy as np
import pytest

from omi import voice_id
from omi.voice_id import VoiceGate, Voiceprints, enroll, speech_only

RATE = 16000


def _voice(pitch: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.1 * np.sin(2 * np.pi * pitch * t)).astype(np.float32)


class _FakeId:
    """Voiceprint = a direction picked by the tone's pitch, with a little noise per piece."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng(1)

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        spectrum = np.abs(np.fft.rfft(pcm[:RATE]))
        vec = np.zeros(8, np.float32)
        vec[int(np.argmax(spectrum)) // 100 % 8] = 1.0
        vec += self.rng.normal(0, 0.05, 8).astype(np.float32)
        return vec / np.linalg.norm(vec)


def test_speech_only_drops_the_pauses():
    pcm = np.concatenate([_voice(200, 1), np.zeros(RATE, np.float32), _voice(200, 1)])
    assert abs(len(speech_only(pcm)) - 2 * RATE) < 2048


def test_enroll_needs_enough_speech_and_sets_a_bounded_match_bar():
    with pytest.raises(ValueError):
        enroll(_voice(200, 2), _FakeId())
    p = enroll(_voice(200, 6), _FakeId())
    assert len(p["vector"]) == 8 and voice_id.MIN_MATCH <= p["threshold"] <= voice_id.MAX_MATCH


def test_gate_lets_saved_voices_through_and_drops_others(tmp_path):
    prints = Voiceprints(tmp_path / "v.json")
    vid = _FakeId()
    prints.save("farzad", enroll(_voice(200, 6), vid))
    on = [True]
    gate = VoiceGate(lambda: on[0], prints, vid)
    assert gate(_voice(200, 2)) is True
    assert gate(_voice(500, 2)) is False  # someone else, the TV
    assert gate(_voice(500, 0.6)) is None  # "stop": too short to judge, never blocked
    on[0] = False
    assert gate(_voice(500, 2)) is None  # switched off in Settings: nothing is checked


def test_voiceprints_are_private_and_can_be_forgotten(tmp_path):
    prints = Voiceprints(tmp_path / "v.json")
    prints.save("a", {"vector": [1.0], "threshold": 0.5})
    prints.save("b", {"vector": [1.0], "threshold": 0.5})
    assert stat.S_IMODE((tmp_path / "v.json").stat().st_mode) == 0o600
    assert prints.names() == ["a", "b"]
    assert prints.forget("a") and prints.names() == ["b"]
    assert not prints.forget("nobody")
    assert prints.forget() and not (tmp_path / "v.json").exists()


async def test_only_my_voice_setting(tmp_path):
    from omi.config import Config
    from omi.daemon import Daemon, RequestError
    from omi.session import Session
    from omi.tools.base import ToolRegistry
    from tests.fakes import CaptureOut, FakeBrain, FakeSTT, FakeTTS

    cfg = Config()
    session = Session(stt=FakeSTT(), tts=FakeTTS(), brain=FakeBrain(lambda _: []), registry=ToolRegistry(),
                      router=lambda _: None, out=CaptureOut(), state_dir=tmp_path)
    daemon = Daemon(session, cfg, socket_path=tmp_path / "s.sock")
    daemon.voiceprints = Voiceprints(tmp_path / "v.json")
    with pytest.raises(RequestError, match="No voice saved yet"):
        await daemon._set("wake.only_my_voice", "on")
    daemon.voiceprints.save("farzad", {"vector": [1.0], "threshold": 0.5})
    assert (await daemon._set("wake.only_my_voice", "on"))["value"] == "on"
    assert daemon.status()["wake"]["voices"] == ["farzad"] and cfg.wake.only_my_voice
    await daemon._set("wake.only_my_voice", "off")
    assert not (tmp_path / "v.json").exists()  # a voiceprint is only kept while it is used


def test_mic_check_hears_a_voice_over_a_quiet_room():
    from omi.look_tune import heard_you

    rng = np.random.default_rng(0)
    room = (rng.normal(0, 0.002, RATE * 4)).astype(np.float32)
    assert heard_you(room, room + np.tile(_voice(180, 1), 4)) is True
    assert heard_you(room, room) is False  # muted or wrong mic
    assert heard_you(room, np.zeros(0, np.float32)) is False
