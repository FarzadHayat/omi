"""Audio sinks and sources, VAD, and PipeWire against a temporary null sink (silent)."""

from __future__ import annotations

import asyncio
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from omi.audio.file import NullOut, WavOut
from omi.audio.pipewire import PipeWireIn, PipeWireOut
from omi.audio.vad import EndpointDetector
from omi.config import SttConfig
from omi.stt import create as create_stt

from test_speech import REFS, overlap

FIX = Path(__file__).parent / "fixtures"
SINK = "omi_test"


def load(name: str) -> np.ndarray:
    return sf.read(FIX / f"{name}.wav", dtype="float32")[0]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt((x**2).mean())) if len(x) else 0.0


# --- file sinks ---


async def test_wav_out_round_trip(tmp_path):
    out = WavOut(tmp_path / "sub" / "reply.wav")
    audio = load("speech_a")
    await out.start(16000)
    await out.write(audio[:8000])
    await out.write(audio[8000:])
    await out.drain()
    back, rate = sf.read(tmp_path / "sub" / "reply.wav", dtype="float32")
    assert rate == 16000 and len(back) == len(audio)
    assert np.abs(back - audio).max() < 1e-3
    with wave.open(str(tmp_path / "sub" / "reply.wav")) as fh:
        assert fh.getnchannels() == 1 and fh.getsampwidth() == 2


async def test_wav_out_stop_keeps_partial(tmp_path):
    out = WavOut(tmp_path / "cut.wav")
    await out.start(22050)
    await out.write(np.zeros(1000, np.float32))
    await out.stop()
    back, rate = sf.read(tmp_path / "cut.wav")
    assert rate == 22050 and len(back) == 1000


async def test_null_out_accepts_everything():
    out = NullOut()
    await out.start(16000)
    await out.write(np.zeros(10, np.float32))
    await out.drain()
    await out.stop()


# --- VAD ---


def run_vad(audio: np.ndarray, silence_ms: int = 300):
    det = EndpointDetector(silence_ms=silence_ms)
    x = np.concatenate([np.zeros(8000, np.float32), audio, np.zeros(16000 + silence_ms * 16, np.float32)])
    saw_speaking, ended_at = False, None
    for i in range(0, len(x), 512):
        det.feed(x[i : i + 512])
        saw_speaking |= det.speaking
        if det.ended and ended_at is None:
            ended_at = i / 16000
    return det, saw_speaking, ended_at, len(audio) / 16000 + 0.5


@pytest.mark.parametrize("name", ["speech_a", "speech_a_noisy", "speech_b"])
def test_vad_cuts_at_the_first_long_pause(name):
    # Every fixture has a pause of 0.8 s or more early on: a 300 ms setting ends the turn there.
    det, saw, ended_at, _ = run_vad(load(name), 300)
    assert saw and det.ended and not det.speaking
    assert 1.2 < ended_at < 2.2


@pytest.mark.parametrize("name", ["speech_a", "speech_a_noisy", "speech_b"])
def test_vad_waits_through_pauses_then_ends(name):
    det, _, ended_at, speech_end = run_vad(load(name), 1200)
    assert det.ended
    # Not before the last words (the files end in a short fade), and 1.2 s after them.
    assert speech_end - 0.8 < ended_at < speech_end + 1.2 + 0.2


def test_vad_silence_never_starts_and_reset_works():
    det = EndpointDetector(silence_ms=300)
    det.feed(np.zeros(32000, np.float32))
    assert not det.speaking and not det.ended
    det.feed(load("speech_a"))
    det.feed(np.zeros(16000, np.float32))
    assert det.ended
    det.reset()
    assert not det.ended and not det.speaking


def test_vad_accepts_odd_chunk_sizes():
    det = EndpointDetector(silence_ms=300)
    x = np.concatenate([load("speech_a"), np.zeros(16000, np.float32)])
    for i in range(0, len(x), 700):
        det.feed(x[i : i + 700])
    assert det.ended


# --- PipeWire, through a throw-away null sink so nothing is audible ---


@pytest.fixture
def null_sink():
    try:
        mod = subprocess.run(
            ["pactl", "load-module", "module-null-sink", f"sink_name={SINK}"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("pactl not available")
    if mod.returncode != 0 or not mod.stdout.strip():
        pytest.skip(f"cannot create null sink: {mod.stderr.strip()}")
    module_id = mod.stdout.strip()
    try:
        yield SINK
    finally:
        subprocess.run(["pactl", "unload-module", module_id], capture_output=True, timeout=10)


async def collect(src: PipeWireIn, seconds: float) -> list[tuple[float, np.ndarray]]:
    """Read chunks for `seconds`, stamping each with its arrival time."""
    got, end = [], time.perf_counter() + seconds
    while time.perf_counter() < end:
        chunk = await asyncio.wait_for(src.read(), 2)
        assert chunk is not None
        got.append((time.perf_counter(), chunk))
    return got


async def test_pipewire_in_chunks_arrive_in_real_time(null_sink):
    src = PipeWireIn(f"{null_sink}.monitor")
    await src.start()
    t = time.perf_counter()
    chunks = [await asyncio.wait_for(src.read(), 2) for _ in range(10)]
    elapsed = time.perf_counter() - t
    audio = await src.stop()
    assert all(len(c) == 512 and c.dtype == np.float32 for c in chunks)
    assert 0.2 < elapsed < 0.6  # 10 chunks of 32 ms, not a burst
    assert len(audio) >= 5120
    assert await src.read() is None  # stopped and empty


async def test_pipewire_play_then_record_and_transcribe(null_sink):
    stt = create_stt(SttConfig(engine="moonshine"))
    audio = load("speech_a")
    src, out = PipeWireIn(f"{null_sink}.monitor"), PipeWireOut(null_sink)
    await src.start()
    await asyncio.sleep(0.3)
    t = time.perf_counter()
    await out.start(16000)
    await out.write(audio[:1600])
    first_write_ms = (time.perf_counter() - t) * 1000
    for i in range(1600, len(audio), 3200):
        await out.write(audio[i : i + 3200])
    t = time.perf_counter()
    await out.drain()
    drain_s = time.perf_counter() - t
    await asyncio.sleep(0.4)
    rec = await src.stop()
    print(f"\nPipeWireOut start-to-first-write {first_write_ms:.1f} ms, drain waited {drain_s:.2f} s")
    assert first_write_ms < 50
    assert drain_s > len(audio) / 16000 * 0.5  # drain waits for the tail, not just the pipe
    assert overlap(REFS["speech_a"], stt.transcribe(rec)) >= 0.8


async def test_pipewire_out_stop_cuts_sound_fast(null_sink):
    audio = load("speech_b")
    src, out = PipeWireIn(f"{null_sink}.monitor"), PipeWireOut(null_sink)
    await src.start()
    await out.start(16000)
    await out.write(np.tile(audio, 3))  # a long reply, all queued at once
    await asyncio.sleep(1.4)  # well inside the first loud phrase
    t = time.perf_counter()
    await out.stop()
    stop_ms = (time.perf_counter() - t) * 1000
    stopped_at = time.perf_counter()
    got = await collect(src, 0.8)
    await src.stop()
    print(f"\nPipeWireOut stop() took {stop_ms:.1f} ms")
    assert stop_ms < 100
    # Everything that reaches the reader more than ~160 ms after stop() must be silent.
    late = np.concatenate([c for ts, c in got if ts - stopped_at > 0.16])
    assert rms(late) < 0.005


async def test_pipewire_out_is_reusable_and_stop_is_idempotent(null_sink):
    out = PipeWireOut(null_sink)
    await out.stop()  # nothing running
    await out.write(np.zeros(100, np.float32))  # ignored before start
    for _ in range(2):
        await out.start(24000)
        await out.write(np.zeros(2400, np.float32))
        await out.drain()
    await out.start(24000)
    await out.stop()
    await out.stop()


# --- no speaker or microphone: a clear error, not silent success ---


@pytest.fixture
def no_devices(tmp_path, monkeypatch):
    """pw-cat and pw-record stand-ins that fail the way PipeWire does with no device."""
    for name in ("pw-cat", "pw-record"):
        exe = tmp_path / name
        exe.write_text("#!/bin/sh\necho 'stream node 1 error: no target node available' >&2\nexit 1\n")
        exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{__import__('os').environ['PATH']}")


async def test_playback_with_no_speaker_raises(no_devices):
    out = PipeWireOut()
    await out.start(16000)
    await out.write(np.zeros(1600, np.float32))
    with pytest.raises(RuntimeError, match="no audio output device"):
        await out.drain()
    assert "no target node" in out.error


async def test_recording_with_no_microphone_reports_why(no_devices):
    mic = PipeWireIn()
    await mic.start()
    await asyncio.sleep(0.2)
    pcm = await mic.stop()
    assert len(pcm) == 0 and "no microphone device" in mic.error


async def test_turn_with_no_speaker_reports_an_error(no_devices, tmp_path):
    from omi.session import Session
    from tests.fakes import FakeBrain, FakeRouter, FakeSTT, FakeTTS, make_registry

    s = Session(stt=FakeSTT(), tts=FakeTTS(), brain=FakeBrain(), registry=make_registry(), router=FakeRouter(),
                out=PipeWireOut(), state_dir=tmp_path)
    res = await s.run_turn("hello")
    assert res.reply == "Okay." and res.error and "no audio output device" in res.error


def test_echo_cancel_references_all_speaker_audio():
    from omi.audio import echo

    conf = echo.build_conf("alsa_input.cam", 'hdmi "out"')
    assert "monitor.mode = true" in conf  # cancels videos and music, not just Omi's voice
    assert 'target.object = "hdmi \\"out\\""' in conf
    assert 'target.object = "alsa_input.cam"' in conf
    assert f'node.name = "{echo.SOURCE}"' in conf
    assert "target.object" not in echo.build_conf("", "hdmi").split("capture.props")[1].split("\n")[0]
