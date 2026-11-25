"""STT and TTS against the audio fixtures. Loads real models, so it takes a few seconds."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hark import stt, tts
from hark.config import SttConfig, TtsConfig
from hark.stt.base import SAMPLE_RATE

FIX = Path(__file__).parent / "fixtures"
REFS = {
    "speech_a": "welcome back today we are testing my cloned voice let's see how it sounds",
    "speech_a_noisy": "welcome back today we are testing my cloned voice let's see how it sounds",
    "speech_b": "let's go this is the moment we've been waiting for big news today and i "
    "couldn't be more excited to share it with you",
}
SENTENCE = "Sure, I've opened Firefox on workspace three."


def words(text: str) -> list[str]:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", text.lower().replace("to day", "today"))).split()


def overlap(ref: str, hyp: str) -> float:
    """Share of reference words found in the hypothesis. Tolerant of one or two misses."""
    ref_w, hyp_w = words(ref), words(hyp)
    pool = list(hyp_w)
    hit = 0
    for w in ref_w:
        if w in pool:
            pool.remove(w)
            hit += 1
    return hit / len(ref_w)


def load(name: str) -> np.ndarray:
    audio, rate = sf.read(FIX / f"{name}.wav", dtype="float32")
    assert rate == SAMPLE_RATE
    return audio


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    n = int(len(audio) * SAMPLE_RATE / rate)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


@pytest.fixture(scope="module", params=["moonshine", "nemotron"])
def engine(request):
    return stt.create(SttConfig(engine=request.param))


@pytest.fixture(scope="module", params=["piper", "kokoro"])
def voice(request):
    return tts.create(TtsConfig(engine=request.param))


@pytest.fixture(scope="module")
def moonshine():
    return stt.create(SttConfig(engine="moonshine"))


@pytest.mark.parametrize("name", list(REFS))
def test_transcribe_fixture(engine, name):
    audio = load(name)
    t = time.perf_counter()
    hyp = engine.transcribe(audio)
    ms = (time.perf_counter() - t) * 1000
    print(f"\n{type(engine).__name__} {name} {len(audio) / SAMPLE_RATE:.2f}s audio: {ms:.0f} ms | {hyp}")
    assert overlap(REFS[name], hyp) >= 0.8, hyp


def test_nemotron_streams_partials_and_matches_final():
    s = stt.create(SttConfig(engine="nemotron"))
    audio = load("speech_a")
    s.start()
    partials, step = [], 2560  # 160 ms
    worst = 0.0
    for i in range(0, len(audio), step):
        t = time.perf_counter()
        partials.append(s.feed(audio[i : i + step]))
        worst = max(worst, time.perf_counter() - t)
    t = time.perf_counter()
    final = s.finish()
    flush_ms = (time.perf_counter() - t) * 1000
    print(f"\nnemotron stream: worst feed {worst * 1000:.0f} ms, finish {flush_ms:.0f} ms | {final}")
    assert any(partials[:-1]), "text should appear before the audio ends"
    assert overlap(REFS["speech_a"], final) >= 0.8
    assert worst < 0.16, "a feed must be faster than the audio it carries"
    s.start()  # a new utterance starts clean
    assert s.feed(np.zeros(2560, np.float32)) == ""


def test_stt_unknown_engine():
    with pytest.raises(ValueError):
        stt.create(SttConfig(engine="nope"))


def test_tts_first_chunk_and_rtf(voice):
    t = time.perf_counter()
    first, chunks = None, []
    for c in voice.synth(SENTENCE):
        first = first if first is not None else time.perf_counter() - t
        chunks.append(c)
    total = time.perf_counter() - t
    audio = np.concatenate(chunks)
    dur = len(audio) / voice.sample_rate
    print(f"\n{type(voice).__name__} first chunk {first * 1000:.0f} ms, audio {dur:.2f}s, RTF {total / dur:.3f}")
    assert audio.dtype == np.float32 and dur > 1.0
    assert np.abs(audio).max() > 0.05
    assert first < 1.5 and total / dur < 0.6  # faster than real time with room to spare


def test_tts_round_trip(voice, moonshine):
    audio = np.concatenate(list(voice.synth(SENTENCE)))
    hyp = moonshine.transcribe(to_16k(audio, voice.sample_rate))
    print(f"\n{type(voice).__name__} round trip: {hyp}")
    assert overlap(SENTENCE.lower(), hyp) >= 0.7, hyp


def test_tts_speed_shortens_audio():
    base = TtsConfig(engine="piper")
    slow = np.concatenate(list(tts.create(base).synth(SENTENCE)))
    fast = np.concatenate(list(tts.create(TtsConfig(engine="piper", speed=1.5)).synth(SENTENCE)))
    assert len(fast) < len(slow) * 0.8


def test_kokoro_speed_shortens_audio():
    slow = np.concatenate(list(tts.create(TtsConfig(engine="kokoro")).synth(SENTENCE)))
    fast = np.concatenate(list(tts.create(TtsConfig(engine="kokoro", speed=1.5)).synth(SENTENCE)))
    assert len(fast) < len(slow) * 0.8


def test_tts_unknown_engine():
    with pytest.raises(ValueError):
        tts.create(TtsConfig(engine="nope"))


# --- speed ---


def _tail_silence_s(audio: np.ndarray, rate: int) -> float:
    loud = np.flatnonzero(np.abs(audio) > 0.01)
    return (len(audio) - loud[-1] - 1) / rate


def test_piper_keeps_only_a_breath_after_a_clause():
    voice = tts.create(TtsConfig(engine="piper"))
    clause = np.concatenate(list(voice.synth("Sure,")))
    sentence = np.concatenate(list(voice.synth("Sure.")))
    assert _tail_silence_s(clause, voice.sample_rate) <= 0.09
    assert _tail_silence_s(sentence, voice.sample_rate) > 0.15  # a sentence end keeps its pause


def test_nemotron_hotword_vocab_is_written_from_tokens(tmp_path):
    from hark.stt.nemotron import write_hotwords

    tokens = tmp_path / "tokens.txt"
    tokens.write_text("<unk> 0\n▁t 1\n▁th 2\n")
    hot, vocab = write_hotwords(tmp_path / "out", tokens, ["Omarchy", "Danzeira"])
    assert Path(vocab).read_text().splitlines() == ["<unk>\t0", "▁t\t-1", "▁th\t-2"]
    assert Path(hot).read_text().splitlines() == ["Omarchy", "Danzeira"]


def test_nemotron_hears_the_users_own_words():
    from hark.stt.nemotron import NemotronSTT

    voice = tts.create(TtsConfig(engine="kokoro", kokoro_voice="af_heart"))
    audio = to_16k(np.concatenate(list(voice.synth("Show me the Omarchy menu."))), voice.sample_rate)
    audio = np.concatenate([np.zeros(3200, np.float32), audio])
    plain = NemotronSTT(SttConfig(engine="nemotron"), hotwords=()).transcribe(audio).lower()
    boosted = NemotronSTT(SttConfig(engine="nemotron")).transcribe(audio).lower()
    assert "omarchy" not in plain and "omarchy" in boosted, (plain, boosted)


def test_default_engine_is_streaming():
    from hark.session import is_streaming

    assert SttConfig().engine == "nemotron"
    assert is_streaming(stt.create(SttConfig()))
