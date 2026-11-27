"""Piper TTS. One chunk per sentence; for a single clause that is one chunk, fast."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from piper import PiperVoice, SynthesisConfig

from omi.config import TtsConfig

# Piper ends each sentence with about 200 ms of silence and starts the next with 60 ms.
# The reply is cut into clauses, so after "Sure," the listener heard a 300 ms pause, a
# full stop. After a clause mark the silence is cut to a short breath.
CLAUSE_PAUSE_S = 0.08
_CLAUSE_MARKS = (",", ";", ":")


def trim_tail(audio: np.ndarray, rate: int, keep_s: float, floor: float = 0.01) -> np.ndarray:
    """Drop trailing near-silence beyond keep_s seconds."""
    loud = np.flatnonzero(np.abs(audio) > floor)
    end = (loud[-1] + 1 if len(loud) else 0) + int(keep_s * rate)
    return audio[:end]


class PiperTTS:
    def __init__(self, cfg: TtsConfig) -> None:
        self._voice = PiperVoice.load(cfg.piper_model)
        self.sample_rate = int(self._voice.config.sample_rate)
        # length_scale below 1 speaks faster, so invert the speed factor.
        self._base_length_scale = self._voice.config.length_scale or 1.0
        self.set_speed(cfg.speed)
        for _ in self.synth("Warm up."):
            pass

    def set_speed(self, speed: float) -> None:
        clamped = min(max(float(speed), 0.5), 2.0)
        self._syn = SynthesisConfig(length_scale=self._base_length_scale / clamped)

    def synth(self, text: str) -> Iterator[np.ndarray]:
        clause = text.rstrip().endswith(_CLAUSE_MARKS)
        for chunk in self._voice.synthesize(text, syn_config=self._syn):
            audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
            yield trim_tail(audio, self.sample_rate, CLAUSE_PAUSE_S) if clause else audio
