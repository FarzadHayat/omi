"""Piper TTS. One chunk per sentence; for a single clause that is one chunk, fast."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from piper import PiperVoice, SynthesisConfig

from hark.config import TtsConfig


class PiperTTS:
    def __init__(self, cfg: TtsConfig) -> None:
        self._voice = PiperVoice.load(cfg.piper_model)
        self.sample_rate = int(self._voice.config.sample_rate)
        # length_scale below 1 speaks faster, so invert the speed factor.
        base = self._voice.config.length_scale or 1.0
        self._syn = SynthesisConfig(length_scale=base / max(cfg.speed, 0.1))
        for _ in self.synth("Warm up."):
            pass

    def synth(self, text: str) -> Iterator[np.ndarray]:
        for chunk in self._voice.synthesize(text, syn_config=self._syn):
            yield np.asarray(chunk.audio_float_array, dtype=np.float32)
