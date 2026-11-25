"""Kokoro TTS (kokoro-onnx, int8). Streams one chunk per phoneme batch."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import numpy as np
from kokoro_onnx import Kokoro

from hark.config import TtsConfig

SAMPLE_RATE = 24000


class KokoroTTS:
    def __init__(self, cfg: TtsConfig) -> None:
        self._k = Kokoro(cfg.kokoro_model, cfg.kokoro_voices)
        self._voice = cfg.kokoro_voice
        self._speed = min(max(cfg.speed, 0.5), 2.0)  # the model accepts 0.5 to 2.0
        # Voice names start with a language letter and a gender letter: bm_ is British.
        self._lang = "en-gb" if self._voice.startswith("b") else "en-us"
        self.sample_rate = SAMPLE_RATE
        for _ in self.synth("Warm up."):
            pass

    def synth(self, text: str) -> Iterator[np.ndarray]:
        # create_stream is async; drive it from a private loop so synth stays a plain
        # blocking generator that is safe to run in any worker thread.
        loop = asyncio.new_event_loop()
        agen = self._k.create_stream(text, voice=self._voice, speed=self._speed, lang=self._lang)
        try:
            while True:
                try:
                    audio, _ = loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    return
                yield np.asarray(audio, dtype=np.float32)
        finally:
            loop.run_until_complete(agen.aclose())
            loop.close()
