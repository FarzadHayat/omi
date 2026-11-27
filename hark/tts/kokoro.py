"""Kokoro TTS (kokoro-onnx). Streams one chunk per phoneme batch."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import numpy as np
from kokoro_onnx import Kokoro

from hark.config import TtsConfig

SAMPLE_RATE = 24000


def _lang_for(voice: str) -> str:
    # Voice names start with a language letter and gender: a=US, b=UK, e=ES, f=FR, i=IT, j=JA, p=PT, z=ZH.
    return {
        "a": "en-us",
        "b": "en-gb",
        "e": "es",
        "f": "fr-fr",
        "i": "it",
        "j": "ja",
        "p": "pt-br",
        "z": "cmn",
    }.get(voice[:1].lower(), "en-us")


class KokoroTTS:
    def __init__(self, cfg: TtsConfig) -> None:
        self._k = Kokoro(cfg.kokoro_model, cfg.kokoro_voices)
        self._voices = set(self._k.get_voices()) if hasattr(self._k, "get_voices") else set()
        self._speed = min(max(cfg.speed, 0.5), 2.0)  # the model accepts 0.5 to 2.0
        self.sample_rate = SAMPLE_RATE
        self.set_voice(cfg.kokoro_voice)
        for _ in self.synth("Warm up."):
            pass

    @property
    def voice(self) -> str:
        return self._voice

    def set_voice(self, voice: str) -> None:
        if self._voices and voice not in self._voices:
            sample = ", ".join(sorted(v for v in self._voices if v.startswith(("af_", "am_", "bf_", "bm_")))[:8])
            raise ValueError(f"unknown kokoro voice {voice!r}; try one of: {sample}")
        self._voice = voice
        self._lang = _lang_for(voice)

    def set_speed(self, speed: float) -> None:
        self._speed = min(max(float(speed), 0.5), 2.0)

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
