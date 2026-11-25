"""Text-to-speech engines. create() loads once and warms up."""

from __future__ import annotations

from hark.config import TtsConfig
from hark.tts.base import TTS

__all__ = ["TTS", "create"]


def create(cfg: TtsConfig) -> TTS:
    # Imports are lazy so choosing one engine never loads the other.
    if cfg.engine == "piper":
        from hark.tts.piper import PiperTTS

        return PiperTTS(cfg)
    if cfg.engine == "kokoro":
        from hark.tts.kokoro import KokoroTTS

        return KokoroTTS(cfg)
    raise ValueError(f"unknown tts engine: {cfg.engine}")
