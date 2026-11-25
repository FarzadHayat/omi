"""Text-to-speech engines. create() loads once and warms up."""

from __future__ import annotations

from pathlib import Path

from hark.config import TtsConfig
from hark.tts.base import TTS

__all__ = ["TTS", "create"]


def create(cfg: TtsConfig) -> TTS:
    files = {"piper": [cfg.piper_model, cfg.piper_model + ".json"],
             "kokoro": [cfg.kokoro_model, cfg.kokoro_voices]}.get(cfg.engine, [])
    for f in files:
        if not Path(f).is_file():
            raise FileNotFoundError(f"speech synthesis model not found: {f}")
    # Imports are lazy so choosing one engine never loads the other.
    if cfg.engine == "piper":
        from hark.tts.piper import PiperTTS

        return PiperTTS(cfg)
    if cfg.engine == "kokoro":
        from hark.tts.kokoro import KokoroTTS

        return KokoroTTS(cfg)
    raise ValueError(f"unknown tts engine: {cfg.engine}")
