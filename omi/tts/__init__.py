"""Text-to-speech engines (local and cloud). create() loads once and warms up."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omi.config import TtsConfig
from omi.tts.base import TTS

__all__ = ["TTS", "available_tts_engines", "create"]

TTS_ORDER = ("kokoro", "piper", "openai", "elevenlabs")


def available_tts_engines(cfg: TtsConfig) -> list[dict[str, Any]]:
    """Report all supported local and cloud TTS engines and whether each is ready."""
    return [
        {
            "name": "kokoro",
            "label": "Kokoro",
            "kind": "local",
            "installed": Path(cfg.kokoro_model).is_file() and Path(cfg.kokoro_voices).is_file(),
            "detail": f"local · neural ({cfg.kokoro_voice})",
        },
        {
            "name": "piper",
            "label": "Piper",
            "kind": "local",
            "installed": Path(cfg.piper_model).is_file() and Path(cfg.piper_model + ".json").is_file(),
            "detail": "local · ultra-light CPU",
        },
        {
            "name": "openai",
            "label": "OpenAI TTS",
            "kind": "cloud",
            "installed": bool(os.environ.get(cfg.openai_api_key_env, "").strip()),
            "detail": f"cloud · {cfg.openai_model} ({cfg.openai_api_key_env})",
        },
        {
            "name": "elevenlabs",
            "label": "ElevenLabs",
            "kind": "cloud",
            "installed": bool(os.environ.get(cfg.elevenlabs_api_key_env, "").strip()),
            "detail": f"cloud · {cfg.elevenlabs_model} ({cfg.elevenlabs_api_key_env})",
        },
    ]


def create(cfg: TtsConfig) -> TTS:
    files = {
        "piper": [cfg.piper_model, cfg.piper_model + ".json"],
        "kokoro": [cfg.kokoro_model, cfg.kokoro_voices],
    }.get(cfg.engine, [])
    for f in files:
        if not Path(f).is_file():
            raise FileNotFoundError(f"speech synthesis model not found: {f}")
    # Imports are lazy so choosing one engine never loads the others.
    if cfg.engine == "piper":
        from omi.tts.piper import PiperTTS

        return PiperTTS(cfg)
    if cfg.engine == "kokoro":
        from omi.tts.kokoro import KokoroTTS

        return KokoroTTS(cfg)
    if cfg.engine == "openai":
        from omi.tts.cloud import OpenAITTS

        return OpenAITTS(cfg)
    if cfg.engine == "elevenlabs":
        from omi.tts.cloud import ElevenLabsTTS

        return ElevenLabsTTS(cfg)
    raise ValueError(f"unknown tts engine: {cfg.engine}")
