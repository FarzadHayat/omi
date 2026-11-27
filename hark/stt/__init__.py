"""Speech-to-text engines (local and cloud). create() loads once and warms up."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hark.config import SttConfig
from hark.stt.base import STT, SAMPLE_RATE, StreamingSTT

__all__ = ["STT", "StreamingSTT", "SAMPLE_RATE", "available_stt_engines", "create"]

MODEL_FILES = {
    "parakeet": ("parakeet_dir", ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")),
    "nemotron": ("nemotron_dir", ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")),
    "moonshine": ("moonshine_dir", ("encoder_model.ort", "decoder_model_merged.ort", "tokens.txt")),
}

CLOUD_STT_ENV = {
    "groq": ("groq_api_key_env", "groq_model", "Groq Whisper"),
    "deepgram": ("deepgram_api_key_env", "deepgram_model", "Deepgram Nova"),
    "openai": ("openai_api_key_env", "openai_model", "OpenAI STT"),
}

STT_ORDER = ("parakeet", "nemotron", "moonshine", "groq", "deepgram", "openai")


def _local_installed(cfg: SttConfig, name: str) -> bool:
    attr, files = MODEL_FILES[name]
    folder = Path(getattr(cfg, attr))
    return all((folder / f).is_file() for f in files)


def available_stt_engines(cfg: SttConfig) -> list[dict[str, Any]]:
    """Report all supported local and cloud STT engines and whether each is ready."""
    rows: list[dict[str, Any]] = []
    local_meta = {
        "parakeet": ("Parakeet v3", "local · punctuated, 25 langs"),
        "nemotron": ("Nemotron", "local · live streaming"),
        "moonshine": ("Moonshine", "local · ultra-light EN"),
    }
    for name in ("parakeet", "nemotron", "moonshine"):
        label, detail = local_meta[name]
        rows.append({
            "name": name,
            "label": label,
            "kind": "local",
            "installed": _local_installed(cfg, name),
            "detail": detail,
        })
    for name in ("groq", "deepgram", "openai"):
        env_attr, model_attr, label = CLOUD_STT_ENV[name]
        env_var = getattr(cfg, env_attr)
        model = getattr(cfg, model_attr)
        has_key = bool(os.environ.get(env_var, "").strip())
        rows.append({
            "name": name,
            "label": label,
            "kind": "cloud",
            "installed": has_key,
            "detail": f"cloud · {model} ({env_var})",
        })
    return rows


def create(cfg: SttConfig) -> STT:
    if cfg.engine in MODEL_FILES:
        attr, files = MODEL_FILES[cfg.engine]
        folder = Path(getattr(cfg, attr))
        missing = [f for f in files if not (folder / f).is_file()]
        if missing:
            # Checked here: sherpa-onnx only says "Invalid fd was supplied: -1".
            raise FileNotFoundError(f"speech recognition model not found: {folder / missing[0]}")
    # Imports are lazy so choosing one engine never loads the others.
    if cfg.engine == "parakeet":
        from hark.stt.parakeet import ParakeetSTT

        return ParakeetSTT(cfg)
    if cfg.engine == "moonshine":
        from hark.stt.moonshine import MoonshineSTT

        return MoonshineSTT(cfg)
    if cfg.engine == "nemotron":
        from hark.stt.nemotron import NemotronSTT

        return NemotronSTT(cfg)
    if cfg.engine == "groq":
        from hark.stt.cloud import GroqSTT

        return GroqSTT(cfg)
    if cfg.engine == "deepgram":
        from hark.stt.cloud import DeepgramSTT

        return DeepgramSTT(cfg)
    if cfg.engine == "openai":
        from hark.stt.cloud import OpenAISTT

        return OpenAISTT(cfg)
    raise ValueError(f"unknown stt engine: {cfg.engine}")
