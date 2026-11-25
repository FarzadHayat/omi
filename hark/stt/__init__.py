"""Speech-to-text engines. create() loads once and warms up."""

from __future__ import annotations

from pathlib import Path

from hark.config import SttConfig
from hark.stt.base import STT, SAMPLE_RATE, StreamingSTT

__all__ = ["STT", "StreamingSTT", "SAMPLE_RATE", "create"]

MODEL_FILES = {
    "moonshine": ("moonshine_dir", ("encoder_model.ort", "decoder_model_merged.ort", "tokens.txt")),
    "nemotron": ("nemotron_dir", ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")),
}


def create(cfg: SttConfig) -> STT:
    if cfg.engine in MODEL_FILES:
        attr, files = MODEL_FILES[cfg.engine]
        folder = Path(getattr(cfg, attr))
        missing = [f for f in files if not (folder / f).is_file()]
        if missing:
            # Checked here: sherpa-onnx only says "Invalid fd was supplied: -1".
            raise FileNotFoundError(f"speech recognition model not found: {folder / missing[0]}")
    # Imports are lazy so choosing one engine never loads the other.
    if cfg.engine == "moonshine":
        from hark.stt.moonshine import MoonshineSTT

        return MoonshineSTT(cfg)
    if cfg.engine == "nemotron":
        from hark.stt.nemotron import NemotronSTT

        return NemotronSTT(cfg)
    raise ValueError(f"unknown stt engine: {cfg.engine}")
