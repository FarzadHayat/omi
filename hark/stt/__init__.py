"""Speech-to-text engines. create() loads once and warms up."""

from __future__ import annotations

from hark.config import SttConfig
from hark.stt.base import STT, SAMPLE_RATE, StreamingSTT

__all__ = ["STT", "StreamingSTT", "SAMPLE_RATE", "create"]


def create(cfg: SttConfig) -> STT:
    # Imports are lazy so choosing one engine never loads the other.
    if cfg.engine == "moonshine":
        from hark.stt.moonshine import MoonshineSTT

        return MoonshineSTT(cfg)
    if cfg.engine == "nemotron":
        from hark.stt.nemotron import NemotronSTT

        return NemotronSTT(cfg)
    raise ValueError(f"unknown stt engine: {cfg.engine}")
