"""Moonshine v2 base through sherpa-onnx: whole utterance in, text out."""

from __future__ import annotations

import numpy as np
import sherpa_onnx as so

from omi.config import SttConfig
from omi.stt.base import SAMPLE_RATE


class MoonshineSTT:
    def __init__(self, cfg: SttConfig) -> None:
        d = cfg.moonshine_dir
        self._rec = so.OfflineRecognizer.from_moonshine_v2(
            encoder=f"{d}/encoder_model.ort", decoder=f"{d}/decoder_model_merged.ort",
            tokens=f"{d}/tokens.txt", num_threads=cfg.threads,
        )
        # One decode on synthetic voiced noise so the first real call pays no init cost.
        t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        self.transcribe(0.1 * np.sin(2 * np.pi * 180 * t) * np.sin(2 * np.pi * 3 * t))

    def transcribe(self, pcm: np.ndarray) -> str:
        stream = self._rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.asarray(pcm, dtype=np.float32))
        self._rec.decode_stream(stream)
        return stream.result.text.strip()
