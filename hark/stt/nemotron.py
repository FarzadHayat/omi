"""Nemotron streaming transducer through sherpa-onnx. Implements StreamingSTT."""

from __future__ import annotations

import numpy as np
import sherpa_onnx as so

from hark.config import SttConfig
from hark.stt.base import SAMPLE_RATE

TAIL_S = 0.32  # silence appended at the end so the last words leave the look-ahead


class NemotronSTT:
    def __init__(self, cfg: SttConfig) -> None:
        d = cfg.nemotron_dir
        self._rec = so.OnlineRecognizer.from_transducer(
            encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
            joiner=f"{d}/joiner.int8.onnx", tokens=f"{d}/tokens.txt",
            num_threads=cfg.threads,
        )
        self._stream = self._rec.create_stream()
        t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        self.transcribe(0.1 * np.sin(2 * np.pi * 180 * t) * np.sin(2 * np.pi * 3 * t))

    def start(self) -> None:
        self._stream = self._rec.create_stream()

    def _decode(self) -> None:
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)

    def _text(self) -> str:
        return str(self._rec.get_result(self._stream)).strip()

    def feed(self, pcm: np.ndarray) -> str:
        self._stream.accept_waveform(SAMPLE_RATE, np.asarray(pcm, dtype=np.float32))
        self._decode()
        return self._text()

    def finish(self) -> str:
        self._stream.accept_waveform(SAMPLE_RATE, np.zeros(int(TAIL_S * SAMPLE_RATE), np.float32))
        self._stream.input_finished()
        self._decode()
        return self._text()

    def transcribe(self, pcm: np.ndarray) -> str:
        self.start()
        self._stream.accept_waveform(SAMPLE_RATE, np.asarray(pcm, dtype=np.float32))
        return self.finish()
