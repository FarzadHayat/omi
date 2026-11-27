"""NVIDIA Parakeet TDT 0.6B v3 (int8) through sherpa-onnx.

Parakeet TDT v3 is the top local open-weight speech recognition model used by Hex and
Superwhisper: native punctuation and capitalization, 25 European languages, and fast
CPU inference via FastConformer + Token-and-Duration Transducer (nemo_transducer).
"""

from __future__ import annotations

import logging
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import sherpa_onnx as so

from omi.config import SttConfig
from omi.stt.base import SAMPLE_RATE

log = logging.getLogger(__name__)

PARAKEET_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
)
PARAKEET_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def parakeet_installed(folder: str | Path) -> bool:
    d = Path(folder)
    return all((d / name).is_file() for name in PARAKEET_FILES)


def ensure_parakeet_model(folder: str | Path) -> Path:
    """Download and extract the Parakeet TDT v3 int8 bundle if missing."""
    dest = Path(folder)
    if parakeet_installed(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading Parakeet TDT v3 int8 model to %s ...", dest)
    tmp = Path(tempfile.mkdtemp(prefix="omi-parakeet-"))
    try:
        archive = tmp / "parakeet.tar.bz2"
        urllib.request.urlretrieve(PARAKEET_URL, archive)
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(tmp, filter="data")
        extracted = next(
            (p for p in tmp.iterdir() if p.is_dir() and parakeet_installed(p)),
            None,
        )
        if extracted is None:
            raise FileNotFoundError("Parakeet archive did not contain expected ONNX files")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(extracted), str(dest))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


class ParakeetSTT:
    def __init__(self, cfg: SttConfig, *, auto_download: bool = False) -> None:
        folder = Path(cfg.parakeet_dir)
        if auto_download and not parakeet_installed(folder):
            ensure_parakeet_model(folder)
        missing = [f for f in PARAKEET_FILES if not (folder / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Parakeet model not found: {folder / missing[0]} "
                f"(download sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 into {folder})"
            )
        self._rec = so.OfflineRecognizer.from_transducer(
            encoder=str(folder / "encoder.int8.onnx"),
            decoder=str(folder / "decoder.int8.onnx"),
            joiner=str(folder / "joiner.int8.onnx"),
            tokens=str(folder / "tokens.txt"),
            num_threads=cfg.threads,
            provider="cpu",
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )
        # Short warm-up so the first live utterance pays no graph initialization cost.
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))

    def transcribe(self, pcm: np.ndarray) -> str:
        audio = np.asarray(pcm, dtype=np.float32)
        if audio.size == 0:
            return ""
        stream = self._rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._rec.decode_stream(stream)
        return str(stream.result.text).strip()
