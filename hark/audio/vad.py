"""Silero VAD through sherpa-onnx (no torch) as a hands-free end-of-turn detector."""

from __future__ import annotations

import shutil
import urllib.request

import numpy as np
import sherpa_onnx as so

from hark.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "silero" / "silero_vad.onnx"
MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
WINDOW = 512  # samples, 32 ms
RATE = 16000
ONSET_WINDOWS = 3  # consecutive speech windows before we call it speech (skips clicks)


def ensure_model() -> str:
    """Download the model on first use."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.replace(MODEL_PATH)
    return str(MODEL_PATH)


class EndpointDetector:
    """Feed 16 kHz float32 audio of any size. `ended` turns True after speech followed by
    `silence_ms` of silence; `speaking` is True while speech is currently detected."""

    def __init__(self, silence_ms: int = 300, threshold: float = 0.5) -> None:
        cfg = so.VadModelConfig(
            # The model's own debounce is zeroed: onset and silence_ms are handled below,
            # so the end of turn lands exactly silence_ms after the last speech.
            silero_vad=so.SileroVadModelConfig(model=ensure_model(), threshold=threshold,
                                               min_silence_duration=0.0,
                                               min_speech_duration=0.0, window_size=WINDOW),
            sample_rate=RATE, num_threads=1,
        )
        self._model = so.VadModel.create(cfg)
        self._silence_windows = max(1, round(silence_ms * RATE / 1000 / WINDOW))
        self.reset()

    def reset(self) -> None:
        self._model.reset()
        self._buf = np.zeros(0, np.float32)
        self._run = 0  # consecutive speech windows
        self._quiet = 0  # consecutive non-speech windows since last speech
        self._started = False
        self.speaking = False
        self.ended = False

    def feed(self, chunk: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, np.asarray(chunk, dtype=np.float32)])
        n = len(self._buf) // WINDOW * WINDOW
        for i in range(0, n, WINDOW):
            self._step(self._model.is_speech(self._buf[i : i + WINDOW]))
        self._buf = self._buf[n:]

    def _step(self, speech: bool) -> None:
        if self.ended:
            return
        if speech:
            self._run += 1
            self._quiet = 0
            if self._run >= ONSET_WINDOWS:
                self._started = self.speaking = True
        else:
            self._run = 0
            self.speaking = False
            if self._started:
                self._quiet += 1
                self.ended = self._quiet >= self._silence_windows
