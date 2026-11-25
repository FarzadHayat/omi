"""Speech-to-text contract. Audio is always float32 mono at 16 kHz, values in [-1, 1]."""

from __future__ import annotations

from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000


class STT(Protocol):
    def transcribe(self, pcm: np.ndarray) -> str:
        """Blocking. Whole utterance in, final text out. Call it from a thread."""
        ...


class StreamingSTT(STT, Protocol):
    def start(self) -> None:
        """Begin a new utterance."""
        ...

    def feed(self, pcm: np.ndarray) -> str:
        """Blocking. Add audio, return the partial text so far."""
        ...

    def finish(self) -> str:
        """Blocking. End the utterance, return the final text."""
        ...
