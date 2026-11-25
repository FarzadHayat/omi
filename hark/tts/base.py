"""Text-to-speech contract."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

import numpy as np


class TTS(Protocol):
    sample_rate: int

    def synth(self, text: str) -> Iterator[np.ndarray]:
        """Blocking generator. One short clause or sentence in, float32 mono chunks out
        at `sample_rate`. The first chunk must come as early as the engine allows.
        Run it in a thread."""
        ...
