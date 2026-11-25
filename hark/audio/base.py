"""Audio in and out contracts. Real implementations use PipeWire; tests use files."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AudioOut(Protocol):
    async def start(self, sample_rate: int) -> None:
        """Open a playback stream. Called once per assistant reply."""
        ...

    async def write(self, pcm: np.ndarray) -> None:
        """Queue float32 mono audio. Must not block for the length of the audio."""
        ...

    async def drain(self) -> None:
        """Wait until everything queued has been played, then close the stream."""
        ...

    async def stop(self) -> None:
        """Cut playback now and drop queued audio (barge-in, stop key)."""
        ...


class AudioIn(Protocol):
    async def start(self) -> None:
        """Start capturing float32 mono 16 kHz from the microphone."""
        ...

    async def read(self) -> np.ndarray | None:
        """Next chunk of about 32 ms, or None once stopped and empty."""
        ...

    async def stop(self) -> np.ndarray:
        """Stop capturing. Return all audio captured since start()."""
        ...
