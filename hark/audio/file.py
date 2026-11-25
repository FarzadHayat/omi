"""File and null sinks: tests and `hark wav --out` use these instead of a sound card."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class WavOut:
    """Collects the reply and writes it as 16-bit mono wav when playback ends."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._rate = 16000
        self._parts: list[np.ndarray] = []

    async def start(self, sample_rate: int) -> None:
        self._rate = sample_rate
        self._parts = []

    async def write(self, pcm: np.ndarray) -> None:
        self._parts.append(np.asarray(pcm, dtype=np.float32))

    def _flush(self) -> None:
        audio = np.concatenate(self._parts) if self._parts else np.zeros(0, np.float32)
        data = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(self._rate)
            fh.writeframes(data.tobytes())

    async def drain(self) -> None:
        self._flush()

    async def stop(self) -> None:
        """Keep what was played so far; a cut-off reply is still worth inspecting."""
        self._flush()


class NullOut:
    """Discards audio. Lets the pipeline run headless."""

    async def start(self, sample_rate: int) -> None:
        pass

    async def write(self, pcm: np.ndarray) -> None:
        pass

    async def drain(self) -> None:
        pass

    async def stop(self) -> None:
        pass
