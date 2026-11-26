"""Cute, soft procedural sound effects for Omi (wake, heard, acting, confirm, done, sleep, error).

All cues are synthesized in pure NumPy using warm soft-bell / marimba harmonics
(fundamental + gentle 2nd harmonic + smooth attack/exponential decay) and cached
in memory. Playback runs asynchronously through PipeWire (`pw-cat`) so it never
blocks the event loop or speech pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
from collections.abc import Callable
from time import monotonic

import numpy as np

from hark.audio.pipewire import PipeWireOut
from hark.stt.base import SAMPLE_RATE

log = logging.getLogger(__name__)

SOUNDS = ("wake", "heard", "acting", "confirm", "done", "sleep", "error")


def _bell(
    freq: float,
    dur_s: float,
    *,
    gain: float = 0.12,
    freq_end: float | None = None,
    overtone: float = 0.22,
    decay: float = 5.5,
    rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """One warm marimba/bell note with soft attack, subtle 2nd harmonic, and exponential tail."""
    n = max(1, int(rate * dur_s))
    t = np.arange(n, dtype=np.float32) / rate
    if freq_end is not None and freq_end != freq:
        # Smooth frequency glide (cute droplet/bloop feel)
        inst_f = np.linspace(freq, freq_end, n, dtype=np.float32)
        phase = 2.0 * np.pi * np.cumsum(inst_f) / rate
    else:
        phase = 2.0 * np.pi * freq * t
    wave = np.sin(phase) + overtone * np.sin(2.0 * phase)
    # Soft attack (6 ms) + smooth exponential decay + clean zero-crossing tail
    attack_n = min(n, max(1, int(rate * 0.006)))
    env = np.exp(-decay * (t / max(dur_s, 1e-3))).astype(np.float32)
    env[:attack_n] *= np.linspace(0.0, 1.0, attack_n, dtype=np.float32)
    tail_n = min(n, max(1, int(rate * 0.008)))
    env[-tail_n:] *= np.linspace(1.0, 0.0, tail_n, dtype=np.float32)
    peak = float(np.max(np.abs(wave))) or 1.0
    return (gain / peak * env * wave).astype(np.float32)


def _mix(notes: list[tuple[float, np.ndarray]], rate: int = SAMPLE_RATE) -> np.ndarray:
    """Layer (offset_s, pcm) notes into a single clip so bells ring naturally into each other."""
    total = max(int(offset * rate) + len(pcm) for offset, pcm in notes)
    out = np.zeros(total, dtype=np.float32)
    for offset, pcm in notes:
        i = int(offset * rate)
        out[i : i + len(pcm)] += pcm
    peak = float(np.max(np.abs(out)))
    if peak > 0.25:
        out *= 0.25 / peak
    return out


@functools.cache
def render(name: str, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate and cache a named Omi sound effect as float32 mono PCM."""
    if name == "wake":
        # Cheerful bubbly rising major-third chime (E5 -> G#5)
        return _mix([
            (0.000, _bell(659.25, 0.075, gain=0.11, freq_end=680.0, decay=6.0, rate=rate)),
            (0.048, _bell(830.61, 0.115, gain=0.13, decay=5.0, rate=rate)),
        ], rate)
    if name == "heard":
        # Tiny cozy marimba pop confirming Omi caught what you said
        return _mix([
            (0.000, _bell(523.25, 0.070, gain=0.10, freq_end=659.25, overtone=0.16, decay=7.5, rate=rate)),
        ], rate)
    if name == "acting":
        # Ultra-light playful two-tap sparkle when starting a tool
        return _mix([
            (0.000, _bell(880.00, 0.055, gain=0.075, overtone=0.18, decay=8.0, rate=rate)),
            (0.040, _bell(1108.73, 0.080, gain=0.080, overtone=0.15, decay=6.5, rate=rate)),
        ], rate)
    if name == "confirm":
        # Inquisitive soft rising triad (F5 -> A5 -> C6)
        return _mix([
            (0.000, _bell(698.46, 0.080, gain=0.10, decay=6.0, rate=rate)),
            (0.055, _bell(880.00, 0.085, gain=0.11, decay=5.8, rate=rate)),
            (0.110, _bell(1046.50, 0.130, gain=0.12, decay=4.8, rate=rate)),
        ], rate)
    if name == "done":
        # Warm cozy resolved fourth (G5 -> C6) when a background job finishes
        return _mix([
            (0.000, _bell(783.99, 0.090, gain=0.10, decay=5.5, rate=rate)),
            (0.060, _bell(1046.50, 0.150, gain=0.12, decay=4.5, rate=rate)),
        ], rate)
    if name == "sleep":
        # Gentle descending sigh (G5 -> D5) when call mode drops to standby
        return _mix([
            (0.000, _bell(783.99, 0.080, gain=0.095, decay=6.0, rate=rate)),
            (0.055, _bell(587.33, 0.130, gain=0.090, freq_end=560.0, decay=5.0, rate=rate)),
        ], rate)
    if name == "error":
        # Friendly soft muted low double-tap
        return _mix([
            (0.000, _bell(349.23, 0.075, gain=0.10, overtone=0.12, decay=7.0, rate=rate)),
            (0.060, _bell(277.18, 0.110, gain=0.10, overtone=0.10, decay=6.0, rate=rate)),
        ], rate)
    raise ValueError(f"unknown sound effect: {name!r}")


class SfxPlayer:
    """Fire-and-forget PipeWire sound effect player with per-cue rate limiting."""

    def __init__(
        self,
        *,
        sink: str = "",
        enabled: Callable[[], bool] = lambda: True,
        min_gap_s: dict[str, float] | None = None,
    ) -> None:
        self.sink = sink
        self._enabled = enabled
        self._min_gap = {"acting": 2.5, "heard": 0.4, "wake": 0.3, **(min_gap_s or {})}
        self._last: dict[str, float] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    def play(self, name: str) -> None:
        if name not in SOUNDS or self.sink == "null" or not self._enabled():
            return
        now = monotonic()
        gap = self._min_gap.get(name, 0.15)
        if now - self._last.get(name, -1e9) < gap:
            return
        self._last[name] = now
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._play_async(name))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _play_async(self, name: str) -> None:
        out = PipeWireOut(self.sink)
        try:
            pcm = render(name, SAMPLE_RATE)
            await out.start(SAMPLE_RATE)
            await out.write(pcm)
            await out.drain()
        except Exception:
            log.debug("sfx %s failed", name, exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                await out.stop()
