"""Fakes for the component contracts, with controllable delays, so the turn pipeline
can be tested and timed without models or speakers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from hark.events import BrainEvent, TextDelta, TurnContext, TurnDone, WorkerEvent
from hark.memory.base import Hit
from hark.tools.base import Tool, ToolRegistry, ToolResult

SR = 16000


class FakeSTT:
    def __init__(self, text: str = "hello there", delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls = 0

    def transcribe(self, pcm: np.ndarray) -> str:
        self.calls += 1
        time.sleep(self.delay)
        return self.text


class FakeTTS:
    """Yields `seconds_per_word` of audio per word, after `delay` seconds of 'compute'."""

    def __init__(self, delay: float = 0.0, seconds_per_word: float = 0.05, name: str = "fake") -> None:
        self.sample_rate = SR
        self.delay = delay
        self.seconds_per_word = seconds_per_word
        self.name = name
        self.calls: list[tuple[float, str]] = []

    def synth(self, text: str) -> Iterator[np.ndarray]:
        self.calls.append((time.perf_counter(), text))
        time.sleep(self.delay)
        n = max(1, int(len(text.split()) * self.seconds_per_word * SR))
        yield np.full(n, 0.1, dtype=np.float32)


class CaptureOut:
    """AudioOut that records writes and simulates real-time playback.
    `speed` > 1 plays faster than real time so tests stay quick."""

    def __init__(self, speed: float = 1.0) -> None:
        self.speed = speed
        self.writes: list[tuple[float, int]] = []  # (time, samples)
        self.starts: list[float] = []
        self.stops: list[float] = []
        self.drains: list[float] = []
        self.sample_rate = SR
        self._play_until = 0.0
        self._stopped = asyncio.Event()

    async def start(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.starts.append(time.perf_counter())
        self._stopped = asyncio.Event()

    async def write(self, pcm: np.ndarray) -> None:
        now = time.perf_counter()
        self.writes.append((now, len(pcm)))
        self._play_until = max(self._play_until, now) + len(pcm) / self.sample_rate / self.speed

    def play_end(self, index: int) -> float:
        """When write `index` finishes playing, given everything before it."""
        until = 0.0
        for t, n in self.writes[: index + 1]:
            until = max(until, t) + n / self.sample_rate / self.speed
        return until

    async def drain(self) -> None:
        wait = self._play_until - time.perf_counter()
        if wait > 0:
            try:
                await asyncio.wait_for(self._stopped.wait(), wait)
            except TimeoutError:
                pass
        self.drains.append(time.perf_counter())

    async def stop(self) -> None:
        self.stops.append(time.perf_counter())
        self._play_until = 0.0
        self._stopped.set()


class FakeWavOut:
    """WavOut-style AudioOut: collects audio and writes a wav on drain."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.sample_rate = SR
        self.chunks: list[np.ndarray] = []

    async def start(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    async def write(self, pcm: np.ndarray) -> None:
        self.chunks.append(pcm)

    async def drain(self) -> None:
        data = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
        sf.write(self.path, data, self.sample_rate)

    async def stop(self) -> None:
        self.chunks.clear()


class FakeAudioIn:
    def __init__(self, seconds: float = 0.5) -> None:
        self.pcm = np.zeros(int(seconds * SR), dtype=np.float32)
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def read(self) -> np.ndarray | None:
        return None

    async def stop(self) -> np.ndarray:
        self.started = False
        return self.pcm


Script = list[tuple[float, BrainEvent]]


class FakeBrain:
    """Yields a scripted list of (delay_before, event). `script` may also be a
    callable taking the user text, for per-turn scripts."""

    def __init__(self, script: Script | Callable[[str], Script] | None = None, name: str = "fake") -> None:
        self.script = script if script is not None else [(0.0, TextDelta("Okay.")), (0.0, TurnDone())]
        self.name = name
        self.turns: list[tuple[str, TurnContext]] = []
        self.yielded: list[tuple[float, BrainEvent]] = []
        self.cancels = 0
        self.closed = False
        self.started = False
        self.started_with: str | None = None
        self._cancel = asyncio.Event()

    async def start(self, memory_context: str = "") -> None:
        self.started = True
        self.started_with = memory_context

    async def turn(self, user_text: str, ctx: TurnContext) -> AsyncIterator[BrainEvent]:
        self.turns.append((user_text, ctx))
        self._cancel = asyncio.Event()
        script = self.script(user_text) if callable(self.script) else self.script
        for delay, ev in script:
            if delay:
                try:
                    await asyncio.wait_for(self._cancel.wait(), delay)
                    yield TurnDone(cancelled=True)
                    return
                except TimeoutError:
                    pass
            if self._cancel.is_set():
                yield TurnDone(cancelled=True)
                return
            self.yielded.append((time.perf_counter(), ev))
            yield ev
            if isinstance(ev, TurnDone):
                return

    async def cancel(self) -> None:
        self.cancels += 1
        self._cancel.set()

    async def close(self) -> None:
        self.closed = True


class FakeWorker:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[WorkerEvent] = asyncio.Queue()
        self.submitted: list[str] = []
        self.closed = False
        self.name = "fake-worker"

    async def start(self) -> None:
        pass

    async def submit(self, task: str) -> str:
        self.submitted.append(task)
        return f"job{len(self.submitted)}"

    async def events(self) -> AsyncIterator[WorkerEvent]:
        while True:
            yield await self.queue.get()

    def push(self, ev: WorkerEvent) -> None:
        self.queue.put_nowait(ev)

    async def cancel(self, job_id: str | None = None) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


@dataclass
class FakeMemory:
    profile: str = "The user is called Sam."
    facts: list[Hit] = field(default_factory=list)
    logged: list[tuple[str, str, str]] = field(default_factory=list)
    ended: list[tuple[str, str]] = field(default_factory=list)

    def context(self) -> str:
        return self.profile

    def recall(self, query: str, k: int = 4) -> list[Hit]:
        return [h for h in self.facts if any(w in h.text.lower() for w in query.lower().split())][:k]

    def save_fact(self, name: str, body: str, description: str = "", type: str = "user") -> str:
        self.facts.append(Hit("fact", name, body, f"facts/{name}.md"))
        return f"facts/{name}.md"

    def forget(self, query: str) -> list[str]:
        return []

    def log_turn(self, session_id: str, role: str, text: str) -> None:
        self.logged.append((session_id, role, text))

    def end_session(self, session_id: str, digest: str) -> None:
        self.ended.append((session_id, digest))


@dataclass
class Route:
    tool: str
    args: dict[str, Any]
    ack: str


class FakeRouter:
    def __init__(self, routes: dict[str, Route] | None = None) -> None:
        self.routes = routes or {}

    def __call__(self, text: str) -> Route | None:
        return self.routes.get(text.lower().strip(" .!?"))


def make_registry(calls: list[tuple[str, dict[str, Any]]] | None = None) -> ToolRegistry:
    """A registry with a reversible volume tool, a destructive close tool and a
    failing tool."""
    calls = calls if calls is not None else []
    reg = ToolRegistry()

    async def set_volume(level: int) -> ToolResult:
        calls.append(("set_volume", {"level": level}))
        return ToolResult(True, f"Volume is {level} percent.")

    async def close_window() -> ToolResult:
        calls.append(("close_window", {}))
        return ToolResult(True, "Closed the window.")

    async def broken() -> ToolResult:
        raise RuntimeError("boom")

    obj = {"type": "object", "properties": {}}
    reg.register(Tool("set_volume", "Set volume.", obj, "reversible", set_volume))
    reg.register(Tool("close_window", "Close the window.", obj, "destructive", close_window, lambda a: "Close the window?"))
    reg.register(Tool("broken", "Always fails.", obj, "reversible", broken))
    return reg


class Events:
    """Collects emitted events with timestamps."""

    def __init__(self) -> None:
        self.items: list[tuple[float, dict[str, Any]]] = []

    async def __call__(self, ev: dict[str, Any]) -> None:
        self.items.append((time.perf_counter(), ev))

    def kinds(self, kind: str) -> list[dict[str, Any]]:
        return [e for _, e in self.items if e.get("event") == kind]

    def states(self) -> list[str]:
        return [e["state"] for e in self.kinds("state")]

    def first_time(self, pred: Callable[[dict[str, Any]], bool]) -> float | None:
        return next((t for t, e in self.items if pred(e)), None)
