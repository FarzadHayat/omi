"""A priority list of brains: the preferred one answers, and when it fails before saying
anything (usage limit, expired login, missing CLI, local server down) the next one takes
the same turn, so the user never has to switch by hand.

A brain that failed sits out for a cooldown, so later turns go straight to the next one
without paying for another failed attempt; after the cooldown the preferred brain gets
tried again. A turn that already spoke or ran a tool is never retried, so nothing is
said or done twice.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from time import monotonic
from typing import Any

from omi.events import BrainEvent, TextDelta, ToolEnd, ToolStart, TurnContext, TurnDone, WorkerDone, WorkerEvent, WorkerProgress

log = logging.getLogger(__name__)

COOLDOWN_S = 15 * 60

# (provider, model); an empty model means the provider's configured model.
Entry = tuple[str, str]
MakeBrain = Callable[[str, str], Any]
MakeWorker = Callable[[str, str], Any]


def parse_entries(raw: list[str] | str) -> list[Entry]:
    """'opencode:opencode/deepseek-free, local' -> [('opencode', 'opencode/deepseek-free'), ('local', '')]."""
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    out: list[Entry] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        provider, _, model = text.partition(":")
        entry = (provider.strip().lower(), model.strip())
        if entry[0] and entry not in out:
            out.append(entry)
    return out


def format_entries(entries: list[Entry]) -> list[str]:
    return [f"{p}:{m}" if m else p for p, m in entries]


class FallbackBrain:
    """Wraps the talker. `name` stays the preferred provider; `active` is who answered last."""

    def __init__(
        self,
        entries: list[Entry],
        make: MakeBrain,
        *,
        cooldown_s: float = COOLDOWN_S,
        on_switch: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        if not entries:
            raise ValueError("a fallback chain needs at least one brain")
        self.entries = entries
        self._make = make
        self.cooldown_s = cooldown_s
        self.on_switch = on_switch
        self._brains: dict[int, Any] = {}
        self._failed_at: dict[int, float] = {}
        self._reasons: dict[int, str] = {}
        self._current: Any = None
        self._memory_context = ""
        self.active_index = 0

    # --- what status shows ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self.entries[0][0]

    @property
    def active(self) -> str:
        provider, model = self.entries[self.active_index]
        return f"{provider}:{model}" if model else provider

    @property
    def error(self) -> str | None:
        """Only an error when no brain in the chain can answer."""
        if self.active_index == 0:
            primary = self._brains.get(0)
            return getattr(primary, "error", None) if primary is not None else None
        return None

    @property
    def fallback_reason(self) -> str:
        """Why the preferred brain is sitting out, empty when it is answering."""
        return self._reasons.get(0, "") if self.active_index != 0 else ""

    # --- lifecycle -----------------------------------------------------------------

    async def _brain(self, idx: int) -> Any:
        brain = self._brains.get(idx)
        if brain is None:
            provider, model = self.entries[idx]
            brain = self._make(provider, model)
            self._brains[idx] = brain
            await brain.start(self._memory_context)
        return brain

    async def start(self, memory_context: str = "") -> None:
        self._memory_context = memory_context
        for idx, brain in list(self._brains.items()):
            if idx != 0:
                await brain.start(memory_context)
        if 0 in self._brains:
            await self._brains[0].start(memory_context)
        else:
            await self._brain(0)

    def _order(self) -> list[int]:
        now = monotonic()
        ready = [i for i in range(len(self.entries)) if now - self._failed_at.get(i, -1e18) >= self.cooldown_s]
        return ready or list(range(len(self.entries)))  # everything failed lately: try them all again

    def _fail(self, idx: int, reason: str) -> None:
        self._failed_at[idx] = monotonic()
        self._reasons[idx] = reason
        log.warning("brain %s failed, falling back: %s", self.entries[idx][0], reason)

    async def turn(self, user_text: str, ctx: TurnContext) -> AsyncIterator[BrainEvent]:
        last_error = "no brain could answer"
        for idx in self._order():
            try:
                brain = await self._brain(idx)
            except Exception as exc:  # could not start (missing CLI, local server failed)
                self._brains.pop(idx, None)
                self._fail(idx, str(exc) or type(exc).__name__)
                last_error = self._reasons[idx]
                continue
            if getattr(brain, "error", None) and len(self.entries) > 1:
                self._fail(idx, brain.error)
                last_error = brain.error
                continue
            await self._switch_to(idx)
            spoke = acted = False
            async for ev in brain.turn(user_text, ctx):
                if isinstance(ev, TurnDone):
                    if ev.error and not ev.cancelled and not spoke and not acted:
                        self._fail(idx, ev.error)
                        last_error = ev.error
                        break  # nothing reached the user: the next brain takes this turn
                    if not ev.error:
                        self._failed_at.pop(idx, None)
                        self._reasons.pop(idx, None)
                    yield ev
                    return
                if isinstance(ev, TextDelta) and ev.text.strip():
                    spoke = True
                elif isinstance(ev, (ToolStart, ToolEnd)):
                    acted = True
                yield ev
            else:
                yield TurnDone()
                return
        yield TurnDone(error=last_error)

    async def _switch_to(self, idx: int) -> None:
        self._current = self._brains[idx]
        if idx != self.active_index:
            self.active_index = idx
            if self.on_switch is not None:
                try:
                    await self.on_switch(self.active, self.fallback_reason)
                except Exception:
                    log.exception("fallback notice failed")

    async def cancel(self) -> None:
        if self._current is not None:
            await self._current.cancel()

    async def close(self) -> None:
        brains, self._brains = list(self._brains.values()), {}
        for brain in brains:
            try:
                await brain.close()
            except Exception:
                log.exception("closing a fallback brain failed")


class FallbackWorker:
    """A priority list of background workers: the preferred one runs jobs, and when it
    fails before doing any work (usage limit, expired login, missing CLI) the next one
    takes the same job under the same job_id.

    A worker that already reported progress (started a tool) is never retried, so no
    side effect runs twice. A failed worker sits out for a cooldown before being tried
    again.
    """

    def __init__(
        self,
        entries: list[Entry],
        make: MakeWorker,
        *,
        cooldown_s: float = COOLDOWN_S,
        on_switch: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        if not entries:
            raise ValueError("a fallback chain needs at least one worker")
        self.entries = entries
        self._make = make
        self.cooldown_s = cooldown_s
        self.on_switch = on_switch
        self._workers: dict[int, Any] = {}
        self._iters: dict[int, AsyncIterator[WorkerEvent]] = {}
        self._failed_at: dict[int, float] = {}
        self._reasons: dict[int, str] = {}
        self._jobs: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._events: asyncio.Queue[WorkerEvent] = asyncio.Queue()
        self._running: str | None = None
        self._cancelled_job: str | None = None
        self._active_worker: Any = None
        self._active_inner_id: str | None = None
        self._dropped: set[str] = set()
        self._runner: asyncio.Task[None] | None = None
        self.active_index = 0

    @property
    def name(self) -> str:
        return self.entries[0][0]

    @property
    def active(self) -> str:
        provider, model = self.entries[self.active_index]
        return f"{provider}:{model}" if model else provider

    @property
    def fallback_reason(self) -> str:
        return self._reasons.get(0, "") if self.active_index != 0 else ""

    async def _worker(self, idx: int) -> Any:
        worker = self._workers.get(idx)
        if worker is None:
            provider, model = self.entries[idx]
            worker = self._make(provider, model)
            try:
                await worker.start()
            except Exception:
                with contextlib.suppress(Exception):
                    await worker.close()
                raise
            self._workers[idx] = worker
            self._iters[idx] = worker.events()
        return worker

    async def start(self) -> None:
        last_exc: Exception | None = None
        started = False
        for idx in range(len(self.entries)):
            try:
                await self._worker(idx)
                await self._switch_to(idx)
                started = True
                break
            except Exception as exc:
                self._workers.pop(idx, None)
                self._iters.pop(idx, None)
                self._fail(idx, str(exc) or type(exc).__name__)
                last_exc = exc
        if not started and last_exc is not None:
            raise last_exc
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run())

    async def submit(self, task: str) -> str:
        job_id = uuid.uuid4().hex[:8]
        self._jobs.put_nowait((job_id, task))
        return job_id

    async def events(self) -> AsyncIterator[WorkerEvent]:
        while True:
            yield await self._events.get()

    async def cancel(self, job_id: str | None = None) -> None:
        queued = list(self._jobs._queue)  # type: ignore[attr-defined]
        for jid, _ in queued:
            if job_id in (None, jid) and jid not in self._dropped:
                self._dropped.add(jid)
                self._events.put_nowait(WorkerDone(jid, False, "Cancelled."))
        if self._running is not None and job_id in (None, self._running):
            self._cancelled_job = self._running
            if self._active_worker is not None:
                await self._active_worker.cancel(self._active_inner_id)

    async def close(self) -> None:
        if self._runner:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
            self._runner = None
        workers, self._workers = list(self._workers.values()), {}
        self._iters.clear()
        for worker in workers:
            try:
                await worker.close()
            except Exception:
                log.exception("closing a fallback worker failed")

    def _order(self) -> list[int]:
        now = monotonic()
        ready = [i for i in range(len(self.entries)) if now - self._failed_at.get(i, -1e18) >= self.cooldown_s]
        return ready or list(range(len(self.entries)))

    def _fail(self, idx: int, reason: str) -> None:
        self._failed_at[idx] = monotonic()
        self._reasons[idx] = reason
        provider, model = self.entries[idx]
        label = f"{provider}:{model}" if model else provider
        log.warning("worker %s failed, falling back: %s", label, reason)

    async def _switch_to(self, idx: int) -> None:
        if idx != self.active_index:
            self.active_index = idx
            if self.on_switch is not None:
                try:
                    await self.on_switch(self.active, self.fallback_reason)
                except Exception:
                    log.exception("worker fallback notice failed")

    async def _run(self) -> None:
        while True:
            job_id, task = await self._jobs.get()
            if job_id in self._dropped:
                self._dropped.discard(job_id)
                continue
            self._running = job_id
            self._cancelled_job = None
            try:
                self._events.put_nowait(await self._job(job_id, task))
            except Exception as exc:  # one failed job must not stop the queue
                self._events.put_nowait(WorkerDone(job_id, False, f"The job failed: {exc}"))
            finally:
                self._running = None
                self._cancelled_job = None

    async def _job(self, job_id: str, task: str) -> WorkerDone:
        last_error = "no worker could run the job"
        for idx in self._order():
            if self._cancelled_job == job_id:
                return WorkerDone(job_id, False, "Stopped.")
            try:
                worker = await self._worker(idx)
            except Exception as exc:
                self._workers.pop(idx, None)
                self._iters.pop(idx, None)
                self._fail(idx, str(exc) or type(exc).__name__)
                last_error = self._reasons[idx]
                continue
            await self._switch_to(idx)
            progressed = False
            try:
                inner_id = await worker.submit(task)
            except Exception as exc:
                self._fail(idx, str(exc) or type(exc).__name__)
                last_error = self._reasons[idx]
                continue
            self._active_worker = worker
            self._active_inner_id = inner_id
            it = self._iters[idx]
            try:
                while True:
                    ev = await anext(it)
                    if ev.job_id != inner_id:
                        continue
                    if isinstance(ev, WorkerProgress):
                        progressed = True
                        self._events.put_nowait(WorkerProgress(job_id, ev.text))
                    elif isinstance(ev, WorkerDone):
                        if ev.ok:
                            self._failed_at.pop(idx, None)
                            self._reasons.pop(idx, None)
                            return WorkerDone(job_id, True, ev.summary)
                        if self._cancelled_job == job_id or progressed:
                            return WorkerDone(job_id, False, ev.summary)
                        self._fail(idx, ev.summary or "worker failed")
                        last_error = ev.summary or last_error
                        break
            except StopAsyncIteration:
                self._workers.pop(idx, None)
                self._iters.pop(idx, None)
                if self._cancelled_job == job_id or progressed:
                    return WorkerDone(job_id, False, "Stopped." if self._cancelled_job == job_id else "The worker stopped.")
                self._fail(idx, "worker event stream ended")
                last_error = self._reasons[idx]
            finally:
                self._active_worker = None
                self._active_inner_id = None
        return WorkerDone(job_id, False, last_error)
