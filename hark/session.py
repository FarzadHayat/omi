"""One conversation: the turn pipeline, spoken output, confirmation and worker narration.

A turn is: audio -> STT (thread) or text -> router; a router hit runs one tool and
speaks a short ack, anything else goes to the brain, whose text is cut into clauses
and synthesised (thread) while earlier clauses play. Everything is cancellable so the
user can talk over the assistant.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

import numpy as np

from hark import config
from hark.audio.base import AudioOut
from hark.brain.base import Brain, Worker
from hark.events import (
    State,
    TextDelta,
    ToolEnd,
    ToolStart,
    TurnContext,
    TurnDone,
    WorkerDone,
    WorkerProgress,
)
from hark.memory.base import Memory
from hark.speech import ClauseChunker, clean_for_speech
from hark.stt.base import STT
from hark.tools.base import ToolRegistry
from hark.tts.base import TTS

log = logging.getLogger(__name__)

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]
RouteFn = Callable[[str], Any]  # returns a Route (tool, args, ack) or None
SessionEndFn = Callable[[str, str], Awaitable[None]]

HISTORY_LIMIT = 12
CANCEL_BUDGET_S = 0.08  # stop() waits at most this long for the turn to wind down
# Local generation runs at about 12 tokens a second, so every word in the first chunk
# costs about 100 ms of silence; 5 words still sounds natural.
FIRST_CHUNK_WORDS = 5
SORRY = "Sorry, something went wrong."

_NO = re.compile(r"\b(no|nope|cancel|stop|do not|don't|dont)\b")
_YES = re.compile(r"\b(yes|yeah|yep|yup|do it|go ahead|confirm|confirmed)\b")


def parse_yes_no(text: str) -> bool:
    """A spoken confirmation answer. Only a clear yes counts; a no wins over a yes."""
    t = text.lower().replace("’", "'")
    if _NO.search(t):
        return False
    return bool(_YES.search(t))


@dataclass
class TurnResult:
    reply: str
    user_text: str
    timing: dict[str, Any]
    cancelled: bool = False
    error: str | None = None


_END = object()


class _Speaker:
    """Text chunks in, audio out, in order. TTS runs in a thread and streams its
    chunks back, and AudioOut.write does not block, so clause 2 is synthesised
    while clause 1 is still playing."""

    def __init__(
        self,
        tts: TTS,
        out: AudioOut,
        quiet: bool,
        on_first: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.tts = tts
        self.out = out
        self.quiet = quiet
        self.on_first = on_first
        self.queued: list[str] = []
        self.spoken: list[str] = []
        self._q: asyncio.Queue[str | None] = asyncio.Queue()
        self._abort = threading.Event()
        self._opened = False
        self._cancelled = False
        self._first_done = False
        self._task = asyncio.create_task(self._run())

    def say(self, text: str) -> None:
        if text and not self._cancelled:
            self.queued.append(text)
            self._q.put_nowait(text)

    async def finish(self) -> None:
        """Wait until everything queued has been played."""
        self._q.put_nowait(None)
        await self._task
        if self._opened:
            await self.out.drain()

    async def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self._abort.set()
        self._task.cancel()
        if self._opened:
            try:
                await self.out.stop()
            except Exception:
                log.exception("audio stop failed")

    async def _first(self) -> None:
        if not self._first_done:
            self._first_done = True
            if self.on_first:
                await self.on_first()

    async def _run(self) -> None:
        while (text := await self._q.get()) is not None:
            if self.quiet:
                self.spoken.append(text)
                await self._first()
                continue
            heard = False
            async for pcm in self._synth(text):
                if not self._opened:
                    await self.out.start(self.tts.sample_rate)
                    self._opened = True
                await self.out.write(pcm)
                if not heard:
                    heard = True
                    self.spoken.append(text)
                    await self._first()

    async def _synth(self, text: str) -> AsyncIterator[np.ndarray]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Any] = asyncio.Queue()
        abort = self._abort
        tts = self.tts

        def put(item: Any) -> None:
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:  # loop closed after a cancel; nobody is listening
                pass

        def work() -> None:
            try:
                for pcm in tts.synth(text):
                    if abort.is_set():
                        break
                    put(pcm)
            except Exception as exc:
                put(exc)
            finally:
                put(_END)

        threading.Thread(target=work, name="hark-tts", daemon=True).start()
        while (item := await q.get()) is not _END:
            if isinstance(item, Exception):
                log.error("tts failed on %r: %s", text, item)
                continue
            yield item


@dataclass
class _Turn:
    sink: EmitFn | None = None
    task: asyncio.Task[TurnResult] | None = None
    speaker: _Speaker | None = None
    cancelled: bool = False
    trace: dict[str, Any] = field(default_factory=dict)


class Session:
    """Holds one conversation and runs its turns, one at a time."""

    def __init__(
        self,
        *,
        stt: STT,
        tts: TTS,
        brain: Brain,
        registry: ToolRegistry,
        router: RouteFn,
        out: AudioOut,
        memory: Memory | None = None,
        worker: Worker | None = None,
        emit: EmitFn | None = None,
        on_session_end: SessionEndFn | None = None,
        state_dir: Path | None = None,
        idle_timeout: float = 300.0,
        confirm_timeout: float = 20.0,
        progress_interval: float = 8.0,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.brain = brain
        self.registry = registry
        self.router = router
        self.out = out
        self.memory = memory
        self.worker = worker
        self.emit = emit
        self.on_session_end = on_session_end
        self.state_dir = Path(state_dir) if state_dir else config.STATE_DIR
        self.idle_timeout = idle_timeout
        self.confirm_timeout = confirm_timeout
        self.progress_interval = progress_interval

        self.state = State.IDLE
        self.session_id = _new_session_id()
        self.history: list[dict[str, str]] = []
        self.transcript: list[dict[str, str]] = []  # whole session, for on_session_end
        self.last_timing: dict[str, Any] | None = None
        self._turn: _Turn | None = None
        self._start_lock = asyncio.Lock()
        self._confirm: asyncio.Future[bool] | None = None
        self._narration: asyncio.Task[None] | None = None
        self._pending_done: list[str] = []
        self._last_progress = -1e9
        self._last_activity = monotonic()
        self._bg: set[asyncio.Task[Any]] = set()
        self._loops: list[asyncio.Task[None]] = []

    # --- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        if self.worker is not None:
            self._loops.append(asyncio.create_task(self._consume_worker()))
        self._loops.append(asyncio.create_task(self._idle_watch()))

    async def close(self) -> None:
        await self.stop()
        for task in self._loops:
            task.cancel()
        await asyncio.gather(*self._loops, return_exceptions=True)
        self._loops.clear()

    # --- turns -------------------------------------------------------------------

    async def run_turn(
        self,
        text: str | None = None,
        pcm: np.ndarray | None = None,
        *,
        out: AudioOut | None = None,
        quiet: bool = False,
        sink: EmitFn | None = None,
    ) -> TurnResult:
        """Run one turn from text or 16 kHz audio. A running turn is cancelled first
        (barge-in), unless a confirmation is pending: then this turn is the answer."""
        t0 = perf_counter()
        self._last_activity = monotonic()
        if self._confirm is not None and not self._confirm.done():
            return await self._answer_by_voice(text, pcm, t0, sink)
        async with self._start_lock:
            await self.stop()
            turn = _Turn(sink=sink)
            self._turn = turn
            turn.task = asyncio.create_task(self._run(turn, text, pcm, out or self.out, quiet, t0))
        # Shielded: a client hanging up does not cancel the turn; stop() does.
        return await asyncio.shield(turn.task)

    async def set_listening(self, on: bool) -> None:
        """Push-to-talk: show that the microphone is open (or closed again)."""
        if on:
            await self._set_state(State.LISTENING)
        elif self.state is State.LISTENING:
            await self._set_state(State.IDLE)

    async def stop(self) -> None:
        """Cut speech and cancel the running turn, confirmation and narration."""
        self._resolve_confirm(False)
        waits: list[asyncio.Future[Any]] = []
        if self._narration is not None and not self._narration.done():
            self._narration.cancel()
            waits.append(self._narration)
        turn = self._turn
        if turn is not None and turn.task is not None and not turn.task.done():
            turn.cancelled = True
            turn.task.cancel()
            waits.append(turn.task)
            waits.append(asyncio.ensure_future(_quietly(self.brain.cancel())))
            if turn.speaker is not None:
                waits.append(asyncio.ensure_future(turn.speaker.cancel()))
        if waits:
            await asyncio.wait(waits, timeout=CANCEL_BUDGET_S)
        if self._turn is turn:
            self._turn = None
        await self._set_state(State.IDLE)

    async def _run(
        self,
        turn: _Turn,
        text: str | None,
        pcm: np.ndarray | None,
        out: AudioOut,
        quiet: bool,
        t0: float,
    ) -> TurnResult:
        timing: dict[str, Any] = {
            "stt_ms": None,
            "route_ms": None,
            "first_text_ms": None,
            "first_audio_ms": None,
            "action_ms": None,
            "total_ms": None,
            "path": None,
        }
        turn.trace = timing

        def ms() -> float:
            return round((perf_counter() - t0) * 1000, 1)

        async def on_first() -> None:
            if not quiet:
                timing["first_audio_ms"] = ms()
            if self.state not in (State.CONFIRMING, State.ACTING):
                await self._set_state(State.SPEAKING, turn)

        user_text = (text or "").strip()
        error: str | None = None
        speaker: _Speaker | None = None
        try:
            await self._set_state(State.THINKING, turn)
            if pcm is not None:
                user_text = (await asyncio.to_thread(self.stt.transcribe, pcm)).strip()
                timing["stt_ms"] = ms()
            if user_text:
                await self._emit_turn(turn, transcript("user", user_text))
                speaker = _Speaker(self.tts, out, quiet, on_first)
                turn.speaker = speaker
                route = self._route(user_text)
                timing["route_ms"] = ms()
                if route is not None:
                    timing["path"] = "router"
                    await self._run_route(turn, route, speaker, timing, ms)
                else:
                    timing["path"] = "talker"
                    error = await self._run_brain(turn, user_text, speaker, timing, ms)
                timing["total_ms"] = ms()
                await speaker.finish()
        except asyncio.CancelledError:
            turn.cancelled = True
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            if speaker is not None:
                await speaker.cancel()
        except Exception as exc:
            log.exception("turn failed")
            error = str(exc) or type(exc).__name__
            if speaker is not None:
                await speaker.cancel()
        if timing["total_ms"] is None:
            timing["total_ms"] = ms()
        return await self._finish_turn(turn, user_text, speaker, timing, error)

    def _route(self, text: str) -> Any:
        try:
            return self.router(text)
        except Exception:
            log.exception("router failed on %r", text)
            return None

    async def _run_route(
        self, turn: _Turn, route: Any, speaker: _Speaker, timing: dict[str, Any], ms: Callable[[], float]
    ) -> None:
        if route.tool == "stop":
            return  # run_turn already cut the previous turn; a spoken "stop" says nothing
        await self._set_state(State.ACTING, turn)
        await self._emit_turn(turn, {"event": "tool", "phase": "start", "name": route.tool, "args": route.args})
        result = await self.registry.call(route.tool, route.args)
        timing["action_ms"] = ms()
        await self._emit_turn(
            turn,
            {"event": "tool", "phase": "end", "name": route.tool, "args": route.args, "ok": result.ok, "text": result.text},
        )
        say = route.ack if (route.ack and result.ok) else result.text
        timing["first_text_ms"] = ms()
        await self._set_state(State.THINKING, turn)
        speaker.say(clean_for_speech(say))

    async def _run_brain(
        self, turn: _Turn, user_text: str, speaker: _Speaker, timing: dict[str, Any], ms: Callable[[], float]
    ) -> str | None:
        ctx = TurnContext(session_id=self.session_id, history=list(self.history))
        if self.memory is not None:
            try:
                ctx.memory_context = self.memory.context()
                ctx.recalled = [h.text for h in self.memory.recall(user_text)]
            except Exception:
                log.exception("memory recall failed")
        chunker = ClauseChunker(first_words=FIRST_CHUNK_WORDS)
        error: str | None = None

        def say(chunks: list[str]) -> None:
            for chunk in chunks:
                cleaned = clean_for_speech(chunk)
                if cleaned:
                    speaker.say(cleaned)

        gen = self.brain.turn(user_text, ctx)
        try:
            async for ev in gen:
                if isinstance(ev, TextDelta):
                    if timing["first_text_ms"] is None and ev.text.strip():
                        timing["first_text_ms"] = ms()
                    say(chunker.feed(ev.text))
                elif isinstance(ev, ToolStart):
                    await self._set_state(State.ACTING, turn)
                    await self._emit_turn(turn, {"event": "tool", "phase": "start", "name": ev.name, "args": ev.args})
                elif isinstance(ev, ToolEnd):
                    timing["action_ms"] = ms()
                    await self._emit_turn(
                        turn, {"event": "tool", "phase": "end", "name": ev.name, "ok": ev.ok, "text": ev.text}
                    )
                    after = State.SPEAKING if timing["first_audio_ms"] or speaker.spoken else State.THINKING
                    await self._set_state(after, turn)
                elif isinstance(ev, TurnDone):
                    error = ev.error
                    break
        finally:
            try:
                await asyncio.wait_for(gen.aclose(), 0.05)
            except BaseException:
                pass
        say(chunker.flush())
        if error and not speaker.queued:
            speaker.say(SORRY)
        return error

    async def _finish_turn(
        self,
        turn: _Turn,
        user_text: str,
        speaker: _Speaker | None,
        timing: dict[str, Any],
        error: str | None,
    ) -> TurnResult:
        if speaker is None:
            reply = ""
        elif turn.cancelled:
            reply = " ".join(speaker.spoken)
        else:
            reply = " ".join(speaker.queued)
        if reply:
            await self._emit_turn(turn, transcript("assistant", reply))
        trace = dict(timing, cancelled=turn.cancelled, session_id=self.session_id, ts=datetime.now().isoformat(timespec="seconds"))
        if user_text:
            self.last_timing = trace
            await self._emit_turn(turn, {"event": "timing", **timing})
            self._spawn(asyncio.to_thread(self._append_trace, trace))
            self._remember(user_text, reply)
        if self._turn is turn:
            self._turn = None
            await self._set_state(State.IDLE, turn)
        self._last_activity = monotonic()
        self._after_idle()
        return TurnResult(reply=reply, user_text=user_text, timing=timing, cancelled=turn.cancelled, error=error)

    def _remember(self, user_text: str, reply: str) -> None:
        msgs = [{"role": "user", "content": user_text}]
        if reply:
            msgs.append({"role": "assistant", "content": reply})
        self.history = (self.history + msgs)[-HISTORY_LIMIT:]
        self.transcript.extend(msgs)
        if self.memory is not None:
            self._spawn(asyncio.to_thread(self._log_memory, self.session_id, msgs))

    def _log_memory(self, session_id: str, msgs: list[dict[str, str]]) -> None:
        assert self.memory is not None
        try:
            for m in msgs:
                self.memory.log_turn(session_id, m["role"], m["content"])
        except Exception:
            log.exception("memory log_turn failed")

    def _append_trace(self, trace: dict[str, Any]) -> None:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            with (self.state_dir / "turns.jsonl").open("a") as fh:
                fh.write(json.dumps(trace) + "\n")
        except OSError:
            log.exception("could not write turns.jsonl")

    # --- confirmation ------------------------------------------------------------

    async def confirm(self, question: str) -> bool:
        """The registry's ConfirmFn: ask aloud, wait for a yes or no. Timeout is no."""
        if self._confirm is not None and not self._confirm.done():
            return False  # one question at a time; a second one is refused
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._confirm = fut
        turn = self._turn
        before = self.state
        await self._set_state(State.CONFIRMING, turn)
        await self._emit_turn(turn, {"event": "confirm_request", "question": question})
        if turn is not None and turn.speaker is not None:
            turn.speaker.say(question)
        else:
            self._narrate(question)
        try:
            return await asyncio.wait_for(asyncio.shield(fut), self.confirm_timeout)
        except TimeoutError:
            return False
        finally:
            if self._confirm is fut:
                self._confirm = None
            if self.state is State.CONFIRMING:
                back = before if before is not State.CONFIRMING else State.IDLE
                if self._turn is None:
                    back = State.IDLE
                await self._set_state(back, self._turn)

    def answer(self, yes: bool) -> bool:
        """Answer the pending confirmation. Returns False when nothing was pending."""
        return self._resolve_confirm(yes)

    def _resolve_confirm(self, yes: bool) -> bool:
        fut = self._confirm
        if fut is None or fut.done():
            return False
        fut.set_result(yes)
        return True

    async def _answer_by_voice(
        self, text: str | None, pcm: np.ndarray | None, t0: float, sink: EmitFn | None
    ) -> TurnResult:
        timing: dict[str, Any] = {"stt_ms": None, "total_ms": None, "path": "confirm"}
        user_text = (text or "").strip()
        if pcm is not None:
            user_text = (await asyncio.to_thread(self.stt.transcribe, pcm)).strip()
            timing["stt_ms"] = round((perf_counter() - t0) * 1000, 1)
        tmp = _Turn(sink=sink)
        if user_text:
            await self._emit_turn(tmp, transcript("user", user_text))
        self.answer(parse_yes_no(user_text))
        timing["total_ms"] = round((perf_counter() - t0) * 1000, 1)
        return TurnResult(reply="", user_text=user_text, timing=timing)

    # --- background worker narration ---------------------------------------------

    async def _consume_worker(self) -> None:
        assert self.worker is not None
        try:
            async for ev in self.worker.events():
                if isinstance(ev, WorkerProgress):
                    await self._emit({"event": "progress", "text": ev.text, "job_id": ev.job_id})
                    if self._can_narrate() and monotonic() - self._last_progress >= self.progress_interval:
                        self._last_progress = monotonic()
                        self._narrate(ev.text)
                elif isinstance(ev, WorkerDone):
                    await self._emit(
                        {"event": "progress", "text": ev.summary, "job_id": ev.job_id, "done": True, "ok": ev.ok}
                    )
                    self._pending_done.append(ev.summary)
                    self.history = (self.history + [{"role": "assistant", "content": ev.summary}])[-HISTORY_LIMIT:]
                    self.transcript.append({"role": "assistant", "content": ev.summary})
                    self._after_idle()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker event stream failed")

    def _can_narrate(self) -> bool:
        return self.state is State.IDLE and self._turn is None and (self._narration is None or self._narration.done())

    def _after_idle(self) -> None:
        """Speak finished-job summaries as soon as nothing else is going on."""
        if self._pending_done and self._can_narrate():
            text = " ".join(self._pending_done)
            self._pending_done.clear()
            self._narrate(text)

    def _narrate(self, text: str) -> None:
        self._narration = asyncio.create_task(self._narrate_run(text))

    async def _narrate_run(self, text: str) -> None:
        speaker = _Speaker(self.tts, self.out, quiet=False)
        if self.state is State.IDLE:
            await self._set_state(State.SPEAKING)
        speaker.say(clean_for_speech(text))
        try:
            await speaker.finish()
        except asyncio.CancelledError:
            await speaker.cancel()
            raise
        except Exception:
            log.exception("narration failed")
        finally:
            if self._turn is None and self.state is State.SPEAKING:
                await self._set_state(State.IDLE)
        if self._narration is asyncio.current_task():
            self._narration = None
        self._after_idle()

    # --- idle session end ----------------------------------------------------------

    async def _idle_watch(self) -> None:
        step = max(0.01, min(5.0, self.idle_timeout / 4))
        while True:
            await asyncio.sleep(step)
            if (
                self.transcript
                and self._turn is None
                and self.state is State.IDLE
                and monotonic() - self._last_activity >= self.idle_timeout
            ):
                self.end_session()

    def end_session(self) -> None:
        """Close this conversation and start a fresh one with no history."""
        sid, lines = self.session_id, self.transcript
        self.session_id = _new_session_id()
        self.history = []
        self.transcript = []
        if self.on_session_end is not None and lines:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in lines)
            self._spawn(_quietly(self.on_session_end(sid, text)))

    # --- events ------------------------------------------------------------------

    async def _set_state(self, state: State, turn: _Turn | None = None) -> None:
        if turn is not None and self._turn is not turn and state is not State.IDLE:
            return  # a cancelled turn must not move the state of the next one
        if state is self.state:
            return
        self.state = state
        await self._emit_turn(turn, {"event": "state", "state": state.value})

    async def _emit_turn(self, turn: _Turn | None, event: dict[str, Any]) -> None:
        await self._emit(event)
        if turn is not None and turn.sink is not None:
            try:
                await turn.sink(event)
            except Exception:
                log.exception("turn sink failed")

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.emit is not None:
            try:
                await self.emit(event)
            except Exception:
                log.exception("emit failed")

    def _spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)


def transcript(role: str, text: str) -> dict[str, Any]:
    return {"event": "transcript", "role": role, "text": text, "final": True}


def _new_session_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


async def _quietly(aw: Awaitable[Any]) -> None:
    try:
        await aw
    except Exception:
        log.exception("background task failed")
