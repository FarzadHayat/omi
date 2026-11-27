"""Background worker: a second Claude Code process that does long multi-step jobs
while the talker keeps chatting. Narrates progress in a few spoken words."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator

from omi.brain.claude_session import ClaudeSession
from omi.brain.local import SpeechFilter
from omi.config import HOME, Config
from omi.brain.base import Worker
from omi.events import WorkerDone, WorkerEvent, WorkerProgress
from omi.tools.base import Tool, ToolResult

WORKER_PROMPT = """You are the background worker of {name}, a voice assistant and on-screen companion on the user's computer.
You get one task at a time and do the real multi-step work yourself: read and edit files,
run commands, control the browser (Claude in Chrome tools), control desktop apps, and answer visual questions about the screen.
Always pick the fastest reliable tool for the job:
1. Direct desktop/window/Omarchy/terminal/T3 Code tools (`launch_or_focus`, `focus_window`, `media`, `volume`, `omarchy`, `terminal_*`, `t3_*`) for OS, terminal, and T3 Code actions.
2. Claude in Chrome tools for anything inside the web browser (reading tabs, navigating, clicking, filling forms, searching).
3. `ui_inspect` + `ui_action` (AT-SPI) for clicking/typing inside desktop apps without screenshots.
4. `capture_screen` (returns the screenshot inline) + `mouse_click` / `mouse_scroll` / `type_text` / `press_keys` for visual or complex GUI tasks.
5. Call `point_at` or `highlight_window` whenever the user asks where something is or asks about something on screen, so your on-screen mascot flies over and points right at it.
Work carefully: check what you find before you report it, and verify the result of every change.
If the user says Omi is failing at a task, or it is clearly outside your tools or keeps failing, hand it off to T3 Code right away with `t3_send` (for a relevant open thread) or `t3_start` (for the relevant project, or omit `project` for scratch) instead of retrying or giving up, and tell the user you handed it to T3 Code.
When you count, list or search a project's files, skip dependency and build folders such as
.venv, venv, node_modules, .git, dist, build, target and __pycache__, unless the task is about them.
Prefer read-only commands; every destructive action has to be approved by the user aloud.
Never read, print or pass on secrets, keys, tokens or passwords.
Your final message is read aloud to the user: end with one or two short spoken sentences
saying what you did and the outcome. No markdown, lists, code, paths or URLs in that final message."""

PROGRESS_EVERY_S = 4.0

_PHRASES = {
    "Read": "Reading files", "Glob": "Looking for files", "Grep": "Searching files",
    "Bash": "Running a command", "Edit": "Editing files", "Write": "Writing a file",
    "NotebookEdit": "Editing a notebook", "WebSearch": "Searching the web",
    "WebFetch": "Reading a web page", "Task": "Starting a helper", "Agent": "Starting a helper",
    "TodoWrite": "Planning the steps",
}


def progress_phrase(tool: str) -> str:
    if tool.startswith("mcp__claude-in-chrome__"):
        return "Using the browser"
    if tool.startswith("mcp__omi__"):
        name = tool.split("__", 2)[-1]
        if name in ("capture_screen", "screen_info", "ui_inspect"):
            return "Looking at the screen"
        if name in ("point_at", "highlight_window"):
            return "Pointing it out"
        if name.startswith("terminal_"):
            return "Checking the terminal"
        if name.startswith("t3_"):
            return "Using T3 Code"
        return "Using the computer"
    if tool in ("capture_screen", "screen_info", "ui_inspect"):
        return "Looking at the screen"
    if tool in ("point_at", "highlight_window"):
        return "Pointing it out"
    if tool.startswith("terminal_"):
        return "Checking the terminal"
    if tool.startswith("t3_"):
        return "Using T3 Code"
    return _PHRASES.get(tool, "Working on it")


class ClaudeWorker:
    def __init__(self, cfg: Config, claude_cmd: list[str] | None = None,
                 use_mcp: bool | None = None) -> None:
        # User settings are loaded on purpose: the worker inherits the user's
        # permission rules and global instructions, like any Claude Code session.
        self.session = ClaudeSession(
            cfg.brain.claude.worker_model, WORKER_PROMPT.format(name=cfg.name),
            tools="default", setting_sources="user", use_mcp=use_mcp, cwd=str(HOME),
            claude_cmd=claude_cmd, chrome=cfg.brain.claude.chrome)
        self.progress_every_s = PROGRESS_EVERY_S
        self._jobs: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._events: asyncio.Queue[WorkerEvent] = asyncio.Queue()
        self._running: str | None = None
        self._dropped: set[str] = set()
        self._runner: asyncio.Task | None = None

    async def start(self) -> None:
        await self.session.start()
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
            await self.session.interrupt()

    async def close(self) -> None:
        if self._runner:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
        await self.session.close()

    async def _run(self) -> None:
        while True:
            job_id, task = await self._jobs.get()
            if job_id in self._dropped:
                self._dropped.discard(job_id)
                continue
            self._running = job_id
            try:
                self._events.put_nowait(await self._job(job_id, task))
            except Exception as exc:  # one failed job must not stop the queue
                self._events.put_nowait(WorkerDone(job_id, False, f"The job failed: {exc}"))
            finally:
                self._running = None

    async def _job(self, job_id: str, task: str) -> WorkerDone:
        last = -self.progress_every_s
        texts: list[str] = []
        result: dict | None = None
        async for msg in self.session.send(task):
            kind = msg.get("type")
            if kind == "assistant":
                for block in (msg.get("message") or {}).get("content") or []:
                    if block.get("type") == "tool_use":
                        now = time.monotonic()
                        if now - last >= self.progress_every_s:
                            last = now
                            self._events.put_nowait(
                                WorkerProgress(job_id, progress_phrase(block.get("name", ""))))
                    elif block.get("type") == "text" and not msg.get("parent_tool_use_id"):
                        texts.append(block.get("text", ""))
            elif kind == "result":
                result = msg
        if self.session.interrupted:
            return WorkerDone(job_id, False, "Stopped.")
        speech = SpeechFilter()
        summary = (result or {}).get("result") or (texts[-1] if texts else "")
        summary = (speech.feed(str(summary)) + speech.flush()).strip()
        ok = bool(result) and not result.get("is_error")
        if not ok:
            return WorkerDone(job_id, False, summary or "The job failed.")
        return WorkerDone(job_id, True, summary or "Done.")


class SwitchableWorker:
    """The session's one worker, which the settings can swap or turn off while Omi runs.

    The delegate tool and the session's narration loop hold this object, so a new
    worker takes over without either noticing. Jobs still running on the old worker
    are stopped and reported as such.
    """

    def __init__(self, worker: Worker | None = None) -> None:
        self.current = worker
        self._events: asyncio.Queue[WorkerEvent] = asyncio.Queue()
        self._open: set[str] = set()
        self._pump: asyncio.Task[None] | None = None

    @property
    def active(self) -> bool:
        return self.current is not None

    @property
    def active_entry(self) -> str:
        return getattr(self.current, "active", "") if self.current is not None else ""

    @property
    def fallback_reason(self) -> str:
        return getattr(self.current, "fallback_reason", "") if self.current is not None else ""

    async def start(self) -> None:
        if self.current is not None:
            await self.current.start()
            self._pump_from(self.current)

    async def submit(self, task: str) -> str:
        if self.current is None:
            raise RuntimeError("the background worker is off")
        job_id = await self.current.submit(task)
        self._open.add(job_id)
        return job_id

    async def events(self) -> AsyncIterator[WorkerEvent]:
        while True:
            yield await self._events.get()

    async def cancel(self, job_id: str | None = None) -> None:
        if self.current is not None:
            await self.current.cancel(job_id)

    async def swap(self, worker: Worker | None) -> None:
        """Hand over to `worker` (already started), or turn the worker off with None."""
        old, self.current = self.current, worker
        await self._stop_pump()
        if old is not None:
            with contextlib.suppress(Exception):
                await old.close()
        for job_id in sorted(self._open):
            self._events.put_nowait(WorkerDone(job_id, False, "Stopped, because the worker was changed."))
        self._open.clear()
        if worker is not None:
            self._pump_from(worker)

    async def close(self) -> None:
        await self._stop_pump()
        if self.current is not None:
            await self.current.close()

    def _pump_from(self, worker: Worker) -> None:
        async def pump() -> None:
            async for ev in worker.events():
                if isinstance(ev, WorkerDone):
                    self._open.discard(ev.job_id)
                self._events.put_nowait(ev)

        self._pump = asyncio.create_task(pump())

    async def _stop_pump(self) -> None:
        if self._pump is not None:
            self._pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump
            self._pump = None


def delegate_tool(worker: Worker) -> Tool:
    async def delegate(task: str) -> ToolResult:
        if not getattr(worker, "active", True):
            return ToolResult(False, "The background worker is off. It can be turned on in Omi's settings.")
        job_id = await worker.submit(task)
        return ToolResult(True, "Started. I will tell you when it is done.", {"job_id": job_id})

    return Tool(
        name="delegate",
        description="Hand a long or multi-step job (coding, research, files, the browser) "
                    "to the background worker. Describe the whole task in plain words.",
        parameters={"type": "object", "properties": {"task": {"type": "string"}},
                    "required": ["task"]},
        safety="reversible",
        fn=delegate,
    )
