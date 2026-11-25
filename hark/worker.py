"""Background worker: a second Claude Code process that does long multi-step jobs
while the talker keeps chatting. Narrates progress in a few spoken words."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator

from hark.brain.claude_session import ClaudeSession
from hark.brain.local import SpeechFilter
from hark.config import HOME, Config
from hark.events import WorkerDone, WorkerEvent, WorkerProgress
from hark.tools.base import Tool, ToolResult

WORKER_PROMPT = """You are the background worker of {name}, a voice assistant on the user's own computer.
You get one task at a time and do the real multi-step work yourself: read and edit files,
run commands, search the web, use the hark device tools.
Work carefully: check what you find before you report it, and verify the result of every change.
When you count, list or search a project's files, skip dependency and build folders such as
.venv, venv, node_modules, .git, dist, build, target and __pycache__, unless the task is about them.
Prefer read-only commands; every command that changes something has to be approved by the user aloud.
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
    if tool.startswith("mcp__hark__"):
        return "Using the computer"
    return _PHRASES.get(tool, "Working on it")


class ClaudeWorker:
    def __init__(self, cfg: Config, claude_cmd: list[str] | None = None,
                 use_mcp: bool | None = None) -> None:
        # User settings are loaded on purpose: the worker inherits the user's
        # permission rules and global instructions, like any Claude Code session.
        self.session = ClaudeSession(
            cfg.brain.claude.worker_model, WORKER_PROMPT.format(name=cfg.name),
            tools="default", setting_sources="user", use_mcp=use_mcp, cwd=str(HOME),
            claude_cmd=claude_cmd)
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


def delegate_tool(worker: ClaudeWorker) -> Tool:
    async def delegate(task: str) -> ToolResult:
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
