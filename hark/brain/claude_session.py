"""One long-lived Claude Code CLI process in headless stream-json mode.

Keeping the process alive is the main speed win: a fresh `claude -p` pays about
0.7 s of startup plus a cold prompt cache on every turn. Permission prompts go to
the hark MCP server's approve tool; permissions are never bypassed.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import importlib.util
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from hark.brain.local import Cancelled, race

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERRUPT_GRACE_S = 2.0  # after this the process is killed and restarted warm
EOF = {"type": "_eof"}


def hark_mcp_available() -> bool:
    try:
        return importlib.util.find_spec("hark.tools.mcp_server") is not None
    except ImportError:
        return False


def mcp_config() -> str:
    return json.dumps({"mcpServers": {"hark": {
        "command": "uv",
        "args": ["run", "--project", str(PROJECT_ROOT), "python", "-m", "hark.tools.mcp_server"],
    }}})


class ClaudeSession:
    def __init__(self, model: str, system_prompt: str, *, tools: str = "",
                 setting_sources: str = "", use_mcp: bool | None = None,
                 cwd: str | None = None, claude_cmd: list[str] | None = None,
                 effort: str = "", chrome: bool = False) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.effort = effort  # "" leaves the CLI's default effort
        self.chrome = chrome  # enable Claude in Chrome browser control
        self.tools = tools  # "" means no built-in tools, only hark MCP tools
        self.setting_sources = setting_sources  # "" skips user hooks, settings and CLAUDE.md
        self.use_mcp = hark_mcp_available() if use_mcp is None else use_mcp
        self.cwd = cwd
        self.claude_cmd = claude_cmd or ["claude"]
        self.turns = 0  # turns run on the current process (its conversation length)
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr: collections.deque[str] = collections.deque(maxlen=20)
        self._pending: dict[str, asyncio.Future] = {}
        self._req = 0
        self._turn_q: asyncio.Queue | None = None
        self._turn_done: asyncio.Future | None = None
        self._cancel = asyncio.Event()
        self._starting: asyncio.Task | None = None
        self._started_at = 0.0
        self._closing = False
        self.min_life_s = 5.0  # a process that dies younger than this is not auto-restarted

    def command(self) -> list[str]:
        cmd = [
            *self.claude_cmd, "-p",
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--include-partial-messages", "--verbose",
            "--model", self.model,
            "--append-system-prompt", self.system_prompt,
            "--tools", self.tools,
            f"--setting-sources={self.setting_sources}",
            "--permission-mode", "default",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--strict-mcp-config",
        ]
        if self.chrome:
            cmd.append("--chrome")
        if self.effort:
            cmd += ["--effort", self.effort]
        if self.use_mcp:
            cmd += ["--mcp-config", mcp_config(), "--permission-prompt-tool", "mcp__hark__approve"]
        else:
            cmd += ["--permission-prompts", "none"]  # nobody to ask, so anything risky is denied
        return cmd

    @property
    def alive(self) -> bool:
        """A process that can take a turn. Its reader ending means stdout hit EOF: the
        process is gone even when asyncio has not reaped it yet."""
        return (self._proc is not None and self._proc.returncode is None
                and self._reader is not None and not self._reader.done())

    # --- process lifecycle ---

    async def start(self) -> None:
        """Spawn the CLI and wait until it answers the initialize handshake."""
        if self._starting and not self._starting.done():
            return await asyncio.shield(self._starting)
        self._starting = asyncio.ensure_future(self._spawn())
        await asyncio.shield(self._starting)

    async def _spawn(self) -> None:
        await self._kill()
        self._closing = False
        self.turns = 0
        self._proc = await asyncio.create_subprocess_exec(
            *self.command(),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=self.cwd, limit=1 << 24,
        )
        self._started_at = time.monotonic()
        self._reader = asyncio.create_task(self._read(self._proc))
        self._stderr_task = asyncio.create_task(self._read_stderr(self._proc))
        await asyncio.wait_for(self._control({"subtype": "initialize", "hooks": None}), 60.0)

    async def restart(self, system_prompt: str | None = None) -> None:
        changed = system_prompt is not None and system_prompt != self.system_prompt
        if self._starting and not self._starting.done():
            await asyncio.shield(self._starting)  # a restart is already on its way
            if not changed:
                return
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self._starting = None
        await self.start()

    async def close(self) -> None:
        self._closing = True
        # A start still on its way would spawn a process after the kill below.
        if self._starting is not None and not self._starting.done():
            await asyncio.gather(asyncio.shield(self._starting), return_exceptions=True)
            self._closing = True  # _spawn resets it
        await self._kill()

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
        if proc is not None:
            with contextlib.suppress(Exception):
                proc._transport.close()  # type: ignore[attr-defined]  # free pipes now, not at GC
        tasks = [t for t in (self._reader, self._stderr_task) if t and not t.done()]
        for task in tasks:
            if task is not asyncio.current_task():
                task.cancel()
        self._fail_turn()

    async def _ensure_ready(self) -> None:
        """Wait out an interrupted turn, then make sure a process is up."""
        if self._turn_done is not None and not self._turn_done.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._turn_done), INTERRUPT_GRACE_S)
            except asyncio.TimeoutError:
                await self.restart()
        if self._starting and not self._starting.done():
            await asyncio.shield(self._starting)
        if not self.alive:
            await self.restart()

    # --- I/O ---

    async def _write(self, obj: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self._proc.stdin.drain()

    async def _control(self, request: dict[str, Any]) -> dict[str, Any]:
        self._req += 1
        rid = f"hark_{self._req}"
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._write({"type": "control_request", "request_id": rid, "request": request})
            return await fut
        finally:
            self._pending.pop(rid, None)

    async def _read(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        try:
            while line := await proc.stdout.readline():
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                await self._dispatch(msg)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("claude exited"))
            if proc is self._proc or self._proc is None:
                self._fail_turn()
                lived = time.monotonic() - self._started_at
                # Restart warm in the background, unless it is crash-looping.
                if not self._closing and proc is self._proc and lived > self.min_life_s:
                    self._starting = asyncio.ensure_future(self._spawn())

    async def _read_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while line := await proc.stderr.readline():
            self._stderr.append(line.decode(errors="replace").rstrip())

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "control_response":
            resp = msg.get("response") or {}
            fut = self._pending.get(resp.get("request_id", ""))
            if fut and not fut.done():
                if resp.get("subtype") == "error":
                    fut.set_exception(RuntimeError(resp.get("error", "control request failed")))
                else:
                    fut.set_result(resp.get("response") or {})
            return
        if kind == "control_request":
            # We register no SDK-side hooks or permission callback; refuse anything asked.
            await self._write({"type": "control_response", "response": {
                "subtype": "error", "request_id": msg.get("request_id"),
                "error": "not supported by hark"}})
            return
        if self._turn_q is not None:
            self._turn_q.put_nowait(msg)
        if kind == "result" and self._turn_done is not None and not self._turn_done.done():
            self._turn_done.set_result(msg)

    def _fail_turn(self) -> None:
        if self._turn_q is not None:
            self._turn_q.put_nowait(EOF)
        if self._turn_done is not None and not self._turn_done.done():
            self._turn_done.set_result(EOF)

    # --- turns ---

    async def send(self, content: str) -> AsyncIterator[dict[str, Any]]:
        """Run one turn. Yields raw stream-json messages, ending with the result.
        Ends early, with no result, once interrupt() is called. A process that died
        since the last turn is replaced once before the turn is given up."""
        q: asyncio.Queue = asyncio.Queue()
        for attempt in (1, 2):
            await self._ensure_ready()
            self._turn_q, self._turn_done = q, asyncio.get_running_loop().create_future()
            cancel = self._cancel = asyncio.Event()
            try:
                await self._write({"type": "user", "session_id": "", "parent_tool_use_id": None,
                                   "message": {"role": "user", "content": content}})
            except (ConnectionError, AssertionError):
                if attempt == 2:
                    q.put_nowait(EOF)
                    break
                await self.restart()
                continue
            self.turns += 1
            if not self.alive:
                q.put_nowait(EOF)  # it died between the check and the write: nobody else will say so
            break
        while True:
            try:
                msg = await race(q.get(), cancel)
            except Cancelled:
                return
            if msg is EOF:
                tail = " ".join(self._stderr)[-300:]
                yield {"type": "result", "subtype": "process_exited", "is_error": True,
                       "result": f"claude exited. {tail}".strip()}
                return
            yield msg
            if msg.get("type") == "result":
                return

    @property
    def interrupted(self) -> bool:
        return self._cancel.is_set()

    async def interrupt(self) -> None:
        """Stop the running turn. Returns at once; the CLI's own interrupt usually
        lands in about 20 ms, and a process that ignores it is replaced."""
        if self._turn_done is None or self._turn_done.done():
            return
        self._cancel.set()
        try:
            await self._write({"type": "control_request", "request_id": "hark_interrupt",
                               "request": {"subtype": "interrupt"}})
        except (ConnectionError, AssertionError):
            pass
        asyncio.ensure_future(self._watch_interrupt(self._turn_done))

    async def _watch_interrupt(self, turn_done: asyncio.Future) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(turn_done), INTERRUPT_GRACE_S)
        except asyncio.TimeoutError:
            if turn_done is self._turn_done and not self._closing:
                await self.restart()
