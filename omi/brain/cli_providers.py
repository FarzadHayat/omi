"""Model-agnostic subscription CLI providers for Omi's talker and background worker.

Supports the major subscription-backed agent CLIs out of the box without requiring API
keys, while giving each one access to Omi's device/GUI/terminal/T3 tools through Omi's
local MCP server (`python -m omi.tools.mcp_server`):

  - opencode : OpenCode CLI (free models, GitHub Copilot, Claude/OpenAI/Google subs)
  - codex    : OpenAI Codex CLI (ChatGPT Plus/Pro subscription)
  - agy      : Antigravity CLI (Google AI Pro/Ultra subscription)
  - gemini   : Google Gemini CLI (Google account / AI Pro subscription)

Claude Code (`claude`) and OpenAI-compatible/local HTTP (`local`/`openai`) have dedicated
optimized implementations in `claude.py` and `local.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omi.brain.base import VOICE_STYLE
from omi.brain.claude import replay, tool_name
from omi.brain.local import SpeechFilter, new_recall, with_recalled
from omi.config import HOME, STATE_DIR, CliProviderConfig, Config
from omi.events import (
    BrainEvent,
    TextDelta,
    ToolEnd,
    ToolStart,
    TurnContext,
    TurnDone,
    WorkerDone,
    WorkerEvent,
    WorkerProgress,
)
from omi.worker import PROGRESS_EVERY_S, WORKER_PROMPT, progress_phrase

CLI_PROVIDERS = ("opencode", "codex", "agy", "gemini")
AUTO_ORDER = ("claude", "opencode", "codex", "agy", "gemini")

PROVIDER_INFO: dict[str, tuple[str, str]] = {
    "claude": ("claude", "Claude Pro/Max subscription (Claude Code CLI)"),
    "opencode": ("opencode", "OpenCode Free / Copilot / multi-provider subs"),
    "codex": ("codex", "ChatGPT Plus/Pro subscription (Codex CLI)"),
    "agy": ("agy", "Google AI Pro/Ultra subscription (Antigravity CLI)"),
    "gemini": ("gemini", "Google account / AI Pro subscription (Gemini CLI)"),
    "local": ("llama-server", "Local llama-server or any OpenAI-compatible URL (Ollama, OpenRouter, xAI)"),
}


def available_providers(cfg: Config) -> list[dict[str, Any]]:
    """Report every supported provider, whether its CLI is installed, and its configured model."""
    rows: list[dict[str, Any]] = []
    for name, (binary, auth) in PROVIDER_INFO.items():
        if name == "local":
            installed = bool(cfg.brain.local.base_url) or shutil.which(binary) is not None or Path(cfg.brain.local.model).is_file()
            model = cfg.brain.local.model_name
        elif name == "claude":
            installed = shutil.which(binary) is not None
            model = cfg.brain.claude.model
        else:
            installed = shutil.which(binary) is not None
            pcfg: CliProviderConfig = getattr(cfg.brain, name)
            model = pcfg.model or "(cli default)"
        rows.append({
            "name": name,
            "binary": binary,
            "installed": installed,
            "model": model,
            "auth": auth,
            "active_talker": cfg.brain.talker == name,
            "active_worker": cfg.brain.worker == name,
        })
    return rows


def resolve_provider(name: str, *, role: str = "talker") -> str:
    """Resolve aliases (`openai` -> `local`) and `auto` to a concrete provider name."""
    low = name.strip().lower()
    if low == "openai":
        return "local"
    if low == "auto":
        for cand in AUTO_ORDER:
            if shutil.which(cand) is not None:
                return cand
        return "local" if role == "talker" else "off"
    return low


def _mcp_command() -> list[str]:
    return [sys.executable, "-m", "omi.tools.mcp_server"]


@dataclass
class ParsedEvent:
    session_id: str | None = None
    text: str = ""
    tool_start: tuple[str, str, dict[str, Any]] | None = None  # (id, name, args)
    tool_end: tuple[str, str, bool, str] | None = None  # (id, name, ok, text)
    error: str | None = None


def build_cli_invocation(
    provider: str,
    prompt: str,
    *,
    model: str = "",
    session_id: str = "",
    effort: str = "",
    use_mcp: bool = True,
    binary: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build exact (argv, extra_env) for one headless JSON-streaming CLI turn."""
    exe = binary or provider
    mcp_cmd = _mcp_command()
    env: dict[str, str] = {}

    if provider == "opencode":
        argv = [exe, "run", "--format", "json", "--archived"]
        if model:
            argv += ["--model", model]
        if session_id:
            argv += ["--session", session_id]
        argv.append(prompt)
        if use_mcp:
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps({
                "mcp": {"omi": {"type": "local", "command": mcp_cmd, "enabled": True}},
            })
        return argv, env

    if provider == "codex":
        argv = [exe, "exec"]
        if session_id:
            argv.append("resume")
        argv += ["--json", "--skip-git-repo-check"]
        if model:
            argv += ["-m", model]
        if use_mcp:
            argv += [
                "-c", f"mcp_servers.omi.command={json.dumps(mcp_cmd[0])}",
                "-c", f"mcp_servers.omi.args={json.dumps(mcp_cmd[1:])}",
            ]
        if session_id:
            argv.append(session_id)
        argv.append(prompt)
        return argv, env

    if provider == "agy":
        argv = [exe, "--output-format", "stream-json", "--disable-slash-commands"]
        if model:
            argv += ["--model", model]
        if effort:
            argv += ["--effort", effort]
        if session_id:
            argv += ["--conversation", session_id]
        if use_mcp:
            argv += ["--settings", json.dumps({
                "mcpServers": {"omi": {"command": mcp_cmd[0], "args": mcp_cmd[1:]}},
            })]
        argv.append(f"-p={prompt}")
        return argv, env

    if provider == "gemini":
        argv = [exe, "--output-format", "stream-json"]
        if model:
            argv += ["-m", model]
        if session_id:
            argv += ["--resume", session_id]
        argv += ["-p", prompt]
        return argv, env

    raise ValueError(f"unsupported CLI provider: {provider}")


def parse_cli_event(provider: str, msg: dict[str, Any]) -> ParsedEvent:
    """Normalize one JSONL event from opencode / codex / agy / gemini into a ParsedEvent."""
    ev = ParsedEvent()
    kind = str(msg.get("type") or "")

    # Common session/conversation id fields across providers
    for key in ("sessionID", "session_id", "thread_id", "conversation_id", "conversationId"):
        if isinstance(msg.get(key), str) and msg[key]:
            ev.session_id = msg[key]
            break

    if provider == "opencode":
        part = msg.get("part") if isinstance(msg.get("part"), dict) else {}
        if not ev.session_id and isinstance(part.get("sessionID"), str):
            ev.session_id = part["sessionID"]
        if kind == "text":
            ev.text = str(part.get("text") or msg.get("text") or "")
        elif kind == "tool_use":
            tname = tool_name(str(part.get("tool") or part.get("name") or ""))
            tid = str(part.get("callID") or part.get("id") or uuid.uuid4().hex[:8])
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            status = state.get("status", "")
            if status in ("completed", "error"):
                ev.tool_end = (tid, tname, status == "completed", str(state.get("output") or state.get("error") or "")[:300])
            else:
                inp = state.get("input") if isinstance(state.get("input"), dict) else {}
                ev.tool_start = (tid, tname, inp)
        elif kind == "error":
            err = msg.get("error")
            if isinstance(err, dict):
                ev.error = str(err.get("message") or err.get("name") or err)
            else:
                ev.error = str(err or msg.get("message") or "opencode error")
        return ev

    if provider == "codex":
        item = msg.get("item") if isinstance(msg.get("item"), dict) else {}
        itype = str(item.get("type") or "")
        if kind == "item.completed" and itype in ("agent_message", "message"):
            ev.text = str(item.get("text") or item.get("content") or "")
        elif kind == "item.started" and itype in ("mcp_tool_call", "command_execution", "function_call"):
            tname = tool_name(str(item.get("tool") or item.get("name") or itype))
            tid = str(item.get("id") or uuid.uuid4().hex[:8])
            args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            ev.tool_start = (tid, tname, args)
        elif kind == "item.completed" and itype in ("mcp_tool_call", "command_execution", "function_call"):
            tname = tool_name(str(item.get("tool") or item.get("name") or itype))
            tid = str(item.get("id") or uuid.uuid4().hex[:8])
            ok = item.get("status") != "failed" and not item.get("error")
            out = str(item.get("output") or item.get("result") or item.get("error") or "")[:300]
            ev.tool_end = (tid, tname, ok, out)
        elif kind == "turn.failed":
            err = msg.get("error")
            ev.error = str(err.get("message") if isinstance(err, dict) else (err or "codex turn failed"))
        elif kind == "error":
            # Codex emits non-fatal reconnect notices; only treat final error if turn fails,
            # unless no reconnect wording is present.
            text = str(msg.get("message") or "")
            if text and not text.startswith("Reconnecting"):
                ev.error = text
        return ev

    # agy and gemini share stream-json conventions (text deltas, assistant blocks, tool calls, result)
    if kind == "stream_event":
        inner = msg.get("event") if isinstance(msg.get("event"), dict) else {}
        delta = inner.get("delta") if isinstance(inner.get("delta"), dict) else {}
        if inner.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
            ev.text = str(delta.get("text") or "")
        return ev
    if kind == "message" and msg.get("role") == "assistant":
        ev.text = str(msg.get("content") or msg.get("text") or "")
        return ev
    if kind == "text":
        ev.text = str(msg.get("text") or msg.get("content") or "")
        return ev
    if kind == "assistant":
        blocks = (msg.get("message") or {}).get("content") if isinstance(msg.get("message"), dict) else msg.get("content")
        if isinstance(blocks, str):
            ev.text = blocks
        elif isinstance(blocks, list):
            texts = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(str(b.get("text") or ""))
                elif b.get("type") == "tool_use":
                    tid = str(b.get("id") or uuid.uuid4().hex[:8])
                    tname = tool_name(str(b.get("name") or ""))
                    ev.tool_start = (tid, tname, b.get("input") if isinstance(b.get("input"), dict) else {})
            if texts:
                ev.text = "".join(texts)
        return ev
    if kind == "tool_use":
        tid = str(msg.get("tool_id") or msg.get("id") or uuid.uuid4().hex[:8])
        tname = tool_name(str(msg.get("tool_name") or msg.get("name") or ""))
        inp = msg.get("parameters") or msg.get("input") or {}
        ev.tool_start = (tid, tname, inp if isinstance(inp, dict) else {})
        return ev
    if kind == "tool_result":
        tid = str(msg.get("tool_id") or msg.get("tool_use_id") or "")
        tname = tool_name(str(msg.get("tool_name") or msg.get("name") or ""))
        ok = msg.get("status") != "error" and not msg.get("is_error")
        ev.tool_end = (tid, tname, ok, str(msg.get("output") or msg.get("content") or "")[:300])
        return ev
    if kind == "result":
        if msg.get("is_error") or msg.get("status") == "error":
            ev.error = str(msg.get("result") or msg.get("error") or f"{provider} error")
        elif isinstance(msg.get("result"), str) and msg["result"]:
            ev.text = msg["result"]
        return ev
    if kind == "error":
        err = msg.get("error") or msg.get("message")
        ev.error = str(err.get("message") if isinstance(err, dict) else (err or f"{provider} error"))
    return ev


class CliBrain:
    """Conversational Brain backed by any subscription CLI (opencode, codex, agy, gemini)."""

    def __init__(
        self,
        provider: str,
        cfg: Config,
        *,
        use_mcp: bool = True,
        binary: str | None = None,
    ) -> None:
        if provider not in CLI_PROVIDERS:
            raise ValueError(f"unsupported CLI provider: {provider}")
        self.name = provider
        self.provider = provider
        self.cfg = cfg
        self.pcfg: CliProviderConfig = getattr(cfg.brain, provider)
        self.use_mcp = use_mcp
        self.binary = binary
        self._style = VOICE_STYLE.format(name=cfg.name)
        self._memory = ""
        self._session_id: str | None = None
        self._cli_session = ""
        self._proc: asyncio.subprocess.Process | None = None
        self._cancelled = False
        self.error: str | None = None
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    async def start(self, memory_context: str = "") -> None:
        self._memory = memory_context
        exe = self.binary or self.provider
        if shutil.which(exe) is None and not Path(exe).is_file():
            self.error = f"{self.provider} CLI is not installed"
        else:
            self.error = None

    async def cancel(self) -> None:
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    async def close(self) -> None:
        await self.cancel()

    def _compose_prompt(self, user_text: str, ctx: TurnContext) -> str:
        content = with_recalled(user_text, new_recall(ctx.recalled, ctx.history, user_text))
        if self._cli_session:
            return content
        parts = [self._style]
        if self._memory:
            parts.append(self._memory)
        if ctx.history:
            parts.append(replay(ctx.history).strip())
        parts.append(f"User: {content}")
        return "\n\n".join(parts)

    async def turn(self, user_text: str, ctx: TurnContext) -> AsyncIterator[BrainEvent]:
        self._cancelled = False
        if ctx.memory_context != self._memory or (self._session_id not in (None, ctx.session_id) and self._cli_session):
            self._memory = ctx.memory_context
            self._cli_session = ""
        self._session_id = ctx.session_id

        if self.error:
            yield TurnDone(error=self.error)
            return

        prompt = self._compose_prompt(user_text, ctx)
        argv, extra_env = build_cli_invocation(
            self.provider,
            prompt,
            model=self.pcfg.model,
            session_id=self._cli_session,
            effort=self.pcfg.effort,
            use_mcp=self.use_mcp,
            binary=self.binary,
        )
        env = {**os.environ, **extra_env} if extra_env else None
        done = TurnDone()
        speech = SpeechFilter()
        emitted_text = False
        last_error: str | None = None
        stderr_lines: list[str] = []

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(STATE_DIR),
                env=env,
                limit=1 << 24,
            )
            self._proc = proc

            async def drain_stderr() -> None:
                assert proc.stderr is not None
                while line := await proc.stderr.readline():
                    stderr_lines.append(line.decode(errors="replace").strip())

            err_task = asyncio.create_task(drain_stderr())
            assert proc.stdout is not None
            while raw := await proc.stdout.readline():
                line = raw.decode(errors="replace").strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                ev = parse_cli_event(self.provider, msg)
                if ev.session_id:
                    self._cli_session = ev.session_id
                if ev.tool_start:
                    yield ToolStart(*ev.tool_start)
                if ev.tool_end:
                    yield ToolEnd(*ev.tool_end)
                if ev.text:
                    # Avoid duplicating final `result` text if deltas already streamed it.
                    if msg.get("type") == "result" and emitted_text:
                        continue
                    cleaned = speech.feed(ev.text)
                    if cleaned:
                        emitted_text = True
                        yield TextDelta(cleaned)
                if ev.error:
                    last_error = ev.error
            tail = speech.flush()
            if tail:
                emitted_text = True
                yield TextDelta(tail)
            await proc.wait()
            await err_task
            if self._cancelled:
                done = TurnDone(cancelled=True)
            elif last_error:
                done = TurnDone(error=last_error)
            elif proc.returncode != 0 and not emitted_text:
                detail = " ".join(stderr_lines[-5:])[-260:]
                done = TurnDone(error=f"{self.provider} exited ({proc.returncode}){': ' + detail if detail else ''}")
        except FileNotFoundError:
            self.error = f"{self.provider} CLI is not installed"
            done = TurnDone(error=self.error)
        except Exception as exc:
            done = TurnDone(cancelled=self._cancelled, error=None if self._cancelled else str(exc))
        finally:
            self._proc = None
        yield done


class CliWorker:
    """Background worker backed by any subscription CLI (opencode, codex, agy, gemini)."""

    def __init__(
        self,
        provider: str,
        cfg: Config,
        *,
        use_mcp: bool = True,
        binary: str | None = None,
    ) -> None:
        if provider not in CLI_PROVIDERS:
            raise ValueError(f"unsupported CLI worker provider: {provider}")
        self.provider = provider
        self.cfg = cfg
        self.pcfg: CliProviderConfig = getattr(cfg.brain, provider)
        self.use_mcp = use_mcp
        self.binary = binary
        self.progress_every_s = PROGRESS_EVERY_S
        self._jobs: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._events: asyncio.Queue[WorkerEvent] = asyncio.Queue()
        self._running: str | None = None
        self._dropped: set[str] = set()
        self._runner: asyncio.Task[None] | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._interrupted = False
        self._cli_session = ""

    async def start(self) -> None:
        exe = self.binary or self.provider
        if shutil.which(exe) is None and not Path(exe).is_file():
            raise RuntimeError(f"{self.provider} CLI is not installed")
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
            self._interrupted = True
            proc = self._proc
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()

    async def close(self) -> None:
        await self.cancel()
        if self._runner:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner

    async def _run(self) -> None:
        while True:
            job_id, task = await self._jobs.get()
            if job_id in self._dropped:
                self._dropped.discard(job_id)
                continue
            self._running = job_id
            try:
                self._events.put_nowait(await self._job(job_id, task))
            except Exception as exc:
                self._events.put_nowait(WorkerDone(job_id, False, f"The job failed: {exc}"))
            finally:
                self._running = None

    async def _job(self, job_id: str, task: str) -> WorkerDone:
        import time

        self._interrupted = False
        prompt = task if self._cli_session else f"{WORKER_PROMPT.format(name=self.cfg.name)}\n\nTask: {task}"
        model = self.pcfg.worker_model or self.pcfg.model
        argv, extra_env = build_cli_invocation(
            self.provider,
            prompt,
            model=model,
            session_id=self._cli_session,
            use_mcp=self.use_mcp,
            binary=self.binary,
        )
        env = {**os.environ, **extra_env} if extra_env else None
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(HOME),
            env=env,
            limit=1 << 24,
        )
        self._proc = proc
        last = -self.progress_every_s
        texts: list[str] = []
        last_error: str | None = None
        stderr_lines: list[str] = []

        async def drain_stderr() -> None:
            assert proc.stderr is not None
            while raw_err := await proc.stderr.readline():
                stderr_lines.append(raw_err.decode(errors="replace").strip())

        err_task = asyncio.create_task(drain_stderr())
        try:
            assert proc.stdout is not None
            while raw := await proc.stdout.readline():
                line = raw.decode(errors="replace").strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(msg, dict):
                    continue
                ev = parse_cli_event(self.provider, msg)
                if ev.session_id:
                    self._cli_session = ev.session_id
                if ev.tool_start:
                    now = time.monotonic()
                    if now - last >= self.progress_every_s:
                        last = now
                        self._events.put_nowait(WorkerProgress(job_id, progress_phrase(ev.tool_start[1])))
                if ev.text:
                    texts.append(ev.text)
                if ev.error:
                    last_error = ev.error
            await proc.wait()
            await err_task
        finally:
            if not err_task.done():
                err_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await err_task
            self._proc = None

        if self._interrupted:
            return WorkerDone(job_id, False, "Stopped.")
        speech = SpeechFilter()
        summary = (speech.feed("".join(texts)) + speech.flush()).strip()
        if last_error or proc.returncode != 0:
            detail = " ".join(stderr_lines[-5:])[-260:]
            fallback_msg = f"{self.provider} exited ({proc.returncode}){': ' + detail if detail else ''}"
            return WorkerDone(job_id, False, summary or last_error or fallback_msg)
        return WorkerDone(job_id, True, summary or "Done.")
