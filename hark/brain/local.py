"""Local talker: any OpenAI-compatible chat endpoint, llama-server by default.

Speed rules: the prompt prefix (system prompt, tools, history) stays byte-stable
between turns so llama.cpp reuses its prompt cache, and per-turn recall rides on
the newest user message instead of the system prompt.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import AsyncIterator, Awaitable
from typing import Any, TypeVar
from urllib.parse import urlparse

import httpx

from hark.brain.base import VOICE_STYLE
from hark.brain.llama_server import LlamaServer
from hark.config import LocalLlmConfig
from hark.events import BrainEvent, TextDelta, ToolEnd, ToolStart, TurnContext, TurnDone
from hark.tools.base import NOT_CONFIRMED, ToolRegistry

MAX_ROUNDS = 4
REFUSED_REPLY = "Okay, I left it alone."
T = TypeVar("T")

_THINK = ("<think>", "</think>")
_MARKDOWN = re.compile(r"[*#`~|]+|__")


class SpeechFilter:
    """Streams text through, dropping think blocks and markdown symbols. Tags may
    be split across chunks, so a possible partial tag is held back."""

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def feed(self, text: str) -> str:
        s, self._buf, out = self._buf + text, "", []
        while s:
            tag = _THINK[1] if self._in_think else _THINK[0]
            i = s.find(tag)
            if i >= 0:
                if not self._in_think:
                    out.append(s[:i])
                s = s[i + len(tag):]
                self._in_think = not self._in_think
                continue
            keep = next((n for n in range(len(tag) - 1, 0, -1) if s.endswith(tag[:n])), 0)
            if not self._in_think:
                out.append(s[: len(s) - keep])
            self._buf = s[len(s) - keep:] if keep else ""
            break
        return _MARKDOWN.sub("", "".join(out))

    def flush(self) -> str:
        rest, self._buf = ("" if self._in_think else self._buf), ""
        return _MARKDOWN.sub("", rest)


def with_recalled(user_text: str, recalled: list[str]) -> str:
    """Recalled memory goes on the user message so the cached prefix never changes."""
    if not recalled:
        return user_text
    return f"{user_text}\n\n(Notes from memory that may help: {' | '.join(recalled)})"


class Cancelled(Exception):
    pass


async def race(aw: Awaitable[T], cancel: asyncio.Event) -> T:
    """Await aw, but give up the moment cancel is set. This is what keeps cancel
    under 100 ms even while waiting on the network."""
    if cancel.is_set():
        if asyncio.iscoroutine(aw):
            aw.close()
        raise Cancelled
    task = asyncio.ensure_future(aw)
    waiter = asyncio.ensure_future(cancel.wait())
    done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        waiter.cancel()
        return task.result()
    task.cancel()
    raise Cancelled


class LocalBrain:
    name = "local"

    def __init__(self, cfg: LocalLlmConfig, registry: ToolRegistry, assistant_name: str = "Hark",
                 max_tokens: int = 400) -> None:
        self.cfg = cfg
        self.registry = registry
        self.max_tokens = max_tokens
        self._style = VOICE_STYLE.format(name=assistant_name)
        self._server = None if cfg.base_url else LlamaServer(cfg)
        self._client: httpx.AsyncClient | None = None
        self._cancel = asyncio.Event()

    # --- request building ---

    @property
    def _base(self) -> str:
        return (self.cfg.base_url or self._server.base_url).rstrip("/")  # type: ignore[union-attr]

    @property
    def _url(self) -> str:
        base = self._base
        return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"

    @property
    def _is_llama(self) -> bool:
        return self._server is not None or urlparse(self._base).hostname in ("127.0.0.1", "localhost")

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.cfg.api_key_env, "") if self.cfg.api_key_env else ""
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _system(self, memory_context: str) -> str:
        return f"{self._style}\n\n{memory_context}" if memory_context else self._style

    def _body(self, messages: list[dict[str, Any]], tools: bool, max_tokens: int, stream: bool) -> dict:
        body: dict[str, Any] = {
            "model": self.cfg.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        if tools and self.registry.tools:
            body["tools"] = self.registry.openai_tools()
        if self._is_llama:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    # --- lifecycle ---

    async def start(self) -> None:
        if self._server is not None:
            await self._server.start()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        # Warm-up: same system prompt and tools as a real turn, so turn one hits the cache.
        messages = [{"role": "system", "content": self._system("")}, {"role": "user", "content": "hi"}]
        try:
            await self._client.post(self._url, json=self._body(messages, True, 1, False),
                                    headers=self._headers())
        except httpx.HTTPError:
            pass  # a cold first turn is better than no assistant

    async def cancel(self) -> None:
        self._cancel.set()

    async def close(self) -> None:
        self._cancel.set()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._server is not None:
            await self._server.close()

    # --- turns ---

    async def turn(self, user_text: str, ctx: TurnContext) -> AsyncIterator[BrainEvent]:
        cancel = self._cancel = asyncio.Event()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system(ctx.memory_context)},
            *ctx.history,
            {"role": "user", "content": with_recalled(user_text, ctx.recalled)},
        ]
        done = TurnDone()
        try:
            if self._client is None:
                raise RuntimeError("local brain not started")
            for rnd in range(MAX_ROUNDS):
                calls: dict[int, dict[str, str]] = {}
                text_parts: list[str] = []
                async for delta in self._stream(messages, rnd < MAX_ROUNDS - 1, calls, cancel):
                    text_parts.append(delta)
                    yield TextDelta(delta)
                if not calls or rnd == MAX_ROUNDS - 1:
                    break  # the last round has no tools; ignore any call it makes up
                messages.append({
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                    "tool_calls": [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                        for c in calls.values()
                    ],
                })
                for c in calls.values():
                    try:
                        args = json.loads(c["args"] or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be an object")
                    except ValueError:
                        args = None
                    yield ToolStart(c["id"], c["name"], args or {})
                    if args is None:
                        ok, text = False, f"Bad arguments for {c['name']}."
                    else:
                        task = asyncio.ensure_future(self.registry.call(c["name"], args))
                        res = await race(asyncio.shield(task), cancel)  # the action itself finishes
                        ok, text = res.ok, res.text
                    yield ToolEnd(c["id"], c["name"], ok, text)
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                if any(m.get("content") == NOT_CONFIRMED for m in messages[-len(calls):]):
                    # The user said no. A small model tends to ask again, or to claim it
                    # did the thing anyway, so the answer is fixed.
                    yield TextDelta(REFUSED_REPLY)
                    break
        except Cancelled:
            done = TurnDone(cancelled=True)
        except Exception as exc:  # network or server trouble ends the turn, never the daemon
            done = TurnDone(cancelled=cancel.is_set(), error=None if cancel.is_set() else str(exc))
        yield done

    async def _stream(self, messages: list[dict[str, Any]], tools: bool,
                      calls: dict[int, dict[str, str]], cancel: asyncio.Event) -> AsyncIterator[str]:
        """One streamed completion. Yields speakable text, fills calls with tool calls."""
        assert self._client is not None
        req = self._client.build_request(
            "POST", self._url, json=self._body(messages, tools, self.max_tokens, True),
            headers=self._headers())
        resp = await race(self._client.send(req, stream=True), cancel)
        speech = SpeechFilter()
        try:
            if resp.status_code != 200:
                body = await race(resp.aread(), cancel)
                raise RuntimeError(f"chat endpoint returned {resp.status_code}: {body[:200]!r}")
            lines = resp.aiter_lines()
            while True:
                try:
                    line = await race(anext(lines), cancel)
                except StopAsyncIteration:
                    break
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("error"):
                    raise RuntimeError(str(chunk["error"])[:200])
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text = speech.feed(delta["content"])
                        if text:
                            yield text
                    for tc in delta.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "args": ""})
                        c["id"] = c["id"] or tc.get("id") or ""
                        fn = tc.get("function") or {}
                        c["name"] += fn.get("name") or ""
                        c["args"] += fn.get("arguments") or ""
            tail = speech.flush()
            if tail:
                yield tail
            for c in calls.values():
                c["id"] = c["id"] or f"call_{uuid.uuid4().hex[:8]}"
        finally:
            await resp.aclose()
