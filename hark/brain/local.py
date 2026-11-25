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
# Every tool schema is prompt the small local model reads on a cache miss. Recall runs
# by itself every turn, so the talker does not get it; these get shorter descriptions.
LOCAL_SKIP = frozenset({"recall"})
LOCAL_DESCRIPTIONS = {
    "remember": "Save a lasting fact about the user.",
    "forget": "Forget a saved fact.",
    "delegate": "Start a long multi-step job on the computer (coding, editing files, web research). Never for a question you can answer yourself.",
}
# A device action whose result text already says what happened ("Opening Firefox.") is
# spoken as is. A second model round would only rephrase it, and costs about 1.4 s of
# silence here (prompt plus a dozen tokens at 12 a second). Measured: tool-turn first
# audio 3.6 s -> 2.3 s.
SPEAK_RESULT = frozenset({
    "launch_or_focus", "focus_window", "list_windows", "active_window", "switch_workspace",
    "move_window_to_workspace", "toggle_fullscreen", "media", "volume", "screenshot",
    "lock_screen", "current_time", "clipboard_get",
})
_MORE_STEPS = re.compile(r"\b(?:and|then|after|afterwards|also|plus)\b")
# The session keeps a sliding window of history, so once it is full its first message
# changes every turn and llama.cpp re-reads the whole history (150 to 180 tokens, about
# 750 ms). The talker cuts the history at a fixed message instead and only moves the cut
# when that message leaves the window, keeping this many messages.
HISTORY_TAIL = 4
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


_WORDS = re.compile(r"[a-z0-9']+")


def new_recall(recalled: list[str], history: list[dict[str, Any]], user_text: str) -> list[str]:
    """Drop recalled snippets the prompt already holds. Recall also finds the running
    conversation in the session log; repeating it cost the local talker 100 to 150
    prompt tokens a turn (300 to 500 ms) and told it nothing new."""
    seen = set(_WORDS.findall(" ".join([user_text, *(str(m.get("content") or "") for m in history)]).lower()))
    out = []
    for snippet in recalled:
        words = _WORDS.findall(snippet.lower())
        if not words or sum(w in seen for w in words) < 0.8 * len(words):
            out.append(snippet)
    return out


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
    try:
        done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:  # the caller itself was cancelled: take the work down with it
        task.cancel()
        waiter.cancel()
        raise
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
        self._memory = ""
        self._sent: tuple[str, list[dict[str, Any]]] | None = None  # (session id, history sent last turn)
        self.error: str | None = None  # why the last start failed; shown in status

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
            body["tools"] = self._tools()
        if self._is_llama:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    def _tools(self) -> list[dict[str, Any]]:
        tools = []
        for t in self.registry.openai_tools():
            fn = t["function"]
            if fn["name"] in LOCAL_SKIP:
                continue
            if fn["name"] in LOCAL_DESCRIPTIONS:
                fn = {**fn, "description": LOCAL_DESCRIPTIONS[fn["name"]]}
            if not fn["parameters"].get("required", True):
                # An empty list is 4 to 5 prompt tokens per tool, saying nothing.
                fn = {**fn, "parameters": {k: v for k, v in fn["parameters"].items() if k != "required"}}
            tools.append({**t, "function": fn})
        return tools

    def _history(self, ctx: TurnContext) -> list[dict[str, str]]:
        """A tail of ctx.history that keeps last turn's history as its prefix, so the
        cached prompt survives the session's sliding window (see HISTORY_TAIL). Matched
        as a whole block: single messages repeat ("what time is it")."""
        history = ctx.history
        prev = self._sent[1] if self._sent is not None and self._sent[0] == ctx.session_id else []
        if prev:
            n = len(prev)
            start = next((i for i in range(len(history) - n + 1) if history[i:i + n] == prev), None)
            # When last turn's start has left the window, move the cut forward.
            history = history[start:] if start is not None else history[-HISTORY_TAIL:]
        while history and history[0].get("role") != "user":
            history = history[1:]
        self._sent = (ctx.session_id, list(history))
        return history

    # --- lifecycle ---

    async def start(self, memory_context: str = "") -> None:
        self._memory = memory_context
        try:
            if self._server is not None:
                await self._server.start()
        except Exception as exc:
            self.error = str(exc)
            raise
        self.error = None
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        # Warm-up: same system prompt and tools as a real turn, so turn one hits the cache.
        messages = [{"role": "system", "content": self._system(memory_context)}, {"role": "user", "content": "hi"}]
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
            *self._history(ctx),
            {"role": "user", "content": with_recalled(user_text, new_recall(ctx.recalled, ctx.history, user_text))},
        ]
        done = TurnDone()
        try:
            if self._client is None or self.error:
                # The last start failed (model missing, server crashed): try again now,
                # so fixing the cause needs no daemon restart.
                try:
                    await race(self.start(ctx.memory_context), cancel)
                except Cancelled:
                    raise
                except Exception as exc:
                    raise RuntimeError(f"the local model is not running: {exc}") from exc
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
                results: list[tuple[str, bool, str]] = []
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
                    results.append((c["name"], ok, text))
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
                if any(text == NOT_CONFIRMED for _, _, text in results):
                    # The user said no. A small model tends to ask again, or to claim it
                    # did the thing anyway, so the answer is fixed.
                    yield TextDelta(REFUSED_REPLY)
                    break
                if all(ok and name in SPEAK_RESULT for name, ok, _ in results) \
                        and not _MORE_STEPS.search(user_text.lower()):
                    yield TextDelta((" " if text_parts else "") + " ".join(text for _, _, text in results))
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
