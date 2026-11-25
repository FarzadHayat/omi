"""Claude talker: the Claude Code CLI as a conversational brain.

The CLI process keeps the conversation, so history is only replayed when a
process had to be restarted mid-session. Device tools arrive through the hark
MCP server, named mcp__hark__<tool>.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from hark.brain.base import VOICE_STYLE
from hark.brain.claude_session import ClaudeSession
from hark.brain.local import SpeechFilter, with_recalled
from hark.config import STATE_DIR, Config
from hark.events import BrainEvent, TextDelta, ToolEnd, ToolStart, TurnContext, TurnDone

MCP_PREFIX = "mcp__hark__"


def tool_name(name: str) -> str:
    return name.removeprefix(MCP_PREFIX)


def result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def replay(history: list[dict[str, str]]) -> str:
    lines = [f"{'User' if h['role'] == 'user' else 'You'}: {h['content']}" for h in history]
    return "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"


class ClaudeBrain:
    name = "claude"

    def __init__(self, cfg: Config, claude_cmd: list[str] | None = None,
                 use_mcp: bool | None = None) -> None:
        self._style = VOICE_STYLE.format(name=cfg.name)
        self._memory = ""
        self._session_id: str | None = None
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.session = ClaudeSession(
            cfg.brain.claude.model, self._style, tools="", setting_sources="",
            use_mcp=use_mcp, cwd=str(STATE_DIR), claude_cmd=claude_cmd)

    def _prompt(self) -> str:
        return f"{self._style}\n\n{self._memory}" if self._memory else self._style

    async def start(self) -> None:
        await self.session.start()

    async def cancel(self) -> None:
        await self.session.interrupt()

    async def close(self) -> None:
        await self.session.close()

    async def turn(self, user_text: str, ctx: TurnContext) -> AsyncIterator[BrainEvent]:
        done = TurnDone()
        try:
            # A new memory context or a new hark session needs a fresh conversation.
            if ctx.memory_context != self._memory or (
                self._session_id not in (None, ctx.session_id) and self.session.turns):
                self._memory = ctx.memory_context
                await self.session.restart(self._prompt())
            self._session_id = ctx.session_id
            content = with_recalled(user_text, ctx.recalled)
            if ctx.history and self.session.turns == 0:
                content = replay(ctx.history) + content
            speech = SpeechFilter()
            names: dict[str, str] = {}
            async for msg in self.session.send(content):
                kind = msg.get("type")
                if kind == "stream_event":
                    ev = msg.get("event") or {}
                    delta = ev.get("delta") or {}
                    if ev.get("type") == "content_block_delta" and delta.get("type") == "text_delta":
                        text = speech.feed(delta.get("text", ""))
                        if text:
                            yield TextDelta(text)
                    elif ev.get("type") == "message_stop":
                        tail = speech.flush()
                        if tail:
                            yield TextDelta(tail)
                elif kind == "assistant" and not msg.get("parent_tool_use_id"):
                    for block in (msg.get("message") or {}).get("content") or []:
                        if block.get("type") == "tool_use" and block.get("id") not in names:
                            names[block["id"]] = tool_name(block.get("name", ""))
                            yield ToolStart(block["id"], names[block["id"]], block.get("input") or {})
                elif kind == "user" and not msg.get("parent_tool_use_id"):
                    content_blocks = (msg.get("message") or {}).get("content")
                    for block in content_blocks if isinstance(content_blocks, list) else []:
                        if block.get("type") == "tool_result":
                            tid = block.get("tool_use_id", "")
                            yield ToolEnd(tid, names.get(tid, ""), not block.get("is_error"),
                                          result_text(block.get("content"))[:300])
                elif kind == "result":
                    if msg.get("is_error") and not self.session.interrupted:
                        done = TurnDone(error=str(msg.get("result") or msg.get("subtype")))
            if self.session.interrupted:
                done = TurnDone(cancelled=True)
        except Exception as exc:  # a broken CLI ends the turn, never the daemon
            done = TurnDone(cancelled=self.session.interrupted, error=str(exc))
        yield done
