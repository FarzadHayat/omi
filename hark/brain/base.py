"""Brain contract: the thing you talk to.

A brain takes the user's words and yields events: text to speak, and tool activity.
The brain runs its own tool calls (through the ToolRegistry it was built with, or
through the hark MCP server for the claude CLI) and reports them as ToolStart/ToolEnd.
Every turn ends with exactly one TurnDone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from hark.events import BrainEvent, TurnContext, WorkerEvent

# Every brain's system prompt starts from this. Keep it short: it is paid on every turn.
VOICE_STYLE = """You are {name}, a fast voice assistant and on-screen companion on the user's computer. \
You hear the user through speech recognition and your reply is spoken aloud.
- Reply in one or two short spoken sentences. No markdown, lists, code, emoji or URLs.
- Use the fastest tool that fits: direct desktop/window/audio/omarchy/terminal/t3 tools first, or `point_at` / `highlight_window` when showing something on screen.
- When asked to do something, call the tool right away, then confirm in a few words.
- For quick one-shot actions that need no follow-up, confirm briefly without adding a filler question. When chatting, troubleshooting, or when clarification helps, stay conversational or ask a short follow-up question.
- If speech recognition clearly garbled a word, go with the most likely meaning.
- For browser tasks, visual inspection, clicking inside apps, coding, research or multi-step work, call `delegate`.
- Never read secrets, keys or passwords aloud or put them in tool arguments."""


class Brain(Protocol):
    name: str

    async def start(self, memory_context: str = "") -> None:
        """Warm up (load model, open the session) so the first turn is fast.
        memory_context is the TurnContext.memory_context the first turn will carry, so
        the warm-up primes the exact prompt prefix and nothing restarts on turn one.
        A brain may expose `error` (str or None): why it cannot answer right now."""
        ...

    def turn(self, user_text: str, ctx: TurnContext) -> AsyncIterator[BrainEvent]:
        """Run one turn. Async generator of BrainEvent, ending with TurnDone."""
        ...

    async def cancel(self) -> None:
        """Abort the running turn quickly (under 100 ms). The generator then yields
        TurnDone(cancelled=True) and ends. Safe to call when idle."""
        ...

    async def close(self) -> None: ...


class Worker(Protocol):
    """Does long multi-step jobs in the background while the talker keeps chatting."""

    async def start(self) -> None: ...

    async def submit(self, task: str) -> str:
        """Queue a job. Returns its id at once; never waits for the job."""
        ...

    def events(self) -> AsyncIterator[WorkerEvent]:
        """Progress and results for all jobs, as they happen. One consumer."""
        ...

    async def cancel(self, job_id: str | None = None) -> None:
        """Cancel one job, or every job when job_id is None."""
        ...

    async def close(self) -> None: ...
