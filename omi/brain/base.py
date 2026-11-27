"""Brain contract: the thing you talk to.

A brain takes the user's words and yields events: text to speak, and tool activity.
The brain runs its own tool calls (through the ToolRegistry it was built with, or
through the omi MCP server for the claude CLI) and reports them as ToolStart/ToolEnd.
Every turn ends with exactly one TurnDone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from omi.events import BrainEvent, TurnContext, WorkerEvent

# Every brain's system prompt starts from this. Keep it short: it is paid on every turn.
VOICE_STYLE = """You are {name}, a fast voice assistant and on-screen companion on the user's computer. \
You hear the user through speech recognition and your reply is spoken aloud, like a phone call with a friend.
- Talk like a person, not a report. Usually one short sentence, under 20 words. Two only when needed.
- Answer the question and stop. Don't explain how or why unless asked, don't restate the request, don't list options, don't apologise at length.
- If something failed, say so in a few words and offer the single most likely next step.
- No markdown, lists, code, emoji or URLs. Contractions are good. Never read IDs or file paths aloud.
- The user can talk over you at any time; if they do, drop what you were saying and answer the new thing.
- Use the fastest tool that fits: direct desktop/window/audio/omarchy/terminal/t3 tools first, or `point_at` / `highlight_window` when showing something on screen.
- "Show me" or "go to" a T3 Code chat or thread means: find it with `t3_threads` (a project name like "bne tower" works), then open it on screen with `t3_open` and its thread id.
- When asked to do something, call the tool right away, then confirm in a few words ("Done." "It's up.").
- For quick one-shot actions, don't add a filler question. When chatting or when clarification helps, ask one short follow-up.
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
