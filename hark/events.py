"""Shared event and state types. Every module speaks these; keep them small and stable."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ACTING = "acting"
    CONFIRMING = "confirming"


# --- Brain output events (yielded by Brain.turn) ---


@dataclass
class TextDelta:
    """A piece of text the assistant should speak. Plain spoken language, no markdown."""

    text: str


@dataclass
class ToolStart:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolEnd:
    id: str
    name: str
    ok: bool
    text: str  # short result, safe to show in the transcript


@dataclass
class TurnDone:
    """Last event of every turn."""

    cancelled: bool = False
    error: str | None = None


BrainEvent = TextDelta | ToolStart | ToolEnd | TurnDone


# --- Turn context handed to the brain ---


@dataclass
class TurnContext:
    session_id: str
    # Stable prefix: user profile plus carry-over from earlier sessions. Changes rarely,
    # so brains can keep it in a prompt cache.
    memory_context: str = ""
    # Per-turn recalled memory snippets (may be empty).
    recalled: list[str] = field(default_factory=list)
    # Earlier turns of this session: [{"role": "user"|"assistant", "content": str}]
    history: list[dict[str, str]] = field(default_factory=list)


# --- Background worker events (long jobs handed off with the `delegate` tool) ---


@dataclass
class WorkerProgress:
    job_id: str
    text: str  # a few spoken words, like "Reading the project files"


@dataclass
class WorkerDone:
    job_id: str
    ok: bool
    summary: str  # one or two spoken sentences with the outcome


WorkerEvent = WorkerProgress | WorkerDone
