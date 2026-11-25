"""Memory contract.

Markdown files under MemoryConfig.dir are the truth (the user can read and edit them
in Obsidian, and they sync between devices). The SQLite index is a cache that can be
rebuilt from the files at any time.

Layout of the memory dir:
    profile.md             who the user is, always in the prompt (about 500 tokens max)
    carryover.md           rolling summary of the last few sessions (about 300 tokens max)
    facts/<slug>.md        one fact per file, frontmatter: name, description, type, updated
    sessions/<ts>-<host>.md  transcript of one session, with a short digest once it ends
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Hit:
    kind: str  # "fact" | "session"
    name: str
    text: str  # short snippet, ready to drop into a prompt
    path: str


class Memory(Protocol):
    def context(self) -> str:
        """Profile plus carry-over, read from RAM. Zero I/O. Stable between writes."""
        ...

    def recall(self, query: str, k: int = 4) -> list[Hit]:
        """Fast keyword search over facts and past sessions. Must take under 10 ms."""
        ...

    def save_fact(self, name: str, body: str, description: str = "", type: str = "user") -> str:
        """Write or update facts/<slug>.md. Returns the path."""
        ...

    def forget(self, query: str) -> list[str]:
        """Delete the facts that best match. Returns the names removed."""
        ...

    def log_turn(self, session_id: str, role: str, text: str) -> None:
        """Append one line to the session transcript. Cheap; called after each turn."""
        ...

    def end_session(self, session_id: str, digest: str) -> None:
        """Store the digest for a finished session and roll it into carryover.md."""
        ...
