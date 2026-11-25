"""Memory: markdown files are the truth, an SQLite FTS5 index makes recall fast."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from hark.config import MemoryConfig
from hark.memory.base import Hit, Memory
from hark.memory.index import Index
from hark.memory.store import FACT_TYPES, Fact, Store

__all__ = ["Hit", "Memory", "FileMemory", "NullMemory", "create"]

log = logging.getLogger("hark.memory")


def _join_context(profile: str, carryover: str) -> str:
    parts = []
    if profile:
        parts.append(f"About the user:\n{profile}")
    if carryover:
        parts.append(f"Recent conversations:\n{carryover}")
    return "\n\n".join(parts)


class FileMemory:
    def __init__(self, cfg: MemoryConfig):
        self._store = Store(cfg.dir)
        self._index = Index(cfg.index, cfg.dir, cfg.shared_dir or None)
        self._lock = threading.RLock()
        self._sessions: dict[str, Path] = {}
        self._context = ""
        self.sync()

    # --- the Memory contract ---

    def context(self) -> str:
        return self._context  # RAM only; rebuilt after writes by _refresh_context

    def recall(self, query: str, k: int = 4) -> list[Hit]:
        return self._index.search(query, k)

    def save_fact(self, name: str, body: str, description: str = "", type: str = "user") -> str:
        if type not in FACT_TYPES:
            type = "user"
        with self._lock:
            path = self._store.write_fact(name, body, description, type)
            self._index.upsert_file(path)
        return str(path)

    def forget(self, query: str) -> list[str]:
        """Move the best-matching own fact into .trash/. Shared facts are never touched."""
        with self._lock:
            hits = self._index.search(query, k=1, own_facts_only=True)
            removed = []
            for hit in hits:
                path = Path(hit.path)
                try:
                    self._store.trash(path)
                except OSError:
                    continue
                self._index.remove_file(path)
                removed.append(hit.name)
            return removed

    def log_turn(self, session_id: str, role: str, text: str) -> None:
        with self._lock:
            path = self._sessions.get(session_id)
            if path is None:
                path = self._store.find_session(session_id)
                if path is not None and self._store.has_digest(path):
                    path = None  # that session already ended: a late turn starts a new file
                path = path or self._store.new_session_path(session_id)
                self._sessions[session_id] = path
            self._store.append_turn(path, role, text)

    def end_session(self, session_id: str, digest: str) -> None:
        with self._lock:
            path = (
                self._sessions.pop(session_id, None)
                or self._store.find_session(session_id)
                or self._store.new_session_path(session_id)
            )
            self._store.set_digest(path, digest)
            self._index.upsert_file(path)
            self._store.roll_carryover(path.stem, digest)
            self._refresh_context()

    # --- extras used by the after-session jobs ---

    def transcript(self, session_id: str) -> str:
        with self._lock:
            path = self._sessions.get(session_id) or self._store.find_session(session_id)
            return self._store.read_transcript(path) if path else ""

    def facts(self, types: tuple[str, ...] | None = None) -> list[Fact]:
        return self._index.facts(types)

    def write_profile(self, text: str) -> None:
        with self._lock:
            self._store.write_profile(text)
            self._refresh_context()

    def sync(self) -> int:
        """Pick up files edited by hand or by a sync tool since the last look."""
        with self._lock:
            changed = self._index.sync()
            self._refresh_context()
            return changed

    def _refresh_context(self) -> None:
        self._context = _join_context(self._store.profile(), self._store.carryover())


class NullMemory:
    """Memory switched off: remembers nothing, recalls nothing."""

    def context(self) -> str:
        return ""

    def recall(self, query: str, k: int = 4) -> list[Hit]:
        return []

    def save_fact(self, name: str, body: str, description: str = "", type: str = "user") -> str:
        return ""

    def forget(self, query: str) -> list[str]:
        return []

    def log_turn(self, session_id: str, role: str, text: str) -> None:
        pass

    def end_session(self, session_id: str, digest: str) -> None:
        pass

    def transcript(self, session_id: str) -> str:
        return ""

    def facts(self, types: tuple[str, ...] | None = None) -> list[Fact]:
        return []

    def write_profile(self, text: str) -> None:
        pass

    def sync(self) -> int:
        return 0


def create(cfg: MemoryConfig) -> Memory:
    return FileMemory(cfg) if cfg.enabled else NullMemory()
