"""SQLite FTS5 index over facts and sessions. A cache: delete the file and it rebuilds.

Own files (facts, session digests and transcripts) and the shared agent-memory facts
(read-only) go in one table. File mtimes decide what to re-read at startup.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from pathlib import Path

from hark.memory.base import Hit
from hark.memory.store import (
    DIGEST_HEADING,
    Fact,
    meta_str,
    meta_type,
    parse_frontmatter,
)

log = logging.getLogger("hark.memory")

SCHEMA_VERSION = "1"
CHUNK_CHARS = 700
SNIPPET_CHARS = 280
MAX_TERMS = 12
# Names in the shared folder that are never facts.
SHARED_SKIP = {"memory.md", "profile.md", "carryover.md"}
PRIVATE_MARKERS = ("private", "personal-details")

STOP_WORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its just me more most my
    no nor not now of off on once only or other our out over own same she should so some such
    than that the their them then there these they this those through to too under until up us
    very was we were what when where which while who whom why will with would you your yours
    hey please okay ok remember tell know said say get got""".split()
)

_SCHEMA = """
CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE files(path TEXT PRIMARY KEY, mtime_ns INTEGER, size INTEGER);
CREATE TABLE chunks(
    id INTEGER PRIMARY KEY, path TEXT NOT NULL, kind TEXT, source TEXT,
    type TEXT, name TEXT, descr TEXT, body TEXT);
CREATE INDEX chunks_path ON chunks(path);
CREATE VIRTUAL TABLE fts USING fts5(
    name, descr, body, content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts(rowid, name, descr, body) VALUES (new.id, new.name, new.descr, new.body);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts(fts, rowid, name, descr, body)
    VALUES ('delete', old.id, old.name, old.descr, old.body);
END;
"""


def build_query(text: str) -> str:
    """Turn free speech into a safe FTS5 query: quoted prefix terms joined by OR.

    Returns "" when nothing useful is left (empty or only stop words).
    """
    terms: list[str] = []
    for word in re.findall(r"[^\W_]+", text.lower()):
        if len(word) < 2 or word in STOP_WORDS or word in terms:
            continue
        terms.append(word)
        if len(terms) >= MAX_TERMS:
            break
    # Prefix match covers "tea" -> "teas"; for the other way round also try the singular.
    singles = [t[:-1] for t in terms if len(t) > 3 and t.endswith("s") and not t.endswith("ss")]
    return " OR ".join(f'"{t}"*' for t in dict.fromkeys(terms + singles))


def _short(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


def _is_private(name: str, description: str, filename: str) -> bool:
    blob = f"{name}\n{description}\n{filename}".lower()
    return any(marker in blob for marker in PRIVATE_MARKERS)


def _split_session(text: str) -> tuple[str, str]:
    _, body = parse_frontmatter(text)
    transcript, _, digest = body.partition(f"\n{DIGEST_HEADING}")
    return transcript.strip(), digest.strip()


def _chunk_lines(text: str) -> list[str]:
    chunks, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + len(line) > CHUNK_CHARS:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks


class Index:
    def __init__(self, db_path: str | Path, root: str | Path, shared_dir: str | Path | None):
        self.db_path = Path(db_path)
        self.root = Path(root)
        self.shared = Path(shared_dir) if shared_dir else None
        self._lock = threading.RLock()
        self._conn = self._open()

    # --- lifecycle ---

    def _open(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in (1, 2):
            try:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=OFF")  # a cache: speed over durability
                row = None
                try:
                    row = conn.execute("SELECT v FROM meta WHERE k='version'").fetchone()
                except sqlite3.OperationalError:
                    pass
                if row is None or row[0] != SCHEMA_VERSION:
                    self._reset(conn)
                return conn
            except sqlite3.DatabaseError:
                if attempt == 2:
                    raise
                log.warning("memory index unreadable, rebuilding %s", self.db_path)
                for suffix in ("", "-wal", "-shm"):
                    Path(f"{self.db_path}{suffix}").unlink(missing_ok=True)
        raise AssertionError("unreachable")

    @staticmethod
    def _reset(conn: sqlite3.Connection) -> None:
        for kind, name in conn.execute("SELECT type, name FROM sqlite_master").fetchall():
            if name.startswith("sqlite_"):
                continue
            if kind in ("trigger", "table", "index", "view") and not name.startswith("fts_"):
                conn.execute(f'DROP {kind} IF EXISTS "{name}"')
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO meta VALUES('version', ?)", (SCHEMA_VERSION,))
        conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- reading files into rows ---

    def _classify(self, path: Path) -> tuple[str, str] | None:
        """(kind, source) for a path we index, or None."""
        if path.suffix != ".md":
            return None
        if path.parent == self.root / "facts":
            return ("fact", "own")
        if path.parent == self.root / "sessions":
            return ("session", "own")
        if self.shared and path.parent == self.shared and path.name.lower() not in SHARED_SKIP:
            return ("fact", "shared")
        return None

    def _rows(self, path: Path, kind: str, source: str) -> list[tuple]:
        """Chunk rows for one file. Empty when the file is skipped."""
        text = path.read_text(encoding="utf-8", errors="replace")
        p = str(path)
        if kind == "session":
            transcript, digest = _split_session(text)
            rows = [(p, "session", source, "", path.stem, "", digest)] if digest else []
            rows += [(p, "session", source, "", path.stem, "", c) for c in _chunk_lines(transcript)]
            return rows
        meta, body = parse_frontmatter(text)
        if source == "shared" and not meta:
            return []  # not a fact file; do not guess
        name = meta_str(meta, "name") or path.stem
        descr = meta_str(meta, "description")
        if source == "shared" and _is_private(name, descr, path.name):
            return []
        return [(p, "fact", source, meta_type(meta), name, descr, body.strip())]

    def _store(self, conn: sqlite3.Connection, path: Path, rows: list[tuple]) -> None:
        conn.execute("DELETE FROM chunks WHERE path=?", (str(path),))
        conn.executemany(
            "INSERT INTO chunks(path, kind, source, type, name, descr, body) VALUES (?,?,?,?,?,?,?)",
            rows,
        )

    # --- keeping the cache in step with the files ---

    def upsert_file(self, path: str | Path) -> None:
        """Re-read one file after we wrote it (or it changed)."""
        path = Path(path)
        cls = self._classify(path)
        if cls is None:
            return
        with self._lock, self._conn as conn:
            try:
                st = path.stat()
                rows = self._rows(path, *cls)
            except OSError:
                self._store(conn, path, [])
                conn.execute("DELETE FROM files WHERE path=?", (str(path),))
                return
            self._store(conn, path, rows)
            conn.execute(
                "INSERT OR REPLACE INTO files VALUES (?,?,?)", (str(path), st.st_mtime_ns, st.st_size)
            )

    def remove_file(self, path: str | Path) -> None:
        with self._lock, self._conn as conn:
            conn.execute("DELETE FROM chunks WHERE path=?", (str(path),))
            conn.execute("DELETE FROM files WHERE path=?", (str(path),))

    def _scan(self) -> dict[Path, tuple[int, int, str, str]]:
        found: dict[Path, tuple[int, int, str, str]] = {}
        spots = [
            (self.root / "facts", "fact", "own"),
            (self.root / "sessions", "session", "own"),
        ]
        if self.shared:
            spots.append((self.shared, "fact", "shared"))
        for folder, kind, source in spots:
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue
            for e in entries:
                if not e.name.endswith(".md") or e.name.startswith("."):
                    continue
                if source == "shared" and e.name.lower() in SHARED_SKIP:
                    continue
                try:
                    if not e.is_file():
                        continue
                    st = e.stat()
                except OSError:
                    continue
                found[Path(e.path)] = (st.st_mtime_ns, st.st_size, kind, source)
        return found

    def sync(self) -> int:
        """Bring the index in step with the files by mtime. Returns how many files changed."""
        found = self._scan()
        changed = 0
        with self._lock, self._conn as conn:
            known = {p: (m, s) for p, m, s in conn.execute("SELECT path, mtime_ns, size FROM files")}
            for path_str in known.keys() - {str(p) for p in found}:
                conn.execute("DELETE FROM chunks WHERE path=?", (path_str,))
                conn.execute("DELETE FROM files WHERE path=?", (path_str,))
                changed += 1
            for path, (mtime, size, kind, source) in found.items():
                if known.get(str(path)) == (mtime, size):
                    continue
                try:
                    rows = self._rows(path, kind, source)
                except OSError:
                    continue
                self._store(conn, path, rows)
                conn.execute("INSERT OR REPLACE INTO files VALUES (?,?,?)", (str(path), mtime, size))
                changed += 1
        return changed

    # --- queries ---

    def search(self, query: str, k: int = 4, own_facts_only: bool = False) -> list[Hit]:
        match = build_query(query)
        if not match:
            return []
        sql = (
            "SELECT c.path, c.kind, c.name, c.descr, c.body, snippet(fts, 2, '', '', '…', 30) "
            "FROM fts JOIN chunks c ON c.id = fts.rowid WHERE fts MATCH ?"
        )
        if own_facts_only:
            sql += " AND c.kind='fact' AND c.source='own'"
        sql += " ORDER BY bm25(fts, 6.0, 3.0, 1.0) LIMIT ?"
        with self._lock:
            try:
                rows = self._conn.execute(sql, (match, max(k * 4, 8))).fetchall()
            except sqlite3.OperationalError as exc:
                log.warning("memory search failed: %s", exc)
                return []
        hits: list[Hit] = []
        seen: set[str] = set()
        for path, kind, name, descr, body, snip in rows:
            if path in seen:
                continue
            seen.add(path)
            if kind == "fact":
                text = body if not descr or descr in body else f"{descr}: {body}"
                text = _short(text)
            else:
                text = _short(snip)
            hits.append(Hit(kind=kind, name=name, text=text, path=path))
            if len(hits) >= k:
                break
        return hits

    def facts(self, types: tuple[str, ...] | None = None) -> list[Fact]:
        sql = "SELECT name, descr, type, body, source FROM chunks WHERE kind='fact'"
        args: tuple = ()
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            args = tuple(types)
        sql += " ORDER BY source ASC, name"  # own facts first
        with self._lock:
            return [Fact(*row) for row in self._conn.execute(sql, args).fetchall()]
