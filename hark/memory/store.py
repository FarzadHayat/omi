"""Markdown file layer: the files are the truth, everything else is a cache.

Every write goes to a temp file in the same folder and is renamed into place, so a
sync tool (Syncthing, Obsidian Sync) never sees a half-written file.
"""

from __future__ import annotations

import json
import os
import re
import socket
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

DIGEST_HEADING = "## Digest"
FACT_TYPES = ("user", "feedback", "project", "reference")
CARRYOVER_KEEP = 5
CARRYOVER_MAX_TOKENS = 300


@dataclass
class Fact:
    name: str
    description: str
    type: str
    body: str
    source: str = "own"  # "own" | "shared"


def approx_tokens(text: str) -> int:
    """Cheap token estimate, deliberately on the high side (4 characters per token)."""
    return (len(text) + 3) // 4


def trim_tokens(text: str, max_tokens: int) -> str:
    """Cut text to the token budget at a line boundary when possible."""
    limit = max_tokens * 4
    if len(text) <= limit:
        return text
    cut = text[:limit]
    nl = cut.rfind("\n")
    return (cut[:nl] if nl > limit // 2 else cut).rstrip()


def slugify(name: str, limit: int = 60) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].strip("-")
    return slug or "fact"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --- frontmatter: a tiny YAML subset, enough for name/description/metadata.type ---

_PLAIN = re.compile(r"^[A-Za-z][A-Za-z0-9 _./()-]*$")
_RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "y", "n"}


def _quote(value: str) -> str:
    value = " ".join(value.split())
    if _PLAIN.match(value) and value.lower() not in _RESERVED:
        return value
    return json.dumps(value, ensure_ascii=False)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return str(json.loads(value))
        except ValueError:
            return value[1:-1]
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file into (frontmatter dict, body). No frontmatter gives ({}, text)."""
    text = text.replace("\r\n", "\n").lstrip("﻿")
    if not text.startswith("---\n"):
        return {}, text
    lines = text.split("\n")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text
    meta: dict = {}
    current: str | None = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        key, value = key.strip(), _unquote(value.strip())
        if line[0] in " \t" and current is not None and isinstance(meta.get(current), dict):
            meta[current][key] = value
        elif value == "":
            meta[key] = {}
            current = key
        else:
            meta[key] = value
            current = None
    return meta, "\n".join(lines[end + 1 :]).lstrip("\n")


def meta_str(meta: dict, key: str) -> str:
    value = meta.get(key, "")
    return value if isinstance(value, str) else ""


def meta_type(meta: dict, default: str = "") -> str:
    nested = meta.get("metadata")
    if isinstance(nested, dict) and nested.get("type"):
        return str(nested["type"])
    return meta_str(meta, "type") or default


def render_fact(name: str, description: str, type: str, body: str) -> str:
    return (
        "---\n"
        f"name: {_quote(name)}\n"
        f"description: {_quote(description)}\n"
        "metadata:\n"
        f"  type: {_quote(type)}\n"
        f"updated: {date.today().isoformat()}\n"
        "---\n\n"
        f"{body.strip()}\n"
    )


def _stem_title(stem: str) -> str:
    """Heading for a session in carryover.md. Unique per session file."""
    try:
        when = datetime.strptime(stem[:15], "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stem
    return f"{when} {stem[16:]}".strip()


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.facts_dir = self.root / "facts"
        self.sessions_dir = self.root / "sessions"
        self.trash_dir = self.root / ".trash"
        self._host = slugify(socket.gethostname().split(".")[0])

    # --- profile and carry-over ---

    def _read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def profile(self) -> str:
        return self._read(self.root / "profile.md")

    def write_profile(self, text: str) -> None:
        atomic_write(self.root / "profile.md", text.strip() + "\n")

    def carryover(self) -> str:
        return self._read(self.root / "carryover.md")

    def roll_carryover(self, session_stem: str, digest: str) -> str:
        """Add one digest, keep the last 5 and stay inside the token budget."""
        entries = [e for e in re.split(r"(?m)^(?=### )", self.carryover()) if e.strip()]
        heading = f"### {_stem_title(session_stem)}"
        entries = [e for e in entries if e.split("\n", 1)[0].strip() != heading]
        entries.append(f"{heading}\n{digest.strip()}\n")
        entries = [e.strip() for e in entries[-CARRYOVER_KEEP:]]
        while len(entries) > 1 and approx_tokens("\n\n".join(entries)) > CARRYOVER_MAX_TOKENS:
            entries.pop(0)
        text = trim_tokens("\n\n".join(entries), CARRYOVER_MAX_TOKENS)
        atomic_write(self.root / "carryover.md", text + "\n")
        return text

    # --- facts ---

    def fact_path(self, name: str) -> Path:
        """Path a fact with this name would use: same name reuses its file, a different
        name that slugs the same gets a numbered file."""
        slug = slugify(name)
        path = self.facts_dir / f"{slug}.md"
        n = 2
        while path.exists():
            meta, _ = parse_frontmatter(self._read(path))
            if (meta_str(meta, "name") or path.stem).casefold() == name.strip().casefold():
                break
            path = self.facts_dir / f"{slug}-{n}.md"
            n += 1
        return path

    def write_fact(self, name: str, body: str, description: str = "", type: str = "user") -> Path:
        name = " ".join(name.split()) or "fact"
        path = self.fact_path(name)
        atomic_write(path, render_fact(name, description or name, type or "user", body))
        return path

    def trash(self, path: Path) -> Path:
        """Move a file into .trash/ instead of deleting it."""
        self.trash_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = self.trash_dir / f"{stamp}-{path.name}"
        n = 2
        while dest.exists():
            dest = self.trash_dir / f"{stamp}-{n}-{path.name}"
            n += 1
        os.replace(path, dest)
        return dest

    # --- sessions ---

    def new_session_path(self, session_id: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{self._host}"
        path = self.sessions_dir / f"{base}.md"
        n = 2
        while path.exists():
            path = self.sessions_dir / f"{base}-{n}.md"
            n += 1
        header = (
            "---\n"
            f"session: {json.dumps(session_id)}\n"
            f"started: {datetime.now().isoformat(timespec='seconds')}\n"
            "---\n\n"
        )
        atomic_write(path, header)
        return path

    def find_session(self, session_id: str) -> Path | None:
        """Look a session file up by the id in its frontmatter (after a restart)."""
        try:
            files = sorted(self.sessions_dir.glob("*.md"), reverse=True)[:50]
        except OSError:
            return None
        needle = f"session: {json.dumps(session_id)}"
        for path in files:
            try:
                with path.open(encoding="utf-8") as fh:
                    head = fh.read(400)
            except OSError:
                continue
            if needle in head:
                return path
        return None

    @staticmethod
    def append_turn(path: Path, role: str, text: str) -> None:
        line = f"{role.strip() or 'user'}: {' '.join(text.split())}\n"
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))  # one write call: lines never interleave
        finally:
            os.close(fd)

    def has_digest(self, path: Path) -> bool:
        return f"\n{DIGEST_HEADING}" in self._read(path)

    def read_transcript(self, path: Path) -> str:
        _, body = parse_frontmatter(self._read(path))
        return body.split(f"\n{DIGEST_HEADING}", 1)[0].strip()

    def set_digest(self, path: Path, digest: str) -> None:
        """Replace or add the Digest section; transcript lines stay untouched."""
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        head = text.split(f"\n{DIGEST_HEADING}", 1)[0].rstrip("\n")
        atomic_write(path, f"{head}\n\n{DIGEST_HEADING}\n\n{digest.strip()}\n")
