"""After-session jobs. The LLM is injected (an async str -> str), so any brain can run them."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from omi.memory.base import Hit, Memory
from omi.memory.store import FACT_TYPES, Fact, looks_secret, trim_tokens

log = logging.getLogger("omi.memory")

LLM = Callable[[str], Awaitable[str]]

MAX_TRANSCRIPT_CHARS = 6000  # a 4B model with an 8k context must fit prompt and answer
PROFILE_MAX_TOKENS = 500
MAX_BULLETS = 5
OPS = ("ADD", "UPDATE", "NOOP")

DIGEST_PROMPT = """Below is a conversation between a voice assistant and its user.
Write 3 to 5 short bullet points that sum it up: what the user asked for, what was done,
any corrections or mistakes the user pointed out about the assistant's behaviour, and
anything left open. Start every line with "- ". No introduction, nothing else.

Conversation:
{transcript}
"""

EXTRACT_PROMPT = """You keep long-term memory about a user of a voice assistant.
Read the session digest and decide what to store.

Rules:
- Keep durable facts about the user, their preferences, their ongoing projects, and any
  corrections or mistakes they asked the assistant not to repeat.
- Store behavioural corrections and lessons learned with "type": "feedback", stating what
  went wrong and what to do instead next time.
- Never store secrets, passwords, keys, tokens, health details or financial details.
- Skip one-off task details (a single search, a timer, one question).
- Prefer UPDATE over ADD: if an existing fact covers the same thing, UPDATE it using its exact name.
- If nothing is worth keeping, answer with [{{"op": "NOOP"}}].

Answer with JSON only: a list of objects with these keys:
  "op": "ADD" or "UPDATE" or "NOOP"
  "name": a short title, 2 to 6 words
  "description": one line saying what the fact is about
  "type": "user" or "feedback" or "project" or "reference"
  "body": the fact in one or two plain sentences

Existing similar facts:
{existing}

Session digest:
{digest}
"""

PROFILE_PROMPT = """Write a short profile of the user for a voice assistant to read before every
conversation. Use plain sentences about who they are, how they like to be helped, and what they
are working on. At most 300 words. No headings, no lists of rules, no introduction.

Facts:
{facts}
"""



def _clean_llm_text(text: str) -> str:
    """Drop code fences and <think> blocks some models wrap around their answer."""
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    return re.sub(r"(?m)^\s*```\w*\s*$", "", text).strip()


async def digest(llm: LLM, transcript: str) -> str:
    """3 to 5 bullet lines for a finished session. Empty if the model gave nothing usable."""
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[-MAX_TRANSCRIPT_CHARS:]
        transcript = transcript.split("\n", 1)[-1]  # start on a whole line
    raw = _clean_llm_text(await llm(DIGEST_PROMPT.format(transcript=transcript)))
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    bullets = [re.sub(r"^([-*•]|\d+[.)])\s*", "", ln) for ln in lines if re.match(r"([-*•]|\d+[.)])\s", ln)]
    bullets = bullets or lines  # model ignored the bullet format: keep its lines
    return "\n".join(f"- {b}" for b in bullets[:MAX_BULLETS])


def _json_candidates(text: str):
    """Yield every JSON value that parses, starting at each [ or { in the text."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[m.start() :])
        except ValueError:
            continue
        yield value


def _as_ops(value: Any) -> list[dict] | None:
    if isinstance(value, dict):
        for key in ("ops", "facts", "operations"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value] if "op" in value else None
    if isinstance(value, list):
        return value
    return None


def _valid_op(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    op = str(raw.get("op", "")).strip().upper()
    if op not in OPS:
        return None
    if op == "NOOP":
        return {"op": "NOOP", "name": "", "description": "", "type": "user", "body": ""}
    name = " ".join(str(raw.get("name") or "").split())
    body = str(raw.get("body") or "").strip()
    if not name or not body:
        return None
    type_ = str(raw.get("type") or "user").strip().lower()
    return {
        "op": op,
        "name": name,
        "description": " ".join(str(raw.get("description") or "").split()),
        "type": type_ if type_ in FACT_TYPES else "user",
        "body": body,
    }


def parse_ops(text: str) -> list[dict]:
    """Pull the list of ops out of a model answer, tolerating fences and surrounding junk."""
    for value in _json_candidates(_clean_llm_text(text)):
        raw_ops = _as_ops(value)
        if raw_ops is None:
            continue
        ops = [o for o in map(_valid_op, raw_ops) if o]
        if ops:
            return ops
    return []


async def extract_facts(llm: LLM, digest_text: str, similar_existing_facts: list[str]) -> list[dict]:
    """Ask the model for ADD/UPDATE/NOOP ops. Bad output gives []."""
    existing = "\n".join(f"- {f}" for f in similar_existing_facts) or "(none)"
    raw = await llm(EXTRACT_PROMPT.format(existing=existing, digest=digest_text))
    return parse_ops(raw)


def apply_ops(memory: Memory, ops: list[dict]) -> list[str]:
    """Write the ADD and UPDATE ops. Returns the names saved. Secret-looking text is dropped."""
    saved = []
    for op in ops:
        if op.get("op") not in ("ADD", "UPDATE"):
            continue
        text = f"{op.get('name', '')} {op.get('description', '')} {op.get('body', '')}"
        if looks_secret(text):
            log.warning("memory: dropped a fact that looked like a secret")
            continue
        memory.save_fact(op["name"], op["body"], op.get("description", ""), op.get("type", "user"))
        saved.append(op["name"])
    return saved


def _fact_line(f: Fact) -> str:
    return f"{f.name} ({f.type}): {f.description or ''} {f.body}".strip()[:300]


async def rebuild_profile(memory: Any, llm: LLM) -> str:
    """Rewrite profile.md from the user and feedback facts. Returns the text ("" if skipped)."""
    facts = memory.facts(("user", "feedback"))[:60]
    if not facts:
        return ""
    prompt = PROFILE_PROMPT.format(facts="\n".join(f"- {_fact_line(f)}" for f in facts))
    text = trim_tokens(_clean_llm_text(await llm(prompt)), PROFILE_MAX_TOKENS)
    if text:
        await asyncio.to_thread(memory.write_profile, text)
    return text


def _fact_hit_line(hit: Hit) -> str:
    return f"{hit.name}: {hit.text}"


async def end_session(memory: Any, session_id: str, llm: LLM) -> str:
    """Digest the session, roll it into the carry-over, extract facts, and rewrite the
    profile when facts changed. Never raises.

    Returns the digest ("" when the session was empty or the model failed).
    """
    try:
        transcript = await asyncio.to_thread(memory.transcript, session_id)
        if not transcript.strip():
            return ""
        text = await digest(llm, transcript)
        if not text:
            return ""
        await asyncio.to_thread(memory.end_session, session_id, text)
    except Exception:
        log.exception("memory: digest failed for session %s", session_id)
        return ""
    try:
        similar = [h for h in await asyncio.to_thread(memory.recall, text, 5) if h.kind == "fact"]
        ops = await extract_facts(llm, text, [_fact_hit_line(h) for h in similar])
        saved = await asyncio.to_thread(apply_ops, memory, ops)
    except Exception:
        log.exception("memory: fact extraction failed for session %s", session_id)
        return text
    if saved:
        try:
            await rebuild_profile(memory, llm)
        except Exception:
            log.exception("memory: profile rebuild failed for session %s", session_id)
    return text


async def profile_if_missing(memory: Any, llm: LLM) -> str:
    """Write profile.md when it is empty but facts exist (a fresh install with shared
    facts, or a deleted profile). Returns the profile text, "" if nothing was done."""
    has_profile = await asyncio.to_thread(lambda: bool(getattr(memory, "profile", lambda: "")()))
    if has_profile:
        return ""
    return await rebuild_profile(memory, llm)
