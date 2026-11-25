"""Voice-facing memory tools: remember, recall, forget."""

from __future__ import annotations

import asyncio

from hark.memory.base import Memory
from hark.tools.base import Tool, ToolResult

_STR = {"type": "string"}


def _name_for(fact: str) -> str:
    words = fact.split()
    return " ".join(words[:8])


def memory_tools(memory: Memory) -> list[Tool]:
    async def remember(fact: str) -> ToolResult:
        fact = " ".join(fact.split())
        if not fact:
            return ToolResult(False, "I did not catch what to remember.")
        await asyncio.to_thread(
            memory.save_fact, _name_for(fact), fact, description=fact[:120], type="user"
        )
        return ToolResult(True, "Got it.")

    async def recall(query: str) -> ToolResult:
        hits = memory.recall(query, k=4)
        if not hits:
            return ToolResult(True, "I don't have anything on that.", [])
        text = "\n".join(f"{h.name}: {h.text}" for h in hits)
        return ToolResult(True, text, [h.name for h in hits])

    async def forget(query: str) -> ToolResult:
        removed = await asyncio.to_thread(memory.forget, query)
        if not removed:
            return ToolResult(True, "I couldn't find anything to forget.", [])
        return ToolResult(True, "Forgot " + ", ".join(removed) + ".", removed)

    return [
        Tool(
            "remember",
            "Save a lasting fact about the user or their preferences so you know it next time.",
            {
                "type": "object",
                "properties": {"fact": {**_STR, "description": "One short self-contained sentence."}},
                "required": ["fact"],
            },
            "reversible",
            remember,
        ),
        Tool(
            "recall",
            "Search saved memories and past conversations for something the user told you before.",
            {
                "type": "object",
                "properties": {"query": {**_STR, "description": "A few keywords."}},
                "required": ["query"],
            },
            "read",
            recall,
        ),
        Tool(
            "forget",
            "Forget a saved fact. It is moved to a trash folder, so it can be restored.",
            {
                "type": "object",
                "properties": {"query": {**_STR, "description": "Keywords of the fact to forget."}},
                "required": ["query"],
            },
            "reversible",
            forget,
        ),
    ]
