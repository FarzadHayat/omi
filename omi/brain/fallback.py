"""A priority list of brains: the preferred one answers, and when it fails before saying
anything (usage limit, expired login, missing CLI, local server down) the next one takes
the same turn, so the user never has to switch by hand.

A brain that failed sits out for a cooldown, so later turns go straight to the next one
without paying for another failed attempt; after the cooldown the preferred brain gets
tried again. A turn that already spoke or ran a tool is never retried, so nothing is
said or done twice.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from time import monotonic
from typing import Any

from omi.events import BrainEvent, TextDelta, ToolEnd, ToolStart, TurnContext, TurnDone

log = logging.getLogger(__name__)

COOLDOWN_S = 15 * 60

# (provider, model); an empty model means the provider's configured model.
Entry = tuple[str, str]
MakeBrain = Callable[[str, str], Any]


def parse_entries(raw: list[str] | str) -> list[Entry]:
    """'opencode:opencode/deepseek-free, local' -> [('opencode', 'opencode/deepseek-free'), ('local', '')]."""
    items = raw.split(",") if isinstance(raw, str) else list(raw)
    out: list[Entry] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        provider, _, model = text.partition(":")
        entry = (provider.strip().lower(), model.strip())
        if entry[0] and entry not in out:
            out.append(entry)
    return out


def format_entries(entries: list[Entry]) -> list[str]:
    return [f"{p}:{m}" if m else p for p, m in entries]


class FallbackBrain:
    """Wraps the talker. `name` stays the preferred provider; `active` is who answered last."""

    def __init__(
        self,
        entries: list[Entry],
        make: MakeBrain,
        *,
        cooldown_s: float = COOLDOWN_S,
        on_switch: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        if not entries:
            raise ValueError("a fallback chain needs at least one brain")
        self.entries = entries
        self._make = make
        self.cooldown_s = cooldown_s
        self.on_switch = on_switch
        self._brains: dict[int, Any] = {}
        self._failed_at: dict[int, float] = {}
        self._reasons: dict[int, str] = {}
        self._current: Any = None
        self._memory_context = ""
        self.active_index = 0

    # --- what status shows ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self.entries[0][0]

    @property
    def active(self) -> str:
        provider, model = self.entries[self.active_index]
        return f"{provider}:{model}" if model else provider

    @property
    def error(self) -> str | None:
        """Only an error when no brain in the chain can answer."""
        if self.active_index == 0:
            primary = self._brains.get(0)
            return getattr(primary, "error", None) if primary is not None else None
        return None

    @property
    def fallback_reason(self) -> str:
        """Why the preferred brain is sitting out, empty when it is answering."""
        return self._reasons.get(0, "") if self.active_index != 0 else ""

    # --- lifecycle -----------------------------------------------------------------

    async def _brain(self, idx: int) -> Any:
        brain = self._brains.get(idx)
        if brain is None:
            provider, model = self.entries[idx]
            brain = self._make(provider, model)
            self._brains[idx] = brain
            await brain.start(self._memory_context)
        return brain

    async def start(self, memory_context: str = "") -> None:
        self._memory_context = memory_context
        for idx, brain in list(self._brains.items()):
            if idx != 0:
                await brain.start(memory_context)
        if 0 in self._brains:
            await self._brains[0].start(memory_context)
        else:
            await self._brain(0)

    def _order(self) -> list[int]:
        now = monotonic()
        ready = [i for i in range(len(self.entries)) if now - self._failed_at.get(i, -1e18) >= self.cooldown_s]
        return ready or list(range(len(self.entries)))  # everything failed lately: try them all again

    def _fail(self, idx: int, reason: str) -> None:
        self._failed_at[idx] = monotonic()
        self._reasons[idx] = reason
        log.warning("brain %s failed, falling back: %s", self.entries[idx][0], reason)

    async def turn(self, user_text: str, ctx: TurnContext) -> AsyncIterator[BrainEvent]:
        last_error = "no brain could answer"
        for idx in self._order():
            try:
                brain = await self._brain(idx)
            except Exception as exc:  # could not start (missing CLI, local server failed)
                self._brains.pop(idx, None)
                self._fail(idx, str(exc) or type(exc).__name__)
                last_error = self._reasons[idx]
                continue
            if getattr(brain, "error", None) and len(self.entries) > 1:
                self._fail(idx, brain.error)
                last_error = brain.error
                continue
            await self._switch_to(idx)
            spoke = acted = False
            async for ev in brain.turn(user_text, ctx):
                if isinstance(ev, TurnDone):
                    if ev.error and not ev.cancelled and not spoke and not acted:
                        self._fail(idx, ev.error)
                        last_error = ev.error
                        break  # nothing reached the user: the next brain takes this turn
                    if not ev.error:
                        self._failed_at.pop(idx, None)
                        self._reasons.pop(idx, None)
                    yield ev
                    return
                if isinstance(ev, TextDelta) and ev.text.strip():
                    spoke = True
                elif isinstance(ev, (ToolStart, ToolEnd)):
                    acted = True
                yield ev
            else:
                yield TurnDone()
                return
        yield TurnDone(error=last_error)

    async def _switch_to(self, idx: int) -> None:
        self._current = self._brains[idx]
        if idx != self.active_index:
            self.active_index = idx
            if self.on_switch is not None:
                try:
                    await self.on_switch(self.active, self.fallback_reason)
                except Exception:
                    log.exception("fallback notice failed")

    async def cancel(self) -> None:
        if self._current is not None:
            await self._current.cancel()

    async def close(self) -> None:
        brains, self._brains = list(self._brains.values()), {}
        for brain in brains:
            try:
                await brain.close()
            except Exception:
                log.exception("closing a fallback brain failed")
