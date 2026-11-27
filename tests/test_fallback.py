"""The brain priority list: a failed brain hands the same turn to the next one."""

from __future__ import annotations

import pytest

from omi.brain.fallback import FallbackBrain, format_entries, parse_entries
from omi.events import TextDelta, ToolEnd, ToolStart, TurnContext, TurnDone
from tests.fakes import FakeBrain

LIMIT = "You've hit your usage limit. Resets 5pm."


def chain(scripts: dict[str, list], switches: list | None = None, **kw) -> tuple[FallbackBrain, dict[str, FakeBrain]]:
    made: dict[str, FakeBrain] = {}

    def make(provider: str, model: str) -> FakeBrain:
        made[provider] = FakeBrain(scripts[provider], name=provider)
        return made[provider]

    async def on_switch(active: str, reason: str) -> None:
        if switches is not None:
            switches.append((active, reason))

    return FallbackBrain([(p, "") for p in scripts], make, on_switch=on_switch, **kw), made


async def run(brain: FallbackBrain, text: str = "hi") -> list:
    return [ev async for ev in brain.turn(text, TurnContext(session_id="s"))]


def test_parse_entries():
    assert parse_entries("opencode:opencode/big-pickle, local,local") == [("opencode", "opencode/big-pickle"), ("local", "")]
    assert format_entries(parse_entries(["Claude:sonnet", " "])) == ["claude:sonnet"]


@pytest.mark.asyncio
async def test_usage_limit_falls_back_to_the_next_brain_in_the_same_turn():
    switches: list = []
    brain, made = chain({
        "claude": [(0.0, TurnDone(error=LIMIT))],
        "opencode": [(0.0, TextDelta("Hi from the backup.")), (0.0, TurnDone())],
    }, switches)
    await brain.start()
    events = await run(brain)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hi from the backup."]
    assert isinstance(events[-1], TurnDone) and events[-1].error is None
    assert brain.active == "opencode" and "usage limit" in brain.fallback_reason
    assert switches == [("opencode", LIMIT)]
    # The failed brain sits out: the next turn goes straight to the backup.
    await run(brain)
    assert len(made["claude"].turns) == 1 and len(made["opencode"].turns) == 2


@pytest.mark.asyncio
async def test_preferred_brain_is_retried_after_the_cooldown():
    brain, made = chain({
        "claude": [(0.0, TurnDone(error=LIMIT))],
        "opencode": [(0.0, TextDelta("ok")), (0.0, TurnDone())],
    }, cooldown_s=0.0)
    await run(brain)
    made["claude"].script = [(0.0, TextDelta("back")), (0.0, TurnDone())]
    events = await run(brain)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["back"]
    assert brain.active == "claude" and brain.fallback_reason == ""


@pytest.mark.asyncio
async def test_a_turn_that_already_spoke_or_acted_is_not_retried():
    brain, made = chain({
        "claude": [(0.0, ToolStart("1", "volume", {})), (0.0, ToolEnd("1", "volume", True, "ok")),
                   (0.0, TurnDone(error=LIMIT))],
        "opencode": [(0.0, TextDelta("never")), (0.0, TurnDone())],
    })
    events = await run(brain)
    assert events[-1].error == LIMIT
    assert "opencode" not in made


@pytest.mark.asyncio
async def test_every_brain_failing_reports_the_last_error():
    brain, _ = chain({"claude": [(0.0, TurnDone(error=LIMIT))], "local": [(0.0, TurnDone(error="server down"))]})
    events = await run(brain)
    assert len(events) == 1 and events[0].error == "server down"


@pytest.mark.asyncio
async def test_a_brain_that_cannot_start_is_skipped():
    made: dict[str, FakeBrain] = {}

    def make(provider: str, model: str):
        if provider == "codex":
            raise RuntimeError("codex is not logged in")
        made[provider] = FakeBrain([(0.0, TextDelta("local here")), (0.0, TurnDone())], name=provider)
        return made[provider]

    brain = FallbackBrain([("codex", ""), ("local", "")], make)
    events = await run(brain)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["local here"]
    assert brain.name == "codex" and brain.active == "local"
