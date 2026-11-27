"""The brain priority list: a failed brain hands the same turn to the next one."""

from __future__ import annotations

import asyncio

import pytest

from omi.brain.fallback import FallbackBrain, FallbackWorker, format_entries, parse_entries
from omi.events import TextDelta, ToolEnd, ToolStart, TurnContext, TurnDone, WorkerDone, WorkerProgress
from tests.fakes import FakeBrain, FakeWorker

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


class ScriptedWorker(FakeWorker):
    """FakeWorker that automatically replies to each submitted job from a script."""

    def __init__(self, runs: list[list], *, fail_start: str | None = None) -> None:
        super().__init__()
        self.runs = list(runs)
        self.fail_start = fail_start
        self.cancelled: list[str | None] = []

    async def start(self) -> None:
        if self.fail_start:
            raise RuntimeError(self.fail_start)

    async def submit(self, task: str) -> str:
        jid = await super().submit(task)
        script = self.runs.pop(0) if self.runs else [("done", True, "Done.")]
        for item in script:
            if item[0] == "progress":
                self.push(WorkerProgress(jid, item[1]))
            elif item[0] == "done":
                self.push(WorkerDone(jid, item[1], item[2]))
        return jid

    async def cancel(self, job_id: str | None = None) -> None:
        self.cancelled.append(job_id)


async def next_worker_events(worker: FallbackWorker, timeout: float = 1.0) -> list:
    out = []
    async for ev in worker.events():
        out.append(ev)
        if isinstance(ev, WorkerDone):
            return out
    return out


@pytest.mark.asyncio
async def test_worker_usage_limit_falls_back_in_the_same_job():
    switches: list = []
    made: dict[tuple[str, str], ScriptedWorker] = {}
    scripts = {
        ("claude", ""): [[("done", False, LIMIT)]],
        ("codex", ""): [[("progress", "Reading files"), ("done", True, "Fixed it.")],
                        [("done", True, "Second job done.")]],
    }

    def make(provider: str, model: str) -> ScriptedWorker:
        w = ScriptedWorker(scripts[(provider, model)])
        made[(provider, model)] = w
        return w

    async def on_switch(active: str, reason: str) -> None:
        switches.append((active, reason))

    w = FallbackWorker([("claude", ""), ("codex", "")], make, on_switch=on_switch)
    await w.start()
    try:
        jid = await w.submit("fix the bug")
        events = await asyncio.wait_for(next_worker_events(w), 1)
        assert [e for e in events if isinstance(e, WorkerProgress)][0] == WorkerProgress(jid, "Reading files")
        done = events[-1]
        assert done == WorkerDone(jid, True, "Fixed it.")
        assert w.active == "codex" and "usage limit" in w.fallback_reason
        assert switches == [("codex", LIMIT)]
        # Failed primary sits out during cooldown: second job goes straight to codex.
        jid2 = await w.submit("next job")
        events2 = await asyncio.wait_for(next_worker_events(w), 1)
        assert events2[-1] == WorkerDone(jid2, True, "Second job done.")
        assert len(made[("claude", "")].submitted) == 1
        assert len(made[("codex", "")].submitted) == 2
    finally:
        await w.close()


@pytest.mark.asyncio
async def test_worker_preferred_model_is_retried_after_cooldown():
    made: dict[tuple[str, str], ScriptedWorker] = {}

    def make(provider: str, model: str) -> ScriptedWorker:
        runs = [[("done", False, LIMIT)]] if (provider, model) == ("claude", "") else [[("done", True, "backup ok")]]
        w = ScriptedWorker(runs)
        made[(provider, model)] = w
        return w

    w = FallbackWorker([("claude", ""), ("claude", "haiku")], make, cooldown_s=0.0)
    await w.start()
    try:
        await w.submit("first")
        done1 = (await asyncio.wait_for(next_worker_events(w), 1))[-1]
        assert done1.ok and done1.summary == "backup ok" and w.active == "claude:haiku"
        made[("claude", "")].runs.append([("done", True, "primary back")])
        await w.submit("second")
        done2 = (await asyncio.wait_for(next_worker_events(w), 1))[-1]
        assert done2.ok and done2.summary == "primary back"
        assert w.active == "claude" and w.fallback_reason == ""
    finally:
        await w.close()


@pytest.mark.asyncio
async def test_worker_that_already_acted_is_not_retried():
    made: dict[tuple[str, str], ScriptedWorker] = {}

    def make(provider: str, model: str) -> ScriptedWorker:
        runs = [[("progress", "Editing files"), ("done", False, LIMIT)]] if provider == "claude" else [[("done", True, "never")]]
        w = ScriptedWorker(runs)
        made[(provider, model)] = w
        return w

    w = FallbackWorker([("claude", ""), ("codex", "")], make)
    await w.start()
    try:
        jid = await w.submit("edit file")
        events = await asyncio.wait_for(next_worker_events(w), 1)
        assert events[-1] == WorkerDone(jid, False, LIMIT)
        assert ("codex", "") not in made
    finally:
        await w.close()


@pytest.mark.asyncio
async def test_worker_that_cannot_start_is_skipped():
    def make(provider: str, model: str) -> ScriptedWorker:
        if provider == "codex":
            return ScriptedWorker([], fail_start="codex CLI is not installed")
        return ScriptedWorker([[("done", True, "opencode handled it")]])

    w = FallbackWorker([("codex", ""), ("opencode", "")], make)
    await w.start()
    try:
        assert w.active == "opencode" and "not installed" in w.fallback_reason
        jid = await w.submit("check logs")
        done = (await asyncio.wait_for(next_worker_events(w), 1))[-1]
        assert done == WorkerDone(jid, True, "opencode handled it")
    finally:
        await w.close()


@pytest.mark.asyncio
async def test_worker_cancel_does_not_fall_back():
    primary = FakeWorker()
    backup_made = []

    def make(provider: str, model: str):
        if provider == "claude":
            return primary
        backup_made.append((provider, model))
        return FakeWorker()

    w = FallbackWorker([("claude", ""), ("codex", "")], make)
    await w.start()
    try:
        jid = await w.submit("slow job")
        while not primary.submitted:
            await asyncio.sleep(0.01)
        await w.cancel(jid)
        primary.push(WorkerDone("job1", False, "Stopped."))
        done = (await asyncio.wait_for(next_worker_events(w), 1))[-1]
        assert done == WorkerDone(jid, False, "Stopped.")
        assert backup_made == []
    finally:
        await w.close()


def test_usable_worker_fallbacks_and_create_worker_chain(monkeypatch):
    from omi import brain
    from omi.config import Config

    cfg = Config()
    cfg.brain.worker = "claude"
    cfg.brain.claude.worker_model = "sonnet"
    cfg.brain.worker_fallbacks = ["claude:sonnet", "claude:haiku", "codex:gpt-5.4", "local", "missing"]
    monkeypatch.setattr(
        "omi.brain.cli_providers.available_providers",
        lambda c: [
            {"name": "claude", "installed": True},
            {"name": "codex", "installed": True},
            {"name": "local", "installed": True},
            {"name": "missing", "installed": False},
        ],
    )
    assert brain.usable_worker_fallbacks(cfg) == [("claude", "haiku"), ("codex", "gpt-5.4")]
    w = brain.create_worker_chain(cfg)
    assert isinstance(w, FallbackWorker)
    assert w.entries == [("claude", ""), ("claude", "haiku"), ("codex", "gpt-5.4")]
    c_haiku = brain.with_worker_model(cfg, "claude", "haiku")
    assert c_haiku.brain.claude.worker_model == "haiku"
    c_codex = brain.with_worker_model(cfg, "codex", "gpt-5.4")
    assert c_codex.brain.worker == "codex" and c_codex.brain.codex.worker_model == "gpt-5.4"
