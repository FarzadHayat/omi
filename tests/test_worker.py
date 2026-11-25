"""ClaudeWorker against the fake `claude`, plus one tiny live job (marked live)."""

from __future__ import annotations

import asyncio
import shutil
import time

import pytest
from test_brain_claude import fake_claude  # noqa: F401  (fixture)

from hark.brain import create_worker
from hark.brain.claude_session import hark_mcp_available
from hark.config import Config
from hark.events import WorkerDone, WorkerProgress
from hark.worker import ClaudeWorker, delegate_tool, progress_phrase


async def next_events(worker: ClaudeWorker, until_done: int = 1, timeout: float = 10) -> list:
    out, gen = [], worker.events()
    async with asyncio.timeout(timeout):
        while sum(isinstance(e, WorkerDone) for e in out) < until_done:
            out.append(await anext(gen))
    return out


@pytest.fixture
async def worker(fake_claude):  # noqa: F811
    cmd, _ = fake_claude
    w = ClaudeWorker(Config(), claude_cmd=cmd, use_mcp=False)
    await w.start()
    yield w
    await w.close()


def test_create_worker_off_returns_none():
    cfg = Config()
    cfg.brain.worker = "off"
    assert create_worker(cfg) is None


def test_progress_phrases():
    assert progress_phrase("Bash") == "Running a command"
    assert progress_phrase("WebSearch") == "Searching the web"
    assert progress_phrase("Read") == "Reading files"
    assert progress_phrase("mcp__hark__set_volume") == "Using the computer"


async def test_job_reports_rate_limited_progress_and_summary(worker):
    t0 = time.perf_counter()
    job = await worker.submit("use a tool three times")
    assert time.perf_counter() - t0 < 0.01  # submit never waits for the job
    events = await next_events(worker)
    assert events == [WorkerProgress(job, "Using the computer"),
                      WorkerDone(job, True, "I did it. All good.")]


async def test_jobs_run_one_at_a_time_in_order(worker):
    a = await worker.submit("hello")
    b = await worker.submit("hello again")
    events = await next_events(worker, until_done=2)
    assert [e.job_id for e in events if isinstance(e, WorkerDone)] == [a, b]


async def test_cancel_running_and_queued_jobs(worker):
    a = await worker.submit("slow job")
    b = await worker.submit("hello")
    await asyncio.sleep(0.2)
    await worker.cancel()
    events = await next_events(worker, until_done=2)
    assert WorkerDone(b, False, "Cancelled.") in events
    assert WorkerDone(a, False, "Stopped.") in events
    c = await worker.submit("hello")  # the queue keeps working
    events = await next_events(worker)
    assert events[-1] == WorkerDone(c, True, "Hello there.")


async def test_delegate_tool(worker):
    tool = delegate_tool(worker)
    assert tool.name == "delegate" and tool.safety == "reversible"
    assert tool.parameters["required"] == ["task"]
    res = await tool.fn(task="hello")
    assert res.ok and res.text == "Started. I will tell you when it is done."
    events = await next_events(worker)
    assert events[-1] == WorkerDone(res.data["job_id"], True, "Hello there.")


@pytest.mark.live
@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")
async def test_live_worker_tiny_job():
    cfg = Config()
    cfg.brain.claude.worker_model = "haiku"
    w = ClaudeWorker(cfg, use_mcp=hark_mcp_available())
    t0 = time.perf_counter()
    await w.start()
    startup = time.perf_counter() - t0
    try:
        t = time.perf_counter()
        job = await w.submit("What is 2 plus 2? Do not use any tools.")
        events = await next_events(w, timeout=90)
        total = time.perf_counter() - t
    finally:
        await w.close()
    print(f"\nLIVE worker startup={startup*1000:.0f}ms job={total*1000:.0f}ms events={events}")
    done = events[-1]
    assert done.job_id == job and done.ok
    assert "4" in done.summary or "four" in done.summary.lower()
