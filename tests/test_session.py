import asyncio
import json
import statistics
import time

import numpy as np
import pytest

from hark.events import State, TextDelta, ToolEnd, ToolStart, TurnDone, WorkerDone, WorkerProgress
from hark.memory.base import Hit
from hark.session import CANCELLED_NOTE, HISTORY_LIMIT, NO_AUDIO, Session, parse_answer, parse_yes_no
from tests.fakes import (
    CaptureOut,
    Events,
    FakeBrain,
    FakeMemory,
    FakeRouter,
    FakeSTT,
    FakeTTS,
    FakeWorker,
    Route,
    make_registry,
)


def make(tmp_path, brain=None, *, tts=None, out=None, routes=None, worker=None, memory=None, **kw):
    events = Events()
    calls: list = []
    reg = make_registry(calls)
    s = Session(
        stt=FakeSTT("what time is it"),
        tts=tts or FakeTTS(),
        brain=brain or FakeBrain(),
        registry=reg,
        router=FakeRouter(routes),
        out=out or CaptureOut(speed=50),
        memory=memory,
        worker=worker,
        emit=events,
        state_dir=tmp_path,
        **kw,
    )
    reg.confirm = s.confirm
    s.calls = calls
    return s, events


async def wait_for(pred, timeout=2.0):
    end = time.perf_counter() + timeout
    while not pred():
        if time.perf_counter() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.002)


async def test_first_audio_before_brain_finishes(tmp_path):
    brain = FakeBrain([
        (0.0, TextDelta("Sure, ")),
        (0.0, TextDelta("I can help with that. ")),
        (0.3, TextDelta("Here is more. ")),
        (0.0, TurnDone()),
    ])
    out = CaptureOut(speed=50)
    s, ev = make(tmp_path, brain, tts=FakeTTS(delay=0.01), out=out)
    res = await s.run_turn("help me")
    done_at = brain.yielded[-1][0]
    assert out.writes[0][0] < done_at - 0.25
    assert res.timing["path"] == "talker"
    assert res.timing["first_audio_ms"] < 100
    assert res.reply == "Sure, I can help with that. Here is more."
    assert ev.states() == ["thinking", "speaking", "idle"]
    assert ev.kinds("transcript")[-1] == {"event": "transcript", "role": "assistant", "text": res.reply, "final": True}


async def test_clause_two_synthesised_while_clause_one_plays(tmp_path):
    brain = FakeBrain([(0.0, TextDelta("One two three, four five six seven. Eight nine. Ten.")), (0.0, TurnDone())])
    tts = FakeTTS(delay=0.03, seconds_per_word=0.1)
    out = CaptureOut(speed=1)
    s, _ = make(tmp_path, brain, tts=tts, out=out)
    await s.run_turn("count")
    assert [t for _, t in tts.calls] == ["One two three,", "four five six seven.", "Eight nine.", "Ten."]
    # clause 2 synthesis started and its audio was queued before clause 1 finished playing
    assert tts.calls[1][0] < out.play_end(0)
    assert out.writes[1][0] < out.play_end(0)
    # and playback ran back to back: drain finished about when the audio ends
    assert out.drains[-1] >= out.play_end(len(out.writes) - 1) - 0.01


async def test_pipeline_overhead_zero_delay(tmp_path):
    """Time from the brain's first TextDelta to the first AudioOut.write."""
    samples = []
    for _ in range(30):
        brain = FakeBrain([(0.0, TextDelta("Hello there, friend.")), (0.0, TurnDone())])
        out = CaptureOut(speed=1000)
        s, _ = make(tmp_path, brain, out=out)
        await s.run_turn("hi")
        samples.append((out.writes[0][0] - brain.yielded[0][0]) * 1000)
    med = statistics.median(samples)
    print(f"\npipeline overhead TextDelta->first write: median {med:.2f} ms, max {max(samples):.2f} ms")
    assert med < 10


async def test_router_path_skips_brain(tmp_path):
    brain = FakeBrain()
    routes = {"volume up": Route("set_volume", {"level": 60}, "Volume up.")}
    s, ev = make(tmp_path, brain, routes=routes)
    res = await s.run_turn("Volume up.")
    assert brain.turns == []
    assert s.calls == [("set_volume", {"level": 60})]
    assert res.reply == "Volume up."
    assert res.timing["path"] == "router"
    assert res.timing["action_ms"] is not None and res.timing["first_audio_ms"] is not None
    tools = ev.kinds("tool")
    assert [t["phase"] for t in tools] == ["start", "end"] and tools[1]["ok"] is True
    assert "acting" in ev.states()


async def test_router_empty_ack_and_failure_speak_result(tmp_path):
    routes = {
        "set it": Route("set_volume", {"level": 5}, ""),
        "break": Route("broken", {}, "Done."),
    }
    s, _ = make(tmp_path, routes=routes)
    assert (await s.run_turn("set it")).reply == "Volume is 5 percent."
    assert (await s.run_turn("break")).reply == "broken failed: boom"


async def slow_brain():
    return FakeBrain([
        (0.0, TextDelta("First part, ")),
        (0.0, TextDelta("then a sentence. ")),
        (5.0, TextDelta("Never said. ")),
        (0.0, TurnDone()),
    ])


async def test_stop_cancels_within_100ms(tmp_path):
    lat = []
    for _ in range(10):
        brain = await slow_brain()
        out = CaptureOut(speed=1)
        s, ev = make(tmp_path, brain, out=out)
        task = asyncio.create_task(s.run_turn("talk"))
        await wait_for(lambda: out.writes)
        t = time.perf_counter()
        await s.stop()
        await wait_for(lambda: s.state is State.IDLE)
        idle_at = ev.first_time(lambda e: e.get("state") == "idle" and True)
        idle_events = [tt for tt, e in ev.items if e.get("state") == "idle" and tt >= t]
        lat.append((idle_events[0] - t) * 1000)
        res = await task
        assert res.cancelled
        assert res.reply == "First part, then a sentence."
        assert brain.cancels >= 1 and out.stops
        assert idle_at is not None
    print(f"\ncancel latency stop()->idle: median {statistics.median(lat):.2f} ms, max {max(lat):.2f} ms")
    assert max(lat) < 100


async def test_stop_during_playback_after_brain_done(tmp_path):
    brain = FakeBrain([(0.0, TextDelta("A long reply that plays for a while.")), (0.0, TurnDone())])
    out = CaptureOut(speed=1)
    s, _ = make(tmp_path, brain, tts=FakeTTS(seconds_per_word=0.5), out=out)
    task = asyncio.create_task(s.run_turn("go"))
    await wait_for(lambda: out.writes)
    await asyncio.sleep(0.05)
    t = time.perf_counter()
    await s.stop()
    res = await task
    assert (time.perf_counter() - t) < 0.1
    assert res.cancelled and out.stops
    assert s.state is State.IDLE


async def test_barge_in_new_turn_cancels_running_one(tmp_path):
    brain = FakeBrain(lambda text: [
        (0.0, TextDelta(f"Answer to {text}. ")),
        (5.0 if text == "first" else 0.0, TextDelta("More. ")),
        (0.0, TurnDone()),
    ])
    out = CaptureOut(speed=1)
    s, ev = make(tmp_path, brain, out=out)
    first = asyncio.create_task(s.run_turn("first"))
    await wait_for(lambda: out.writes)
    t = time.perf_counter()
    second = await s.run_turn("second")
    r1 = await first
    assert r1.cancelled and r1.reply == "Answer to first."
    assert not second.cancelled and second.reply == "Answer to second. More."
    assert time.perf_counter() - t < 1.0
    assert s.state is State.IDLE
    # history has both turns, cancelled one with what was spoken
    assert [m["content"] for m in s.history] == ["first", "Answer to first.", "second", "Answer to second. More."]


async def confirm_setup(tmp_path, **kw):
    routes = {"close it": Route("close_window", {}, "Closed.")}
    s, ev = make(tmp_path, routes=routes, **kw)
    task = asyncio.create_task(s.run_turn("close it"))
    await wait_for(lambda: ev.kinds("confirm_request"))
    assert s.state is State.CONFIRMING
    return s, ev, task


async def test_confirm_by_command(tmp_path):
    s, ev, task = await confirm_setup(tmp_path)
    assert ev.kinds("confirm_request")[0]["question"] == "Close the window?"
    assert s.answer(True)
    res = await task
    assert res.reply == "Closed." and s.calls == [("close_window", {})]  # the question is not in the reply
    assert not s.answer(True)  # nothing pending now


async def test_confirm_by_command_no(tmp_path):
    s, _, task = await confirm_setup(tmp_path)
    s.answer(False)
    res = await task
    assert "Cancelled, not confirmed." in res.reply and s.calls == []


@pytest.mark.parametrize("said,ok", [("yeah do it", True), ("Go ahead.", True), ("nope", False), ("wait", False)])
async def test_confirm_by_voice(tmp_path, said, ok):
    s, _, task = await confirm_setup(tmp_path)
    ans = await s.run_turn(said)
    assert ans.timing["path"] == "confirm" and ans.reply == ""
    res = await task
    assert (s.calls == [("close_window", {})]) is ok
    assert res.cancelled is False


async def test_confirm_by_voice_audio(tmp_path):
    s, _, task = await confirm_setup(tmp_path)
    s.stt = FakeSTT("yes please")
    await s.run_turn(pcm=np.zeros(1600, np.float32))
    await task
    assert s.calls == [("close_window", {})]


async def test_confirm_timeout_means_no(tmp_path):
    s, _, task = await confirm_setup(tmp_path, confirm_timeout=0.1)
    res = await task
    assert s.calls == [] and "Cancelled" in res.reply
    assert s.state is State.IDLE


async def test_confirm_when_idle_speaks_question(tmp_path):
    tts = FakeTTS()
    s, ev = make(tmp_path, tts=tts)
    task = asyncio.create_task(s.confirm("Delete the file?"))
    await wait_for(lambda: ev.kinds("confirm_request"))
    await wait_for(lambda: tts.calls)
    assert tts.calls[0][1] == "Delete the file?"
    s.answer(True)
    assert await task is True
    await wait_for(lambda: s.state is State.IDLE)


def test_parse_yes_no():
    for t in ["yes", "Yeah.", "yep", "do it", "go ahead", "confirm", "yes, go ahead"]:
        assert parse_yes_no(t), t
    for t in ["no", "nope", "cancel", "stop", "do not", "don't do it", "yes no", "sure", "", "yesterday"]:
        assert not parse_yes_no(t), t


async def test_worker_progress_rate_limited_and_only_when_idle(tmp_path):
    worker = FakeWorker()
    tts = FakeTTS()
    s, ev = make(tmp_path, tts=tts, worker=worker, progress_interval=8)
    await s.start()
    worker.push(WorkerProgress("j1", "Reading files"))
    await wait_for(lambda: tts.calls)
    await wait_for(lambda: s.state is State.IDLE)
    worker.push(WorkerProgress("j1", "Still reading"))
    await wait_for(lambda: len(ev.kinds("progress")) == 2)
    await asyncio.sleep(0.05)
    assert [t for _, t in tts.calls] == ["Reading files"]
    await s.close()


async def test_worker_progress_not_spoken_during_turn(tmp_path):
    worker = FakeWorker()
    tts = FakeTTS()
    brain = FakeBrain([(0.0, TextDelta("Working on it. ")), (0.2, TurnDone())])
    s, ev = make(tmp_path, brain, tts=tts, worker=worker)
    await s.start()
    task = asyncio.create_task(s.run_turn("hi"))
    await wait_for(lambda: tts.calls)
    worker.push(WorkerProgress("j1", "Reading files"))
    await wait_for(lambda: ev.kinds("progress"))
    await task
    await asyncio.sleep(0.05)
    assert [t for _, t in tts.calls] == ["Working on it."]
    await s.close()


async def test_worker_done_spoken_when_idle(tmp_path):
    worker = FakeWorker()
    tts = FakeTTS()
    brain = FakeBrain([(0.0, TextDelta("Sure. ")), (0.2, TurnDone())])
    s, ev = make(tmp_path, brain, tts=tts, worker=worker)
    await s.start()
    task = asyncio.create_task(s.run_turn("hi"))
    await wait_for(lambda: tts.calls)
    worker.push(WorkerDone("j1", True, "The report is ready."))
    await wait_for(lambda: ev.kinds("progress"))
    assert [t for _, t in tts.calls] == ["Sure."]  # not while the turn runs
    await task
    await wait_for(lambda: len(tts.calls) == 2)
    assert tts.calls[1][1] == "The report is ready."
    assert ev.kinds("progress")[0]["done"] is True
    # idle: spoken at once
    worker.push(WorkerDone("j2", False, "The build failed."))
    await wait_for(lambda: len(tts.calls) == 3)
    await s.close()


async def test_idle_ends_session(tmp_path):
    ended = []

    async def on_end(sid, text):
        ended.append((sid, text))

    s, _ = make(tmp_path, idle_timeout=0.1, on_session_end=on_end)
    await s.start()
    old = s.session_id
    await s.run_turn("hello")
    await wait_for(lambda: ended, timeout=1)
    assert ended[0][0] == old and "user: hello" in ended[0][1] and "assistant: Okay." in ended[0][1]
    assert s.session_id != old and s.history == [] and s.transcript == []
    await asyncio.sleep(0.3)
    assert len(ended) == 1  # an empty session is not ended again
    await s.close()


async def test_history_memory_trace_and_stt(tmp_path):
    memory = FakeMemory(facts=[Hit("fact", "dog", "Sam has a dog named Rex.", "facts/dog.md")])
    brain = FakeBrain()
    s, ev = make(tmp_path, brain, memory=memory)
    res = await s.run_turn(pcm=np.zeros(1600, np.float32))
    assert res.user_text == "what time is it" and res.timing["stt_ms"] is not None
    await s.run_turn("tell me about my dog")
    ctx = brain.turns[-1][1]
    assert ctx.recalled == ["Sam has a dog named Rex."] and ctx.memory_context == memory.profile
    assert ctx.history == [
        {"role": "user", "content": "what time is it"},
        {"role": "assistant", "content": "Okay."},
    ]
    for i in range(10):
        await s.run_turn(f"turn {i}")
    assert len(s.history) == HISTORY_LIMIT
    await wait_for(lambda: len(memory.logged) == 24)
    lines = (tmp_path / "turns.jsonl").read_text().splitlines()
    await wait_for(lambda: len((tmp_path / "turns.jsonl").read_text().splitlines()) == 12)
    rec = json.loads(lines[0])
    assert set(rec) >= {"stt_ms", "route_ms", "first_text_ms", "first_audio_ms", "action_ms", "total_ms", "path"}
    assert ev.kinds("timing")[0]["path"] == "talker"


async def test_brain_tool_events_and_error(tmp_path):
    brain = FakeBrain([
        (0.0, ToolStart("1", "set_volume", {"level": 3})),
        (0.0, ToolEnd("1", "set_volume", True, "Volume is 3.")),
        (0.0, TextDelta("Done.")),
        (0.0, TurnDone()),
    ])
    s, ev = make(tmp_path, brain)
    res = await s.run_turn("quiet please")
    assert [t["phase"] for t in ev.kinds("tool")] == ["start", "end"]
    assert res.timing["action_ms"] is not None and res.reply == "Done."
    s.brain = FakeBrain([(0.0, TurnDone(error="model down"))])
    res = await s.run_turn("hi")
    assert res.reply == "Sorry, something went wrong." and res.error == "model down"


async def test_quiet_turn_skips_tts(tmp_path):
    tts = FakeTTS()
    out = CaptureOut()
    s, _ = make(tmp_path, tts=tts, out=out)
    res = await s.run_turn("hi", quiet=True)
    assert res.reply == "Okay." and tts.calls == [] and out.writes == []
    assert res.timing["first_audio_ms"] is None


async def test_stop_bounded_when_brain_cancel_is_slow(tmp_path):
    brain = await slow_brain()

    async def slow_cancel():
        await asyncio.sleep(0.5)

    brain.cancel = slow_cancel
    out = CaptureOut(speed=1)
    s, _ = make(tmp_path, brain, out=out)
    task = asyncio.create_task(s.run_turn("talk"))
    await wait_for(lambda: out.writes)
    t = time.perf_counter()
    await s.stop()
    took = (time.perf_counter() - t) * 1000
    res = await task
    assert took < 100 and res.cancelled and s.state is State.IDLE


# --- fixes ---


@pytest.mark.parametrize(("said", "want"), [
    ("yes", True), ("Yes please.", True), ("yeah, go ahead", True), ("do it", True),
    ("Can you confirm what that deletes first?", None), ("What happens if I do it?", None),
    ("Yeah, wait, which window?", None), ("Hmm, yeah I guess not", False),
    ("Hold on, is that the right folder? Yep, the old one, right?", None), ("Yes, but only the second one", None),
    ("no", False), ("don't", False), ("not now", False), ("wait", False), ("hmm maybe", None),
    ("what is the weather", None), ("yes yes yes do it now please", None), ("", None),
])
def test_parse_answer_is_strict(said, want):
    assert parse_answer(said) is want


async def test_new_request_during_a_question_cancels_it_and_still_runs(tmp_path):
    brain = FakeBrain()
    s, ev, task = await confirm_setup(tmp_path, brain=brain)
    tts = s.tts
    res = await s.run_turn("what is the weather")
    first = await task
    assert s.calls == [] and first.cancelled  # the window was not closed
    assert [t for t, _ in brain.turns] == ["what is the weather"] and res.reply == "Okay."
    assert CANCELLED_NOTE in [t for _, t in tts.calls]  # said aloud, but not part of the reply


async def test_unclear_answer_asks_again_then_counts_as_no(tmp_path):
    tts = FakeTTS()
    s, ev, task = await confirm_setup(tmp_path, tts=tts)
    await s.run_turn("hmm maybe")
    await wait_for(lambda: any("Please say yes or no" in t for _, t in tts.calls))
    assert s.pending_confirm is not None
    await s.run_turn("uh")
    res = await task
    assert s.calls == [] and "Cancelled" in res.reply


async def test_answer_needs_the_current_question_id(tmp_path):
    s, ev, task = await confirm_setup(tmp_path)
    qid = ev.kinds("confirm_request")[0]["id"]
    assert not s.answer(True, "stale-id")
    assert s.answer(True, qid)
    await task
    assert s.calls == [("close_window", {})]


async def test_yes_spoken_before_the_question_does_not_answer_it(tmp_path):
    tts = FakeTTS()
    s, ev, task = await confirm_setup(tmp_path, tts=tts)
    before = s._question.asked_at - 1.0
    await s.run_turn("yes", started=before)
    assert s.pending_confirm is not None and s.calls == []
    await s.run_turn("yes")
    await task
    assert s.calls == [("close_window", {})]


async def test_total_ms_includes_playback(tmp_path):
    brain = FakeBrain([(0.0, TextDelta("A reply that takes a while to play.")), (0.0, TurnDone())])
    s, _ = make(tmp_path, brain, tts=FakeTTS(seconds_per_word=0.05), out=CaptureOut(speed=1))
    res = await s.run_turn("hi")
    t = res.timing
    assert t["first_audio_ms"] <= t["total_ms"] and t["brain_done_ms"] <= t["total_ms"]
    assert t["total_ms"] - t["first_audio_ms"] > 200  # about 0.4 s of audio played in between


class CountingSTT:
    def __init__(self, delay):
        self.delay, self.active, self.peak, self.calls = delay, 0, 0, 0
        self.lock = __import__("threading").Lock()

    def transcribe(self, pcm):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return "what time is it"


async def test_barge_in_does_not_stack_transcriptions(tmp_path):
    s, _ = make(tmp_path)
    stt = s.stt = CountingSTT(0.3)
    pcm = np.zeros(1600, np.float32)
    tasks = []
    for _ in range(4):
        tasks.append(asyncio.create_task(s.run_turn(pcm=pcm)))
        await asyncio.sleep(0.05)
    results = await asyncio.gather(*tasks)
    assert stt.peak <= 2  # one stale transcription at most, beside the current one
    assert stt.calls < 4  # the ones still waiting when the user spoke again were dropped
    assert not results[-1].cancelled and results[-1].user_text == "what time is it"
    await s.close()


class FakeStreamingSTT:
    def __init__(self, decode_s=0.0):
        self.decode_s, self.fed, self.events = decode_s, 0, []

    def start(self):
        self.fed = 0
        self.events.append("start")

    def feed(self, pcm):
        time.sleep(self.decode_s)
        self.fed += len(pcm)
        self.events.append("feed")
        return ""

    def finish(self):
        self.events.append("finish")
        return "what time is it"

    def transcribe(self, pcm):
        raise AssertionError("streaming engines get chunks, not whole files")


async def test_streaming_utterance_decodes_while_talking(tmp_path):
    s, ev = make(tmp_path)
    stt = s.stt = FakeStreamingSTT(decode_s=0.02)
    await s.begin_utterance()
    assert s.state is State.LISTENING
    for _ in range(20):  # 20 chunks of 32 ms, decoded as they arrive
        s.feed_audio(np.zeros(512, np.float32))
        await asyncio.sleep(0.03)
    res = await s.end_utterance()
    assert stt.fed == 20 * 512 and stt.events[0] == "start" and stt.events[-1] == "finish"
    assert res.user_text == "what time is it" and res.timing["stt_ms"] < 100
    assert s.state is State.IDLE
    await s.close()


async def test_utterance_with_no_audio_is_an_error(tmp_path):
    s, _ = make(tmp_path)
    await s.begin_utterance()
    res = await s.end_utterance()
    assert res.error == NO_AUDIO and s.state is State.IDLE
    await s.close()


async def test_whole_engine_utterance_transcribes_all_chunks(tmp_path):
    s, _ = make(tmp_path)
    seen = []
    s.stt.transcribe = lambda pcm: seen.append(len(pcm)) or "hello there"
    await s.begin_utterance()
    for _ in range(3):
        s.feed_audio(np.zeros(1600, np.float32))
    res = await s.end_utterance()
    assert seen == [4800] and res.user_text == "hello there"
    await s.close()
