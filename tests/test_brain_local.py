"""LocalBrain against a fake OpenAI-compatible server that streams SSE, plus live
turns against the real llama-server (marked live)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from hark.brain.base import VOICE_STYLE
from hark.brain.llama_server import LlamaServer
from hark.brain.local import MAX_ROUNDS, REFUSED_REPLY, LocalBrain, SpeechFilter
from hark.config import LocalLlmConfig
from hark.events import TextDelta, ToolEnd, ToolStart, TurnContext, TurnDone
from hark.tools.base import Tool, ToolRegistry, ToolResult


class FakeServer:
    """Tiny HTTP/1.1 server: /health plus a streaming /v1/chat/completions."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.status = 200
        self.server: asyncio.base_events.Server | None = None

    async def __aenter__(self) -> FakeServer:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        return self

    async def __aexit__(self, *exc) -> None:
        self.server.close()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = (await reader.readuntil(b"\r\n\r\n")).decode()
            request_line, *header_lines = head.split("\r\n")
            headers = {k.lower(): v.strip() for k, _, v in
                       (h.partition(":") for h in header_lines if h)}
            body = await reader.readexactly(int(headers.get("content-length", 0)))
            if request_line.startswith("GET /health"):
                return await self._send(writer, 200, b'{"status":"ok"}')
            req = json.loads(body)
            self.requests.append(req)
            self.headers.append(headers)
            if self.status != 200:
                return await self._send(writer, self.status, b'{"error":"boom"}')
            if not req.get("stream"):
                return await self._send(writer, 200, json.dumps(
                    {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode())
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Connection: close\r\n\r\n")
            for delta, pause in self._script(req):
                chunk = {"choices": [{"index": 0, "delta": delta}]}
                writer.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await writer.drain()
                await asyncio.sleep(pause)
            writer.write(b"data: [DONE]\n\n")
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def _send(self, writer: asyncio.StreamWriter, status: int, body: bytes) -> None:
        writer.write(f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n"
                     f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
        await writer.drain()

    def _script(self, req: dict[str, Any]) -> list[tuple[dict, float]]:
        last = req["messages"][-1]
        text = last.get("content") or ""
        looping = any("loop" in (m.get("content") or "") for m in req["messages"])
        if last["role"] == "tool" and not looping:
            return [({"content": "Volume set "}, 0), ({"content": "to thirty."}, 0)]
        if "slow" in text:
            return [({"content": "word "}, 0.05)] * 100
        if "volume" in text or looping:
            return [
                ({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                  "function": {"name": "set_volume", "arguments": ""}}]}, 0),
                ({"tool_calls": [{"index": 0, "function": {"arguments": '{"perc'}}]}, 0),
                ({"tool_calls": [{"index": 0, "function": {"arguments": 'ent": 30}'}}]}, 0),
            ]
        return [({"content": "<think>hmm</think>Hel"}, 0), ({"content": "lo **the"}, 0),
                ({"content": "re**."}, 0)]


def make_registry(calls: list[dict]) -> ToolRegistry:
    async def set_volume(percent: int) -> ToolResult:
        calls.append({"percent": percent})
        return ToolResult(True, f"Volume is {percent} percent.")

    reg = ToolRegistry()
    reg.register(Tool("set_volume", "Set output volume percent",
                      {"type": "object", "properties": {"percent": {"type": "integer"}},
                       "required": ["percent"]}, "reversible", set_volume))
    return reg


async def run(brain: LocalBrain, text: str, ctx: TurnContext | None = None) -> list:
    return [ev async for ev in brain.turn(text, ctx or TurnContext("s"))]


@pytest.fixture
async def setup():
    async with FakeServer() as server:
        calls: list[dict] = []
        brain = LocalBrain(LocalLlmConfig(base_url=server.url), make_registry(calls))
        await brain.start()
        yield server, brain, calls
        await brain.close()


def test_speech_filter_handles_split_tags():
    f = SpeechFilter()
    out = f.feed("<thi") + f.feed("nk>secret</th") + f.feed("ink>**Hi** `x` <") + f.flush()
    assert out == "Hi x <"


async def test_chat_reply_is_cleaned(setup):
    server, brain, _ = setup
    events = await run(brain, "hello")
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Hello there."
    assert events[-1] == TurnDone()
    assert sum(isinstance(e, TurnDone) for e in events) == 1


async def test_warmup_uses_the_same_prefix(setup):
    server, brain, _ = setup
    await run(brain, "hello")
    warm, turn = server.requests[0], server.requests[1]
    assert warm["stream"] is False and warm["max_tokens"] == 1
    assert warm["messages"][0] == turn["messages"][0]
    assert warm["tools"] == turn["tools"]
    assert warm["chat_template_kwargs"] == {"enable_thinking": False}


async def test_tool_call_runs_and_result_is_fed_back(setup):
    server, brain, calls = setup
    events = await run(brain, "set the volume to 30")
    kinds = [type(e).__name__ for e in events]
    assert kinds[:2] == ["ToolStart", "ToolEnd"] and kinds[-1] == "TurnDone"
    assert events[0] == ToolStart("call_1", "set_volume", {"percent": 30})
    assert events[1] == ToolEnd("call_1", "set_volume", True, "Volume is 30 percent.")
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Volume set to thirty."
    assert calls == [{"percent": 30}]
    follow = server.requests[-1]["messages"]
    assert follow[-2]["tool_calls"][0]["function"] == {"name": "set_volume",
                                                       "arguments": '{"percent": 30}'}
    assert follow[-1] == {"role": "tool", "tool_call_id": "call_1",
                          "content": "Volume is 30 percent."}


async def test_prefix_is_byte_stable_and_recall_rides_on_user_message(setup):
    server, brain, _ = setup
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."}]
    await run(brain, "hello", TurnContext("s", "Profile: likes tea.", ["fact one"], history))
    await run(brain, "hello", TurnContext("s", "Profile: likes tea.", ["fact two"], history))
    a, b = server.requests[-2], server.requests[-1]
    expected = VOICE_STYLE.format(name="Hark") + "\n\nProfile: likes tea."
    assert a["messages"][0]["content"] == expected
    assert json.dumps(a["messages"][:-1]) == json.dumps(b["messages"][:-1])
    assert json.dumps(a["tools"]) == json.dumps(b["tools"])
    assert b["messages"][-1]["content"] == "hello\n\n(Notes from memory that may help: fact two)"


async def test_tool_loop_stops_after_max_rounds(setup):
    server, brain, calls = setup
    events = await run(brain, "loop forever")
    assert len(server.requests) == 1 + MAX_ROUNDS  # warm-up plus rounds
    assert "tools" not in server.requests[-1]  # the last round must answer in words
    assert len(calls) == MAX_ROUNDS - 1
    assert sum(isinstance(e, TurnDone) for e in events) == 1


async def test_no_second_ask_after_the_user_says_no(setup):
    server, brain, calls = setup
    brain.registry.tools["set_volume"].safety = "destructive"
    asked: list[str] = []

    async def say_no(question: str) -> bool:
        asked.append(question)
        return False

    brain.registry.confirm = say_no
    events = await run(brain, "loop forever")  # the fake model would call the tool every round
    assert len(asked) == 1 and calls == []
    assert len(server.requests) == 2  # warm-up and the call; the reply is fixed
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == REFUSED_REPLY


async def test_cancel_mid_reply_is_fast(setup):
    server, brain, _ = setup
    gen = brain.turn("slow please", TurnContext("s"))
    assert isinstance(await anext(gen), TextDelta)
    t0 = time.perf_counter()
    await brain.cancel()
    rest = [ev async for ev in gen]
    assert rest[-1] == TurnDone(cancelled=True)
    assert time.perf_counter() - t0 < 0.1
    events = await run(brain, "hello")  # the brain is usable right after
    assert events[-1] == TurnDone()


async def test_http_error_ends_turn_with_error(setup):
    server, brain, _ = setup
    server.status = 500
    events = await run(brain, "hello")
    assert len(events) == 1 and events[0].error and "500" in events[0].error


async def test_api_key_is_read_from_named_env_at_call_time(monkeypatch):
    async with FakeServer() as server:
        brain = LocalBrain(LocalLlmConfig(base_url=server.url + "/v1", api_key_env="HARK_TEST_KEY"),
                           ToolRegistry())
        await brain.start()
        monkeypatch.setenv("HARK_TEST_KEY", "test-value")
        await run(brain, "hello")
        await brain.close()
    assert "authorization" not in server.headers[0]
    assert server.headers[1]["authorization"] == "Bearer test-value"


async def test_llama_server_reuses_a_server_already_listening():
    async with FakeServer() as server:
        ls = LlamaServer(LocalLlmConfig(port=server.port), binary="/nonexistent/llama-server")
        await ls.start()
        assert not ls.owned and ls.base_url == server.url
        await ls.close()


# --- live: the real llama-server on port 47910 ---


async def _healthy(url: str) -> bool:
    try:
        async with httpx.AsyncClient() as c:
            return (await c.get(f"{url}/health", timeout=1)).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.live
async def test_live_local_brain():
    calls: list[dict] = []
    cfg = LocalLlmConfig()
    brain = LocalBrain(cfg, make_registry(calls))
    t0 = time.perf_counter()
    await brain.start()
    startup = time.perf_counter() - t0
    try:
        async def first_of(text: str, kind: type) -> tuple[float | None, list]:
            t, first, events = time.perf_counter(), None, []
            async for ev in brain.turn(text, TurnContext("live")):
                if first is None and isinstance(ev, kind):
                    first = time.perf_counter() - t
                events.append(ev)
            return first, events

        await first_of("Hi there.", TextDelta)  # one more warm turn
        chat_s, chat = await first_of("What is the capital of France?", TextDelta)
        tool_s, tooled = await first_of("Set the volume to 30 percent.", ToolStart)
        gen = brain.turn("Tell me a long story about a lighthouse keeper.", TurnContext("live"))
        async for ev in gen:
            if isinstance(ev, TextDelta):
                break
        t = time.perf_counter()
        await brain.cancel()
        rest = [ev async for ev in gen]
        cancel_s = time.perf_counter() - t
    finally:
        owned = brain._server.owned
        await brain.close()
    chat_text = "".join(e.text for e in chat if isinstance(e, TextDelta))
    tool_text = "".join(e.text for e in tooled if isinstance(e, TextDelta))
    print(f"\nLIVE local startup={startup*1000:.0f}ms owned={owned} "
          f"first_text={chat_s*1000:.0f}ms first_tool_start={tool_s*1000:.0f}ms "
          f"cancel={cancel_s*1000:.1f}ms chat={chat_text!r} tool_reply={tool_text!r} calls={calls} tool_turn={[e for e in tooled if not isinstance(e, TextDelta)]}")
    assert "paris" in chat_text.lower()
    starts = [e for e in tooled if isinstance(e, ToolStart)]
    assert starts and starts[0].name == "set_volume" and starts[0].args == {"percent": 30}
    assert rest[-1] == TurnDone(cancelled=True) and cancel_s < 0.1
    assert not await _healthy(f"http://127.0.0.1:{cfg.port}") or not owned
