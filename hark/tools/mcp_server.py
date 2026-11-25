"""The hark MCP server (stdio): device tools for outside agents, plus "approve".

Run with "python -m hark.tools.mcp_server". Claude sees the tools as
mcp__hark__<name>. "approve" answers Claude Code's --permission-prompt-tool: it
returns one text block holding {"behavior": "allow"} or
{"behavior": "deny", "message": ...}, and no structured content.

Confirmation goes to the user through the hark daemon ("ask_confirm" on its
socket). No daemon means nobody can say yes, so the answer is no.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import Any

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from hark import config
from hark.tools import build_registry
from hark.tools.base import Tool, ToolRegistry, ToolResult
from hark.tools.safety import classify_claude_tool

ASK_TIMEOUT = 120.0  # the user has to hear the question and answer it
APPROVAL_TTL = 120.0

APPROVE_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string"},
        "input": {"type": "object"},
        "tool_use_id": {"type": "string"},
    },
    "required": ["tool_name", "input"],
}


async def ask_confirm(question: str, socket_path: str | None = None) -> bool:
    """Ask the user through the daemon. True only on a clear yes."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path or config.SOCKET_PATH)), 1.0
        )
    except (OSError, TimeoutError):
        return False
    try:
        writer.write((json.dumps({"cmd": "ask_confirm", "question": question}) + "\n").encode())
        await writer.drain()
        deadline = time.monotonic() + ASK_TIMEOUT
        while (left := deadline - time.monotonic()) > 0:
            line = await asyncio.wait_for(reader.readline(), left)
            if not line:
                return False
            msg = json.loads(line)
            if "event" in msg:
                continue
            return bool(msg.get("ok")) and msg.get("answer") is True
        return False
    except (OSError, TimeoutError, ValueError):
        return False
    finally:
        writer.close()


async def daemon_call(req: dict[str, Any], socket_path: str | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """One request to the daemon; its final (non-event) reply line."""
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(socket_path or config.SOCKET_PATH)), 1.0
    )
    try:
        writer.write((json.dumps(req) + "\n").encode())
        await writer.drain()
        while line := await asyncio.wait_for(reader.readline(), timeout):
            msg = json.loads(line)
            if "event" not in msg:
                return msg
        return {"ok": False, "error": "the daemon closed the connection"}
    finally:
        writer.close()


def daemon_tools() -> list[Tool]:
    """Tools that live in the daemon (memory and the background worker), so Claude can
    use them too. Built here only for their names and schemas; calls go over the socket."""
    from hark.memory import NullMemory
    from hark.memory.tools import memory_tools
    from hark.worker import delegate_tool

    def proxy(name: str):
        async def fn(**args: Any) -> ToolResult:
            try:
                reply = await daemon_call({"cmd": "tool", "name": name, "args": args})
            except (OSError, TimeoutError, ValueError) as exc:
                return ToolResult(False, f"The hark daemon is not reachable: {exc}")
            if not reply.get("ok"):
                return ToolResult(False, str(reply.get("error", "failed")))
            return ToolResult(bool(reply["result"]["ok"]), str(reply["result"]["text"]))

        return fn

    tools = [*memory_tools(NullMemory()), delegate_tool(None)]  # type: ignore[arg-type]
    return [replace(t, fn=proxy(t.name)) for t in tools]


class Approvals:
    """Questions the user already said yes to in "approve", so the tool call that
    follows does not ask a second time."""

    def __init__(self) -> None:
        self._granted: dict[str, float] = {}

    def grant(self, question: str) -> None:
        self._granted[question] = time.monotonic() + APPROVAL_TTL

    def take(self, question: str) -> bool:
        expiry = self._granted.pop(question, 0.0)
        return expiry > time.monotonic()


def hark_question(registry: ToolRegistry, name: str, args: dict[str, Any]) -> str | None:
    """The exact question the registry would ask for this hark tool call."""
    if name == "omarchy":
        from hark.tools.omarchy import confirm_question

        return confirm_question(str(args.get("command", "")), args.get("args"))
    tool = registry.tools.get(name)
    if tool is None:
        return None
    return tool.confirm_prompt(args) if tool.confirm_prompt else f"Run {name}?"


def question_for(tool_name: str, tool_input: dict[str, Any], reason: str, registry: ToolRegistry) -> str:
    if tool_name.startswith("mcp__hark__"):
        q = hark_question(registry, tool_name.removeprefix("mcp__hark__"), tool_input)
        if q:
            return q
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        return f"Claude wants to run: {cmd[:160]}. Allow it?"
    target = tool_input.get("file_path") or tool_input.get("path") or ""
    return f"Claude wants to use {tool_name}{' on ' + str(target) if target else ''}. {reason} Allow it?"


async def approve(
    tool_name: str, tool_input: dict[str, Any], registry: ToolRegistry, approvals: Approvals, ask=ask_confirm
) -> dict[str, Any]:
    decision, reason = classify_claude_tool(tool_name, tool_input)
    if decision == "allow":
        return {"behavior": "allow"}
    if decision == "deny":
        return {"behavior": "deny", "message": reason}
    question = question_for(tool_name, tool_input, reason, registry)
    if await ask(question):
        if tool_name.startswith("mcp__hark__"):
            approvals.grant(question)
        return {"behavior": "allow"}
    return {"behavior": "deny", "message": f"The user did not approve. {reason}"}


def build_server() -> Server:
    registry = build_registry(config.load())
    for tool in daemon_tools():
        registry.register(tool)
    approvals = Approvals()

    async def confirm(question: str) -> bool:
        return approvals.take(question) or await ask_confirm(question)

    registry.confirm = confirm
    tools = [types.Tool(name=t.name, description=t.description, input_schema=t.parameters)
             for t in registry.tools.values()]
    tools.append(types.Tool(
        name="approve",
        description="Permission prompt handler for Claude Code; not for direct use.",
        input_schema=APPROVE_SCHEMA,
    ))

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        args = params.arguments or {}
        if params.name == "approve":
            answer = await approve(str(args.get("tool_name", "")), args.get("input") or {}, registry, approvals)
            return types.CallToolResult(content=[types.TextContent(text=json.dumps(answer))])
        result = await registry.call(params.name, args)
        text = result.text
        if result.data is not None:
            text += "\n" + json.dumps(result.data, default=str)[:4000]
        return types.CallToolResult(content=[types.TextContent(text=text)], is_error=not result.ok)

    return Server("hark", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def main() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
