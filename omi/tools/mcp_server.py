"""The omi MCP server (stdio): device tools for outside agents, plus "approve".

Run with "python -m omi.tools.mcp_server". Claude sees the tools as
mcp__omi__<name>. "approve" answers Claude Code's --permission-prompt-tool: it
returns one text block holding {"behavior": "allow"} or
{"behavior": "deny", "message": ...}, and no structured content.

Confirmation goes to the user through the omi daemon ("ask_confirm" on its
socket). No daemon means nobody can say yes, so the answer is no.

"--role worker" marks the background worker's server. With brain.worker_mode set to
"unrestricted" it never asks: what would need a yes runs, and what is blocked (secrets)
stays blocked. The mode is read from the config file on every call, so a change in
Settings applies to the running worker at once.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from dataclasses import replace
from typing import Any

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from omi import config
from omi.tools import build_registry
from omi.tools.base import Tool, ToolRegistry, ToolResult
from omi.tools.safety import classify_claude_tool, describe_command, guarded_write

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


async def ask_confirm(question: str, socket_path: str | None = None, detail: str = "") -> bool:
    """Ask the user through the daemon. True only on a clear yes. `detail` (the full
    command) is shown on screen next to the spoken question."""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(socket_path or config.SOCKET_PATH)), 1.0
        )
    except (OSError, TimeoutError):
        return False
    try:
        req = {"cmd": "ask_confirm", "question": question}
        if detail:
            req["detail"] = detail
        writer.write((json.dumps(req) + "\n").encode())
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
    from omi.memory import NullMemory
    from omi.memory.tools import memory_tools
    from omi.worker import delegate_tool

    def proxy(name: str):
        async def fn(**args: Any) -> ToolResult:
            try:
                reply = await daemon_call({"cmd": "tool", "name": name, "args": args})
            except (OSError, TimeoutError, ValueError) as exc:
                return ToolResult(False, f"The omi daemon is not reachable: {exc}")
            if not reply.get("ok"):
                return ToolResult(False, str(reply.get("error", "failed")))
            return ToolResult(bool(reply["result"]["ok"]), str(reply["result"]["text"]))

        return fn

    from omi.look_setup import camera_tool, look_setup_tool

    from omi.worker import worker_jobs_tool

    tools = [*memory_tools(NullMemory()), delegate_tool(None), look_setup_tool(None),  # type: ignore[arg-type]
             camera_tool(None), worker_jobs_tool(None)]
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


def omi_question(registry: ToolRegistry, name: str, args: dict[str, Any]) -> str | None:
    """The exact question the registry would ask for this omi tool call, when it asks."""
    tool = registry.tools.get(name)
    if tool is None:
        return None
    try:
        if tool.safety_of(args) != "destructive":
            return None
    except (ValueError, TypeError):
        return None
    return tool.confirm_prompt(args) if tool.confirm_prompt else f"Run {name}?"


SPEAK_WHOLE = 160  # a command this short is read out in full
SPEAK_RISKY = 220  # otherwise its risky parts are, when they fit


def _join_words(items: list[str]) -> str:
    items = list(dict.fromkeys(items))
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def bash_question(cmd: str) -> tuple[str, str]:
    """(spoken question, on-screen detail) for a shell command. Every risky action is
    named in plain words; a command is never cut short in a way that hides one."""
    _, steps = describe_command(cmd)
    what = _join_words([s.action for s in steps] or ["run a command"])
    flat = " ".join(cmd.split())
    if len(flat) <= SPEAK_WHOLE:
        return f"Claude wants to {what}: {flat}. Allow it?", ""
    risky = "; ".join(s.text for s in steps)
    if risky and len(risky) <= SPEAK_RISKY:
        return f"Claude wants to {what}. The risky part is: {risky}. The full command is on screen. Allow it?", cmd
    return (f"Claude wants to {what}, in a long command of {len(flat)} characters. "
            f"It is shown on screen. Allow it?", cmd)


def question_for(tool_name: str, tool_input: dict[str, Any], reason: str, registry: ToolRegistry) -> tuple[str, str]:
    """(spoken question, on-screen detail) for a call that needs the user's yes."""
    if tool_name.startswith("mcp__omi__"):
        q = omi_question(registry, tool_name.removeprefix("mcp__omi__"), tool_input)
        if q:
            return q, ""
    if tool_name == "Bash":
        return bash_question(str(tool_input.get("command", "")))
    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        what = guarded_write(path) if path else None
        return f"Claude wants to change {what + ': ' if what else ''}{path}. Allow it?", ""
    if tool_name == "WebFetch":
        return f"Claude wants to open the web address {tool_input.get('url', '')}. {reason} Allow it?", ""
    target = tool_input.get("file_path") or tool_input.get("path") or tool_input.get("url") or ""
    shown = json.dumps(tool_input)[:2000]
    return f"Claude wants to use {tool_name.removeprefix('mcp__omi__')}{' on ' + str(target) if target else ''}. " \
           f"{reason} Allow it?", shown


async def approve(
    tool_name: str, tool_input: dict[str, Any], registry: ToolRegistry, approvals: Approvals, ask=ask_confirm,
    *, unattended: bool = False,
) -> dict[str, Any]:
    decision, reason = classify_claude_tool(tool_name, tool_input)
    if decision == "allow":
        return {"behavior": "allow"}
    if decision == "deny":
        return {"behavior": "deny", "message": reason}
    if unattended:
        return {"behavior": "allow"}
    question, detail = question_for(tool_name, tool_input, reason, registry)
    if await ask(question, detail=detail):
        if tool_name.startswith("mcp__omi__"):
            approvals.grant(question)
        return {"behavior": "allow"}
    return {"behavior": "deny", "message": f"The user did not approve. {reason}"}


def worker_unrestricted(role: str) -> bool:
    if role != "worker":
        return False
    try:
        return config.load().brain.worker_mode == "unrestricted"
    except Exception:
        return False  # an unreadable config never loosens safety


def build_server(role: str = "talker") -> Server:
    cfg = config.load()
    registry = build_registry(cfg)
    for tool in daemon_tools():
        registry.register(tool)
    approvals = Approvals()

    async def confirm(question: str) -> bool:
        return worker_unrestricted(role) or approvals.take(question) or await ask_confirm(question)

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
            answer = await approve(str(args.get("tool_name", "")), args.get("input") or {}, registry, approvals,
                                   unattended=worker_unrestricted(role))
            return types.CallToolResult(content=[types.TextContent(text=json.dumps(answer))])
        result = await registry.call(params.name, args)
        text = result.text
        if result.data is not None:
            text += "\n" + json.dumps(result.data, default=str)[:4000]
        content: list[types.ContentBlock] = [types.TextContent(text=text)]
        for mime_type, raw in result.images:
            content.append(types.ImageContent(
                data=base64.b64encode(raw).decode("ascii"),
                mime_type=mime_type,
            ))
        return types.CallToolResult(content=content, is_error=not result.ok)

    return Server("omi", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def role_from_argv(argv: list[str]) -> str:
    if "--role" in argv[:-1]:
        return argv[argv.index("--role") + 1]
    return "talker"


async def main() -> None:
    server = build_server(role_from_argv(sys.argv[1:]))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
