"""Device tool contract.

A Tool is one thing the assistant can do on the device. Tools are the portable
boundary: the same registry serves the local talker (in process), the fast router,
and outside agents through the MCP server in hark.tools.mcp_server. A new platform
means a new set of tools behind the same names.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

# read: looks, changes nothing.
# reversible: changes something that is trivial to undo (volume, focus, play/pause).
# destructive: hard or impossible to undo (close a window, delete, send, power off).
Safety = Literal["read", "reversible", "destructive"]


@dataclass
class ToolResult:
    ok: bool
    text: str  # short, speakable summary of what happened
    data: Any = None


# Asks the user to approve a destructive action. Gets a short spoken question,
# returns True only on a clear yes. No ConfirmFn means: refuse.
ConfirmFn = Callable[[str], Awaitable[bool]]


@dataclass
class Tool:
    name: str  # snake_case, stable across platforms
    description: str  # one line, written for the model
    parameters: dict[str, Any]  # JSON Schema object
    safety: Safety
    fn: Callable[..., Awaitable[ToolResult]]  # async, called with the arguments as kwargs
    # For destructive tools: builds the question to ask from the arguments.
    confirm_prompt: Callable[[dict[str, Any]], str] | None = None


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)
    confirm: ConfirmFn | None = None
    dry_run: bool = False  # tests: non-read tools report what they would do

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self.tools[tool.name] = tool

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self.tools.values()
        ]

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        """The single entry point for running a tool. Never raises: errors come back
        as ToolResult(ok=False). Enforces the safety model."""
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(False, f"There is no tool called {name}.")
        if self.dry_run and tool.safety != "read":
            return ToolResult(True, f"Dry run: would call {name}.", {"args": args})
        if tool.safety == "destructive":
            question = tool.confirm_prompt(args) if tool.confirm_prompt else f"Run {name}?"
            if self.confirm is None or not await self.confirm(question):
                return ToolResult(False, "Cancelled, not confirmed.")
        try:
            return await tool.fn(**args)
        except TypeError as exc:
            return ToolResult(False, f"Bad arguments for {name}: {exc}")
        except Exception as exc:  # tools talk to the outside world; report, do not crash
            return ToolResult(False, f"{name} failed: {exc}")
