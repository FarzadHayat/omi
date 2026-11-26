"""Device tools for this machine (Omarchy on Hyprland)."""

from __future__ import annotations

import os
from dataclasses import replace

from hark.config import Config
from hark.tools.base import Tool, ToolRegistry, ToolResult


def build_registry(cfg: Config) -> ToolRegistry:
    """Every device tool. The caller sets registry.confirm (daemon: ask by voice;
    MCP server: ask through the daemon socket). With HARK_DRY_RUN=1 the device
    tools that change something only report what they would do."""
    from hark.tools import dev, gui, hyprland, media, omarchy, system

    registry = ToolRegistry()
    dry = os.environ.get("HARK_DRY_RUN") == "1" or os.environ.get("OMI_DRY_RUN") == "1"
    for tool in [*hyprland.tools(), *media.tools(), *system.tools(), *gui.tools(), *dev.tools(), omarchy.make_tool()]:
        registry.register(_dry(tool) if dry and tool.safety != "read" else tool)
    return registry


def _dry(tool: Tool) -> Tool:
    """Same tool and same safety checks (confirmation is still asked), but the action
    is a no-op. Unlike ToolRegistry.dry_run this keeps memory and delegate working."""

    async def fn(**args: object) -> ToolResult:
        return ToolResult(True, f"Dry run: would call {tool.name}.", {"args": args})

    return replace(tool, fn=fn)
