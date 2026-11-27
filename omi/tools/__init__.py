"""Device tools for this machine (Omarchy on Hyprland)."""

from __future__ import annotations

import os
from dataclasses import replace

from omi.config import Config
from omi.tools.base import Tool, ToolRegistry, ToolResult


def build_registry(cfg: Config) -> ToolRegistry:
    """Every device tool. The caller sets registry.confirm (daemon: ask by voice;
    MCP server: ask through the daemon socket). With OMI_DRY_RUN=1 the device
    tools that change something only report what they would do."""
    from omi.tools import dev, gui, hyprland, media, omarchy, system

    registry = ToolRegistry()
    dry = os.environ.get("OMI_DRY_RUN") == "1" or os.environ.get("OMI_DRY_RUN") == "1"
    for tool in [*hyprland.tools(), *media.tools(), *system.tools(), *gui.tools(), *dev.tools(), omarchy.make_tool()]:
        registry.register(_dry(tool) if dry and tool.safety != "read" else tool)
    return registry


def _dry(tool: Tool) -> Tool:
    """Same tool and same safety checks (confirmation is still asked), but the action
    is a no-op. Unlike ToolRegistry.dry_run this keeps memory and delegate working."""

    async def fn(**args: object) -> ToolResult:
        return ToolResult(True, f"Dry run: would call {tool.name}.", {"args": args})

    return replace(tool, fn=fn)
