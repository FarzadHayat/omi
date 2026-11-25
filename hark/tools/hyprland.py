"""Window and workspace tools, talking to Hyprland's request socket directly.

Hyprland's request socket answers one request per connection and then closes it,
so there is no persistent connection to hold; a raw connect-send-read takes about
0.1 ms, against 3 to 4 ms for spawning hyprctl.

Hyprland 0.56 with a Lua config takes "dispatch <lua expression>". Every command
builder returns a list of alternatives: the Lua form first, the legacy dispatcher
last. The first one Hyprland answers "ok" to wins.
"""

from __future__ import annotations

import json
import os
import re
import socket
from typing import Any

from hark.tools.base import Tool, ToolResult
from hark.tools.omarchy import App, resolve_app
from hark.tools.system import spawn


def socket_path() -> str:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return f"{runtime}/hypr/{os.environ.get('HYPRLAND_INSTANCE_SIGNATURE', '')}/.socket.sock"


def request(msg: str, timeout: float = 1.0) -> str:
    """One request on Hyprland's socket. Blocking is fine: it answers in well under a ms."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(socket_path())
        sock.sendall(msg.encode())
        chunks = []
        while chunk := sock.recv(65536):
            chunks.append(chunk)
    return b"".join(chunks).decode(errors="replace")


def hypr_query(what: str) -> Any:
    return json.loads(request(f"j/{what}") or "null")


def send_first_ok(candidates: list[str]) -> tuple[bool, str]:
    reply = ""
    for msg in candidates:
        reply = request(msg).strip()
        if reply == "ok":
            return True, msg
    return False, reply


# --- command builders (pure, tested) ---


def build_focus_window(address: str) -> list[str]:
    return [f'dispatch hl.dsp.focus({{ window = "address:{address}" }})', f"dispatch focuswindow address:{address}"]


def build_close_window(address: str) -> list[str]:
    return [f'dispatch hl.dsp.window.close({{ window = "address:{address}" }})', f"dispatch closewindow address:{address}"]


def build_switch_workspace(n: int) -> list[str]:
    # Omarchy gives each monitor its own workspaces 1 to 10; o.workspaces maps the
    # spoken number to this monitor's real id, as the SUPER+number keys do.
    return [f"eval o.workspaces.focus({n})()", f'dispatch hl.dsp.focus({{ workspace = "{n}" }})', f"dispatch workspace {n}"]


def build_move_to_workspace(n: int) -> list[str]:
    return [
        f"eval o.workspaces.move_window({n})()",
        f'dispatch hl.dsp.window.move({{ workspace = "{n}" }})',
        f"dispatch movetoworkspace {n}",
    ]


def build_fullscreen() -> list[str]:
    return ['dispatch hl.dsp.window.fullscreen({ mode = "fullscreen" })', "dispatch fullscreen 0"]


def _check_workspace(n: Any) -> int:
    n = int(n)
    if not 1 <= n <= 10:
        raise ValueError("workspaces go from 1 to 10")
    return n


# --- window matching ---


def _summary(c: dict) -> dict:
    return {
        "address": c.get("address"),
        "class": c.get("class") or c.get("initialClass") or "",
        "title": c.get("title") or "",
        "workspace": (c.get("workspace") or {}).get("name"),
    }


def _word(pattern: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(pattern.lower())}(?![a-z0-9])", re.I)


def find_app_window(clients: list[dict], app: App) -> dict | None:
    pats = [_word(c) for c in app.classes]
    for c in clients:
        klass = (c.get("class") or c.get("initialClass") or "").lower()
        if any(p.search(klass) for p in pats):
            return c
    return None


def find_window(clients: list[dict], query: str) -> dict | None:
    """Best window for a spoken query: an app match first, then class, then title."""
    q = query.strip().lower()
    if not q:
        return None
    app = resolve_app(q)
    if app and (hit := find_app_window(clients, app)):
        return hit
    pat = _word(q)
    for field in ("class", "title"):
        for c in clients:
            if pat.search(c.get(field) or ""):
                return c
    for c in clients:
        if q in (c.get("title") or "").lower():
            return c
    return None


def _label(c: dict) -> str:
    title = (c.get("title") or "").strip()
    klass = c.get("class") or "a window"
    return f"{klass}: {title[:60]}" if title and title.lower() != klass.lower() else klass


# --- tools ---


async def list_windows() -> ToolResult:
    clients = [c for c in hypr_query("clients") if c.get("mapped", True)]
    data = [_summary(c) for c in clients]
    if not data:
        return ToolResult(True, "No windows are open.", data)
    names = ", ".join(_label(c) for c in clients[:8])
    return ToolResult(True, f"{len(data)} windows: {names}.", data)


async def active_window() -> ToolResult:
    c = hypr_query("activewindow") or {}
    if not c.get("address"):
        return ToolResult(True, "No window is focused.", None)
    return ToolResult(True, f"The focused window is {_label(c)}.", _summary(c))


async def focus_window(query: str) -> ToolResult:
    hit = find_window(hypr_query("clients"), query)
    if hit is None:
        return ToolResult(False, f"I can't find a window for {query}.")
    ok, _ = send_first_ok(build_focus_window(hit["address"]))
    return ToolResult(ok, f"Focused {_label(hit)}." if ok else "Hyprland refused to focus it.")


async def close_window(query: str = "") -> ToolResult:
    if query.strip():
        hit = find_window(hypr_query("clients"), query)
    else:
        hit = hypr_query("activewindow") or None
    if not hit or not hit.get("address"):
        return ToolResult(False, f"I can't find a window for {query or 'that'}.")
    ok, _ = send_first_ok(build_close_window(hit["address"]))
    return ToolResult(ok, f"Closed {_label(hit)}." if ok else "Hyprland refused to close it.")


async def launch_or_focus(app: str) -> ToolResult:
    spec = resolve_app(app)
    if spec is None:
        return ToolResult(False, f"I don't know an app called {app}.")
    hit = find_app_window(hypr_query("clients"), spec)
    if hit is not None:
        ok, _ = send_first_ok(build_focus_window(hit["address"]))
        return ToolResult(ok, f"Switched to {spec.name}." if ok else "Hyprland refused to focus it.")
    spawn(list(spec.launch))
    return ToolResult(True, f"Opening {spec.name}.")


async def switch_workspace(n: int) -> ToolResult:
    ok, _ = send_first_ok(build_switch_workspace(_check_workspace(n)))
    return ToolResult(ok, f"Workspace {n}." if ok else "Hyprland refused to switch.")


async def move_window_to_workspace(n: int) -> ToolResult:
    ok, _ = send_first_ok(build_move_to_workspace(_check_workspace(n)))
    return ToolResult(ok, f"Moved it to workspace {n}." if ok else "Hyprland refused to move it.")


async def toggle_fullscreen() -> ToolResult:
    ok, _ = send_first_ok(build_fullscreen())
    return ToolResult(ok, "Toggled full screen." if ok else "Hyprland refused.")


def tools() -> list[Tool]:
    def obj(props: dict | None = None, required: list[str] | None = None) -> dict:
        return {"type": "object", "properties": props or {}, "required": required or []}

    ws = {"type": "integer", "minimum": 1, "maximum": 10}
    q = {"type": "string", "description": "app name or words from the window title"}
    return [
        Tool("launch_or_focus", "Open an app, or switch to it if it is already open.",
             obj({"app": {"type": "string", "description": "app name, like browser or spotify"}}, ["app"]),
             "reversible", launch_or_focus),
        Tool("list_windows", "List the open windows.", obj(), "read", list_windows),
        Tool("active_window", "Tell which window is focused.", obj(), "read", active_window),
        Tool("focus_window", "Switch to an open window by app name or title.",
             obj({"query": q}, ["query"]), "reversible", focus_window),
        Tool("close_window", "Close a window by app name or title; empty closes the focused one.",
             obj({"query": q}), "destructive", close_window,
             lambda a: f"Close {a.get('query') or 'this window'}?"),
        Tool("switch_workspace", "Go to workspace 1 to 10.", obj({"n": ws}, ["n"]), "reversible", switch_workspace),
        Tool("move_window_to_workspace", "Move the focused window to workspace 1 to 10.",
             obj({"n": ws}, ["n"]), "reversible", move_window_to_workspace),
        Tool("toggle_fullscreen", "Make the focused window full screen, or back.", obj(), "reversible",
             toggle_fullscreen),
    ]
