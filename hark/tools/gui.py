"""Desktop GUI, vision, pointer, and on-screen Omi companion tools.

Tiered by speed and reliability on Wayland / Hyprland:
  1. Screen metadata & window geometry via Hyprland IPC (~0.5 ms).
  2. Semantic UI tree & direct actions via AT-SPI over D-Bus (no screenshots, works
     live on Chromium, Electron, GTK, and Qt/Quickshell apps via /usr/bin/python).
  3. Universal pointer input via ydotool (with Hyprland movecursor + AT-SPI hit-test
     fallback when ydotool is not installed) and fast grim screenshots for multimodal
     workers.
  4. On-screen mascot pointing (`point_at` / `highlight_window`) via daemon events so
     Omi glides to the target, looks at it, spotlights it, and shows a label.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hark import config
from hark.tools.base import Safety, Tool, ToolResult
from hark.tools.hyprland import find_window, hypr_query, send_first_ok
from hark.tools.system import focused_is_terminal, run

SYS_PYTHON = "/usr/bin/python"
MAX_ELEMENTS = 100
_DESTRUCTIVE_LABELS = re.compile(
    r"\b(delete|remove|erase|wipe|format|uninstall|destroy|send|submit|post|publish|"
    r"pay|purchase|buy|checkout|place order|confirm order|transfer|shut ?down|reboot|power ?off|sign out|log out)\b",
    re.I,
)


@dataclass
class UiElement:
    index: int
    app: str
    role: str
    name: str
    x: int
    y: int
    w: int
    h: int
    actions: list[str]
    path: list[int]  # child index path from Atspi desktop root


_LAST_ELEMENTS: dict[int, UiElement] = {}


# --- mascot bridge (best-effort, never breaks a tool call) ---


async def notify_mascot(payload: dict[str, Any], socket_path: Path | None = None) -> bool:
    """Send a mascot event command to the running daemon so the on-screen Omi moves or points."""
    path = Path(socket_path or config.SOCKET_PATH)
    if not path.exists():
        return False
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(str(path)), 0.4)
    except (OSError, TimeoutError):
        return False
    try:
        req = {"cmd": "mascot", **payload}
        writer.write((json.dumps(req) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), 1.0)
        if not line:
            return False
        return bool(json.loads(line).get("ok"))
    except (OSError, TimeoutError, ValueError):
        return False
    finally:
        writer.close()


# --- pure command builders (unit-tested without touching the desktop) ---


def build_capture_screen(path: Path, scale: float = 0.5) -> list[str]:
    scale = min(max(float(scale), 0.25), 1.0)
    return ["grim", "-t", "jpeg", "-q", "82", "-s", f"{scale:.2f}", str(path)]


def build_movecursor(x: int, y: int) -> list[str]:
    return [f"dispatch movecursor {int(x)} {int(y)}"]


def _ydotool_button_code(button: str) -> str:
    # ydotool click hex codes: 0xC0 = left click (down+up), 0xC1 = right, 0xC2 = middle
    codes = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
    code = codes.get(button.lower())
    if code is None:
        raise ValueError(f"unknown mouse button {button!r}; use left, right or middle")
    return code


def build_ydotool_click(button: str = "left", clicks: int = 1) -> list[str]:
    clicks = max(1, min(int(clicks), 3))
    code = _ydotool_button_code(button)
    argv = ["ydotool", "click"]
    if clicks > 1:
        argv += ["--repeat", str(clicks), "--next-delay", "60"]
    argv.append(code)
    return argv


def build_ydotool_scroll(dy: int) -> list[str]:
    # Negative dy scrolls down, positive scrolls up (ydotool mousemove -w -y <delta>).
    dy = max(-20, min(int(dy), 20))
    return ["ydotool", "mousemove", "--wheel", "--", "0", str(dy)]


def ydotool_available() -> bool:
    return shutil.which("ydotool") is not None


# --- screen geometry & vision ---


def _focused_monitor() -> dict[str, Any]:
    monitors = hypr_query("monitors") or []
    for m in monitors:
        if m.get("focused"):
            return m
    return monitors[0] if monitors else {}


def logical_screen_size(mon: dict[str, Any] | None = None) -> tuple[int, int, float]:
    mon = mon if mon is not None else _focused_monitor()
    scale = float(mon.get("scale") or 1.0)
    w = int(round((mon.get("width") or 1920) / scale))
    h = int(round((mon.get("height") or 1080) / scale))
    return w, h, scale


async def screen_info() -> ToolResult:
    mon = _focused_monitor()
    w, h, scale = logical_screen_size(mon)
    cursor = hypr_query("cursorpos") or {}
    win = hypr_query("activewindow") or {}
    info = {
        "monitor": mon.get("name", ""),
        "logical_width": w,
        "logical_height": h,
        "scale": scale,
        "cursor": {"x": cursor.get("x", 0), "y": cursor.get("y", 0)},
        "active_window": {
            "class": win.get("class") or "",
            "title": win.get("title") or "",
            "at": win.get("at") or [0, 0],
            "size": win.get("size") or [0, 0],
        } if win.get("address") else None,
    }
    title = info["active_window"]["title"][:50] if info["active_window"] else "none"
    return ToolResult(True, f"Screen {w} by {h} logical pixels, focused window: {title}.", info)


async def capture_screen(scale: float = 0.5) -> ToolResult:
    """Fast JPEG screenshot in /tmp with logical coordinate metadata for visual inspection."""
    w, h, mon_scale = logical_screen_size()
    out_dir = Path(tempfile.gettempdir()) / "omi-vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "screen.jpg"
    rc, _, err = await run(build_capture_screen(path, scale=scale), timeout=5.0)
    if rc != 0:
        return ToolResult(False, f"Screen capture failed: {err.strip() or rc}")
    return ToolResult(
        True,
        f"Captured screen ({w}x{h} logical coords) to {path}.",
        {"path": str(path), "logical_width": w, "logical_height": h, "monitor_scale": mon_scale},
    )


# --- AT-SPI semantic inspection & actions (runs via /usr/bin/python gi Atspi) ---

_ATSPI_INSPECT_SCRIPT = r"""
import json, sys
import gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi

app_filter = sys.argv[1].strip().lower()
query = sys.argv[2].strip().lower()
limit = int(sys.argv[3])
win_x, win_y, win_w, win_h = map(int, sys.argv[4:8])

INTERACTIVE = {
    'push button', 'toggle button', 'check box', 'radio button', 'menu item',
    'check menu item', 'radio menu item', 'combo box', 'text', 'entry',
    'password text', 'link', 'page tab', 'spin button', 'slider', 'list item',
    'tree item', 'table cell', 'icon', 'document web',
}

desktop = Atspi.get_desktop(0)
results = []

def walk(node, app_name, path, depth):
    if len(results) >= limit or depth > 14:
        return
    try:
        role = node.get_role_name() or ''
        name = (node.get_name() or '').strip()
        states = node.get_state_set()
        if states and not (states.contains(Atspi.StateType.SHOWING) or states.contains(Atspi.StateType.VISIBLE)):
            return
    except Exception:
        return

    actions = []
    try:
        ai = node.get_action_iface()
        if ai:
            for i in range(min(ai.get_n_actions(), 4)):
                an = ai.get_action_name(i)
                if an:
                    actions.append(an)
    except Exception:
        pass

    rect = None
    try:
        ci = node.get_component_iface()
        if ci:
            ext = ci.get_extents(Atspi.CoordType.SCREEN)
            if ext.width > 0 and ext.height > 0:
                rect = (int(ext.x), int(ext.y), int(ext.width), int(ext.height))
    except Exception:
        pass

    keep = (role in INTERACTIVE or bool(actions)) and rect is not None
    if keep and name == '' and role not in ('text', 'entry', 'combo box'):
        keep = False
    if keep and query and query not in name.lower() and query not in role.lower():
        keep = False
    if keep and win_w > 0 and win_h > 0 and not app_filter:
        cx, cy = rect[0] + rect[2] // 2, rect[1] + rect[3] // 2
        if not (win_x - 20 <= cx <= win_x + win_w + 20 and win_y - 20 <= cy <= win_y + win_h + 20):
            keep = False

    if keep:
        results.append({
            'app': app_name,
            'role': role,
            'name': name[:100],
            'x': rect[0], 'y': rect[1], 'w': rect[2], 'h': rect[3],
            'actions': actions,
            'path': path,
        })

    try:
        n = min(node.get_child_count(), 120)
    except Exception:
        return
    for i in range(n):
        if len(results) >= limit:
            break
        try:
            child = node.get_child_at_index(i)
            if child:
                walk(child, app_name, path + [i], depth + 1)
        except Exception:
            continue

n_apps = desktop.get_child_count()
for idx in range(n_apps):
    if len(results) >= limit:
        break
    try:
        app = desktop.get_child_at_index(idx)
        aname = (app.get_name() or '').strip() if app else ''
    except Exception:
        continue
    if app_filter and app_filter not in aname.lower():
        continue
    walk(app, aname, [idx], 0)

print(json.dumps(results))
"""

_ATSPI_ACT_SCRIPT = r"""
import json, sys
import gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi

path = json.loads(sys.argv[1])
action = sys.argv[2]
text = sys.argv[3]

node = Atspi.get_desktop(0)
for idx in path:
    node = node.get_child_at_index(idx)
    if node is None:
        print(json.dumps({'ok': False, 'error': 'element is no longer present'}))
        sys.exit(0)

if action == 'type':
    try:
        ei = node.get_editable_text_iface()
        if ei:
            ei.set_text_contents(text)
            print(json.dumps({'ok': True, 'method': 'editable_text'}))
            sys.exit(0)
    except Exception:
        pass
    try:
        ci = node.get_component_iface()
        if ci:
            ci.grab_focus()
    except Exception:
        pass
    print(json.dumps({'ok': False, 'fallback': 'wtype'}))
    sys.exit(0)

try:
    ai = node.get_action_iface()
    if ai and ai.get_n_actions() > 0:
        chosen = 0
        for i in range(ai.get_n_actions()):
            nm = (ai.get_action_name(i) or '').lower()
            if action.lower() in nm or nm in ('click', 'press', 'activate'):
                chosen = i
                break
        ok = bool(ai.do_action(chosen))
        print(json.dumps({'ok': ok, 'method': 'atspi_action', 'action': ai.get_action_name(chosen)}))
        sys.exit(0)
except Exception as exc:
    print(json.dumps({'ok': False, 'error': str(exc)}))
    sys.exit(0)

print(json.dumps({'ok': False, 'error': 'element has no AT-SPI action'}))
"""

_ATSPI_HIT_SCRIPT = r"""
import json, sys
import gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi

tx, ty = int(sys.argv[1]), int(sys.argv[2])
desktop = Atspi.get_desktop(0)
best = None

def walk(node, depth):
    global best
    if depth > 14:
        return
    try:
        ci = node.get_component_iface()
        if ci:
            ext = ci.get_extents(Atspi.CoordType.SCREEN)
            if ext.width > 0 and ext.height > 0 and ext.x <= tx <= ext.x + ext.width and ext.y <= ty <= ext.y + ext.height:
                ai = node.get_action_iface()
                if ai and ai.get_n_actions() > 0:
                    area = ext.width * ext.height
                    if best is None or area < best[0]:
                        best = (area, node, ai)
    except Exception:
        pass
    try:
        n = min(node.get_child_count(), 120)
    except Exception:
        return
    for i in range(n):
        try:
            c = node.get_child_at_index(i)
            if c:
                walk(c, depth + 1)
        except Exception:
            continue

for i in range(desktop.get_child_count()):
    try:
        a = desktop.get_child_at_index(i)
        if a:
            walk(a, 0)
    except Exception:
        continue

if best is None:
    print(json.dumps({'ok': False, 'error': 'no clickable AT-SPI element at those coordinates'}))
else:
    _, node, ai = best
    ok = bool(ai.do_action(0))
    print(json.dumps({'ok': ok, 'name': node.get_name() or '', 'role': node.get_role_name() or ''}))
"""


def _atspi_available() -> bool:
    return Path(SYS_PYTHON).is_file()


def _format_elements(elements: list[UiElement]) -> str:
    if not elements:
        return "No matching interactive UI elements found."
    lines = [
        f"[{e.index}] {e.role} \"{e.name}\" @({e.x},{e.y},{e.w},{e.h})"
        for e in elements
    ]
    return f"Found {len(elements)} elements:\n" + "\n".join(lines)


async def ui_inspect(app: str = "", query: str = "", max_elements: int = 50) -> ToolResult:
    """List interactive UI elements of the focused window (or named app) via AT-SPI."""
    if not _atspi_available():
        return ToolResult(False, "System Python AT-SPI bridge is not available.")
    limit = max(1, min(int(max_elements), MAX_ELEMENTS))
    win_x = win_y = win_w = win_h = 0
    if not app.strip():
        try:
            win = hypr_query("activewindow") or {}
            at = win.get("at") or [0, 0]
            sz = win.get("size") or [0, 0]
            win_x, win_y, win_w, win_h = int(at[0]), int(at[1]), int(sz[0]), int(sz[1])
        except (OSError, ValueError):
            pass
    rc, out, err = await run(
        [SYS_PYTHON, "-c", _ATSPI_INSPECT_SCRIPT, app, query, str(limit),
         str(win_x), str(win_y), str(win_w), str(win_h)],
        timeout=6.0,
    )
    if rc != 0:
        return ToolResult(False, f"UI inspection failed: {(err or out).strip().splitlines()[-1] if (err or out).strip() else rc}")
    try:
        raw = json.loads(out)
    except ValueError:
        return ToolResult(False, "UI inspection returned malformed output.")
    _LAST_ELEMENTS.clear()
    items: list[UiElement] = []
    for i, row in enumerate(raw):
        el = UiElement(
            index=i,
            app=str(row.get("app", "")),
            role=str(row.get("role", "")),
            name=str(row.get("name", "")),
            x=int(row.get("x", 0)),
            y=int(row.get("y", 0)),
            w=int(row.get("w", 0)),
            h=int(row.get("h", 0)),
            actions=list(row.get("actions") or []),
            path=list(row.get("path") or []),
        )
        _LAST_ELEMENTS[i] = el
        items.append(el)
    summary = f"Found {len(items)} UI elements." if items else "No interactive UI elements found."
    return ToolResult(True, summary, _format_elements(items))


def ui_action_safety(args: dict[str, Any]) -> Safety:
    idx = args.get("index")
    el = _LAST_ELEMENTS.get(int(idx)) if isinstance(idx, (int, str)) and str(idx).isdigit() else None
    if focused_is_terminal():
        return "destructive"
    if el and _DESTRUCTIVE_LABELS.search(el.name):
        return "destructive"
    if str(args.get("action", "")).lower() == "type" and ("\n" in str(args.get("text", "")) or "\r" in str(args.get("text", ""))):
        return "destructive"
    return "reversible"


def ui_action_question(args: dict[str, Any]) -> str:
    idx = args.get("index")
    el = _LAST_ELEMENTS.get(int(idx)) if isinstance(idx, (int, str)) and str(idx).isdigit() else None
    label = f'{el.role} "{el.name}"' if el else f"element {idx}"
    action = str(args.get("action", "click"))
    if action == "type":
        return f"Type into {label}?"
    return f"Click {label}?"


async def ui_action(index: int, action: str = "click", text: str = "") -> ToolResult:
    """Click, activate, or type into an indexed element from ui_inspect."""
    el = _LAST_ELEMENTS.get(int(index))
    if el is None:
        return ToolResult(False, f"No element {index}. Call ui_inspect first.")
    cx, cy = el.x + el.w // 2, el.y + el.h // 2
    await notify_mascot({"action": "point", "x": cx, "y": cy, "w": el.w, "h": el.h,
                         "label": el.name[:40] or el.role, "duration": 4})
    rc, out, err = await run(
        [SYS_PYTHON, "-c", _ATSPI_ACT_SCRIPT, json.dumps(el.path), action, text],
        timeout=5.0,
    )
    if rc != 0:
        return ToolResult(False, f"UI action failed: {(err or out).strip()}")
    try:
        res = json.loads(out)
    except ValueError:
        return ToolResult(False, "UI action returned malformed output.")
    if res.get("ok"):
        return ToolResult(True, f"{'Typed into' if action == 'type' else 'Clicked'} {el.name or el.role}.")
    if res.get("fallback") == "wtype" and action == "type":
        rc2, _, err2 = await run(["wtype", "--", text], timeout=3.0)
        if rc2 == 0:
            return ToolResult(True, f"Typed into {el.name or el.role}.")
        return ToolResult(False, f"Typing failed: {err2.strip()}")
    # If AT-SPI couldn't invoke it directly, try clicking its center coordinates.
    if action != "type":
        return await mouse_click(cx, cy)
    return ToolResult(False, str(res.get("error", "UI action failed.")))


# --- universal pointer control ---


def mouse_safety(args: dict[str, Any]) -> Safety:
    return "destructive" if focused_is_terminal() else "reversible"


async def mouse_move(x: int, y: int) -> ToolResult:
    x, y = int(x), int(y)
    ok, _ = send_first_ok(build_movecursor(x, y))
    await notify_mascot({"action": "move", "x": x, "y": y, "duration": 4})
    return ToolResult(ok, f"Moved cursor to {x}, {y}." if ok else "Hyprland refused to move the cursor.")


async def mouse_click(x: int, y: int, button: str = "left", clicks: int = 1) -> ToolResult:
    x, y = int(x), int(y)
    send_first_ok(build_movecursor(x, y))
    await notify_mascot({"action": "point", "x": x, "y": y, "w": 24, "h": 24, "label": "", "duration": 3})
    if ydotool_available():
        argv = build_ydotool_click(button=button, clicks=clicks)
        rc, out, err = await run(argv, timeout=3.0)
        if rc == 0:
            return ToolResult(True, f"Clicked at {x}, {y}.")
        msg = (err or out).strip()
        if "ydotoold" in msg.lower() or "socket" in msg.lower():
            return ToolResult(False, "ydotoold is not running; start it with: systemctl --user start ydotoold")
        return ToolResult(False, f"Click failed: {msg}")
    # Fallback without ydotool: try AT-SPI hit-test click at (x, y).
    if button == "left" and _atspi_available():
        rc, out, _ = await run([SYS_PYTHON, "-c", _ATSPI_HIT_SCRIPT, str(x), str(y)], timeout=4.0)
        if rc == 0:
            try:
                res = json.loads(out)
                if res.get("ok"):
                    label = res.get("name") or res.get("role") or f"{x}, {y}"
                    return ToolResult(True, f"Clicked {label}.")
            except ValueError:
                pass
    return ToolResult(
        False,
        "Could not click that spot without ydotool. Install ydotool (`omarchy pkg add ydotool`) for raw pixel clicks, or use ui_inspect / Chrome tools.",
    )


async def mouse_scroll(x: int, y: int, dy: int = -3) -> ToolResult:
    x, y, dy = int(x), int(y), int(dy)
    send_first_ok(build_movecursor(x, y))
    if ydotool_available():
        rc, out, err = await run(build_ydotool_scroll(dy), timeout=3.0)
        if rc == 0:
            return ToolResult(True, "Scrolled down." if dy < 0 else "Scrolled up.")
        return ToolResult(False, f"Scroll failed: {(err or out).strip()}")
    key = "Next" if dy < 0 else "Prior"
    rc, _, err = await run(["wtype", "-k", key], timeout=2.0)
    return ToolResult(rc == 0, ("Scrolled down." if dy < 0 else "Scrolled up.") if rc == 0 else f"Scroll failed: {err.strip()}")


# --- on-screen Omi pointer / spotlight ---


async def point_at(x: int, y: int, label: str = "", w: int = 0, h: int = 0, duration: int = 8) -> ToolResult:
    """Move Omi on screen next to (x, y), look at it, highlight it, and show a label."""
    x, y = int(x), int(y)
    w, h = max(0, int(w)), max(0, int(h))
    duration = max(2, min(int(duration), 30))
    ok = await notify_mascot({
        "action": "point",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "label": str(label)[:140],
        "duration": duration,
    })
    return ToolResult(True, f"Pointing at {label or f'{x}, {y}'}.", {"x": x, "y": y, "notified": ok})


async def highlight_window(query: str, label: str = "", duration: int = 8) -> ToolResult:
    """Move Omi to a window on screen and highlight it."""
    hit = find_window(hypr_query("clients"), query)
    if hit is None:
        return ToolResult(False, f"I can't find a window for {query}.")
    at = hit.get("at") or [0, 0]
    sz = hit.get("size") or [0, 0]
    cx, cy = int(at[0]) + int(sz[0]) // 2, int(at[1]) + int(sz[1]) // 2
    shown = label or (hit.get("title") or hit.get("class") or query)[:80]
    return await point_at(cx, cy, label=shown, w=int(sz[0]), h=int(sz[1]), duration=duration)


def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


def tools() -> list[Tool]:
    i = {"type": "integer"}
    s = {"type": "string"}
    return [
        Tool("screen_info", "Get logical screen size, cursor position, and active window bounds.",
             _schema(), "read", screen_info),
        Tool("capture_screen", "Capture a fast JPEG screenshot and return its path and logical coordinates.",
             _schema({"scale": {"type": "number", "minimum": 0.25, "maximum": 1.0}}),
             "read", capture_screen),
        Tool("ui_inspect", "List visible interactive desktop UI elements with indices and screen coordinates (no screenshot needed).",
             _schema({"app": s, "query": s, "max_elements": i}), "read", ui_inspect),
        Tool("ui_action", "Click or type into an indexed element returned by ui_inspect.",
             _schema({"index": i, "action": {"type": "string", "enum": ["click", "type"]}, "text": s}, ["index"]),
             "reversible", ui_action, ui_action_question, ui_action_safety),
        Tool("mouse_move", "Move the mouse cursor to logical screen coordinates (x, y).",
             _schema({"x": i, "y": i}, ["x", "y"]), "reversible", mouse_move),
        Tool("mouse_click", "Click at logical screen coordinates (x, y).",
             _schema({
                 "x": i, "y": i,
                 "button": {"type": "string", "enum": ["left", "right", "middle"]},
                 "clicks": {"type": "integer", "minimum": 1, "maximum": 3},
             }, ["x", "y"]),
             "reversible", mouse_click,
             lambda a: f"Click at {a.get('x')}, {a.get('y')}?", mouse_safety),
        Tool("mouse_scroll", "Scroll at logical screen coordinates (x, y); negative dy scrolls down.",
             _schema({"x": i, "y": i, "dy": i}, ["x", "y"]), "reversible", mouse_scroll),
        Tool("point_at", "Move Omi on screen to point at coordinates (x, y) with a spotlight and optional label.",
             _schema({"x": i, "y": i, "label": s, "w": i, "h": i, "duration": i}, ["x", "y"]),
             "read", point_at),
        Tool("highlight_window", "Move Omi on screen to point at and highlight an open window.",
             _schema({"query": s, "label": s, "duration": i}, ["query"]),
             "read", highlight_window),
    ]
