"""System tools: clipboard, typing, screenshot, notify, URLs, time, lock, power.

Every state-changing tool has a pure build_* function that returns the exact argv
it would run, so tests can check the command without touching the desktop.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from hark.tools.base import Tool, ToolResult


async def run(argv: list[str], timeout: float = 3.0, stdin: bytes | None = None) -> tuple[int, str, str]:
    """Run a short command and wait for it. Returns (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


def spawn(argv: list[str]) -> None:
    """Start a long-lived program detached from hark (own session, no pipes), so
    apps outlive the daemon and never block a turn."""
    subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


async def run_ok(argv: list[str], done: str, timeout: float = 3.0) -> ToolResult:
    """Run a command and turn its exit status into a speakable result."""
    rc, out, err = await run(argv, timeout)
    if rc != 0:
        msg = (err or out).strip().splitlines()
        return ToolResult(False, f"That failed: {msg[-1] if msg else 'exit ' + str(rc)}")
    return ToolResult(True, done)


# --- typing and keys (wtype) ---

_MODS = {"ctrl": "ctrl", "control": "ctrl", "shift": "shift", "alt": "alt", "super": "logo",
         "win": "logo", "windows": "logo", "meta": "logo", "cmd": "logo", "altgr": "altgr"}
_KEYS = {"enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape", "tab": "Tab",
         "space": "space", "backspace": "BackSpace", "delete": "Delete", "del": "Delete",
         "up": "Up", "down": "Down", "left": "Left", "right": "Right", "home": "Home", "end": "End",
         "pageup": "Prior", "pagedown": "Next", "page_up": "Prior", "page_down": "Next",
         "insert": "Insert", "print": "Print"}


def build_type_text(text: str) -> list[str]:
    return ["wtype", "--", text]


def build_press_keys(combo: str) -> list[str]:
    """'ctrl+shift+t' -> wtype -M ctrl -M shift -k t -m shift -m ctrl."""
    parts = [p.strip().lower() for p in combo.replace(" ", "+").split("+") if p.strip()]
    if not parts:
        raise ValueError("no keys given")
    mods = [_MODS[p] for p in parts[:-1] if p in _MODS]
    if len(mods) != len(parts) - 1:
        raise ValueError(f"unknown modifier in {combo!r}")
    last = parts[-1]
    if last in _MODS:
        raise ValueError("a combo must end with a key, not a modifier")
    key = _KEYS.get(last) or (last.upper() if len(last) <= 3 and last[0] == "f" and last[1:].isdigit() else last)
    argv = ["wtype"]
    for m in mods:
        argv += ["-M", m]
    argv += ["-k", key]
    for m in reversed(mods):
        argv += ["-m", m]
    return argv


async def type_text(text: str) -> ToolResult:
    return await run_ok(build_type_text(text), "Typed it.")


# Window classes of terminals. Omarchy runs its TUIs (org.omarchy.*) in one too.
_TERMINALS = re.compile(
    r"(?<![a-z0-9])(foot|alacritty|kitty|ghostty|wezterm|konsole|xterm|terminator|tilix|"
    r"gnome-terminal|st-256color|urxvt|warp)(?![a-z0-9])|^org\.omarchy\.", re.I)
TYPE_LIMIT = 300  # longer text is never typed into a terminal


def focused_is_terminal() -> bool:
    """True when the focused window is a terminal, or when that cannot be told."""
    from hark.tools import hyprland

    try:
        c = hyprland.hypr_query("activewindow") or {}
    except (OSError, ValueError):
        return True
    return bool(_TERMINALS.search(str(c.get("class") or c.get("initialClass") or "")))


def type_text_safety(args: dict) -> str:
    """Typing into a terminal, or pressing Enter, can run or send anything: ask first."""
    text = str(args.get("text", ""))
    if "\n" in text or "\r" in text or focused_is_terminal():
        if len(text) > TYPE_LIMIT:
            raise ValueError("that is too long to type into a terminal or with Enter")
        return "destructive"
    return "reversible"


def type_text_question(args: dict) -> str:
    text = str(args.get("text", "")).replace("\r\n", "\n").replace("\r", "\n")
    spoken = " [Enter] ".join(text.split("\n")).strip()
    where = "the terminal" if focused_is_terminal() else "the focused window"
    return f"Type {spoken} into {where}?"


async def press_keys(combo: str) -> ToolResult:
    return await run_ok(build_press_keys(combo), f"Pressed {combo}.")


# --- clipboard ---

async def clipboard_get() -> ToolResult:
    rc, out, _ = await run(["wl-paste", "--no-newline", "--type", "text"], timeout=2.0)
    if rc != 0 or not out:
        return ToolResult(True, "The clipboard is empty.", "")
    return ToolResult(True, out[:500], out)


def build_clipboard_set(text: str) -> tuple[list[str], bytes]:
    return ["wl-copy"], text.encode()


async def clipboard_set(text: str) -> ToolResult:
    # Text goes on stdin so it never shows in the process list. wl-copy forks a child
    # that holds the selection and keeps any output pipe open, so give it none.
    argv, data = build_clipboard_set(text)
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    await asyncio.wait_for(proc.communicate(data), 2.0)
    ok = proc.returncode == 0
    return ToolResult(ok, "Copied to the clipboard." if ok else "Copy failed.")


# --- screenshot, notify, url, time ---

def screenshot_dir() -> Path:
    env = os.environ.get("OMARCHY_SCREENSHOT_DIR") or os.environ.get("XDG_PICTURES_DIR")
    return Path(env) if env else Path.home() / "Pictures"


def build_screenshot(now: dt.datetime | None = None) -> tuple[list[str], Path]:
    now = now or dt.datetime.now()
    path = screenshot_dir() / f"hark-screenshot-{now:%Y%m%d-%H%M%S}.png"
    return ["grim", str(path)], path


async def screenshot() -> ToolResult:
    argv, path = build_screenshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    result = await run_ok(argv, "Screenshot saved.", timeout=5.0)
    if result.ok:
        result.data = {"path": str(path)}
    return result


def build_notify(title: str, body: str = "") -> list[str]:
    return ["notify-send", "--app-name=Hark", "--", title, body]


async def notify(title: str, body: str = "") -> ToolResult:
    return await run_ok(build_notify(title, body), "Notification sent.")


def build_open_url(url: str) -> list[str]:
    if "://" not in url and re.match(r"^[a-z][a-z0-9+.-]*:(?!\d)", url, re.I):
        raise ValueError("only http and https web addresses can be opened")
    parsed = urlparse(url if "://" in url else "https://" + url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("only http and https web addresses can be opened")
    return ["omarchy-launch-browser", parsed.geturl()]


async def open_url(url: str) -> ToolResult:
    spawn(build_open_url(url))
    return ToolResult(True, "Opened it in the browser.")


async def current_time(part: str = "both") -> ToolResult:
    now = dt.datetime.now().astimezone()
    clock = now.strftime("%-I:%M %p")
    day = now.strftime("%A, %B %-d")
    text = {"time": f"It's {clock}.", "date": f"It's {day}."}.get(part, f"It's {clock} on {day}.")
    return ToolResult(True, text, {"iso": now.isoformat(timespec="seconds")})


# --- lock and power ---

LOCK_CMD = ["omarchy-system-lock"]
POWER_CMDS = {
    "suspend": ["systemctl", "suspend"],
    "reboot": ["omarchy-system", "reboot"],
    "shutdown": ["omarchy-system", "shutdown"],
}


async def lock_screen() -> ToolResult:
    spawn(LOCK_CMD)
    return ToolResult(True, "Locked.")


async def power(action: str) -> ToolResult:
    argv = POWER_CMDS.get(action)
    if argv is None:
        return ToolResult(False, f"Unknown power action {action}.")
    spawn(argv)
    return ToolResult(True, f"Okay, {action}.")


def _schema(props: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


def tools() -> list[Tool]:
    s = {"type": "string"}
    return [
        Tool("type_text", "Type text into the focused window as if on the keyboard.",
             _schema({"text": s}, ["text"]), "reversible", type_text,
             type_text_question, type_text_safety),
        Tool("press_keys", "Press a key combo in the focused window, like ctrl+t or enter.",
             _schema({"combo": s}, ["combo"]), "destructive", press_keys,
             lambda a: f"Press {a.get('combo', 'those keys')}?"),
        Tool("clipboard_get", "Read the text on the clipboard.", _schema(), "read", clipboard_get),
        Tool("clipboard_set", "Put text on the clipboard.", _schema({"text": s}, ["text"]),
             "reversible", clipboard_set),
        Tool("screenshot", "Take a screenshot of the whole screen and save it to a file.",
             _schema(), "reversible", screenshot),
        Tool("notify", "Show a desktop notification.",
             _schema({"title": s, "body": s}, ["title"]), "reversible", notify),
        Tool("open_url", "Open a web address in the browser.", _schema({"url": s}, ["url"]),
             "reversible", open_url),
        Tool("current_time", "Get the current time and date.",
             _schema({"part": {"type": "string", "enum": ["time", "date", "both"]}}),
             "read", current_time),
        Tool("lock_screen", "Lock the screen.", _schema(), "reversible", lock_screen),
        Tool("power", "Suspend, reboot or shut down the computer.",
             _schema({"action": {"type": "string", "enum": list(POWER_CMDS)}}, ["action"]),
             "destructive", power, lambda a: f"Really {a.get('action', 'power off')} the computer?"),
    ]
