"""Developer tools: terminal control (Herdr / tmux) and T3 Code control (`t3` CLI).

Pure `build_*` functions return exact argv lists for testing. Every terminal command
passes through `omi.tools.safety` so secrets are blocked and non-read commands ask
the user for confirmation aloud before running.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from omi.tools import hyprland, safety
from omi.tools.base import Safety, Tool, ToolResult
from omi.tools.system import run, spawn

HERDR_SOCK = Path.home() / ".config/herdr/herdr.sock"
T3_RUNTIME = Path.home() / ".t3/runtime"


# --- backend discovery ---


def find_herdr_binary() -> str | None:
    return shutil.which("herdr")


def herdr_available() -> bool:
    return bool(find_herdr_binary()) and HERDR_SOCK.exists()


def find_tmux_binary() -> str | None:
    return shutil.which("tmux")


def find_t3_binary() -> str | None:
    """Locate the `t3` CLI matching the running T3 Code runtime if present, else PATH."""
    if env := os.environ.get("T3_BIN"):
        if os.access(env, os.X_OK):
            return env
    state_file = T3_RUNTIME / "service-state.json"
    if state_file.is_file():
        try:
            ver = json.loads(state_file.read_text(encoding="utf-8")).get("activeVersion")
            if isinstance(ver, str) and ver:
                cand = T3_RUNTIME / "versions" / ver / "t3"
                if cand.is_file() and os.access(cand, os.X_OK):
                    return str(cand)
        except (OSError, ValueError):
            pass
    versions_dir = T3_RUNTIME / "versions"
    if versions_dir.is_dir():
        cands = sorted(versions_dir.glob("*/t3"), reverse=True)
        for cand in cands:
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    return shutil.which("t3")


def _expand_path(path: str) -> str:
    return str(Path(os.path.expandvars(os.path.expanduser(path))).resolve()) if path else ""


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _parse_json(text: str) -> Any:
    raw = text.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        for line in reversed(raw.splitlines()):
            line = line.strip()
            if line.startswith(("{", "[")):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
    return None


# --- terminal command builders ---


def build_herdr_list(workspace: str = "") -> list[str]:
    argv = ["herdr", "pane", "list"]
    if workspace:
        argv += ["--workspace", workspace]
    return argv


def build_herdr_read(pane: str, lines: int = 40) -> list[str]:
    return ["herdr", "pane", "read", "--source", "recent", "--lines", str(_clamp(lines, 1, 200)), pane]


def build_herdr_run(pane: str, command: str) -> list[str]:
    return ["herdr", "pane", "run", pane, "--", command]


def build_herdr_split(directory: str = "", direction: str = "right") -> list[str]:
    argv = ["herdr", "pane", "split", "--current", "--focus"]
    if direction in ("right", "down"):
        argv += ["--direction", direction]
    if directory:
        argv += ["--cwd", _expand_path(directory)]
    return argv


def build_tmux_list() -> list[str]:
    return [
        "tmux", "list-panes", "-a", "-F",
        "#{session_name}:#{window_index}.#{pane_index}\t#{pane_current_path}\t#{pane_current_command}\t#{pane_title}\t#{pane_active}",
    ]


def build_tmux_read(pane: str = "", lines: int = 40) -> list[str]:
    argv = ["tmux", "capture-pane", "-p", "-S", f"-{_clamp(lines, 1, 200)}"]
    if pane:
        argv += ["-t", pane]
    return argv


def build_tmux_send(pane: str, command: str) -> list[list[str]]:
    target = ["-t", pane] if pane else []
    return [
        ["tmux", "send-keys", *target, "-l", "--", command],
        ["tmux", "send-keys", *target, "Enter"],
    ]


# --- terminal safety ---


def terminal_command_safety(args: dict[str, Any]) -> Safety:
    cmd = str(args.get("command", "")).strip()
    if not cmd:
        return "reversible"
    if reason := safety.secret_reason(cmd):
        raise ValueError(f"blocked because {reason}")
    if len(cmd) > 400:
        raise ValueError("that terminal command is too long")
    level = safety.classify_command(cmd)
    return "reversible" if level == "read" else "destructive"


def terminal_run_question(args: dict[str, Any]) -> str:
    cmd = " ".join(str(args.get("command", "")).strip().split())
    if len(cmd) > 120:
        cmd = cmd[:117] + "..."
    return f"Run {cmd} in the terminal?"


def terminal_open_question(args: dict[str, Any]) -> str:
    cmd = " ".join(str(args.get("command", "")).strip().split())
    if len(cmd) > 120:
        cmd = cmd[:117] + "..."
    return f"Open a terminal and run {cmd}?" if cmd else "Open a terminal?"


# --- terminal tools ---


async def _resolve_herdr_pane(pane: str) -> str | None:
    if pane:
        return pane
    rc, out, _ = await run(["herdr", "pane", "current"], timeout=3.0)
    if rc == 0:
        data = _parse_json(out)
        if isinstance(data, dict):
            pid = ((data.get("result") or {}).get("pane") or {}).get("pane_id")
            if pid:
                return str(pid)
    rc, out, _ = await run(["herdr", "pane", "list"], timeout=3.0)
    if rc == 0:
        data = _parse_json(out)
        panes = ((data or {}).get("result") or {}).get("panes") or []
        if panes:
            focused = next((p for p in panes if p.get("focused")), panes[0])
            return str(focused.get("pane_id", "")) or None
    return None


async def _tmux_active() -> bool:
    if not find_tmux_binary():
        return False
    rc, _, _ = await run(["tmux", "list-sessions"], timeout=2.0)
    return rc == 0


async def terminal_list(workspace: str = "") -> ToolResult:
    if herdr_available():
        rc, out, err = await run(build_herdr_list(workspace), timeout=4.0)
        if rc != 0:
            return ToolResult(False, f"Could not list Herdr panes: {(err or out).strip()}")
        data = _parse_json(out) or {}
        panes = ((data.get("result") or {}).get("panes")) or []
        if not panes:
            return ToolResult(True, "No open terminal panes in Herdr.", {"backend": "herdr", "panes": []})
        items = [
            {
                "pane": p.get("pane_id", ""),
                "workspace": p.get("workspace_id", ""),
                "cwd": p.get("foreground_cwd") or p.get("cwd", ""),
                "title": p.get("terminal_title_stripped") or p.get("terminal_title", ""),
                "focused": bool(p.get("focused")),
            }
            for p in panes
        ]
        summary = "; ".join(
            f"{p['pane']} ({Path(p['cwd']).name or p['cwd']}{', focused' if p['focused'] else ''})"
            for p in items[:6]
        )
        return ToolResult(True, f"{len(items)} terminal pane(s) in Herdr: {summary}.",
                          {"backend": "herdr", "panes": items})

    if await _tmux_active():
        rc, out, err = await run(build_tmux_list(), timeout=3.0)
        if rc != 0:
            return ToolResult(False, f"Could not list tmux panes: {(err or out).strip()}")
        items = []
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) >= 5:
                items.append({
                    "pane": parts[0], "cwd": parts[1], "command": parts[2],
                    "title": parts[3], "focused": parts[4] == "1",
                })
        summary = "; ".join(f"{p['pane']} ({p['command']} in {Path(p['cwd']).name})" for p in items[:6])
        return ToolResult(True, f"{len(items)} tmux pane(s): {summary}." if items else "No tmux panes.",
                          {"backend": "tmux", "panes": items})

    return ToolResult(False, "No Herdr or tmux terminal session is running. Use terminal_open to start a terminal.")


async def terminal_read(pane: str = "", lines: int = 40) -> ToolResult:
    n = _clamp(lines, 1, 200)
    if herdr_available():
        pid = await _resolve_herdr_pane(pane)
        if not pid:
            return ToolResult(False, "No open Herdr terminal pane found.")
        rc, out, err = await run(build_herdr_read(pid, n), timeout=4.0)
        if rc != 0:
            return ToolResult(False, f"Could not read pane {pid}: {(err or out).strip()}")
        text = out.strip()
        tail = "\n".join(text.splitlines()[-n:]) if text else "(empty pane)"
        return ToolResult(True, f"Output from terminal pane {pid}:\n{tail[-3000:]}",
                          {"backend": "herdr", "pane": pid, "output": tail})

    if await _tmux_active():
        rc, out, err = await run(build_tmux_read(pane, n), timeout=3.0)
        if rc != 0:
            return ToolResult(False, f"Could not read tmux pane: {(err or out).strip()}")
        text = out.strip()
        tail = "\n".join(text.splitlines()[-n:]) if text else "(empty pane)"
        return ToolResult(True, f"Output from tmux pane {pane or 'current'}:\n{tail[-3000:]}",
                          {"backend": "tmux", "pane": pane or "current", "output": tail})

    return ToolResult(False, "No Herdr or tmux terminal session is running.")


async def terminal_run(command: str, pane: str = "") -> ToolResult:
    cmd = command.strip()
    if not cmd:
        return ToolResult(False, "No command given.")
    if reason := safety.secret_reason(cmd):
        return ToolResult(False, f"Blocked because {reason}.")

    if herdr_available():
        pid = await _resolve_herdr_pane(pane)
        if not pid:
            return ToolResult(False, "No open Herdr terminal pane found.")
        rc, out, err = await run(build_herdr_run(pid, cmd), timeout=5.0)
        if rc != 0:
            return ToolResult(False, f"Could not run command in pane {pid}: {(err or out).strip()}")
        return ToolResult(True, f"Sent command to terminal pane {pid}.", {"backend": "herdr", "pane": pid})

    if await _tmux_active():
        for argv in build_tmux_send(pane, cmd):
            rc, out, err = await run(argv, timeout=3.0)
            if rc != 0:
                return ToolResult(False, f"Could not send command to tmux: {(err or out).strip()}")
        return ToolResult(True, f"Sent command to tmux pane {pane or 'current'}.",
                          {"backend": "tmux", "pane": pane or "current"})

    return ToolResult(False, "No Herdr or tmux terminal session is running. Open one first with terminal_open.")


async def terminal_open(directory: str = "", command: str = "", direction: str = "right") -> ToolResult:
    cmd = command.strip()
    if cmd and (reason := safety.secret_reason(cmd)):
        return ToolResult(False, f"Blocked because {reason}.")
    cwd = _expand_path(directory) if directory else ""

    if herdr_available():
        rc, out, err = await run(build_herdr_split(cwd, direction), timeout=5.0)
        if rc == 0:
            data = _parse_json(out) or {}
            pid = ((data.get("result") or {}).get("pane") or {}).get("pane_id", "")
            if cmd and pid:
                await run(build_herdr_run(str(pid), cmd), timeout=5.0)
            return ToolResult(True, f"Opened a new terminal pane{f' ({pid})' if pid else ''}.",
                              {"backend": "herdr", "pane": pid})
        # Fall back to launching a terminal window if Herdr has no active window to split.

    if await _tmux_active():
        argv = ["tmux", "split-window", "-h" if direction != "down" else "-v"]
        if cwd:
            argv += ["-c", cwd]
        rc, out, err = await run(argv, timeout=3.0)
        if rc == 0:
            if cmd:
                for step in build_tmux_send("", cmd):
                    await run(step, timeout=3.0)
            return ToolResult(True, "Opened a new tmux pane.", {"backend": "tmux"})

    term = shutil.which("omarchy-launch-terminal") or shutil.which("ghostty") or shutil.which("alacritty") or shutil.which("foot")
    if not term:
        return ToolResult(False, "Could not find a terminal launcher on this system.")
    argv = [term]
    if cwd and Path(term).name != "omarchy-launch-terminal":
        argv += ["--working-directory", cwd]
    spawn(argv)
    return ToolResult(True, "Opened the terminal.", {"backend": Path(term).name})


# --- T3 Code command builders ---


def build_t3_threads(t3_bin: str, project: str = "") -> list[str]:
    argv = [t3_bin, "thread", "list", "--json"]
    if project:
        argv += ["--project", _expand_path(project) if project.startswith(("~", "/", ".")) else project]
    return argv


def build_t3_read(t3_bin: str, thread: str, messages: int = 5) -> list[str]:
    return [t3_bin, "thread", "show", "--json", "--messages", str(_clamp(messages, 1, 20)), thread]


def build_t3_start(t3_bin: str, project: str, message: str, title: str = "") -> list[str]:
    proj = _expand_path(project) if project.startswith(("~", "/", ".")) else project
    argv = [t3_bin, "thread", "start", "--json"]
    if title:
        argv += ["--title", title]
    else:
        argv.append("--generate-title")
    argv += [proj, message]
    return argv


def build_t3_send(t3_bin: str, thread: str, message: str) -> list[str]:
    return [t3_bin, "thread", "send", "--json", thread, message]


def build_t3_open(t3_bin: str, project: str) -> list[str]:
    proj = _expand_path(project) if project.startswith(("~", "/", ".")) else project
    return [t3_bin, "app", proj]


# --- T3 Code tools ---


def _require_t3() -> str | None:
    return find_t3_binary()


async def t3_threads(project: str = "", limit: int = 8) -> ToolResult:
    t3 = _require_t3()
    if not t3:
        return ToolResult(False, "T3 Code CLI is not installed on this machine.")
    rc, out, err = await run(build_t3_threads(t3, project), timeout=15.0)
    if rc != 0 and not project.strip():
        scratch = Path.home() / ".t3/scratch"
        if scratch.is_dir():
            rc, out, err = await run(build_t3_threads(t3, str(scratch)), timeout=15.0)
    if rc != 0:
        msg = (err or out).strip().splitlines()
        return ToolResult(False, f"Could not list T3 Code threads: {msg[0] if msg else 'exit ' + str(rc)}")
    data = _parse_json(out)
    threads = data if isinstance(data, list) else (data.get("threads") if isinstance(data, dict) else [])
    threads = threads or []
    n = _clamp(limit, 1, 25)
    slim = [
        {
            "id": t.get("threadId") or t.get("id", ""),
            "title": t.get("title", "Untitled"),
            "status": t.get("status", ""),
            "model": t.get("model", ""),
            "projectId": t.get("projectId", ""),
            "updatedAt": t.get("updatedAt", ""),
        }
        for t in threads[:n]
    ]
    if not slim:
        return ToolResult(True, "No T3 Code threads found.", {"threads": []})
    spoken = "; ".join(f"{t['title']} ({t['status']}, id {t['id'][:8]})" for t in slim[:5])
    return ToolResult(True, f"{len(threads)} T3 Code threads. Most recent: {spoken}.", {"threads": slim})


async def t3_read(thread: str, messages: int = 5) -> ToolResult:
    t3 = _require_t3()
    if not t3:
        return ToolResult(False, "T3 Code CLI is not installed on this machine.")
    if not thread.strip():
        return ToolResult(False, "Give a T3 Code thread ID or title.")
    rc, out, err = await run(build_t3_read(t3, thread.strip(), messages), timeout=15.0)
    if rc != 0:
        msg = (err or out).strip().splitlines()
        return ToolResult(False, f"Could not read T3 Code thread: {msg[0] if msg else 'exit ' + str(rc)}")
    data = _parse_json(out)
    if isinstance(data, dict):
        title = data.get("title") or (data.get("thread") or {}).get("title") or thread
        status = data.get("status") or (data.get("thread") or {}).get("status") or ""
        msgs = data.get("messages") or data.get("items") or []
        lines = []
        for m in msgs[-_clamp(messages, 1, 20):]:
            role = m.get("role") or m.get("kind") or "msg"
            text = str(m.get("text") or m.get("content") or "").strip()
            if text:
                lines.append(f"{role}: {text[:500]}")
        body = "\n".join(lines) if lines else out.strip()[:2000]
        return ToolResult(True, f"Thread {title} ({status}):\n{body[-3000:]}", data)
    return ToolResult(True, out.strip()[-3000:] or "Empty thread.", {"raw": out})


async def t3_start(project: str, message: str, title: str = "") -> ToolResult:
    t3 = _require_t3()
    if not t3:
        return ToolResult(False, "T3 Code CLI is not installed on this machine.")
    if not project.strip() or not message.strip():
        return ToolResult(False, "Provide both a project path or ID and a message.")
    rc, out, err = await run(build_t3_start(t3, project.strip(), message.strip(), title.strip()), timeout=20.0)
    if rc != 0:
        msg = (err or out).strip().splitlines()
        return ToolResult(False, f"Could not start T3 Code thread: {msg[0] if msg else 'exit ' + str(rc)}")
    data = _parse_json(out)
    tid = (data.get("threadId") or data.get("id") or "") if isinstance(data, dict) else ""
    label = (data.get("title") or title or tid[:8]) if isinstance(data, dict) else title
    return ToolResult(True, f"Started a new T3 Code thread{f' ({label})' if label else ''}.",
                      data if isinstance(data, dict) else {"raw": out})


async def t3_send(thread: str, message: str) -> ToolResult:
    t3 = _require_t3()
    if not t3:
        return ToolResult(False, "T3 Code CLI is not installed on this machine.")
    if not thread.strip() or not message.strip():
        return ToolResult(False, "Provide both a thread ID and a message.")
    rc, out, err = await run(build_t3_send(t3, thread.strip(), message.strip()), timeout=20.0)
    if rc != 0:
        msg = (err or out).strip().splitlines()
        return ToolResult(False, f"Could not message T3 Code thread: {msg[0] if msg else 'exit ' + str(rc)}")
    data = _parse_json(out)
    return ToolResult(True, f"Sent your message to T3 Code thread {thread.strip()[:8]}.",
                      data if isinstance(data, dict) else {"raw": out})


async def t3_open(project: str = "") -> ToolResult:
    if project.strip():
        t3 = _require_t3()
        if not t3:
            return ToolResult(False, "T3 Code CLI is not installed on this machine.")
        rc, out, err = await run(build_t3_open(t3, project.strip()), timeout=10.0)
        if rc != 0:
            msg = (err or out).strip().splitlines()
            return ToolResult(False, f"Could not open project in T3 Code: {msg[0] if msg else 'exit ' + str(rc)}")
        return ToolResult(True, f"Opened {Path(project.strip()).name or project.strip()} in T3 Code.")
    return await hyprland.launch_or_focus("t3code")


# --- tool schemas ---


def _schema(props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props or {}, "required": required or []}


def tools() -> list[Tool]:
    s = {"type": "string"}
    i = {"type": "integer"}
    return [
        Tool("terminal_list", "List open terminal panes in Herdr or tmux with their IDs and working directories.",
             _schema({"workspace": s}), "read", terminal_list),
        Tool("terminal_read", "Read recent text output from a Herdr or tmux terminal pane.",
             _schema({"pane": s, "lines": i}), "read", terminal_read),
        Tool("terminal_run", "Run a shell command inside a live Herdr or tmux terminal pane.",
             _schema({"command": s, "pane": s}, ["command"]), "reversible", terminal_run,
             terminal_run_question, terminal_command_safety),
        Tool("terminal_open", "Open or split a terminal pane (Herdr, tmux, or system terminal), optionally in a directory.",
             _schema({"directory": s, "command": s, "direction": {"type": "string", "enum": ["right", "down"]}}),
             "reversible", terminal_open, terminal_open_question, terminal_command_safety),
        Tool("t3_threads", "List T3 Code threads for a project folder path or ID (defaults to current or Scratch project).",
             _schema({"project": s, "limit": i}), "read", t3_threads),
        Tool("t3_read", "Read recent messages and status of a T3 Code thread by ID.",
             _schema({"thread": s, "messages": i}, ["thread"]), "read", t3_read),
        Tool("t3_start", "Start a new T3 Code thread in a project directory or ID with an initial prompt.",
             _schema({"project": s, "message": s, "title": s}, ["project", "message"]), "reversible", t3_start),
        Tool("t3_send", "Send a follow-up message to an existing T3 Code thread.",
             _schema({"thread": s, "message": s}, ["thread", "message"]), "reversible", t3_send),
        Tool("t3_open", "Focus T3 Code or open a project folder inside the T3 Code desktop app.",
             _schema({"project": s}), "reversible", t3_open),
    ]
