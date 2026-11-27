"""Media (MPRIS over D-Bus with busctl) and output volume.

playerctl is not installed, so players are driven with busctl. Volume goes through
Omarchy's sink resolution, because the default-sink shortcut in wpctl fails on
Bluetooth output and a DSP sink is not where loudness lives.
"""

from __future__ import annotations

import json
import re

from omi.tools.base import Tool, ToolResult
from omi.tools.system import run, run_ok

MPRIS_PATH = "/org/mpris/MediaPlayer2"
MPRIS_PLAYER = "org.mpris.MediaPlayer2.Player"
_METHODS = {"play_pause": "PlayPause", "play": "Play", "pause": "Pause", "next": "Next", "previous": "Previous"}
_DONE = {"play_pause": "", "play": "Playing.", "pause": "Paused.", "next": "Next track.", "previous": "Previous track."}

VOLUME_SCRIPT = "omarchy-audio-output-volume"
SINK_SCRIPT = "omarchy-audio-output-sink"
VOLUME_STEP = 10


# --- MPRIS ---


async def players() -> list[str]:
    rc, out, _ = await run(["busctl", "--user", "list", "--json=short"], timeout=2.0)
    if rc != 0:
        return []
    return [d["name"] for d in json.loads(out) if d.get("name", "").startswith("org.mpris.MediaPlayer2.")]


def build_status_query(player: str) -> list[str]:
    return ["busctl", "--user", "-j", "get-property", player, MPRIS_PATH, MPRIS_PLAYER, "PlaybackStatus", "Metadata"]


def build_media_call(player: str, action: str) -> list[str]:
    return ["busctl", "--user", "call", player, MPRIS_PATH, MPRIS_PLAYER, _METHODS[action]]


def parse_status(out: str) -> tuple[str, dict]:
    """busctl -j prints one JSON object per property."""
    docs = [json.loads(line) for line in out.splitlines() if line.strip()]
    status = docs[0]["data"] if docs else "Stopped"
    meta = {k: v.get("data") for k, v in (docs[1]["data"] if len(docs) > 1 else {}).items()}
    return status, meta


async def _status(player: str) -> tuple[str, dict]:
    rc, out, _ = await run(build_status_query(player), timeout=2.0)
    return parse_status(out) if rc == 0 else ("Stopped", {})


async def pick_player() -> tuple[str, str, dict] | None:
    """The playing player if any, else the first one: the one the user means."""
    found = await players()
    first = None
    for name in found:
        status, meta = await _status(name)
        if status == "Playing":
            return name, status, meta
        first = first or (name, status, meta)
    return first


async def media(action: str) -> ToolResult:
    picked = await pick_player()
    if picked is None:
        return ToolResult(False, "Nothing is playing.")
    name, status, meta = picked
    if action == "now_playing":
        title = meta.get("xesam:title") or "something"
        artists = meta.get("xesam:artist") or []
        by = f" by {', '.join(artists)}" if artists else ""
        state = "Playing" if status == "Playing" else "Paused on"
        return ToolResult(True, f"{state} {title}{by}.", {"player": name, "status": status, "metadata": meta})
    if action not in _METHODS:
        return ToolResult(False, f"Unknown media action {action}.")
    done = _DONE[action] or ("Paused." if status == "Playing" else "Playing.")
    return await run_ok(build_media_call(name, action), done)


# --- volume ---


def parse_volume(out: str) -> int | None:
    m = re.search(r"(\d+)%", out)
    return int(m.group(1)) if m else None


def build_volume(action: str, value: int | None, sink: str = "", current: int | None = None) -> list[list[str]]:
    """The commands for one volume action. set needs the sink's current level to
    turn the target into the relative step the Omarchy script (and its OSD) takes."""
    if action in ("up", "down"):
        step = max(1, min(100, int(value or VOLUME_STEP)))
        return [[VOLUME_SCRIPT, f"{'+' if action == 'up' else '-'}{step}"]]
    if action in ("mute", "unmute"):
        return [["pactl", "set-sink-mute", sink, "1" if action == "mute" else "0"]]
    if action == "set":
        if value is None:
            raise ValueError("set needs a value from 0 to 100")
        target = max(0, min(100, int(value)))
        delta = target - (current or 0)
        if delta == 0:
            return [["pactl", "set-sink-mute", sink, "0"]]
        return [[VOLUME_SCRIPT, f"{'+' if delta > 0 else '-'}{abs(delta)}"]]
    raise ValueError(f"unknown volume action {action}")


async def _sink() -> str:
    rc, out, _ = await run([SINK_SCRIPT], timeout=2.0)
    sink = out.strip()
    if rc != 0 or not sink:
        raise RuntimeError("no audio output to control")
    return sink


async def volume(action: str, value: int | None = None) -> ToolResult:
    sink, current = "", None
    if action in ("mute", "unmute", "set"):
        sink = await _sink()
    if action == "set":
        _, out, _ = await run(["pactl", "get-sink-volume", sink], timeout=2.0)
        current = parse_volume(out)
    for argv in build_volume(action, value, sink, current):
        result = await run_ok(argv, "")
        if not result.ok:
            return result
    text = {"up": "Volume up.", "down": "Volume down.", "mute": "Muted.", "unmute": "Unmuted."}.get(action)
    return ToolResult(True, text or f"Volume {max(0, min(100, int(value or 0)))} percent.")


def tools() -> list[Tool]:
    return [
        Tool("media", "Control the music or video player, or say what is playing.",
             {"type": "object", "properties": {"action": {"type": "string",
              "enum": ["play_pause", "play", "pause", "next", "previous", "now_playing"]}},
              "required": ["action"]},
             "reversible", media),
        Tool("volume", "Change the speaker volume: set to a percent, up, down, mute or unmute.",
             {"type": "object", "properties": {
                 "action": {"type": "string", "enum": ["set", "up", "down", "mute", "unmute"]},
                 "value": {"type": "integer", "minimum": 0, "maximum": 100,
                           "description": "percent for set, step for up and down"}},
              "required": ["action"]},
             "reversible", volume),
    ]
