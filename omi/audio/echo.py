"""PipeWire echo cancellation, so Omi only hears the person, never the speakers.

A small PipeWire process runs the WebRTC echo canceller in monitor mode: its reference is
everything the speakers play (Omi's voice, a video, music), and it removes that from what
the microphone hears. Cancelling only Omi's own voice is not enough, since a video playing
on the computer would still sound like the user talking. Measured on HDMI speakers and a
webcam mic: a spoken clip played on the computer drops from 11 s of detected speech to
none. That is also what makes talking over Omi (barge-in) safe without headphones.

The process is restarted on every Omi start, so it follows the current speakers, and is
stopped when Omi exits.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

SOURCE = "omi_echo_cancel_source"
_LEGACY_ARG = f"source_name={SOURCE}"  # the PulseAudio module older versions loaded
_proc: subprocess.Popen | None = None


def _runtime_dir() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir())


def _pidfile() -> Path:
    return _runtime_dir() / "omi-echo-cancel.pid"


def _names(kind: str) -> set[str]:
    out = subprocess.run(["pactl", "list", "short", kind], capture_output=True, text=True, timeout=5).stdout
    return {line.split("\t")[1] for line in out.splitlines() if "\t" in line}


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_conf(source_master: str, speakers: str) -> str:
    """A standalone PipeWire config hosting the echo canceller in monitor mode."""
    capture = '{ node.name = "omi-echo-capture" node.passive = true'
    if source_master:
        capture += f" target.object = {_quote(source_master)}"
    reference = f'{{ node.name = "omi-echo-reference" target.object = {_quote(speakers)} }}'
    return f"""context.properties = {{ log.level = 1 }}
context.spa-libs = {{
  audio.convert.* = audioconvert/libspa-audioconvert
  support.*       = support/libspa-support
}}
context.modules = [
  {{ name = libpipewire-module-rt flags = [ ifexists nofail ] }}
  {{ name = libpipewire-module-protocol-native }}
  {{ name = libpipewire-module-client-node }}
  {{ name = libpipewire-module-adapter }}
  {{ name = libpipewire-module-echo-cancel
    args = {{
      library.name = aec/libspa-aec-webrtc
      monitor.mode = true
      capture.props = {capture} }}
      sink.props = {reference}
      source.props = {{ node.name = "{SOURCE}" node.description = "Omi microphone (echo cancelled)" }}
    }}
  }}
]
"""


def _default_sink() -> str:
    return subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=5).stdout.strip()


def _unload_legacy() -> None:
    """Older Omi left a PulseAudio echo-cancel module loaded under the same source name."""
    out = subprocess.run(["pactl", "list", "short", "modules"], capture_output=True, text=True, timeout=5).stdout
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] == "module-echo-cancel" and _LEGACY_ARG in parts[2]:
            subprocess.run(["pactl", "unload-module", parts[0]], capture_output=True, timeout=5)


def _kill_stale() -> None:
    """A canceller left by an Omi that did not exit cleanly may track old speakers."""
    pidfile = _pidfile()
    try:
        pid = int(pidfile.read_text())
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return
    if b"omi-echo-cancel" in cmdline:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.05)
    pidfile.unlink(missing_ok=True)


def stop() -> None:
    global _proc
    proc, _proc = _proc, None
    if proc is None:
        return
    proc.terminate()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=2)
    _pidfile().unlink(missing_ok=True)


def ensure(source_master: str = "", sink: str = "") -> tuple[str, str] | None:
    """(source, sink) for Omi: the echo-cancelled mic and the speakers it plays to. None
    when PipeWire cannot provide it; Omi then keeps the plain devices and turn-taking."""
    global _proc
    pipewire = shutil.which("pipewire")
    if shutil.which("pactl") is None or pipewire is None:
        return None
    try:
        stop()
        _kill_stale()
        _unload_legacy()
        speakers = sink or _default_sink()
        if not speakers:
            log.warning("echo cancellation unavailable: no speakers to listen to")
            return None
        conf = _runtime_dir() / "omi-echo-cancel.conf"
        conf.write_text(build_conf(source_master, speakers))
        _proc = subprocess.Popen([pipewire, "-c", str(conf)], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _pidfile().write_text(str(_proc.pid))
        for _ in range(30):  # the source appears a moment after the process starts
            if _proc.poll() is not None:
                break
            if SOURCE in _names("sources"):
                return SOURCE, sink
            time.sleep(0.1)
        log.warning("echo cancellation unavailable: the PipeWire echo canceller did not start")
        stop()
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("echo cancellation unavailable: %s", exc)
        stop()
    return None


atexit.register(stop)
