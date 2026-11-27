"""PipeWire echo cancellation, so Omi can keep listening while it talks.

Omi plays its voice into a virtual sink and records from a virtual source; the WebRTC
canceller in between removes whatever went out through that sink from what the
microphone hears. Measured on monitor speakers and a webcam mic: Omi's own voice drops
from 0.8 s of detected speech to none, while another voice in the room still comes
through. That is what makes talking over Omi (barge-in) safe without headphones.

The module stays loaded after Omi exits and is reused on the next start.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time

log = logging.getLogger(__name__)

SOURCE = "omi_echo_cancel_source"
SINK = "omi_echo_cancel_sink"


def _names(kind: str) -> set[str]:
    out = subprocess.run(["pactl", "list", "short", kind], capture_output=True, text=True, timeout=5).stdout
    return {line.split("\t")[1] for line in out.splitlines() if "\t" in line}


def build_load(source_master: str = "", sink_master: str = "") -> list[str]:
    argv = ["pactl", "load-module", "module-echo-cancel", "aec_method=webrtc",
            f"source_name={SOURCE}", f"sink_name={SINK}"]
    if source_master:
        argv.append(f"source_master={source_master}")
    if sink_master:
        argv.append(f"sink_master={sink_master}")
    return argv


def ensure(source_master: str = "", sink_master: str = "") -> tuple[str, str] | None:
    """(source, sink) of the echo-cancelled pair, loading it when missing. None when
    PipeWire cannot provide it; Omi then keeps the plain devices and turn-taking."""
    if shutil.which("pactl") is None:
        return None
    try:
        if SOURCE in _names("sources") and SINK in _names("sinks"):
            return SOURCE, SINK
        res = subprocess.run(build_load(source_master, sink_master), capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            log.warning("echo cancellation unavailable: %s", (res.stderr or res.stdout).strip())
            return None
        for _ in range(20):  # the nodes appear a moment after the module loads
            if SOURCE in _names("sources") and SINK in _names("sinks"):
                return SOURCE, SINK
            time.sleep(0.1)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("echo cancellation unavailable: %s", exc)
    return None
