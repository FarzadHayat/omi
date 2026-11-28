"""PipeWire audio through the pw-cat and pw-record subprocesses.

A subprocess gives hard barge-in for free: killing it silences the stream at once, and no
Python audio library has to live in the event loop. Each keeps the tail of its stderr, so
a missing speaker or microphone ("no target node available") becomes a clear error
instead of silence.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import shutil

import numpy as np

CHUNK = 512  # samples, 32 ms at 16 kHz
IN_RATE = 16000
DUCK_VOLUME = "40%"  # about -24 dB: still audible, out of the way of the user's voice


async def _collect(stream: asyncio.StreamReader | None, tail: collections.deque[str]) -> None:
    if stream is None:
        return
    while line := await stream.readline():
        tail.append(line.decode(errors="replace").strip())


def _failure(what: str, rc: int | None, tail: collections.deque[str]) -> str:
    detail = next((t for t in reversed(tail) if t), "") or f"exit status {rc}"
    if "no target node" in detail:
        detail = f"no {what} device is available ({detail})"
    return detail


class PipeWireOut:
    """Streams raw f32 mono into `pw-cat --playback`. Empty sink means the default."""

    def __init__(self, sink: str = "") -> None:
        self.sink = sink
        self.error: str | None = None  # why the last reply could not be played
        self._proc: asyncio.subprocess.Process | None = None
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._pump: asyncio.Task | None = None
        self._stderr: collections.deque[str] = collections.deque(maxlen=5)
        self._errs: asyncio.Task | None = None

    async def start(self, sample_rate: int) -> None:
        await self.stop()
        exe = shutil.which("pw-cat") or shutil.which("pw-play")
        if exe is None:
            raise RuntimeError("pw-cat not found; install pipewire")
        cmd = [exe, "--playback", "--raw", "--format", "f32", "--rate", str(sample_rate),
               "--channels", "1", "--latency", "20ms", "--media-role", "Assistant",
               "-P", "application.name=Omi", "-P", "node.name=omi-playback",
               # A lowered voice (duck) must not be remembered for the next reply.
               "-P", "state.restore-props=false"]
        if self.sink:
            cmd += ["--target", self.sink]
        cmd.append("-")
        self._queue = asyncio.Queue()
        self._stderr = collections.deque(maxlen=5)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        self._pump = asyncio.create_task(self._run(self._proc, self._queue))
        self._errs = asyncio.create_task(_collect(self._proc.stderr, self._stderr))

    @staticmethod
    async def _run(proc: asyncio.subprocess.Process, queue: asyncio.Queue) -> None:
        """Feed the pipe from a queue so write() never waits on pipe back-pressure."""
        stdin = proc.stdin
        assert stdin is not None
        try:
            while (data := await queue.get()) is not None:
                stdin.write(data)
                await stdin.drain()
            stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            pass

    async def write(self, pcm: np.ndarray) -> None:
        if self._proc is not None:
            self._queue.put_nowait(np.ascontiguousarray(pcm, dtype="<f4").tobytes())

    async def drain(self) -> None:
        proc, pump = self._proc, self._pump
        if proc is None or pump is None:
            return
        self._queue.put_nowait(None)
        await pump
        rc = await proc.wait()  # pw-cat plays out its buffer at EOF, then exits
        if self._errs is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._errs, 1.0)
        self._proc = self._pump = self._errs = None
        if rc != 0:
            self.error = _failure("audio output", rc, self._stderr)
            raise RuntimeError(f"nothing was played: {self.error}")
        self.error = None

    async def duck(self, on: bool) -> None:
        """Lower Omi's voice at once while the user talks over it, or bring it back.
        Queued audio is already in the stream, so this sets the stream volume itself."""
        if self._proc is None or shutil.which("pactl") is None:
            return
        listing = await asyncio.create_subprocess_exec(
            "pactl", "-f", "json", "list", "sink-inputs",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await listing.communicate()
        try:
            streams = json.loads(out or b"[]")
        except ValueError:
            return
        for stream in streams:
            if (stream.get("properties") or {}).get("node.name") == "omi-playback":
                setter = await asyncio.create_subprocess_exec(
                    "pactl", "set-sink-input-volume", str(stream["index"]), DUCK_VOLUME if on else "100%",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await setter.wait()

    async def stop(self) -> None:
        proc, pump = self._proc, self._pump
        self._proc = self._pump = None
        if proc is None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        if pump is not None:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
        await proc.wait()
        if self._errs is not None:
            self._errs.cancel()
            self._errs = None


class PipeWireIn:
    """Reads raw f32 mono 16 kHz from `pw-record`. Empty source means the default mic."""

    def __init__(self, source: str = "", *, keep_chunks: bool = True) -> None:
        self.source = source
        self.keep_chunks = keep_chunks
        self.error: str | None = None  # why the last recording got no audio
        self._proc: asyncio.subprocess.Process | None = None
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._chunks: list[np.ndarray] = []
        self._reader: asyncio.Task | None = None
        self._stderr: collections.deque[str] = collections.deque(maxlen=5)
        self._errs: asyncio.Task | None = None

    async def start(self) -> None:
        await self.stop()
        exe = shutil.which("pw-record")
        if exe is None:
            raise RuntimeError("pw-record not found; install pipewire")
        cmd = [exe, "--raw", "--format", "f32", "--rate", str(IN_RATE), "--channels", "1",
               "--latency", "32ms", "--media-role", "Communication",
               "-P", "application.name=Omi", "-P", "node.name=omi-capture"]
        if self.source.endswith(".monitor"):
            # pactl-style monitor name: capture the sink's output, which is what tests record.
            cmd += ["--target", self.source.removesuffix(".monitor"), "-P", "stream.capture.sink=true"]
        elif self.source:
            cmd += ["--target", self.source]
        cmd.append("-")
        self._queue = asyncio.Queue()
        self._chunks = []
        self._stderr = collections.deque(maxlen=5)
        self.error = None
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        self._reader = asyncio.create_task(self._read_loop(self._proc))
        self._errs = asyncio.create_task(_collect(self._proc.stderr, self._stderr))

    async def _read_loop(self, proc: asyncio.subprocess.Process) -> None:
        stdout = proc.stdout
        assert stdout is not None
        size = CHUNK * 4
        try:
            while True:
                data = await stdout.readexactly(size)
                chunk = np.frombuffer(data, dtype="<f4").copy()
                if self.keep_chunks:
                    self._chunks.append(chunk)
                self._queue.put_nowait(chunk)
        except asyncio.IncompleteReadError as err:
            if err.partial:
                chunk = np.frombuffer(err.partial[: len(err.partial) // 4 * 4], dtype="<f4").copy()
                if self.keep_chunks:
                    self._chunks.append(chunk)
                self._queue.put_nowait(chunk)
        finally:
            self._queue.put_nowait(None)

    async def read(self) -> np.ndarray | None:
        chunk = await self._queue.get()
        if chunk is None:
            self._queue.put_nowait(None)  # stay at end for later callers
        return chunk

    async def stop(self) -> np.ndarray:
        proc, reader, errs = self._proc, self._reader, self._errs
        self._proc = self._reader = self._errs = None
        if proc is not None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            if reader is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await reader
            rc = await proc.wait()
            if errs is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(errs, 1.0)
            if rc not in (0, -9):  # -9 is our kill; anything else means pw-record failed
                self.error = _failure("microphone", rc, self._stderr)
        chunks, self._chunks = self._chunks, []
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def parse_mic_users(outputs: list[dict], sources: str) -> list[str]:
    """Apps other than Omi recording a microphone, from `pactl -f json list source-outputs`
    and `pactl list short sources`. Desktop-audio recorders (".monitor" sources), level
    meters, paused streams and the echo canceller's own capture do not count."""
    monitors = set()
    for line in sources.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].endswith(".monitor"):
            monitors.add(parts[0].strip())
    users: list[str] = []
    for out in outputs:
        props = out.get("properties") or {}
        node = str(props.get("node.name") or "")
        app = str(props.get("application.name") or props.get("application.process.binary") or "")
        if out.get("corked") or str(out.get("source")) in monitors:
            continue
        if app == "Omi" or node.startswith(("omi-", "echo-cancel")):
            continue
        if str(props.get("stream.monitor") or "").lower() == "true":
            continue
        users.append(app or node or "another app")
    return list(dict.fromkeys(users))


def mic_users() -> list[str]:
    """Apps other than Omi using a microphone right now (a call, dictation). Empty when
    PipeWire cannot be asked."""
    import json
    import subprocess

    if shutil.which("pactl") is None:
        return []
    try:
        outs = subprocess.run(["pactl", "-f", "json", "list", "source-outputs"],
                              capture_output=True, text=True, timeout=2.0, check=True).stdout
        srcs = subprocess.run(["pactl", "list", "short", "sources"],
                              capture_output=True, text=True, timeout=2.0, check=True).stdout
        return parse_mic_users(json.loads(outs or "[]"), srcs)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
