"""The hark daemon: one Session behind a Unix socket speaking hark.protocol.

Newline-delimited JSON. Turn commands stream that turn's events back on the same
connection; "subscribe" connections get every event. One turn runs at a time and a
new one cancels the running one (the Session enforces that).

Every connection has its own bounded outbox, so a client that stops reading never
slows a turn or the stop command: once its outbox is full it is dropped.

Only the user answers a yes/no question: a "confirm" must name the question's id and
come from a same-user process that is not one the daemon started (Claude and its
shell are), and while a question waits those processes cannot start turns either.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import struct
from collections.abc import Awaitable, Callable
from time import monotonic
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from hark import config
from hark.audio.base import AudioIn, AudioOut
from hark.config import Config
from hark.protocol import COMMANDS
from hark.session import Session
from hark.stt.base import SAMPLE_RATE

log = logging.getLogger(__name__)

Factory = Callable[[str], Awaitable[Any]]
Job = Callable[[], Awaitable[Any]]
# Settable keys and the Session attribute each one swaps.
SETTABLE = {"brain.talker": "brain", "tts.engine": "tts", "stt.engine": "stt"}
MAX_REQUEST = 1 << 22  # bytes in one request line
OUTBOX_LIMIT = 2000  # queued lines per client before a client that does not read is dropped
SHUTDOWN_JOB_S = 5.0  # the after-session memory job gets this long when the daemon stops
WAV_CHUNK = 1600  # samples, 100 ms: wav files go through the same path as live audio
IDLE_JOB_GRACE_S = 2.0  # idle jobs start only after this much quiet
IDLE_POLL_S = 0.25
TURN_CMDS = ("say", "wav", "ptt_start", "ptt_stop")


class Conn:
    """One client connection and its outbox. send() never waits."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(OUTBOX_LIMIT)
        self._pump = asyncio.create_task(self._run())
        self.closed = False

    def send(self, obj: dict[str, Any]) -> bool:
        if self.closed or self.writer.is_closing():
            return False
        try:
            self._queue.put_nowait(obj)
            return True
        except asyncio.QueueFull:
            log.warning("dropping a client that stopped reading")
            self.abort()
            return False

    async def _run(self) -> None:
        try:
            while (obj := await self._queue.get()) is not None:
                self.writer.write((json.dumps(obj) + "\n").encode())
                await self.writer.drain()
        except (ConnectionError, RuntimeError):
            pass
        finally:
            self.closed = True

    async def close(self, timeout: float = 1.0) -> None:
        """Flush what is queued (bounded), then close."""
        if not self.closed:
            try:
                self._queue.put_nowait(None)
                await asyncio.wait_for(asyncio.shield(self._pump), timeout)
            except (asyncio.QueueFull, TimeoutError):
                pass
        self.abort()

    def abort(self) -> None:
        self.closed = True
        self._pump.cancel()
        self.writer.close()


def peer_of(writer: asyncio.StreamWriter) -> tuple[int, int] | None:
    """(pid, uid) of the process on the other end of a Unix socket."""
    sock = writer.get_extra_info("socket")
    if sock is None:
        return None
    try:
        pid, uid, _ = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    except OSError:
        return None
    return pid, uid


def descends_from(pid: int, ancestor: int) -> bool:
    """pid is a child, grandchild, ... of ancestor (not ancestor itself)."""
    seen = 0
    while pid > 1 and seen < 64:
        try:
            with open(f"/proc/{pid}/stat") as fh:
                stat = fh.read()
            pid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            return False
        if pid == ancestor:
            return True
        seen += 1
    return False


def is_user_client(writer: asyncio.StreamWriter) -> bool:
    """A process of this user that the daemon did not start (CLI, bar, phone bridge)."""
    peer = peer_of(writer)
    if peer is None:
        return False
    pid, uid = peer
    return uid == os.getuid() and not descends_from(pid, os.getpid())


class RequestError(Exception):
    pass


class Daemon:
    def __init__(
        self,
        session: Session,
        cfg: Config,
        *,
        audio_in: AudioIn | None = None,
        factories: dict[str, Factory] | None = None,
        wav_out: Callable[[str], AudioOut] | None = None,
        socket_path: Path | None = None,
        warmups: list[Job] | dict[str, Job] | None = None,
        closers: list[Job] | None = None,
        idle_jobs: list[Job] | None = None,
        errors: dict[str, str] | None = None,
        web: Any = None,
    ) -> None:
        self.session = session
        self.web = web  # phone and tablet server (hark.web.server.WebServer), started with the socket
        self.cfg = cfg
        self.audio_in = audio_in
        self.factories = factories or {}
        self.wav_out = wav_out
        self.socket_path = Path(socket_path or config.SOCKET_PATH)
        if isinstance(warmups, dict):
            self.warmups = warmups
        else:
            self.warmups = {f"warmup {i + 1}": w for i, w in enumerate(warmups or [])}
        self.closers = closers or []
        # Run once after warm-up, only while nobody is talking (a turn cancels and retries them).
        self.idle_jobs = idle_jobs or []
        # Components that failed to load or warm up, by name: shown in status.
        self.errors: dict[str, str] = dict(errors or {})
        self.ready = asyncio.Event()
        self._subs: set[Conn] = set()
        self._server: asyncio.Server | None = None
        self._listening = False
        self._pump: asyncio.Task[int] | None = None
        self._warm_task: asyncio.Task[None] | None = None
        self._jobs: list[asyncio.Task[Any]] = []
        self._stopping = asyncio.Event()
        session.emit = self.broadcast

    # --- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        """Listen on the socket, then warm the components in the background."""
        directory = self.socket_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        if self.socket_path.exists():
            if _is_live(self.socket_path):
                raise RuntimeError(f"a hark daemon is already running on {self.socket_path}")
            self.socket_path.unlink()
        old = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(self._client, path=str(self.socket_path), limit=MAX_REQUEST)
        finally:
            os.umask(old)
        os.chmod(self.socket_path, 0o600)
        if self.web is not None:
            try:
                await self.web.start()
            except OSError as exc:  # a busy port must not take the voice loop down
                log.error("phone page failed to start: %s", exc)
                self.errors["web"] = str(exc)
        await self.session.start()
        self._warm_task = asyncio.create_task(self._warm())

    async def _warm(self) -> None:
        names = list(self.warmups)
        results = await asyncio.gather(*(w() for w in self.warmups.values()), return_exceptions=True)
        for name, r in zip(names, results):
            if isinstance(r, BaseException):
                log.error("warm-up of %s failed: %s", name, r)
                self.errors[name] = str(r) or type(r).__name__
        self.ready.set()
        self._jobs = [asyncio.create_task(self._when_idle(job)) for job in self.idle_jobs]

    async def _when_idle(self, job: Job, attempts: int = 5) -> None:
        """Run job while the session is idle; if the user starts talking, cancel it and
        try again later. Keeps background model work off the user's turns."""
        s = self.session
        for _ in range(attempts):
            quiet_since = None
            while True:
                idle = s.state.value == "idle" and s.pending_confirm is None and not self._listening
                quiet_since = (quiet_since or monotonic()) if idle else None
                if quiet_since is not None and monotonic() - quiet_since >= IDLE_JOB_GRACE_S:
                    break
                await asyncio.sleep(IDLE_POLL_S)
            task = asyncio.ensure_future(job())
            while not task.done():
                await asyncio.wait({task}, timeout=IDLE_POLL_S)
                if not task.done() and (s.state.value != "idle" or self._listening):
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    break
            if task.done() and not task.cancelled():
                if task.exception() is not None:
                    log.error("background job failed: %s", task.exception())
                return

    async def run(self) -> None:
        """Serve until SIGTERM or SIGINT, then shut down cleanly."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._stopping.set)
        try:
            await self.start()
            log.info("hark daemon listening on %s", self.socket_path)
            await self._stopping.wait()
        finally:
            await self.close()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)

    def request_stop(self) -> None:
        self._stopping.set()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
        for c in list(self._subs):
            c.abort()
        self._subs.clear()
        if self.web is not None:
            await _quietly(self.web.close())
        if self._warm_task is not None:
            self._warm_task.cancel()
        for job in self._jobs:
            job.cancel()
        if self._listening and self.audio_in is not None:
            await _quietly(self.audio_in.stop())
        await _quietly(self.session.close())
        await self._end_session_job()
        await _quietly(self.session.out.stop())
        await _quietly(self.session.brain.close())
        if self.session.worker is not None:
            await _quietly(self.session.worker.close())
        for closer in self.closers:
            await _quietly(closer())
        if self._server is not None:  # never remove another daemon's socket
            self._server = None
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    async def _end_session_job(self) -> None:
        """Digest the conversation before the daemon goes, bounded so stopping stays quick."""
        job = self.session.end_session() if self.session.transcript else None
        if job is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(job), SHUTDOWN_JOB_S)
        except TimeoutError:
            log.warning("memory job did not finish in %.0f s; skipped", SHUTDOWN_JOB_S)
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)

    # --- connections ---------------------------------------------------------------

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = Conn(writer)
        try:
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, asyncio.LimitOverrunError):
                    await _skip_line(reader)  # so the client is still there to read the answer
                    conn.send({"ok": False, "error": f"request too large (over {MAX_REQUEST} bytes)"})
                    break
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    req = json.loads(line)
                    if not isinstance(req, dict):
                        raise ValueError
                except ValueError:
                    conn.send({"ok": False, "error": "malformed JSON request"})
                    continue
                try:
                    reply = await self.handle(req, conn)
                except RequestError as exc:
                    reply = {"ok": False, "error": str(exc)}
                except Exception as exc:
                    log.exception("request failed: %r", req)
                    reply = {"ok": False, "error": f"internal error: {exc}"}
                if reply is not None:
                    conn.send(reply)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # client went away
        finally:
            self._subs.discard(conn)
            await conn.close()

    async def broadcast(self, event: dict[str, Any]) -> None:
        for c in list(self._subs):
            if not c.send(event):
                self._subs.discard(c)

    # --- commands ----------------------------------------------------------------

    async def handle(self, req: dict[str, Any], conn: Conn) -> dict[str, Any] | None:
        cmd = req.get("cmd")
        if cmd not in COMMANDS:
            raise RequestError(f"unknown command: {cmd}")
        s = self.session
        if cmd in TURN_CMDS and s.pending_confirm is not None and not is_user_client(conn.writer):
            raise RequestError("a question is waiting for the user's answer")
        if cmd == "say":
            text = req.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RequestError("say needs non-empty text")
            return await self._turn(conn, req, text=text)
        if cmd == "wav":
            path = req.get("path")
            if not isinstance(path, str):
                raise RequestError("wav needs a path")
            pcm = await asyncio.to_thread(read_wav, path)
            await self.ready.wait()
            await s.begin_utterance()
            for i in range(0, len(pcm), WAV_CHUNK):
                s.feed_audio(pcm[i:i + WAV_CHUNK])
            return await self._turn(conn, req, utterance=True)
        if cmd == "ptt_start":
            if self.audio_in is None:
                raise RequestError("no microphone configured")
            if not self._listening:
                await s.begin_utterance()  # pressing the key talks over the assistant
                try:
                    await self.audio_in.start()
                except Exception as exc:
                    await s.cancel_utterance()
                    self.errors["microphone"] = str(exc)
                    raise RequestError(f"cannot open the microphone: {exc}") from exc
                self._listening = True
                self._pump = asyncio.create_task(self._pump_audio())
            return {"ok": True}
        if cmd == "ptt_stop":
            if self.audio_in is None or not self._listening:
                raise RequestError("not listening")
            self._listening = False
            pcm = await self.audio_in.stop()
            fed = await self._pump if self._pump is not None else 0
            self._pump = None
            if len(pcm) > fed:
                s.feed_audio(pcm[fed:])
            mic_error = getattr(self.audio_in, "error", None)
            if mic_error:
                self.errors["microphone"] = mic_error
            elif len(pcm):
                self.errors.pop("microphone", None)
            final = await self._turn(conn, req, utterance=True)
            if not len(pcm) and mic_error:
                final["error"] = f"no audio came in: {mic_error}"
            return final
        if cmd == "stop":
            if await self.release_mic():
                await s.cancel_utterance()
            await s.stop()
            return {"ok": True}
        if cmd == "confirm":
            answer, cid = req.get("answer"), req.get("id")
            if not isinstance(answer, bool):
                raise RequestError("confirm needs answer true or false")
            if not isinstance(cid, str) or not cid:
                raise RequestError("confirm needs the id of the question it answers")
            if not is_user_client(conn.writer):
                raise RequestError("only the user can answer a question")
            if not s.answer(answer, cid):
                raise RequestError("nothing to confirm" if s.pending_confirm is None
                                   else "that question is no longer the one waiting")
            return {"ok": True}
        if cmd == "ask_confirm":
            question, detail = req.get("question"), req.get("detail") or ""
            if not isinstance(question, str) or not question.strip() or not isinstance(detail, str):
                raise RequestError("ask_confirm needs a question")
            return {"ok": True, "answer": await s.confirm(question, detail)}
        if cmd == "tool":
            name, args = req.get("name"), req.get("args") or {}
            if not isinstance(name, str) or not isinstance(args, dict):
                raise RequestError("tool needs a name and an args object")
            res = await s.registry.call(name, args)
            return {"ok": True, "result": {"ok": res.ok, "text": res.text}}
        if cmd == "status":
            return self.status()
        if cmd == "set":
            return await self._set(req.get("key"), req.get("value"))
        # subscribe
        self._subs.add(conn)
        return {"ok": True, "subscribed": True}

    async def release_mic(self) -> bool:
        """Close the push-to-talk microphone without running a turn (stop, or a phone
        starts talking). True if it was open."""
        if not self._listening or self.audio_in is None:
            return False
        self._listening = False
        await self.audio_in.stop()
        if self._pump is not None:
            await self._pump
            self._pump = None
        return True

    async def _pump_audio(self) -> int:
        """Hand microphone chunks to the session as they are captured. Returns samples fed."""
        assert self.audio_in is not None
        fed = 0
        while (chunk := await self.audio_in.read()) is not None:
            self.session.feed_audio(chunk)
            fed += len(chunk)
        return fed

    def status(self) -> dict[str, Any]:
        s = self.session
        errors = dict(self.errors)
        if hasattr(s.brain, "error"):  # the brain knows its own health better than the warm-up did
            errors.pop("talker", None)
            if s.brain.error:
                errors["talker"] = s.brain.error
        if out_error := getattr(s.out, "error", None):
            errors["speaker"] = out_error
        return {
            "ok": True,
            "state": s.state.value,
            "ready": self.ready.is_set(),
            "talker": self.cfg.brain.talker,
            "worker": self.cfg.brain.worker if s.worker is not None else "off",
            "engines": {"stt": self.cfg.stt.engine, "tts": self.cfg.tts.engine},
            "session_id": s.session_id,
            "timing": s.last_timing,
            "confirm": s.pending_confirm,
            "errors": errors,
        }

    async def _set(self, key: Any, value: Any) -> dict[str, Any]:
        if key not in SETTABLE:
            raise RequestError(f"cannot set {key}; settable: {', '.join(SETTABLE)}")
        if not isinstance(value, str) or not value:
            raise RequestError("set needs a string value")
        factory = self.factories.get(key)
        if factory is None:
            raise RequestError(f"{key} cannot be changed live")
        try:
            new = await factory(value)
        except Exception as exc:
            raise RequestError(f"could not switch {key} to {value}: {exc}") from exc
        attr = SETTABLE[key]
        old = getattr(self.session, attr)
        if attr == "brain":
            await self.session.stop()  # never swap a brain out from under a running turn
        setattr(self.session, attr, new)
        if attr == "brain":
            asyncio.ensure_future(_quietly(old.close()))
        section, name = key.split(".")
        setattr(getattr(self.cfg, section), name, value)
        return {"ok": True, "key": key, "value": value}

    async def _turn(self, conn: Conn, req: dict[str, Any], utterance: bool = False, **kw: Any) -> dict[str, Any]:
        await self.ready.wait()
        out_path = req.get("out")
        out = None
        if out_path:
            if self.wav_out is None:
                raise RequestError("wav output is not available")
            out = self.wav_out(str(out_path))

        async def sink(event: dict[str, Any]) -> None:
            if conn not in self._subs:  # subscribers already got it
                conn.send(event)

        quiet = bool(req.get("quiet")) and out is None
        if utterance:
            res = await self.session.end_utterance(out=out, quiet=quiet, sink=sink)
        else:
            res = await self.session.run_turn(out=out, quiet=quiet, sink=sink, **kw)
        final: dict[str, Any] = {
            "ok": res.error is None,
            "done": True,
            "reply": res.reply,
            "timing": res.timing,
            "cancelled": res.cancelled,
        }
        if res.error is not None:
            final["error"] = res.error
        return final


def read_wav(path: str) -> np.ndarray:
    """A 16 kHz wav as float32 mono."""
    try:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise RequestError(f"cannot read {path}: {exc}") from exc
    if rate != SAMPLE_RATE:
        raise RequestError(f"{path} is {rate} Hz; hark needs {SAMPLE_RATE} Hz")
    return data.mean(axis=1)


async def _skip_line(reader: asyncio.StreamReader, cap: int = 64 * MAX_REQUEST) -> None:
    """Read and drop the rest of an over-long line (bounded)."""
    seen = 0
    while seen < cap:
        chunk = await reader.read(1 << 16)
        if not chunk or b"\n" in chunk:
            return
        seen += len(chunk)


def _is_live(path: Path) -> bool:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        try:
            sock.connect(str(path))
            return True
        except OSError:
            return False


async def _quietly(aw: Awaitable[Any]) -> None:
    try:
        await aw
    except Exception:
        log.exception("shutdown step failed")
