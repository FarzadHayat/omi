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
from hark.dictation import (
    DICTATION_MODES,
    ActiveContext,
    MediaPauser,
    capture_active_context,
    format_dictation,
    insert_text,
)
from hark.protocol import COMMANDS
from hark.session import Session
from hark.stt import available_stt_engines
from hark.stt.base import SAMPLE_RATE
from hark.tts import available_tts_engines
from hark.tts.cloud import OPENAI_VOICES

log = logging.getLogger(__name__)

Factory = Callable[[str], Awaitable[Any]]
Job = Callable[[], Awaitable[Any]]
# Settable keys and the Session attribute each one swaps.
SETTABLE = {
    "brain.talker": "brain",
    "brain.model": "brain",
    "tts.engine": "tts",
    "tts.voice": "tts",
    "stt.engine": "stt",
}
WAKE_SENSITIVITY = {"low": 0.22, "medium": 0.15, "high": 0.10}
WAKE_CALL_MODES = ("auto", "continuous", "single")
KOKORO_VOICES = ("af_heart", "af_bella", "am_adam", "bf_emma", "bm_george")


def sensitivity_label(threshold: float) -> str:
    if threshold >= 0.19:
        return "low"
    if threshold <= 0.12:
        return "high"
    return "medium"
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
        wake: Any = None,
        track_cursor: bool = False,
        config_path: Path | None = None,
        play_sfx: Callable[[str], None] | None = None,
        dictation_llm: Callable[[str], Awaitable[str]] | None = None,
        dictation_context_fn: Callable[..., Awaitable[ActiveContext]] | None = None,
        dictation_insert_fn: Callable[..., Awaitable[tuple[bool, str]]] | None = None,
        media_pauser: MediaPauser | None = None,
    ) -> None:
        self.session = session
        self.web = web  # phone and tablet server (hark.web.server.WebServer), started with the socket
        self.wake = wake  # optional always-on WakeListener (hark.audio.wake.WakeListener)
        self.cfg = cfg
        self.config_path = config_path  # when set, `set` with save=true persists to this TOML file
        self.play_sfx = play_sfx
        self.dictation_llm = dictation_llm
        self.dictation_context_fn = dictation_context_fn or capture_active_context
        self.dictation_insert_fn = dictation_insert_fn or insert_text
        self.media_pauser = media_pauser or MediaPauser()
        self.audio_in = audio_in
        self.factories = factories or {}
        self.wav_out = wav_out
        self.socket_path = Path(socket_path or config.SOCKET_PATH)
        self.track_cursor = track_cursor
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
        self._dictating = False
        self._dictate_chunks: list[np.ndarray] = []
        self._dictate_pump: asyncio.Task[int] | None = None
        self._dictate_ctx: ActiveContext | None = None
        self._pump: asyncio.Task[int] | None = None
        self._warm_task: asyncio.Task[None] | None = None
        self._cursor_task: asyncio.Task[None] | None = None
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
        if self.track_cursor:
            self._cursor_task = asyncio.create_task(self._cursor_loop())

    async def _cursor_loop(self) -> None:
        """Broadcast cursor coordinates (~12 Hz, only on movement >= 5 px) while subscribers exist,
        so Omi's eyes in the bar and companion overlay can smoothly follow the pointer."""
        from hark.tools.hyprland import hypr_query

        last: tuple[int, int] | None = None
        while not self._stopping.is_set():
            await asyncio.sleep(0.08 if self._subs else 0.5)
            if not self._subs:
                continue
            try:
                pos = await asyncio.to_thread(hypr_query, "cursorpos")
            except (OSError, ValueError):
                await asyncio.sleep(1.0)
                continue
            if not isinstance(pos, dict) or "x" not in pos or "y" not in pos:
                continue
            x, y = int(pos["x"]), int(pos["y"])
            if last is None or abs(x - last[0]) >= 5 or abs(y - last[1]) >= 5:
                last = (x, y)
                await self.broadcast({"event": "cursor", "x": x, "y": y})

    async def _warm(self) -> None:
        names = list(self.warmups)
        results = await asyncio.gather(*(w() for w in self.warmups.values()), return_exceptions=True)
        for name, r in zip(names, results):
            if isinstance(r, BaseException):
                log.error("warm-up of %s failed: %s", name, r)
                self.errors[name] = str(r) or type(r).__name__
        self.ready.set()
        if self.wake is not None and self.cfg.wake.enabled:
            try:
                await self.wake.start()
                await self.broadcast({"event": "wake", "enabled": True})
            except Exception as exc:
                log.error("wake word failed to start: %s", exc)
                self.errors["wake"] = str(exc) or type(exc).__name__
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
        if self.wake is not None:
            await _quietly(self.wake.stop())
        if self._warm_task is not None:
            self._warm_task.cancel()
        if self._cursor_task is not None:
            self._cursor_task.cancel()
        for job in self._jobs:
            job.cancel()
        if (self._listening or self._dictating) and self.audio_in is not None:
            await _quietly(self.audio_in.stop())
        if self._dictating:
            self._dictating = False
            await _quietly(self.media_pauser.resume_if_paused())
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
                if self.play_sfx is not None and not req.get("quiet"):
                    self.play_sfx("wake")
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
        if cmd == "dictate_start":
            return await self._dictate_start(quiet=bool(req.get("quiet")))
        if cmd == "dictate_stop":
            return await self._dictate_stop(
                mode_override=req.get("mode"),
                dry_run=bool(req.get("dry_run")),
                quiet=bool(req.get("quiet")),
            )
        if cmd == "dictate_toggle":
            if self._dictating:
                return await self._dictate_stop(
                    mode_override=req.get("mode"),
                    dry_run=bool(req.get("dry_run")),
                    quiet=bool(req.get("quiet")),
                )
            return await self._dictate_start(quiet=bool(req.get("quiet")))
        if cmd == "stop":
            if self._dictating:
                await self._cancel_dictation()
            if await self.release_mic() or s._utt is not None:
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
        if cmd == "mascot":
            action = str(req.get("action") or "point")
            if action not in ("point", "move", "clear"):
                raise RequestError("mascot action must be point, move or clear")
            ev: dict[str, Any] = {"event": "mascot", "action": action}
            if action != "clear":
                ev["x"] = int(req.get("x", 0))
                ev["y"] = int(req.get("y", 0))
                ev["w"] = max(0, int(req.get("w", 0)))
                ev["h"] = max(0, int(req.get("h", 0)))
                ev["label"] = str(req.get("label") or "")[:140]
                ev["duration"] = max(1, min(int(req.get("duration", 8)), 30))
            await self.broadcast(ev)
            return {"ok": True}
        if cmd == "status":
            return self.status()
        if cmd == "set":
            return await self._set(req.get("key"), req.get("value"), save=bool(req.get("save")))
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

    async def _pump_dictation(self) -> int:
        assert self.audio_in is not None
        fed = 0
        while (chunk := await self.audio_in.read()) is not None:
            arr = np.asarray(chunk, dtype=np.float32)
            self._dictate_chunks.append(arr)
            fed += len(arr)
        return fed

    async def _dictate_start(self, *, quiet: bool = False) -> dict[str, Any]:
        if self.audio_in is None:
            raise RequestError("no microphone configured")
        if self._dictating:
            return {"ok": True, "dictating": True}
        if await self.release_mic() or self.session._utt is not None:
            await self.session.cancel_utterance()
        await self.session.stop()
        self._dictate_ctx = await self.dictation_context_fn(read_selection=self.cfg.dictation.use_selection)
        if self.cfg.dictation.pause_media:
            await self.media_pauser.pause_if_playing()
        self._dictate_chunks = []
        try:
            await self.audio_in.start()
        except Exception as exc:
            await self.media_pauser.resume_if_paused()
            self.errors["microphone"] = str(exc)
            raise RequestError(f"cannot open the microphone: {exc}") from exc
        self._dictating = True
        self._dictate_pump = asyncio.create_task(self._pump_dictation())
        if self.play_sfx is not None and not quiet:
            self.play_sfx("wake")
        await self.broadcast({"event": "state", "state": "listening"})
        await self.broadcast({"event": "progress", "text": f"Dictating ({self.cfg.dictation.mode})…"})
        return {"ok": True, "dictating": True}

    async def _cancel_dictation(self) -> None:
        if not self._dictating:
            return
        self._dictating = False
        if self.audio_in is not None:
            await _quietly(self.audio_in.stop())
        if self._dictate_pump is not None:
            await _quietly(self._dictate_pump)
            self._dictate_pump = None
        self._dictate_chunks = []
        self._dictate_ctx = None
        await _quietly(self.media_pauser.resume_if_paused())
        await self.broadcast({"event": "state", "state": "idle"})

    async def _dictate_stop(
        self,
        *,
        mode_override: str | None = None,
        dry_run: bool = False,
        quiet: bool = False,
    ) -> dict[str, Any]:
        if self.audio_in is None or not self._dictating:
            raise RequestError("not dictating")
        self._dictating = False
        pcm = await self.audio_in.stop()
        fed = await self._dictate_pump if self._dictate_pump is not None else 0
        self._dictate_pump = None
        if len(pcm) > fed:
            self._dictate_chunks.append(np.asarray(pcm[fed:], dtype=np.float32))
        audio = np.concatenate(self._dictate_chunks) if self._dictate_chunks else np.asarray(pcm, dtype=np.float32)
        self._dictate_chunks = []
        ctx = self._dictate_ctx or ActiveContext()
        self._dictate_ctx = None
        await _quietly(self.media_pauser.resume_if_paused())

        mic_error = getattr(self.audio_in, "error", None)
        if mic_error:
            self.errors["microphone"] = mic_error
        elif len(audio):
            self.errors.pop("microphone", None)

        if len(audio) == 0:
            await self.broadcast({"event": "state", "state": "idle"})
            reply: dict[str, Any] = {"ok": True, "done": True, "raw": "", "text": "", "mode": self.cfg.dictation.mode, "inserted": False}
            if mic_error:
                reply["error"] = f"no audio came in: {mic_error}"
            return reply

        if self.play_sfx is not None and not quiet:
            self.play_sfx("heard")
        await self.broadcast({"event": "state", "state": "thinking"})
        await self.ready.wait()
        try:
            raw = await asyncio.to_thread(self.session.stt.transcribe, audio)
        except Exception as exc:
            await self.broadcast({"event": "state", "state": "idle"})
            if self.play_sfx is not None and not quiet:
                self.play_sfx("error")
            raise RequestError(f"dictation transcription failed: {exc}") from exc

        res = await format_dictation(
            raw,
            self.cfg.dictation,
            ctx,
            llm=self.dictation_llm,
            mode_override=mode_override if isinstance(mode_override, str) else None,
        )
        inserted = False
        method = "dry_run" if dry_run else ""
        if res.text and not dry_run:
            inserted, method = await self.dictation_insert_fn(
                res.text,
                is_terminal=ctx.is_terminal,
                method=self.cfg.dictation.paste_method,
                restore_clipboard=self.cfg.dictation.restore_clipboard,
            )
            res.inserted = inserted
            res.method = method
        if res.text:
            preview = res.text if len(res.text) <= 80 else res.text[:77] + "…"
            await self.broadcast({"event": "progress", "text": f"Dictated ({res.mode}): {preview}"})
            if self.play_sfx is not None and not quiet:
                self.play_sfx("done" if (inserted or dry_run) else "error")
        await self.broadcast({"event": "state", "state": "idle"})
        out: dict[str, Any] = {
            "ok": True,
            "done": True,
            "raw": res.raw,
            "text": res.text,
            "mode": res.mode,
            "inserted": inserted,
            "method": method,
            "rewrote_selection": res.rewrote_selection,
        }
        if res.error:
            out["warning"] = res.error
        return out

    def _active_voice(self) -> str:
        if self.cfg.tts.engine == "openai":
            return self.cfg.tts.openai_voice
        if self.cfg.tts.engine == "elevenlabs":
            return self.cfg.tts.elevenlabs_voice_id
        return self.cfg.tts.kokoro_voice

    def _voice_list(self) -> list[str]:
        if self.cfg.tts.engine == "openai":
            return list(OPENAI_VOICES)
        if self.cfg.tts.engine == "kokoro":
            return list(KOKORO_VOICES)
        return []

    def status(self) -> dict[str, Any]:
        s = self.session
        errors = dict(self.errors)
        if hasattr(s.brain, "error"):  # the brain knows its own health better than the warm-up did
            errors.pop("talker", None)
            if s.brain.error:
                errors["talker"] = s.brain.error
        if out_error := getattr(s.out, "error", None):
            errors["speaker"] = out_error
        if self.wake is not None and getattr(self.wake, "error", None):
            errors["wake"] = self.wake.error
        from hark.brain.cli_providers import available_providers

        talker = self.cfg.brain.talker
        talker_model = self._active_model(talker)
        return {
            "ok": True,
            "name": self.cfg.name,
            "state": "listening" if self._dictating else s.state.value,
            "ready": self.ready.is_set(),
            "talker": talker,
            "talker_model": talker_model,
            "worker": self.cfg.brain.worker if s.worker is not None else "off",
            "providers": [
                {"name": p["name"], "installed": p["installed"], "model": p["model"]}
                for p in available_providers(self.cfg)
            ],
            "stt_engines": available_stt_engines(self.cfg.stt),
            "tts_engines": available_tts_engines(self.cfg.tts),
            "voices": self._voice_list(),
            "dictation": {
                "mode": self.cfg.dictation.mode,
                "pause_media": self.cfg.dictation.pause_media,
                "active": self._dictating,
                "modes": list(DICTATION_MODES),
            },
            "wake": {
                "enabled": bool(self.cfg.wake.enabled and self.wake is not None and self.wake.active),
                "configured": self.cfg.wake.enabled,
                "call_mode": self._call_mode(),
                "conversational": self._call_mode() != "single",
                "conversing": bool(self.wake is not None and getattr(self.wake, "conversing", False)),
                "threshold": self.cfg.wake.threshold,
                "sensitivity": sensitivity_label(self.cfg.wake.threshold),
                "phrases": list(self.cfg.wake.phrases),
            },
            "engines": {
                "stt": self.cfg.stt.engine,
                "tts": self.cfg.tts.engine,
                "voice": self._active_voice(),
                "speed": self.cfg.tts.speed,
            },
            "audio": {
                "sounds": self.cfg.audio.sounds,
            },
            "session_id": s.session_id,
            "timing": s.last_timing,
            "confirm": s.pending_confirm,
            "errors": errors,
        }

    def _persist(self, section: str, key: str, value: Any, save: bool) -> None:
        if save and self.config_path is not None:
            try:
                config.save_override(section, key, value, self.config_path)
            except Exception as exc:
                log.warning("could not save %s.%s to %s: %s", section, key, self.config_path, exc)

    def _call_mode(self) -> str:
        from hark.audio.wake import effective_call_mode

        return effective_call_mode(self.cfg.wake)

    def _settings_event(self) -> dict[str, Any]:
        mode = self._call_mode()
        return {
            "event": "settings",
            "talker": self.cfg.brain.talker,
            "talker_model": self._active_model(self.cfg.brain.talker),
            "voice": self._active_voice(),
            "voices": self._voice_list(),
            "stt_engines": available_stt_engines(self.cfg.stt),
            "tts_engines": available_tts_engines(self.cfg.tts),
            "engines": {
                "stt": self.cfg.stt.engine,
                "tts": self.cfg.tts.engine,
                "voice": self._active_voice(),
                "speed": self.cfg.tts.speed,
            },
            "dictation": {
                "mode": self.cfg.dictation.mode,
                "pause_media": self.cfg.dictation.pause_media,
            },
            "wake": {
                "enabled": bool(self.cfg.wake.enabled and self.wake is not None and self.wake.active),
                "call_mode": mode,
                "conversational": mode != "single",
                "threshold": self.cfg.wake.threshold,
                "sensitivity": sensitivity_label(self.cfg.wake.threshold),
            },
            "audio": {
                "sounds": self.cfg.audio.sounds,
            },
        }

    def _active_model(self, talker: str) -> str:
        if talker == "claude":
            return self.cfg.brain.claude.model
        if talker in ("local", "openai"):
            return self.cfg.brain.local.model_name
        pcfg = getattr(self.cfg.brain, talker, None)
        return getattr(pcfg, "model", "") if pcfg is not None else ""

    @staticmethod
    def _parse_bool(value: Any, key: str) -> bool:
        raw = str(value).strip().lower() if isinstance(value, (str, bool)) else ""
        if raw in ("true", "on", "1", "yes"):
            return True
        if raw in ("false", "off", "0", "no"):
            return False
        raise RequestError(f"{key} expects on or off")

    async def _set(self, key: Any, value: Any, *, save: bool = False) -> dict[str, Any]:
        if key == "tts.speed":
            try:
                speed = round(float(value), 2)
            except (TypeError, ValueError) as exc:
                raise RequestError("tts.speed expects a number between 0.5 and 2.0") from exc
            if not 0.5 <= speed <= 2.0:
                raise RequestError("tts.speed must be between 0.5 and 2.0")
            if hasattr(self.session.tts, "set_speed"):
                self.session.tts.set_speed(speed)
            self.cfg.tts.speed = speed
            self._persist("tts", "speed", speed, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": f"{speed:g}"}
        if key == "dictation.mode":
            raw = str(value).strip().lower() if isinstance(value, str) else ""
            if raw == "super":
                raw = "auto"
            if raw not in DICTATION_MODES:
                raise RequestError(f"dictation.mode expects {', '.join(DICTATION_MODES)}")
            self.cfg.dictation.mode = raw
            self._persist("dictation", "mode", raw, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": raw}
        if key == "dictation.pause_media":
            want = self._parse_bool(value, key)
            self.cfg.dictation.pause_media = want
            self._persist("dictation", "pause_media", want, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": "on" if want else "off"}
        if key == "audio.sounds":
            want = self._parse_bool(value, key)
            self.cfg.audio.sounds = want
            self._persist("audio", "sounds", want, save)
            if want and self.play_sfx is not None:
                self.play_sfx("heard")
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": "on" if want else "off"}
        if key == "wake.enabled":
            if self.wake is None:
                raise RequestError("wake.enabled cannot be changed live (no wake listener)")
            want = self._parse_bool(value, key)
            if want:
                try:
                    await self.wake.start()
                except Exception as exc:
                    self.errors["wake"] = str(exc) or type(exc).__name__
                    raise RequestError(f"could not start wake word: {exc}") from exc
                self.errors.pop("wake", None)
            else:
                await self.wake.stop()
                self.errors.pop("wake", None)
            self.cfg.wake.enabled = want
            self._persist("wake", "enabled", want, save)
            await self.broadcast({"event": "wake", "enabled": want})
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": "on" if want else "off"}
        if key == "wake.call_mode":
            raw = str(value).strip().lower() if isinstance(value, str) else ""
            if raw == "smart":
                raw = "auto"
            if raw not in WAKE_CALL_MODES:
                raise RequestError(f"wake.call_mode expects {', '.join(WAKE_CALL_MODES)}")
            self.cfg.wake.call_mode = raw
            self.cfg.wake.conversational = raw != "single"
            self._persist("wake", "call_mode", raw, save)
            self._persist("wake", "conversational", raw != "single", save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": raw}
        if key == "wake.conversational":
            want = self._parse_bool(value, key)
            self.cfg.wake.conversational = want
            self.cfg.wake.call_mode = "continuous" if want else "single"
            self._persist("wake", "conversational", want, save)
            self._persist("wake", "call_mode", self.cfg.wake.call_mode, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": "on" if want else "off"}
        if key == "wake.sensitivity":
            raw = str(value).strip().lower() if isinstance(value, (str, int, float)) else ""
            if raw in WAKE_SENSITIVITY:
                thresh = WAKE_SENSITIVITY[raw]
                label = raw
            else:
                try:
                    thresh = float(raw)
                except ValueError as exc:
                    raise RequestError("wake.sensitivity expects low, medium, high, or a number 0.05-0.6") from exc
                if not 0.05 <= thresh <= 0.6:
                    raise RequestError("wake.sensitivity must be between 0.05 and 0.6")
                label = sensitivity_label(thresh)
            self.cfg.wake.threshold = thresh
            if self.wake is not None and hasattr(self.wake, "reload_detector"):
                try:
                    await self.wake.reload_detector()
                except Exception as exc:
                    raise RequestError(f"could not update wake sensitivity: {exc}") from exc
            self._persist("wake", "threshold", thresh, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": label}
        if key not in SETTABLE:
            all_keys = (*SETTABLE, "tts.speed", "dictation.mode", "dictation.pause_media",
                        "audio.sounds", "wake.enabled", "wake.call_mode", "wake.conversational", "wake.sensitivity")
            raise RequestError(f"cannot set {key}; settable: {', '.join(all_keys)}")
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
        if key == "tts.voice":
            if self.cfg.tts.engine == "openai":
                self.cfg.tts.openai_voice = value
                self._persist("tts", "openai_voice", value, save)
            elif self.cfg.tts.engine == "elevenlabs":
                self.cfg.tts.elevenlabs_voice_id = value
                self._persist("tts", "elevenlabs_voice_id", value, save)
            else:
                self.cfg.tts.kokoro_voice = value
                self.cfg.tts.engine = "kokoro"
                self._persist("tts", "kokoro_voice", value, save)
                self._persist("tts", "engine", "kokoro", save)
        elif key == "brain.talker":
            resolved = getattr(new, "name", value) if value in ("auto", "openai") else value
            self.cfg.brain.talker = resolved
            self._persist("brain", "talker", resolved, save)
        elif key == "brain.model":
            talker = self.cfg.brain.talker
            if talker == "claude":
                self.cfg.brain.claude.model = value
                self._persist("brain.claude", "model", value, save)
            elif talker in ("local", "openai"):
                self.cfg.brain.local.model_name = value
                self._persist("brain.local", "model_name", value, save)
            elif (pcfg := getattr(self.cfg.brain, talker, None)) is not None and hasattr(pcfg, "model"):
                pcfg.model = value
                self._persist(f"brain.{talker}", "model", value, save)
        else:
            section, name = key.split(".")
            setattr(getattr(self.cfg, section), name, value)
            self._persist(section, name, value, save)
        await self.broadcast(self._settings_event())
        return {"ok": True, "key": key, "value": self.cfg.brain.talker if key == "brain.talker" else value}

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
