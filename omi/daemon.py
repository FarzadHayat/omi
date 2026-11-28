"""The omi daemon: one Session behind a Unix socket speaking omi.protocol.

Newline-delimited JSON. Turn commands stream that turn's events back on the same
connection; "subscribe" connections get every event. One turn runs at a time and a
new one cancels the running one (the Session enforces that).

Every connection has its own bounded outbox, so a client that stops reading never
slows a turn or the stop command: once its outbox is full it is dropped.

A stop (systemctl restart after an update, say) waits for a pause in the conversation,
up to RESTART_WAIT_S, and hands the conversation to the next daemon through a file in
the state directory, so a restart costs a few seconds of Omi, not the conversation.

Only the user answers a yes/no question: a "confirm" must name the question's id and
come from a same-user process that is not one the daemon started (Claude and its
shell are), and while a question waits those processes cannot start turns either.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import struct
from collections.abc import Awaitable, Callable
from time import monotonic, time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from omi import config
from omi.audio.base import AudioIn, AudioOut
from omi.config import Config
from omi.events import State
from omi.protocol import COMMANDS
from omi.session import Session
from omi.stt import available_stt_engines
from omi.stt.base import SAMPLE_RATE
from omi.tts import available_tts_engines
from omi.tts.cloud import OPENAI_VOICES

log = logging.getLogger(__name__)

Factory = Callable[[str], Awaitable[Any]]
Job = Callable[[], Awaitable[Any]]
# Settable keys and the Session attribute each one swaps.
WORKER_MODES = ("auto", "unrestricted")  # brain.worker_mode
# Look to talk numbers `omi look-tune` sets, with the range each may take.
LOOK_TUNING = {"wake.lip_threshold": (0.005, 0.2), "wake.yaw_center": (-1.0, 1.0), "wake.yaw_range": (0.1, 0.8)}

SETTABLE = {
    "brain.talker": "brain",
    "brain.model": "brain",
    "brain.fallbacks": "brain",
    "brain.local_model": "brain",
    "brain.worker": "worker",
    "brain.worker_model": "worker",
    "brain.worker_fallbacks": "worker",
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
# A stop waits this long for the user and Omi to pause before it cuts in anyway. Well
# under systemd's 90 s stop timeout, and short enough that logging out never hangs long.
RESTART_WAIT_S = 30.0
RESTART_QUIET_S = 0.3  # the pause must last this long (a turn starting has tiny gaps)
RESTART_POLL_S = 0.05
HANDOFF_FILE = "handoff.json"


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
    ) -> None:
        self.session = session
        self.web = web  # phone and tablet server (omi.web.server.WebServer), started with the socket
        self.wake = wake  # optional always-on WakeListener (omi.audio.wake.WakeListener)
        self.cfg = cfg
        self.config_path = config_path  # when set, `set` with save=true persists to this TOML file
        self.play_sfx = play_sfx
        self.voiceprints: Any = None  # omi.voice_id.Voiceprints, made on first use
        from omi.look_setup import GuidedSetup

        self.look_setup = GuidedSetup(self)  # Omi tunes look to talk with you, out loud
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
        self._pump: asyncio.Task[int] | None = None
        self._warm_task: asyncio.Task[None] | None = None
        self._cursor_task: asyncio.Task[None] | None = None
        self._jobs: list[asyncio.Task[Any]] = []
        self._stopping = asyncio.Event()
        self._drain_task: asyncio.Task[None] | None = None
        self._resume_call = False  # the previous daemon stopped in the middle of a call
        self._started_at = monotonic()
        session.emit = self.broadcast

    # --- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        """Listen on the socket, then warm the components in the background."""
        directory = self.socket_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        if self.socket_path.exists():
            if _is_live(self.socket_path):
                raise RuntimeError(f"a omi daemon is already running on {self.socket_path}")
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
        self._take_handoff()
        if self.wake is not None and (self.cfg.wake.enabled or self.cfg.wake.look_to_talk) \
                and hasattr(self.wake, "listen_early"):
            try:  # hear what is said while the models load; the listener catches up once ready
                await self.wake.listen_early()
            except Exception as exc:
                log.warning("could not open the mic early: %s", exc)
        await self.session.start()
        self._warm_task = asyncio.create_task(self._warm())
        if self.track_cursor:
            self._cursor_task = asyncio.create_task(self._cursor_loop())

    async def _cursor_loop(self) -> None:
        """Broadcast cursor coordinates (~12 Hz, only on movement >= 5 px) while subscribers exist,
        so Omi's eyes in the bar and companion overlay can smoothly follow the pointer."""
        from omi.tools.hyprland import hypr_query

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
        log.info("ready %.1f s after start", monotonic() - self._started_at)
        if self.wake is not None and (self.cfg.wake.enabled or self.cfg.wake.look_to_talk):
            try:
                if self._resume_call and hasattr(self.wake, "resume_call"):
                    self.wake.resume_call()
                await self.wake.start()
                await self.broadcast({"event": "wake", "enabled": self.cfg.wake.enabled})
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
        """Serve until SIGTERM or SIGINT. The first one waits for a pause in the
        conversation (bounded), a second one stops at once. The conversation is handed
        to the next daemon."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._on_stop_signal)
        try:
            await self.start()
            log.info("omi daemon listening on %s", self.socket_path)
            await self._stopping.wait()
        finally:
            if self._drain_task is not None:
                self._drain_task.cancel()
            await self.close(handoff=True)
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)

    def _on_stop_signal(self) -> None:
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._stop_when_quiet())
        else:
            self._stopping.set()

    def quiet(self) -> bool:
        """Nothing a restart would cut: no turn, speech, question or request being spoken.
        An open call mic that has heard nothing yet counts as quiet; the call resumes."""
        s = self.session
        if self._listening or s._turn is not None or s.pending_confirm is not None:
            return False
        if s.state is State.IDLE:
            return s._utt is None
        if s.state is State.LISTENING:
            return s._utt is not None and bool(getattr(self.wake, "waiting_for_speech", False))
        return False

    async def _stop_when_quiet(self) -> None:
        deadline = monotonic() + RESTART_WAIT_S
        quiet_since: float | None = None
        told = False
        while monotonic() < deadline:
            now = monotonic()
            if self.quiet():
                quiet_since = quiet_since or now
                if now - quiet_since >= RESTART_QUIET_S:
                    break
            else:
                quiet_since = None
                if not told:
                    told = True
                    log.info("stop requested mid-conversation; waiting up to %.0f s for a pause", RESTART_WAIT_S)
            await asyncio.sleep(RESTART_POLL_S)
        else:
            log.warning("conversation did not pause in %.0f s; stopping anyway", RESTART_WAIT_S)
        self._stopping.set()

    def request_stop(self) -> None:
        self._stopping.set()

    async def close(self, *, handoff: bool = False) -> None:
        """Shut down. With handoff, the conversation is saved for the next daemon instead
        of being closed (and digested into memory) now, which also makes stopping quick."""
        in_call = bool(self.wake is not None and getattr(self.wake, "conversing", False))
        if self._server is not None:
            self._server.close()
        for c in list(self._subs):
            c.abort()
        self._subs.clear()
        if self.web is not None:
            await _quietly(self.web.close())
        await _quietly(self.look_setup.cancel())
        if self.wake is not None:
            await _quietly(self.wake.stop())
        if self._warm_task is not None:
            self._warm_task.cancel()
        if self._cursor_task is not None:
            self._cursor_task.cancel()
        for job in self._jobs:
            job.cancel()
        if self._listening and self.audio_in is not None:
            await _quietly(self.audio_in.stop())
        await _quietly(self.session.close())
        if not (handoff and self._save_handoff(in_call)):
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

    @property
    def handoff_path(self) -> Path:
        return self.session.state_dir / HANDOFF_FILE

    def _save_handoff(self, in_call: bool) -> bool:
        """Write the conversation for the next daemon. True when it was saved."""
        snap = self.session.snapshot()
        if not snap["transcript"] and not in_call:
            return False
        snap.update(saved_at=time(), in_call=in_call)
        path = self.handoff_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # it holds what was said
            with os.fdopen(fd, "w") as fh:
                json.dump(snap, fh)
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not save the conversation for the next start: %s", exc)
            return False
        log.info("conversation %s handed to the next start", snap["session_id"])
        return True

    def _take_handoff(self) -> None:
        """Carry on the conversation the previous daemon saved, if it is still fresh.
        A stale one is closed and digested into memory once nobody is talking."""
        path = self.handoff_path
        try:
            snap = json.loads(path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            log.warning("ignoring an unreadable conversation handoff: %s", exc)
            snap = None
        with contextlib.suppress(OSError):
            path.unlink()
        if not isinstance(snap, dict):
            return
        try:
            idle_s = max(0.0, time() - float(snap.get("saved_at", 0))) + float(snap.get("idle_s", 0))
        except (TypeError, ValueError):
            return
        s = self.session
        if idle_s < s.idle_timeout:
            s.restore(snap, idle_s)
            self._resume_call = bool(snap.get("in_call"))
            log.info("carrying on conversation %s (%d lines)", s.session_id, len(s.transcript))
            return
        lines = [m for m in snap.get("transcript") or [] if isinstance(m, dict)]
        end = s.on_session_end
        if lines and end is not None:
            sid = str(snap.get("session_id") or "")
            text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in lines)
            self.idle_jobs.append(lambda: end(sid, text))

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
        if cmd == "look_tune":
            return {"ok": True, "text": self.look_setup.start()}
        if cmd == "stop":
            await self.look_setup.cancel()
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
        if self.wake is not None and getattr(self.wake, "look_error", None):
            errors["look"] = self.wake.look_error
        from omi.brain.cli_providers import available_providers

        talker = self.cfg.brain.talker
        talker_model = self._active_model(talker)
        return {
            "ok": True,
            "name": self.cfg.name,
            "state": s.state.value,
            "ready": self.ready.is_set(),
            "talker": talker,
            "talker_model": talker_model,
            "worker": self.cfg.brain.worker if getattr(s.worker, "active", s.worker is not None) else "off",
            "brain": self._brain_info(),
            "providers": [
                {"name": p["name"], "installed": p["installed"], "model": p["model"]}
                for p in available_providers(self.cfg)
            ],
            "stt_engines": available_stt_engines(self.cfg.stt),
            "tts_engines": available_tts_engines(self.cfg.tts),
            "voices": self._voice_list(),
            "wake": {
                "enabled": bool(self.cfg.wake.enabled and self.wake is not None and self.wake.active),
                "configured": self.cfg.wake.enabled,
                "call_mode": self._call_mode(),
                "conversational": self._call_mode() != "single",
                "conversing": bool(self.wake is not None and getattr(self.wake, "conversing", False)),
                "threshold": self.cfg.wake.threshold,
                "sensitivity": sensitivity_label(self.cfg.wake.threshold),
                "phrases": list(self.cfg.wake.phrases),
                "look_to_talk": self._look_on(),
                "look_paused_by": str(getattr(self.wake, "look_paused_by", "") or "") if self.wake else "",
                "lip_check": self.cfg.wake.lip_check,
                "only_my_voice": self.cfg.wake.only_my_voice,
                "voices": self._voices().names(),
                "camera": self.cfg.wake.camera,
            },
            "cameras": self._cameras(),
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

    def _voices(self) -> Any:
        if self.voiceprints is None:
            from omi.voice_id import Voiceprints

            self.voiceprints = Voiceprints()
        return self.voiceprints

    def _cameras(self) -> list[dict[str, str]]:
        from omi.look import list_cameras

        try:
            return list_cameras()
        except Exception:
            log.debug("could not list cameras", exc_info=True)
            return []

    def _look_on(self) -> bool:
        return bool(self.cfg.wake.look_to_talk and self.wake is not None
                    and getattr(self.wake, "looking_active", False))

    async def _sync_listener(self) -> None:
        """The listener runs for the wake word, look to talk, or both."""
        assert self.wake is not None
        if self.cfg.wake.enabled or self.cfg.wake.look_to_talk:
            await self.wake.start()
        else:
            await self.wake.stop()

    def _call_mode(self) -> str:
        from omi.audio.wake import effective_call_mode

        return effective_call_mode(self.cfg.wake)

    def _settings_event(self) -> dict[str, Any]:
        mode = self._call_mode()
        return {
            "event": "settings",
            "talker": self.cfg.brain.talker,
            "talker_model": self._active_model(self.cfg.brain.talker),
            "brain": self._brain_info(),
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
            "wake": {
                "enabled": bool(self.cfg.wake.enabled and self.wake is not None and self.wake.active),
                "call_mode": mode,
                "conversational": mode != "single",
                "threshold": self.cfg.wake.threshold,
                "sensitivity": sensitivity_label(self.cfg.wake.threshold),
                "look_to_talk": self._look_on(),
                "look_paused_by": str(getattr(self.wake, "look_paused_by", "") or "") if self.wake else "",
                "lip_check": self.cfg.wake.lip_check,
                "only_my_voice": self.cfg.wake.only_my_voice,
                "voices": self._voices().names(),
                "camera": self.cfg.wake.camera,
            },
            "cameras": self._cameras(),
            "audio": {
                "sounds": self.cfg.audio.sounds,
            },
        }

    def _brain_info(self) -> dict[str, Any]:
        from omi.brain.catalog import brain_catalog, worker_catalog
        from omi.brain.cli_providers import available_providers

        installed = {p["name"] for p in available_providers(self.cfg) if p["installed"]}
        brain = self.session.brain
        worker = self.session.worker
        info = brain_catalog(self.cfg, installed)
        info.update(worker_catalog(self.cfg, installed))
        info.update({
            "talker": self.cfg.brain.talker,
            "model": self._active_model(self.cfg.brain.talker),
            "fallbacks": list(self.cfg.brain.fallbacks),
            "active": getattr(brain, "active", self.cfg.brain.talker),
            "fallback_reason": getattr(brain, "fallback_reason", ""),
            "worker_fallbacks": list(self.cfg.brain.worker_fallbacks),
            "worker_active": getattr(worker, "active_entry", "") or info.get("worker", ""),
            "worker_fallback_reason": getattr(worker, "fallback_reason", ""),
            "worker_mode": self.cfg.brain.worker_mode,
        })
        return info

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
            self.cfg.wake.enabled = want
            try:
                await self._sync_listener()
            except Exception as exc:
                self.cfg.wake.enabled = False
                self.errors["wake"] = str(exc) or type(exc).__name__
                raise RequestError(f"could not start wake word: {exc}") from exc
            self.errors.pop("wake", None)
            self._persist("wake", "enabled", want, save)
            await self.broadcast({"event": "wake", "enabled": want})
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": "on" if want else "off"}
        if key == "wake.look_to_talk":
            if self.wake is None:
                raise RequestError("wake.look_to_talk cannot be changed live (no wake listener)")
            want = self._parse_bool(value, key)
            self.cfg.wake.look_to_talk = want
            try:
                await self._sync_listener()
            except Exception as exc:
                self.cfg.wake.look_to_talk = False
                with contextlib.suppress(Exception):
                    await self._sync_listener()
                raise RequestError(f"could not start look to talk: {exc}") from exc
            self._persist("wake", "look_to_talk", want, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": "on" if want else "off"}
        if key == "brain.worker_mode":
            raw = str(value).strip().lower() if isinstance(value, str) else ""
            if raw not in WORKER_MODES:
                raise RequestError(f"brain.worker_mode expects {', '.join(WORKER_MODES)}")
            self.cfg.brain.worker_mode = raw
            # Always saved: the worker's permission check reads it from the config file.
            self._persist("brain", "worker_mode", raw, True)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": raw}
        if key == "wake.only_my_voice":
            want = self._parse_bool(value, key)
            if want and not self._voices().names():
                raise RequestError("No voice saved yet. Run `omi look-tune` (Tune in Settings) to save yours first.")
            if not want:
                self._voices().forget()  # a voiceprint is only kept while it is used
            self.cfg.wake.only_my_voice = want
            self._persist("wake", "only_my_voice", want, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": "on" if want else "off"}
        if key in LOOK_TUNING:
            lo, hi = LOOK_TUNING[key]
            try:
                num = float(value)
            except (TypeError, ValueError) as exc:
                raise RequestError(f"{key} expects a number from {lo} to {hi}") from exc
            if not lo <= num <= hi:
                raise RequestError(f"{key} expects a number from {lo} to {hi}")
            setattr(self.cfg.wake, key.removeprefix("wake."), num)
            if self.wake is not None and hasattr(self.wake, "reload_look"):
                await self.wake.reload_look()
            self._persist("wake", key.removeprefix("wake."), num, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": f"{num:g}"}
        if key == "wake.lip_check":
            want = self._parse_bool(value, key)
            self.cfg.wake.lip_check = want
            if self.wake is not None and hasattr(self.wake, "reload_look"):
                try:
                    await self.wake.reload_look()
                except Exception as exc:
                    raise RequestError(f"could not restart look to talk: {exc}") from exc
            self._persist("wake", "lip_check", want, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": "on" if want else "off"}
        if key == "wake.camera":
            raw = str(value).strip() if isinstance(value, str) else ""
            if raw.lower() in ("auto", "default"):
                raw = ""
            cameras = self._cameras()
            if raw and raw not in {c["id"] for c in cameras}:
                names = ", ".join(f'{c["name"]} ({c["id"]})' for c in cameras) or "none found"
                raise RequestError(f"no camera {raw}; cameras: auto, {names}")
            self.cfg.wake.camera = raw
            if self.wake is not None and hasattr(self.wake, "reload_look"):
                try:
                    await self.wake.reload_look()
                except Exception as exc:
                    raise RequestError(f"could not switch camera: {exc}") from exc
            self._persist("wake", "camera", raw, save)
            await self.broadcast(self._settings_event())
            return {"ok": True, "key": key, "value": raw or "auto"}
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
            all_keys = (*SETTABLE, "tts.speed", "audio.sounds", "wake.enabled", "wake.look_to_talk", "wake.lip_check", "wake.only_my_voice", *LOOK_TUNING, "wake.camera", "brain.worker_mode",
                        "wake.call_mode", "wake.conversational", "wake.sensitivity")
            raise RequestError(f"cannot set {key}; settable: {', '.join(all_keys)}")
        if not isinstance(value, str) or (
            not value and key not in ("brain.fallbacks", "brain.model", "brain.worker_model", "brain.worker_fallbacks")
        ):
            raise RequestError("set needs a string value")
        factory = self.factories.get(key)
        if SETTABLE[key] == "worker" and not hasattr(self.session.worker, "swap"):
            factory = None
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
        if attr == "worker":
            await old.swap(new)  # the session keeps one worker; only what runs inside changes
        else:
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
        elif key == "brain.worker":
            self.cfg.brain.worker = value
            self._persist("brain", "worker", value, save)
        elif key == "brain.worker_model":
            from omi.brain.cli_providers import resolve_provider

            provider = resolve_provider(self.cfg.brain.worker, role="worker")
            getattr(self.cfg.brain, provider).worker_model = value
            self._persist(f"brain.{provider}", "worker_model", value, save)
        elif key == "brain.worker_fallbacks":
            from omi.brain.fallback import format_entries, parse_entries

            self.cfg.brain.worker_fallbacks = format_entries(parse_entries(value))
            self._persist("brain", "worker_fallbacks", self.cfg.brain.worker_fallbacks, save)
        elif key == "brain.local_model":
            self.cfg.brain.local.model = os.path.expanduser(value)
            self._persist("brain.local", "model", self.cfg.brain.local.model, save)
        elif key == "brain.fallbacks":
            from omi.brain.fallback import format_entries, parse_entries

            self.cfg.brain.fallbacks = format_entries(parse_entries(value))
            self._persist("brain", "fallbacks", self.cfg.brain.fallbacks, save)
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
        raise RequestError(f"{path} is {rate} Hz; omi needs {SAMPLE_RATE} Hz")
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
