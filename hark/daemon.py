"""The hark daemon: one Session behind a Unix socket speaking hark.protocol.

Newline-delimited JSON. Turn commands stream that turn's events back on the same
connection; "subscribe" connections get every event. One turn runs at a time and a
new one cancels the running one (the Session enforces that).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
from collections.abc import Awaitable, Callable
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
# Settable keys and the Session attribute each one swaps.
SETTABLE = {"brain.talker": "brain", "tts.engine": "tts", "stt.engine": "stt"}


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
        warmups: list[Callable[[], Awaitable[Any]]] | None = None,
        closers: list[Callable[[], Awaitable[Any]]] | None = None,
    ) -> None:
        self.session = session
        self.cfg = cfg
        self.audio_in = audio_in
        self.factories = factories or {}
        self.wav_out = wav_out
        self.socket_path = Path(socket_path or config.SOCKET_PATH)
        self.warmups = warmups or []
        self.closers = closers or []
        self.ready = asyncio.Event()
        self._subs: set[asyncio.StreamWriter] = set()
        self._server: asyncio.Server | None = None
        self._listening = False
        self._warm_task: asyncio.Task[None] | None = None
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
            self._server = await asyncio.start_unix_server(self._client, path=str(self.socket_path))
        finally:
            os.umask(old)
        os.chmod(self.socket_path, 0o600)
        await self.session.start()
        self._warm_task = asyncio.create_task(self._warm())

    async def _warm(self) -> None:
        results = await asyncio.gather(*(w() for w in self.warmups), return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                log.error("warm-up failed: %s", r)
        self.ready.set()

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
        for w in list(self._subs):
            w.close()
        self._subs.clear()
        if self._warm_task is not None:
            self._warm_task.cancel()
        if self._listening and self.audio_in is not None:
            await _quietly(self.audio_in.stop())
        await _quietly(self.session.close())
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

    # --- connections ---------------------------------------------------------------

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                if not line.strip():
                    continue
                try:
                    req = json.loads(line)
                    if not isinstance(req, dict):
                        raise ValueError
                except ValueError:
                    await send(writer, {"ok": False, "error": "malformed JSON request"})
                    continue
                try:
                    reply = await self.handle(req, writer)
                except RequestError as exc:
                    reply = {"ok": False, "error": str(exc)}
                except Exception as exc:
                    log.exception("request failed: %r", req)
                    reply = {"ok": False, "error": f"internal error: {exc}"}
                if reply is not None:
                    await send(writer, reply)
        except (ConnectionError, asyncio.IncompleteReadError, ValueError):
            pass  # client went away, or sent a line over the stream limit
        finally:
            self._subs.discard(writer)
            writer.close()

    async def broadcast(self, event: dict[str, Any]) -> None:
        for w in list(self._subs):
            if not await send(w, event):
                self._subs.discard(w)

    # --- commands ----------------------------------------------------------------

    async def handle(self, req: dict[str, Any], writer: asyncio.StreamWriter) -> dict[str, Any] | None:
        cmd = req.get("cmd")
        if cmd not in COMMANDS:
            raise RequestError(f"unknown command: {cmd}")
        s = self.session
        if cmd == "say":
            text = req.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RequestError("say needs non-empty text")
            return await self._turn(writer, req, text=text)
        if cmd == "wav":
            path = req.get("path")
            if not isinstance(path, str):
                raise RequestError("wav needs a path")
            pcm = await asyncio.to_thread(read_wav, path)
            return await self._turn(writer, req, pcm=pcm)
        if cmd == "ptt_start":
            if self.audio_in is None:
                raise RequestError("no microphone configured")
            if s.state.value != "confirming":
                await s.stop()  # pressing the key talks over the assistant
            if not self._listening:
                await self.audio_in.start()
                self._listening = True
            await s.set_listening(True)
            return {"ok": True}
        if cmd == "ptt_stop":
            if self.audio_in is None or not self._listening:
                raise RequestError("not listening")
            self._listening = False
            pcm = await self.audio_in.stop()
            await s.set_listening(False)
            return await self._turn(writer, req, pcm=pcm)
        if cmd == "stop":
            if self._listening and self.audio_in is not None:
                self._listening = False
                await self.audio_in.stop()
                await s.set_listening(False)
            await s.stop()
            return {"ok": True}
        if cmd == "confirm":
            answer = req.get("answer")
            if not isinstance(answer, bool):
                raise RequestError("confirm needs answer true or false")
            if not s.answer(answer):
                raise RequestError("nothing to confirm")
            return {"ok": True}
        if cmd == "ask_confirm":
            question = req.get("question")
            if not isinstance(question, str) or not question.strip():
                raise RequestError("ask_confirm needs a question")
            return {"ok": True, "answer": await s.confirm(question)}
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
        self._subs.add(writer)
        return {"ok": True, "subscribed": True}

    def status(self) -> dict[str, Any]:
        s = self.session
        return {
            "ok": True,
            "state": s.state.value,
            "ready": self.ready.is_set(),
            "talker": self.cfg.brain.talker,
            "worker": self.cfg.brain.worker if s.worker is not None else "off",
            "engines": {"stt": self.cfg.stt.engine, "tts": self.cfg.tts.engine},
            "session_id": s.session_id,
            "timing": s.last_timing,
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

    async def _turn(self, writer: asyncio.StreamWriter, req: dict[str, Any], **kw: Any) -> dict[str, Any]:
        await self.ready.wait()
        out_path = req.get("out")
        out = None
        if out_path:
            if self.wav_out is None:
                raise RequestError("wav output is not available")
            out = self.wav_out(str(out_path))

        async def sink(event: dict[str, Any]) -> None:
            if writer not in self._subs:  # subscribers already got it
                await send(writer, event)

        res = await self.session.run_turn(
            out=out, quiet=bool(req.get("quiet")) and out is None, sink=sink, **kw
        )
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


async def send(writer: asyncio.StreamWriter, obj: dict[str, Any]) -> bool:
    """Write one JSON line. False when the client is gone."""
    if writer.is_closing():
        return False
    try:
        writer.write((json.dumps(obj) + "\n").encode())
        await writer.drain()
        return True
    except (ConnectionError, RuntimeError):
        return False


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
