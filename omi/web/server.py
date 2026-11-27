"""The phone and tablet page: a small HTTP and WebSocket server inside the daemon.

It listens on 127.0.0.1 only and is reached through `tailscale serve`, which proxies
from tailscaled (a root process) and adds the Tailscale-User-Login and X-Forwarded-For
headers. A request is answered only when it came from that proxy, from another device
on the tailnet (tailscaled also stamps the owner's login on requests made from this PC,
so those are refused by address), and the login is on the allowlist; a WebSocket also
needs an Origin of the page itself. The wire format is in omi.protocol.

Each phone gets its own WebSocket. Its turns play their reply on the phone (PhoneOut),
never on the PC speakers, and it sees every session event like a subscriber.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
import struct
import tomllib
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from omi.config import WebConfig
from omi.protocol import WEB_AUDIO_HEADER, WEB_COMMANDS
from omi.web.websocket import WebSocket, accept_key

if TYPE_CHECKING:
    from omi.daemon import Daemon

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/mic-worklet.js": ("mic-worklet.js", "text/javascript; charset=utf-8"),
    "/sw.js": ("sw.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "/icon.svg": ("icon.svg", "image/svg+xml"),
    "/icon-192.png": ("icon-192.png", "image/png"),
    "/icon-512.png": ("icon-512.png", "image/png"),
}
HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; connect-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; "
        "worker-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "microphone=(self)",
    "Cache-Control": "no-cache",
}
LOGIN_HEADER = "tailscale-user-login"
FORWARDED_HEADER = "x-forwarded-for"  # set by tailscaled to the connecting device's address
TAILNET = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))
MAX_HEAD = 16384  # bytes of request line plus headers
KEEPALIVE_S = 30.0
OUTBOX_LIMIT = 4000  # queued frames per phone before a phone that does not read is dropped
DRAIN_SLACK_S = 1.0  # a reply counts as played this long after its expected end, ack or not
MAX_UTTERANCE_S = 60.0  # a phone's push-to-talk is cut after this much audio or wall time
MAX_SAY_CHARS = 2000
INTERNAL_ERROR = "something went wrong on the PC"
THEME_KEYS = {
    "background": "bg", "foreground": "fg", "lighter_background": "surface", "dark_background": "bg-deep",
    "dark_foreground": "muted", "accent": "accent", "red": "red", "green": "green", "yellow": "yellow",
    "orange": "orange", "blue": "blue", "magenta": "magenta", "cyan": "cyan",
}


# --- framing -------------------------------------------------------------------------


def decode_mic(data: bytes) -> np.ndarray:
    """A phone mic frame (16 kHz mono Int16 little-endian) as float32."""
    n = len(data) // 2 * 2
    return np.frombuffer(data[:n], dtype="<i2").astype(np.float32) / 32768.0


def encode_reply(pcm: np.ndarray, sample_rate: int, seq: int) -> bytes:
    """Reply audio for the phone: the header (sample rate, turn number), then Int16 LE."""
    data = (np.clip(np.asarray(pcm, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2")
    return struct.pack(WEB_AUDIO_HEADER, sample_rate, seq) + data.tobytes()


def decode_reply(frame: bytes) -> tuple[int, int, np.ndarray]:
    """(sample_rate, seq, float32 pcm) from a reply frame; the inverse of encode_reply."""
    size = struct.calcsize(WEB_AUDIO_HEADER)
    rate, seq = struct.unpack(WEB_AUDIO_HEADER, frame[:size])
    return rate, seq, decode_mic(frame[size:])


# --- who may connect -----------------------------------------------------------------


def owner_login(status: dict[str, Any]) -> str | None:
    """The login of the user who owns this machine, from `tailscale status --json`."""
    try:
        uid = status["Self"]["UserID"]
        return status["User"][str(uid)]["LoginName"] or None
    except (KeyError, TypeError):
        return None


def own_addresses(status: dict[str, Any]) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """This machine's own tailnet addresses, from `tailscale status --json`."""
    out = set()
    try:
        for ip in status["Self"]["TailscaleIPs"] or ():
            out.add(ipaddress.ip_address(ip))
    except (KeyError, TypeError, ValueError):
        return set()
    return out


async def tailnet_status() -> dict[str, Any]:
    """`tailscale status --json`, or {} when it cannot be read."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "tailscale", "status", "--json", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await asyncio.wait_for(proc.communicate(), 5.0)
        status = json.loads(out)
        return status if isinstance(status, dict) else {}
    except (OSError, TimeoutError, ValueError) as exc:
        log.error("could not read the tailnet status: %s", exc)
        return {}


def forwarded_for(req: Request) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The connecting device's address as tailscaled wrote it (the last hop is its own)."""
    raw = req.headers.get(FORWARDED_HEADER, "").rsplit(",", 1)[-1].strip()
    try:
        return ipaddress.ip_address(raw)
    except ValueError:
        return None


def _proc_addr(host: str, port: int) -> str:
    return "%08X:%04X" % (struct.unpack("<I", socket.inet_aton(host))[0], port)


def peer_uid(sock: socket.socket, table: str = "/proc/net/tcp") -> int | None:
    """Owner uid of the process on the other end of a loopback TCP connection: its socket
    is the /proc/net/tcp row whose local address is our peer's address."""
    try:
        here, there = sock.getsockname(), sock.getpeername()
        want = (_proc_addr(there[0], there[1]), _proc_addr(here[0], here[1]))
        with open(table) as fh:
            for line in fh.readlines()[1:]:
                parts = line.split()
                if (parts[1], parts[2]) == want:
                    return int(parts[7])
    except (OSError, ValueError, IndexError):
        pass
    return None


def theme_css(path: str | Path) -> str:
    """CSS variables from the active Omarchy theme; empty when it cannot be read (the
    page then keeps its dark neutral palette)."""
    try:
        with open(path, "rb") as fh:
            colors = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    lines = []
    for key, var in THEME_KEYS.items():
        value = colors.get(key)
        if isinstance(value, str) and _is_hex(value):
            lines.append(f"  --{var}: {value};")
    if not lines:
        return ""
    scheme = "light" if colors.get("mode") == "light" else "dark"
    return ":root {\n  color-scheme: %s;\n%s\n}\n" % (scheme, "\n".join(lines))


def _is_hex(value: str) -> bool:
    v = value[1:]
    return value.startswith("#") and len(v) in (3, 6, 8) and all(c in "0123456789abcdefABCDEF" for c in v)


# --- audio to the phone ----------------------------------------------------------------


class PhoneOut:
    """AudioOut that streams one turn's reply to the phone that asked. drain() waits
    until the phone says it finished playing (or the expected end plus a little)."""

    def __init__(self, client: WebClient, seq: int) -> None:
        self.client = client
        self.seq = seq
        self.rate = 16000
        self._t0: float | None = None
        self._samples = 0
        self._stopped = asyncio.Event()
        self._played = client.played.setdefault(seq, asyncio.Event())

    async def start(self, sample_rate: int) -> None:
        self.rate = sample_rate

    async def write(self, pcm: np.ndarray) -> None:
        if self._t0 is None:
            self._t0 = monotonic()
        self._samples += len(pcm)
        self.client.send_bytes(encode_reply(pcm, self.rate, self.seq))

    async def drain(self) -> None:
        try:
            if self._t0 is None or self.client.closed:
                return
            self.client.send({"event": "audio", "phase": "end", "seq": self.seq})
            budget = max(0.0, self._t0 + self._samples / self.rate - monotonic()) + DRAIN_SLACK_S
            waits = [asyncio.ensure_future(e.wait()) for e in (self._played, self._stopped, self.client.gone)]
            try:
                await asyncio.wait(waits, timeout=budget, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for w in waits:
                    w.cancel()
        finally:
            self.client.played.pop(self.seq, None)

    async def stop(self) -> None:
        self._stopped.set()
        self.client.played.pop(self.seq, None)
        self.client.send({"event": "audio", "phase": "stop", "seq": self.seq})


# --- one phone ---------------------------------------------------------------------------


class RequestError(Exception):
    pass


class WebClient:
    """One phone's WebSocket. Duck-types the daemon's Conn (send, abort) so it can sit in
    the daemon's subscriber set and get every event."""

    def __init__(self, daemon: Daemon, ws: WebSocket, login: str) -> None:
        self.daemon = daemon
        self.ws = ws
        self.login = login
        self.closed = False
        self.gone = asyncio.Event()
        self.played: dict[int, asyncio.Event] = {}
        self._seq = 0
        self._utt: Any = None  # the session utterance this phone opened
        self._turns: set[asyncio.Task[Any]] = set()
        self._limit: asyncio.TimerHandle | None = None  # ends a push-to-talk that runs too long
        self._queue: asyncio.Queue[str | bytes | None] = asyncio.Queue(OUTBOX_LIMIT)
        self._pump = asyncio.create_task(self._run_pump())

    # Conn interface
    def send(self, obj: dict[str, Any]) -> bool:
        return self._put(json.dumps(obj))

    def send_bytes(self, data: bytes) -> bool:
        return self._put(data)

    def _put(self, item: str | bytes) -> bool:
        if self.closed:
            return False
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            log.warning("dropping a phone that stopped reading")
            self.abort()
            return False

    def abort(self) -> None:
        self.closed = True
        self.gone.set()
        self._pump.cancel()
        self.ws.writer.close()

    async def _run_pump(self) -> None:
        try:
            while (item := await self._queue.get()) is not None:
                await self.ws.send(item)
        except (ConnectionError, RuntimeError):
            pass
        finally:
            self.closed = True
            self.gone.set()

    # session side
    @property
    def session(self) -> Any:
        return self.daemon.session

    def _owns_utterance(self) -> bool:
        # The session keeps one open utterance; the desktop key or another phone may
        # have taken it over since this phone pressed talk.
        return self._utt is not None and getattr(self.session, "_utt", None) is self._utt

    async def serve(self) -> None:
        s = self.session
        self.daemon._subs.add(self)
        self.send({"event": "hello", "name": self.daemon.cfg.name, "state": s.state.value,
                   "confirm": s.pending_confirm, "ready": self.daemon.ready.is_set()})
        try:
            while (msg := await self.ws.recv()) is not None:
                if isinstance(msg, bytes):
                    if self._owns_utterance():
                        if self._utt.samples + len(msg) // 2 > MAX_UTTERANCE_S * 16000:
                            await self._drop_utterance("talked for too long; that was not sent")
                        else:
                            s.feed_audio(decode_mic(msg))
                    continue
                try:
                    req = json.loads(msg)
                    if not isinstance(req, dict):
                        raise ValueError
                except ValueError:
                    self.send({"event": "error", "text": "malformed JSON"})
                    continue
                try:
                    await self.handle(req)
                except RequestError as exc:
                    self.send({"event": "error", "text": str(exc), "cmd": req.get("cmd")})
                except Exception:
                    log.exception("phone request failed: %r", req.get("cmd"))
                    self.send({"event": "error", "text": INTERNAL_ERROR, "cmd": req.get("cmd")})
        finally:
            self.daemon._subs.discard(self)
            self._cancel_limit()
            if self._owns_utterance():
                self._utt = None
                await s.cancel_utterance()
            await self.ws.close()
            self.closed = True
            self.gone.set()
            self._pump.cancel()

    async def handle(self, req: dict[str, Any]) -> None:
        cmd = req.get("cmd")
        if cmd not in WEB_COMMANDS:
            raise RequestError(f"unknown command: {cmd}")
        s = self.session
        if cmd == "ptt_start":
            await self.daemon.release_mic()  # the desktop key's capture must not feed this one
            await s.begin_utterance()  # talks over the assistant
            self._utt = getattr(s, "_utt", None)
            self._cancel_limit()
            # A phone that drops off the network while holding talk sends nothing more.
            self._limit = asyncio.get_running_loop().call_later(
                MAX_UTTERANCE_S, lambda utt=self._utt: self._spawn(self._expire(utt)))
        elif cmd == "ptt_stop":
            self._cancel_limit()
            if not self._owns_utterance():
                self._utt = None
                raise RequestError("not listening")
            self._utt = None
            self._spawn(self._turn(None))
        elif cmd == "say":
            text = req.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RequestError("say needs non-empty text")
            if len(text) > MAX_SAY_CHARS:
                raise RequestError(f"say text is limited to {MAX_SAY_CHARS} characters")
            self._spawn(self._turn(text))
        elif cmd == "stop":
            self._cancel_limit()
            if self._owns_utterance():
                self._utt = None
                await s.cancel_utterance()
            await s.stop()
        elif cmd == "confirm":
            answer, cid = req.get("answer"), req.get("id")
            if not isinstance(answer, bool) or not isinstance(cid, str) or not cid:
                raise RequestError("confirm needs answer true or false and the question's id")
            if not s.answer(answer, cid):
                raise RequestError("nothing to confirm" if s.pending_confirm is None
                                   else "that question is no longer the one waiting")
        elif cmd == "played":
            ev = self.played.get(req.get("seq"))  # type: ignore[arg-type]
            if ev is not None:
                ev.set()

    def _cancel_limit(self) -> None:
        if self._limit is not None:
            self._limit.cancel()
            self._limit = None

    async def _drop_utterance(self, why: str) -> None:
        self._cancel_limit()
        self._utt = None
        await self.session.cancel_utterance()
        self.send({"event": "error", "text": why, "cmd": "ptt_start"})

    async def _expire(self, utt: Any) -> None:
        if self._utt is utt and self._owns_utterance():
            await self._drop_utterance("talked for too long; that was not sent")

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._turns.add(task)
        task.add_done_callback(self._turns.discard)

    async def _turn(self, text: str | None) -> None:
        """One turn whose reply plays on this phone."""
        s = self.session
        self._seq += 1
        out = PhoneOut(self, self._seq)
        try:
            if text is None:
                res = await s.end_utterance(out=out)
            else:
                await self.daemon.ready.wait()
                res = await s.run_turn(text, out=out)
        finally:
            self.played.pop(out.seq, None)
        final: dict[str, Any] = {"event": "done", "ok": res.error is None, "reply": res.reply,
                                 "timing": res.timing, "cancelled": res.cancelled}
        if res.error is not None:
            final["error"] = res.error
        self.send(final)


# --- the server ------------------------------------------------------------------------


class Request:
    def __init__(self, method: str, path: str, headers: dict[str, str]) -> None:
        self.method = method
        self.path = path
        self.headers = headers


def parse_head(raw: bytes) -> Request | None:
    try:
        text = raw.decode("latin-1")
        first, *rest = text.split("\r\n")
        method, target, version = first.split(" ")
        if not version.startswith("HTTP/1."):
            return None
    except ValueError:
        return None
    headers: dict[str, str] = {}
    for line in rest:
        if not line:
            continue
        if ":" not in line:
            return None
        k, v = line.split(":", 1)
        k = k.strip().lower()
        if k in (LOGIN_HEADER, FORWARDED_HEADER) and k in headers:
            return None  # one identity per request
        headers[k] = v.strip()
    return Request(method, target.split("?", 1)[0], headers)


class WebServer:
    def __init__(
        self,
        daemon: Daemon,
        cfg: WebConfig,
        *,
        logins: set[str] | None = None,
        own_ips: set[str] | None = None,
        peer_ok: Callable[[socket.socket], bool] | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self.daemon = daemon
        self.cfg = cfg
        self.logins = set(logins) if logins is not None else None
        # This PC's tailnet addresses: requests from them are local programs, not a phone.
        self.own_ips = {ipaddress.ip_address(ip) for ip in own_ips} if own_ips is not None else None
        self.peer_ok = peer_ok or (lambda sock: peer_uid(sock) == cfg.proxy_uid)
        self.host = host
        self.port = cfg.port
        self.clients: set[WebClient] = set()
        self._server: asyncio.Server | None = None
        self._conns: set[asyncio.Task[Any]] = set()

    async def start(self) -> None:
        status: dict[str, Any] | None = None
        if self.logins is None:
            self.logins = set(self.cfg.allowed_logins)
            if not self.logins:
                status = await tailnet_status()
                if owner := owner_login(status):
                    self.logins = {owner}
        if self.own_ips is None:
            self.own_ips = own_addresses(status if status is not None else await tailnet_status())
        if not self.logins:
            self.daemon.errors["web"] = "no Tailscale login is allowed; the phone page refuses everyone"
        elif not self.own_ips:
            self.daemon.errors["web"] = "could not read this PC's tailnet addresses; the phone page refuses everyone"
        self._server = await asyncio.start_server(self._conn, self.host, self.port, limit=MAX_HEAD)
        self.port = self._server.sockets[0].getsockname()[1]
        log.info("phone page on http://%s:%d", self.host, self.port)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        for c in list(self.clients):
            c.abort()
        for t in list(self._conns):
            t.cancel()
        await asyncio.gather(*self._conns, return_exceptions=True)

    def allowed(self, req: Request, sock: socket.socket | None) -> str | None:
        """None when the request may be answered, else why not."""
        if sock is None or not self.peer_ok(sock):
            return "only through tailscale serve"
        ip = forwarded_for(req)
        if ip is None or not any(ip in net for net in TAILNET):
            return "only from a device on the tailnet"
        if not self.own_ips or ip in self.own_ips:
            return "only from another device, not from this PC"
        login = req.headers.get(LOGIN_HEADER, "")
        if not login or login not in (self.logins or ()):
            return "not an allowed Tailscale login"
        return None

    def origin_ok(self, req: Request) -> bool:
        origin, host = req.headers.get("origin", ""), req.headers.get("host", "")
        return bool(origin) and (origin in self.cfg.origins or (bool(host) and origin == f"https://{host}"))

    async def _conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._conns.add(task)
        sock = writer.get_extra_info("socket")
        handed_off = False
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), KEEPALIVE_S)
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError, TimeoutError,
                        ConnectionError):
                    break
                req = parse_head(raw)
                if req is None:
                    await _respond(writer, 400, b"bad request")
                    break
                if req.headers.get("content-length", "0") != "0" or "transfer-encoding" in req.headers:
                    await _respond(writer, 400, b"no request bodies")
                    break
                why = self.allowed(req, sock)
                if why is not None:
                    log.warning("phone page refused %s %s: %s", req.method, req.path, why)
                    await _respond(writer, 403, why.encode())
                    break
                if req.path == "/ws":
                    handed_off = True
                    await self._websocket(req, reader, writer)
                    return
                if req.method not in ("GET", "HEAD"):
                    await _respond(writer, 405, b"method not allowed")
                    break
                await self._static(req, writer)
                if req.headers.get("connection", "").lower() == "close":
                    break
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            if task is not None:
                self._conns.discard(task)
            if not handed_off:
                writer.close()

    async def _static(self, req: Request, writer: asyncio.StreamWriter) -> None:
        if req.path == "/theme.css":
            body = (await asyncio.to_thread(theme_css, self.cfg.theme)).encode()
            ctype = "text/css; charset=utf-8"
        elif req.path in FILES:
            name, ctype = FILES[req.path]
            body = await asyncio.to_thread((STATIC / name).read_bytes)
        else:
            await _respond(writer, 404, b"not found", keep=True)
            return
        await _respond(writer, 200, body, ctype, keep=True, head=req.method == "HEAD")

    async def _websocket(self, req: Request, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        h = req.headers
        key = h.get("sec-websocket-key", "")
        if (req.method != "GET" or h.get("upgrade", "").lower() != "websocket"
                or "upgrade" not in h.get("connection", "").lower() or h.get("sec-websocket-version") != "13" or not key):
            await _respond(writer, 400, b"websocket upgrade expected")
            writer.close()
            return
        if not self.origin_ok(req):
            log.warning("phone page refused a websocket from origin %r", h.get("origin"))
            await _respond(writer, 403, b"origin not allowed")
            writer.close()
            return
        writer.write((
            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n"
        ).encode())
        await writer.drain()
        sock = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client = WebClient(self.daemon, WebSocket(reader, writer), h.get(LOGIN_HEADER, ""))
        self.clients.add(client)
        try:
            await client.serve()
        finally:
            self.clients.discard(client)


async def _respond(
    writer: asyncio.StreamWriter, status: int, body: bytes, ctype: str = "text/plain; charset=utf-8",
    keep: bool = False, head: bool = False,
) -> None:
    reason = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found", 405: "Method Not Allowed"}[status]
    lines = [f"HTTP/1.1 {status} {reason}", f"Content-Type: {ctype}", f"Content-Length: {len(body)}",
             f"Connection: {'keep-alive' if keep else 'close'}"]
    lines += [f"{k}: {v}" for k, v in HEADERS.items()]
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode() + (b"" if head else body))
    await writer.drain()
