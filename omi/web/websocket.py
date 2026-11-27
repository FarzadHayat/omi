"""A small RFC 6455 WebSocket over asyncio streams, server and client side.

uvicorn needs the websockets or wsproto package for WebSockets and neither is
installed, so this covers what omi needs: text and binary messages, fragments,
ping, pong and close. No extensions.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import ssl as ssl_mod
import struct
from typing import Any

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_MESSAGE = 1 << 20  # bytes; a 20 ms mic frame is 640

CONT, TEXT, BINARY, CLOSE, PING, PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class ProtocolError(Exception):
    pass


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()


def encode_frame(opcode: int, payload: bytes, mask: bool = False) -> bytes:
    n = len(payload)
    head = bytearray([0x80 | opcode])
    bit = 0x80 if mask else 0
    if n < 126:
        head.append(bit | n)
    elif n < 1 << 16:
        head.append(bit | 126)
        head += struct.pack("!H", n)
    else:
        head.append(bit | 127)
        head += struct.pack("!Q", n)
    if not mask:
        return bytes(head) + payload
    key = os.urandom(4)
    return bytes(head) + key + _mask(payload, key)


def _mask(data: bytes, key: bytes) -> bytes:
    n = len(data)
    if not n:
        return b""
    full = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "little") ^ int.from_bytes(full, "little")).to_bytes(n, "little")


class WebSocket:
    """One open connection. recv() returns str, bytes, or None once closed."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, client: bool = False) -> None:
        self.reader = reader
        self.writer = writer
        self.client = client  # a client masks what it sends; a server requires masked input
        self.closed = False
        self._lock = asyncio.Lock()

    async def _send(self, opcode: int, payload: bytes) -> None:
        if self.closed and opcode != CLOSE:
            raise ConnectionError("websocket is closed")
        async with self._lock:
            self.writer.write(encode_frame(opcode, payload, mask=self.client))
            await self.writer.drain()

    async def send(self, data: str | bytes) -> None:
        if isinstance(data, str):
            await self._send(TEXT, data.encode())
        else:
            await self._send(BINARY, bytes(data))

    async def _frame(self) -> tuple[bool, int, bytes]:
        b0, b1 = await self.reader.readexactly(2)
        if b0 & 0x70:
            raise ProtocolError("reserved bits set")
        fin, opcode, masked, n = bool(b0 & 0x80), b0 & 0x0F, bool(b1 & 0x80), b1 & 0x7F
        if masked == self.client:
            raise ProtocolError("wrong masking")
        if n == 126:
            (n,) = struct.unpack("!H", await self.reader.readexactly(2))
        elif n == 127:
            (n,) = struct.unpack("!Q", await self.reader.readexactly(8))
        if n > MAX_MESSAGE:
            raise ProtocolError("message too large")
        key = await self.reader.readexactly(4) if masked else b""
        data = await self.reader.readexactly(n)
        return fin, opcode, _mask(data, key) if masked else data

    async def recv(self) -> str | bytes | None:
        parts: list[bytes] = []
        kind = None
        try:
            while not self.closed:
                fin, opcode, data = await self._frame()
                if opcode == PING:
                    await self._send(PONG, data)
                elif opcode == PONG:
                    pass
                elif opcode == CLOSE:
                    await self.close()
                    return None
                elif opcode in (TEXT, BINARY, CONT):
                    if (opcode == CONT) == (kind is None):
                        raise ProtocolError("bad fragment")
                    kind = kind if opcode == CONT else opcode
                    parts.append(data)
                    if sum(map(len, parts)) > MAX_MESSAGE:
                        raise ProtocolError("message too large")
                    if fin:
                        raw = b"".join(parts)
                        return raw.decode() if kind == TEXT else raw
                else:
                    raise ProtocolError(f"unknown opcode {opcode}")
        except (asyncio.IncompleteReadError, ConnectionError, ProtocolError, UnicodeDecodeError, ValueError):
            await self.close(1002)
        return None

    async def close(self, code: int = 1000) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            await asyncio.wait_for(self._send(CLOSE, struct.pack("!H", code)), 1.0)
        except (ConnectionError, TimeoutError, RuntimeError):
            pass
        self.writer.close()


async def connect(
    url: str, headers: dict[str, str] | None = None, ssl: ssl_mod.SSLContext | bool | None = None
) -> WebSocket:
    """Open a client connection to ws:// or wss:// url. Raises ConnectionError with the
    HTTP status line when the server refuses the upgrade."""
    scheme, rest = url.split("://", 1)
    hostport, _, path = rest.partition("/")
    host, _, port = hostport.partition(":")
    secure = scheme == "wss"
    reader, writer = await asyncio.open_connection(
        host, int(port or (443 if secure else 80)), ssl=(ssl or True) if secure else None, limit=MAX_MESSAGE * 2
    )
    key = base64.b64encode(os.urandom(16)).decode()
    lines = [
        f"GET /{path} HTTP/1.1",
        f"Host: {hostport}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ] + [f"{k}: {v}" for k, v in (headers or {}).items()]
    writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
    await writer.drain()
    head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
    status = head.split("\r\n", 1)[0]
    got: dict[str, Any] = {}
    for line in head.split("\r\n")[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            got[k.strip().lower()] = v.strip()
    if " 101 " not in status + " " or got.get("sec-websocket-accept") != accept_key(key):
        writer.close()
        raise ConnectionError(status)
    return WebSocket(reader, writer, client=True)
