"""Owns the local llama-server process, so the local talker works with no setup.

Reuses a server already listening on the port, and only stops one it started. The
server dies with the daemon (parent-death signal through setpriv), and a pidfile lets
the next daemon adopt and later stop a server an earlier daemon started, so a 4B
model never lingers in memory after hark is gone. Its log goes to a file whose tail
explains a failed start.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import time
from pathlib import Path

import httpx

from hark import config
from hark.config import LocalLlmConfig

LLAMA_SERVER = "/usr/bin/llama-server"
LOG_TAIL_LINES = 3


class LlamaServer:
    def __init__(self, cfg: LocalLlmConfig, binary: str = LLAMA_SERVER, state_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.binary = binary
        self.base_url = f"http://127.0.0.1:{cfg.port}"
        self.state_dir = Path(state_dir or config.STATE_DIR)
        self._proc: asyncio.subprocess.Process | None = None
        self._adopted: int | None = None  # pid of a server an earlier daemon started

    @property
    def owned(self) -> bool:
        return self._proc is not None or self._adopted is not None

    @property
    def pidfile(self) -> Path:
        return self.state_dir / f"llama-server-{self.cfg.port}.pid"

    @property
    def logfile(self) -> Path:
        return self.state_dir / f"llama-server-{self.cfg.port}.log"

    async def _healthy(self, client: httpx.AsyncClient) -> bool:
        try:
            r = await client.get(f"{self.base_url}/health", timeout=1.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def _ours(self) -> int | None:
        """The pid in our pidfile, if that process is still a llama-server on our port."""
        try:
            pid = int(self.pidfile.read_text().strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            stat = Path(f"/proc/{pid}/stat").read_text()
            ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            return None
        if ppid == os.getpid():
            return None  # this daemon started it, through another LlamaServer: not ours to stop
        if not any(Path(a.decode(errors="replace")).name == Path(self.binary).name for a in cmdline[:3]):
            return None
        return pid if str(self.cfg.port).encode() in cmdline else None

    def _log_tail(self) -> str:
        try:
            lines = [ln for ln in self.logfile.read_text(errors="replace").splitlines() if ln.strip()]
        except OSError:
            return ""
        return " ".join(lines[-LOG_TAIL_LINES:])[-400:]

    async def start(self, timeout: float = 120.0) -> None:
        async with httpx.AsyncClient() as client:
            if self._proc is None or self._proc.returncode is not None:
                if await self._healthy(client):
                    # Someone already serves this port: reuse it, and own it when an
                    # earlier hark daemon started it.
                    self._adopted = self._ours()
                    return
                await self._spawn()
            proc = self._proc
            assert proc is not None
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if proc.returncode is not None:
                    self._proc = None
                    with contextlib.suppress(OSError):
                        self.pidfile.unlink()
                    tail = self._log_tail()
                    raise RuntimeError(f"llama-server exited during startup{': ' + tail if tail else ''}")
                if await self._healthy(client):
                    return
                await asyncio.sleep(0.1)
            await self.close()
            raise RuntimeError("llama-server did not become healthy in time")

    async def _spawn(self) -> None:
        if not Path(self.cfg.model).is_file():
            raise RuntimeError(f"the local model file is missing: {self.cfg.model}")
        args = [
            self.binary,
            "-m", self.cfg.model,
            "--host", "127.0.0.1",
            "--port", str(self.cfg.port),
            "-ngl", str(self.cfg.gpu_layers),
            "-c", str(self.cfg.ctx),
            # No "-np 1": measured here it raised warm time to first token from 280 to 640 ms.
            "--jinja",
        ]
        setpriv = shutil.which("setpriv")
        if setpriv:
            args = [setpriv, "--pdeathsig", "TERM", "--", *args]  # dies with the daemon
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.logfile, "wb") as log:
            self._proc = await asyncio.create_subprocess_exec(
                *args, stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log)
        self.pidfile.write_text(f"{self._proc.pid}\n")

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        adopted, self._adopted = self._adopted, None
        if adopted is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(adopted, signal.SIGTERM)
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 5.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        if proc is not None or adopted is not None:
            with contextlib.suppress(OSError):
                self.pidfile.unlink()
