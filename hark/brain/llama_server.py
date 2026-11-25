"""Owns the local llama-server process, so the local talker works with no setup.

Reuses a server already listening on the port, and only stops one it started.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import httpx

from hark.config import LocalLlmConfig

LLAMA_SERVER = "/usr/bin/llama-server"


class LlamaServer:
    def __init__(self, cfg: LocalLlmConfig, binary: str = LLAMA_SERVER) -> None:
        self.cfg = cfg
        self.binary = binary
        self.base_url = f"http://127.0.0.1:{cfg.port}"
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def owned(self) -> bool:
        return self._proc is not None

    async def _healthy(self, client: httpx.AsyncClient) -> bool:
        try:
            r = await client.get(f"{self.base_url}/health", timeout=1.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def start(self, timeout: float = 120.0) -> None:
        async with httpx.AsyncClient() as client:
            if await self._healthy(client):
                return  # someone already serves this port: reuse it
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
            self._proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self._proc.returncode is not None:
                    self._proc = None
                    raise RuntimeError("llama-server exited during startup")
                if await self._healthy(client):
                    return
                await asyncio.sleep(0.1)
            await self.close()
            raise RuntimeError("llama-server did not become healthy in time")

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 5.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
