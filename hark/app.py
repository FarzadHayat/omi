"""Wire the real components into a Daemon. Imports are lazy so `hark say` and the
tests never load models or brains."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any

import numpy as np

from hark.config import Config
from hark.daemon import Daemon
from hark.events import TextDelta, TurnContext, TurnDone

log = logging.getLogger(__name__)


async def build(cfg: Config) -> Daemon:
    """Create every component (model loads run concurrently in threads) and return a
    Daemon whose start() warms the brains, worker and engines concurrently."""
    from hark import brain as brains
    from hark import memory as memories
    from hark import stt as stts
    from hark import tts as ttss
    from hark.audio.file import NullOut, WavOut
    from hark.audio.pipewire import PipeWireIn, PipeWireOut
    from hark.memory import jobs
    from hark.memory.tools import memory_tools
    from hark.router import route, warm
    from hark.session import Session
    from hark.tools import build_registry
    from hark.tools.base import ToolRegistry

    stt, tts, memory = await asyncio.gather(
        asyncio.to_thread(stts.create, cfg.stt),
        asyncio.to_thread(ttss.create, cfg.tts),
        asyncio.to_thread(memories.create, cfg.memory),
    )
    registry = build_registry(cfg)
    for tool in memory_tools(memory):
        registry.register(tool)
    worker = brains.create_worker(cfg)
    if worker is not None:
        from hark.worker import delegate_tool

        registry.register(delegate_tool(worker))
    talker = brains.create_talker(cfg, registry)

    async def llm(prompt: str) -> str:
        """One-shot completion for after-session jobs. A separate, tool-less brain so
        the job never adds to the live conversation or calls device tools; the local
        one reuses the running llama-server."""
        brain = brains.create_talker(cfg, ToolRegistry())
        await brain.start()
        try:
            parts: list[str] = []
            async for ev in brain.turn(prompt, TurnContext(session_id="memory-job")):
                if isinstance(ev, TextDelta):
                    parts.append(ev.text)
                elif isinstance(ev, TurnDone):
                    if ev.error:
                        raise RuntimeError(ev.error)
                    break
            return "".join(parts)
        finally:
            await brain.close()

    async def on_session_end(session_id: str, transcript: str) -> None:
        # jobs.end_session reads the transcript from memory (Session logged it there).
        await jobs.end_session(memory, session_id, llm)

    session = Session(
        stt=stt,
        tts=tts,
        brain=talker,
        registry=registry,
        router=route,
        out=NullOut() if cfg.audio.sink == "null" else PipeWireOut(cfg.audio.sink),
        memory=memory,
        worker=worker,
        on_session_end=on_session_end,
    )
    registry.confirm = session.confirm

    async def new_talker(value: str) -> Any:
        brain = brains.create_talker(replace(cfg, brain=replace(cfg.brain, talker=value)), registry)
        await brain.start()
        return brain

    async def new_tts(value: str) -> Any:
        engine = await asyncio.to_thread(ttss.create, replace(cfg.tts, engine=value))
        await asyncio.to_thread(_warm_tts, engine)
        return engine

    async def new_stt(value: str) -> Any:
        engine = await asyncio.to_thread(stts.create, replace(cfg.stt, engine=value))
        await asyncio.to_thread(_warm_stt, engine)
        return engine

    warmups = [
        talker.start,
        lambda: asyncio.to_thread(_warm_tts, tts),
        lambda: asyncio.to_thread(_warm_stt, stt),
        lambda: asyncio.to_thread(warm),
    ]
    if worker is not None:
        warmups.append(worker.start)

    return Daemon(
        session,
        cfg,
        audio_in=PipeWireIn(cfg.audio.source),
        factories={"brain.talker": new_talker, "tts.engine": new_tts, "stt.engine": new_stt},
        wav_out=WavOut,
        warmups=warmups,
    )


def _warm_tts(tts: Any) -> None:
    """First synthesis pays for graph setup; do it before the user is waiting."""
    for _ in tts.synth("Ready."):
        pass


def _warm_stt(stt: Any) -> None:
    stt.transcribe(np.zeros(8000, dtype=np.float32))
