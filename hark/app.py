"""Wire the real components into a Daemon. Imports are lazy so `hark say` and the
tests never load models or brains."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from typing import Any

import numpy as np

from hark.config import Config
from hark.daemon import Daemon
from hark.events import State, TextDelta, TurnContext, TurnDone

log = logging.getLogger(__name__)

MEMORY_SYNC_S = 15.0  # how often memory files changed by others are picked up


class UnavailableSTT:
    """Stands in for an STT engine that failed to load, so typed turns still work."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def transcribe(self, pcm: np.ndarray) -> str:
        raise RuntimeError(f"speech recognition is unavailable: {self.reason}")


class UnavailableTTS:
    """Stands in for a TTS engine that failed to load: replies still come back as text."""

    sample_rate = 16000

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def synth(self, text: str):
        raise RuntimeError(f"speech synthesis is unavailable: {self.reason}")


async def build(cfg: Config) -> Daemon:
    """Create every component (model loads run concurrently in threads) and return a
    Daemon whose start() warms the brains, worker and engines concurrently. A part that
    fails to load is replaced by a stand-in that explains itself, and is listed in
    status, so the rest of hark still starts."""
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

    errors: dict[str, str] = {}
    stt, tts, memory = await asyncio.gather(
        asyncio.to_thread(stts.create, cfg.stt),
        asyncio.to_thread(ttss.create, cfg.tts),
        asyncio.to_thread(memories.create, cfg.memory),
        return_exceptions=True,
    )
    if isinstance(stt, BaseException):
        errors["stt"] = str(stt) or type(stt).__name__
        stt = UnavailableSTT(errors["stt"])
    if isinstance(tts, BaseException):
        errors["tts"] = str(tts) or type(tts).__name__
        tts = UnavailableTTS(errors["tts"])
    if isinstance(memory, BaseException):
        errors["memory"] = str(memory) or type(memory).__name__
        memory = memories.NullMemory()
    for name, error in errors.items():
        log.error("%s failed to load: %s", name, error)
    if hasattr(memory, "start_autosync"):
        memory.start_autosync(MEMORY_SYNC_S)
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
        before = memory.context()
        await jobs.end_session(memory, session_id, llm)
        after = memory.context()
        # Warm the talker on the new memory now, while nobody talks, so the next
        # session's first turn pays for no Claude restart or local prompt re-read.
        if after != before and not session.closed and session.state is State.IDLE and session.pending_confirm is None:
            await session.brain.start(after)

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
        await brain.start(memory.context())
        return brain

    async def new_model(value: str) -> Any:
        talker = getattr(session.brain, "name", cfg.brain.talker)
        if talker == "claude":
            bcfg = replace(cfg.brain, talker="claude", claude=replace(cfg.brain.claude, model=value))
        elif talker in ("local", "openai"):
            bcfg = replace(cfg.brain, talker="local", local=replace(cfg.brain.local, model_name=value))
        elif hasattr(cfg.brain, talker):
            pcfg = getattr(cfg.brain, talker)
            bcfg = replace(cfg.brain, talker=talker, **{talker: replace(pcfg, model=value)})
        else:
            raise ValueError(f"cannot set model for unknown talker {talker}")
        brain = brains.create_talker(replace(cfg, brain=bcfg), registry)
        await brain.start(memory.context())
        return brain

    async def new_tts(value: str) -> Any:
        engine = await asyncio.to_thread(ttss.create, replace(cfg.tts, engine=value))
        await asyncio.to_thread(_warm_tts, engine)
        return engine

    async def new_voice(value: str) -> Any:
        if hasattr(session.tts, "set_voice"):
            session.tts.set_voice(value)
            return session.tts
        engine = await asyncio.to_thread(ttss.create, replace(cfg.tts, engine="kokoro", kokoro_voice=value))
        await asyncio.to_thread(_warm_tts, engine)
        return engine

    async def new_stt(value: str) -> Any:
        engine = await asyncio.to_thread(stts.create, replace(cfg.stt, engine=value))
        await asyncio.to_thread(_warm_stt, engine)
        return engine

    warmups = {
        # The talker warms up with the memory context turn one will carry, so the
        # prompt prefix is cached and nothing restarts on the first turn.
        "talker": lambda: talker.start(memory.context()),
        "tts": lambda: asyncio.to_thread(_warm_tts, tts),
        "stt": lambda: asyncio.to_thread(_warm_stt, stt),
        "router": lambda: asyncio.to_thread(warm),
    }
    if worker is not None:
        warmups["worker"] = worker.start

    async def stop_sync() -> None:
        if hasattr(memory, "stop_autosync"):
            await asyncio.to_thread(memory.stop_autosync)

    daemon = Daemon(
        session,
        cfg,
        audio_in=PipeWireIn(cfg.audio.source),
        factories={
            "brain.talker": new_talker,
            "brain.model": new_model,
            "tts.engine": new_tts,
            "tts.voice": new_voice,
            "stt.engine": new_stt,
        },
        wav_out=WavOut,
        warmups=warmups,
        closers=[stop_sync],
        # A missing profile is written from the facts once, while nobody is talking.
        idle_jobs=[lambda: jobs.profile_if_missing(memory, llm)],
        errors=errors,
        track_cursor=bool(os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")),
        config_path=config.CONFIG_PATH,
    )
    if cfg.web.enabled:
        from hark.web.server import WebServer

        daemon.web = WebServer(daemon, cfg.web)
    from hark.audio.wake import WakeListener

    daemon.wake = WakeListener(
        session,
        cfg.wake,
        audio_source=cfg.audio.source,
        audio_sink=cfg.audio.sink,
        is_ptt_active=lambda: daemon._listening,
        broadcast=daemon.broadcast,
    )
    return daemon


def _warm_tts(tts: Any) -> None:
    """First synthesis pays for graph setup; do it before the user is waiting."""
    if isinstance(tts, UnavailableTTS):
        return
    for _ in tts.synth("Ready."):
        pass


def _warm_stt(stt: Any) -> None:
    if isinstance(stt, UnavailableSTT):
        return
    stt.transcribe(np.zeros(8000, dtype=np.float32))
