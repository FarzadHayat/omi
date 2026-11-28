"""Wire the real components into a Daemon. Imports are lazy so `omi say` and the
tests never load models or brains."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import replace
from typing import Any

import numpy as np

from omi.config import CONFIG_PATH, Config
from omi.daemon import Daemon
from omi.events import State, TextDelta, TurnContext, TurnDone

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
    status, so the rest of omi still starts."""
    from omi import brain as brains
    from omi import memory as memories
    from omi import stt as stts
    from omi import tts as ttss
    from omi.audio.file import NullOut, WavOut
    from omi.audio.pipewire import PipeWireIn, PipeWireOut
    from omi.audio.sfx import SfxPlayer
    from omi.memory import jobs
    from omi.memory.tools import memory_tools
    from omi.router import route, warm
    from omi.session import Session
    from omi.tools import build_registry
    from omi.tools.base import ToolRegistry

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
    from omi.worker import SwitchableWorker, delegate_tool, worker_jobs_tool

    notices: list[Any] = []  # set to the daemon's broadcast once it exists

    async def on_switch(active: str, reason: str) -> None:
        log.info("talker is now %s%s", active, f" ({reason})" if reason else "")
        for send in notices:
            await send({"event": "brain", "active": active, "reason": reason})

    async def on_worker_switch(active: str, reason: str) -> None:
        log.info("worker is now %s%s", active, f" ({reason})" if reason else "")
        for send in notices:
            await send({"event": "brain", "worker_active": active, "worker_reason": reason})

    # Always present, so the settings can turn the worker on, off or over to another
    # provider while Omi runs.
    worker = SwitchableWorker(brains.create_worker_chain(cfg, on_worker_switch))
    registry.register(delegate_tool(worker))
    registry.register(worker_jobs_tool(worker))

    def chain(c: Config) -> Any:
        return brains.create_chain(c, registry, on_switch)

    talker = chain(cfg)

    async def llm(prompt: str) -> str:
        """One-shot completion for after-session jobs. A separate, tool-less brain so
        the job never adds to the live conversation or calls device tools; the local
        one reuses the running llama-server."""
        brain = brains.create_chain(cfg, ToolRegistry())
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

    # Echo cancellation: Omi plays into, and listens from, a cancelled pair so it never
    # hears itself and the user can talk over it.
    source, sink, echo = cfg.audio.source, cfg.audio.sink, None
    if cfg.audio.echo_cancel and cfg.audio.sink != "null" and not cfg.audio.source.endswith(".monitor"):
        from omi.audio import echo as echo_cancel

        echo = await asyncio.to_thread(echo_cancel.ensure, cfg.audio.source, cfg.audio.sink)
        if echo is not None:
            source, sink = echo
    if cfg.audio.barge_in and echo is None:
        log.warning("barge-in is off: echo cancellation is not available, so Omi would hear itself")
    sfx = SfxPlayer(sink=sink, enabled=lambda: cfg.audio.sounds)
    session = Session(
        stt=stt,
        tts=tts,
        brain=talker,
        registry=registry,
        router=route,
        out=NullOut() if cfg.audio.sink == "null" else PipeWireOut(sink),
        memory=memory,
        worker=worker,
        on_session_end=on_session_end,
        play_sfx=sfx.play,
    )
    registry.confirm = session.confirm

    async def new_talker(value: str) -> Any:
        brain = chain(replace(cfg, brain=replace(cfg.brain, talker=value)))
        await brain.start(memory.context())
        return brain

    async def new_model(value: str) -> Any:
        talker = getattr(session.brain, "name", cfg.brain.talker)
        if value:
            c = brains.with_model(cfg, talker, value)
        elif talker in ("claude", "local") or not hasattr(cfg.brain, talker):
            raise ValueError("pick a model")
        else:  # back to the CLI's own default model
            c = replace(cfg, brain=replace(cfg.brain, **{talker: replace(getattr(cfg.brain, talker), model="")}))
        brain = chain(c)
        await brain.start(memory.context())
        return brain

    async def new_fallbacks(value: str) -> Any:
        from omi.brain.fallback import format_entries, parse_entries

        brain = chain(replace(cfg, brain=replace(cfg.brain, fallbacks=format_entries(parse_entries(value)))))
        await brain.start(memory.context())
        return brain

    async def new_local_model(value: str) -> Any:
        path = os.path.expanduser(value)
        if not os.path.isfile(path):
            raise ValueError(f"no model file at {path}")
        # The running llama-server holds the old model on the same port: stop it first,
        # or the new brain would adopt it.
        await session.stop()
        await session.brain.close()
        brain = chain(replace(cfg, brain=replace(cfg.brain, local=replace(cfg.brain.local, model=path))))
        try:
            await brain.start(memory.context())
        except Exception:
            await brain.close()
            await session.brain.start(memory.context())  # bring the old one back
            raise
        return brain

    async def started_worker(c: Config) -> Any:
        new = brains.create_worker_chain(c, on_worker_switch)
        if new is not None:
            try:
                await new.start()
            except Exception:
                await new.close()
                raise
        return new

    async def new_worker(value: str) -> Any:
        return await started_worker(replace(cfg, brain=replace(cfg.brain, worker=value)))

    async def new_worker_model(value: str) -> Any:
        from omi.brain.cli_providers import resolve_provider

        provider = resolve_provider(cfg.brain.worker, role="worker")
        if provider == "off":
            raise ValueError("the worker is off")
        if provider == "claude":
            if not value:
                raise ValueError("pick a model")
            c = replace(cfg, brain=replace(cfg.brain, claude=replace(cfg.brain.claude, worker_model=value)))
        else:
            pcfg = getattr(cfg.brain, provider)
            c = replace(cfg, brain=replace(cfg.brain, **{provider: replace(pcfg, worker_model=value)}))
        return await started_worker(c)

    async def new_worker_fallbacks(value: str) -> Any:
        from omi.brain.fallback import format_entries, parse_entries

        return await started_worker(
            replace(cfg, brain=replace(cfg.brain, worker_fallbacks=format_entries(parse_entries(value))))
        )

    async def new_tts(value: str) -> Any:
        engine = await asyncio.to_thread(ttss.create, replace(cfg.tts, engine=value))
        await asyncio.to_thread(_warm_tts, engine)
        return engine

    async def new_voice(value: str) -> Any:
        if hasattr(session.tts, "set_voice"):
            session.tts.set_voice(value)
            return session.tts
        engine_name = cfg.tts.engine
        if engine_name == "openai":
            engine = await asyncio.to_thread(ttss.create, replace(cfg.tts, openai_voice=value))
        elif engine_name == "elevenlabs":
            engine = await asyncio.to_thread(ttss.create, replace(cfg.tts, elevenlabs_voice_id=value))
        else:
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
        "models": lambda: load_models(),
    }

    async def load_models() -> None:
        from omi.brain.catalog import load_cli_models

        await asyncio.to_thread(load_cli_models)
        for send in notices:
            await send(daemon._settings_event())
    warmups["worker"] = worker.start

    async def stop_sync() -> None:
        if hasattr(memory, "stop_autosync"):
            await asyncio.to_thread(memory.stop_autosync)

    daemon = Daemon(
        session,
        cfg,
        audio_in=PipeWireIn(source),
        factories={
            "brain.talker": new_talker,
            "brain.model": new_model,
            "brain.fallbacks": new_fallbacks,
            "brain.local_model": new_local_model,
            "brain.worker": new_worker,
            "brain.worker_model": new_worker_model,
            "brain.worker_fallbacks": new_worker_fallbacks,
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
        config_path=CONFIG_PATH,
        play_sfx=sfx.play,
    )
    notices.append(daemon.broadcast)
    if cfg.web.enabled:
        from omi.web.server import WebServer

        daemon.web = WebServer(daemon, cfg.web)
    from omi.audio.wake import WakeListener
    from omi.voice_id import VoiceGate

    daemon.wake = WakeListener(
        session,
        cfg.wake,
        audio_source=source,
        audio_sink=sink,
        barge_in=cfg.audio.barge_in and echo is not None,
        is_ptt_active=lambda: daemon._listening,
        broadcast=daemon.broadcast,
        play_sfx=sfx.play,
        voice_gate=VoiceGate(lambda: cfg.wake.only_my_voice),
    )
    from omi.look_setup import camera_tool, look_setup_tool

    registry.register(look_setup_tool(daemon.look_setup))
    registry.register(camera_tool(daemon))
    return daemon


def _warm_tts(tts: Any) -> None:
    """First synthesis pays for graph setup on local ONNX models; skip remote APIs."""
    if isinstance(tts, UnavailableTTS):
        return
    if type(tts).__module__.endswith(".cloud"):
        return
    for _ in tts.synth("Ready."):
        pass


def _warm_stt(stt: Any) -> None:
    if isinstance(stt, UnavailableSTT):
        return
    if type(stt).__module__.endswith(".cloud"):
        return
    stt.transcribe(np.zeros(8000, dtype=np.float32))
