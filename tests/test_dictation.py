"""Unit tests for Hex + Superwhisper-style dictation, formatting, paste, and daemon commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hark import config
from hark.audio.file import NullOut
from hark.config import Config, DictationConfig
from hark.daemon import Daemon
from hark.dictation import (
    ActiveContext,
    MediaPauser,
    apply_replacements,
    build_format_prompt,
    format_dictation,
    looks_like_rewrite_command,
    paste_combo_for,
    resolve_app_mode,
    strip_llm_wrapper,
)
from hark.session import Session
from hark.tools.base import ToolRegistry


class FakeAudioIn:
    def __init__(self, samples: np.ndarray) -> None:
        self._samples = samples
        self._sent = False
        self.started = False
        self.error = None

    async def start(self) -> None:
        self.started = True
        self._sent = False

    async def read(self) -> np.ndarray | None:
        if not self.started or self._sent:
            await asyncio.sleep(0.05)
            return None
        self._sent = True
        return self._samples

    async def stop(self) -> np.ndarray:
        self.started = False
        return self._samples


class FakeSTT:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, pcm: np.ndarray) -> str:
        return self.text


class FakeTTS:
    sample_rate = 16000

    def synth(self, text: str):
        yield np.zeros(160, dtype=np.float32)


class FakeBrain:
    name = "fake"

    async def start(self, memory: str = "") -> None:
        pass

    async def close(self) -> None:
        pass


class FakeMediaPauser(MediaPauser):
    def __init__(self) -> None:
        super().__init__()
        self.paused = 0
        self.resumed = 0

    async def pause_if_playing(self) -> bool:
        self.paused += 1
        return True

    async def resume_if_paused(self) -> bool:
        self.resumed += 1
        return True


def test_apply_replacements_longest_first_and_word_boundaries() -> None:
    rules = {"omi": "Omi", "super whisper": "Superwhisper", "t3": "T3 Code"}
    out = apply_replacements("ask omi about super whisper and t3 in ominous", rules)
    assert out == "ask Omi about Superwhisper and T3 Code in ominous"


def test_resolve_app_mode_defaults_and_custom_overrides() -> None:
    assert resolve_app_mode("com.mitchellh.ghostty", "zsh") == "raw"
    assert resolve_app_mode("kitty", "nvim") == "raw"
    assert resolve_app_mode("code", "main.py") == "raw"
    assert resolve_app_mode("Slack", "#engineering") == "message"
    assert resolve_app_mode("discord", "General") == "message"
    assert resolve_app_mode("org.mozilla.Thunderbird", "Inbox") == "mail"
    assert resolve_app_mode("obsidian", "Daily Notes") == "note"
    assert resolve_app_mode("chromium", "Google Docs") == "clean"
    assert resolve_app_mode("chromium", "Gmail - Inbox") == "mail"
    assert resolve_app_mode("chromium", "GitHub PR", {"github": "code"}) == "code"


def test_looks_like_rewrite_command() -> None:
    assert looks_like_rewrite_command("make this shorter")
    assert looks_like_rewrite_command("Please fix the grammar")
    assert looks_like_rewrite_command("translate to Spanish")
    assert not looks_like_rewrite_command("We should ship this on Friday.")
    assert not looks_like_rewrite_command("Hey team quick update")


def test_strip_llm_wrapper_removes_preamble_and_quotes() -> None:
    assert strip_llm_wrapper('Here is the cleaned text:\n"Ship it today."', "raw") == "Ship it today."
    assert strip_llm_wrapper("```\nhello world\n```", "raw") == "hello world"
    assert strip_llm_wrapper("", "fallback") == "fallback"


def test_paste_combo_for_terminal_vs_gui() -> None:
    assert paste_combo_for(is_terminal=True) == "ctrl+shift+v"
    assert paste_combo_for(is_terminal=False) == "ctrl+v"


@pytest.mark.asyncio
async def test_format_dictation_raw_skips_llm() -> None:
    called = False

    async def llm(prompt: str) -> str:
        nonlocal called
        called = True
        return "SHOULD NOT RUN"

    cfg = DictationConfig(mode="auto", replacements={"omi": "Omi"})
    ctx = ActiveContext(window_class="com.mitchellh.ghostty", is_terminal=True)
    res = await format_dictation("hello omi", cfg, ctx, llm=llm)
    assert res.mode == "raw"
    assert res.text == "hello Omi"
    assert not called


@pytest.mark.asyncio
async def test_format_dictation_selection_rewrite_in_auto_mode() -> None:
    prompts: list[str] = []

    async def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "Concise version."

    cfg = DictationConfig(mode="auto", use_selection=True)
    ctx = ActiveContext(window_class="chromium", selection="This is a very long and wordy sentence.")
    res = await format_dictation("make this shorter", cfg, ctx, llm=llm)
    assert res.mode == "rewrite"
    assert res.rewrote_selection is True
    assert res.text == "Concise version."
    assert "very long and wordy" in prompts[0]


@pytest.mark.asyncio
async def test_format_dictation_falls_back_on_llm_error() -> None:
    async def broken_llm(prompt: str) -> str:
        raise RuntimeError("offline")

    cfg = DictationConfig(mode="mail")
    ctx = ActiveContext(window_class="thunderbird")
    res = await format_dictation("see you tomorrow", cfg, ctx, llm=broken_llm)
    assert res.text == "see you tomorrow"
    assert res.mode == "raw"
    assert res.error == "offline"


@pytest.mark.asyncio
async def test_daemon_dictate_flow_and_settings(tmp_path: Path) -> None:
    cfg = Config()
    cfg.dictation.mode = "auto"
    cfg.dictation.replacements = {"omi": "Omi"}
    cfg_file = tmp_path / "config.toml"
    sock = tmp_path / "omi.sock"

    inserted: list[tuple[str, bool]] = []
    sfx_events: list[str] = []
    pauser = FakeMediaPauser()

    async def fake_ctx(*, read_selection: bool = True) -> ActiveContext:
        return ActiveContext(window_class="Slack", window_title="#general", is_terminal=False)

    async def fake_insert(text: str, *, is_terminal: bool = False, **kw: Any) -> tuple[bool, str]:
        inserted.append((text, is_terminal))
        return True, "paste:ctrl+v"

    async def fake_llm(prompt: str) -> str:
        assert "Mode (message)" in prompt
        return "Hey team, Omi is live!"

    session = Session(
        stt=FakeSTT("um hey team omi is live"),
        tts=FakeTTS(),
        brain=FakeBrain(),
        registry=ToolRegistry(),
        router=lambda _text: None,
        out=NullOut(),
    )
    daemon = Daemon(
        session,
        cfg,
        audio_in=FakeAudioIn(np.ones(3200, dtype=np.float32) * 0.1),
        socket_path=sock,
        config_path=cfg_file,
        play_sfx=sfx_events.append,
        dictation_llm=fake_llm,
        dictation_context_fn=fake_ctx,
        dictation_insert_fn=fake_insert,
        media_pauser=pauser,
    )
    await daemon.start()
    await daemon.ready.wait()

    async def rpc(req: dict[str, Any]) -> dict[str, Any]:
        r, w = await asyncio.open_unix_connection(str(sock))
        try:
            w.write((json.dumps(req) + "\n").encode())
            await w.drain()
            return json.loads(await r.readline())
        finally:
            w.close()

    try:
        r1 = await rpc({"cmd": "dictate_start"})
        assert r1["ok"] and r1["dictating"]
        assert pauser.paused == 1

        r2 = await rpc({"cmd": "dictate_stop"})
        assert r2["ok"] and r2["done"]
        assert r2["mode"] == "message"
        assert r2["text"] == "Hey team, Omi is live!"
        assert r2["inserted"] is True
        assert inserted == [("Hey team, Omi is live!", False)]
        assert pauser.resumed == 1
        assert sfx_events == ["wake", "heard", "done"]

        # Live dictation settings persist cleanly to config.toml
        r3 = await rpc({"cmd": "set", "key": "dictation.mode", "value": "raw", "save": True})
        assert r3 == {"ok": True, "key": "dictation.mode", "value": "raw"}
        r4 = await rpc({"cmd": "set", "key": "dictation.pause_media", "value": "off", "save": True})
        assert r4 == {"ok": True, "key": "dictation.pause_media", "value": "off"}

        reloaded = config.load(cfg_file)
        assert reloaded.dictation.mode == "raw"
        assert reloaded.dictation.pause_media is False
    finally:
        await daemon.close()
