"""app.build wiring, with temp folders and no real brain or audio device."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from omi import app, config
from omi.config import Config
from omi.memory import jobs


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    """Turn traces go to a temp folder, never the real ~/.local/state/omi."""
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")


@pytest.fixture
def sock_dir():
    d = Path(tempfile.mkdtemp(prefix="hk", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


async def test_missing_models_do_not_stop_the_daemon(tmp_path, sock_dir, monkeypatch):
    monkeypatch.setenv("OMI_DRY_RUN", "1")
    cfg = Config()
    cfg.audio.sink = "null"
    cfg.stt.engine = "moonshine"
    cfg.stt.moonshine_dir = str(tmp_path / "no-stt")
    cfg.tts.piper_model = str(tmp_path / "no-tts.onnx")
    cfg.tts.kokoro_model = str(tmp_path / "no-tts.onnx")
    cfg.brain.talker = "local"
    cfg.brain.worker = "off"
    cfg.brain.fallbacks = []
    cfg.brain.local.base_url = "http://127.0.0.1:9"  # nothing listens: no llama-server is started
    cfg.memory.dir = str(tmp_path / "mem" / "voice")
    cfg.memory.shared_dir = str(tmp_path / "mem")
    cfg.memory.index = str(tmp_path / "mem.sqlite")
    cfg.web.port = 0  # never the real phone page port
    d = await app.build(cfg)
    d.socket_path = sock_dir / "omi.sock"
    await d.start()
    try:
        await d.ready.wait()
        st = d.status()
        assert "speech recognition model not found" in st["errors"]["stt"] and "no-stt" in st["errors"]["stt"]
        assert "speech synthesis model not found" in st["errors"]["tts"]
        res = await d.session.run_turn("what time is it", quiet=True)  # typed turns still work
        assert res.reply.startswith("It's") and res.timing["path"] == "router"
    finally:
        await d.close()
    json.dumps(st)


async def test_talker_is_warmed_on_the_new_memory_after_a_session_ends(tmp_path, sock_dir, monkeypatch):
    monkeypatch.setenv("OMI_DRY_RUN", "1")
    cfg = Config()
    cfg.audio.sink = "null"
    cfg.stt.engine = "moonshine"
    cfg.stt.moonshine_dir = str(tmp_path / "no-stt")
    cfg.tts.piper_model = str(tmp_path / "no-tts.onnx")
    cfg.tts.kokoro_model = str(tmp_path / "no-tts.onnx")
    cfg.brain.talker = "local"
    cfg.brain.worker = "off"
    cfg.brain.fallbacks = []
    cfg.brain.local.base_url = "http://127.0.0.1:9"
    cfg.memory.dir = str(tmp_path / "mem" / "voice")
    cfg.memory.shared_dir = str(tmp_path / "mem")
    cfg.memory.index = str(tmp_path / "mem.sqlite")
    cfg.web.enabled = False

    async def digest(memory, session_id, llm):
        memory.end_session(session_id, "- The user likes green tea.")
        return "ok"

    monkeypatch.setattr(jobs, "end_session", digest)
    d = await app.build(cfg)
    d.socket_path = sock_dir / "omi.sock"
    await d.start()
    try:
        await d.ready.wait()
        starts: list[str] = []

        async def start(memory_context=""):
            starts.append(memory_context)

        d.session.brain.start = start
        await d.session.run_turn("what time is it", quiet=True)
        await d.session.end_session()
        assert len(starts) == 1 and "green tea" in starts[0] and starts[0] == d.session.memory.context()
        await d.session.run_turn("what time is it", quiet=True)
        d.session.closed = True  # shutting down: no new talker process
        await d.session.end_session()
        assert len(starts) == 1
    finally:
        await d.close()
