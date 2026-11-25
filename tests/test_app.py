"""app.build wiring, with temp folders and no real brain or audio device."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from hark import app
from hark.config import Config


@pytest.fixture
def sock_dir():
    d = Path(tempfile.mkdtemp(prefix="hk", dir="/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


async def test_missing_models_do_not_stop_the_daemon(tmp_path, sock_dir, monkeypatch):
    monkeypatch.setenv("HARK_DRY_RUN", "1")
    cfg = Config()
    cfg.audio.sink = "null"
    cfg.stt.moonshine_dir = str(tmp_path / "no-stt")
    cfg.tts.piper_model = str(tmp_path / "no-tts.onnx")
    cfg.brain.worker = "off"
    cfg.brain.local.base_url = "http://127.0.0.1:9"  # nothing listens: no llama-server is started
    cfg.memory.dir = str(tmp_path / "mem" / "voice")
    cfg.memory.shared_dir = str(tmp_path / "mem")
    cfg.memory.index = str(tmp_path / "mem.sqlite")
    d = await app.build(cfg)
    d.socket_path = sock_dir / "hark.sock"
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
