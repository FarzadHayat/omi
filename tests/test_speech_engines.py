"""Unit tests for local & cloud STT/TTS engines and availability discovery."""

from __future__ import annotations

import json
import numpy as np
import httpx
import pytest

from hark.config import SttConfig, TtsConfig
from hark.stt import available_stt_engines, create as create_stt
from hark.stt.cloud import DeepgramSTT, GroqSTT, OpenAISTT, pcm_to_wav_bytes
from hark.tts import available_tts_engines, create as create_tts
from hark.tts.cloud import ElevenLabsTTS, OpenAITTS


def test_pcm_to_wav_bytes_roundtrip() -> None:
    pcm = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    wav = pcm_to_wav_bytes(pcm)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) > 3200


def test_groq_and_openai_stt_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"text": "Hello from cloud!"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")

    cfg = SttConfig(hotwords=["Omi", "Omarchy"])
    pcm = np.ones(1600, dtype=np.float32) * 0.2

    g = GroqSTT(cfg, client=client)
    assert g.transcribe(pcm) == "Hello from cloud!"
    assert "api.groq.com" in str(seen[-1].url)
    assert seen[-1].headers["Authorization"] == "Bearer gsk_test"

    o = OpenAISTT(cfg, client=client)
    assert o.transcribe(pcm) == "Hello from cloud!"
    assert "api.openai.com" in str(seen[-1].url)
    assert seen[-1].headers["Authorization"] == "Bearer sk_test"


def test_deepgram_stt_keyterms(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json={"results": {"channels": [{"alternatives": [{"transcript": "Deepgram Nova three."}]}]}},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg_test")
    dg = DeepgramSTT(SttConfig(hotwords=["Omi", "Hyprland"]), client=client)
    assert dg.transcribe(np.ones(1600, dtype=np.float32) * 0.2) == "Deepgram Nova three."
    url = str(seen[0].url)
    assert "model=nova-3" in url
    assert "keyterm=Omi" in url
    assert seen[0].headers["Authorization"] == "Token dg_test"


def test_cloud_tts_streaming_pcm(monkeypatch: pytest.MonkeyPatch) -> None:
    pcm16 = (np.linspace(-0.5, 0.5, 480, dtype=np.float32) * 32767).astype("<i2").tobytes()
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, content=pcm16)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("OPENAI_API_KEY", "sk_test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el_test")

    ot = OpenAITTS(TtsConfig(openai_voice="coral", speed=1.1), client=client)
    chunks = list(ot.synth("Hello world."))
    assert len(chunks) >= 1
    out = np.concatenate(chunks)
    assert out.dtype == np.float32
    assert len(out) == 480
    body = json.loads(seen[0].content)
    assert body["voice"] == "coral"
    assert body["response_format"] == "pcm"

    et = ElevenLabsTTS(TtsConfig(), client=client)
    chunks2 = list(et.synth("Hello eleven."))
    assert len(np.concatenate(chunks2)) == 480
    assert "output_format=pcm_24000" in str(seen[1].url)
    assert seen[1].headers["xi-api-key"] == "el_test"


def test_missing_cloud_key_errors_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        create_stt(SttConfig(engine="groq"))
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        create_tts(TtsConfig(engine="elevenlabs"))


def test_available_engines_reflect_env_and_local_files(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_123")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk_123")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    stt_rows = {r["name"]: r for r in available_stt_engines(SttConfig(parakeet_dir=str(tmp_path / "none")))}
    assert stt_rows["parakeet"]["installed"] is False
    assert stt_rows["groq"]["installed"] is True
    assert stt_rows["deepgram"]["installed"] is False
    assert stt_rows["openai"]["installed"] is True

    tts_rows = {r["name"]: r for r in available_tts_engines(TtsConfig(kokoro_model=str(tmp_path / "none")))}
    assert tts_rows["kokoro"]["installed"] is False
    assert tts_rows["openai"]["installed"] is True
    assert tts_rows["elevenlabs"]["installed"] is False
