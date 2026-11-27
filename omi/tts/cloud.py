"""Cloud text-to-speech providers over plain httpx streaming raw 24 kHz PCM.

Supported providers:
  - openai:     OpenAI gpt-4o-mini-tts / tts-1 (OPENAI_API_KEY)
  - elevenlabs: ElevenLabs eleven_flash_v2_5 (ELEVENLABS_API_KEY)

Both stream 16-bit signed little-endian mono PCM at 24 kHz directly into PipeWire so the
first audio chunk plays as soon as the first HTTP packets arrive.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import numpy as np

from omi.config import TtsConfig

PCM_RATE = 24000
CHUNK_BYTES = 4800  # 100 ms of 24 kHz 16-bit mono PCM
TIMEOUT_S = 30.0
OPENAI_VOICES = ("coral", "alloy", "ash", "ballad", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse")


def _require_key(env_name: str, provider: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(f"{provider} TTS requires the {env_name} environment variable")
    return key


def _pcm16_chunks(byte_stream: Iterator[bytes]) -> Iterator[np.ndarray]:
    """Convert arbitrary HTTP byte chunks into aligned float32 mono numpy chunks."""
    carry = b""
    for raw in byte_stream:
        if not raw:
            continue
        data = carry + raw
        even = len(data) - (len(data) % 2)
        if even:
            samples = np.frombuffer(data[:even], dtype="<i2").astype(np.float32) / 32768.0
            yield samples
        carry = data[even:]


class OpenAITTS:
    sample_rate = PCM_RATE

    def __init__(self, cfg: TtsConfig, client: httpx.Client | None = None) -> None:
        self._api_key = _require_key(cfg.openai_api_key_env, "OpenAI")
        self.url = f"{cfg.openai_base_url.rstrip('/')}/audio/speech"
        self.model = cfg.openai_model
        self.voice = cfg.openai_voice
        self.speed = float(cfg.speed)
        self._client = client or httpx.Client(timeout=TIMEOUT_S)

    def set_voice(self, voice: str) -> None:
        self.voice = voice

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.25, min(float(speed), 4.0))

    def synth(self, text: str) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        body = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": "pcm",
            "speed": max(0.25, min(self.speed, 4.0)),
        }
        with self._client.stream(
            "POST",
            self.url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        ) as resp:
            if resp.status_code >= 400:
                detail = resp.read().decode(errors="replace").strip()[:200]
                raise RuntimeError(f"OpenAI TTS failed ({resp.status_code}): {detail}")
            yield from _pcm16_chunks(resp.iter_bytes(CHUNK_BYTES))


class ElevenLabsTTS:
    sample_rate = PCM_RATE

    def __init__(self, cfg: TtsConfig, client: httpx.Client | None = None) -> None:
        self._api_key = _require_key(cfg.elevenlabs_api_key_env, "ElevenLabs")
        self.model = cfg.elevenlabs_model
        self.voice_id = cfg.elevenlabs_voice_id
        self.speed = float(cfg.speed)
        self._client = client or httpx.Client(timeout=TIMEOUT_S)

    def set_voice(self, voice_id: str) -> None:
        self.voice_id = voice_id

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.7, min(float(speed), 1.2))

    def synth(self, text: str) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        body = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
                "speed": max(0.7, min(self.speed, 1.2)),
            },
        }
        with self._client.stream(
            "POST",
            url,
            params={"output_format": "pcm_24000"},
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
            },
            json=body,
        ) as resp:
            if resp.status_code >= 400:
                detail = resp.read().decode(errors="replace").strip()[:200]
                raise RuntimeError(f"ElevenLabs TTS failed ({resp.status_code}): {detail}")
            yield from _pcm16_chunks(resp.iter_bytes(CHUNK_BYTES))
