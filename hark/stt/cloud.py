"""Cloud speech-to-text providers over plain httpx (zero extra SDKs).

Supported providers:
  - groq:     Groq Whisper Large v3 Turbo (~150 ms cloud latency, GROQ_API_KEY)
  - deepgram: Deepgram Nova-3 (smart formatting + keyterms, DEEPGRAM_API_KEY)
  - openai:   OpenAI gpt-4o-mini-transcribe / gpt-4o-transcribe (OPENAI_API_KEY)

API keys are always read from environment variables; secrets are never stored in config.
"""

from __future__ import annotations

import io
import os
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from hark.config import SttConfig
from hark.stt.base import SAMPLE_RATE

TIMEOUT_S = 20.0


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode float32 mono PCM in [-1, 1] into compact 16-bit WAV bytes in memory."""
    audio = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _require_key(env_name: str, provider: str) -> str:
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(f"{provider} STT requires the {env_name} environment variable")
    return key


class OpenAICompatSTT:
    """Multipart /audio/transcriptions client shared by Groq and OpenAI."""

    def __init__(
        self,
        *,
        provider: str,
        url: str,
        model: str,
        api_key_env: str,
        language: str = "",
        hotwords: list[str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider = provider
        self.url = url
        self.model = model
        self.api_key_env = api_key_env
        self.language = language.strip()
        self.hotwords = list(hotwords or [])
        self._api_key = _require_key(api_key_env, provider)
        self._client = client or httpx.Client(timeout=TIMEOUT_S)

    def transcribe(self, pcm: np.ndarray) -> str:
        audio = np.asarray(pcm, dtype=np.float32)
        if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-5:
            return ""
        wav = pcm_to_wav_bytes(audio)
        data: dict[str, Any] = {"model": self.model, "response_format": "json"}
        if self.language:
            data["language"] = self.language
        if self.hotwords:
            data["prompt"] = ", ".join(self.hotwords[:40])
        resp = self._client.post(
            self.url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            data=data,
            files={"file": ("audio.wav", wav, "audio/wav")},
        )
        if resp.status_code >= 400:
            detail = resp.text.strip()[:200]
            raise RuntimeError(f"{self.provider} STT failed ({resp.status_code}): {detail}")
        payload = resp.json()
        return str(payload.get("text") or "").strip()


class GroqSTT(OpenAICompatSTT):
    def __init__(self, cfg: SttConfig, client: httpx.Client | None = None) -> None:
        super().__init__(
            provider="Groq",
            url="https://api.groq.com/openai/v1/audio/transcriptions",
            model=cfg.groq_model,
            api_key_env=cfg.groq_api_key_env,
            language=cfg.language,
            hotwords=cfg.hotwords,
            client=client,
        )


class OpenAISTT(OpenAICompatSTT):
    def __init__(self, cfg: SttConfig, client: httpx.Client | None = None) -> None:
        base = cfg.openai_base_url.rstrip("/")
        super().__init__(
            provider="OpenAI",
            url=f"{base}/audio/transcriptions",
            model=cfg.openai_model,
            api_key_env=cfg.openai_api_key_env,
            language=cfg.language,
            hotwords=cfg.hotwords,
            client=client,
        )


class DeepgramSTT:
    """Deepgram Nova-3 pre-recorded REST client with smart_format and keyterms."""

    URL = "https://api.deepgram.com/v1/listen"

    def __init__(self, cfg: SttConfig, client: httpx.Client | None = None) -> None:
        self.model = cfg.deepgram_model
        self.language = cfg.language.strip()
        self.hotwords = list(cfg.hotwords or [])
        self._api_key = _require_key(cfg.deepgram_api_key_env, "Deepgram")
        self._client = client or httpx.Client(timeout=TIMEOUT_S)

    def transcribe(self, pcm: np.ndarray) -> str:
        audio = np.asarray(pcm, dtype=np.float32)
        if audio.size == 0 or float(np.max(np.abs(audio))) < 1e-5:
            return ""
        wav = pcm_to_wav_bytes(audio)
        params: list[tuple[str, str]] = [
            ("model", self.model),
            ("smart_format", "true"),
            ("punctuate", "true"),
        ]
        if self.language:
            params.append(("language", self.language))
        else:
            params.append(("detect_language", "true"))
        for word in self.hotwords[:30]:
            w = word.strip()
            if w:
                params.append(("keyterm", w))
        resp = self._client.post(
            self.URL,
            params=params,
            headers={
                "Authorization": f"Token {self._api_key}",
                "Content-Type": "audio/wav",
            },
            content=wav,
        )
        if resp.status_code >= 400:
            detail = resp.text.strip()[:200]
            raise RuntimeError(f"Deepgram STT failed ({resp.status_code}): {detail}")
        payload = resp.json()
        try:
            alt = payload["results"]["channels"][0]["alternatives"][0]
            return str(alt.get("transcript") or "").strip()
        except (KeyError, IndexError, TypeError):
            return ""
