"""Always-on local wake word detector ("Hey Omi") using sherpa-onnx KeywordSpotter
and Silero VAD for hands-free end-of-turn capture.

Runs completely locally on CPU (~1.7% of one core) using the 3.3M GigaSpeech
streaming Zipformer KWS model. Supports any custom English wake phrase via BPE
tokenization against the model's `tokens.txt`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic
from typing import Any, Protocol

import numpy as np
import sherpa_onnx as so

from hark.audio.base import AudioIn
from hark.audio.pipewire import PipeWireIn, PipeWireOut
from hark.audio.vad import EndpointDetector
from hark.config import WakeConfig
from hark.events import State
from hark.stt.base import SAMPLE_RATE

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2"
)
MODEL_FILES = (
    "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "tokens.txt",
)

# Phonetic spelling helpers for Omi so varied accents trigger reliably without false positives.
_PHONETIC_ALIASES: dict[str, tuple[str, ...]] = {
    "hey omi": ("hey omi", "hey oh me", "hey omy"),
    "okay omi": ("okay omi", "ok omi", "okay oh me"),
    "ok omi": ("ok omi", "okay omi"),
}

POST_SPEECH_COOLDOWN_S = 0.7


class Detector(Protocol):
    def feed(self, pcm: np.ndarray) -> str: ...
    def reset(self) -> None: ...


class Endpoint(Protocol):
    speaking: bool
    ended: bool
    def feed(self, chunk: np.ndarray) -> None: ...
    def reset(self) -> None: ...


def ensure_model(model_dir: str | Path) -> Path:
    """Ensure the GigaSpeech 3.3M KWS model exists locally, downloading on first use."""
    dest = Path(model_dir)
    if all((dest / f).is_file() for f in MODEL_FILES):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="omi-kws-", dir=str(dest.parent)))
    archive = tmp_dir / "kws.tar.bz2"
    try:
        log.info("downloading wake word model to %s", dest)
        with urllib.request.urlopen(MODEL_URL, timeout=120) as src, archive.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        with tarfile.open(archive, "r:bz2") as tf:
            members = tf.getmembers()
            for m in members:
                parts = Path(m.name).parts
                if len(parts) < 2:
                    continue
                rel = Path(*parts[1:])
                if rel.is_absolute() or ".." in rel.parts:
                    continue
                target = tmp_dir / "extracted" / rel
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif m.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(m) as fh, target.open("wb") as out:  # type: ignore[union-attr]
                        shutil.copyfileobj(fh, out)
        extracted = tmp_dir / "extracted"
        if dest.exists():
            shutil.rmtree(dest)
        extracted.replace(dest)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return dest


def load_vocab(tokens_path: Path) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        piece, _, idx = line.rpartition(" ")
        if piece and idx.isdigit():
            vocab[piece] = int(idx)
    return vocab


def encode_word(word: str, vocab: dict[str, int]) -> list[str]:
    """Encode an ASCII word into GigaSpeech BPE pieces using DP (fewest pieces, lowest ids)."""
    clean = re.sub(r"[^A-Za-z']", "", word).upper()
    if not clean:
        return []
    s = "▁" + clean
    n = len(s)
    inf = 10**9
    dp: list[tuple[int, int, list[str]]] = [(inf, inf, [])] * (n + 1)
    dp[0] = (0, 0, [])
    for i in range(n):
        if dp[i][0] >= inf:
            continue
        for j in range(i + 1, n + 1):
            sub = s[i:j]
            if sub in vocab:
                cand = (dp[i][0] + 1, dp[i][1] + vocab[sub], [*dp[i][2], sub])
                if cand[:2] < dp[j][:2]:
                    dp[j] = cand
    if dp[n][0] >= inf:
        raise ValueError(f"cannot encode wake word token {word!r} in model vocabulary")
    return dp[n][2]


def encode_phrase(phrase: str, vocab: dict[str, int], label: str = "") -> str:
    pieces: list[str] = []
    for word in phrase.strip().split():
        pieces.extend(encode_word(word, vocab))
    if not pieces:
        raise ValueError(f"empty wake phrase: {phrase!r}")
    line = " ".join(pieces)
    if label:
        line += f" @{label}"
    return line


def build_keywords_text(phrases: list[str] | tuple[str, ...], tokens_path: Path) -> str:
    vocab = load_vocab(tokens_path)
    lines: list[str] = []
    seen: set[str] = set()
    for raw in phrases:
        norm = " ".join(raw.lower().split())
        if not norm:
            continue
        label = re.sub(r"[^a-z0-9]+", "_", norm).strip("_") or "wake"
        variants = _PHONETIC_ALIASES.get(norm, (norm,))
        for v in variants:
            encoded = encode_phrase(v, vocab, label)
            if encoded not in seen:
                seen.add(encoded)
                lines.append(encoded)
    if not lines:
        raise ValueError("at least one wake phrase is required")
    return "\n".join(lines) + "\n"


def wake_chime_pcm(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Gentle 95 ms rising two-tone chime (660 Hz -> 880 Hz) confirming Omi woke up."""
    dur1, dur2 = 0.045, 0.050
    t1 = np.arange(int(sample_rate * dur1), dtype=np.float32) / sample_rate
    t2 = np.arange(int(sample_rate * dur2), dtype=np.float32) / sample_rate
    env1 = np.hanning(len(t1)).astype(np.float32)
    env2 = np.hanning(len(t2)).astype(np.float32)
    tone1 = 0.11 * env1 * np.sin(2.0 * np.pi * 660.0 * t1)
    tone2 = 0.11 * env2 * np.sin(2.0 * np.pi * 880.0 * t2)
    return np.concatenate([tone1, tone2]).astype(np.float32)


class WakeDetector:
    """Streaming sherpa-onnx KeywordSpotter wrapper for arbitrary English wake phrases."""

    def __init__(self, cfg: WakeConfig) -> None:
        model_dir = ensure_model(cfg.model_dir)
        tokens_path = model_dir / "tokens.txt"
        kw_text = build_keywords_text(cfg.phrases, tokens_path)
        folder = Path(tempfile.mkdtemp(prefix="omi-kws-cfg-"))
        kw_file = folder / "keywords.txt"
        kw_file.write_text(kw_text, encoding="utf-8")
        try:
            self._kws = so.KeywordSpotter(
                tokens=str(tokens_path),
                encoder=str(model_dir / MODEL_FILES[0]),
                decoder=str(model_dir / MODEL_FILES[1]),
                joiner=str(model_dir / MODEL_FILES[2]),
                keywords_file=str(kw_file),
                num_threads=1,
                keywords_score=cfg.score,
                keywords_threshold=cfg.threshold,
            )
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        self._stream = self._kws.create_stream()

    def reset(self) -> None:
        self._kws.reset_stream(self._stream)

    def feed(self, pcm: np.ndarray) -> str:
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if not len(samples):
            return ""
        self._stream.accept_waveform(SAMPLE_RATE, samples)
        hit = ""
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)
            res = self._kws.get_result(self._stream)
            if res:
                hit = res
                self._kws.reset_stream(self._stream)
        return hit


class WakeListener:
    """Background continuous listener: spots the wake word, captures the spoken
    command until Silero VAD detects end-of-turn, and hands it to Session."""

    def __init__(
        self,
        session: Any,
        cfg: WakeConfig,
        *,
        audio_source: str = "",
        audio_sink: str = "",
        is_ptt_active: Callable[[], bool] = lambda: False,
        broadcast: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        audio_in_factory: Callable[[], AudioIn] | None = None,
        detector: Detector | None = None,
        vad_factory: Callable[[], Endpoint] | None = None,
        chime_fn: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.session = session
        self.cfg = cfg
        self.audio_source = audio_source
        self.audio_sink = audio_sink
        self._is_ptt_active = is_ptt_active
        self._broadcast = broadcast
        self._audio_in_factory = audio_in_factory or (lambda: PipeWireIn(audio_source, keep_chunks=False))
        self._detector = detector
        self._vad_factory = vad_factory or (lambda: EndpointDetector(silence_ms=cfg.silence_ms))
        self._chime_fn = chime_fn
        self._vad: Endpoint | None = None
        self._audio: AudioIn | None = None
        self._task: asyncio.Task[None] | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._running = False
        self.error: str | None = None

    @property
    def active(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.active:
            return
        self.error = None
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="omi-wake")
        loop = asyncio.get_running_loop()
        if self._detector is None:
            self._detector = await loop.run_in_executor(self._pool, WakeDetector, self.cfg)
        if self._vad is None:
            self._vad = await loop.run_in_executor(self._pool, self._vad_factory)
        self._running = True
        self._task = asyncio.create_task(self._run(), name="omi-wake-loop")

    async def stop(self) -> None:
        self._running = False
        if self._audio is not None:
            with contextlib.suppress(Exception):
                await self._audio.stop()
            self._audio = None
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    async def _play_chime(self) -> None:
        if not self.cfg.chime or self.audio_sink == "null":
            return
        if self._chime_fn is not None:
            with contextlib.suppress(Exception):
                await self._chime_fn()
            return
        out = PipeWireOut(self.audio_sink)
        try:
            await out.start(SAMPLE_RATE)
            await out.write(wake_chime_pcm(SAMPLE_RATE))
            await out.drain()
        except Exception:
            log.debug("wake chime failed", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                await out.stop()

    async def _run(self) -> None:
        assert self._detector is not None and self._vad is not None and self._pool is not None
        loop = asyncio.get_running_loop()
        while self._running:
            audio = self._audio_in_factory()
            self._audio = audio
            try:
                await audio.start()
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                log.warning("wake listener could not open mic: %s", exc)
                self._audio = None
                await asyncio.sleep(3.0)
                continue

            capturing = False
            speech_seen = False
            capture_started = 0.0
            cooldown_until = 0.0
            was_busy = False
            turn_task: asyncio.Task[Any] | None = None

            try:
                while self._running:
                    chunk = await audio.read()
                    if chunk is None:
                        break
                    now = monotonic()
                    busy = (
                        self._is_ptt_active()
                        or (turn_task is not None and not turn_task.done())
                        or self.session.state in (State.SPEAKING, State.THINKING, State.ACTING)
                    )
                    if capturing:
                        # Abort wake capture if PTT took over or the user pressed Stop.
                        if self._is_ptt_active() or self.session._utt is None:
                            capturing = False
                            await loop.run_in_executor(self._pool, self._detector.reset)
                            cooldown_until = now + 0.3
                            continue
                        self.session.feed_audio(chunk)
                        await loop.run_in_executor(self._pool, self._vad.feed, chunk)
                        if self._vad.speaking:
                            speech_seen = True
                        elapsed = now - capture_started
                        if self._vad.ended or elapsed >= self.cfg.max_utterance_s:
                            capturing = False
                            await loop.run_in_executor(self._pool, self._detector.reset)
                            if speech_seen:
                                turn_task = asyncio.create_task(self.session.end_utterance())
                                cooldown_until = now + POST_SPEECH_COOLDOWN_S
                            else:
                                await self.session.cancel_utterance()
                                cooldown_until = now + 0.3
                        elif not speech_seen and elapsed >= self.cfg.initial_timeout_s:
                            capturing = False
                            await self.session.cancel_utterance()
                            await loop.run_in_executor(self._pool, self._detector.reset)
                            cooldown_until = now + 0.3
                        continue

                    if busy:
                        was_busy = True
                        continue
                    if was_busy:
                        was_busy = False
                        cooldown_until = max(cooldown_until, now + POST_SPEECH_COOLDOWN_S)
                        await loop.run_in_executor(self._pool, self._detector.reset)
                        continue
                    if now < cooldown_until:
                        continue

                    hit = await loop.run_in_executor(self._pool, self._detector.feed, chunk)
                    if hit:
                        log.info("wake word detected: %s", hit)
                        if self._broadcast is not None:
                            await self._broadcast({"event": "wake", "phrase": hit})
                        asyncio.create_task(self._play_chime())
                        await self.session.begin_utterance()
                        await loop.run_in_executor(self._pool, self._vad.reset)
                        capturing = True
                        speech_seen = False
                        capture_started = monotonic()
            except asyncio.CancelledError:
                if capturing:
                    with contextlib.suppress(Exception):
                        await self.session.cancel_utterance()
                raise
            except Exception:
                log.exception("wake listener loop error")
                await asyncio.sleep(1.0)
            finally:
                with contextlib.suppress(Exception):
                    await audio.stop()
                if self._audio is audio:
                    self._audio = None
            if self._running:
                await asyncio.sleep(1.0)
