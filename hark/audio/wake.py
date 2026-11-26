"""Always-on local wake word detector ("Hey Omi") with continuous phone-call
conversation mode using sherpa-onnx KeywordSpotter and Silero VAD.

Runs completely locally on CPU (~1.7% of one core) using the 3.3M GigaSpeech
streaming Zipformer KWS model. Once woken, Omi stays in conversational mode —
automatically listening again after every response until you say a farewell
("goodbye", "see ya", "that's all", "stop listening", ...) or stay silent for
`followup_timeout_s` seconds.
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

from hark.audio import sfx
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

# Phonetic spelling helpers for Omi so varied accents and softer speech trigger reliably
# while keeping false positives near zero (benchmarked 31/32 TP, 0/24 FP at thresh=0.15).
_PHONETIC_ALIASES: dict[str, tuple[str, ...]] = {
    "omi": ("omi", "omie", "omee", "ohmi", "omy"),
    "hey omi": ("hey omi", "hey oh me", "hey omy", "hey omie", "hey omee", "hey ohmi"),
    "okay omi": ("okay omi", "ok omi", "okay oh me", "okay omie"),
    "ok omi": ("ok omi", "okay omi", "okay oh me", "okay omie"),
}

POST_SPEECH_COOLDOWN_S = 0.55
CONFIRM_REOPEN_DELAY_S = 1.2
QUIET_RMS_THRESHOLD = 0.008
QUIET_RESET_S = 2.5

_GOODBYE_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|alright|all right|thanks?|thank you)(?:\s+omi)?\s+)*"
    r"(?:good\s*bye|bye(?:\s+bye)?|see\s+(?:ya|you)(?:\s+later)?|"
    r"talk\s+(?:to\s+you\s+)?later|catch\s+you\s+later|"
    r"that(?:'s| is)\s+(?:all|it)(?:\s+for\s+now)?|"
    r"(?:i(?:'m| am)|we(?:'re| are))\s+(?:done|good|all\s+set)|"
    r"all\s+(?:done|good)|nothing\s+else|"
    r"end\s+(?:the\s+)?call|hang\s+up|stop\s+listening|go\s+to\s+sleep|dismiss|"
    r"stop|stop\s+talking|never\s*mind|cancel|be\s+quiet|shut\s+up)"
    r"(?:\s+(?:omi|please|thanks|thank\s+you|now|for\s+now))*$",
    re.I,
)


_KEEP_TALKING_RE = re.compile(
    r"\b(?:let's\s+(?:talk|chat)|talk\s+to\s+me|stay\s+on\s+the\s+line|keep\s+listening|"
    r"chat\s+with\s+me|brainstorm)\b",
    re.I,
)


def is_goodbye(text: str) -> bool:
    """True when a short spoken utterance ends an active hands-free conversation."""
    clean = re.sub(r"[^a-zA-Z0-9'\s]+", " ", text.replace("’", "'")).strip()
    clean = re.sub(r"\s+", " ", clean)
    if not clean or len(clean.split()) > 7:
        return False
    return bool(_GOODBYE_RE.fullmatch(clean))


def effective_call_mode(cfg: WakeConfig) -> str:
    """Resolved call mode: 'auto' (smart), 'continuous', or 'single'."""
    if not cfg.conversational:
        return "single"
    mode = (cfg.call_mode or "auto").strip().lower()
    return mode if mode in ("auto", "continuous", "single") else "auto"


def should_continue_conversation(
    mode: str,
    *,
    user_text: str,
    reply: str = "",
    path: str | None = None,
    action_ms: float | None = None,
    cancelled: bool = False,
) -> bool:
    """Decide whether Omi should keep the mic open after finishing a turn.

    - 'single': never continue.
    - 'continuous': always continue unless cancelled, empty, or user said goodbye.
    - 'auto' (smart): drop back to standby after one-shot OS/tool commands with short
      confirmations; keep listening for chats, explanations, troubleshooting, or
      whenever Omi asked a follow-up question.
    """
    if cancelled or not user_text.strip() or is_goodbye(user_text):
        return False
    if mode == "single":
        return False
    if mode == "continuous":
        return True
    # Smart ('auto') mode:
    if _KEEP_TALKING_RE.search(user_text):
        return True
    if path == "router":
        return False
    clean_reply = reply.strip()
    if "?" in clean_reply:
        return True
    # A tool ran and Omi gave a short closed confirmation (e.g. "Switched to Gruvbox."):
    # treat as a finished one-shot command rather than an open conversation.
    if action_ms is not None and len(clean_reply.split()) <= 14:
        return False
    return True


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
            for m in tf.getmembers():
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
    """Gentle bubbly rising bell chime confirming Omi woke up."""
    return sfx.render("wake", sample_rate)


def sleep_chime_pcm(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Gentle descending bell chime when call mode drops back to standby."""
    return sfx.render("sleep", sample_rate)


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
    """Background continuous listener: spots the wake word, enters continuous phone-call
    conversation mode (keeping the mic open for follow-up turns after each reply), and
    drops back to wake-word standby on a farewell phrase or prolonged silence."""

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
        sleep_chime_fn: Callable[[], Awaitable[None]] | None = None,
        play_sfx: Callable[[str], None] | None = None,
        post_speech_cooldown_s: float = POST_SPEECH_COOLDOWN_S,
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
        self._sleep_chime_fn = sleep_chime_fn
        self._play_sfx = play_sfx
        self._post_speech_cooldown_s = post_speech_cooldown_s
        self._vad: Endpoint | None = None
        self._audio: AudioIn | None = None
        self._task: asyncio.Task[None] | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._running = False
        self.conversing = False
        self.error: str | None = None

    @property
    def active(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    async def reload_detector(self) -> None:
        """Rebuild the KWS detector live after a sensitivity/threshold change."""
        if not self.active or self._pool is None:
            return
        loop = asyncio.get_running_loop()
        self._detector = await loop.run_in_executor(self._pool, WakeDetector, self.cfg)

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
        self.conversing = False
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

    async def _play_tone(
        self,
        cue: str,
        pcm_fn: Callable[[int], np.ndarray],
        hook: Callable[[], Awaitable[None]] | None,
    ) -> None:
        if not self.cfg.chime or self.audio_sink == "null":
            return
        if hook is not None:
            with contextlib.suppress(Exception):
                await hook()
            return
        if self._play_sfx is not None:
            self._play_sfx(cue)
            return
        out = PipeWireOut(self.audio_sink)
        try:
            await out.start(SAMPLE_RATE)
            await out.write(pcm_fn(SAMPLE_RATE))
            await out.drain()
        except Exception:
            log.debug("wake chime failed", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                await out.stop()

    async def _end_conversation(self, *, chime: bool = False) -> None:
        was = self.conversing
        self.conversing = False
        if chime:
            asyncio.create_task(self._play_tone("sleep", sleep_chime_pcm, self._sleep_chime_fn))
        if was and self._broadcast is not None:
            await self._broadcast({"event": "wake", "conversing": False})

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
            active_timeout = self.cfg.initial_timeout_s
            cooldown_until = 0.0
            reopen_at: float | None = None
            confirm_reopen_at: float | None = None
            was_busy = False
            audio_seen_since_reset = False
            quiet_since: float | None = None
            turn_task: asyncio.Task[Any] | None = None

            async def start_capture(timeout_s: float) -> None:
                nonlocal capturing, speech_seen, capture_started, active_timeout, reopen_at, confirm_reopen_at
                reopen_at = None
                confirm_reopen_at = None
                await self.session.begin_utterance()
                await loop.run_in_executor(self._pool, self._vad.reset)
                capturing = True
                speech_seen = False
                capture_started = monotonic()
                active_timeout = timeout_s

            try:
                while self._running:
                    chunk = await audio.read()
                    if chunk is None:
                        break
                    now = monotonic()

                    # Collect completed turn result to decide whether to continue the call.
                    if turn_task is not None and turn_task.done():
                        res = None
                        with contextlib.suppress(Exception):
                            res = turn_task.result()
                        turn_task = None
                        if self.conversing:
                            user_text = getattr(res, "user_text", "") if res is not None else ""
                            reply = getattr(res, "reply", "") if res is not None else ""
                            timing = getattr(res, "timing", None) or {} if res is not None else {}
                            cancelled = bool(getattr(res, "cancelled", False)) if res is not None else False
                            mode = effective_call_mode(self.cfg)
                            keep = should_continue_conversation(
                                mode,
                                user_text=user_text,
                                reply=reply,
                                path=timing.get("path"),
                                action_ms=timing.get("action_ms"),
                                cancelled=cancelled,
                            )
                            if not keep:
                                await self._end_conversation(chime=False)
                                cooldown_until = now + self._post_speech_cooldown_s
                                await loop.run_in_executor(self._pool, self._detector.reset)
                            else:
                                reopen_at = now + self._post_speech_cooldown_s

                    # If user took over with PTT, drop continuous call mode cleanly.
                    if self._is_ptt_active():
                        if capturing:
                            capturing = False
                        if self.conversing:
                            await self._end_conversation(chime=False)
                        reopen_at = None
                        confirm_reopen_at = None
                        was_busy = True
                        continue

                    if capturing:
                        # User pressed Stop (or another turn cancelled the open utterance).
                        if self.session._utt is None:
                            capturing = False
                            await self._end_conversation(chime=False)
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
                                cooldown_until = now + self._post_speech_cooldown_s
                            else:
                                await self.session.cancel_utterance()
                                await self._end_conversation(chime=self.conversing)
                                cooldown_until = now + 0.3
                        elif not speech_seen and elapsed >= active_timeout:
                            capturing = False
                            await self.session.cancel_utterance()
                            await self._end_conversation(chime=self.conversing)
                            await loop.run_in_executor(self._pool, self._detector.reset)
                            cooldown_until = now + 0.3
                        continue

                    # Hands-free voice confirmation during an active call: open mic once question is spoken.
                    if self.conversing and self.session.state is State.CONFIRMING and self.session._utt is None:
                        if confirm_reopen_at is None:
                            confirm_reopen_at = now + CONFIRM_REOPEN_DELAY_S
                        elif now >= confirm_reopen_at:
                            await start_capture(self.cfg.followup_timeout_s)
                        continue
                    confirm_reopen_at = None

                    busy = (
                        (turn_task is not None and not turn_task.done())
                        or self.session.state in (State.SPEAKING, State.THINKING, State.ACTING)
                    )
                    if busy:
                        was_busy = True
                        continue

                    # Active phone-call follow-up: reopen mic automatically once reply audio settles.
                    if self.conversing and reopen_at is not None:
                        if now >= reopen_at:
                            await start_capture(self.cfg.followup_timeout_s)
                        continue

                    if was_busy:
                        was_busy = False
                        cooldown_until = max(cooldown_until, now + self._post_speech_cooldown_s)
                        await loop.run_in_executor(self._pool, self._detector.reset)
                        audio_seen_since_reset = False
                        quiet_since = None
                        continue
                    if now < cooldown_until:
                        continue

                    # Periodically reset the streaming KWS decoder after quiet stretches so
                    # long-running background noise never degrades wake sensitivity.
                    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32)))) if len(chunk) else 0.0
                    if rms >= QUIET_RMS_THRESHOLD:
                        audio_seen_since_reset = True
                        quiet_since = None
                    elif audio_seen_since_reset:
                        if quiet_since is None:
                            quiet_since = now
                        elif now - quiet_since >= QUIET_RESET_S:
                            await loop.run_in_executor(self._pool, self._detector.reset)
                            audio_seen_since_reset = False
                            quiet_since = None

                    hit = await loop.run_in_executor(self._pool, self._detector.feed, chunk)
                    if hit:
                        log.info("wake word detected: %s", hit)
                        self.conversing = effective_call_mode(self.cfg) != "single"
                        if self._broadcast is not None:
                            await self._broadcast({"event": "wake", "phrase": hit, "conversing": self.conversing})
                        asyncio.create_task(self._play_tone("wake", wake_chime_pcm, self._chime_fn))
                        audio_seen_since_reset = False
                        quiet_since = None
                        await start_capture(self.cfg.initial_timeout_s)
            except asyncio.CancelledError:
                if capturing:
                    with contextlib.suppress(Exception):
                        await self.session.cancel_utterance()
                raise
            except Exception:
                log.exception("wake listener loop error")
                await asyncio.sleep(1.0)
            finally:
                self.conversing = False
                with contextlib.suppress(Exception):
                    await audio.stop()
                if self._audio is audio:
                    self._audio = None
            if self._running:
                await asyncio.sleep(1.0)
