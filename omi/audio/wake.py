"""Always-on local wake word detector ("Hey Omi") with continuous phone-call
conversation mode using sherpa-onnx KeywordSpotter and Silero VAD.

Runs completely locally on CPU (~1.7% of one core) using the 3.3M GigaSpeech
streaming Zipformer KWS model. Once woken, Omi stays in conversational mode,
automatically listening again after every response until you say a farewell
("goodbye", "see ya", "that's all", "stop listening", ...) or stay silent for
`followup_timeout_s` seconds.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import re
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Protocol

import numpy as np
import sherpa_onnx as so

from omi.audio import sfx
from omi.audio.base import AudioIn
from omi.audio.pipewire import PipeWireIn, PipeWireOut
from omi.audio.vad import EndpointDetector
from omi.config import WakeConfig
from omi.events import State
from omi.stt.base import SAMPLE_RATE

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
# Talking over Omi: this many 32 ms chunks of steady speech (after the VAD's own onset)
# interrupt it, about 0.3 s, long enough to skip a cough or an "mm-hm" click.
BARGE_IN_CHUNKS = 6
EARLY_MAX_S = 30.0  # audio heard while Omi starts up, kept for the listener to catch up on
PREROLL_CHUNKS = 16  # audio kept while Omi talks, so the first words of an interruption count
BARGE_IN_THRESHOLD = 0.6  # stricter than end-of-turn detection
# Talking while Omi speaks first lowers its voice, then decides what you meant:
OVERLAP_TAKEOVER_CHUNKS = 38  # ~1.2 s of your speech is a real interruption, whatever it says
OVERLAP_PAUSE_CHUNKS = 10  # ~0.3 s of quiet ends a short remark, which is then transcribed
OVERLAP_MAX_S = 4.0  # still undecided after this: Omi carries on at full volume
CONFIRM_REOPEN_DELAY_S = 1.2
QUIET_RMS_THRESHOLD = 0.008
QUIET_RESET_S = 2.5
# Look to talk: steady speech for this many chunks while you face the screen opens a turn.
LOOK_ONSET_CHUNKS = 6
# ~1.3 s, so the words said before the decision still count: the camera only opens once
# a voice is heard and takes about 0.3 s to its first reading.
LOOK_PREROLL_CHUNKS = 40
LOOK_KEEP_RATIO = 0.5  # face away for more of the request than this and it was not for Omi

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


# Short sounds of agreement said while someone else talks. Hyphens are dropped first, so
# "mm-hmm" is "mmhmm" and "uh-huh" is "uhhuh".
BACKCHANNEL_WORDS = frozenset({
    "mm", "mmm", "hm", "hmm", "mhm", "mmhm", "mmhmm", "uhhuh", "uh", "um", "ah", "oh", "aha",
    "yeah", "yes", "yep", "yup", "ya", "ok", "okay", "alright", "right", "sure", "cool", "nice",
    "great", "wow", "true", "totally", "exactly", "gotcha", "got", "it", "i", "see", "interesting",
})


def is_backchannel(text: str) -> bool:
    """True when something said over Omi only acknowledges it ("mm-hm", "yeah, right"),
    or is no words at all (a cough). Omi then keeps talking."""
    words = re.findall(r"[a-z]+", text.lower().replace("-", ""))
    return len(words) <= 4 and all(w in BACKCHANNEL_WORDS for w in words)


@dataclass
class _Overlap:
    """The user talking while Omi speaks, until it is clear whether they mean to interrupt."""

    started: float
    chunks: list[np.ndarray]
    speech: int  # chunks of speech heard so far
    quiet: int = 0  # chunks of quiet since the last speech
    checked: int = 0  # `speech` when the last transcription began
    check: asyncio.Task[str] | None = field(default=None, repr=False)


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


class Attention(Protocol):
    """Whether you are facing the screen (omi.look.LookTracker)."""
    error: str | None
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def looking(self, now: float | None = None) -> bool: ...
    def addressed(self, now: float | None = None) -> bool: ...
    def wake(self) -> None: ...
    def facing_ratio(self, since: float, until: float | None = None) -> float | None: ...


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


class Backlog:
    """A mic opened before the listener is ready. It keeps the newest EARLY_MAX_S of
    audio so the listener can go through what was said during start-up, then carries on
    with live audio from the same stream, so nothing falls between the two."""

    def __init__(self, audio: AudioIn, max_s: float = EARLY_MAX_S) -> None:
        self.audio = audio
        self.max_samples = int(max_s * SAMPLE_RATE)
        self.samples = 0
        self._chunks: collections.deque[np.ndarray] = collections.deque()
        self._more = asyncio.Event()
        self._ended = False
        self._task: asyncio.Task[None] | None = None

    @property
    def error(self) -> str | None:
        return getattr(self.audio, "error", None)

    @property
    def seconds(self) -> float:
        return self.samples / SAMPLE_RATE

    async def start(self) -> None:
        if self._task is None:  # the listener calls start() too; the mic is already open
            await self.audio.start()
            self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        try:
            while (chunk := await self.audio.read()) is not None:
                self._chunks.append(chunk)
                self.samples += len(chunk)
                while self.samples > self.max_samples and len(self._chunks) > 1:
                    self.samples -= len(self._chunks.popleft())
                self._more.set()
        finally:
            self._ended = True
            self._more.set()

    async def read(self) -> np.ndarray | None:
        while not self._chunks:
            if self._ended:
                return None
            self._more.clear()
            await self._more.wait()
        chunk = self._chunks.popleft()
        self.samples -= len(chunk)
        return chunk

    async def stop(self) -> np.ndarray:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._chunks.clear()
        self.samples = 0
        return await self.audio.stop()


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
        barge_in: bool = False,
        barge_vad_factory: Callable[[], Endpoint] | None = None,
        attention_factory: Callable[[], Attention] | None = None,
        look_vad_factory: Callable[[], Endpoint] | None = None,
        voice_gate: Callable[[np.ndarray], bool | None] | None = None,
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
        # Barge-in: while Omi thinks or talks, listen for the user talking over it. Only
        # safe when the mic does not hear Omi itself (echo cancellation or headphones).
        self.barge_in = barge_in
        self._barge_vad_factory = barge_vad_factory or (
            lambda: EndpointDetector(silence_ms=cfg.silence_ms, threshold=BARGE_IN_THRESHOLD))
        self._barge_vad: Endpoint | None = None
        # Look to talk: the webcam decides whether speech is meant for Omi, no wake word.
        self._attention_factory = attention_factory or self._default_attention
        self._look_vad_factory = look_vad_factory or (
            lambda: EndpointDetector(silence_ms=cfg.silence_ms, threshold=BARGE_IN_THRESHOLD))
        self._look: Attention | None = None
        self._look_vad: Endpoint | None = None
        # "Only my voice": True/None lets a request through, False drops it (someone else).
        self._voice_gate = voice_gate
        self._look_failed: str | None = None  # why look to talk could not start
        self._control = asyncio.Lock()  # start, stop and look toggles can arrive together
        self._vad: Endpoint | None = None
        self._audio: AudioIn | None = None
        self._task: asyncio.Task[None] | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._running = False
        self.conversing = False
        # The mic is open for a request but nobody has started talking yet (a call
        # waiting for the follow-up): a restart here loses nothing.
        self.waiting_for_speech = False
        self._resume_call = False
        self._early: Backlog | None = None
        self.error: str | None = None

    @property
    def active(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    @property
    def looking_active(self) -> bool:
        return self._look is not None

    @property
    def look_paused_by(self) -> str:
        """The call or dictation app holding the mic while look to talk rests, or ""."""
        return str(getattr(self._look, "paused_by", "") or "") if self._look is not None else ""

    @property
    def look_error(self) -> str | None:
        if self._look is None:
            return self._look_failed
        return self._look.error

    def _default_attention(self) -> Attention:
        from omi.look import LookTracker

        return LookTracker(self.cfg.camera, lip_check=self.cfg.lip_check, lip_threshold=self.cfg.lip_threshold,
                           yaw_center=self.cfg.yaw_center, yaw_range=self.cfg.yaw_range)

    async def sync_look(self) -> None:
        """Turn the webcam on or off to match `look_to_talk` while the listener runs."""
        async with self._control:
            await self._sync_look()

    async def _sync_look(self) -> None:
        loop = asyncio.get_running_loop()
        want = bool(self.cfg.look_to_talk) and self._running and self._pool is not None
        if not want:
            self._look_failed = None
        if want and self._look is None:
            assert self._pool is not None
            if self._look_vad is None:
                self._look_vad = await loop.run_in_executor(self._pool, self._look_vad_factory)
            if self.barge_in and self._barge_vad is None:
                self._barge_vad = await loop.run_in_executor(self._pool, self._barge_vad_factory)
            look = self._attention_factory()
            try:
                await loop.run_in_executor(None, look.start)
            except Exception as exc:
                self._look_failed = str(exc) or type(exc).__name__
                raise
            self._look_failed = None
            self._look = look
        elif not want and self._look is not None:
            look, self._look = self._look, None
            await loop.run_in_executor(None, look.stop)

    async def reload_look(self) -> None:
        """Reopen the webcam after the chosen camera changed."""
        async with self._control:
            if self._look is not None:
                look, self._look = self._look, None
                await asyncio.get_running_loop().run_in_executor(None, look.stop)
            await self._sync_look()

    async def reload_detector(self) -> None:
        """Rebuild the KWS detector live after a sensitivity/threshold change."""
        if not self.active or self._pool is None:
            return
        loop = asyncio.get_running_loop()
        self._detector = await loop.run_in_executor(self._pool, WakeDetector, self.cfg)

    async def start(self) -> None:
        async with self._control:
            await self._start()

    async def _start(self) -> None:
        if self.active:
            await self._sync_look()
            return
        self.error = None
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="omi-wake")
        loop = asyncio.get_running_loop()
        if self._detector is None:
            self._detector = await loop.run_in_executor(self._pool, WakeDetector, self.cfg)
        if self._vad is None:
            self._vad = await loop.run_in_executor(self._pool, self._vad_factory)
        if self.barge_in and self._barge_vad is None:
            self._barge_vad = await loop.run_in_executor(self._pool, self._barge_vad_factory)
        self._running = True
        self._task = asyncio.create_task(self._run(), name="omi-wake-loop")
        try:
            await self._sync_look()
        except Exception as exc:
            # The camera side failing must never take the wake word down with it.
            log.warning("look to talk could not start, the wake word still works: %s", exc)

    async def stop(self) -> None:
        async with self._control:
            await self._stop()

    async def listen_early(self) -> None:
        """Open the mic now, while the models still load. What is said meanwhile is kept
        and the listener goes through it once it starts, so nothing is lost."""
        if self._early is not None or self.active:
            return
        early = Backlog(self._audio_in_factory())
        await early.start()
        self._early = early

    def resume_call(self) -> None:
        """Carry on the call a previous daemon had open: listen for the follow-up as soon
        as the mic opens, with no wake word."""
        self._resume_call = True

    async def _stop(self) -> None:
        self._running = False
        self.conversing = False
        self.waiting_for_speech = False
        if self._early is not None:
            early, self._early = self._early, None
            with contextlib.suppress(Exception):
                await early.stop()
        if self._look is not None:
            look, self._look = self._look, None
            with contextlib.suppress(Exception):
                await asyncio.get_running_loop().run_in_executor(None, look.stop)
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

    async def _voice_ok(self, heard: list[np.ndarray]) -> bool:
        """False only when "only my voice" is on and this was clearly someone else."""
        if self._voice_gate is None or not heard:
            return True
        pcm = np.concatenate(heard)
        try:
            verdict = await asyncio.get_running_loop().run_in_executor(self._pool, self._voice_gate, pcm)
        except Exception:
            log.warning("voice check failed; letting the request through", exc_info=True)
            return True
        return verdict is not False

    def _turned_away(self, since: float, until: float) -> bool:
        """True when the webcam saw you facing away for most of a look-opened request."""
        if self._look is None:
            return False
        ratio = self._look.facing_ratio(since, until)
        return ratio is not None and ratio < LOOK_KEEP_RATIO

    async def _follow_overlap(self, o: _Overlap, chunk: np.ndarray, now: float) -> bool | None:
        """One chunk heard while the user talks over Omi. True: they are interrupting.
        False: only an acknowledgement or a noise, Omi carries on. None: not clear yet."""
        if self.session.state is State.ACTING:
            return False  # never interrupt a tool that is running
        vad = self._barge_vad
        assert vad is not None and self._pool is not None
        loop = asyncio.get_running_loop()
        o.chunks.append(chunk)
        await loop.run_in_executor(self._pool, vad.feed, chunk)
        if vad.ended:
            await loop.run_in_executor(self._pool, vad.reset)
        if vad.speaking:
            o.speech += 1
            o.quiet = 0
        else:
            o.quiet += 1
        if o.speech >= OVERLAP_TAKEOVER_CHUNKS:
            return True
        if o.check is not None and o.check.done():
            check, o.check = o.check, None
            heard = ""
            if not check.cancelled() and check.exception() is None:
                heard = check.result()
            if not is_backchannel(heard):
                log.info("overlap: heard %r, Omi stops", heard)
                return True
            if o.speech == o.checked:
                return False
            # More was said after that clip: check the whole of it at the next pause.
        if now - o.started >= OVERLAP_MAX_S:
            return False
        if o.check is None and o.quiet >= OVERLAP_PAUSE_CHUNKS and o.speech > o.checked:
            o.checked = o.speech
            o.check = asyncio.create_task(self.session.transcribe_clip(np.concatenate(o.chunks)))
        return None

    @staticmethod
    def _drop_overlap(o: _Overlap) -> None:
        if o.check is not None:
            o.check.cancel()

    async def _run(self) -> None:
        assert self._detector is not None and self._vad is not None and self._pool is not None
        loop = asyncio.get_running_loop()
        while self._running:
            audio, self._early = self._early or self._audio_in_factory(), None
            self._audio = audio
            if isinstance(audio, Backlog) and audio.seconds >= 0.1:
                log.info("catching up on %.1f s heard while starting", audio.seconds)
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
            speech_started: float | None = None
            active_timeout = self.cfg.initial_timeout_s
            cooldown_until = 0.0
            reopen_at: float | None = None
            confirm_reopen_at: float | None = None
            was_busy = False
            audio_seen_since_reset = False
            quiet_since: float | None = None
            turn_task: asyncio.Task[Any] | None = None
            preroll: collections.deque[np.ndarray] = collections.deque(maxlen=PREROLL_CHUNKS)
            barge_run = 0
            overlap: _Overlap | None = None  # the user talking while Omi speaks, not yet decided
            look_preroll: collections.deque[np.ndarray] = collections.deque(maxlen=LOOK_PREROLL_CHUNKS)
            look_run = 0
            look_turn_since: float | None = None  # set while capturing a turn opened by looking
            heard: list[np.ndarray] = []  # this request's audio, for the "only my voice" check

            def feed(chunk: np.ndarray) -> None:
                self.session.feed_audio(chunk)
                heard.append(chunk)

            async def start_capture(timeout_s: float) -> None:
                nonlocal capturing, speech_seen, capture_started, speech_started, active_timeout, reopen_at, confirm_reopen_at
                nonlocal look_turn_since
                reopen_at = None
                confirm_reopen_at = None
                look_turn_since = None
                heard.clear()
                await self.session.begin_utterance()
                await loop.run_in_executor(self._pool, self._vad.reset)
                capturing = True
                speech_seen = False
                capture_started = monotonic()
                speech_started = None
                active_timeout = timeout_s

            async def take_over(buffered: list[np.ndarray]) -> None:
                """The user interrupts Omi: stop it and make what they said the next turn."""
                nonlocal turn_task, was_busy, barge_run, speech_seen, speech_started
                log.info("barge-in: the user talked over Omi")
                turn_task = None  # its result no longer decides anything
                was_busy = False
                barge_run = 0
                await start_capture(self.cfg.followup_timeout_s)
                for past in buffered:
                    feed(past)
                    await loop.run_in_executor(self._pool, self._vad.feed, past)
                speech_seen = bool(buffered)
                if speech_seen:
                    speech_started = monotonic()

            if self._resume_call:
                self._resume_call = False
                if effective_call_mode(self.cfg) != "single":
                    self.conversing = True
                    reopen_at = monotonic()
                    if self._broadcast is not None:
                        await self._broadcast({"event": "wake", "phrase": "resume", "conversing": True})

            try:
                while self._running:
                    self.waiting_for_speech = capturing and not speech_seen
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
                        if overlap is not None:
                            overlap = self._drop_overlap(overlap)
                            await self.session.duck(False)
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
                        feed(chunk)
                        await loop.run_in_executor(self._pool, self._vad.feed, chunk)
                        if self._vad.speaking and look_turn_since is not None and self._look is not None:
                            self._look.wake()  # keep watching for the turned-away check
                        if self._vad.speaking:
                            if not speech_seen:
                                speech_started = now
                            speech_seen = True
                            self.waiting_for_speech = False
                        elapsed = now - capture_started
                        speech_elapsed = (now - speech_started) if speech_started is not None else 0.0
                        if self._vad.ended or (speech_seen and speech_elapsed >= self.cfg.max_utterance_s):
                            capturing = False
                            await loop.run_in_executor(self._pool, self._detector.reset)
                            if speech_seen and look_turn_since is not None and self._turned_away(look_turn_since, now):
                                log.info("look to talk: you turned away, so that was not for Omi")
                                await self.session.cancel_utterance()
                                cooldown_until = now + 0.3
                            elif speech_seen and not await self._voice_ok(heard):
                                log.info("only my voice: that was someone else, so it was ignored")
                                await self.session.cancel_utterance()
                                cooldown_until = now + 0.3
                            elif speech_seen:
                                turn_task = asyncio.create_task(self.session.end_utterance())
                                cooldown_until = now + self._post_speech_cooldown_s
                            else:
                                await self.session.cancel_utterance()
                                await self._end_conversation(chime=self.conversing)
                                cooldown_until = now + 0.3
                        elif not speech_seen and active_timeout > 0 and elapsed >= active_timeout:
                            capturing = False
                            await self.session.cancel_utterance()
                            await self._end_conversation(chime=self.conversing)
                            await loop.run_in_executor(self._pool, self._detector.reset)
                            cooldown_until = now + 0.3
                        continue

                    if overlap is not None:
                        verdict = await self._follow_overlap(overlap, chunk, now)
                        if verdict is None:
                            continue
                        buffered, overlap = overlap.chunks, self._drop_overlap(overlap)
                        preroll.clear()
                        if verdict:
                            await take_over(buffered)
                            await self.session.duck(False)  # in case a yes/no question kept playing
                        else:
                            log.info("overlap: only an acknowledgement, Omi carries on")
                            barge_run = 0
                            await self.session.duck(False)
                            if self._barge_vad is not None:
                                await loop.run_in_executor(self._pool, self._barge_vad.reset)
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
                        if not was_busy:
                            preroll.clear()
                            barge_run = 0
                            if self._barge_vad is not None:
                                await loop.run_in_executor(self._pool, self._barge_vad.reset)
                        was_busy = True
                        # Never interrupt a tool that is running: only thinking or talking.
                        if not self.barge_in or self.session.state not in (State.SPEAKING, State.THINKING):
                            continue
                        preroll.append(chunk)
                        interrupt = False
                        voice = False  # sustained speech, as opposed to the wake word
                        if self.conversing and self._barge_vad is not None:
                            vad = self._barge_vad
                            await loop.run_in_executor(self._pool, vad.feed, chunk)
                            if vad.ended:
                                await loop.run_in_executor(self._pool, vad.reset)
                            barge_run = barge_run + 1 if vad.speaking else 0
                            interrupt = voice = barge_run >= BARGE_IN_CHUNKS
                        elif not self.conversing:
                            hit = ""
                            if self.cfg.enabled:
                                hit = await loop.run_in_executor(self._pool, self._detector.feed, chunk)
                            if hit:
                                log.info("wake word while busy: %s", hit)
                                self.conversing = effective_call_mode(self.cfg) != "single"
                                if self._broadcast is not None:
                                    await self._broadcast({"event": "wake", "phrase": hit, "conversing": self.conversing})
                                preroll.clear()  # the wake word itself is not part of the request
                                interrupt = True
                            elif self._look is not None and self._barge_vad is not None:
                                # Look to talk: talking over Omi while facing the screen.
                                vad = self._barge_vad
                                await loop.run_in_executor(self._pool, vad.feed, chunk)
                                if vad.ended:
                                    await loop.run_in_executor(self._pool, vad.reset)
                                barge_run = barge_run + 1 if vad.speaking else 0
                                if vad.speaking:
                                    self._look.wake()
                                interrupt = voice = barge_run >= BARGE_IN_CHUNKS and self._look.addressed(now)
                        if interrupt and voice and self.session.state is State.SPEAKING:
                            # Lower Omi's voice and listen before deciding: "mm-hm" lets it
                            # carry on, "wait" or a longer remark stops it.
                            log.info("overlap: the user talks while Omi speaks, lowering its voice")
                            overlap = _Overlap(now, list(preroll), speech=barge_run)
                            preroll.clear()
                            await self.session.duck(True)
                        elif interrupt:
                            buffered = list(preroll)
                            preroll.clear()
                            await take_over(buffered)
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
                        look_preroll.clear()
                        look_run = 0
                        if self._look_vad is not None:
                            await loop.run_in_executor(self._pool, self._look_vad.reset)
                        continue
                    if now < cooldown_until:
                        continue

                    # Look to talk: speech while you face the screen is for Omi.
                    look = self._look
                    if look is not None and self._look_vad is not None:
                        look_preroll.append(chunk)
                        vad = self._look_vad
                        await loop.run_in_executor(self._pool, vad.feed, chunk)
                        if vad.ended:
                            await loop.run_in_executor(self._pool, vad.reset)
                        look_run = look_run + 1 if vad.speaking else 0
                        if vad.speaking:
                            look.wake()  # the camera only runs while someone talks
                        if look_run >= LOOK_ONSET_CHUNKS and look.addressed(now):
                            log.info("look to talk: talking while facing the screen")
                            buffered = list(look_preroll)
                            look_preroll.clear()
                            look_run = 0
                            chunk_s = len(chunk) / SAMPLE_RATE
                            since = now - len(buffered) * chunk_s
                            await loop.run_in_executor(self._pool, vad.reset)
                            if self._broadcast is not None:
                                await self._broadcast({"event": "wake", "phrase": "look", "conversing": False})
                            await start_capture(self.cfg.initial_timeout_s)
                            look_turn_since = since
                            for past in buffered:
                                feed(past)
                                await loop.run_in_executor(self._pool, self._vad.feed, past)
                            speech_seen = True
                            speech_started = since
                            continue
                    if not self.cfg.enabled:
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
                self.waiting_for_speech = False
                if overlap is not None:
                    overlap = self._drop_overlap(overlap)
                    with contextlib.suppress(Exception):
                        await self.session.duck(False)
                with contextlib.suppress(Exception):
                    await audio.stop()
                if self._audio is audio:
                    self._audio = None
            if self._running:
                await asyncio.sleep(1.0)
