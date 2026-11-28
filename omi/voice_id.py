"""Only my voice: Omi answers the people who saved their voice and ignores everyone else.

A voiceprint is 192 numbers that describe how someone sounds, made by NVIDIA's TitaNet
small model (40 MB, run by sherpa-onnx on the CPU, ~40 ms a check). It is saved in
~/.local/share/omi/voiceprints.json, readable only by you, never leaves the computer,
and is deleted when "only my voice" is turned off. No recording is kept.

A finished request is compared with every saved voice; when none is close enough it is
dropped. Requests too short to judge ("stop", "yes") are always let through, so you are
never ignored when it matters most.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from omi.config import DATA_DIR, MODELS_DIR

log = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "speaker" / "nemo_en_titanet_small.onnx"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
             "nemo_en_titanet_small.onnx")
MODEL_SHA256 = "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"
PRINTS_PATH = DATA_DIR / "voiceprints.json"

RATE = 16000
FRAME = 512  # 32 ms
MIN_CHECK_S = 1.2  # less speech than this is too short to judge: let it through
MIN_ENROLL_S = 3.0
ENROLL_CHUNK_S = 1.5
# Match bar: how alike a request must sound to a saved voice. Set per voice from how
# alike that person's own enrollment pieces were, within these bounds. Different
# speakers scored 0.31 on average (0.78 at worst, two similar synthetic voices).
MIN_MATCH, MAX_MATCH, MATCH_MARGIN = 0.4, 0.6, 0.15


def ensure_model() -> Path:
    from omi.look import _fetch

    return _fetch(MODEL_PATH, MODEL_URL, MODEL_SHA256, "voice model")


def speech_only(pcm: np.ndarray) -> np.ndarray:
    """Drop the quiet frames so pauses do not water down the voiceprint."""
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    n = len(pcm) // FRAME
    if n == 0:
        return pcm[:0]
    frames = pcm[: n * FRAME].reshape(n, FRAME)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    loud = rms >= max(0.004, 0.25 * float(np.percentile(rms, 95)))
    return frames[loud].reshape(-1)


class VoiceId:
    """Turns speech into a voiceprint. Loads the model on first use."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or MODEL_PATH
        self._extractor = None
        self._lock = threading.Lock()

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        import sherpa_onnx as so

        with self._lock:
            if self._extractor is None:
                self._extractor = so.SpeakerEmbeddingExtractor(
                    so.SpeakerEmbeddingExtractorConfig(model=str(self.model_path), num_threads=1))
            stream = self._extractor.create_stream()
            stream.accept_waveform(RATE, np.asarray(pcm, dtype=np.float32))
            stream.input_finished()
            vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return vec / (np.linalg.norm(vec) or 1.0)


def enroll(pcm: np.ndarray, voice: VoiceId) -> dict:
    """A voiceprint from a few seconds of someone reading aloud."""
    speech = speech_only(pcm)
    if len(speech) < MIN_ENROLL_S * RATE:
        raise ValueError("not enough speech to learn the voice")
    size = int(ENROLL_CHUNK_S * RATE)
    pieces = [speech[i:i + size] for i in range(0, len(speech) - size + 1, size)]
    vecs = [voice.embed(p) for p in pieces]
    mean = np.mean(vecs, axis=0)
    mean /= np.linalg.norm(mean) or 1.0
    own = min(float(v @ mean) for v in vecs)  # how alike this person's own pieces were
    threshold = min(MAX_MATCH, max(MIN_MATCH, own - MATCH_MARGIN))
    return {"vector": [round(float(x), 6) for x in mean], "threshold": round(threshold, 3),
            "saved": time.strftime("%Y-%m-%d")}


class Voiceprints:
    """The saved voices, by name, in a file only you can read."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or PRINTS_PATH
        self._cache: tuple[float, dict] | None = None

    def load(self) -> dict[str, dict]:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            return {}
        if self._cache is None or self._cache[0] != mtime:
            try:
                self._cache = (mtime, json.loads(self.path.read_text()))
            except (OSError, ValueError):
                return {}
        return self._cache[1]

    def names(self) -> list[str]:
        return sorted(self.load())

    def save(self, name: str, print_: dict) -> None:
        prints = dict(self.load())
        prints[name] = print_
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(prints, fh)
        tmp.replace(self.path)
        self._cache = None

    def forget(self, name: str | None = None) -> bool:
        """Delete one voice, or every voice when no name is given."""
        prints = dict(self.load())
        if name is None or set(prints) <= {name}:
            existed = self.path.exists()
            self.path.unlink(missing_ok=True)
            self._cache = None
            return existed
        if name not in prints:
            return False
        del prints[name]
        self.save_all(prints)
        return True

    def save_all(self, prints: dict) -> None:
        self.path.unlink(missing_ok=True)
        self._cache = None
        for k, v in prints.items():
            self.save(k, v)


class VoiceGate:
    """The listener's check: True (a saved voice), None (cannot tell, or the check is
    off), False (someone else)."""

    def __init__(self, enabled, prints: Voiceprints | None = None, voice: VoiceId | None = None) -> None:
        self._enabled = enabled  # callable, so a change in Settings applies at once
        self.prints = prints or Voiceprints()
        self.voice = voice or VoiceId()

    def __call__(self, pcm: np.ndarray) -> bool | None:
        if not self._enabled():
            return None
        prints = self.prints.load()
        if not prints:
            return None
        speech = speech_only(pcm)
        if len(speech) < MIN_CHECK_S * RATE:
            return None
        vec = self.voice.embed(speech)
        best = max(float(vec @ np.asarray(p["vector"], np.float32)) - float(p["threshold"])
                   for p in prints.values())
        log.debug("voice check margin %.3f", best)
        return best >= 0
