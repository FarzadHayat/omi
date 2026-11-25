"""Nemotron streaming transducer through sherpa-onnx. Implements StreamingSTT.

Names speech recognition has never seen (Omarchy, Danzeira) are boosted as hotwords.
sherpa-onnx needs a SentencePiece vocab to split them into the model's pieces; the
model ships only tokens.txt, so a vocab is written from it (a piece's score is minus
its id, which is the BPE merge order).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import sherpa_onnx as so

from hark.config import SttConfig
from hark.stt.base import SAMPLE_RATE

# Silence appended at the end so the last words leave the look-ahead. Each 160 ms of it
# costs about 70 ms after speech. On 96 test commands 0.32 s dropped a last word now and
# then ("open ghost", "workspace" without "two"); 0.48 s got every number and two more
# app names; 0.64 s gained one more name.
TAIL_S = 0.48
# Measured on 96 spoken commands (Piper and two Kokoro voices): these words went from
# 3 of 21 heard right to 11, word errors from 15.6% to 13.0%, commands the router
# caught from 58 to 60 of 63, at about 6 ms more after speech. Adding the 114 installed
# app names lowered word errors further but routed fewer (58); score 3.0 broke app names.
HOTWORDS = ("Omarchy", "Hyprland", "Danzeira", "Superwhisper")
HOTWORDS_SCORE = 2.0


def write_hotwords(folder: Path, tokens: Path, words: tuple[str, ...] | list[str]) -> tuple[str, str]:
    """Write the hotwords file and the SentencePiece vocab sherpa-onnx needs to encode it."""
    folder.mkdir(parents=True, exist_ok=True)
    vocab, hot = folder / "nemotron-bpe.vocab", folder / "nemotron-hotwords.txt"
    lines = []
    for line in tokens.read_text(encoding="utf-8").splitlines():
        piece, _, idx = line.rpartition(" ")
        lines.append(f"{piece}\t{-int(idx)}")
    vocab.write_text("\n".join(lines) + "\n", encoding="utf-8")
    hot.write_text("\n".join(words) + "\n", encoding="utf-8")
    return str(hot), str(vocab)


class NemotronSTT:
    def __init__(self, cfg: SttConfig, hotwords: tuple[str, ...] | list[str] = HOTWORDS,
                 hotwords_score: float = HOTWORDS_SCORE) -> None:
        d = cfg.nemotron_dir
        extra: dict = {}
        if hotwords:
            folder = Path(tempfile.mkdtemp(prefix="hark-stt-"))
            hot, vocab = write_hotwords(folder, Path(d) / "tokens.txt", hotwords)
            extra = dict(decoding_method="modified_beam_search", hotwords_file=hot,
                         hotwords_score=hotwords_score, modeling_unit="bpe", bpe_vocab=vocab)
        try:
            self._rec = so.OnlineRecognizer.from_transducer(
                encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
                joiner=f"{d}/joiner.int8.onnx", tokens=f"{d}/tokens.txt",
                num_threads=cfg.threads, **extra,
            )
        finally:
            if hotwords:
                shutil.rmtree(folder, ignore_errors=True)  # read once, at load
        self._stream = self._rec.create_stream()
        t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
        self.transcribe(0.1 * np.sin(2 * np.pi * 180 * t) * np.sin(2 * np.pi * 3 * t))

    def start(self) -> None:
        self._stream = self._rec.create_stream()

    def _decode(self) -> None:
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)

    def _text(self) -> str:
        return str(self._rec.get_result(self._stream)).strip()

    def feed(self, pcm: np.ndarray) -> str:
        self._stream.accept_waveform(SAMPLE_RATE, np.asarray(pcm, dtype=np.float32))
        self._decode()
        return self._text()

    def finish(self) -> str:
        self._stream.accept_waveform(SAMPLE_RATE, np.zeros(int(TAIL_S * SAMPLE_RATE), np.float32))
        self._stream.input_finished()
        self._decode()
        return self._text()

    def transcribe(self, pcm: np.ndarray) -> str:
        self.start()
        self._stream.accept_waveform(SAMPLE_RATE, np.asarray(pcm, dtype=np.float32))
        return self.finish()
