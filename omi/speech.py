"""Turn streamed model text into speakable pieces.

ClauseChunker cuts the text stream so TTS can start on the first clause while the
model is still writing, which is most of the perceived speed. clean_for_speech
removes what a model writes for screens but should never be read aloud.
"""

from __future__ import annotations

import re

_SENTENCE_END = ".!?"
_CLAUSE_END = ",;:…"
_CLOSERS = "\"')]”’"
# A period after these is not the end of a sentence. "etc." is left out on purpose:
# it usually does end one.
_ABBREVIATIONS = frozenset(
    "e.g. i.e. vs. dr. mr. mrs. ms. prof. sr. jr. st. mt. no. approx. inc. ltd. co. "
    "fig. cf. al. jan. feb. mar. apr. jun. jul. aug. sep. sept. oct. nov. dec.".split()
)
_INITIALS = re.compile(r"(?:[a-z]\.)+")
_WORD = re.compile(r"\S+\s")


class ClauseChunker:
    """Feed model text as it arrives; get back chunks ready for TTS.

    The first chunk ends at the first clause boundary (or after `first_words` words)
    so audio starts early. Later chunks end at sentence ends, which sound more
    natural; a very long sentence still breaks at a clause after `max_words` words,
    or at a word after `max_words` words when it has no clause marks at all.
    Fenced code blocks are dropped, since they are never worth reading aloud.
    """

    def __init__(self, first_words: int = 8, max_words: int = 30) -> None:
        self.first_words = first_words
        self.max_words = max_words
        self._raw = ""  # unprocessed input, may end in a partial ``` fence
        self._buf = ""  # speakable text waiting for a boundary
        self._in_fence = False
        self._first = True

    def feed(self, text: str) -> list[str]:
        self._raw += text
        self._strip_fences(final=False)
        return self._drain(final=False)

    def flush(self) -> list[str]:
        """End of the reply: return whatever is left."""
        self._strip_fences(final=True)
        return self._drain(final=True)

    def _strip_fences(self, final: bool) -> None:
        raw = self._raw
        while True:
            idx = raw.find("```")
            if self._in_fence:
                if idx < 0:
                    raw = "" if final else raw[-2:]
                    break
                raw = raw[idx + 3 :]
                self._in_fence = False
            else:
                if idx < 0:
                    keep = 0 if final else len(raw) - len(raw.rstrip("`"))
                    self._buf += raw[: len(raw) - keep]
                    raw = raw[len(raw) - keep :]
                    break
                self._buf += raw[:idx] + " "
                raw = raw[idx + 3 :]
                self._in_fence = True
        self._raw = raw

    def _drain(self, final: bool) -> list[str]:
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            self._emit(self._buf[:cut], out)
            self._buf = self._buf[cut:].lstrip()
        if final:
            self._emit(self._buf, out)
            self._buf = ""
        return out

    def _emit(self, chunk: str, out: list[str]) -> None:
        chunk = chunk.strip()
        if any(ch.isalnum() for ch in chunk):
            out.append(chunk)
            self._first = False

    def _find_cut(self) -> int | None:
        """Index just after the next boundary in the buffer, or None to wait."""
        buf = self._buf
        word_cut = None
        if self._first:
            words = _WORD.finditer(buf)
            for n, m in enumerate(words, 1):
                if n == self.first_words:
                    word_cut = m.end()
                    break
        i = 0
        n = len(buf)
        while i < n:
            if word_cut is not None and i >= word_cut:
                return word_cut
            c = buf[i]
            if c in _SENTENCE_END or c in _CLAUSE_END:
                j = i
                while j < n and buf[j] in _SENTENCE_END + _CLAUSE_END:
                    j += 1
                while j < n and buf[j] in _CLOSERS:
                    j += 1
                if j >= n:
                    # The first sentence end is cut without waiting for the next token
                    # (about 80 ms locally), unless it may be a number or abbreviation.
                    if self._first and c in _SENTENCE_END and i > 0 and buf[i - 1].isalpha() \
                            and not (c == "." and self._is_abbreviation(buf, i)):
                        return j
                    return word_cut  # cannot tell yet what follows
                if not buf[j].isspace():
                    i = j
                    continue
                run = buf[i:j]
                is_ellipsis = run.startswith("..") or "…" in run
                if run[0] == "." and len(run.rstrip(_CLOSERS)) == 1 and self._is_abbreviation(buf, i):
                    i = j
                    continue
                sentence = any(ch in "!?" for ch in run) or (run[0] == "." and not is_ellipsis)
                if self._first or sentence:
                    return j
                if len(buf[:i].split()) >= self.max_words:
                    return j
                i = j
                continue
            i += 1
        if word_cut is None:
            # A run-on with no usable boundary: speak it in max_words pieces rather than
            # leaving the user in silence until the model finishes.
            for n, m in enumerate(_WORD.finditer(buf), 1):
                if n == self.max_words:
                    return m.end()
        return word_cut

    @staticmethod
    def _is_abbreviation(buf: str, dot: int) -> bool:
        start = dot
        while start > 0 and not buf[start - 1].isspace():
            start -= 1
        token = buf[start : dot + 1].lstrip("\"'([“‘").lower()
        return token in _ABBREVIATIONS or bool(_INITIALS.fullmatch(token))


# --- cleaning -------------------------------------------------------------------

_FENCE = re.compile(r"```.*?(?:```|$)", re.S)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>()]*[^\s<>().,;:!?'\"]")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_QUOTE = re.compile(r"^\s*>+\s?", re.M)
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_BOLD = re.compile(r"(\*\*|__|~~)(.+?)\1", re.S)
_EM_STAR = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)")
_EM_UNDER = re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w])")
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff️‍⃣]"
)
_SYMBOLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<=\d)\s*[–—-]\s*(?=\d)"), " to "),
    (re.compile(r"\s*(?:->|=>|→)\s*"), " to "),
    (re.compile(r"\s*[–—]\s*"), ", "),
    (re.compile(r"\s+-\s+"), ", "),
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"(\d)\s*%"), r"\1 percent"),
    (re.compile(r"\$(\d[\d,]*(?:\.\d+)?)"), r"\1 dollars"),
    (re.compile(r"€(\d[\d,]*(?:\.\d+)?)"), r"\1 euros"),
    (re.compile(r"£(\d[\d,]*(?:\.\d+)?)"), r"\1 pounds"),
    (re.compile(r"\s*°\s*C\b"), " degrees Celsius"),
    (re.compile(r"\s*°\s*F\b"), " degrees Fahrenheit"),
    (re.compile(r"\s*°"), " degrees"),
    (re.compile(r"~\s*(?=\d)"), "about "),
    (re.compile(r"#(?=\d)"), "number "),
    (re.compile(r"(?<=\d)\s*\+\s*(?=\d)"), " plus "),
    (re.compile(r"(?<=\w)@(?=\w)"), " at "),
    (re.compile(r"[*#`|\\_]+"), " "),
]


def clean_for_speech(text: str) -> str:
    """Strip markdown, code, URLs and emoji, and spell out common symbols."""
    text = _FENCE.sub(" ", text)
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("a link", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _BOLD.sub(r"\2", text)
    text = _EM_STAR.sub(r"\1", text)
    text = _EM_UNDER.sub(r"\1", text)
    text = _EMOJI.sub("", text)
    for pattern, repl in _SYMBOLS:
        text = pattern.sub(repl, text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if lines and lines[-1][-1] not in _SENTENCE_END + _CLAUSE_END:
            lines[-1] += "."
        lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", text)
    return text.lstrip(" ,;:").rstrip()
