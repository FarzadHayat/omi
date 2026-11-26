"""Fast router: common spoken commands straight to a tool, with no LLM.

Conservative on purpose. Every pattern must match the whole normalised utterance,
so anything with extra words, questions or chat falls through to the talker
(route returns None). A wrong action is worse than a slower right one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class Route:
    tool: str  # a registry tool name, or "stop" (the daemon cancels speech and work)
    args: dict = field(default_factory=dict)
    ack: str = ""  # spoken on success; empty means speak the ToolResult.text


_LEAD = re.compile(
    r"^(?:(?:hey|hi|ok|okay|yo)\s+)?(?:(?:omi|ommy|oh me|oh my|hark)\s+)?"
    r"(?:(?:please|can you|could you|would you|will you|can u|just|now|so|um|uh|erm|hmm)\s+)*"
)
_TRAIL = re.compile(r"(?:\s+(?:please|for me|thanks|thank you|right now|now|omi|hark))+$")
_FILLER = re.compile(r"\b(?:um+|uh+|uhm|erm|er|hmm+|ah)\b")

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_ORDINALS = {w: i for i, w in enumerate(
    "_ first second third fourth fifth sixth seventh eighth ninth tenth".split()) if w != "_"}
_ORD = rf"({'|'.join(_ORDINALS)})"
# Homophones of small numbers, only trusted right after "workspace" or "desktop".
_SOUNDS_LIKE = {"to": 2, "too": 2, "for": 4, "won": 1}
_NUM = rf"(\d{{1,3}}|(?:(?:{'|'.join([*_UNITS, *_TENS, 'hundred', 'a'])})\s?)+)"


def normalise(text: str) -> str:
    t = text.lower().replace("%", " percent ").replace("-", " ").replace("’", "'")
    t = re.sub(r"\bwhat's\b|\bwhats\b", "what is", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t).replace("'", "")
    t = re.sub(r"\s+", " ", _FILLER.sub(" ", t)).strip()
    for _ in range(3):  # "hey hark can you please ..."
        t = _LEAD.sub("", t, count=1).strip()
    t = re.sub(r"^turned ", "turn ", t)  # speech recognition hears "turn the" as "turned the"
    return _TRAIL.sub("", t).strip()


def parse_number(text: str) -> int | None:
    """'fifty' -> 50, 'twenty five' -> 25, 'a hundred' -> 100, '75' -> 75."""
    text = text.strip()
    if text.isdigit():
        return int(text)
    total, seen = 0, False
    for w in text.split():
        if w in _UNITS:
            total += _UNITS[w]
        elif w in _TENS:
            total += _TENS[w]
        elif w == "hundred":
            total = (total or 1) * 100
        elif w == "a" and not seen:
            continue
        else:
            return None
        seen = True
    return total if seen else None


def _app_names() -> set[str]:
    global _APPS
    if _APPS is None:
        from hark.tools.omarchy import app_names

        _APPS = app_names()
    return _APPS


_APPS: set[str] | None = None


def warm() -> None:
    """Load the installed app names (about 60 ms once) so the first route is fast."""
    _app_index()


# --- misheard app names -------------------------------------------------------------
# Speech recognition splits and respells names it does not know ("fire folks",
# "spot if i", "obsidion"). A name is matched on its letters and on a rough sound key,
# and only when both are close and no other app comes near: a wrong launch is worse
# than asking the talker. Thresholds were tuned on 60 spoken commands and 40
# conversational phrases: no false matches.
FUZZY_SOUND = 0.85  # sound key similarity needed
FUZZY_LETTERS = 0.7  # letter similarity needed
FUZZY_MARGIN = 0.05  # how far ahead of any other app the best match must be
_SOUND_RULES = [(re.compile(p), r) for p, r in (
    ("ph", "f"), ("ck", "k"), ("x", "ks"), ("q", "k"), ("c(?=[eiy])", "s"), ("c", "k"),
    ("z", "s"), ("wh", "w"), ("gh", "g"), ("dg", "j"), ("y", "i"), ("(?<=[^aeiou])h", ""))]
_INDEX: tuple[set[str], dict[str, str], list[tuple[str, str, str]]] | None = None


def _compact(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name)


def sound_key(name: str) -> str:
    """A rough consonant skeleton: "fire folks" -> "frflks", "firefox" -> "frfks"."""
    s = _compact(name)
    for pattern, repl in _SOUND_RULES:
        s = pattern.sub(repl, s)
    s = s[:1] + re.sub(r"[aeiou]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", s)


def _app_index() -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    global _INDEX
    names = _app_names()
    if _INDEX is None or _INDEX[0] is not names:
        compact = {}
        for n in sorted(names, key=len):
            compact.setdefault(_compact(n), n)
        _INDEX = (names, compact, [(n, _compact(n), sound_key(n)) for n in sorted(names)])
    return _INDEX[1], _INDEX[2]


def match_app(spoken: str) -> str | None:
    """The installed app a spoken (maybe misheard) name means, or None."""
    names = _app_names()
    if spoken in names:
        return spoken
    compact, entries = _app_index()
    c = _compact(spoken)
    if c in compact:  # "moon light" -> "moonlight"
        return compact[c]
    k = sound_key(spoken)
    if len(c) < 4 or len(k) < 3:
        return None
    scored: list[tuple[float, str]] = []
    for name, name_c, name_k in entries:
        if len(c) < 0.75 * len(name_c):
            continue  # a cut-off word ("open spot") is not a misheard name
        sm = SequenceMatcher(None, k, name_k)
        if sm.real_quick_ratio() < FUZZY_SOUND or sm.quick_ratio() < FUZZY_SOUND:
            continue
        sound = sm.ratio()
        if sound < FUZZY_SOUND:
            continue
        letters = SequenceMatcher(None, c, name_c).ratio()
        if letters >= FUZZY_LETTERS:
            scored.append(((sound + letters) / 2, name))
    if not scored:
        return None
    scored.sort(reverse=True)
    if len(scored) > 1 and scored[0][0] - scored[1][0] < FUZZY_MARGIN:
        return None  # two apps sound alike: let the talker ask
    return scored[0][1]

_STOP = {"stop", "stop it", "stop talking", "be quiet", "quiet", "shut up", "shush", "cancel", "cancel that",
         "never mind", "nevermind", "forget it", "enough", "thats enough", "hush", "silence"}
_GOODBYE = {"goodbye", "good bye", "bye", "bye bye", "see ya", "see you", "see you later",
            "talk to you later", "catch you later", "thats all", "thats it", "im done", "were done",
            "end call", "end the call", "hang up", "stop listening", "nothing else", "all done"}
_MEDIA_NOUN = r"(?:the )?(?:music|song|track|video|playback|audio|podcast|media)"
_MEDIA = [
    (rf"(?:play|resume|unpause|continue)(?: {_MEDIA_NOUN})?|play some music", "play", ""),
    (rf"pause(?: {_MEDIA_NOUN})?|stop {_MEDIA_NOUN}", "pause", ""),
    (rf"(?:next|skip)(?: (?:this )?(?:song|track|video))?|(?:play the )?next (?:song|track)|skip it|skip this", "next", ""),
    (r"(?:previous|last|go back a|play the previous|play the last)(?: song| track| video)|previous|go back a song", "previous", ""),
    (r"what is (?:playing|this song|the song|this track)|what song is (?:this|playing)|whats playing|what is playing right now", "now_playing", ""),
]
_VOL = r"(?:the )?(?:volume|sound|audio)"
_BIT = r" (?:a (?:little )?bit|a little|a tad|slightly|some|more)"
_TIME = {"what time is it", "what is the time", "tell me the time", "time", "the time", "current time",
         "what is the time now", "what time is it now", "do you have the time"}
_DATE = {"what is the date", "what is todays date", "what day is it", "what day is it today", "todays date",
         "what is the date today", "tell me the date", "the date", "date", "what is today", "what date is it"}


def _media(t: str) -> Route | None:
    for pattern, action, ack in _MEDIA:
        if re.fullmatch(pattern, t):
            return Route("media", {"action": action}, ack)
    return None


def _volume(t: str) -> Route | None:
    if re.fullmatch(rf"mute(?: {_VOL}| the speakers?| it)?", t):
        return Route("volume", {"action": "mute"}, "")
    if re.fullmatch(rf"unmute(?: {_VOL}| the speakers?| it)?|turn {_VOL} (?:back )?on", t):
        return Route("volume", {"action": "unmute"}, "")
    if re.fullmatch(rf"turn (?:off {_VOL}|{_VOL} off)", t):
        return Route("volume", {"action": "mute"}, "")
    m = re.fullmatch(rf"(?:set |change |put |turn )?{_VOL}(?: to| at)? {_NUM}(?: percent)?", t) or \
        re.fullmatch(rf"(?:set|change|put|turn) (?:the )?(?:volume|sound) (?:to|at) {_NUM}(?: percent)?", t)
    if m:
        n = parse_number(m.group(1))
        return Route("volume", {"action": "set", "value": n}, "") if n is not None and 0 <= n <= 100 else None
    if re.fullmatch(r"(?:max|maximum|full) volume|volume (?:max|maximum|all the way up)", t):
        return Route("volume", {"action": "set", "value": 100}, "")
    by = rf"(?:{_BIT}|(?: by)? {_NUM}(?: percent)?)?"
    m = re.fullmatch(rf"(?:{_VOL} up|turn (?:it|{_VOL}) up|turn up {_VOL}|(?:raise|increase) {_VOL}|louder|make it(?:{_BIT})? louder|{_VOL} higher){by}", t)
    if m:
        return _step("up", m)
    m = re.fullmatch(rf"(?:{_VOL} down|turn (?:it|{_VOL}) down|turn down {_VOL}|(?:lower|decrease|reduce) {_VOL}|quieter|softer|make it(?:{_BIT})? (?:quieter|softer)|{_VOL} lower){by}", t)
    if m:
        return _step("down", m)
    return None


def _step(action: str, m: re.Match) -> Route | None:
    args: dict = {"action": action}
    if m.lastindex and m.group(m.lastindex):
        n = parse_number(m.group(m.lastindex))
        if n is None or not 1 <= n <= 100:
            return None
        args["value"] = n
    return Route("volume", args, "")


def _workspace(t: str) -> Route | None:
    which = rf"(?:(?:workspace|desktop) (?:number )?(?:{_NUM}|({'|'.join(_SOUNDS_LIKE)}))|(?:the )?{_ORD} (?:workspace|desktop))"
    for tool, lead in (("switch_workspace", r"(?:(?:switch|go|change|jump|move) to |show |open )?"),
                       ("move_window_to_workspace",
                        r"(?:move|send|put) (?:this|it|this window|the window|the current window) to ")):
        m = re.fullmatch(lead + which, t)
        if m:
            word = next(g for g in m.groups() if g)
            n = _ORDINALS.get(word) or _SOUNDS_LIKE.get(word) or parse_number(word)
            return Route(tool, {"n": n}, "") if n and 1 <= n <= 10 else None
    return None


def _app(t: str) -> Route | None:
    m = re.fullmatch(r"(?:open|launch|start|run|bring up|show me|show|switch to|go to|focus on|focus) (?:up )?(.+)", t)
    if not m:
        return None
    name = re.sub(r"^(?:the|my|a|an) ", "", m.group(1))
    name = re.sub(r" (?:app|application|window)$", "", name)
    app = match_app(name)
    return Route("launch_or_focus", {"app": app}, "") if app else None


_FULLSCREEN = re.compile(
    r"(?:toggle |go |make (?:it |this |this window |the window )?|exit |leave |enter )?full ?screen(?: mode)?")
_SCREENSHOT = re.compile(r"(?:take |grab |capture |make )?(?:a )?screen ?shot|capture the screen")
_LOCK = re.compile(r"lock (?:the |my )?(?:screen|computer|laptop|pc|machine|desktop)|lock screen")
_CLIPBOARD = re.compile(
    r"what is (?:on|in) (?:my|the) clipboard|read (?:me )?(?:my |the )?clipboard|what did i copy|"
    r"whats on (?:my|the) clipboard")


def route(text: str) -> Route | None:
    """The tool call for a common spoken command, or None to let the talker handle it."""
    t = normalise(text)
    if not t or len(t) > 60:
        return None
    if t in _STOP:
        return Route("stop", {}, "")
    if t in _GOODBYE:
        return Route("goodbye", {}, "See ya.")
    if t in _TIME:
        return Route("current_time", {"part": "time"}, "")
    if t in _DATE:
        return Route("current_time", {"part": "date"}, "")
    if _FULLSCREEN.fullmatch(t):
        return Route("toggle_fullscreen", {}, "")
    if _SCREENSHOT.fullmatch(t):
        return Route("screenshot", {}, "Screenshot saved.")
    if _LOCK.fullmatch(t):
        return Route("lock_screen", {}, "")
    if _CLIPBOARD.fullmatch(t):
        return Route("clipboard_get", {}, "")
    for fn in (_volume, _workspace, _media, _app):
        if (r := fn(t)) is not None:
            return r
    return None
