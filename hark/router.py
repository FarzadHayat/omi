"""Fast router: common spoken commands straight to a tool, with no LLM.

Conservative on purpose. Every pattern must match the whole normalised utterance,
so anything with extra words, questions or chat falls through to the talker
(route returns None). A wrong action is worse than a slower right one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Route:
    tool: str  # a registry tool name, or "stop" (the daemon cancels speech and work)
    args: dict = field(default_factory=dict)
    ack: str = ""  # spoken on success; empty means speak the ToolResult.text


_LEAD = re.compile(
    r"^(?:(?:hey|hi|ok|okay|yo)\s+)?(?:hark\s+)?"
    r"(?:(?:please|can you|could you|would you|will you|can u|just|now|so|um|uh|erm|hmm)\s+)*"
)
_TRAIL = re.compile(r"(?:\s+(?:please|for me|thanks|thank you|now|hark))+$")

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_NUM = rf"(\d{{1,3}}|(?:(?:{'|'.join([*_UNITS, *_TENS, 'hundred', 'a'])})\s?)+)"


def normalise(text: str) -> str:
    t = text.lower().replace("%", " percent ").replace("-", " ").replace("’", "'")
    t = re.sub(r"\bwhat's\b|\bwhats\b", "what is", t)
    t = re.sub(r"[^a-z0-9' ]+", " ", t).replace("'", "")
    t = re.sub(r"\s+", " ", t).strip()
    for _ in range(3):  # "hey hark can you please ..."
        t = _LEAD.sub("", t, count=1).strip()
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
    _app_names()

_STOP = {"stop", "stop it", "stop talking", "be quiet", "quiet", "shut up", "shush", "cancel", "cancel that",
         "never mind", "nevermind", "forget it", "enough", "thats enough", "hush", "silence"}
_MEDIA_NOUN = r"(?:the )?(?:music|song|track|video|playback|audio|podcast|media)"
_MEDIA = [
    (rf"(?:play|resume|unpause|continue)(?: {_MEDIA_NOUN})?|play some music", "play", ""),
    (rf"pause(?: {_MEDIA_NOUN})?|stop {_MEDIA_NOUN}", "pause", ""),
    (rf"(?:next|skip)(?: (?:this )?(?:song|track|video))?|(?:play the )?next (?:song|track)|skip it|skip this", "next", ""),
    (r"(?:previous|last|go back a|play the previous|play the last)(?: song| track| video)|previous|go back a song", "previous", ""),
    (r"what is (?:playing|this song|the song|this track)|what song is (?:this|playing)|whats playing|what is playing right now", "now_playing", ""),
]
_VOL = r"(?:the )?(?:volume|sound|audio)"
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
    if re.fullmatch(rf"unmute(?: {_VOL}| the speakers?| it)?", t):
        return Route("volume", {"action": "unmute"}, "")
    m = re.fullmatch(rf"(?:set |change |put |turn )?{_VOL}(?: to| at)? {_NUM}(?: percent)?", t) or \
        re.fullmatch(rf"(?:set|change|put|turn) (?:the )?(?:volume|sound) (?:to|at) {_NUM}(?: percent)?", t)
    if m:
        n = parse_number(m.group(1))
        return Route("volume", {"action": "set", "value": n}, "") if n is not None and 0 <= n <= 100 else None
    if re.fullmatch(r"(?:max|maximum|full) volume|volume (?:max|maximum|all the way up)", t):
        return Route("volume", {"action": "set", "value": 100}, "")
    by = rf"(?: by {_NUM}(?: percent)?)?"
    m = re.fullmatch(rf"(?:{_VOL} up|turn (?:it|{_VOL}) up|turn up {_VOL}|(?:raise|increase) {_VOL}|louder|make it louder|{_VOL} higher){by}", t)
    if m:
        return _step("up", m)
    m = re.fullmatch(rf"(?:{_VOL} down|turn (?:it|{_VOL}) down|turn down {_VOL}|(?:lower|decrease|reduce) {_VOL}|quieter|softer|make it quieter|{_VOL} lower){by}", t)
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
    m = re.fullmatch(rf"(?:(?:switch|go|change|jump|move) to |show |open )?(?:workspace|desktop) (?:number )?{_NUM}", t)
    if m:
        n = parse_number(m.group(1))
        return Route("switch_workspace", {"n": n}, "") if n and 1 <= n <= 10 else None
    m = re.fullmatch(rf"(?:move|send|put) (?:this|it|this window|the window|the current window) to (?:workspace|desktop) (?:number )?{_NUM}", t)
    if m:
        n = parse_number(m.group(1))
        return Route("move_window_to_workspace", {"n": n}, "") if n and 1 <= n <= 10 else None
    return None


def _app(t: str) -> Route | None:
    m = re.fullmatch(r"(?:open|launch|start|run|bring up|show me|show|switch to|go to|focus on|focus) (?:up )?(.+)", t)
    if not m:
        return None
    name = re.sub(r"^(?:the|my|a|an) ", "", m.group(1))
    name = re.sub(r" (?:app|application|window)$", "", name)
    if name in _app_names():
        return Route("launch_or_focus", {"app": name}, "")
    return None


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
