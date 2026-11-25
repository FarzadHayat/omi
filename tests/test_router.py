import time

import pytest

from hark.router import Route, normalise, parse_number, route, warm

POSITIVE = [
    ("Volume up.", "volume", {"action": "up"}),
    ("turn the volume down please", "volume", {"action": "down"}),
    ("Hey Hark, mute", "volume", {"action": "mute"}),
    ("unmute the sound", "volume", {"action": "unmute"}),
    ("volume to fifty percent", "volume", {"action": "set", "value": 50}),
    ("set the volume to 75%", "volume", {"action": "set", "value": 75}),
    ("set volume to twenty five", "volume", {"action": "set", "value": 25}),
    ("volume 30", "volume", {"action": "set", "value": 30}),
    ("max volume", "volume", {"action": "set", "value": 100}),
    ("louder", "volume", {"action": "up"}),
    ("make it quieter", "volume", {"action": "down"}),
    ("turn it up", "volume", {"action": "up"}),
    ("volume up by ten", "volume", {"action": "up", "value": 10}),
    ("Can you play the music?", "media", {"action": "play"}),
    ("Play.", "media", {"action": "play"}),
    ("resume", "media", {"action": "play"}),
    ("pause", "media", {"action": "pause"}),
    ("pause the music", "media", {"action": "pause"}),
    ("stop the music", "media", {"action": "pause"}),
    ("next song", "media", {"action": "next"}),
    ("skip this track", "media", {"action": "next"}),
    ("okay hark, next track please", "media", {"action": "next"}),
    ("previous track", "media", {"action": "previous"}),
    ("what's playing", "media", {"action": "now_playing"}),
    ("what song is this", "media", {"action": "now_playing"}),
    ("open spotify", "launch_or_focus", {"app": "spotify"}),
    ("open up spotify", "launch_or_focus", {"app": "spotify"}),
    ("Open the browser.", "launch_or_focus", {"app": "browser"}),
    ("launch terminal", "launch_or_focus", {"app": "terminal"}),
    ("could you open files please", "launch_or_focus", {"app": "files"}),
    ("start the editor", "launch_or_focus", {"app": "editor"}),
    ("open chromium", "launch_or_focus", {"app": "chromium"}),
    ("focus on chromium", "launch_or_focus", {"app": "chromium"}),
    ("bring up the file manager", "launch_or_focus", {"app": "file manager"}),
    ("workspace three", "switch_workspace", {"n": 3}),
    ("go to workspace 3", "switch_workspace", {"n": 3}),
    ("switch to desktop two", "switch_workspace", {"n": 2}),
    ("move this window to workspace four", "move_window_to_workspace", {"n": 4}),
    ("send it to workspace ten", "move_window_to_workspace", {"n": 10}),
    ("fullscreen", "toggle_fullscreen", {}),
    ("toggle full screen", "toggle_fullscreen", {}),
    ("make it full screen", "toggle_fullscreen", {}),
    ("take a screenshot", "screenshot", {}),
    ("screen shot", "screenshot", {}),
    ("what time is it", "current_time", {"part": "time"}),
    ("What's the time?", "current_time", {"part": "time"}),
    ("what's today's date", "current_time", {"part": "date"}),
    ("what day is it", "current_time", {"part": "date"}),
    ("lock the screen", "lock_screen", {}),
    ("lock screen", "lock_screen", {}),
    ("what's on my clipboard", "clipboard_get", {}),
    ("read the clipboard", "clipboard_get", {}),
    ("stop", "stop", {}),
    ("hark stop", "stop", {}),
    ("never mind", "stop", {}),
    ("be quiet", "stop", {}),
]

NEGATIVE = [
    "open up to me about volume",
    "what do you think of spotify",
    "play it by ear",
    "what's the volume of a sphere",
    "can you open spotify and play some jazz",
    "how do I take a screenshot",
    "is it time to go",
    "mute the microphone",
    "stop the war",
    "lock the door",
    "lock it",
    "next week",
    "previous meeting",
    "open the pod bay doors",
    "what time is it in tokyo",
    "take a screenshot and send it to slack",
    "full screen mode is annoying",
    "workspace eleven",
    "set the volume to 150",
    "volume up by a lot",
    "tell me about the browser wars",
    "why is the music so loud",
    "play something by radiohead",
    "open my email and reply to john",
    "go to sleep",
    "switch to dark mode",
    "what is on my screen",
    "close spotify",
    "turn up the heat",
    "could you tell me a joke",
    "what's the time difference to london",
    "is spotify open",
    "open",
    "volume",
    "show me the money",
    "skip leg day",
    "the next big thing",
    "pause for a second and think",
    "I want to open a bakery",
    "workspaces are confusing",
    "",
]


@pytest.mark.parametrize(("text", "tool", "args"), POSITIVE)
def test_positive(text, tool, args):
    r = route(text)
    assert r is not None, text
    assert (r.tool, r.args) == (tool, args)


@pytest.mark.parametrize("text", NEGATIVE)
def test_negative(text):
    assert route(text) is None


def test_table_sizes():
    assert len(POSITIVE) >= 40 and len(NEGATIVE) >= 30


def test_normalise_and_numbers():
    assert normalise("Hey Hark, could you please Open Spotify, thanks!") == "open spotify"
    assert parse_number("twenty five") == 25
    assert parse_number("a hundred") == 100
    assert parse_number("75") == 75
    assert parse_number("lots") is None


def test_route_type():
    r = route("screenshot")
    assert isinstance(r, Route) and r.ack


def test_speed():
    warm()
    phrases = [p[0] for p in POSITIVE] + NEGATIVE
    n = 20
    t = time.perf_counter()
    for _ in range(n):
        for p in phrases:
            route(p)
    per_call_ms = (time.perf_counter() - t) * 1000 / (n * len(phrases))
    worst = 0.0
    for p in phrases:
        t = time.perf_counter()
        route(p)
        worst = max(worst, (time.perf_counter() - t) * 1000)
    print(f"\nrouter: mean {per_call_ms * 1000:.1f} us, worst {worst * 1000:.1f} us per call")
    assert per_call_ms < 2 and worst < 2
