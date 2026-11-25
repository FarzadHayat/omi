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


# --- speech-recognition form: no punctuation, number words, filler, mishearings ---
# Installed apps are fixed here so the fuzzy matching is tested against a known list.

SPOKEN_APPS = {"firefox", "chromium", "spotify", "obsidian", "discord", "signal", "terminal", "files", "file manager",
        "browser", "web browser", "editor", "text editor", "whatsapp", "zoom", "ghostty", "moonlight", "kdenlive",
        "libreoffice writer", "libreoffice calc", "obs studio", "localsend", "superwhisper", "nautilus", "btop",
        "youtube", "google maps", "google photos", "hey", "basecamp", "bitwarden", "neovim", "x", "signed", "cursor",
        "papers", "minecraft", "pinta", "qbittorrent", "omacut", "omakeys", "omarchy", "t3 code", "docker"}
SPOKEN = [
    ("turn the volume down", "volume", {"action": "down"}),
    ("turned the volume down", "volume", {"action": "down"}),
    ("um turn it up a bit", "volume", {"action": "up"}),
    ("volume to fifty", "volume", {"action": "set", "value": 50}),
    ("set volume to thirty five percent", "volume", {"action": "set", "value": 35}),
    ("can you make it louder", "volume", {"action": "up"}),
    ("make it a little quieter", "volume", {"action": "down"}),
    ("volume up by ten", "volume", {"action": "up", "value": 10}),
    ("mute", "volume", {"action": "mute"}),
    ("unmute the sound", "volume", {"action": "unmute"}),
    ("set the volume at seventy", "volume", {"action": "set", "value": 70}),
    ("turn the sound off", "volume", {"action": "mute"}),
    ("lower the volume please", "volume", {"action": "down"}),
    ("volume down twenty percent", "volume", {"action": "down", "value": 20}),
    ("turn it down a little bit", "volume", {"action": "down"}),
    ("set the volume to one hundred", "volume", {"action": "set", "value": 100}),
    ("open firefox", "launch_or_focus", {"app": "firefox"}),
    ("open fire folks", "launch_or_focus", {"app": "firefox"}),
    ("launch spot if i", "launch_or_focus", {"app": "spotify"}),
    ("uh open up chromium", "launch_or_focus", {"app": "chromium"}),
    ("start obsidion", "launch_or_focus", {"app": "obsidian"}),
    ("open dis cord", "launch_or_focus", {"app": "discord"}),
    ("open the terminal", "launch_or_focus", {"app": "terminal"}),
    ("switch to firefox", "launch_or_focus", {"app": "firefox"}),
    ("open whats app", "launch_or_focus", {"app": "whatsapp"}),
    ("open ghost tea", "launch_or_focus", {"app": "ghostty"}),
    ("open moon light", "launch_or_focus", {"app": "moonlight"}),
    ("okay hark open the browser", "launch_or_focus", {"app": "browser"}),
    ("go to workspace three", "switch_workspace", {"n": 3}),
    ("workspace two", "switch_workspace", {"n": 2}),
    ("switch to desktop number four", "switch_workspace", {"n": 4}),
    ("move this window to workspace five", "move_window_to_workspace", {"n": 5}),
    ("send it to desktop one", "move_window_to_workspace", {"n": 1}),
    ("go to the third workspace", "switch_workspace", {"n": 3}),
    ("switch to workspace to", "switch_workspace", {"n": 2}),
    ("workspace for", "switch_workspace", {"n": 4}),
    ("next song", "media", {"action": "next"}),
    ("skip", "media", {"action": "next"}),
    ("play the previous song", "media", {"action": "previous"}),
    ("pause the music", "media", {"action": "pause"}),
    ("resume", "media", {"action": "play"}),
    ("play music", "media", {"action": "play"}),
    ("what song is this", "media", {"action": "now_playing"}),
    ("whats playing", "media", {"action": "now_playing"}),
    ("next track please", "media", {"action": "next"}),
    ("what time is it", "current_time", {"part": "time"}),
    ("um what time is it right now", "current_time", {"part": "time"}),
    ("whats todays date", "current_time", {"part": "date"}),
    ("what day is it today", "current_time", {"part": "date"}),
    ("take a screenshot", "screenshot", {}),
    ("grab a screen shot", "screenshot", {}),
    ("lock the screen", "lock_screen", {}),
    ("lock my computer", "lock_screen", {}),
    ("full screen", "toggle_fullscreen", {}),
    ("make this window full screen", "toggle_fullscreen", {}),
    ("whats on my clipboard", "clipboard_get", {}),
    ("stop", "stop", {}),
    ("never mind", "stop", {}),
    ("shut up", "stop", {}),
    ("hey hark uh stop talking", "stop", {}),
]
SPOKEN_NEGATIVE = [
    "whats the weather like today", "tell me a joke", "how loud is a jet engine", "can you open the window",
    "what time does the store close", "what day is christmas this year", "play a song by queen",
    "turn the lights off", "remember to buy milk", "set a timer for ten minutes", "what is the volume of a sphere",
    "how do i lock my car", "next time remind me", "i love firefox", "what is firefox", "who made spotify",
    "take me home", "stop the war", "pause for a moment and think", "switch to decaf", "open a new tab",
    "go to sleep", "what is two plus two", "its a moot point", "date night ideas", "how are you", "good morning",
    "thanks", "open sesame", "start a timer", "open fire", "launch the rocket", "show me a picture of a cat",
    "focus on your breathing", "start over", "open the door", "go to google", "run a mile",
    "open the pod bay doors", "start the car",
]


@pytest.fixture
def spoken_apps(monkeypatch):
    from hark import router

    monkeypatch.setattr(router, "_APPS", set(SPOKEN_APPS))
    monkeypatch.setattr(router, "_INDEX", None)


@pytest.mark.parametrize(("text", "tool", "args"), SPOKEN)
def test_spoken_command(spoken_apps, text, tool, args):
    r = route(text)
    assert r is not None, text
    assert (r.tool, r.args) == (tool, args)


@pytest.mark.parametrize("text", SPOKEN_NEGATIVE)
def test_spoken_conversation_is_left_to_the_talker(spoken_apps, text):
    assert route(text) is None


@pytest.mark.parametrize("text", ["open spot", "open fire", "open what", "switch to decaf", "start the car",
                                  "open steam", "open slack", "open hay", "open sesame"])
def test_fuzzy_app_match_never_guesses_at_plain_words(spoken_apps, text):
    assert route(text) is None


def test_spoken_table_sizes():
    assert len(SPOKEN) >= 60 and len(SPOKEN_NEGATIVE) >= 40


def test_sound_key():
    from hark.router import sound_key

    assert sound_key("fire folks") == "frflks" and sound_key("firefox") == "frfks"
    assert sound_key("spot if i") == sound_key("spotify")


def test_spoken_speed(spoken_apps):
    phrases = [p[0] for p in SPOKEN] + SPOKEN_NEGATIVE
    route("open fire folks")  # builds the app index once
    worst = 0.0
    for _ in range(5):
        for p in phrases:
            t = time.perf_counter()
            route(p)
            worst = max(worst, (time.perf_counter() - t) * 1000)
    assert worst < 2
