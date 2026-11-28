"""The guided look to talk setup: the numbers it picks and the flow a new user sees."""

from __future__ import annotations

import time

import numpy as np
import pytest

from omi import look_tune
from omi.look_tune import TuneError, choose

FACING = [0.02, -0.03, 0.05, 0.0, 0.04, -0.02, 0.03, 0.01, 0.06, -0.01]
QUIET = [0.012, 0.015, 0.018, 0.014, 0.02, 0.016, 0.028]  # his still face
TALKING = [0.039, 0.074, 0.067, 0.065, 0.061, 0.09, 0.045, 0.066, 0.059, 0.057]  # him reading


def test_threshold_sits_between_quiet_and_talking_nearer_quiet():
    t = choose(QUIET, TALKING, FACING, [0.7, 0.8, 0.75, 0.9])
    assert t.quiet < t.lip_threshold < t.talking
    assert t.lip_threshold - t.quiet < t.talking - t.lip_threshold  # missing words is worse
    assert 0.03 <= t.lip_threshold <= 0.04  # where his hand tuning landed
    assert t.yaw_center == pytest.approx(0.015, abs=0.01)
    assert 0.1 < t.yaw_range < 0.7  # between facing (~0.05 off) and looking aside (~0.7)


def test_a_webcam_on_another_monitor_shifts_what_facing_means():
    off_center = [y + 0.45 for y in FACING]  # the screen you use is beside the camera
    t = choose(QUIET, TALKING, off_center, [-0.3, -0.25, -0.35])
    assert t.yaw_center == pytest.approx(0.465, abs=0.02)
    from omi.look import facing_from_faces
    import numpy as np

    def face(dx):  # a YuNet row with the nose dx eye-widths off centre
        return np.array([[200, 60, 120, 144, 236, 100, 284, 100, 260 + dx * 48, 130, 245, 150, 275, 150, 0.9]])

    assert facing_from_faces(face(0.45), 640, t.yaw_center, t.yaw_range)  # looking at your screen
    assert not facing_from_faces(face(0.45), 640)  # the old default would have said no


def test_bad_runs_explain_what_to_change():
    with pytest.raises(TuneError, match="see your face"):
        choose(QUIET, TALKING, [0.0, 0.1], [])
    with pytest.raises(TuneError, match="see your mouth"):
        choose([], TALKING, FACING, [])
    with pytest.raises(TuneError, match="as much while quiet as while talking"):
        choose(QUIET, [0.02, 0.025, 0.03, 0.022], FACING, [])


def test_face_vanishing_when_turned_still_gives_a_sane_range():
    t = choose(QUIET, TALKING, FACING, [])  # turned far enough that no face was seen
    assert 0.15 <= t.yaw_range <= 0.6


class _FakeTracker:
    mode = "quiet"

    def __init__(self, *a, **k) -> None:
        self.error = None

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def wake(self) -> None: ...

    def lips_moving(self, now=None):
        return {"quiet": 0.015, "talk": 0.06, "away": None}[_FakeTracker.mode]

    def readings(self, since, until=None):
        yaw = 0.8 if _FakeTracker.mode == "away" else 0.02
        return [(True, None, yaw)] * 10


class _FakeMic:
    def start(self) -> None: ...
    def stop(self) -> None: ...

    def between(self, since, until):
        return np.zeros(16000, np.float32)


class _Screen(look_tune._Screen):
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.live = False

    def say(self, text: str = "") -> None:
        self.lines.append(text)
        if "Step 2" in text:
            _FakeTracker.mode = "talk"
        elif "Step 3" in text:
            _FakeTracker.mode = "away"


def test_guided_run_saves_and_tells_the_user(monkeypatch):
    for name, value in (("QUIET_S", 0.15), ("TALK_S", 0.15), ("AWAY_S", 0.15), ("SETTLE_S", 0.0),
                        ("COUNTDOWN_S", 0), ("TICK_S", 0.01)):
        monkeypatch.setattr(look_tune, name, value)
    monkeypatch.setattr(look_tune.look, "LookTracker", _FakeTracker)
    _FakeTracker.mode = "quiet"
    saved = []
    screen = _Screen()
    start = time.monotonic()
    code = look_tune.run("", lambda t: saved.append(t) or "to your settings", ask=lambda _: "", screen=screen,
                         mic=_FakeMic())
    assert code == 0 and time.monotonic() - start < 5
    (t,) = saved
    assert 0.015 < t.lip_threshold < 0.06
    text = "\n".join(screen.lines)
    for expected in ("Step 1 of 3", "Stay quiet", "Step 2 of 3", look_tune.SENTENCE, "Step 3 of 3",
                     "All set", "Saved to your settings"):
        assert expected in text


def test_guided_run_changes_nothing_when_it_cannot_tell(monkeypatch):
    for name, value in (("QUIET_S", 0.1), ("TALK_S", 0.1), ("AWAY_S", 0.1), ("SETTLE_S", 0.0),
                        ("COUNTDOWN_S", 0), ("TICK_S", 0.01)):
        monkeypatch.setattr(look_tune, name, value)
    monkeypatch.setattr(look_tune.look, "LookTracker", _FakeTracker)
    monkeypatch.setattr(_FakeTracker, "lips_moving", lambda self, now=None: 0.02)  # same quiet or talking
    saved = []
    screen = _Screen()
    assert look_tune.run("", saved.append, ask=lambda _: "", screen=screen, mic=_FakeMic()) == 1
    assert saved == [] and any("Nothing was changed" in line for line in screen.lines)


def test_voice_is_saved_only_when_the_user_says_yes(monkeypatch):
    for name, value in (("QUIET_S", 0.1), ("TALK_S", 0.1), ("AWAY_S", 0.1), ("SETTLE_S", 0.0),
                        ("COUNTDOWN_S", 0), ("TICK_S", 0.01)):
        monkeypatch.setattr(look_tune, name, value)
    monkeypatch.setattr(look_tune.look, "LookTracker", _FakeTracker)
    monkeypatch.setattr(look_tune, "heard_you", lambda q, t: True)
    for answer, saved_voice in (("", False), ("y", True)):
        _FakeTracker.mode = "quiet"
        voices = []
        screen = _Screen()
        code = look_tune.run("", lambda t: "here", ask=lambda q: answer if "voice" in q else "", screen=screen,
                             mic=_FakeMic(), save_voice=lambda pcm: voices.append(pcm) or "✓ Saved")
        assert code == 0 and bool(voices) is saved_voice  # opt-in: Enter alone means no
        assert any("I heard you" in line for line in screen.lines)


def test_a_silent_mic_is_reported_and_no_voice_is_offered(monkeypatch):
    for name, value in (("QUIET_S", 0.1), ("TALK_S", 0.1), ("AWAY_S", 0.1), ("SETTLE_S", 0.0),
                        ("COUNTDOWN_S", 0), ("TICK_S", 0.01)):
        monkeypatch.setattr(look_tune, name, value)
    monkeypatch.setattr(look_tune.look, "LookTracker", _FakeTracker)
    _FakeTracker.mode = "quiet"
    asked = []
    screen = _Screen()
    look_tune.run("", lambda t: "here", ask=lambda q: asked.append(q) or "y", screen=screen, mic=_FakeMic(),
                  save_voice=lambda pcm: "saved")
    assert any("couldn't hear you" in line for line in screen.lines)
    assert not any("voice" in q for q in asked)
