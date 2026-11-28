"""`omi look-tune`: a 20-second guided setup that fits look to talk to one person.

Three short steps in front of the webcam: stay quiet, read two sentences out loud, then
turn to look beside you. From them it picks how much lip movement counts as talking (some
people barely move their lips, some cameras are blurry) and which head turn means
"facing the screen" (the webcam may sit on a different monitor than the one you use).
It also checks the mic heard you, and offers to save your voice so Omi ignores other
people (omi.voice_id). Nothing is recorded: only those numbers are kept.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from omi import look

SENTENCE = ("Hey Omi, could you check what I have on tomorrow afternoon? "
            "Then remind me to call the dentist before lunch on Friday.")
QUIET_S, TALK_S, AWAY_S = 5.0, 10.0, 4.0
HEARD_SHARE = 0.2  # share of step 2 that must sound like a voice for "I heard you"
SETTLE_S = 1.5  # the first moments of a step are people getting into position
COUNTDOWN_S = 3
TICK_S = 0.25
FIND_FACE_S = 6.0


class TuneError(Exception):
    """The run cannot give good numbers; the message says what to change."""


@dataclass
class Tuning:
    lip_threshold: float
    yaw_center: float
    yaw_range: float
    quiet: float  # typical lip movement while quiet
    talking: float  # typical lip movement while talking


class Mic:
    """Records the mic in the background while the steps run, with a time for each piece."""

    def __init__(self, source: str = "") -> None:
        self.source = source
        self._proc: subprocess.Popen | None = None
        self._chunks: list[tuple[float, np.ndarray]] = []
        self._lock = threading.Lock()
        self.error = ""

    def start(self) -> None:
        exe = shutil.which("pw-record")
        if exe is None:
            self.error = "pw-record is missing"
            return
        cmd = [exe, "--raw", "--format", "f32", "--rate", "16000", "--channels", "1", "--latency", "32ms",
               "-P", "application.name=Omi", "-P", "node.name=omi-tune"]
        if self.source:
            cmd += ["--target", self.source]
        try:
            self._proc = subprocess.Popen([*cmd, "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        except OSError as exc:
            self.error = str(exc)
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while data := self._proc.stdout.read(512 * 4):
            with self._lock:
                self._chunks.append((time.monotonic(), np.frombuffer(data[: len(data) // 4 * 4], "<f4").copy()))

    def between(self, since: float, until: float) -> np.ndarray:
        with self._lock:
            parts = [c for t, c in self._chunks if since <= t <= until]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait()
            self._proc = None


def heard_you(quiet: np.ndarray, talk: np.ndarray) -> bool:
    """True when step 2 clearly had a voice in it compared with the quiet step."""
    def frame_rms(pcm: np.ndarray) -> np.ndarray:
        n = len(pcm) // 512
        return np.sqrt(np.mean(pcm[: n * 512].reshape(n, 512) ** 2, axis=1)) if n else np.zeros(0)

    q, t = frame_rms(quiet), frame_rms(talk)
    if not len(t):
        return False
    floor = max(0.004, 4 * float(np.median(q)) if len(q) else 0.0)
    return float(np.mean(t > floor)) >= HEARD_SHARE


def mic_source(audio_source: str) -> str:
    """The mic Omi itself listens to: its echo-cancelled copy when that exists."""
    from omi.audio import echo

    try:
        out = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True,
                             timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    return echo.SOURCE if any(line.split("\t")[1:2] == [echo.SOURCE] for line in out.splitlines()) else audio_source


def _pct(values: list[float], q: float) -> float:
    vals = sorted(values)
    return vals[min(len(vals) - 1, max(0, round(q * (len(vals) - 1))))]


def choose(quiet: list[float], talking: list[float], facing_yaws: list[float], away_yaws: list[float]) -> Tuning:
    """Numbers from one run. quiet/talking: lip spreads; *_yaws: head turns seen while
    facing the screen and while looking beside you."""
    if len(facing_yaws) < 8:
        raise TuneError("I couldn't see your face well enough. Sit in front of the camera with some "
                        "light on your face, then try again.")
    center = statistics.median(facing_yaws)
    facing_dev = _pct([abs(y - center) for y in facing_yaws], 0.9)
    away_devs = [abs(y - center) for y in away_yaws]
    if len(away_devs) >= 3 and statistics.median(away_devs) > facing_dev + 0.1:
        yaw_range = (facing_dev + _pct(away_devs, 0.2)) / 2  # halfway between the two
    else:
        yaw_range = facing_dev + 0.15  # turned away the face vanished, which already counts as away
    yaw_range = min(0.6, max(0.15, yaw_range))

    if len(quiet) < 3 or len(talking) < 3:
        raise TuneError("I couldn't see your mouth clearly. Keep your hands away from your face, add "
                        "some light, and try again.")
    q, t = _pct(quiet, 0.9), statistics.median(talking)
    if t < q * 1.5 or t - q < 0.012:
        raise TuneError("Your lips moved about as much while quiet as while talking. Try again: stay "
                        "still in step 1 and read clearly in step 2. If it keeps happening, the room "
                        "may be too dark for the camera.")
    threshold = min(0.2, max(0.005, q + (t - q) / 3))  # closer to quiet: missing your words is worse
    return Tuning(round(threshold, 3), round(center, 3), round(yaw_range, 3), round(q, 3), round(t, 3))


class _Screen:
    """Prints steps; redraws one live line when attached to a terminal."""

    def __init__(self, out=sys.stdout) -> None:
        self.out = out
        self.live = out.isatty()

    def say(self, text: str = "") -> None:
        self.out.write(text + "\n")
        self.out.flush()

    def status(self, text: str) -> None:
        if self.live:
            self.out.write("\r\033[K" + text)
            self.out.flush()

    def end_status(self, text: str) -> None:
        if self.live:
            self.out.write("\r\033[K")
        self.say(text)


def _bar(done: float, total: float, width: int = 20) -> str:
    filled = min(width, round(width * done / total))
    return "█" * filled + "░" * (width - filled)


def _face_note(tracker: look.LookTracker) -> str:
    seen = tracker.readings(time.monotonic() - 0.6)
    return "✓ I can see you" if any(yaw is not None for _, _, yaw in seen) else "✗ can't see your face"


def _step(screen: _Screen, tracker: look.LookTracker, n: int, title: str, lines: list[str],
          seconds: float, collect: Callable[[], None]) -> float:
    screen.say(f"\nStep {n} of 3 · {title}")
    for line in lines:
        screen.say(f"  {line}")
    for left in range(COUNTDOWN_S, 0, -1):
        tracker.wake()
        screen.status(f"  Starting in {left}…")
        time.sleep(1.0)
    start = time.monotonic()
    while (elapsed := time.monotonic() - start) < seconds:
        tracker.wake()
        if elapsed >= SETTLE_S:
            collect()
        screen.status(f"  {_bar(elapsed, seconds)} {seconds - elapsed:3.0f}s   {_face_note(tracker)}")
        time.sleep(TICK_S)
    screen.end_status(f"  {_bar(1, 1)} done")
    return start


def run(camera: str, save: Callable[[Tuning], str], *, ask: Callable[[str], str] = input,
        screen: _Screen | None = None, mic: Mic | None = None,
        save_voice: Callable[[np.ndarray], str] | None = None, has_voice: bool = False) -> int:
    screen = screen or _Screen()
    screen.say("Look to talk setup")
    screen.say("Omi will learn what you look like when you talk to it. It takes about 20 seconds,")
    screen.say("and nothing is recorded: only three numbers are kept.")
    screen.say("\nSit the way you normally do, facing the screen you use.")
    try:
        ask("Press Enter to start (Ctrl+C to cancel) ")
    except EOFError:
        pass

    tracker = look.LookTracker(camera, mic_users=lambda: [], measure_all=True)
    mic = mic or Mic()
    try:
        mic.start()
        tracker.start()
        tracker.wake()
        screen.status("Finding your face…")
        deadline = time.monotonic() + FIND_FACE_S
        while time.monotonic() < deadline:
            tracker.wake()
            if any(yaw is not None for _, _, yaw in tracker.readings(time.monotonic() - 1.0)):
                break
            time.sleep(TICK_S)
        else:
            screen.end_status("")
            why = f" ({tracker.error})" if tracker.error else ""
            raise TuneError(f"I can't see a face{why}. Check the camera is plugged in and not used by "
                            "another app, sit in front of it with some light, then try again.")
        screen.end_status("✓ Found you.")

        quiet: list[float] = []
        talking: list[float] = []

        def spread_into(bucket: list[float]) -> Callable[[], None]:
            def collect() -> None:
                spread = tracker.lips_moving()
                if spread is not None:
                    bucket.append(spread)
            return collect

        t1 = _step(screen, tracker, 1, "Stay quiet",
                   ["Look at the screen and keep your mouth still."], QUIET_S, spread_into(quiet))
        t2 = _step(screen, tracker, 2, "Talk to Omi",
                   ["Keep looking at the screen and read this out loud, at your normal pace:",
                    f'"{SENTENCE}"'], TALK_S, spread_into(talking))
        quiet_audio = mic.between(t1 + SETTLE_S, t1 + QUIET_S)
        talk_audio = mic.between(t2, t2 + TALK_S)
        facing_yaws = [yaw for _, _, yaw in tracker.readings(t1 + SETTLE_S) if yaw is not None]
        t3 = _step(screen, tracker, 3, "Look away",
                   ["Turn your head to look at something beside you,",
                    "as if someone in the room were talking to you."], AWAY_S, lambda: None)
        away_yaws = [yaw for _, _, yaw in tracker.readings(t3 + SETTLE_S) if yaw is not None]
    except KeyboardInterrupt:
        screen.end_status("\nCancelled. Nothing was changed.")
        return 130
    except TuneError as exc:
        screen.say(f"\n✗ {exc}")
        return 1
    finally:
        tracker.stop()
        mic.stop()

    heard = heard_you(quiet_audio, talk_audio)
    try:
        tuning = choose(quiet, talking, facing_yaws, away_yaws)
    except TuneError as exc:
        screen.say(f"\n✗ {exc}\nNothing was changed. Run `omi look-tune` again when ready.")
        return 1
    where = save(tuning)
    screen.say("\n✓ All set! Look to talk now fits you.")
    screen.say(f"  Talking:  your lips move about {tuning.talking:.3f} (still: {tuning.quiet:.3f}); "
               f"anything above {tuning.lip_threshold:.3f} counts.")
    side = "straight at the camera" if abs(tuning.yaw_center) < 0.1 else "off to one side of the camera"
    screen.say(f"  Facing:   your screen is {side}; Omi now knows which way that is.")
    if heard:
        screen.say("  Mic:      ✓ I heard you clearly.")
    else:
        screen.say("  Mic:      ✗ I couldn't hear you. Check the right microphone is picked and not muted")
        screen.say("            (bar panel → Settings, or your system sound settings).")
    screen.say(f"  Saved {where}.")
    if heard and save_voice is not None:
        _offer_voice(screen, ask, save_voice, talk_audio, has_voice)
    screen.say("\nCheck it any time with `omi look-test`, or run `omi look-tune` again if you move")
    screen.say("your camera or desk.")
    return 0


def _offer_voice(screen: _Screen, ask: Callable[[str], str], save_voice: Callable[[np.ndarray], str],
                 talk_audio: np.ndarray, has_voice: bool) -> None:
    screen.say("\nOnly my voice (optional)")
    screen.say("  Omi can learn your voice from what you just read, then ignore other people,")
    screen.say("  the TV and videos. Your voiceprint stays on this PC and is deleted if you turn")
    screen.say("  this off. Short words like \"stop\" always work.")
    question = "  Update your saved voice? [Y/n] " if has_voice else "  Only answer your voice? [y/N] "
    try:
        answer = ask(question).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if not (answer in ("y", "yes") or (has_voice and answer == "")):
        screen.say("  Skipped. Turn it on later by running `omi look-tune` again.")
        return
    try:
        screen.say("  " + save_voice(talk_audio))
    except Exception as exc:
        screen.say(f"  ✗ Couldn't save your voice ({exc}). Run `omi look-tune` again and read a bit louder.")
