"""Omi tunes look to talk with you, out loud: no terminal.

Start it from the Tune button in Settings, or ask Omi ("tune look to talk"). Omi says
each step and shows it in its speech bubble: stay quiet, read a sentence, look away.
Then it says what it learned, asks by voice (or the Yes/No buttons) whether it should
only answer your voice, and offers to turn look to talk on. The numbers come from
omi.look_tune.choose, the same as `omi look-tune` in a terminal.

The hands-free listener rests while this runs, so reading the sentence does not start
a request and the webcam is free for the setup.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

import numpy as np

from omi import look, look_tune
from omi.session import parse_answer

log = logging.getLogger(__name__)

FACE_LOST_S = 2.5  # tell the user once when the camera loses them this long
ANSWER_S = 8.0  # how long to listen for a spoken yes or no
IDLE_WAIT_S = 15.0  # let Omi finish what it is saying before the setup starts
TITLE = "Tune look to talk"


class GuidedSetup:
    """One guided run at a time, driven by the daemon."""

    def __init__(self, daemon: Any) -> None:
        self.daemon = daemon
        self.task: asyncio.Task | None = None
        self.title = TITLE
        self.saved = False  # the new numbers are in, so stopping now keeps them

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self) -> str:
        if self.running:
            return "The look to talk setup is already running."
        self.task = asyncio.create_task(self._guard(), name="omi-look-setup")
        return "Starting the look to talk setup now."

    async def cancel(self) -> None:
        if self.running:
            assert self.task is not None
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.task

    # --- the conversation ---------------------------------------------------------

    @property
    def session(self) -> Any:
        return self.daemon.session

    async def say(self, text: str, shown: str | None = None) -> None:
        await self.card(shown or text)
        await self.session.announce(text, shown)

    async def card(self, text: str, note: str = "", seconds: float = 0.0, total: float = 0.0) -> None:
        """The large card in the middle of the screen that stays up for the whole setup,
        so each prompt can be read while looking at the screen."""
        await self.daemon.broadcast({"event": "tune", "active": True, "title": self.title, "text": text,
                                     "note": note, "seconds": seconds, "total": total or seconds})

    async def step_label(self, text: str) -> None:
        await self.daemon.broadcast({"event": "progress", "text": text})

    async def _guard(self) -> None:
        # Wait for Omi to finish its own reply ("Sure, let's do it") before starting.
        deadline = time.monotonic() + IDLE_WAIT_S
        while self.session.state.value != "idle" and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        await self.session.finish_speaking()
        wake = self.daemon.wake
        was_running = bool(wake is not None and wake.active)
        if was_running:
            await wake.stop()
        self.saved = False
        try:
            await self._run()
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await self.card("Stopped. Your new tuning is saved." if self.saved
                                else "Cancelled. Nothing was changed.")
            raise
        except Exception:
            log.exception("look to talk setup failed")
            with contextlib.suppress(Exception):
                await self.say("Sorry, the setup hit a problem. Nothing was changed.")
        finally:
            if was_running:
                with contextlib.suppress(Exception):
                    await wake.start()
            self.title = TITLE
            with contextlib.suppress(Exception):
                await self.daemon.broadcast({"event": "progress", "text": ""})
                await self.daemon.broadcast({"event": "tune", "active": False})

    async def _run(self) -> None:
        cfg = self.daemon.cfg
        loop = asyncio.get_running_loop()
        tracker = look.LookTracker(cfg.wake.camera, mic_users=lambda: [], measure_all=True)
        mic = look_tune.Mic(await asyncio.to_thread(look_tune.mic_source, cfg.audio.source))
        await self.say("Let's tune look to talk. It takes about twenty seconds. "
                       "Sit the way you normally do and look at your screen.")
        try:
            await loop.run_in_executor(None, tracker.start)
            mic.start()
            if not await self._find_face(tracker):
                why = f" ({tracker.error})" if tracker.error else ""
                await self.say("I can't see your face. Check the camera is plugged in and has some light, "
                               "then ask me to tune look to talk again.",
                               f"I can't see your face{why}. Check the camera and the light, then try again.")
                return
            quiet: list[float] = []
            talking: list[float] = []
            t1 = await self._step(tracker, 1, "Stay quiet",
                                  "First, keep looking at the screen and stay quiet for a few seconds.",
                                  None, look_tune.QUIET_S, quiet)
            t2 = await self._step(tracker, 2, "Talk to Omi",
                                  "Now keep looking at the screen and read the sentence on screen out loud.",
                                  f'Read this out loud:\n"{look_tune.SENTENCE}"', look_tune.TALK_S, talking)
            facing_yaws = [y for _, _, y in tracker.readings(t1 + look_tune.SETTLE_S) if y is not None]
            t3 = await self._step(tracker, 3, "Look away",
                                  "Last one. Turn your head and look at something beside you, "
                                  "as if someone in the room were talking to you.",
                                  None, look_tune.AWAY_S, None)
            away_yaws = [y for _, _, y in tracker.readings(t3 + look_tune.SETTLE_S) if y is not None]
            quiet_audio = mic.between(t1 + look_tune.SETTLE_S, t1 + look_tune.QUIET_S)
            talk_audio = mic.between(t2, t2 + look_tune.TALK_S)
        finally:
            mic.stop()
            await loop.run_in_executor(None, tracker.stop)
        self.title = TITLE

        heard = look_tune.heard_you(quiet_audio, talk_audio)
        try:
            tuning = look_tune.choose(quiet, talking, facing_yaws, away_yaws)
        except look_tune.TuneError as exc:
            hint = "" if heard else " I also couldn't hear you, so check your microphone."
            await self.say(f"{exc}{hint} Nothing was changed.")
            return
        for key in ("lip_threshold", "yaw_center", "yaw_range"):
            await self.daemon._set(f"wake.{key}", str(getattr(tuning, key)), save=True)
        self.saved = True
        await self.say("All done. Look to talk now fits your face and camera.",
                       f"✓ Look to talk is tuned. Lips: talking {tuning.talking:.3f}, still {tuning.quiet:.3f}, "
                       f"counts from {tuning.lip_threshold:.3f}.")
        if not heard:
            await self.say("But I couldn't hear you while you read. Check the right microphone is picked "
                           "and not muted.")
        elif await self._ask("Want me to only answer your voice? I'd keep a voiceprint on this computer, "
                             "and short words like stop always work."):
            await self._save_voice(talk_audio)
        if not cfg.wake.look_to_talk and await self._ask("Should I turn look to talk on now?"):
            await self.daemon._set("wake.look_to_talk", "on", save=True)
            await self.say("Look to talk is on. Just face the screen and talk to me.")

    async def _find_face(self, tracker: look.LookTracker) -> bool:
        deadline = time.monotonic() + look_tune.FIND_FACE_S
        while time.monotonic() < deadline:
            tracker.wake()
            if any(y is not None for _, _, y in tracker.readings(time.monotonic() - 1.0)):
                return True
            await asyncio.sleep(look_tune.TICK_S)
        return False

    async def _step(self, tracker: look.LookTracker, n: int, title: str, spoken: str, shown: str | None,
                    seconds: float, bucket: list[float] | None) -> float:
        self.title = f"Step {n} of 3 · {title}"
        await self.step_label(self.title)
        tracker.wake()
        body = shown or spoken
        await self.card(body)
        await self.session.announce(spoken, f"{self.title}\n{body}")
        await self.card(body, seconds=seconds)  # the clock starts once Omi stops talking
        start = time.monotonic()
        last_face = start
        warned = False
        while (elapsed := time.monotonic() - start) < seconds:
            tracker.wake()
            now = time.monotonic()
            if any(y is not None for _, _, y in tracker.readings(now - 0.6)):
                last_face = now
            elif bucket is not None and not warned and now - last_face >= FACE_LOST_S:
                warned = True
                await self.step_label("✗ I can't see your face: move back into view")
                await self.card(body, "I can't see your face: move back into view",
                                max(0.0, seconds - (now - start)), seconds)
            if bucket is not None and elapsed >= look_tune.SETTLE_S:
                spread = tracker.lips_moving()
                if spread is not None:
                    bucket.append(spread)
            await asyncio.sleep(look_tune.TICK_S)
        return start

    async def _ask(self, question: str) -> bool:
        """Ask out loud; a spoken yes or no, or the Yes/No buttons, answers it."""
        await self.card(question)
        answer = asyncio.create_task(self.session.confirm(question))
        while self.session.pending_confirm is None and not answer.done():
            await asyncio.sleep(0.02)
        await self.session.finish_speaking()
        listen = asyncio.create_task(self._listen_for_answer())
        try:
            return await answer
        finally:
            listen.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await listen

    async def _listen_for_answer(self) -> None:
        from omi.audio.vad import EndpointDetector

        cfg = self.daemon.cfg
        mic = look_tune.Mic(await asyncio.to_thread(look_tune.mic_source, cfg.audio.source))
        mic.start()
        loop = asyncio.get_running_loop()
        try:
            while self.session.pending_confirm is not None:
                vad = EndpointDetector(silence_ms=700)
                start = seen = time.monotonic()
                while time.monotonic() - start < ANSWER_S and not vad.ended:
                    await asyncio.sleep(0.1)
                    now = time.monotonic()
                    vad.feed(mic.between(seen, now))
                    seen = now
                if not vad.ended:
                    continue
                pcm = mic.between(start, time.monotonic())
                text = await loop.run_in_executor(None, self.session.stt.transcribe, pcm)
                verdict = parse_answer(text or "")
                if verdict is not None:
                    self.session.answer(verdict)
                    return
        finally:
            mic.stop()

    async def _save_voice(self, talk_audio: np.ndarray) -> None:
        from omi import voice_id

        if not voice_id.MODEL_PATH.is_file():
            await self.say("One moment, I'm getting the voice model. It's a one-time download.")
        try:
            await asyncio.to_thread(voice_id.ensure_model)
            print_ = await asyncio.to_thread(voice_id.enroll, talk_audio, voice_id.VoiceId())
        except ValueError:
            await self.say("I didn't hear enough of your voice to learn it. Try the setup again and read a bit louder.")
            return
        import getpass

        try:
            name = getpass.getuser()
        except Exception:
            name = "me"
        self.daemon._voices().save(name, print_)
        await self.daemon._set("wake.only_my_voice", "on", save=True)
        await self.say("Got it. I'll only answer your voice now.")


def look_setup_tool(setup: GuidedSetup | None):
    """Lets the talker start the setup when asked ("tune look to talk")."""
    from omi.tools.base import Tool, ToolResult

    async def tune_look_to_talk() -> ToolResult:
        if setup is None:
            return ToolResult(False, "The look to talk setup is not available.")
        return ToolResult(True, setup.start() + " Say one short line, then let the setup talk.")

    return Tool(
        name="tune_look_to_talk",
        description="Start the guided setup that fits look to talk (talking to Omi by facing the screen, "
                    "no wake word) to the user's face, camera and voice, and can set 'only my voice'. Use when "
                    "they ask to set up, tune or calibrate look to talk, the webcam, or only answering them.",
        parameters={"type": "object", "properties": {}},
        safety="reversible",
        fn=tune_look_to_talk,
    )


def camera_tool(daemon: Any):
    """Lets the talker switch the look to talk webcam when asked ("use the desk camera")."""
    from omi.tools.base import Tool, ToolResult

    async def choose_camera(camera: str = "") -> ToolResult:
        if daemon is None:
            return ToolResult(False, "Choosing a camera is not available.")
        cameras = daemon._cameras()
        names = ", ".join(c["name"] for c in cameras) or "none found"
        want = camera.strip().lower()
        if not want:
            current = next((c["name"] for c in cameras if c["id"] == daemon.cfg.wake.camera), "automatic")
            return ToolResult(True, f"Using {current}. Cameras: {names}.", {"cameras": cameras})
        if want in ("auto", "automatic", "default", "first"):
            pick = ""
        else:
            hits = [c for c in cameras if want in (c["id"].lower(), c["name"].lower())] or \
                   [c for c in cameras if want in c["name"].lower() or want in c["id"].lower()]
            if len(hits) != 1:
                why = "more than one camera matches" if hits else "no camera matches"
                return ToolResult(False, f"{why} '{camera}'. Cameras: {names}.", {"cameras": cameras})
            pick = hits[0]["id"]
        await daemon._set("wake.camera", pick or "auto", save=True)
        chosen = next((c["name"] for c in cameras if c["id"] == pick), "the first camera")
        return ToolResult(True, f"Look to talk now uses {chosen}.")

    return Tool(
        name="choose_camera",
        description="List the webcams or pick which one look to talk uses. Leave camera empty to hear the "
                    "choices; give part of a camera's name, or 'auto' for the first camera.",
        parameters={"type": "object", "properties": {
            "camera": {"type": "string", "description": "Part of the camera's name, its id, or 'auto'."}}},
        safety="reversible",
        fn=choose_camera,
    )
