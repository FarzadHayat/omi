"""The `omi` command: runs the daemon, or talks to it over its Unix socket."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from omi import config

EXIT_ERROR = 1
EXIT_NOT_RUNNING = 2


class NotRunning(Exception):
    pass


class Incomplete(Exception):
    """The daemon closed the connection before its final reply."""


def _connect(path: Path) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(path))
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        sock.close()
        raise NotRunning from exc
    return sock


def request(path: Path, req: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Send one request; yield every line the daemon sends back until the final one
    (for turns, the line with "done"; for subscribe, until the daemon goes away)."""
    sock = _connect(path)
    with sock, sock.makefile("rb") as fh:
        sock.sendall((json.dumps(req) + "\n").encode())
        streaming = req["cmd"] in ("say", "wav", "ptt_stop")
        for raw in fh:
            msg = json.loads(raw)
            yield msg
            if req["cmd"] == "subscribe":
                continue
            if "event" in msg:
                continue
            if not streaming or msg.get("done") or not msg.get("ok", False):
                return
        if req["cmd"] != "subscribe":
            raise Incomplete


def _timing_line(t: dict[str, Any]) -> str:
    parts = [f"path={t.get('path')}"]
    for key in ("stt_ms", "route_ms", "first_text_ms", "first_audio_ms", "action_ms", "brain_done_ms", "total_ms"):
        if t.get(key) is not None:
            parts.append(f"{key[:-3]}={t[key]:.0f}ms")
    return "timing: " + " ".join(parts)


def _print_human(msg: dict[str, Any]) -> None:
    ev = msg.get("event")
    if ev == "transcript":
        who = "you" if msg["role"] == "user" else "omi"
        print(f"{who}: {msg['text']}")
    elif ev == "tool" and msg.get("phase") == "end":
        mark = "ok" if msg.get("ok") else "failed"
        print(f"[{msg['name']} {mark}] {msg.get('text', '')}")
    elif ev == "confirm_request":
        print(f"confirm? {msg['question']}  (answer with: omi yes / omi no)")
        if msg.get("detail"):
            print(f"  full command: {msg['detail']}")
    elif ev == "progress":
        print(f"... {msg['text']}")
    elif ev is None and msg.get("done"):
        if msg.get("cancelled"):
            print("(cancelled)")
        if msg.get("timing"):
            print(_timing_line(msg["timing"]))


def _print_status(msg: dict[str, Any]) -> None:
    engines = msg.get("engines") or {}
    voice = f" ({engines['voice']})" if engines.get("voice") else ""
    tmodel = f" ({msg['talker_model']})" if msg.get("talker_model") else ""
    print(f"state: {msg.get('state')}{'' if msg.get('ready', True) else ' (warming up)'}")
    print(f"talker: {msg.get('talker')}{tmodel}  worker: {msg.get('worker')}")
    print(f"stt: {engines.get('stt')}  tts: {engines.get('tts')}{voice}")
    wake = msg.get("wake") or {}
    if wake:
        phrases = ", ".join(wake.get("phrases") or [])
        state = f"on ({phrases})" if wake.get("enabled") else ("starting" if wake.get("configured") and not msg.get("ready", True) else "off")
        print(f"wake: {state}")
        if wake.get("look_to_talk"):
            by = wake.get("look_paused_by")
            print(f"look to talk: paused while {by} uses the mic" if by else "look to talk: on")
    print(f"session: {msg.get('session_id')}")
    if msg.get("confirm"):
        print(f"waiting for yes or no: {msg['confirm']['question']}")
    for name, error in (msg.get("errors") or {}).items():
        print(f"problem with {name}: {error}")
    if msg.get("timing"):
        print(_timing_line(msg["timing"]))


def _build_request(args: argparse.Namespace) -> dict[str, Any]:
    c = args.command
    if c == "say":
        req: dict[str, Any] = {"cmd": "say", "text": " ".join(args.text)}
    elif c == "wav":
        req = {"cmd": "wav", "path": str(Path(args.path).resolve())}
        if args.out:
            req["out"] = str(Path(args.out).resolve())
    elif c == "ptt":
        req = {"cmd": f"ptt_{args.action}"}
    elif c in ("yes", "no"):
        req = {"cmd": "confirm", "answer": c == "yes"}
    elif c == "set":
        req = {"cmd": "set", "key": args.key, "value": args.value}
        if getattr(args, "save", False):
            req["save"] = True
    elif c == "point":
        req = {"cmd": "mascot", "action": "point", "x": args.x, "y": args.y,
               "label": " ".join(args.label), "duration": args.duration}
    elif c == "events":
        req = {"cmd": "subscribe"}
    else:
        req = {"cmd": c}
    if getattr(args, "quiet", False):
        req["quiet"] = True
    return req


def _parser() -> argparse.ArgumentParser:
    # The global flags work before or after the subcommand. Each parser gets its own
    # copy: argparse shares action objects between parents and children.
    def flags(top: bool) -> argparse.ArgumentParser:
        c = argparse.ArgumentParser(add_help=False)
        dflt = (lambda v: v) if top else (lambda v: argparse.SUPPRESS)
        c.add_argument("--socket", type=Path, default=dflt(None), help="daemon socket path")
        c.add_argument("--json", action="store_true", default=dflt(False), help="print raw JSON lines")
        c.add_argument("--quiet", action="store_true", default=dflt(False), help="no audio for this turn")
        return c

    p = argparse.ArgumentParser(prog="omi", description="Omi voice assistant and desktop companion.", parents=[flags(True)])
    sub = p.add_subparsers(dest="command", required=True)
    _add = sub.add_parser

    def add_parser(name: str, **kw: Any) -> argparse.ArgumentParser:
        return _add(name, parents=[flags(False)], **kw)

    sub.add_parser = add_parser  # type: ignore[method-assign]
    sub.add_parser("daemon", help="run the daemon in the foreground")
    say = sub.add_parser("say", help="run a turn from text")
    say.add_argument("text", nargs="+")
    wav = sub.add_parser("wav", help="run a turn from a 16 kHz wav")
    wav.add_argument("path")
    wav.add_argument("--out", help="write the reply audio to this wav instead of playing it")
    ptt = sub.add_parser("ptt", help="push-to-talk")
    ptt.add_argument("action", choices=["start", "stop"])
    sub.add_parser("stop", help="stop speaking and cancel the running turn")
    sub.add_parser("status", help="show daemon state")
    sub.add_parser("yes", help="approve the pending confirmation")
    sub.add_parser("no", help="refuse the pending confirmation")
    st = sub.add_parser("set", help="change a setting live (brain.*, stt.engine, tts.*, audio.sounds, wake.*)")
    st.add_argument("key")
    st.add_argument("value")
    st.add_argument("--save", action="store_true", help="also persist this setting to config.toml")
    sub.add_parser("providers", help="list supported AI providers and which CLIs are installed")
    sub.add_parser("engines", help="list local and cloud STT/TTS engines and whether each is ready")
    pt = sub.add_parser("point", help="show an arrow from Omi pointing at (x, y) on screen")
    pt.add_argument("x", type=int)
    pt.add_argument("y", type=int)
    pt.add_argument("label", nargs="*")
    pt.add_argument("--duration", type=int, default=6)
    sub.add_parser("events", help="print every daemon event as JSON lines")
    lt = sub.add_parser("look-test", help="watch the webcam live: facing, lip movement, and whether look to talk would fire")
    lt.add_argument("seconds", type=float, nargs="?", default=30.0)
    tune = sub.add_parser("look-tune", help="20-second guided setup that fits look to talk to your face and camera")
    tune.add_argument("--wait", action="store_true", help="wait for Enter at the end (when opened from the panel)")
    vo = sub.add_parser("voice", help="the voices Omi answers when 'only my voice' is on")
    vo.add_argument("action", choices=["list", "add", "forget"])
    vo.add_argument("name", nargs="?", default="")
    for name, what in (("install", "add the Omarchy keys, mascot widget and service"),
                       ("uninstall", "remove what omi install added")):
        sp = sub.add_parser(name, help=what)
        sp.add_argument("--no-activate", action="store_true",
                        help="only write or remove the files; leave systemd, the shell and Hyprland alone")
    return p


def _run_daemon() -> int:
    import asyncio

    from omi import app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request drowns the log

    async def main() -> None:
        daemon = await app.build(config.load())
        await daemon.run()

    asyncio.run(main())
    return 0


def _print_providers(sock_path: Path, *, as_json: bool = False) -> int:
    from omi.brain.cli_providers import available_providers

    cfg = config.load()
    rows = available_providers(cfg)
    try:
        for msg in request(sock_path, {"cmd": "status"}):
            if msg.get("ok"):
                live_talker = msg.get("talker")
                live_worker = msg.get("worker")
                for r in rows:
                    r["active_talker"] = r["name"] == live_talker
                    r["active_worker"] = r["name"] == live_worker
                break
    except (NotRunning, ConnectionError, Incomplete, json.JSONDecodeError):
        pass
    if as_json:
        print(json.dumps(rows))
        return 0
    for r in rows:
        status = "installed" if r["installed"] else "not found"
        tags = []
        if r["active_talker"]:
            tags.append("talker")
        if r["active_worker"]:
            tags.append("worker")
        active = f" [{', '.join(tags)}]" if tags else ""
        print(f"{r['name']:10} {status:10} model={r['model']:24}{active}  - {r['auth']}")
    return 0


def _print_engines(sock_path: Path, *, as_json: bool = False) -> int:
    from omi.stt import available_stt_engines
    from omi.tts import available_tts_engines

    cfg = config.load()
    stt_rows = available_stt_engines(cfg.stt)
    tts_rows = available_tts_engines(cfg.tts)
    active_stt, active_tts = cfg.stt.engine, cfg.tts.engine
    try:
        for msg in request(sock_path, {"cmd": "status"}):
            if msg.get("ok"):
                eng = msg.get("engines") or {}
                active_stt = eng.get("stt") or active_stt
                active_tts = eng.get("tts") or active_tts
                if msg.get("stt_engines"):
                    stt_rows = msg["stt_engines"]
                if msg.get("tts_engines"):
                    tts_rows = msg["tts_engines"]
                break
    except (NotRunning, ConnectionError, Incomplete, json.JSONDecodeError):
        pass
    if as_json:
        print(json.dumps({"active": {"stt": active_stt, "tts": active_tts}, "stt": stt_rows, "tts": tts_rows}))
        return 0
    print("STT engines:")
    for r in stt_rows:
        st = "ready" if r["installed"] else "not set up"
        mark = " [active]" if r["name"] == active_stt else ""
        print(f"  {r['name']:11} {st:11} {r['detail']}{mark}")
    print("TTS engines:")
    for r in tts_rows:
        st = "ready" if r["installed"] else "not set up"
        mark = " [active]" if r["name"] == active_tts else ""
        print(f"  {r['name']:11} {st:11} {r['detail']}{mark}")
    return 0


def _pending_question(path: Path) -> dict[str, Any] | None:
    for msg in request(path, {"cmd": "status"}):
        return msg.get("confirm")
    return None


def _daemon_set(key: str, value: str, save: bool = False) -> bool:
    req = {"cmd": "set", "key": key, "value": value, "save": save}
    try:
        return any(m.get("ok") for m in request(config.SOCKET_PATH, req) if "event" not in m)
    except (NotRunning, OSError, ValueError, Incomplete):
        return False


def _daemon_look_on() -> bool:
    try:
        for msg in request(config.SOCKET_PATH, {"cmd": "status"}):
            if "event" not in msg:
                return bool((msg.get("wake") or {}).get("look_to_talk"))
    except (NotRunning, OSError, ValueError, Incomplete):
        pass
    return False


def _look_test(seconds: float, camera: str) -> int:
    """Print what look to talk sees twice a second, to check the lip threshold by hand."""
    import time

    from omi import look

    # A webcam serves one app at a time: Omi's own look to talk rests during the test.
    paused = _daemon_look_on() and _daemon_set("wake.look_to_talk", "off")
    if paused:
        print("Omi's look to talk is paused during the test.")
    try:
        return _look_test_run(seconds, camera, look, time)
    finally:
        if paused:
            _daemon_set("wake.look_to_talk", "on")


def _look_tune(wait: bool = False) -> int:
    try:
        return _look_tune_run()
    finally:
        if wait:
            try:
                input("\nPress Enter to close.")
            except (EOFError, KeyboardInterrupt):
                pass


def _look_tune_run() -> int:
    # Omi itself guides the setup when it is running; the terminal version is the fallback.
    try:
        for msg in request(config.SOCKET_PATH, {"cmd": "look_tune"}):
            if "event" not in msg and msg.get("ok"):
                print(f"{msg.get('text', 'Starting.')} Omi will talk you through it.")
                return 0
    except (NotRunning, OSError, ValueError, Incomplete):
        pass
    return _look_tune_terminal()


def _look_tune_terminal() -> int:
    from omi import look_tune

    was_on = _daemon_look_on()
    paused = was_on and _daemon_set("wake.look_to_talk", "off")

    def save(t: look_tune.Tuning) -> str:
        values = {"lip_threshold": t.lip_threshold, "yaw_center": t.yaw_center, "yaw_range": t.yaw_range}
        if all(_daemon_set(f"wake.{k}", f"{v:g}", save=True) for k, v in values.items()):
            return "to your Omi settings (already in use)"
        for k, v in values.items():
            config.save_override("wake", k, v, config.CONFIG_PATH)
        return f"to {config.CONFIG_PATH}"

    from omi.voice_id import Voiceprints

    me = _me()
    cfg = config.load()
    try:
        code = look_tune.run(cfg.wake.camera, save, mic=look_tune.Mic(look_tune.mic_source(cfg.audio.source)),
                             save_voice=lambda pcm: _save_voice(me, pcm), has_voice=me in Voiceprints().names())
    finally:
        if paused:
            _daemon_set("wake.look_to_talk", "on")
    if code == 0 and not was_on and _daemon_look_on() is False and sys.stdin.isatty():
        try:
            yes = input("\nTurn on Look to talk now? [Y/n] ").strip().lower() in ("", "y", "yes")
        except (EOFError, KeyboardInterrupt):
            yes = False
        if yes and _daemon_set("wake.look_to_talk", "on", save=True):
            print("✓ Look to talk is on. Just face the screen and talk to Omi.")
        elif yes:
            print("Omi isn't running. Start it, then turn Look to talk on from the bar panel.")
    return code


def _me() -> str:
    import getpass

    try:
        return getpass.getuser()
    except Exception:
        return "me"


def _save_voice(name: str, pcm) -> str:  # noqa: ANN001
    from omi import voice_id

    if not voice_id.MODEL_PATH.is_file():
        print("  Downloading the voice model (40 MB, only once)…", flush=True)
    voice_id.ensure_model()
    voice_id.Voiceprints().save(name, voice_id.enroll(pcm, voice_id.VoiceId()))
    if not _daemon_set("wake.only_my_voice", "on", save=True):
        config.save_override("wake", "only_my_voice", True, config.CONFIG_PATH)
    return f"✓ Saved as \"{name}\". Omi now only answers your voice. Add others with `omi voice add NAME`."


def _voice(action: str, name: str) -> int:
    from omi import look_tune, voice_id

    prints = voice_id.Voiceprints()
    if action == "list":
        names = prints.names()
        print("\n".join(names) if names else "No voices saved. `omi look-tune` saves yours.")
        return 0
    if action == "forget":
        if not prints.forget(name or None):
            print(f"omi: no saved voice called {name}" if name else "No voices saved.", file=sys.stderr)
            return EXIT_ERROR
        if not prints.names():
            _daemon_set("wake.only_my_voice", "off", save=True)
            print("Forgotten. No voices are left, so Omi answers everyone again.")
        else:
            print(f"Forgot {name}.")
        return 0
    if not name:
        print("omi: say whose voice it is, like: omi voice add sam", file=sys.stderr)
        return EXIT_ERROR
    screen = look_tune._Screen()
    screen.say(f"Saving {name}'s voice. {name}, read this out loud at your normal pace:")
    screen.say(f'  "{look_tune.SENTENCE}"')
    mic = look_tune.Mic(look_tune.mic_source(config.load().audio.source))
    mic.start()
    try:
        for left in range(look_tune.COUNTDOWN_S, 0, -1):
            screen.status(f"  Starting in {left}…")
            time.sleep(1.0)
        start = time.monotonic()
        while (elapsed := time.monotonic() - start) < look_tune.TALK_S:
            screen.status(f"  {look_tune._bar(elapsed, look_tune.TALK_S)} {look_tune.TALK_S - elapsed:3.0f}s")
            time.sleep(look_tune.TICK_S)
        screen.end_status(f"  {look_tune._bar(1, 1)} done")
        pcm = mic.between(start, time.monotonic())
    except KeyboardInterrupt:
        print("\nCancelled. Nothing was saved.")
        return 130
    finally:
        mic.stop()
    try:
        print(_save_voice(name, pcm).replace("your voice", "saved voices"))
    except ValueError:
        print("✗ I didn't hear enough. Check the microphone and try again, reading a bit louder.")
        return EXIT_ERROR
    return 0


def _look_untuned() -> bool:
    from omi.config import WakeConfig

    w, d = config.load().wake, WakeConfig()
    return (w.lip_threshold, w.yaw_center, w.yaw_range) == (d.lip_threshold, d.yaw_center, d.yaw_range)


def _look_test_run(seconds: float, camera: str, look, time) -> int:  # noqa: ANN001
    w = config.load().wake
    tracker = look.LookTracker(camera, mic_users=lambda: [], lip_threshold=w.lip_threshold,
                               yaw_center=w.yaw_center, yaw_range=w.yaw_range)
    try:
        tracker.start()
    except Exception as exc:
        print(f"omi: look test could not start: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"lips count as moving at spread >= {tracker.lip_threshold:.3f}. Ctrl+C to stop.")
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            tracker.wake()
            time.sleep(0.5)
            spread = tracker.lips_moving()
            lips = "unseen" if spread is None else f"{spread:.3f}"
            fire = "yes" if tracker.addressed() else "no"
            note = f"  ({tracker.error})" if tracker.error else ""
            print(f"facing {'yes' if tracker.looking() else 'no ':3}  lips {lips:>6}  would fire {fire}{note}",
                  flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "daemon":
        return _run_daemon()
    if args.command == "look-test":
        return _look_test(args.seconds, config.load().wake.camera)
    if args.command == "look-tune":
        return _look_tune(args.wait)
    if args.command == "voice":
        return _voice(args.action, args.name)
    if args.command == "providers":
        return _print_providers(args.socket or config.SOCKET_PATH, as_json=args.json)
    if args.command == "engines":
        return _print_engines(args.socket or config.SOCKET_PATH, as_json=args.json)
    if args.command in ("install", "uninstall"):
        from omi import install

        return install.main(args.command, activate=not args.no_activate)
    path = args.socket or config.SOCKET_PATH
    req = _build_request(args)
    status = 0
    try:
        if req["cmd"] == "confirm":
            # Answer exactly the question that is waiting now, and say which one it was.
            pending = _pending_question(path)
            if pending is None:
                print("omi: nothing to confirm", file=sys.stderr)
                return EXIT_ERROR
            req["id"] = pending["id"]
            if not args.json:
                print(f"{'yes' if req['answer'] else 'no'} to: {pending['question']}")
        for msg in request(path, req):
            if args.json or args.command == "events":
                print(json.dumps(msg), flush=True)
            elif args.command == "status" and msg.get("ok"):
                _print_status(msg)
            else:
                _print_human(msg)
            if "event" not in msg and msg.get("ok") is False:
                print(f"omi: {msg.get('error', 'request failed')}", file=sys.stderr)
                status = EXIT_ERROR
    except NotRunning:
        print(f"omi: the daemon is not running (no socket at {path}). Start it with: omi daemon", file=sys.stderr)
        return EXIT_NOT_RUNNING
    except KeyboardInterrupt:
        return 130
    except (ConnectionError, json.JSONDecodeError) as exc:
        print(f"omi: lost the daemon connection: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Incomplete:
        print("omi: the daemon closed the connection without an answer", file=sys.stderr)
        return EXIT_ERROR
    if (status == 0 and args.command == "set" and args.key == "wake.look_to_talk"
            and args.value.lower() in ("on", "true", "1", "yes") and _look_untuned()):
        print("Tip: run `omi look-tune` once (20 seconds) so Look to talk fits your face and camera.")
    return status


if __name__ == "__main__":
    sys.exit(main())
