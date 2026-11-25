"""The `hark` command: runs the daemon, or talks to it over its Unix socket."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hark import config

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
        who = "you" if msg["role"] == "user" else "hark"
        print(f"{who}: {msg['text']}")
    elif ev == "tool" and msg.get("phase") == "end":
        mark = "ok" if msg.get("ok") else "failed"
        print(f"[{msg['name']} {mark}] {msg.get('text', '')}")
    elif ev == "confirm_request":
        print(f"confirm? {msg['question']}  (answer with: hark yes / hark no)")
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
    print(f"state: {msg.get('state')}{'' if msg.get('ready', True) else ' (warming up)'}")
    print(f"talker: {msg.get('talker')}  worker: {msg.get('worker')}")
    print(f"stt: {engines.get('stt')}  tts: {engines.get('tts')}")
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

    p = argparse.ArgumentParser(prog="hark", description="Fast voice assistant.", parents=[flags(True)])
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
    st = sub.add_parser("set", help="change a setting live (brain.talker, tts.engine, stt.engine)")
    st.add_argument("key")
    st.add_argument("value")
    sub.add_parser("events", help="print every daemon event as JSON lines")
    for name, what in (("install", "add the Omarchy keys, bar widget and service"),
                       ("uninstall", "remove what hark install added")):
        sp = sub.add_parser(name, help=what)
        sp.add_argument("--no-activate", action="store_true",
                        help="only write or remove the files; leave systemd, the shell and Hyprland alone")
    return p


def _run_daemon() -> int:
    import asyncio

    from hark import app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request drowns the log

    async def main() -> None:
        daemon = await app.build(config.load())
        await daemon.run()

    asyncio.run(main())
    return 0


def _pending_question(path: Path) -> dict[str, Any] | None:
    for msg in request(path, {"cmd": "status"}):
        return msg.get("confirm")
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "daemon":
        return _run_daemon()
    if args.command in ("install", "uninstall"):
        from hark import install

        return install.main(args.command, activate=not args.no_activate)
    path = args.socket or config.SOCKET_PATH
    req = _build_request(args)
    status = 0
    try:
        if req["cmd"] == "confirm":
            # Answer exactly the question that is waiting now, and say which one it was.
            pending = _pending_question(path)
            if pending is None:
                print("hark: nothing to confirm", file=sys.stderr)
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
                print(f"hark: {msg.get('error', 'request failed')}", file=sys.stderr)
                status = EXIT_ERROR
    except NotRunning:
        print(f"hark: the daemon is not running (no socket at {path}). Start it with: hark daemon", file=sys.stderr)
        return EXIT_NOT_RUNNING
    except KeyboardInterrupt:
        return 130
    except (ConnectionError, json.JSONDecodeError) as exc:
        print(f"hark: lost the daemon connection: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Incomplete:
        print("hark: the daemon closed the connection without an answer", file=sys.stderr)
        return EXIT_ERROR
    return status


if __name__ == "__main__":
    sys.exit(main())
