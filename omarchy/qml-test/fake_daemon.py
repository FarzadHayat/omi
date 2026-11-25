"""A stand-in hark daemon for the bar widget harness (shell.qml).

Usage: fake_daemon.py SOCKET LOG. Speaks hark.protocol on SOCKET, plays a short
event script to subscribers, writes every request it gets to LOG as JSON lines,
and shuts down (so the widget sees the daemon go) once it has had a say and a stop.
"""

import asyncio
import json
import sys

SCRIPT = [
    {"event": "state", "state": "listening"},
    {"event": "transcript", "role": "user", "text": "delete my notes", "final": True},
    {"event": "state", "state": "thinking"},
    {"event": "state", "state": "speaking"},
    {"event": "transcript", "role": "assistant", "text": "<b>not bold</b>", "final": True},
    {"event": "state", "state": "acting"},
    {"event": "state", "state": "confirming"},
    {"event": "confirm_request", "id": "q-1", "question": "Delete the file notes.txt?", "detail": "rm notes.txt"},
]


async def main(sock_path: str, log_path: str) -> None:
    subs: list[asyncio.StreamWriter] = []
    seen: set[str] = set()
    done = asyncio.Event()
    log = open(log_path, "a")

    def send(w: asyncio.StreamWriter, obj: dict) -> None:
        w.write((json.dumps(obj) + "\n").encode())

    async def client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                req = json.loads(line)
                log.write(json.dumps(req) + "\n")
                log.flush()
                cmd = req.get("cmd")
                seen.add(cmd)
                if cmd == "status":
                    send(writer, {"ok": True, "state": "idle", "ready": True, "confirm": None, "errors": {}})
                elif cmd == "subscribe":
                    send(writer, {"ok": True, "subscribed": True})
                    subs.append(writer)
                    for ev in SCRIPT:
                        await asyncio.sleep(0.05)
                        send(writer, ev)
                elif cmd == "confirm":
                    send(writer, {"ok": True})
                    for w in subs:
                        send(w, {"event": "state", "state": "idle"})
                elif cmd == "say":
                    send(writer, {"ok": False, "error": "a question is waiting for the user's answer"})
                else:
                    send(writer, {"ok": True})
                await writer.drain()
                if {"say", "stop"} <= seen:
                    await asyncio.sleep(0.3)
                    done.set()
        except (ConnectionError, json.JSONDecodeError):
            pass

    server = await asyncio.start_unix_server(client, path=sock_path)
    try:
        await asyncio.wait_for(done.wait(), 20)
    finally:
        server.close()
        for w in subs:
            w.close()
        await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
