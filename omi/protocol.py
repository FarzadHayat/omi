"""Wire protocol between the omi daemon and its clients (CLI, bar widget, phone and
tablet page). Newline-delimited JSON over a Unix socket at config.SOCKET_PATH; the
phone's WebSocket is described at the end.

Client sends one request object per line:

    {"cmd": "say", "text": "..."}        run a turn from text (as if spoken)
    {"cmd": "wav", "path": "...", "out": "..."}  run a turn from a 16 kHz wav file;
                                         with "out" the reply audio is written to
                                         that wav instead of being played
    {"cmd": "ptt_start"}                 start listening (push-to-talk pressed)
    {"cmd": "ptt_stop"}                  stop listening and run the turn
    {"cmd": "stop"}                      stop speaking and cancel whatever is running
    {"cmd": "confirm", "answer": true, "id": "..."}  answer the pending confirm_request
                                         with that id. Only from a same-user process the
                                         daemon did not start (CLI, bar); Claude and its
                                         shell are refused, and while a question waits
                                         they cannot start turns either
    {"cmd": "ask_confirm", "question": "...", "detail": "..."}  used by the omi MCP
                                         server: the daemon asks the user (voice plus a
                                         confirm_request event; detail, like a full command,
                                         is only shown), replies {"ok": true, "answer": true|false}
    {"cmd": "tool", "name": "remember", "args": {...}}  used by the omi MCP server for
                                         tools that live in the daemon (memory, delegate);
                                         replies {"ok": true, "result": {"ok": ..., "text": ...}}
    {"cmd": "status"}                    reply with one status object; it includes
                                         "confirm": {"id", "question", "detail"} or null,
                                         and "errors": {component: message}
    {"cmd": "set", "key": "brain.talker", "value": "claude"}  change a setting live
    {"cmd": "subscribe"}                 keep the connection open and receive all events

Any request may carry "quiet": true to skip audio playback for that turn (tests).
A request line may be up to 4 MiB; a longer one gets {"ok": false, "error": "request too large ..."}.

The daemon answers each request with {"ok": true, ...} or {"ok": false, "error": "..."}.
For "say", "wav" and "ptt_stop" the events of that turn are streamed on the same
connection, and the final line is {"ok": true, "done": true, "reply": "...", "timing": {...}}.

Events (each has "event"):

    {"event": "state", "state": "idle|listening|thinking|speaking|acting|confirming"}
    {"event": "transcript", "role": "user|assistant", "text": "...", "final": true}
    {"event": "tool", "phase": "start|end", "name": "...", "args": {...}, "ok": true, "text": "..."}
    {"event": "confirm_request", "question": "...", "id": "...", "detail": "..."}
    {"event": "progress", "text": "..."}          background worker narration
    {"event": "timing", ...}                      per-turn latency trace, ms from end of speech:
        stt_ms, route_ms, first_text_ms, first_audio_ms (first audio handed to the
        output device), action_ms, brain_done_ms (reply fully generated), total_ms
        (reply finished playing), path ("router|talker|confirm")

Timing traces are also appended to config.STATE_DIR / "turns.jsonl".

Phones and tablets (omi.web) use one WebSocket at /ws on the daemon's web port, reached
through `tailscale serve`. Every request needs the Tailscale-User-Login header of an
allowed login and an X-Forwarded-For of another tailnet device (programs on the PC
itself are refused), and the WebSocket an Origin of the page itself. Push-to-talk is
cut after 60 s and typed text is limited to 2000 characters.

    phone -> daemon, binary: 16 kHz mono Int16 little-endian PCM, sent between
                             ptt_start and ptt_stop (20 ms frames)
    phone -> daemon, text:   {"cmd": "ptt_start"}   start talking (talks over the assistant)
                             {"cmd": "ptt_stop"}    stop talking and run the turn
                             {"cmd": "say", "text": "..."}
                             {"cmd": "stop"}
                             {"cmd": "confirm", "answer": true, "id": "..."}
                             {"cmd": "played", "seq": n}  the reply of turn n finished playing
    daemon -> phone, binary: reply audio, streamed clause by clause: a WEB_AUDIO_HEADER
                             (uint32 sample rate, uint32 turn number seq) then Int16 LE PCM
    daemon -> phone, text:   every event above, plus
                             {"event": "hello", "name", "state", "confirm", "ready"} on connect
                             {"event": "audio", "phase": "end|stop", "seq": n}  all of turn n's
                                 audio was sent, or drop what is left of it now
                             {"event": "done", "ok", "reply", "timing", "cancelled", "error"?}
                             {"event": "error", "text": "...", "cmd": "..."}

A phone's turns play their reply on that phone only.
"""

WEB_COMMANDS = ("ptt_start", "ptt_stop", "say", "stop", "confirm", "played")
WEB_AUDIO_HEADER = "<II"  # struct format: sample rate, seq

COMMANDS = ("say", "wav", "ptt_start", "ptt_stop", "stop", "confirm", "ask_confirm", "tool", "mascot",
            "status", "set", "subscribe")
EVENTS = ("state", "transcript", "speech", "tool", "confirm_request", "progress", "mascot", "cursor", "wake", "settings", "timing")
