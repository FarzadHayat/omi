"""Wire protocol between the hark daemon and its clients (CLI, bar widget, later
phone or tablet apps). Newline-delimited JSON over a Unix socket at config.SOCKET_PATH.

Client sends one request object per line:

    {"cmd": "say", "text": "..."}        run a turn from text (as if spoken)
    {"cmd": "wav", "path": "...", "out": "..."}  run a turn from a 16 kHz wav file;
                                         with "out" the reply audio is written to
                                         that wav instead of being played
    {"cmd": "ptt_start"}                 start listening (push-to-talk pressed)
    {"cmd": "ptt_stop"}                  stop listening and run the turn
    {"cmd": "stop"}                      stop speaking and cancel whatever is running
    {"cmd": "confirm", "answer": true}   answer a pending confirm_request
    {"cmd": "ask_confirm", "question": "..."}  used by the hark MCP server: the daemon
                                         asks the user (voice plus confirm_request event),
                                         replies {"ok": true, "answer": true|false}
    {"cmd": "status"}                    reply with one status object
    {"cmd": "set", "key": "brain.talker", "value": "claude"}  change a setting live
    {"cmd": "subscribe"}                 keep the connection open and receive all events

Any request may carry "quiet": true to skip audio playback for that turn (tests).

The daemon answers each request with {"ok": true, ...} or {"ok": false, "error": "..."}.
For "say", "wav" and "ptt_stop" the events of that turn are streamed on the same
connection, and the final line is {"ok": true, "done": true, "reply": "...", "timing": {...}}.

Events (each has "event"):

    {"event": "state", "state": "idle|listening|thinking|speaking|acting|confirming"}
    {"event": "transcript", "role": "user|assistant", "text": "...", "final": true}
    {"event": "tool", "phase": "start|end", "name": "...", "args": {...}, "ok": true, "text": "..."}
    {"event": "confirm_request", "question": "..."}
    {"event": "progress", "text": "..."}          background worker narration
    {"event": "timing", ...}                      per-turn latency trace, ms from end of speech:
        stt_ms, route_ms, first_text_ms, first_audio_ms, action_ms, total_ms, path ("router|talker")

Timing traces are also appended to config.STATE_DIR / "turns.jsonl".
"""

COMMANDS = ("say", "wav", "ptt_start", "ptt_stop", "stop", "confirm", "ask_confirm", "status", "set", "subscribe")
EVENTS = ("state", "transcript", "tool", "confirm_request", "progress", "timing")
