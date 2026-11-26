# Omi

[![Built for Omarchy](https://raw.githubusercontent.com/tcballard/omarchy-badges/85f859029e7b05ada6dbe73506d07a91/badges/v1/built-for-omarchy.svg)](https://github.com/tcballard/omarchy-badges)

A local-first voice assistant and animated on-screen companion for Omarchy & Hyprland, with desktop, browser, terminal, and T3 Code control.

## Features

- **Fast voice loop + conversational wake word**: Always-on **"Hey Omi"** / **"Okay Omi"** wake word (`sherpa-onnx` 3.3M streaming keyword spotter + Silero VAD, ~1.7% CPU, toggleable live) with continuous phone-call conversation mode (keeps listening after every reply until you say *"goodbye"* / *"see ya"* / *"that's all"* or pause for 12 s), streaming local STT (`nemotron` or `moonshine`), and neural local TTS (`kokoro` v1.0 or `piper`).
- **Animated companion mascot**: Lives in the Omarchy bar and pops up as a click-through floating overlay with speech bubbles, pointing/highlighting (`point_at`, `highlight_window`), and cursor-following eyes.
- **Follows your Omarchy theme live**: Bar icon, floating companion, conversation panel, and mobile web PWA automatically adapt to your active Omarchy theme (`Commons.Color` / `~/.local/state/omarchy/current/theme/colors.toml`).
- **Desktop, browser, terminal & T3 Code control**:
  - **Hyprland & Omarchy**: Switch/focus/close windows and workspaces, control media/volume, clipboard, screenshots, notifications, and vetted `omarchy-*` commands.
  - **GUI & vision**: AT-SPI accessibility inspection and actions (`ui_inspect`, `ui_action`), screen capture (`capture_screen`), and pointer/keyboard control (`ydotool` + `wtype`).
  - **Browser**: Claude in Chrome MCP integration on the background worker for reading tabs, navigating, clicking, and filling forms.
  - **Terminal & T3 Code**: Inspect and run commands in live **Herdr** or **tmux** panes, and list, read, start, or message **T3 Code** threads via the `t3` CLI.
- **Safety guardrails**: Hard-blocks reading secrets/keys/tokens and asks for spoken or on-screen confirmation before destructive commands, terminal mutations, or sensitive actions.
- **Model-agnostic & subscription-first**: Works with your existing AI CLI subscriptions (no API keys needed) — **Claude Code** (`claude`), **OpenCode** (`opencode`, free models / Copilot / multi-provider subs), **Codex** (`codex`, ChatGPT Plus/Pro), **Antigravity** (`agy`, Google AI Pro/Ultra), **Gemini CLI** (`gemini`), or any **local / OpenAI-compatible** endpoint (`llama-server`, Ollama, LM Studio, OpenRouter, xAI/Grok). Switch providers or models live with `omi set brain.talker <provider>` and `omi set brain.model <model>`.
- **Persistent compacting memory**: Markdown facts + rolling session digests + SQLite FTS5 search, with optional read-only shared memory directory (e.g. an Obsidian/Syncthing folder).
- **Private mobile/tablet web UI**: Optional local web PWA served on `127.0.0.1:47911` for private access via `tailscale serve`.

## Screenshots

<!-- Add a screenshot of Omi in the Omarchy bar and the floating companion pointing on screen before publishing -->
_Screenshot coming soon._

## Keybindings

`omi install` appends one marked block to `~/.config/hypr/bindings.lua` using keys that do not conflict with Omarchy defaults:

| Shortcut | Action |
| --- | --- |
| `Super + Alt + V` (hold) | Push-to-talk (hold to speak, release to send) |
| `Super + Alt + O` | Toggle the Omi panel in the Omarchy bar |
| `Super + Alt + Escape` | Stop speaking / cancel current turn or job |

## Requirements

- Arch Linux / [Omarchy](https://omarchy.org) with Hyprland, `omarchy-shell` (Quickshell), PipeWire, `wtype`, `wl-clipboard`, `grim`
- Python 3.12+ and [`uv`](https://docs.astral.sh/uv/)
- At least one AI backend (subscription CLI or local/OpenAI-compatible):
  - **Claude Code** (`claude` — Claude Pro/Max), **OpenCode** (`opencode` — free / Copilot / subs), **Codex** (`codex` — ChatGPT Plus/Pro), **Antigravity** (`agy` — Google AI Pro/Ultra), **Gemini CLI** (`gemini`), **or** `llama-server` / Ollama / any OpenAI-compatible endpoint (`brain.talker = "local"`)
- Optional:
  - `ydotool` (enabled as a user service) for visual mouse clicks/scrolls
  - `python-gobject` + `at-spi2-core` for AT-SPI UI inspection
  - `herdr` or `tmux` for live terminal pane control
  - `t3` (T3 Code) for thread control
  - `tailscale` for the private mobile web UI

## Quickstart

```bash
git clone <repo-url> omi
cd omi
uv sync
```

Download speech models into `~/.local/share/omi/models/` (default locations):

- **Nemotron streaming STT** (`~/.local/share/omi/models/nemotron-streaming-160ms/`): `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt` (from `sherpa-onnx` `sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14`)
- **Kokoro TTS** (`~/.local/share/omi/models/kokoro/`): `kokoro-v1.0.onnx` and `voices-v1.0.bin` (from `kokoro-onnx` releases)
- **Wake word & VAD** (`kws-gigaspeech-3.3M` + `silero_vad.onnx`): downloaded automatically on first start when `[wake].enabled = true`

Install the Omarchy bar plugin, systemd user service, and Hyprland keybindings:

```bash
uv run omi install
omi status
```

## Configuration

Copy [`config.example.toml`](config.example.toml) to `~/.config/omi/config.toml` and edit only what you want to override. All settings have sensible XDG defaults:

- Config: `~/.config/omi/config.toml`
- Models & memory: `~/.local/share/omi/`
- Runtime socket: `$XDG_RUNTIME_DIR/omi/omi.sock`
- State & backups: `~/.local/state/omi/`

Live runtime tweaks:

```bash
omi providers                  # see all supported providers & installed CLIs
omi set brain.talker opencode  # switch provider live (claude|opencode|codex|agy|gemini|local|auto)
omi set brain.model sonnet     # switch active talker model live
omi set tts.voice bm_george    # switch Kokoro voice live
omi set wake.enabled off       # pause/resume always-on wake word live
omi say "what's running in my terminal?"
```

## Uninstall

Remove the systemd user service, Omarchy bar plugin, and keybinding block in one command:

```bash
omi uninstall
```

Your models, memory files, config (`~/.config/omi/config.toml`), and automatic config backups (`~/.local/state/omi/backups/`) are left untouched so you do not lose data; delete `~/.local/share/omi`, `~/.config/omi`, and `~/.local/state/omi` if you want a full wipe.

## License

[MIT](LICENSE)
