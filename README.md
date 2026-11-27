# Omi

[![Built for Omarchy](https://raw.githubusercontent.com/tcballard/omarchy-badges/85f859029e7b05ada6dbe73506d07a91/badges/v1/built-for-omarchy.svg)](https://github.com/tcballard/omarchy-badges)

A local-first voice assistant and animated on-screen companion for Omarchy & Hyprland, with desktop, browser, terminal, and T3 Code control.

![Omi](docs/cover.jpg)

## Features

- **Talks like a phone call**: Interrupt Omi mid-sentence just by talking (PipeWire WebRTC echo cancellation keeps it from hearing itself, no headphones needed), and replies stay short and conversational.
- **Never stuck on a limit**: The brain is a priority list (`brain.fallbacks`). If your preferred model hits its usage limit or fails before answering, the next one (for example a free OpenCode model, then a local model) takes the same turn, and the preferred one is retried after 15 minutes.
- **Fast voice loop + smart wake word**: Always-on **"Hey Omi"** / **"Okay Omi"** wake word (`sherpa-onnx` 3.3M streaming keyword spotter + Silero VAD, ~1.7% CPU, toggleable live) with **Smart (`auto`)**, **Continuous**, and **Single-turn** call modes, plus soft procedural sound effects.
- **Best local & cloud STT and TTS**:
  - **Local STT**: **NVIDIA Parakeet TDT 0.6B v3** (default — 10x faster than streaming, native punctuation & capitalization, 25 languages), **Nemotron Streaming 160ms**, or **Moonshine**.
  - **Cloud STT**: **Groq Whisper Large v3 Turbo** (`GROQ_API_KEY`), **Deepgram Nova-3** (`DEEPGRAM_API_KEY`), or **OpenAI `gpt-4o-mini-transcribe`** (`OPENAI_API_KEY`).
  - **Local & Cloud TTS**: **Kokoro v1.0** (default local neural), **Piper**, **OpenAI `gpt-4o-mini-tts`** (`OPENAI_API_KEY`), or **ElevenLabs `eleven_flash_v2_5`** (`ELEVENLABS_API_KEY`). Switch live from the bar Settings drawer or `omi engines`.
- **Animated companion mascot**: Lives in the Omarchy bar and pops up as a click-through floating overlay with speech bubbles, pointing/highlighting (`point_at`, `highlight_window`), and cursor-following eyes.
- **Follows your Omarchy theme live**: Bar icon, floating companion, conversation panel, and mobile web PWA automatically adapt to your active Omarchy theme (`Commons.Color` / `~/.local/state/omarchy/current/theme/colors.toml`).
- **Desktop, browser, terminal & T3 Code control**:
  - **Hyprland & Omarchy**: Switch/focus/close windows and workspaces, control media/volume, clipboard, screenshots, notifications, and vetted `omarchy-*` commands.
  - **GUI & vision**: AT-SPI accessibility inspection and actions (`ui_inspect`, `ui_action`), screen capture (`capture_screen`), and pointer/keyboard control (`ydotool` + `wtype`).
  - **Browser**: Claude in Chrome MCP integration on the background worker for reading tabs, navigating, clicking, and filling forms.
  - **Terminal & T3 Code**: Inspect and run commands in live **Herdr** or **tmux** panes, and list, read, start, message, or open on screen **T3 Code** threads (by project name, e.g. "show me the bne tower chat") via the `t3` CLI and T3's desktop deep links.
- **Safety guardrails**: Hard-blocks reading secrets/keys/tokens and asks for spoken or on-screen confirmation before destructive commands, terminal mutations, or sensitive actions.
- **Model-agnostic & subscription-first**: Works with your existing AI CLI subscriptions (no API keys needed) — **Claude Code** (`claude`), **OpenCode** (`opencode`, free models / Copilot / multi-provider subs), **Codex** (`codex`, ChatGPT Plus/Pro), **Antigravity** (`agy`, Google AI Pro/Ultra), **Gemini CLI** (`gemini`), or any **local / OpenAI-compatible** endpoint (`llama-server`, Ollama, LM Studio, OpenRouter, xAI/Grok). Switch providers or models live from the bar Settings drawer (dropdowns labelled "on this PC" or "cloud", with a "best for this PC" pick among your downloaded local models) or with `omi set brain.talker <provider>`, `omi set brain.model <model>`, `omi set brain.local_model <file.gguf>` and `omi set brain.fallbacks "opencode,local"`.
- **Persistent compacting memory**: Markdown facts + rolling session digests + SQLite FTS5 search, with optional read-only shared memory directory (e.g. an Obsidian/Syncthing folder).
- **Private mobile/tablet web UI**: Optional local web PWA served on `127.0.0.1:47911` for private access via `tailscale serve`.

## Keybindings

`omi install` appends one marked block to `~/.config/hypr/bindings.lua` using keys that do not conflict with Omarchy defaults:

| Shortcut | Action |
| --- | --- |
| `Super + Alt + V` (hold) | Push-to-talk with Omi (hold to speak, release to send) |
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
git clone https://github.com/FarzadHayat/omi.git
cd omi
uv sync
```

Download speech models into `~/.local/share/omi/models/` (default locations):

- **Parakeet TDT v3 STT (default)** (`~/.local/share/omi/models/parakeet-tdt-0.6b-v3-int8/`): `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8` from `sherpa-onnx` releases (or **Nemotron Streaming** in `nemotron-streaming-160ms/`)
- **Kokoro TTS (default)** (`~/.local/share/omi/models/kokoro/`): `kokoro-v1.0.onnx` and `voices-v1.0.bin` (from `kokoro-onnx` releases)
- **Wake word & VAD** (`kws-gigaspeech-3.3M` + `silero_vad.onnx`): downloaded automatically on first start when `[wake].enabled = true`
- **Cloud STT/TTS (optional)**: export `GROQ_API_KEY`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, or `ELEVENLABS_API_KEY` and switch live with `omi engines` / `omi set` or the bar Settings drawer.

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
omi providers                  # see all supported AI providers & installed CLIs
omi engines                    # see local & cloud STT/TTS engines and readiness
omi set brain.talker opencode  # switch provider live (claude|opencode|codex|agy|gemini|local|auto)
omi set stt.engine parakeet    # switch STT live (parakeet|nemotron|moonshine|groq|deepgram|openai)
omi set tts.engine kokoro      # switch TTS live (kokoro|piper|openai|elevenlabs)
omi set tts.voice bm_george    # switch active voice live
omi set wake.enabled off       # pause/resume always-on wake word live
omi say "what's running in my terminal?"
```

## Uninstall

Remove the systemd user service, Omarchy bar plugin, and keybinding block in one command:

```bash
omi uninstall
```

Your models, memory files, config (`~/.config/omi/config.toml`), and automatic config backups (`~/.local/state/omi/backups/`) are left untouched so you do not lose data; delete `~/.local/share/omi`, `~/.config/omi`, and `~/.local/state/omi` if you want a full wipe.

## More for Omarchy

See everything else I've built for Omarchy, each with a one-line install, at **[omarchy.farzadhayat.dev](https://omarchy.farzadhayat.dev)**.

## Support

If this saves you time, please [sponsor me on GitHub](https://github.com/sponsors/FarzadHayat). It keeps these projects maintained and free.

## License

[MIT](LICENSE)
