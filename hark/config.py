"""Configuration: defaults here, overrides in ~/.config/hark/config.toml.

The TOML file mirrors the dataclass tree, for example:

    [brain]
    talker = "claude"

    [tts]
    engine = "kokoro"
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

HOME = Path.home()
CONFIG_PATH = Path(os.environ.get("HARK_CONFIG", HOME / ".config/hark/config.toml"))
DATA_DIR = Path(os.environ.get("HARK_DATA_DIR", HOME / ".local/share/hark"))
STATE_DIR = Path(os.environ.get("HARK_STATE_DIR", HOME / ".local/state/hark"))
RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "hark"
SOCKET_PATH = Path(os.environ.get("HARK_SOCKET", RUNTIME_DIR / "hark.sock"))
MODELS_DIR = DATA_DIR / "models"


@dataclass
class AudioConfig:
    source: str = ""  # PipeWire source name; empty means the default microphone
    sink: str = ""  # PipeWire sink name; empty means the default output, "null" plays nothing
    silence_ms: int = 300  # hands-free mode: silence that ends a turn


@dataclass
class SttConfig:
    # moonshine | nemotron. Nemotron streams while you speak: on 96 spoken commands it
    # heard 60 of 63 routable ones right (Moonshine 48), at about 215 ms after speech (45).
    engine: str = "nemotron"
    moonshine_dir: str = str(MODELS_DIR / "moonshine-base-en")
    nemotron_dir: str = str(MODELS_DIR / "nemotron-streaming-160ms")
    threads: int = 4


@dataclass
class TtsConfig:
    engine: str = "piper"  # piper | kokoro
    piper_model: str = str(MODELS_DIR / "piper/en_US-lessac-medium.onnx")
    kokoro_model: str = str(MODELS_DIR / "kokoro/kokoro-v1.0.int8.onnx")
    kokoro_voices: str = str(MODELS_DIR / "kokoro/voices-v1.0.bin")
    kokoro_voice: str = "bm_george"
    speed: float = 1.0


@dataclass
class LocalLlmConfig:
    # Any OpenAI-compatible chat-completions endpoint. When base_url is empty hark
    # starts its own llama-server with `model` on `port`.
    base_url: str = ""
    api_key_env: str = ""  # name of the env var holding the key; never the key itself
    model_name: str = "local"
    model: str = str(HOME / "models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf")
    port: int = 47910
    gpu_layers: int = 99
    ctx: int = 8192


@dataclass
class ClaudeConfig:
    model: str = "haiku"  # talker and worker model for the claude CLI
    worker_model: str = "sonnet"


@dataclass
class BrainConfig:
    # local | claude : who you converse with. Claude measured faster here (first audio
    # 881 vs 1284 ms on chat, action 852 vs 1733 ms on tool turns) and answered 10 of 10
    # test prompts well against about 8 for the local 4B model. Local works offline.
    talker: str = "claude"
    worker: str = "claude"  # claude | off  : who does long multi-step jobs
    local: LocalLlmConfig = field(default_factory=LocalLlmConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)


@dataclass
class MemoryConfig:
    enabled: bool = True
    dir: str = str(HOME / "Sync/Obsidian/Agent-Memory/voice")
    # Facts other agents already keep about the user. Read and searched, never written.
    shared_dir: str = str(HOME / "Sync/Obsidian/Agent-Memory")
    index: str = str(DATA_DIR / "memory.sqlite")
    embeddings_url: str = "http://127.0.0.1:47801"  # optional; never on the blocking path


@dataclass
class WebConfig:
    # Phone and tablet page, reached privately through `tailscale serve`. The server only
    # listens on 127.0.0.1 and only answers requests that came through tailscaled.
    enabled: bool = True
    port: int = 47911
    # Tailscale logins (Tailscale-User-Login header) let in. Empty means the login that
    # owns this machine on the tailnet, read from `tailscale status --json` at startup.
    allowed_logins: list[str] = field(default_factory=list)
    # Extra websocket origins accepted besides https://<host the page was loaded from>.
    origins: list[str] = field(default_factory=list)
    proxy_uid: int = 0  # owner of the process that may connect: tailscaled runs as root
    theme: str = str(Path(os.environ.get("XDG_STATE_HOME", HOME / ".local/state")) / "omarchy/current/theme/colors.toml")


@dataclass
class Config:
    name: str = "Hark"  # what the assistant calls itself
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    web: WebConfig = field(default_factory=WebConfig)


def _merge(obj, data: dict) -> None:
    known = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"config key {key} must be a table")
            _merge(current, value)
        else:
            setattr(obj, key, value)


def load(path: Path | None = None) -> Config:
    cfg = Config()
    path = path or CONFIG_PATH
    if path.exists():
        with path.open("rb") as fh:
            _merge(cfg, tomllib.load(fh))
    return cfg
