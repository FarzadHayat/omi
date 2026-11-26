"""Configuration: defaults here, overrides in ~/.config/omi/config.toml.

The TOML file mirrors the dataclass tree, for example:

    [brain]
    talker = "claude"

    [tts]
    engine = "kokoro"
    kokoro_voice = "af_heart"
"""

from __future__ import annotations

import os
import re
import tempfile
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

HOME = Path.home()


def _env_path(primary: str, legacy: str, default: Path) -> Path:
    val = os.environ.get(primary) or os.environ.get(legacy)
    return Path(val) if val else default


def _first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


CONFIG_PATH = _env_path(
    "OMI_CONFIG",
    "HARK_CONFIG",
    _first_existing(HOME / ".config/omi/config.toml", HOME / ".config/hark/config.toml"),
)
DATA_DIR = _env_path(
    "OMI_DATA_DIR",
    "HARK_DATA_DIR",
    _first_existing(HOME / ".local/share/omi", HOME / ".local/share/hark"),
)
STATE_DIR = _env_path("OMI_STATE_DIR", "HARK_STATE_DIR", HOME / ".local/state/omi")
RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "omi"
SOCKET_PATH = _env_path("OMI_SOCKET", "HARK_SOCKET", RUNTIME_DIR / "omi.sock")
MODELS_DIR = _first_existing(DATA_DIR / "models", HOME / ".local/share/hark/models")


def _model_path(rel: str, *fallbacks: Path) -> str:
    return str(_first_existing(MODELS_DIR / rel, HOME / ".local/share/hark/models" / rel, *fallbacks))


@dataclass
class AudioConfig:
    source: str = ""  # PipeWire source name; empty means the default microphone
    sink: str = ""  # PipeWire sink name; empty means the default output, "null" plays nothing
    silence_ms: int = 300  # hands-free mode: silence that ends a turn
    sounds: bool = True  # cute UI/voice sound effects (wake, heard, acting, confirm, done, sleep)


@dataclass
class SttConfig:
    # moonshine | nemotron. Nemotron streams while you speak: on 96 spoken commands it
    # heard 60 of 63 routable ones right (Moonshine 48), at about 215 ms after speech (45).
    engine: str = "nemotron"
    moonshine_dir: str = _model_path("moonshine-base-en")
    nemotron_dir: str = _model_path("nemotron-streaming-160ms")
    threads: int = 4
    hotwords: list[str] = field(default_factory=lambda: ["Omi", "Omarchy", "Hyprland"])


@dataclass
class TtsConfig:
    engine: str = "kokoro"  # kokoro | piper
    piper_model: str = _model_path("piper/en_US-lessac-medium.onnx")
    # Full-precision Kokoro fp32 sounds markedly cleaner than int8 (~498 ms first chunk vs 356 ms).
    kokoro_model: str = _model_path(
        "kokoro/kokoro-v1.0.onnx",
        HOME / ".local/share/kokoro-tts/kokoro-v1.0.onnx",
        MODELS_DIR / "kokoro/kokoro-v1.0.int8.onnx",
        HOME / ".local/share/hark/models/kokoro/kokoro-v1.0.int8.onnx",
    )
    kokoro_voices: str = _model_path(
        "kokoro/voices-v1.0.bin",
        HOME / ".local/share/kokoro-tts/voices-v1.0.bin",
    )
    kokoro_voice: str = "af_heart"
    speed: float = 1.0


@dataclass
class LocalLlmConfig:
    # Any OpenAI-compatible chat-completions endpoint. When base_url is empty Omi
    # starts its own llama-server with `model` on `port`.
    base_url: str = ""
    api_key_env: str = ""  # name of the env var holding the key; never the key itself
    model_name: str = "local"
    model: str = _model_path(
        "llm/Qwen3.5-4B-Q4_K_M.gguf",
        HOME / "models/Qwen3.5-4B/Qwen3.5-4B-Q4_K_M.gguf",
    )
    port: int = 47910
    gpu_layers: int = 99
    ctx: int = 8192


@dataclass
class ClaudeConfig:
    model: str = "haiku"  # talker model for the claude CLI
    worker_model: str = "sonnet"  # background worker model
    chrome: bool = True  # enable Claude in Chrome browser control on the worker


@dataclass
class CliProviderConfig:
    model: str = ""  # empty = use the CLI's default configured model
    worker_model: str = ""  # empty = fall back to model or CLI default
    effort: str = ""  # optional reasoning effort (e.g. "low" for fast voice talker)


@dataclass
class BrainConfig:
    # claude | opencode | codex | agy | gemini | local | openai | auto
    talker: str = "claude"
    # claude | opencode | codex | agy | gemini | off | auto
    worker: str = "claude"
    local: LocalLlmConfig = field(default_factory=LocalLlmConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    opencode: CliProviderConfig = field(default_factory=CliProviderConfig)
    codex: CliProviderConfig = field(default_factory=CliProviderConfig)
    agy: CliProviderConfig = field(default_factory=lambda: CliProviderConfig(effort="low"))
    gemini: CliProviderConfig = field(default_factory=CliProviderConfig)


@dataclass
class MemoryConfig:
    enabled: bool = True
    dir: str = str(DATA_DIR / "memory")
    # Optional folder of facts other agents already keep about the user. Read-only.
    shared_dir: str = ""
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
class WakeConfig:
    enabled: bool = True
    phrases: list[str] = field(default_factory=lambda: ["omi", "hey omi", "okay omi"])
    model_dir: str = _model_path("kws-gigaspeech-3.3M")
    threshold: float = 0.15
    score: float = 1.0
    silence_ms: int = 750  # hands-free end-of-turn pause while speaking
    initial_timeout_s: float = 4.0  # cancel if nothing is said right after wake word
    call_mode: str = "auto"  # auto (smart) | continuous | single
    conversational: bool = True  # legacy bool; false forces single-turn mode
    followup_timeout_s: float = 12.0  # drop back to standby after this much silence in call mode
    max_utterance_s: float = 15.0
    chime: bool = True


@dataclass
class Config:
    name: str = "Omi"  # what the assistant calls itself
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
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


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"unsupported TOML value: {value!r}")


def save_override(section: str, key: str, value: Any, path: Path | None = None) -> Path:
    """Persist one [section] key = value override into config.toml while keeping
    existing comments and unrelated settings intact."""
    target = path or (HOME / ".config/omi/config.toml")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    lines = text.splitlines()
    header = f"[{section}]"
    entry = f"{key} = {_toml_value(value)}"
    key_re = re.compile(rf"^\s*{re.escape(key)}\s*=")

    sec_start: int | None = None
    sec_end = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if sec_start is not None:
                sec_end = idx
                break
            if stripped == header:
                sec_start = idx

    if sec_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend([header, entry])
    else:
        replaced = False
        for idx in range(sec_start + 1, sec_end):
            if key_re.match(lines[idx]):
                lines[idx] = entry
                replaced = True
                break
        if not replaced:
            insert_at = sec_end
            while insert_at > sec_start + 1 and not lines[insert_at - 1].strip():
                insert_at -= 1
            lines.insert(insert_at, entry)

    rendered = "\n".join(lines).rstrip("\n") + "\n"
    # Validate rendered TOML before replacing the user's config file.
    tomllib.loads(rendered)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target
