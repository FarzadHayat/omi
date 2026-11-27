"""What the settings panel shows for the brain: every provider with whether it runs on
this computer or in the cloud, the models each one offers, and which downloaded local
model suits this machine best.

The local pick is a rule of thumb that matches measurements on a Ryzen 7 8745HS with
Radeon 780M graphics (shared memory): token speed follows the bytes read per token, so a
mixture-of-experts model like Gemma 4 26B-A4B (about 4B active) speaks faster than a
dense 4B while knowing far more. A model only counts if it fits in memory next to the
desktop.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from omi.config import MODELS_DIR, Config

log = logging.getLogger(__name__)

# provider -> (label, where it runs)
PROVIDERS: dict[str, tuple[str, str]] = {
    "claude": ("Claude", "cloud"),
    "opencode": ("OpenCode", "cloud"),
    "codex": ("Codex", "cloud"),
    "agy": ("Antigravity", "cloud"),
    "gemini": ("Gemini", "cloud"),
    "local": ("On this computer", "local"),
}
CLAUDE_MODELS = [
    {"value": "haiku", "label": "Haiku · fastest"},
    {"value": "sonnet", "label": "Sonnet · smarter"},
    {"value": "opus", "label": "Opus · smartest, slower"},
]
LOCAL_DIRS = (MODELS_DIR / "llm", Path.home() / "models")
HEADROOM_GB = 10.0  # left for the desktop and apps when a local model is loaded
RUNTIME_GB = 1.5  # context cache and buffers on top of the file size
FAST_ACTIVE_GB = 3.0  # read per token at most this much to talk at conversational speed

_PARAMS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)B(?![a-z])", re.I)
_ACTIVE = re.compile(r"A(\d+(?:\.\d+)?)B", re.I)

_cli_cache: dict[str, list[dict[str, str]]] = {}
_cli_lock = threading.Lock()


def ram_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    return 16.0


def describe_gguf(path: Path) -> dict[str, Any]:
    """Size, parameter counts read from the file name, and the bytes read per token."""
    size_gb = path.stat().st_size / 1e9
    name = path.stem
    active = _ACTIVE.search(name)
    total = next((float(m.group(1)) for m in _PARAMS.finditer(name) if not name[max(0, m.start() - 1)] in "Aa"), None)
    active_b = float(active.group(1)) if active else total
    share = (active_b / total) if (total and active_b and active_b < total) else 1.0
    return {
        "path": str(path),
        "size_gb": round(size_gb, 1),
        "params_b": total,
        "active_gb": round(size_gb * share, 2),
    }


def find_local_models(dirs: tuple[Path, ...] = LOCAL_DIRS) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    out = []
    for root in dirs:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("**/*.gguf")):
            real = path.resolve()
            if real in seen or "mmproj" in path.name.lower() or len(path.relative_to(root).parts) > 3:
                continue
            seen.add(real)
            try:
                out.append(describe_gguf(path))
            except OSError:
                continue
    return out


def recommend(models: list[dict[str, Any]], memory_gb: float) -> str:
    """The smartest model that fits and talks fast enough, else the fastest that fits."""
    fits = [m for m in models if m["size_gb"] + RUNTIME_GB <= memory_gb - HEADROOM_GB]
    if not fits:
        return ""
    fast = [m for m in fits if m["active_gb"] <= FAST_ACTIVE_GB]
    if fast:
        return max(fast, key=lambda m: (m["params_b"] or 0, -m["active_gb"]))["path"]
    return min(fits, key=lambda m: m["active_gb"])["path"]


def _pretty(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[-_.](it|instruct|qat|UD|Q\d\w*|IQ\d\w*|K_[A-Z]+|F16|BF16)\b.*$", "", stem, flags=re.I)
    return stem.replace("-", " ").replace("_", " ")


def local_options(cfg: Config) -> tuple[list[dict[str, Any]], str]:
    models = find_local_models()
    best = recommend(models, ram_gb())
    options = []
    for m in models:
        tag = " · best for this PC" if m["path"] == best else ""
        too_big = m["size_gb"] + RUNTIME_GB > ram_gb() - HEADROOM_GB
        note = " · too big" if too_big else ""
        options.append({"value": m["path"], "label": f"{_pretty(m['path'])} ({m['size_gb']:g} GB){tag}{note}"})
    current = cfg.brain.local.model
    if current and not any(o["value"] == current for o in options):
        options.insert(0, {"value": current, "label": _pretty(current)})
    return options, best


def _load_cli_models(provider: str) -> list[dict[str, str]]:
    if provider != "opencode" or shutil.which("opencode") is None:
        return []
    try:
        out = subprocess.run(["opencode", "models"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    names = [ln.strip() for ln in out.splitlines() if ln.strip()]
    free = [n for n in names if n.endswith("-free") or n == "opencode/big-pickle"]
    return [{"value": n, "label": n.replace("opencode-go/", "go: ").replace("opencode/", "") + " · free"} for n in free]


def load_cli_models() -> None:
    """Fill the CLI model lists (OpenCode takes a few seconds, so run it in a thread)."""
    for provider in ("opencode",):
        models = _load_cli_models(provider)
        with _cli_lock:
            _cli_cache[provider] = models


def model_options(cfg: Config, provider: str) -> list[dict[str, str]]:
    if provider == "claude":
        options = list(CLAUDE_MODELS)
        current = cfg.brain.claude.model
    elif provider == "local":
        return local_options(cfg)[0]
    else:
        pcfg = getattr(cfg.brain, provider, None)
        current = getattr(pcfg, "model", "") if pcfg is not None else ""
        with _cli_lock:
            options = [{"value": "", "label": "CLI default"}, *_cli_cache.get(provider, [])]
    if current and not any(o["value"] == current for o in options):
        options.append({"value": current, "label": current})
    return options


def brain_catalog(cfg: Config, installed: set[str]) -> dict[str, Any]:
    local, best = local_options(cfg)
    providers = []
    for name, (label, kind) in PROVIDERS.items():
        if name in installed:
            providers.append({"value": name, "label": f"{label} · {kind}" if name != "local" else label, "kind": kind})
    return {
        "providers": providers,
        "models": {p["value"]: (local if p["value"] == "local" else model_options(cfg, p["value"])) for p in providers},
        "local_model": cfg.brain.local.model,
        "local_recommended": best,
        "local_endpoint": cfg.brain.local.base_url,
        "ram_gb": round(ram_gb()),
    }
