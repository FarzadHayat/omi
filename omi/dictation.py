"""Hex + Superwhisper-style voice dictation for Omi.

Hold the dictate key (Super+Alt+D) to speak, release to insert directly at the cursor:
  - raw:     Hex-style instant verbatim transcription + custom replacements (0 ms LLM wait).
  - clean:   Strip filler words and fix punctuation/grammar while preserving your voice.
  - message: Concise, natural chat message (Slack, Discord, Telegram, WhatsApp).
  - mail:    Structured email with greeting, clean paragraphs, and sign-off.
  - note:    Organized markdown notes / bullet points (Obsidian, Notion, Logseq).
  - code:    Developer-friendly prompt/note preserving identifiers, flags, and file paths.
  - auto:    Superwhisper-style smart mode:
               1. If text is highlighted (Wayland primary selection) and you speak an edit
                  command ("make this shorter", "fix grammar", "translate to ..."), rewrites
                  the selection in place.
               2. Otherwise chooses the mode from the focused Hyprland window (terminals &
                  code editors default to instant `raw`, chat -> `message`, mail -> `mail`,
                  notes -> `note`, everything else -> `clean`).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from omi.config import DictationConfig
from omi.tools.system import build_press_keys, build_type_text, focused_is_terminal, run

log = logging.getLogger(__name__)

DICTATION_MODES = ("auto", "raw", "clean", "message", "mail", "note", "code")
LLM_TIMEOUT_S = 12.0

# Ordered window (class / title) patterns -> default dictation mode in `auto`.
DEFAULT_APP_MODES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?<![a-z0-9])(ghostty|alacritty|kitty|foot|wezterm|konsole|xterm|terminator|"
            r"tilix|gnome-terminal|urxvt|warp|herdr|tmux|code|code-oss|cursor|windsurf|zed|"
            r"neovim|nvim|vim|t3-?code|jetbrains|idea|pycharm|webstorm|rustrover)(?![a-z0-9])|^org\.omarchy\.",
            re.I,
        ),
        "raw",
    ),
    (
        re.compile(r"\b(slack|discord|vesktop|telegram|signal|whatsapp|element|beeper|teams|zulip|mattermost|messages)\b", re.I),
        "message",
    ),
    (
        re.compile(r"\b(thunderbird|betterbird|geary|evolution|omamail|gmail|fastmail|proton ?mail|outlook|mail)\b", re.I),
        "mail",
    ),
    (
        re.compile(r"\b(obsidian|logseq|notion|joplin|typora|anytype|simplenote|notes)\b", re.I),
        "note",
    ),
)

# Phrases that clearly signal "transform the currently highlighted text" when selection is present.
_REWRITE_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"rewrite|rephrase|reword|shorten|condense|tighten|simplify|summarize|summarise|"
    r"expand|elaborate|polish|improve|clean up|proofread|fix(?:\s+(?:this|the|my|grammar|spelling|typos?))?|"
    r"translate|convert|format|capitalize|title case|bullet(?:ize)?|"
    r"make (?:this|it)|turn (?:this|it) into|change (?:this|it) to|"
    r"reply to this|respond to this|answer this"
    r")\b",
    re.I,
)

# Common accidental LLM preambles stripped if a model ignores the "only output text" rule.
_PREAMBLE_RE = re.compile(
    r"^(?:here(?:'s| is)(?: the)? (?:cleaned|formatted|rewritten|revised|updated|final|edited)?\s*"
    r"(?:text|version|message|email|note|output|result)\s*:\s*\n*)",
    re.I,
)


@dataclass
class ActiveContext:
    window_class: str = ""
    window_title: str = ""
    is_terminal: bool = False
    selection: str = ""


@dataclass
class DictationResult:
    raw: str
    text: str
    mode: str
    inserted: bool = False
    method: str = ""
    rewrote_selection: bool = False
    error: str | None = None


def apply_replacements(text: str, replacements: dict[str, str] | None) -> str:
    """Apply case-insensitive word/phrase replacements (longest phrases matched first)."""
    out = text.strip()
    if not out or not replacements:
        return out
    items = sorted(
        ((k.strip(), str(v)) for k, v in replacements.items() if str(k).strip()),
        key=lambda kv: len(kv[0]),
        reverse=True,
    )
    for src, dst in items:
        pat = re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.I)
        out = pat.sub(lambda _m, r=dst: r, out)
    return out


def resolve_app_mode(
    window_class: str,
    window_title: str,
    custom_app_modes: dict[str, str] | None = None,
) -> str:
    """Pick the best dictation mode for the currently focused Hyprland window."""
    haystack = f"{window_class} {window_title}".strip()
    if custom_app_modes:
        for pattern, mode in custom_app_modes.items():
            m = mode.strip().lower()
            if m not in DICTATION_MODES or m == "auto":
                continue
            try:
                if re.search(pattern, haystack, re.I):
                    return m
            except re.error:
                if pattern.lower() in haystack.lower():
                    return m
    for rx, mode in DEFAULT_APP_MODES:
        if rx.search(window_class) or rx.search(window_title):
            return mode
    return "clean"


def looks_like_rewrite_command(text: str) -> bool:
    """True when the spoken utterance is clearly an instruction to transform highlighted text."""
    return bool(_REWRITE_PREFIX.search(text or ""))


def build_format_prompt(
    raw: str,
    mode: str,
    *,
    app_name: str = "",
    selection: str = "",
) -> str:
    """Build a compact zero-chatter prompt for Superwhisper-style formatting or rewrite."""
    if mode == "rewrite":
        return (
            "You are a precision text editor for voice dictation. "
            "Apply the user's spoken instruction to the selected text.\n"
            "Rules:\n"
            "- Output ONLY the final replacement text. No preamble, no quotes, no explanation.\n"
            "- Preserve formatting and intent unless asked to change them.\n\n"
            f"Selected text:\n{selection}\n\n"
            f"Instruction: {raw}"
        )
    mode_rules = {
        "clean": (
            "Remove filler words (um, uh, like, you know), false starts, and self-corrections. "
            "Fix punctuation, capitalization, and light grammar while preserving the speaker's exact wording and tone."
        ),
        "message": (
            "Format as a concise, natural chat message (Slack/Discord/text). "
            "Remove filler words and false starts; keep it casual, direct, and easy to read."
        ),
        "mail": (
            "Format as a clean, well-structured email with natural paragraphs, clear punctuation, "
            "and appropriate greeting/sign-off if implied by the speaker."
        ),
        "note": (
            "Format as clean markdown notes. Use concise bullet points or short headings if the "
            "speaker lists multiple items; otherwise clean paragraphs."
        ),
        "code": (
            "Format as a crisp technical prompt or developer note. Preserve identifiers, file paths, "
            "CLI flags, code symbols, and technical terms accurately; remove spoken filler words."
        ),
    }
    rule = mode_rules.get(mode, mode_rules["clean"])
    app_hint = f"Target app: {app_name}.\n" if app_name else ""
    return (
        "You are a voice-dictation formatter. Clean up the dictated transcript according to the mode.\n"
        f"Mode ({mode}): {rule}\n"
        f"{app_hint}"
        "Rules:\n"
        "- Output ONLY the final text to paste at the cursor.\n"
        "- Never add commentary, quotes around the whole answer, or answers to questions in the dictation.\n"
        "- Never invent facts the speaker did not say.\n\n"
        f"Dictation:\n{raw}"
    )


def strip_llm_wrapper(text: str, fallback: str) -> str:
    """Clean accidental quotes or chatty preambles from an LLM formatting pass."""
    out = text.strip()
    if not out:
        return fallback
    out = _PREAMBLE_RE.sub("", out).strip()
    if len(out) >= 2 and ((out[0] == '"' and out[-1] == '"') or (out[0] == "'" and out[-1] == "'")):
        inner = out[1:-1].strip()
        if inner:
            out = inner
    if out.startswith("```") and out.endswith("```"):
        lines = out.splitlines()
        if len(lines) >= 3:
            out = "\n".join(lines[1:-1]).strip()
    return out or fallback


async def capture_active_context(*, read_selection: bool = True) -> ActiveContext:
    """Inspect focused Hyprland window and optional Wayland primary selection."""
    from omi.tools.hyprland import hypr_query

    win_class = ""
    win_title = ""
    try:
        win = await asyncio.to_thread(hypr_query, "activewindow")
        if isinstance(win, dict):
            win_class = str(win.get("class") or win.get("initialClass") or "")
            win_title = str(win.get("title") or "")
    except (OSError, ValueError):
        pass

    is_term = bool(DEFAULT_APP_MODES[0][0].search(win_class)) if win_class else focused_is_terminal()
    selection = ""
    if read_selection:
        try:
            rc, out, _ = await run(["wl-paste", "--primary", "--no-newline", "--type", "text"], timeout=0.6)
            if rc == 0 and out.strip():
                selection = out[:4000]
        except (OSError, TimeoutError):
            pass
    return ActiveContext(
        window_class=win_class,
        window_title=win_title,
        is_terminal=is_term,
        selection=selection,
    )


async def _clipboard_read_text() -> str | None:
    try:
        rc, out, _ = await run(["wl-paste", "--no-newline", "--type", "text"], timeout=0.6)
        return out if rc == 0 else None
    except (OSError, TimeoutError):
        return None


async def _clipboard_write_text(text: str) -> bool:
    import subprocess

    try:
        proc = await asyncio.create_subprocess_exec(
            "wl-copy",
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.communicate(text.encode("utf-8")), 1.5)
        return proc.returncode == 0
    except (OSError, TimeoutError):
        return False


def paste_combo_for(*, is_terminal: bool) -> str:
    return "ctrl+shift+v" if is_terminal else "ctrl+v"


async def insert_text(
    text: str,
    *,
    is_terminal: bool = False,
    method: str = "auto",
    restore_clipboard: bool = True,
) -> tuple[bool, str]:
    """Insert dictated text into the focused Wayland window cleanly and fast."""
    if not text:
        return False, "empty"

    use_wtype_direct = method == "wtype" or (
        method == "auto" and len(text) <= 24 and "\n" not in text and text.isascii()
    )
    if use_wtype_direct:
        try:
            rc, _, _ = await run(build_type_text(text), timeout=3.0)
            if rc == 0:
                return True, "wtype"
        except (OSError, TimeoutError):
            pass
        if method == "wtype":
            return False, "wtype"

    # Clipboard-preserving instant paste (safe for unicode, multiline, and long dictations).
    prev_clip = await _clipboard_read_text() if restore_clipboard else None
    if await _clipboard_write_text(text):
        await asyncio.sleep(0.03)
        combo = paste_combo_for(is_terminal=is_terminal)
        try:
            rc, _, _ = await run(build_press_keys(combo), timeout=2.0)
            if rc == 0:
                if restore_clipboard and prev_clip is not None:
                    asyncio.create_task(_restore_clipboard_later(prev_clip))
                return True, f"paste:{combo}"
        except (OSError, TimeoutError):
            pass

    # Last-resort fallback to direct wtype if paste failed.
    try:
        rc, _, _ = await run(build_type_text(text), timeout=4.0)
        return rc == 0, "wtype-fallback"
    except (OSError, TimeoutError):
        return False, "failed"


async def _restore_clipboard_later(previous: str, delay_s: float = 0.22) -> None:
    await asyncio.sleep(delay_s)
    await _clipboard_write_text(previous)


class MediaPauser:
    """Auto-pause active MPRIS media while dictating, then resume only what Omi paused."""

    def __init__(self) -> None:
        self._paused_player: str | None = None

    async def pause_if_playing(self) -> bool:
        from omi.tools.media import build_media_call, pick_player

        self._paused_player = None
        try:
            picked = await pick_player()
        except Exception:
            return False
        if picked is None:
            return False
        player, status, _ = picked
        if status != "Playing":
            return False
        try:
            rc, _, _ = await run(build_media_call(player, "pause"), timeout=1.2)
            if rc == 0:
                self._paused_player = player
                return True
        except (OSError, TimeoutError):
            pass
        return False

    async def resume_if_paused(self) -> bool:
        from omi.tools.media import build_media_call

        player = self._paused_player
        self._paused_player = None
        if not player:
            return False
        try:
            rc, _, _ = await run(build_media_call(player, "play"), timeout=1.2)
            return rc == 0
        except (OSError, TimeoutError):
            return False


async def format_dictation(
    raw_text: str,
    cfg: DictationConfig,
    ctx: ActiveContext,
    *,
    llm: Callable[[str], Awaitable[str]] | None = None,
    mode_override: str | None = None,
) -> DictationResult:
    """Process raw STT output through replacements, app-mode resolution, and optional LLM cleanup."""
    cleaned_raw = apply_replacements(raw_text, cfg.replacements)
    if not cleaned_raw:
        return DictationResult(raw=raw_text, text="", mode="raw")

    requested = (mode_override or cfg.mode or "auto").strip().lower()
    if requested not in DICTATION_MODES:
        requested = "auto"

    rewrote_selection = False
    if requested == "auto":
        if cfg.use_selection and ctx.selection.strip() and looks_like_rewrite_command(cleaned_raw):
            effective_mode = "rewrite"
            rewrote_selection = True
        else:
            effective_mode = resolve_app_mode(ctx.window_class, ctx.window_title, cfg.app_modes)
    else:
        effective_mode = requested

    if effective_mode == "raw" or llm is None:
        return DictationResult(raw=cleaned_raw, text=cleaned_raw, mode=effective_mode)

    prompt = build_format_prompt(
        cleaned_raw,
        effective_mode,
        app_name=ctx.window_class or ctx.window_title,
        selection=ctx.selection if rewrote_selection else "",
    )
    try:
        formatted = await asyncio.wait_for(llm(prompt), timeout=LLM_TIMEOUT_S)
        final = apply_replacements(strip_llm_wrapper(formatted, cleaned_raw), cfg.replacements)
        return DictationResult(
            raw=cleaned_raw,
            text=final,
            mode=effective_mode,
            rewrote_selection=rewrote_selection,
        )
    except Exception as exc:
        log.warning("dictation formatting (%s) failed, falling back to raw text: %s", effective_mode, exc)
        return DictationResult(
            raw=cleaned_raw,
            text=cleaned_raw,
            mode="raw",
            error=str(exc),
        )
