"""Safety classification for shell commands and for Claude Code tool calls.

classify_command sorts a shell command into read / reversible / destructive.
secret_reason flags commands that would expose secrets; those are never run, not
even with the user's yes. classify_claude_tool turns both into the allow / confirm /
deny answer the hark MCP "approve" tool gives Claude Code.
"""

from __future__ import annotations

import functools
import os
import re
import shlex
from pathlib import Path
from typing import Any, Literal

from hark.tools.base import Safety

Decision = Literal["allow", "confirm", "deny"]
HOME = Path.home()

# --- secret paths ---

_SECRET_NAMES = re.compile(
    r"""^(
        \.env(\..*)?|.*\.env|secrets?\.(env|ya?ml|json|toml)|\.netrc|\.pgpass|\.git-credentials|
        \.npmrc|\.pypirc|\.fizzy\.ya?ml|credentials(\..*)?|.*credentials.*\.json|
        id_(rsa|dsa|ecdsa|ed25519)(_sk)?|.*\.(pem|key|p12|pfx|keystore|jks|kdbx|gpg|asc)|
        age\.txt|keys\.txt|hosts\.yml|token(s)?(\.(txt|json))?|.*_token|auth\.json
    )$""",
    re.X | re.I,
)
# Directories whose contents are credentials (relative to home).
_SECRET_DIRS = (".ssh", ".gnupg", ".aws", ".password-store", ".config/chezmoi", ".kube",
                ".docker", ".config/gh", ".local/share/keyrings", ".azure", ".config/gcloud")
_SSH_PUBLIC = re.compile(r".*\.pub$|^known_hosts.*$|^config$|^authorized_keys$")
# Files an agent may read but must not change without asking.
_SENSITIVE_EDITS = re.compile(r"^(\.bashrc|\.zshrc|\.profile|\.bash_profile|\.zprofile|settings(\.local)?\.json)$")


def _expand(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path.strip("'\"")))).absolute()


def is_secret_path(path: str) -> bool:
    if not path:
        return False
    p = _expand(path)
    name = p.name
    if name.endswith((".example", ".sample", ".template", ".dist")) and not name.startswith(".env"):
        return False
    if _SECRET_NAMES.match(name):
        return True
    try:
        rel = p.relative_to(HOME).as_posix()
    except ValueError:
        return str(p).startswith(("/etc/shadow", "/etc/gshadow", "/etc/sudoers"))
    for d in _SECRET_DIRS:
        if rel == d or rel.startswith(d + "/"):
            return not (d == ".ssh" and _SSH_PUBLIC.match(name))
    return False


def in_secret_dir(path: str) -> bool:
    """Under a credential directory like ~/.ssh, public files included."""
    try:
        rel = _expand(path).relative_to(HOME).as_posix()
    except ValueError:
        return False
    return any(rel == d or rel.startswith(d + "/") for d in _SECRET_DIRS)


def inside_home(path: str) -> bool:
    try:
        _expand(path).resolve().relative_to(HOME)
        return True
    except ValueError:
        return False


# --- shell parsing ---

_SPLIT = re.compile(r"\|\||&&|[;\n|&]")
_SUBST = re.compile(r"\$\(([^()]*)\)|`([^`]*)`|<\(([^()]*)\)")
_WRAPPERS = {"time", "nice", "nohup", "command", "builtin", "exec", "stdbuf", "ionice", "timeout", "env"}
_SHELLS = {"sh", "bash", "zsh", "fish", "dash", "ksh"}
_SECRET_VAR = re.compile(r"\$\{?[A-Z0-9_]*(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|PRIVATE_KEY|CREDENTIAL)[A-Z0-9_]*\}?", re.I)

_READ_CMDS = {
    "ls", "cat", "head", "tail", "less", "more", "grep", "rg", "ag", "fd", "find", "pwd", "echo", "printf",
    "wc", "which", "whereis", "type", "stat", "file", "du", "df", "ps", "pgrep", "date", "uname", "whoami",
    "id", "hostname", "uptime", "free", "lsblk", "lscpu", "lsusb", "lspci", "tree", "diff", "cmp", "sort",
    "uniq", "cut", "tr", "jq", "yq", "realpath", "dirname", "basename", "readlink", "true", "false", "test",
    "[", "nproc", "sha256sum", "md5sum", "column", "awk", "sed", "busctl", "wpctl", "pactl", "hyprctl",
    "journalctl", "man", "tldr", "bat", "eza", "zoxide", "cal", "seq", "sleep", "xxd", "od", "strings",
}
_DESTRUCTIVE_CMDS = {
    "rm", "rmdir", "shred", "dd", "wipefs", "fdisk", "sfdisk", "parted", "gdisk", "mkswap", "truncate",
    "sudo", "pkexec", "doas", "su", "run0", "kill", "pkill", "killall", "xkill",
    "shutdown", "reboot", "poweroff", "halt", "suspend", "hibernate",
    "pacman", "yay", "paru", "apt", "apt-get", "dnf", "zypper", "flatpak", "snap", "brew", "makepkg",
    "mail", "mailx", "sendmail", "mutt", "neomutt", "msmtp", "himalaya", "signal-cli", "telegram-send",
    "ssh", "scp", "sftp", "crontab", "chsh", "passwd", "useradd", "userdel", "usermod", "mount", "umount",
    "omarchy-system", "omarchy-update", "chezmoi", "systemd-run",
}
_PKG_SUBCMDS = {"install", "i", "add", "remove", "rm", "uninstall", "un", "update", "upgrade", "up",
                "publish", "sync", "link", "unlink", "ci", "global"}
_PKG_TOOLS = {"pip", "pip3", "uv", "npm", "pnpm", "yarn", "bun", "cargo", "gem", "go", "mise", "pipx", "npx", "composer"}
_SYSTEMCTL_BAD = {"stop", "disable", "mask", "kill", "poweroff", "reboot", "suspend", "hibernate", "halt",
                  "isolate", "kexec", "emergency", "rescue", "default", "hybrid-sleep", "suspend-then-hibernate",
                  "daemon-reexec", "reset-failed"}
_SYSTEMCTL_READ = {"status", "is-active", "is-enabled", "is-failed", "list-units", "list-unit-files",
                   "list-timers", "show", "cat", "list-sockets", "list-dependencies"}
_GIT_READ = {"status", "log", "diff", "show", "blame", "grep", "ls-files", "rev-parse", "describe",
             "shortlog", "reflog", "remote", "config", "branch", "tag", "fetch"}
_GH_BAD = re.compile(r"\b(create|comment|merge|close|delete|edit|review|reopen|ready|lock|transfer|archive|"
                     r"rename|upload|set|remove|add|fork|sync)\b")
_RANK: dict[Safety, int] = {"read": 0, "reversible": 1, "destructive": 2}


def _segments(cmd: str) -> list[str]:
    """Split a command line into simple commands, including ones inside $(), `` and <()."""
    cmd = re.sub(r"\d*>&(\d+|-)", " ", cmd).replace("&>", ">")
    inner = [a or b or c for a, b, c in _SUBST.findall(cmd)]
    outer = _SUBST.sub(" SUBST ", cmd)
    parts = [s.strip() for s in _SPLIT.split(outer) if s.strip()]
    for sub in inner:
        parts += _segments(sub)
    return parts


def _words(segment: str) -> list[str]:
    try:
        words = shlex.split(segment, comments=True)
    except ValueError:
        words = segment.split()
    while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
        words = words[1:]  # VAR=value prefix
    while words and words[0] in _WRAPPERS:
        words = words[1:]
        while words and (words[0].startswith("-") or re.match(r"^[A-Za-z_]\w*=|^\d+[smhd]?$", words[0])):
            words = words[1:]
    return words


def secret_reason(cmd: str) -> str | None:
    """Why this command would expose a secret, or None."""
    low = cmd.lower()
    if re.search(r"\bheroku\s+(config|run|pg:credentials|auth:token|releases:info)", low):
        return "it prints Heroku credentials"
    if _SECRET_VAR.search(cmd):
        return "it reads a secret variable"
    for segment in _segments(cmd):
        words = _words(segment)
        raw = segment.split()
        if raw and Path(raw[0]).name == "env" and not words:
            return "it dumps the environment"
        if not words:
            continue
        name = Path(words[0]).name
        if name in ("env", "printenv") or (name == "set" and len(words) == 1):
            return "it dumps the environment"
        if name in ("export", "declare", "typeset") and len(words) == 2 and words[1] in ("-p", "-x", "-px"):
            return "it dumps the environment"
        if name == "mise" and words[1:2] == ["env"] or name == "launchctl" and "getenv" in words:
            return "it dumps the environment"
        for w in [*words, *re.findall(r"[<>]\s*(\S+)", segment)]:
            if w.startswith("-") and "=" in w:
                w = w.split("=", 1)[1]
            if not w.startswith("-") and is_secret_path(w):
                return f"it touches the secret file {Path(w).name}"
    return None


def _writable(path: str) -> bool:
    """Places an agent may write without asking: home and /tmp."""
    return inside_home(path) or _expand(path).as_posix().startswith("/tmp/")


def _redirect_targets(segment: str) -> list[str]:
    targets = re.findall(r"(?<![<&0-9])>>?\s*([^\s;&|]+)", segment)
    return [t for t in targets if not t.startswith("&") and t not in ("/dev/null", "/dev/stdout", "/dev/stderr")]


def _mv_overwrites(words: list[str]) -> bool:
    args = [w for w in words[1:] if not w.startswith("-")]
    if "-n" in words or "--no-clobber" in words:
        return False
    if len(args) < 2:
        return True
    dest = _expand(args[-1])
    if any(not _writable(a) for a in args):
        return True
    if dest.is_dir():
        return any((dest / _expand(src).name).exists() for src in args[:-1])
    return dest.exists()


def _classify_segment(segment: str) -> Safety:
    level = _classify_words(segment)
    targets = _redirect_targets(segment)
    if any(not _writable(t) for t in targets):
        return "destructive"
    if targets and level == "read":
        return "reversible"
    return level


def _classify_words(segment: str) -> Safety:
    words = _words(segment)
    if not words:
        return "read"
    name = Path(words[0]).name
    rest = words[1:]
    sub = next((w for w in rest if not w.startswith("-")), "")

    if name in _DESTRUCTIVE_CMDS or name.startswith("mkfs"):
        if name in ("pacman", "yay", "paru") and rest and re.match(r"^-(Q|Ss|Si|F)", rest[0]):
            return "read"
        return "destructive"
    if name in _SHELLS or name == "eval":
        if "-c" in rest:
            return classify_command(" ".join(rest[rest.index("-c") + 1:][:1]))
        return "destructive" if name == "eval" or rest else "reversible"
    if name.startswith("python") or name in ("node", "perl", "ruby", "deno"):
        code = " ".join(rest)
        if re.search(r"rmtree|os\.remove|unlink|os\.system|subprocess|shutil\.move|child_process|\bsystem\(", code):
            return "destructive"
        return "reversible"
    if name == "mv":
        return "destructive" if _mv_overwrites(words) else "reversible"
    if name in ("cp", "install", "ln", "tee", "touch", "mkdir", "chmod", "chown", "chgrp", "rsync"):
        targets = [w for w in rest if not w.startswith("-")]
        if name == "rsync" and any(re.match(r"^[\w.@-]+:", t) for t in targets):
            return "destructive"
        if name == "rsync" and "--delete" in " ".join(rest):
            return "destructive"
        if name in ("chmod", "chown", "chgrp") and "-R" in rest:
            return "destructive"
        check = targets if name in ("tee", "touch", "mkdir", "chmod", "chown", "chgrp") else targets[-1:]
        return "destructive" if any(not _writable(t) for t in check) else "reversible"
    if name == "find":
        return "destructive" if {"-delete", "-exec", "-execdir", "-ok"} & set(rest) else "read"
    if name == "sed":
        return "reversible" if any(w.startswith("-i") for w in rest) else "read"
    if name == "git":
        flags = set(rest)
        if sub in ("push", "clean", "filter-branch", "filter-repo", "gc", "prune"):
            return "destructive"
        if sub == "reset" and "--hard" in flags or sub == "checkout" and ({"-f", "--force", "--", "."} & flags):
            return "destructive"
        if sub == "branch" and {"-D", "-d", "--delete"} & flags or sub == "stash" and {"drop", "clear"} & flags:
            return "destructive"
        if sub == "restore" or sub == "rebase" or sub == "update-ref":
            return "destructive"
        if sub in _GIT_READ and not ({"-d", "-D", "-m", "-M", "--delete"} & flags) and sub != "fetch":
            return "read"
        return "reversible"
    if name == "gh":
        if sub == "api":
            return "destructive" if re.search(r"(-X|--method)\s*(POST|PUT|PATCH|DELETE)|-f\b|-F\b|--field|--input", segment) else "read"
        return "destructive" if _GH_BAD.search(" ".join(rest[:3])) else "read"
    if name in ("curl", "wget", "http", "xh"):
        if re.search(r"\s(-d|--data\S*|-F|--form\S*|-T|--upload-file|--post-\w+|--json)\b|(-X|--request)\s*(POST|PUT|PATCH|DELETE)", segment):
            return "destructive"  # sends data somewhere
        if name == "wget" or re.search(r"\s(-o|-O|--output)\b", segment):
            return "reversible"  # downloads a file
        return "read"
    if name in _PKG_TOOLS:
        if name == "uv" and sub == "pip":
            sub = next((w for w in rest[1:] if not w.startswith("-")), "")
        if name == "npx" or sub in _PKG_SUBCMDS or (name == "go" and sub == "get"):
            return "destructive"
        return "reversible"
    if name == "systemctl" or name == "loginctl":
        if sub in _SYSTEMCTL_BAD or name == "loginctl" and sub in ("terminate-session", "terminate-user", "kill-session", "kill-user", "poweroff", "reboot", "suspend"):
            return "destructive"
        return "read" if sub in _SYSTEMCTL_READ or name == "loginctl" and sub.startswith(("list", "show")) else "reversible"
    if name == "hyprctl":
        return "destructive" if sub in ("kill", "dispatch", "keyword", "eval", "reload") else "read"
    if name.startswith("omarchy"):
        from hark.tools.omarchy import command_safety

        return command_safety(name)
    if name in _READ_CMDS:
        return "read"
    return "reversible"


def classify_command(cmd: str) -> Safety:
    """read / reversible / destructive for a shell command line. Secret-exposing
    commands count as destructive here; secret_reason is what hard-denies them."""
    if secret_reason(cmd):
        return "destructive"
    segments = _segments(cmd)
    # Piping a download into a shell runs code nobody has read.
    names = [Path((_words(s) or [""])[0]).name for s in segments]
    if any(n in ("curl", "wget") for n in names) and any(n in _SHELLS or n.startswith("python") for n in names[1:]):
        return "destructive"
    worst: Safety = "read"
    for segment in segments:
        level = _classify_segment(segment)
        if _RANK[level] > _RANK[worst]:
            worst = level
            if worst == "destructive":
                break
    return worst


# --- Claude Code tools ---

_ALWAYS_ALLOW = {"WebFetch", "WebSearch", "TodoWrite", "ToolSearch", "Task", "Agent"}


@functools.cache
def _hark_safety() -> dict[str, Safety]:
    from hark.config import Config
    from hark.tools import build_registry
    from hark.tools.mcp_server import daemon_tools

    tools = [*build_registry(Config()).tools.values(), *daemon_tools()]
    return {tool.name: tool.safety for tool in tools}


def _hark_tool_safety(name: str, tool_input: dict[str, Any]) -> Safety | None:
    if name == "omarchy":
        from hark.tools.omarchy import command_safety

        return command_safety(str(tool_input.get("command", "")))
    if name == "approve":
        return "read"
    return _hark_safety().get(name)


def classify_claude_tool(tool_name: str, tool_input: dict[str, Any]) -> tuple[Decision, str]:
    """allow / confirm / deny for one Claude Code tool call, with a short reason."""
    tool_input = tool_input or {}
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        if reason := secret_reason(cmd):
            return "deny", f"Blocked because {reason}."
        level = classify_command(cmd)
        if level == "destructive":
            return "confirm", "This command can change or delete things."
        return "allow", f"{level} command"
    if tool_name in ("Read", "Grep", "Glob", "LS", "NotebookRead"):
        paths = [str(tool_input.get(k, "")) for k in ("file_path", "path", "notebook_path")]
        paths.append(str(tool_input.get("pattern", "")) if tool_name == "Glob" else "")
        paths.append(str(tool_input.get("glob", "")))
        for p in paths:
            if p and any(is_secret_path(p.replace("*", fill)) for fill in ("", "x")):
                return "deny", f"Blocked because {Path(p).name} may hold secrets."
        return "allow", "reading"
    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        if not path:
            return "confirm", "Edit with no path."
        if is_secret_path(path) or in_secret_dir(path):
            return "deny", f"Blocked because {Path(path).name} holds secrets."
        if not inside_home(path):
            return "confirm", f"This writes {path}, outside the home folder."
        if _SENSITIVE_EDITS.match(Path(path).name):
            return "confirm", f"This changes {Path(path).name}, a settings file."
        return "allow", "editing a file in the home folder"
    if tool_name in _ALWAYS_ALLOW:
        return "allow", "safe built-in tool"
    if tool_name.startswith("mcp__hark__"):
        safety = _hark_tool_safety(tool_name.removeprefix("mcp__hark__"), tool_input)
        if safety is None:
            return "confirm", "Unknown hark tool."
        if safety == "destructive":
            return "confirm", "This hark action is hard to undo."
        return "allow", f"{safety} hark tool"
    return "confirm", f"{tool_name} is not a tool hark knows."
