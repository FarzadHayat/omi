"""Safety classification for shell commands and for Claude Code tool calls.

classify_command sorts a shell command into read / reversible / destructive. A
command is "read" only when every program in it is on a vetted read-only list with
arguments that keep it read-only, with no output redirection, no command
substitution, no unresolved variables and no secret paths. Anything else, including
any program omi does not know, is at least "reversible". secret_reason flags
commands that would expose secrets; those are never run, not even with the user's
yes. classify_claude_tool turns both into the allow / confirm / deny answer the omi
MCP "approve" tool gives Claude Code: only read commands run without asking.

Paths are expanded and resolved (~, $VARS, "..", symlinks) before any check, so a
detour like ~/Work/../.ssh is seen as ~/.ssh.
"""

from __future__ import annotations

import functools
import glob
import itertools
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from omi.tools.base import Safety

Decision = Literal["allow", "confirm", "deny"]
HOME = Path(os.path.realpath(Path.home()))
OMI_ROOT = Path(__file__).resolve().parents[2]  # omi's own code


def _shared_memory_index() -> Path | None:
    from omi import config

    try:
        shared = config.load().memory.shared_dir
    except Exception:
        return None
    return _expand(str(Path(shared) / "MEMORY.md")) if shared else None

# --- secret paths ---

_SECRET_NAMES = re.compile(
    r"""^(
        \.env(\..*)?|.*\.env|secrets?\.(env|ya?ml|json|toml)|\.netrc|\.pgpass|\.git-credentials|
        \.npmrc|\.pypirc|\.fizzy\.ya?ml|credentials(\..*)?|.*credentials.*\.json|
        id_(rsa|dsa|ecdsa|ed25519)(_sk)?|.*\.(pem|key|p12|pfx|keystore|jks|kdbx|gpg|asc|keytab)|
        age\.txt|keys\.txt|hosts\.yml|token(s)?(\.(txt|json))?|.*_token|auth\.json|
        rclone\.conf|cookies(\.sqlite)?|login\ data|logins\.json|key[34]\.db|\.htpasswd
    )$""",
    re.X | re.I,
)
# Directories whose contents are credentials (relative to home).
_SECRET_DIRS = (".ssh", ".gnupg", ".aws", ".password-store", ".config/chezmoi", ".kube",
                ".docker", ".config/gh", ".local/share/keyrings", ".azure", ".config/gcloud",
                ".config/rclone", ".config/op", ".config/1Password", ".config/Bitwarden CLI",
                ".config/sops", ".mozilla", ".thunderbird", ".config/chromium",
                ".config/google-chrome", ".config/BraveSoftware", ".config/vivaldi",
                ".config/microsoft-edge")
_SSH_PUBLIC = re.compile(r".*\.pub$|^known_hosts.*$|^config$|^authorized_keys$")
_PROC_ENV = re.compile(r"^/proc/[^/]+/(task/[^/]+/)?environ$")


def _expand(path: str, base: str | None = None) -> Path:
    """~, $VARS, relative parts and symlinks resolved, so every check sees one path."""
    p = os.path.expandvars(os.path.expanduser(path.strip("'\"")))
    if base and not os.path.isabs(p):
        p = os.path.join(base, p)
    return Path(os.path.realpath(p))


def _rel_home(p: Path) -> str | None:
    try:
        return p.relative_to(HOME).as_posix()
    except ValueError:
        return None


def is_secret_path(path: str, base: str | None = None) -> bool:
    if not path:
        return False
    p = _expand(path, base)
    names = {p.name, Path(os.path.normpath(path)).name}
    if all(n.endswith((".example", ".sample", ".template", ".dist")) and not n.startswith(".env") for n in names):
        return False
    if any(_SECRET_NAMES.match(n) for n in names):
        return True
    s = p.as_posix()
    if _PROC_ENV.match(s) or s.startswith(("/etc/shadow", "/etc/gshadow", "/etc/sudoers")):
        return True
    rel = _rel_home(p)
    if rel is None:
        return False
    for d in _SECRET_DIRS:
        if rel == d or rel.startswith(d + "/"):
            return not (d == ".ssh" and _SSH_PUBLIC.match(p.name))
    return False


def in_secret_dir(path: str) -> bool:
    """Under a credential directory like ~/.ssh, public files included."""
    rel = _rel_home(_expand(path))
    return rel is not None and any(rel == d or rel.startswith(d + "/") for d in _SECRET_DIRS)


def covers_secrets(path: str, base: str | None = None) -> bool:
    """A search rooted here would reach into a credential directory (home, ~/.config, /)."""
    root = _expand(path or ".", base).as_posix().rstrip("/") + "/"
    return any((HOME / d).as_posix().startswith(root) for d in _SECRET_DIRS)


def inside_home(path: str) -> bool:
    return _rel_home(_expand(path)) is not None


def _candidates(word: str, base: str | None) -> list[str]:
    """The word, plus what it matches when it is a glob, so ~/.ss?/id_* is caught."""
    if not any(c in word for c in "*?["):
        return [word]
    out = [word.replace("*", ""), word.replace("*", "x")]
    pattern = os.path.expandvars(os.path.expanduser(word))
    if base and not os.path.isabs(pattern):
        pattern = os.path.join(base, pattern)
    try:
        out += itertools.islice(glob.iglob(pattern, recursive="**" in pattern, include_hidden=True), 200)
    except (OSError, ValueError):
        pass
    return out


# --- shell parsing ---

_PUNCT = ";&|()<>\n"
_SEPARATORS = set(";&|()\n")
_WRAPPERS = {"time", "nice", "nohup", "command", "builtin", "exec", "stdbuf", "ionice", "timeout", "env",
             "chronic", "setsid", "unbuffer"}
_SHELLS = {"sh", "bash", "zsh", "fish", "dash", "ksh"}
_SECRET_VAR = re.compile(r"\$\{?[A-Z0-9_]*(TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|PRIVATE_KEY|CREDENTIAL)[A-Z0-9_]*\}?", re.I)
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_BRACES = re.compile(r"\{[^{}]*(,|\.\.)[^{}]*\}")


@dataclass
class _Seg:
    words: list[str]  # the simple command, VAR=value prefixes included
    redirects: list[tuple[str, str]]  # (operator, target)


def _tokens(cmd: str) -> list[str] | None:
    """shlex words, with ; & | ( ) < > and newlines as their own tokens. None when
    the quoting does not parse. '#' is never a comment here: reading too much is safe,
    reading too little is not."""
    lx = shlex.shlex(cmd, posix=True, punctuation_chars=_PUNCT)
    lx.whitespace = " \t\r"
    lx.whitespace_split = True
    lx.commenters = ""
    try:
        return list(lx)
    except ValueError:
        return None


def _is_op(token: str) -> bool:
    return bool(token) and all(c in _PUNCT for c in token)


def _split(tokens: list[str]) -> list[_Seg]:
    segs: list[_Seg] = []
    cur = _Seg([], [])
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_op(tok) and ("<" in tok or ">" in tok) and not tok.endswith("("):
            if cur.words and cur.words[-1].isdigit():
                cur.words.pop()  # "2>file": the 2 is a file descriptor
            target = tokens[i + 1] if i + 1 < len(tokens) and not _is_op(tokens[i + 1]) else ""
            cur.redirects.append((tok, target))
            i += 2 if target else 1
            continue
        if _is_op(tok):
            if cur.words or cur.redirects:
                segs.append(cur)
            cur = _Seg([], [])
        else:
            cur.words.append(tok)
        i += 1
    if cur.words or cur.redirects:
        segs.append(cur)
    return segs


def _substitutions(cmd: str) -> list[str]:
    """Inner commands of $(...), `...`, <(...) and >(...), outside single quotes."""
    out: list[str] = []
    i, n, quote = 0, len(cmd), ""
    while i < n:
        c = cmd[i]
        if c == "\\" and quote != "'":
            i += 2
            continue
        if quote == "'":
            quote = "" if c == "'" else quote
            i += 1
            continue
        if c == "'" and not quote:
            quote = "'"
        elif c == '"':
            quote = "" if quote == '"' else '"'
        elif c == "`":
            j = cmd.find("`", i + 1)
            j = n if j < 0 else j
            out.append(cmd[i + 1:j])
            i = j + 1
            continue
        elif c in "$<>" and cmd.startswith("(", i + 1):
            depth, j = 1, i + 2
            while j < n and depth:
                depth += {"(": 1, ")": -1}.get(cmd[j], 0)
                j += 1
            out.append(cmd[i + 2:j - 1] if depth == 0 else cmd[i + 2:])
            i = j
            continue
        i += 1
    return out


_KNOWN_VARS = {"HOME", "USER", "LOGNAME", "SHELL", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
               "XDG_CACHE_HOME", "XDG_RUNTIME_DIR"}


def _unknown_vars(cmd: str) -> bool:
    """A $variable outside single quotes whose value omi cannot know."""
    i, n, quote = 0, len(cmd), ""
    while i < n:
        c = cmd[i]
        if c == "\\" and quote != "'":
            i += 2
            continue
        if quote == "'":
            quote = "" if c == "'" else quote
        elif c == "'" and not quote:
            quote = "'"
        elif c == '"':
            quote = "" if quote == '"' else '"'
        elif c == "$" and i + 1 < n:
            m = re.match(r"\{?([A-Za-z_][A-Za-z0-9_]*)\}?", cmd[i + 1:])
            if m is None:
                if cmd[i + 1] not in " \t\n\"'":
                    return True  # $1, $@, $$, ${...}
            elif m.group(1) not in _KNOWN_VARS:
                return True
        i += 1
    return False


def _unwrap(words: list[str]) -> tuple[list[str], bool]:
    """Drop VAR=value prefixes and wrappers like nice or timeout. The flag is True when
    anything was set in the environment (LD_PRELOAD=..., env FOO=1), which can make
    any program run other code."""
    env = False
    while words and _ASSIGN.match(words[0]):
        words, env = words[1:], True
    while words and Path(words[0]).name in _WRAPPERS:
        if Path(words[0]).name == "env" and len(words) > 1:
            env = True
        words = words[1:]
        while words and (words[0].startswith("-") or _ASSIGN.match(words[0]) or re.match(r"^\d+(\.\d+)?[smhd]?$", words[0])):
            env = env or bool(_ASSIGN.match(words[0])) or words[0] in ("-S", "--split-string")
            words = words[1:]
    return words, env


def _shell_inner(name: str, rest: list[str]) -> str | None:
    """The command string bash -c, sh -c or eval would run."""
    if name == "eval":
        return " ".join(rest)
    if name in _SHELLS:
        for i, w in enumerate(rest):
            if w == "-c" or (w.startswith("-") and not w.startswith("--") and "c" in w):
                return rest[i + 1] if i + 1 < len(rest) else ""
    return None


# --- secrets ---

_PASSWORD_TOOLS = {"pass", "gopass", "secret-tool", "op", "bw", "rbw", "keepassxc-cli", "kwallet-query",
                   "lpass", "vault"}
_INTERPRETERS = re.compile(r"^(python[\d.]*|node|nodejs|deno|bun|perl|ruby|php|awk|gawk|mawk|lua\d*|jq|yq)$")
_ENV_CODE = re.compile(r"os\.environ|getenv|process\.env|ENVIRON|\$ENV\b|Deno\.env|%ENV|\bENV\[|(^|[|\s(])env\b")


def secret_reason(cmd: str, base: str | None = None, depth: int = 0) -> str | None:
    """Why this command would expose a secret, or None."""
    low = cmd.lower()
    if re.search(r"\bheroku\s+(config|run|pg:credentials|auth:token|releases:info)", low):
        return "it prints Heroku credentials"
    if _SECRET_VAR.search(cmd):
        return "it reads a secret variable"
    if depth > 4:
        return None
    for inner in _substitutions(cmd):
        if reason := secret_reason(inner, base, depth + 1):
            return reason
    tokens = _tokens(cmd) or cmd.split()
    for seg in _split(tokens):
        raw = seg.words
        words, _ = _unwrap(raw)
        if raw and Path(raw[0]).name == "env" and not words:
            return "it dumps the environment"
        if not words:
            continue
        name, rest = Path(words[0]).name, words[1:]
        if reason := _program_secret(name, rest):
            return reason
        inner = _shell_inner(name, rest)
        if inner and (reason := secret_reason(inner, base, depth + 1)):
            return reason
        patterns = _search_patterns(rest) if name in _SEARCHERS else set()
        checks = [w for w in words if w not in patterns] + [t for _, t in seg.redirects]
        for w in checks:
            if w.startswith("-") and "=" in w:
                w = w.split("=", 1)[1]
            if w.startswith("-"):
                continue
            for c in _candidates(w, base):
                if is_secret_path(c, base):
                    return f"it touches the secret file {Path(c).name or c}"
        # tar -C ~ .ssh, git -C ~ ...: members are relative to the -C folder.
        for i, w in enumerate(rest):
            folder = rest[i + 1] if w in ("-C", "--directory") and i + 1 < len(rest) else (
                w.split("=", 1)[1] if w.startswith("--directory=") else None)
            if folder:
                for member in rest[i + 2:]:
                    if not member.startswith("-") and is_secret_path(member, str(_expand(folder, base))):
                        return f"it touches the secret file {member}"
        if name in ("cd", "pushd") and rest:
            base = str(_expand(rest[-1], base))
    return None


_SEARCHERS = {"grep", "egrep", "fgrep", "rg", "ag"}


def _search_patterns(rest: list[str]) -> set[str]:
    """What grep or rg searches for: text, not a path (grep -r token src is fine)."""
    given = {rest[i + 1] for i, w in enumerate(rest) if w in ("-e", "--regexp") and i + 1 < len(rest)}
    return given or set(_positionals(rest, {"-m", "-A", "-B", "-C", "-g", "--glob", "-t", "--type"})[:1])


def _program_secret(name: str, rest: list[str]) -> str | None:
    if name in ("env", "printenv") or (name == "set" and not rest):
        return "it dumps the environment"
    if name in ("export", "declare", "typeset") and len(rest) == 1 and rest[0] in ("-p", "-x", "-px"):
        return "it dumps the environment"
    if name == "mise" and rest[:1] == ["env"] or name == "launchctl" and "getenv" in rest:
        return "it dumps the environment"
    if name == "direnv" and rest[:1] and rest[0] in ("dump", "export", "exec"):
        return "it dumps the environment"
    if name == "systemctl" and "show-environment" in rest:
        return "it dumps the environment"
    if name == "ps" and any(not w.startswith("-") and "e" in w and re.fullmatch(r"[a-zA-Z]+", w) for w in rest):
        return "it shows the environment of running programs"
    if name in _PASSWORD_TOOLS:
        return "it reads the password store"
    if name == "gh" and (rest[:2] == ["auth", "token"] or rest[:1] == ["secret"]
                         or rest[:2] == ["auth", "status"] and {"-t", "--show-token"} & set(rest)):
        return "it prints GitHub credentials"
    if name in ("gpg", "gpg2") and any(w.startswith("--export-secret") or w in ("-d", "--decrypt") for w in rest):
        return "it exports or decrypts secret keys"
    if _INTERPRETERS.match(name) and any(_ENV_CODE.search(w) for w in rest):
        return "it reads environment variables"
    return None


# --- read-only vetting ---

_PLAIN_READ = {
    "ls", "cat", "head", "tail", "less", "more", "wc", "pwd", "echo", "printf", "which", "whereis", "type",
    "stat", "du", "df", "pgrep", "uname", "whoami", "id", "uptime", "free", "lsblk", "lscpu", "lsusb",
    "lspci", "cmp", "diff", "cut", "tr", "realpath", "dirname", "basename", "readlink", "true", "false",
    "test", "[", "nproc", "sha256sum", "sha1sum", "sha512sum", "md5sum", "b2sum", "column", "bat", "eza",
    "cal", "seq", "sleep", "od", "strings", "man", "whatis", "apropos", "cd", "pushd", "popd", "nl", "rev",
    "fold", "comm", "join", "paste", "expand", "unexpand", "file", "lsof", "ss", "sensors", "tac", "ps",
}
_GIT_READ = {"status", "log", "diff", "show", "blame", "ls-files", "ls-tree", "rev-parse", "describe",
             "shortlog", "cat-file", "rev-list", "merge-base", "name-rev", "for-each-ref", "show-ref",
             "count-objects", "check-ignore", "version", "help", "grep"}
_GIT_GLOBAL_ARG = {"-C", "--git-dir", "--work-tree", "--namespace"}
_GIT_GLOBAL_FLAG = {"--no-pager", "-P", "--no-replace-objects", "--literal-pathspecs", "--bare", "--no-optional-locks"}
_SYSTEMCTL_READ = {"status", "is-active", "is-enabled", "is-failed", "list-units", "list-unit-files",
                   "list-timers", "show", "cat", "list-sockets", "list-dependencies", "list-jobs", "help",
                   "get-default"}
_HYPRCTL_READ = {"clients", "activewindow", "activeworkspace", "workspaces", "monitors", "devices", "layers",
                 "version", "binds", "getoption", "cursorpos", "globalshortcuts", "instances", "splash",
                 "systeminfo", "animations", "rollinglog", "workspacerules", "layouts", "configerrors",
                 "decorations", "getprop"}
_GH_READ_VERBS = {"view", "list", "status", "diff", "checks"}
_CURL_SAFE_FLAG = re.compile(r"^(-[sSfLIiv]+|--(silent|show-error|fail|location|head|include|verbose|compressed|"
                             r"fail-with-body|no-progress-meter))$")
_CURL_SAFE_OPT = {"-m", "--max-time", "--connect-timeout", "--retry", "-w", "--write-out"}


def _positionals(rest: list[str], with_value: set[str] = frozenset()) -> list[str]:
    out, skip = [], False
    for w in rest:
        if skip:
            skip = False
        elif w in with_value:
            skip = True
        elif not w.startswith("-") or w == "-":
            out.append(w)
    return out


def _web_url_ok(url: str) -> bool:
    """A plain web address with no query string or login, so it carries no data out."""
    parsed = urlparse(url if "://" in url else "https://" + url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc) and not parsed.query \
        and "@" not in parsed.netloc


def _search_ok(rest: list[str], base: str | None, with_value: set[str], recursive: bool) -> bool:
    """grep -r, rg and ag must not sweep a folder that holds credentials."""
    if not recursive:
        return True
    given = any(w in ("-e", "-f", "--regexp", "--file") or w.startswith(("--regexp=", "--file=")) for w in rest)
    roots = _positionals(rest, with_value)[0 if given else 1:] or ["."]
    return not any(covers_secrets(r, base) for r in roots)


def _sed_ok(rest: list[str]) -> bool:
    scripts: list[str] = []
    explicit, i = False, 0
    while i < len(rest):
        w = rest[i]
        if w.startswith("--"):
            if w.startswith(("--in-place", "--file", "--separate")):
                return False
            if w == "--expression":
                scripts.append(rest[i + 1] if i + 1 < len(rest) else "")
                explicit, i = True, i + 1
            elif w.startswith("--expression="):
                scripts.append(w.split("=", 1)[1])
                explicit = True
        elif w.startswith("-") and len(w) > 1:
            if set(w[1:]) & set("ifs"):
                return False  # -i edits in place (also inside -ni), -f reads a script file
            if "e" in w[1:]:
                scripts.append(rest[i + 1] if i + 1 < len(rest) else "")
                explicit, i = True, i + 1
        i += 1
    if not explicit:
        scripts = _positionals(rest)[:1]
    addr = r"(\d+|\$|/[^/]*/)(,(\d+|\$|/[^/]*/|\+\d+))?"
    subst = r"s(?P<d>[^\\\n\w\s])(?:(?!(?P=d)).|\\.)*(?P=d)(?:(?!(?P=d)).|\\.)*(?P=d)[gpiImM0-9]*"
    for script in scripts:
        for part in re.split(r"[;\n]", script):
            part = part.strip()
            if part and not re.fullmatch(rf"({addr})?\s*!?\s*({subst}|[pdq=nNlgGhHxzD]|)\s*", part):
                return False
    return True


def _awk_ok(rest: list[str]) -> bool:
    if any(w.startswith(("-f", "--file", "-i", "--include", "-l", "--load", "-E", "--exec")) for w in rest):
        return False
    return not any(re.search(r"system|getline|[|>]|ENVIRON|PROCINFO|close\(|fflush", w) for w in rest)


def _git_read(rest: list[str]) -> bool:
    i = 0
    while i < len(rest):
        w = rest[i]
        if w in _GIT_GLOBAL_ARG:
            i += 2
        elif w.split("=", 1)[0] in _GIT_GLOBAL_ARG or w in _GIT_GLOBAL_FLAG:
            i += 1
        elif w.startswith("-"):
            return False  # -c sets config (core.fsmonitor runs code), --exec-path, ...
        else:
            break
    sub, args = (rest[i], rest[i + 1:]) if i < len(rest) else ("", [])
    flags = set(args)
    if any(a.startswith(("--output", "--ext-diff", "-O", "--open-files-in-pager", "--exec")) for a in args):
        return False
    if sub in _GIT_READ:
        return True
    if sub == "branch":
        return not _positionals(args) or bool({"-l", "--list", "--contains", "--merged", "--no-merged"} & flags) \
            and not ({"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "-f", "--force", "-u",
                      "--set-upstream-to", "--unset-upstream", "--edit-description"} & flags)
    if sub == "tag":
        return not _positionals(args) or bool({"-l", "--list"} & flags) and not ({"-d", "--delete", "-f", "-a", "-s", "-m"} & flags)
    if sub == "remote":
        return not args or args in (["-v"], ["--verbose"]) or args[:1] in (["get-url"], ["show"])
    if sub == "stash":
        return args[:1] in (["list"], ["show"])
    if sub == "config":
        return bool({"--get", "--get-all", "--get-regexp", "--list", "-l"} & flags)
    if sub == "worktree":
        return args[:1] == ["list"]
    if sub == "reflog":
        return not args or args[0] == "show" or args[0].startswith("-")
    if sub == "submodule":
        return args[:1] == ["status"]
    return False


def _read_ok(name: str, rest: list[str], base: str | None) -> bool:
    """True only for a program and arguments known to change nothing."""
    flags = set(rest)
    if name in _PLAIN_READ:
        if name == "file":
            return "-C" not in flags
        if name == "ps":
            return all(w.startswith("-") or "e" not in w for w in rest)
        return True
    if name in ("grep", "egrep", "fgrep"):
        recursive = any(re.fullmatch(r"-[a-zA-Z]*[rR][a-zA-Z]*|--recursive|--dereference-recursive", w) for w in rest)
        return _search_ok(rest, base, {"-e", "-f", "-m", "-A", "-B", "-C", "--include", "--exclude"}, recursive)
    if name in ("rg", "ag"):
        if any(w.startswith(("--pre", "--search-zip", "-z")) for w in rest):
            return False
        return _search_ok(rest, base, {"-e", "-g", "--glob", "-t", "--type", "-m", "-A", "-B", "-C", "-T"}, True)
    if name in ("fd", "fdfind"):
        return not any(w.startswith(("-x", "-X", "--exec")) for w in rest)
    if name == "find":
        return not ({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"} & flags)
    if name == "tree":
        return not any(w.startswith(("-o", "--output")) for w in rest)
    if name == "sort":
        return not any(w.startswith(("-o", "--output", "--compress-program")) for w in rest)
    if name == "uniq":
        return len(_positionals(rest)) <= 1
    if name == "xxd":
        return len(_positionals(rest, {"-c", "-g", "-l", "-s", "-o"})) <= 1
    if name == "jq":
        return True  # environment reads are refused by secret_reason
    if name == "yq":
        return not any(w.startswith(("-i", "--inplace")) for w in rest)
    if name == "date":
        return not any(w.startswith(("-s", "--set")) for w in rest) and all(
            w.startswith(("-", "+")) for w in rest)
    if name == "hostname":
        return all(w in ("-f", "-i", "-I", "-s", "-d", "-A", "--fqdn", "--short") for w in rest)
    if name == "journalctl":
        return not any(w.startswith(("--vacuum", "--rotate", "--flush", "--sync", "--relinquish", "--smart-relinquish",
                                     "--setup-keys", "--update-catalog")) for w in rest)
    if name == "tldr":
        return not ({"-u", "--update", "--clear-cache"} & flags)
    if name == "sed":
        return _sed_ok(rest)
    if name in ("awk", "gawk", "mawk"):
        return _awk_ok(rest)
    if name == "git":
        return _git_read(rest)
    if name == "gh":
        if rest[:1] == ["api"]:
            return not re.search(r"(-X|--method)\s*(?!GET)|-f\b|-F\b|--field|--raw-field|--input", " ".join(rest)) \
                and all(_web_url_ok("https://api.github.com/" + p) for p in _positionals(rest[1:]))
        verbs = _positionals(rest)
        return len(verbs) >= 2 and verbs[1] in _GH_READ_VERBS or verbs[:2] == ["auth", "status"]
    if name in ("systemctl", "loginctl"):
        sub = next((w for w in rest if not w.startswith("-")), "")
        if name == "loginctl":
            return sub.startswith(("list", "show")) or sub in ("session-status", "user-status", "seat-status")
        return sub in _SYSTEMCTL_READ
    if name == "hyprctl":
        if any(w in ("--batch",) for w in rest):
            return False
        sub = next((w for w in _positionals(rest, {"-i", "--instance"})), "")
        return sub in _HYPRCTL_READ
    if name == "wpctl":
        sub = rest[:1]
        return sub in (["status"], ["inspect"], ["get-volume"]) or (sub == ["settings"] and len(rest) <= 2)
    if name == "pactl":
        sub = next((w for w in rest if not w.startswith("-")), "")
        return sub in ("list", "info", "stat", "subscribe") or sub.startswith("get-")
    if name == "busctl":
        sub = next((w for w in rest if not w.startswith("-")), "")
        return sub in ("list", "status", "tree", "introspect", "get-property", "monitor")
    if name in ("pacman", "yay", "paru"):
        return bool(rest) and bool(re.fullmatch(r"-(Q[a-zA-Z]*|S[si]+|F[lsx]*|T)", rest[0]))
    if name == "curl":
        args = iter(rest)
        urls = []
        for w in args:
            if w in _CURL_SAFE_OPT:
                next(args, None)
            elif w.startswith("-"):
                if not _CURL_SAFE_FLAG.match(w):
                    return False
            else:
                urls.append(w)
        return bool(urls) and all(_web_url_ok(u) for u in urls)
    if name == "wget":
        urls = _positionals(rest)
        return bool({"-qO-", "-O-"} & flags or rest[:2] == ["-O", "-"]) and bool(urls) and \
            all(u == "-" or _web_url_ok(u) for u in urls)
    return False


# --- levels and plain words for what a command does ---

_DESTRUCTIVE_CMDS = {
    "rm", "rmdir", "shred", "dd", "wipefs", "fdisk", "sfdisk", "parted", "gdisk", "mkswap", "truncate", "unlink",
    "sudo", "pkexec", "doas", "su", "run0", "kill", "pkill", "killall", "xkill",
    "shutdown", "reboot", "poweroff", "halt", "suspend", "hibernate",
    "pacman", "yay", "paru", "apt", "apt-get", "dnf", "zypper", "flatpak", "snap", "brew", "makepkg",
    "mail", "mailx", "sendmail", "mutt", "neomutt", "msmtp", "himalaya", "signal-cli", "telegram-send",
    "ssh", "scp", "sftp", "nc", "ncat", "netcat", "socat", "crontab", "at", "batch", "chsh", "passwd",
    "useradd", "userdel", "usermod", "mount", "umount", "omarchy-system", "omarchy-update", "chezmoi",
    "systemd-run",
}
_ACTIONS = [  # (programs, plain words), first match wins
    ({"rm", "rmdir", "shred", "unlink", "truncate"}, "delete files"),
    ({"dd", "wipefs", "fdisk", "sfdisk", "parted", "gdisk", "mkswap"}, "write straight to a disk"),
    ({"sudo", "pkexec", "doas", "su", "run0"}, "run something as administrator"),
    ({"kill", "pkill", "killall", "xkill"}, "stop running programs"),
    ({"shutdown", "reboot", "poweroff", "halt", "suspend", "hibernate"}, "power off or restart the computer"),
    ({"pacman", "yay", "paru", "apt", "apt-get", "dnf", "zypper", "flatpak", "snap", "brew", "makepkg",
      "pip", "pip3", "pipx", "npm", "pnpm", "yarn", "bun", "cargo", "gem", "go", "mise", "composer", "uv", "npx",
      "uvx"}, "install or remove software"),
    ({"mail", "mailx", "sendmail", "mutt", "neomutt", "msmtp", "himalaya", "signal-cli", "telegram-send"},
     "send a message"),
    ({"ssh", "scp", "sftp", "nc", "ncat", "netcat", "socat"}, "connect to another computer"),
    ({"crontab", "at", "batch", "systemd-run"}, "schedule a command to run later"),
    ({"systemctl", "loginctl"}, "change system services"),
    ({"mount", "umount"}, "mount or unmount a disk"),
    ({"chmod", "chown", "chgrp"}, "change file permissions"),
    ({"cp", "mv", "ln", "install", "rsync", "tee", "touch", "mkdir"}, "copy, move or write files"),
    ({"curl", "wget", "http", "xh"}, "send or fetch data over the network"),
    ({"docker", "podman"}, "run a container"),
    ({"hyprctl"}, "change the desktop"),
    ({"xargs"}, "run a command on a list of files"),
    ({"find"}, "run a command on the files it finds"),
    ({"chezmoi"}, "change your dotfiles"),
]
_PKG_SUBCMDS = {"install", "i", "add", "remove", "rm", "uninstall", "un", "update", "upgrade", "up",
                "publish", "sync", "link", "unlink", "ci", "global"}
_PKG_TOOLS = {"pip", "pip3", "uv", "npm", "pnpm", "yarn", "bun", "cargo", "gem", "go", "mise", "pipx", "npx",
              "composer"}
_SYSTEMCTL_BAD = {"stop", "disable", "mask", "kill", "poweroff", "reboot", "suspend", "hibernate", "halt",
                  "isolate", "kexec", "emergency", "rescue", "default", "hybrid-sleep", "suspend-then-hibernate",
                  "daemon-reexec", "reset-failed"}
_GH_BAD = re.compile(r"\b(create|comment|merge|close|delete|edit|review|reopen|ready|lock|transfer|archive|"
                     r"rename|upload|set|remove|add|fork|sync)\b")
_CODE_BAD = re.compile(r"rmtree|remove|unlink|rmdir|rename|replace\(|system|subprocess|spawn|exec|eval|"
                       r"__import__|getattr|write|open\(|socket|request|urllib|http|child_process|fs\.|shutil|"
                       r"chmod|kill")
_RANK: dict[Safety, int] = {"read": 0, "reversible": 1, "destructive": 2}


@dataclass
class Step:
    """One simple command and what it would do."""

    level: Safety
    action: str  # plain words, like "delete files"
    text: str  # the command as written, for the spoken question


def _writable(path: str, base: str | None) -> bool:
    """Places an agent may write without asking first: home and /tmp."""
    p = _expand(path, base)
    return _rel_home(p) is not None or p.as_posix().startswith("/tmp/")


def _overwrites(target: str, base: str | None) -> bool:
    return not _writable(target, base) or _expand(target, base).exists()


def _level(name: str, rest: list[str], base: str | None, depth: int) -> tuple[Safety, str | None]:
    """How bad a non-read program is, and plain words for it when they are more exact
    than the program's general action."""
    flags = set(rest)
    sub = next((w for w in rest if not w.startswith("-")), "")
    if name in _DESTRUCTIVE_CMDS or name.startswith("mkfs"):
        return "destructive", "format a disk" if name.startswith("mkfs") else None
    if name.startswith("omarchy"):
        from omi.tools.omarchy import command_safety

        return command_safety(name), f"run {name.replace('-', ' ')}"
    if name.startswith("python") or name in ("node", "nodejs", "perl", "ruby", "deno", "bun", "php", "lua"):
        return ("destructive" if _CODE_BAD.search(" ".join(rest)) else "reversible"), "run a script"
    if name in ("awk", "gawk", "mawk"):
        return ("destructive" if not _awk_ok(rest) else "reversible"), "run an awk program"
    if name == "sed":
        return ("reversible" if any(w.startswith(("-i", "--in-place")) for w in rest) else "destructive"), "edit files"
    if name == "mv":
        dest = _positionals(rest)[-1:] or [""]
        return ("destructive" if "-n" not in flags and _overwrites(dest[0], base) else "reversible"), "move files"
    if name in ("cp", "install", "ln", "rsync"):
        targets = _positionals(rest)
        if name == "rsync" and ("--delete" in " ".join(rest) or any(re.match(r"^[\w.@-]+:", t) for t in targets)):
            return "destructive", "sync files to another place"
        return ("destructive" if targets and _overwrites(targets[-1], base) else "reversible"), None
    if name in ("tee", "touch", "mkdir", "chmod", "chown", "chgrp"):
        targets = _positionals(rest)
        if name in ("chmod", "chown", "chgrp") and ("-R" in flags or any(_expand(t, base) == HOME for t in targets)):
            return "destructive", None
        bad = any(not _writable(t, base) for t in targets) or name == "tee" and any(
            _overwrites(t, base) for t in targets if "-a" not in flags)
        return ("destructive" if bad else "reversible"), None
    if name == "git":
        words = rest
        while words and words[0].startswith("-"):
            words = words[2:] if words[0] in _GIT_GLOBAL_ARG or words[0] == "-c" else words[1:]
        sub, gflags = (words[0], set(words[1:])) if words else ("", set())
        if sub in ("push", "clean", "filter-branch", "filter-repo", "gc", "prune", "rm", "restore", "rebase",
                   "update-ref", "config"):
            return "destructive", {"push": "push code to a remote", "clean": "delete untracked files",
                                   "rm": "delete files from git", "config": "change git settings"}.get(sub, "change git history")
        if sub == "reset" and "--hard" in gflags or sub == "checkout" and ({"-f", "--force", "--", "."} & gflags):
            return "destructive", "throw away git changes"
        if sub == "branch" and {"-D", "-d", "--delete"} & gflags or sub == "stash" and {"drop", "clear"} & gflags:
            return "destructive", "delete git work"
        return "reversible", f"run git {sub}".strip()
    if name == "gh":
        if sub == "api":
            return ("destructive" if re.search(r"(-X|--method)\s*(POST|PUT|PATCH|DELETE)|-f\b|-F\b|--field|--input",
                                               " ".join(rest)) else "reversible"), "call the GitHub API"
        return ("destructive" if _GH_BAD.search(" ".join(rest[:3])) else "reversible"), "change things on GitHub"
    if name in ("curl", "wget", "http", "xh"):
        if re.search(r"(^|\s)(-d|--data\S*|-F|--form\S*|-T|--upload-file|--post-\w+|--json)\b|(-X|--request)\s*(POST|PUT|PATCH|DELETE)",
                     " ".join(rest)) or any(urlparse(u).query for u in _positionals(rest)):
            return "destructive", "send data over the network"
        return "reversible", None
    if name in _PKG_TOOLS or name in ("uvx",):
        if name == "uv" and sub == "pip":
            sub = next((w for w in rest[1:] if not w.startswith("-")), "")
        if name in ("npx", "uvx") or sub in _PKG_SUBCMDS or (name == "go" and sub == "get") or sub in ("dlx", "x"):
            return "destructive", None
        return "reversible", f"run {name} {sub}".strip()
    if name in ("systemctl", "loginctl"):
        if sub in _SYSTEMCTL_BAD or name == "loginctl" and sub.startswith(("terminate", "kill", "poweroff", "reboot", "suspend")):
            return "destructive", None
        return "reversible", None
    if name == "hyprctl":
        return ("destructive" if {"kill", "dispatch", "keyword", "eval", "reload", "--batch"} & flags else "reversible"), None
    if name == "busctl":
        return ("destructive" if sub in ("call", "set-property", "emit") else "reversible"), "call a system service"
    if name in ("docker", "podman"):
        return ("destructive" if sub in ("run", "exec", "rm", "rmi", "system", "volume", "compose", "kill", "stop")
                else "reversible"), None
    if name in ("make", "just", "task"):
        return "reversible", f"run {name} {sub}".strip()
    if name in _SEARCHERS:
        return "reversible", "search inside a folder that holds keys and tokens"
    return "reversible", None


def _action(name: str) -> str:
    for names, words in _ACTIONS:
        if name in names:
            return words
    return f"run {name}" if name else "run a command"


def _seg_steps(seg: _Seg, base: str | None, depth: int) -> list[Step]:
    text = " ".join(seg.words) + "".join(f" {op} {t}".rstrip() for op, t in seg.redirects)
    words, env = _unwrap(seg.words)
    if not words:
        if env and not seg.redirects:
            return [Step("read", "set a variable", text)]  # FOO=1 on its own changes nothing outside
        words = []
    name = Path(words[0]).name if words else ""
    rest = words[1:]
    steps: list[Step] = []
    inner = _shell_inner(name, rest) if words else None
    if inner is not None:
        steps = _steps(inner, base, depth + 1)
        level: Safety = max((s.level for s in steps), key=_RANK.__getitem__, default="read")
        action = None
    elif name == "xargs":
        cmd = _positionals(rest, {"-a", "-d", "-E", "-I", "-L", "-n", "-P", "-s", "--arg-file", "--delimiter",
                                  "--max-args", "--max-procs", "--replace"})
        steps = _steps(shlex.join(cmd), base, depth + 1) if cmd else []
        level = "destructive" if any(s.level != "read" for s in steps) else "reversible"
        action = "run a command on a list of files"
    elif name == "find" and {"-exec", "-execdir", "-ok", "-okdir"} & set(rest):
        i = next(i for i, w in enumerate(rest) if w in ("-exec", "-execdir", "-ok", "-okdir"))
        cmd = [w for w in rest[i + 1:] if w not in (";", "+", "{}", "\\;")]
        steps = _steps(shlex.join(cmd), base, depth + 1) if cmd else []
        level = "destructive" if any(s.level != "read" for s in steps) else "reversible"
        action = "run a command on the files it finds"
    elif name == "find" and "-delete" in rest:
        level, action = "destructive", "delete files"
    elif words and _read_ok(name, rest, base) and not env and not any(_BRACES.search(w) for w in words):
        level, action = "read", "look at things"
    else:
        level, action = _level(name, rest, base, depth)
    for op, target in seg.redirects:
        if ">" not in op or target in ("/dev/null", "/dev/stdout", "/dev/stderr") or op in (">&", "<&") and (
                target.isdigit() or target == "-"):
            if "<<" in op and level == "read":
                level, action = "reversible", action  # a here-document: keep it simple, ask
            continue
        bad = op != ">>" and _overwrites(target, base) or not _writable(target, base)
        new: Safety = "destructive" if bad else "reversible"
        if _RANK[new] > _RANK[level]:
            level, action = new, "overwrite a file" if bad else "write a file"
        elif level == "read":
            level, action = "reversible", "write a file"
    if not steps or level != "read" and action:
        steps = [Step(level, action or _action(name), text), *[s for s in steps if s.level != "read"]]
    return steps


def _steps(cmd: str, base: str | None = None, depth: int = 0) -> list[Step]:
    """Every simple command in cmd, including the ones inside $(), `` and <()."""
    if depth > 4:
        return [Step("destructive", "run deeply nested commands", cmd)]
    subs = _substitutions(cmd)
    steps: list[Step] = []
    for inner in subs:
        for s in _steps(inner, base, depth + 1):
            # The output of a substitution becomes part of another command: never "read".
            steps.append(Step("reversible", s.action, s.text) if s.level == "read" else s)
    tokens = _tokens(cmd)
    if tokens is None:
        return [*steps, Step("destructive", "run a command omi cannot read", cmd)]
    unknown = _unknown_vars(cmd)
    for seg in _split(tokens):
        for s in _seg_steps(seg, base, depth):
            # A variable omi cannot see could name any file: never "read".
            steps.append(Step("reversible", "use a shell variable", s.text) if unknown and s.level == "read" else s)
        words, _ = _unwrap(seg.words)
        if words and Path(words[0]).name in ("cd", "pushd") and words[1:]:
            base = str(_expand(words[-1], base))
    return steps


def describe_command(cmd: str) -> tuple[Safety, list[Step]]:
    """The command's level and its steps that are not read-only, worst first."""
    if secret_reason(cmd):
        return "destructive", [Step("destructive", "read a secret", cmd)]
    steps = _steps(cmd)
    # Piping a download into a shell runs code nobody has read.
    names = [s.text.split(" ", 1)[0] for s in steps]
    if any(Path(n).name in ("curl", "wget") for n in names) and any(
            Path(n).name in _SHELLS or Path(n).name.startswith("python") for n in names[1:]):
        steps.append(Step("destructive", "run code downloaded from the internet", cmd))
    risky = sorted((s for s in steps if s.level != "read"), key=lambda s: -_RANK[s.level])
    level: Safety = risky[0].level if risky else "read"
    return level, risky


def classify_command(cmd: str) -> Safety:
    """read / reversible / destructive for a shell command line. Secret-exposing
    commands count as destructive here; secret_reason is what hard-denies them."""
    return describe_command(cmd)[0]


# --- writes Claude must ask about ---

_RC_NAMES = re.compile(r"^(\.bashrc|\.zshrc|\.zshenv|\.zprofile|\.zlogin|\.zlogout|\.profile|\.bash_profile|"
                       r"\.bash_login|\.bash_logout|\.inputrc|\.xprofile|\.xinitrc|\.envrc|\.gitconfig|"
                       r"settings(\.local)?\.json)$")
_GUARDED_DIRS = (  # relative to home, with what a write there means
    (".config/systemd", "a login service"), (".local/share/systemd", "a login service"),
    (".config/autostart", "an autostart entry"), (".config/hypr", "the Hyprland config"),
    (".config/uwsm", "the session environment"), (".config/environment.d", "the session environment"),
    (".local/bin", "a program in your command path"), (".claude", "Claude's own settings"),
    (".config/omi", "omi's settings"), (".config/omi", "Omi's settings"),
    (".config/omarchy/hooks", "an Omarchy hook"),
    (".config/fish", "your shell startup files"), (".config/nushell", "your shell startup files"),
    (".config/git", "your git settings"), (".local/share/applications", "an app launcher entry"),
)


def guarded_write(path: str) -> str | None:
    """Plain words for a file whose change can switch off omi's checks or run code at
    login or on the next command, or None for an ordinary file."""
    p = _expand(path)
    if p == OMI_ROOT or OMI_ROOT in p.parents:
        return "omi's own code"
    shared_index = _shared_memory_index()
    if shared_index is not None and p == shared_index:
        return "the shared agent memory index"
    if _RC_NAMES.match(p.name):
        return "your shell or settings file " + p.name
    parts = p.parts
    if ".git" in parts and (parts[parts.index(".git") + 1:2 + parts.index(".git")] in (("hooks",), ("config",))):
        return "a git hook or git setting"
    rel = _rel_home(p)
    for d, words in _GUARDED_DIRS:
        if rel is not None and (rel == d or rel.startswith(d + "/")):
            return words
    return None


# --- Claude Code tools ---

_ALWAYS_ALLOW = {"WebSearch", "TodoWrite", "ToolSearch", "Task", "Agent"}


@functools.cache
def _omi_tools() -> dict[str, Any]:
    from omi.config import Config
    from omi.tools import build_registry
    from omi.tools.mcp_server import daemon_tools

    return {tool.name: tool for tool in [*build_registry(Config()).tools.values(), *daemon_tools()]}


_CHROME_READ = {
    "tabs_context_mcp", "read_page", "find", "get_page_text",
    "read_console_messages", "read_network_requests", "shortcuts_list",
}
_CHROME_REVERSIBLE = {
    "tabs_create_mcp", "navigate", "form_input", "computer", "resize_window", "switch_browser",
}


def classify_claude_tool(tool_name: str, tool_input: dict[str, Any]) -> tuple[Decision, str]:
    """allow / confirm / deny for one Claude Code tool call, with a short reason."""
    tool_input = tool_input or {}
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        if reason := secret_reason(cmd):
            return "deny", f"Blocked because {reason}."
        level = classify_command(cmd)
        if level != "read":
            return "confirm", "This command can change or delete things."
        return "allow", "read-only command"
    if tool_name in ("Read", "Grep", "Glob", "LS", "NotebookRead"):
        paths = [str(tool_input.get(k, "")) for k in ("file_path", "path", "notebook_path")]
        paths.append(str(tool_input.get("pattern", "")) if tool_name == "Glob" else "")
        paths.append(str(tool_input.get("glob", "")))
        for p in paths:
            if p and any(is_secret_path(c) for c in _candidates(p, None)):
                return "deny", f"Blocked because {Path(p).name} may hold secrets."
        if tool_name == "Grep" and not (tool_input.get("glob") or tool_input.get("type")) and covers_secrets(
                str(tool_input.get("path") or ".")):
            return "confirm", "This searches inside every file of a folder that holds keys and tokens."
        return "allow", "reading"
    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
        if not path:
            return "confirm", "Edit with no path."
        if is_secret_path(path) or in_secret_dir(path):
            return "deny", f"Blocked because {Path(path).name} holds secrets."
        if not inside_home(path):
            return "confirm", f"This writes {path}, outside the home folder."
        if what := guarded_write(path):
            return "confirm", f"This changes {what}."
        return "allow", "editing a file in the home folder"
    if tool_name == "WebFetch":
        if urlparse(str(tool_input.get("url", ""))).query:
            return "confirm", "The web address carries data in it."
        return "allow", "reading a web page"
    if tool_name in _ALWAYS_ALLOW:
        return "allow", "safe built-in tool"
    if tool_name.startswith("mcp__claude-in-chrome__"):
        name = tool_name.removeprefix("mcp__claude-in-chrome__")
        if name in _CHROME_READ:
            return "allow", "read-only browser tool"
        if name == "navigate":
            url = str(tool_input.get("url", "")).strip()
            if url and url.lower() not in ("back", "forward"):
                if "://" not in url and re.match(r"^[a-z][a-z0-9+.-]*:(?!\d)", url, re.I):
                    return "deny", "Blocked because only http and https browser navigation is allowed."
                parsed = urlparse(url if "://" in url else "https://" + url)
                if parsed.scheme not in ("http", "https"):
                    return "deny", "Blocked because only http and https browser navigation is allowed."
            return "allow", "browser navigation"
        if name in _CHROME_REVERSIBLE:
            return "allow", "browser interaction"
        return "confirm", f"This browser action ({name}) can run scripts or upload data."
    if tool_name.startswith("mcp__omi__"):
        name = tool_name.removeprefix("mcp__omi__")
        if name == "approve":
            return "allow", "read omi tool"
        tool = _omi_tools().get(name)
        if tool is None:
            return "confirm", "Unknown omi tool."
        try:
            safety = tool.safety_of(tool_input)
        except (ValueError, TypeError) as exc:
            return "deny", f"Blocked: {exc}."
        if name == "open_url" and urlparse(str(tool_input.get("url", ""))).query:
            return "confirm", "The web address carries data in it."
        if safety == "destructive":
            return "confirm", "This omi action is hard to undo."
        return "allow", f"{safety} omi tool"
    return "confirm", f"{tool_name} is not a tool omi knows."
