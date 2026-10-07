"""Small helpers shared across modules."""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

_META = re.compile(r"[;|&<>`$\n\r]")

ALLOWED_INTERPRETERS = {"python", "python3", "pytest"}


class UnsafeCommand(ValueError):
    pass


def parse_command(command: str) -> list[str]:
    """Split a user/LLM-proposed command into argv, rejecting shell syntax.

    Commands never go through a shell. They must start with python/python3/pytest.
    The sandbox is still the real security boundary; this just keeps proposals sane.
    """
    command = command.strip()
    if not command:
        raise UnsafeCommand("empty command")
    if _META.search(command):
        raise UnsafeCommand("shell metacharacters are not allowed")
    argv = shlex.split(command)
    if argv[0] not in ALLOWED_INTERPRETERS:
        raise UnsafeCommand(f"command must start with one of {sorted(ALLOWED_INTERPRETERS)}")
    return argv


def truncate_middle(text: str, head: int = 3000, tail: int = 12000) -> tuple[str, bool]:
    """Keep the start and (more importantly) the end of long output; tracebacks live at the end."""
    if len(text) <= head + tail:
        return text, False
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [{omitted} characters omitted] ...\n{text[-tail:]}", True


def tail_lines(text: str, n: int = 60) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def safe_join(root: Path, rel: str) -> Path:
    """Resolve `rel` under `root`; raise if it escapes (.., absolute paths, symlinks)."""
    if not rel or rel.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", rel):
        raise ValueError(f"path must be relative: {rel!r}")
    root_r = root.resolve()
    target = (root_r / rel).resolve()
    if target != root_r and root_r not in target.parents:
        raise ValueError(f"path escapes the workspace: {rel!r}")
    return target


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of model output.

    Handles <think>...</think> blocks, ```json fences and prose around the object.
    Returns None if nothing parses.
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S | re.I) if "</think>" in text else text
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    candidates = list(reversed(fenced))
    # balanced-brace scan for bare objects
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : i + 1])
                start = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
