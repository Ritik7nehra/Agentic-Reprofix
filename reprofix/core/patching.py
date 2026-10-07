"""Applying LLM-proposed edits safely, reverting them, and producing a real unified diff.

Edits are exact search/replace pairs rather than model-written diffs: line numbers and context
hunks written by an LLM are unreliable, while an exact, unique match either applies or fails loudly.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..util import safe_join

FORBIDDEN_TOP = {".deps", ".git", ".home", "__pycache__", ".pytest_cache"}
MAX_FILES_PER_PATCH = 5
MAX_FILE_BYTES = 200_000


class PatchError(ValueError):
    pass


@dataclass
class Edit:
    path: str
    search: str | None = None
    replace: str | None = None
    create: str | None = None  # full content for a new file

    def describe(self) -> str:
        return f"create {self.path}" if self.create is not None else f"edit {self.path}"

    def to_dict(self) -> dict:
        return {"path": self.path, "search": self.search, "replace": self.replace, "create": self.create}


def parse_edits(raw) -> list[Edit]:
    if not isinstance(raw, list) or not raw:
        raise PatchError("'edits' must be a non-empty list")
    edits: list[Edit] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise PatchError(f"edit #{i}: needs a string 'path'")
        if isinstance(item.get("create"), str):
            edits.append(Edit(path=item["path"], create=item["create"]))
        elif isinstance(item.get("search"), str) and isinstance(item.get("replace"), str):
            if item["search"] == "":
                raise PatchError(f"edit #{i}: 'search' must not be empty")
            if item["search"] == item["replace"]:
                raise PatchError(f"edit #{i}: 'search' and 'replace' are identical")
            edits.append(Edit(path=item["path"], search=item["search"], replace=item["replace"]))
        else:
            raise PatchError(f"edit #{i}: provide either 'create' or both 'search' and 'replace'")
    return edits


def _glob_to_regex(pat: str) -> re.Pattern[str]:
    out = ""
    i = 0
    while i < len(pat):
        c = pat[i]
        if pat[i : i + 3] == "**/":
            out += "(?:.*/)?"
            i += 3
            continue
        if pat[i : i + 2] == "**":
            out += ".*"
            i += 2
            continue
        out += "[^/]*" if c == "*" else "[^/]" if c == "?" else re.escape(c)
        i += 1
    return re.compile(out + r"\Z")


def is_protected(rel_path: str, patterns: list[str]) -> bool:
    rel = re.sub(r"^(?:\./)+", "", rel_path.replace("\\", "/"))  # strip leading "./" only; ".env" must stay ".env"
    return any(_glob_to_regex(p).match(rel) for p in patterns)


@dataclass
class AppliedPatch:
    backups: dict[str, bytes | None] = field(default_factory=dict)  # path -> previous bytes (None = didn't exist)
    changed: list[str] = field(default_factory=list)


def _closest_hint(content: str, search: str) -> str:
    first = next((ln.strip() for ln in search.splitlines() if ln.strip()), "")
    if not first:
        return ""
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    close = difflib.get_close_matches(first, lines, n=1, cutoff=0.6)
    return f" Closest line in the file: {close[0]!r}." if close else ""


def apply_edits(workdir: Path, edits: list[Edit], protected: list[str]) -> AppliedPatch:
    """Apply atomically: on any failure nothing is left changed."""
    if len({e.path for e in edits}) > MAX_FILES_PER_PATCH:
        raise PatchError(f"too many files in one patch (max {MAX_FILES_PER_PATCH})")
    applied = AppliedPatch()
    try:
        for e in edits:
            top = e.path.replace("\\", "/").split("/")[0]
            if top in FORBIDDEN_TOP:
                raise PatchError(f"{e.path}: editing {top}/ is not allowed")
            if is_protected(e.path, protected):
                raise PatchError(f"{e.path} is protected (tests, README and similar files may not be edited)")
            target = safe_join(workdir, e.path)
            existed = target.exists()
            if e.path not in applied.backups:
                applied.backups[e.path] = target.read_bytes() if existed else None
            if e.create is not None:
                if existed:
                    raise PatchError(f"{e.path} already exists; use search/replace to modify it")
                if len(e.create.encode()) > MAX_FILE_BYTES:
                    raise PatchError(f"{e.path}: new file is too large")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(e.create, encoding="utf-8")
            else:
                if not existed or not target.is_file():
                    raise PatchError(f"{e.path} does not exist")
                try:
                    content = target.read_text(encoding="utf-8")
                except UnicodeDecodeError as exc:
                    raise PatchError(f"{e.path} is not a UTF-8 text file") from exc
                count = content.count(e.search or "")
                if count == 0:
                    raise PatchError(f"{e.path}: 'search' text not found." + _closest_hint(content, e.search or ""))
                if count > 1:
                    raise PatchError(f"{e.path}: 'search' text matches {count} places; include more surrounding lines")
                target.write_text(content.replace(e.search or "", e.replace or "", 1), encoding="utf-8")
            if e.path not in applied.changed:
                applied.changed.append(e.path)
    except Exception:
        revert(workdir, applied)
        raise
    return applied


def revert(workdir: Path, applied: AppliedPatch) -> None:
    for rel, old in applied.backups.items():
        target = workdir / rel
        if old is None:
            if target.exists():
                target.unlink()
        else:
            target.write_bytes(old)


def _hardcode_hits(text: str, expected: float, metric_name: str) -> list[str]:
    """Lines of `text` that look like the expected metric value written into code that computes/prints a metric."""
    forms = {f"{expected}", f"{expected:.2f}", f"{expected:.3f}", f"{expected:.4f}", f"{expected * 100:g}", f"{expected * 100:.1f}"}
    context = re.compile(r"print|accuracy|acc|metric|" + re.escape(metric_name), re.I)
    hits = []
    for line in text.splitlines():
        if not context.search(line):
            continue
        for form in forms:
            if re.search(rf"(?<![\d.]){re.escape(form)}(?![\d])", line):
                hits.append(line.strip()[:120])
                break
    return hits


def literal_hardcoding_findings(edits: list[Edit], expected: float | None, metric_name: str) -> list[str]:
    """Heuristic guard against 'fixing' a metric by writing the expected number into the code."""
    if expected is None:
        return []
    findings = []
    for e in edits:
        added = e.create if e.create is not None else (e.replace or "")
        findings += [f"{e.path}: line contains the expected value: {hit}" for hit in _hardcode_hits(added, expected, metric_name)]
    return findings


def diff_hardcoding_findings(diff: str, expected: float | None, metric_name: str) -> list[str]:
    """The same rule applied to the lines a unified diff adds (used by the report's computed checks)."""
    if expected is None:
        return []
    added = "\n".join(ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    return _hardcode_hits(added, expected, metric_name)


# ----------------------------------------------------------------------------- diff
def _is_text(b: bytes) -> bool:
    if b"\x00" in b:
        return False
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _lines(b: bytes | None) -> list[str]:
    return [] if b is None else b.decode("utf-8").splitlines(keepends=True)


def _emit(diff_iter) -> list[str]:
    out = []
    for ln in diff_iter:
        if ln.endswith("\n"):
            out.append(ln)
        else:
            out.append(ln + "\n\\ No newline at end of file\n")
    return out


def make_diff(orig_dir: Path, work_dir: Path, paths: list[str]) -> tuple[str, list[dict]]:
    """Unified diff (git-apply compatible, `a/` `b/` prefixes) for `paths`, original vs workspace."""
    chunks: list[str] = []
    files: list[dict] = []
    for rel in sorted(set(paths)):
        a = orig_dir / rel
        b = work_dir / rel
        old = a.read_bytes() if a.is_file() else None
        new = b.read_bytes() if b.is_file() else None
        if old == new:
            continue
        if (old is not None and not _is_text(old)) or (new is not None and not _is_text(new)):
            continue
        head = [f"diff --git a/{rel} b/{rel}\n"]
        if old is None:
            head.append("new file mode 100644\n")
        elif new is None:
            head.append("deleted file mode 100644\n")
        body = _emit(difflib.unified_diff(
            _lines(old), _lines(new),
            fromfile="/dev/null" if old is None else f"a/{rel}",
            tofile="/dev/null" if new is None else f"b/{rel}",
        ))
        added = sum(1 for ln in body if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in body if ln.startswith("-") and not ln.startswith("---"))
        chunks.append("".join(head + body))
        files.append({"path": rel, "status": "added" if old is None else "deleted" if new is None else "modified",
                      "added": added, "removed": removed})
    return "".join(chunks), files
