"""Repair agent (Nemotron Super): propose exact search/replace edits for one hypothesis."""
from __future__ import annotations

from pathlib import Path

from ..core.patching import Edit, PatchError, parse_edits
from ..inference import ModelRouter
from ..models import Observation, RunRequest
from ..util import safe_join
from .common import call_json, fence, system_prompt
from .diagnoser import Hypothesis

PROMPT = """
You are the repair agent of ReproFix. Fix the ONE hypothesis you are given with the smallest correct change.

Rules:
- Return edits as exact search/replace pairs. `search` must be copied verbatim from the file shown (include enough
  surrounding lines to make it unique) and must appear exactly once. Use {{"path": ..., "create": "<full content>"}} only for a new file.
- Change only what the hypothesis requires. Prefer fixing the root cause over working around it.
- Never edit tests or the README. Never hardcode the expected metric value or print a made-up number.
- Do not silence errors with try/except or by deleting the failing code. Do not add network access.
- Dependency problems are fixed in requirements files (pin/unpin/add), not in code.

Return JSON:
{"edits": [{"path": "...", "search": "...", "replace": "..."}], "rationale": "one or two sentences", "expected_effect": "what should change when rerun"}
"""

MAX_FILE_CHARS = 14_000


def _file_blob(workdir: Path, rel: str) -> str:
    try:
        p = safe_join(workdir, rel)
        if p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_CHARS]
    except (ValueError, OSError):
        pass
    return ""


def repair(router: ModelRouter, *, workdir: Path, request: RunRequest, hypothesis: Hypothesis, observation: Observation,
           requirement_files: list[str], escalate: int, protected: list[str]) -> tuple[list[Edit], dict]:
    rels: list[str] = []
    for f in hypothesis.files + [e.get("file", "") for e in hypothesis.evidence if e.get("kind") == "code"]:
        if f and f not in rels:
            rels.append(f)
    if observation.location:
        loc = observation.location.rsplit(":", 1)[0]
        if loc not in rels:
            rels.append(loc)
    if observation.kind == "install_failure":
        rels += [r for r in requirement_files if r not in rels]
    blobs = [(r, _file_blob(workdir, r)) for r in rels[:5]]
    blobs = [(r, b) for r, b in blobs if b]
    base = (
        f"Goal: {request.goal}\nHypothesis ({hypothesis.category}): {hypothesis.statement}\n"
        f"How to test it: {hypothesis.test}\nFailure: {observation.summary}\n\n"
        + "\n".join(fence(f"file {r}", b, MAX_FILE_CHARS) for r, b in blobs)
        + "\n" + fence("log tail", observation.log_tail, 4000)
    )
    messages = [{"role": "system", "content": system_prompt("repair", PROMPT)}, {"role": "user", "content": base}]
    last_err = ""
    for attempt in range(3):
        obj, _ = call_json(router, "repair", messages, escalate=escalate)
        try:
            edits = parse_edits(obj.get("edits"))
            for e in edits:  # fail early on an obviously bad target so the model can correct itself
                if e.search is not None and e.path in dict(blobs) and e.search not in dict(blobs)[e.path]:
                    raise PatchError(f"{e.path}: 'search' text not found in the file shown above")
            return edits, {"rationale": str(obj.get("rationale", ""))[:500], "expected_effect": str(obj.get("expected_effect", ""))[:300]}
        except PatchError as exc:
            last_err = str(exc)
            messages += [{"role": "assistant", "content": str(obj)[:3000]},
                         {"role": "user", "content": f"Your edits could not be used: {last_err}\nReply again with corrected edits."}]
    raise PatchError(last_err)
