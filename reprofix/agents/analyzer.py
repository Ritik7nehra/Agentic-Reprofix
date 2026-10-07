"""Repository analyzer (Nemotron Nano): summarise what the project is. Deterministic facts win over the model."""
from __future__ import annotations

from pathlib import Path

from ..core.repo import RepoScan
from ..inference import ModelRouter
from ..models import RunRequest
from ..util import UnsafeCommand, parse_command
from .common import AgentError, call_json, fence, system_prompt

PROMPT = """
You are the repository analyzer of ReproFix, a tool that reproduces and repairs broken ML repositories.
Summarise the project from the scan below. Do not guess: use null when the repository does not say.

Return JSON:
{"framework": "pytorch|tensorflow|jax|sklearn|numpy|other|unknown",
 "task": "short phrase, e.g. image classification",
 "model": "short phrase or null",
 "dataset": "short phrase or null",
 "run_command": "the command that reproduces the documented experiment, e.g. python train.py, or null",
 "expected_metric": {"name": "metric name", "value": 0.86, "source": "exact README text"} or null,
 "notes": "one or two sentences"}
"""


def analyze(router: ModelRouter, scan: RepoScan, workdir: Path, request: RunRequest) -> dict:
    """Returns a validated analysis dict. Never raises: falls back to scan-only facts."""
    files = "\n".join(scan.files[:80])
    reqs = "\n\n".join(
        f"# {r}\n{(workdir / r).read_text(encoding='utf-8', errors='replace')[:1500]}" for r in scan.requirements[:3]
    )
    user = (
        f"User goal: {request.goal}\n\n"
        f"Frameworks detected from imports/requirements: {scan.frameworks or 'none'}\n"
        f"Python entry points: {scan.entry_points or 'none'}\n"
        f"README numbers that look like metrics: {scan.readme_claims or 'none'}\n"
        f"Commands found in the README: {scan.readme_commands or 'none'}\n\n"
        + fence("file listing", files, 4000) + "\n"
        + fence("requirements", reqs or "(none)", 4000) + "\n"
        + fence("README", scan.readme_text[:6000] or "(no README)", 6000)
    )
    llm: dict = {}
    error = None
    try:
        llm, _ = call_json(router, "summarize", [
            {"role": "system", "content": system_prompt("analyze", PROMPT)},
            {"role": "user", "content": user},
        ], max_tokens=2048)
    except Exception as exc:  # model failure must not stop the run
        error = f"{type(exc).__name__}: {exc}"

    analysis: dict = {
        "framework": scan.frameworks[0] if scan.frameworks else (llm.get("framework") or "unknown"),
        "task": llm.get("task"), "model": llm.get("model"), "dataset": llm.get("dataset"),
        "notes": llm.get("notes"), "frameworks_detected": scan.frameworks,
        "entry_points": scan.entry_points, "requirements": scan.requirements, "has_tests": scan.has_tests,
        "run_command": None, "run_command_source": None, "expected_metric": None, "llm_error": error,
    }
    # run command: user > README fenced command > LLM (validated, script must exist) > entry point
    candidates: list[tuple[str, str]] = []
    if request.command:
        parse_command(request.command)  # an explicit command is never silently replaced by a guess: raises UnsafeCommand
        candidates.append((request.command, "user"))
    candidates += [(c, "readme") for c in scan.readme_commands if not c.startswith("pytest")]
    if isinstance(llm.get("run_command"), str):
        candidates.append((llm["run_command"], "model"))
    if scan.entry_points:
        candidates.append((f"python {scan.entry_points[0]}", "entry-point"))
    for cmd, src in candidates:
        try:
            argv = parse_command(cmd)
        except UnsafeCommand:
            continue
        script = next((a for a in argv[1:] if a.endswith(".py")), None)
        if src != "user" and script and not (workdir / script).is_file():
            continue
        analysis["run_command"], analysis["run_command_source"] = cmd, src
        break
    # expected metric: only accept a model-proposed value that really appears in the README
    em = llm.get("expected_metric")
    if isinstance(em, dict) and isinstance(em.get("value"), (int, float)):
        val = float(em["value"])
        if any(abs(c["value"] - val) < 1e-6 for c in scan.readme_claims):
            analysis["expected_metric"] = {"name": str(em.get("name") or "metric"), "value": val,
                                           "source": "README (value verified present)"}
    if analysis["expected_metric"] is None and scan.readme_claims:
        c = scan.readme_claims[0]
        analysis["expected_metric"] = {"name": c["metric_hint"], "value": c["value"],
                                       "source": "README regex (first match; confirm it)"}
    return analysis
