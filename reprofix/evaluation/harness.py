"""ReproBench harness: validate tasks, run the agent on them, score stages, aggregate measured results.

Scoring is deterministic (no LLM judging):
  understood        the agent's inferred run command equals the task's command (the command is NOT given to it)
  reproduced        the agent's baseline run failed in the way the task documents (crash / install failure /
                    wrong metric, plus the error pattern when one is specified)
  root_cause        confirmed hypotheses cover every ground-truth file (and, for single-fault tasks, the category)
  patch_generated   at least one file was changed and kept
  patch_verified    the agent's own verification passed AND a hidden check passed: protected files untouched,
                    a fresh run with a different data seed lands within the hidden tolerance, and the printed
                    metric equals an independent recomputation from the saved predictions.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..core.orchestrator import Orchestrator
from ..core.patching import apply_edits, is_protected, parse_edits
from ..core.verify import TOLERANCE_EPS, make_verdict, parse_metric
from ..inference.base import ChatBackend
from ..models import MetricSpec, RunRequest
from ..sandbox import Sandbox
from ..sandbox.base import DEPS_DIR
from ..util import parse_command
from .scripted import OracleBackend

STAGES = ["understood", "reproduced", "root_cause", "patch_generated", "patch_verified"]
DEFAULT_PROTECTED = ["tests/**", "test_*.py", "**/test_*.py", "README*", "LICENSE*"]


@dataclass
class Task:
    id: str
    dir: Path
    spec: dict

    @property
    def repo(self) -> Path:
        return self.dir / "repo"

    @property
    def timeout_s(self) -> int:
        """Per-command timeout. Most tasks run in seconds; the PyTorch task needs minutes to install torch."""
        return int(self.spec.get("timeout_s", 300))


def load_tasks(root: Path, only: list[str] | None = None) -> list[Task]:
    tasks = []
    for d in sorted(root.iterdir()):
        if (d / "task.json").is_file() and (not only or d.name in only):
            tasks.append(Task(id=d.name, dir=d, spec=json.loads((d / "task.json").read_text(encoding="utf-8"))))
    return tasks


def _metric(spec: dict) -> MetricSpec:
    m = spec["metric"]
    return MetricSpec(name=m["name"], expected=m["expected"], tolerance=m["tolerance"])


def _standalone_verdict(sandbox: Sandbox, workdir: Path, spec: dict, timeout: int | None = None):
    """Install + run + tests with no agent involved. Returns (verdict, install_result, run_result)."""
    timeout = timeout or int(spec.get("timeout_s", 300))
    shutil.rmtree(workdir / DEPS_DIR, ignore_errors=True)
    metric = _metric(spec)
    reqs = ["requirements.txt"] if (workdir / "requirements.txt").is_file() else []
    install = sandbox.install(workdir, reqs, timeout) if reqs else None
    if install is not None and (install.exit_code != 0 or install.timed_out):
        return make_verdict(install=install, run=None, tests=None, metric=metric, workdir=workdir), install, None
    run = sandbox.run(workdir, parse_command(spec["command"]), timeout)
    tests = sandbox.run(workdir, ["pytest", "-q", "-p", "no:cacheprovider"], timeout) if (workdir / "tests").is_dir() else None
    return make_verdict(install=None, run=run, tests=tests, metric=metric, workdir=workdir), install, run


# ---------------------------------------------------------------------------- validation
def validate_task(task: Task, sandbox: Sandbox, scratch: Path) -> dict:
    """Prove the task is well-formed: broken state fails as documented; reference fix passes."""
    spec = task.spec
    out: dict = {"id": task.id, "category": spec["category"], "ok": False, "problems": []}
    work = scratch / task.id
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(task.repo, work)

    bv, inst, run = _standalone_verdict(sandbox, work, spec)
    out["broken"] = {"stage": bv.stage, "reasons": bv.reasons}
    kind = spec["symptom"]["kind"]
    want = {"install_failure": "install_failure", "crash": "crash", "wrong_result": "wrong_result"}[kind]
    if bv.verified:
        out["problems"].append("broken repo already passes verification")
    elif bv.stage != want:
        out["problems"].append(f"expected symptom '{want}' but got '{bv.stage}' ({'; '.join(bv.reasons)})")
    pat = spec["symptom"].get("pattern")
    if pat and not bv.verified:
        text = ((inst.stderr + inst.stdout) if inst else "") + ((run.stderr + run.stdout) if run else "")
        if not re.search(pat, text, re.I):
            out["problems"].append(f"error pattern {pat!r} not found in output")

    apply_edits(work, parse_edits(spec["fix"]), [])
    fv, _, _ = _standalone_verdict(sandbox, work, spec)
    out["fixed"] = {"stage": fv.stage, "reasons": fv.reasons}
    if not fv.verified:
        out["problems"].append(f"reference fix does not verify: {'; '.join(fv.reasons)}")
    hc = hidden_check(task, work, sandbox)
    out["hidden_check"] = hc
    if not hc["passed"]:
        out["problems"].append("hidden check fails on the reference fix: " + "; ".join(hc["failures"]))
    out["ok"] = not out["problems"]
    return out


# ---------------------------------------------------------------------------- hidden check
def hidden_check(task: Task, workdir: Path, sandbox: Sandbox) -> dict:
    spec = task.spec
    failures: list[str] = []
    details: dict = {}
    # 1. protected files untouched
    for p in task.repo.rglob("*"):
        if p.is_file():
            rel = p.relative_to(task.repo).as_posix()
            if is_protected(rel, DEFAULT_PROTECTED):
                q = workdir / rel
                if not q.is_file() or q.read_bytes() != p.read_bytes():
                    failures.append(f"protected file modified: {rel}")
    # 2. fresh run with a different data seed
    argv = parse_command(spec["command"]) + ["--seed", str(spec["hidden"]["seed"])]
    res = sandbox.run(workdir, argv, int(spec.get("timeout_s", 300)))
    metric = _metric(spec)
    val = parse_metric(res.stdout, metric)
    details["hidden_seed_metric"] = val
    if res.exit_code != 0 or val is None:
        failures.append("hidden-seed run failed or printed no metric")
    else:
        if abs(val - spec["metric"]["expected"]) > spec["hidden"]["tolerance"] + TOLERANCE_EPS:
            failures.append(f"hidden-seed metric {val:.4f} outside +/-{spec['hidden']['tolerance']} of {spec['metric']['expected']}")
        # 3. independent recomputation from saved predictions
        try:
            import numpy as np
            z = np.load(workdir / "artifacts" / "predictions.npz")
            recomputed = float(np.mean(z["preds"] == z["labels"]))
            details["recomputed"] = recomputed
            if abs(recomputed - val) > 1e-3:
                failures.append(f"printed metric {val:.4f} != recomputed {recomputed:.4f}")
        except Exception as exc:  # missing numpy or artifacts
            failures.append(f"could not recompute metric from predictions: {type(exc).__name__}")
    return {"passed": not failures, "failures": failures, **details}


# ---------------------------------------------------------------------------- run one task
def run_task(task: Task, *, settings: Settings, sandbox: Sandbox, backend: ChatBackend, run_dir: Path,
             router_mode: str = "router", max_attempts: int = 6, tavily=None, give_command: bool = False) -> dict:
    spec = task.spec
    s = copy.copy(settings)
    s.allow_local_paths = True
    run_dir.mkdir(parents=True, exist_ok=True)
    req = RunRequest(
        local_path=str(task.repo), goal=spec["goal"], command=spec["command"] if give_command else None,
        metric=_metric(spec), max_attempts=max_attempts, router_mode=router_mode,  # type: ignore[arg-type]
        max_run_seconds=max(900, 3 * task.timeout_s), command_timeout_s=task.timeout_s,
    )
    t0 = time.time()
    orch = Orchestrator(settings=s, request=req, run_dir=run_dir, sandbox=sandbox, backend=backend, tavily=tavily)
    report = orch.run()
    elapsed = time.time() - t0
    return {"task": task, "report": report, "orch": orch, "elapsed_s": elapsed}


def score(task: Task, outcome: dict, sandbox: Sandbox) -> dict:
    spec, report, orch = task.spec, outcome["report"], outcome["orch"]
    gt = spec["ground_truth"]
    base = (report["results"] or {}).get("baseline")
    kind = spec["symptom"]["kind"]
    understood = report["environment"].get("run_command") == spec["command"]
    reproduced = bool(base) and not base["verified"] and {
        "install_failure": base["stage"] == "install_failure", "crash": base["stage"] == "crash",
        "wrong_result": base["stage"] == "wrong_result"}[kind]
    pat = spec["symptom"].get("pattern")
    if reproduced and pat:
        text = "".join((e.stdout + e.stderr) for e in orch.execs[:3])
        reproduced = bool(re.search(pat, text, re.I))
    confirmed = orch.graph.accepted_hypotheses()
    covered = set()
    cats = set()
    for n in confirmed:
        covered |= set(n["data"].get("files", []))
        cats.add(n["data"].get("category"))
    root = set(gt["files"]) <= covered and (gt["category"] == "multi" or gt["category"] in cats)
    generated = bool(report["files_modified"])
    verified = False
    hidden: dict = {"passed": False, "failures": ["not run"]}
    if report["status"] == "verified":
        hidden = hidden_check(task, orch.work, sandbox)
        verified = hidden["passed"]
    usage = report["usage"]
    return {
        "id": task.id, "category": spec["category"], "difficulty": spec["difficulty"], "status": report["status"],
        "stages": {"understood": understood, "reproduced": reproduced, "root_cause": root,
                   "patch_generated": generated, "patch_verified": verified},
        "hidden_check": hidden, "attempts": len(report["experiments"]), "duration_s": round(outcome["elapsed_s"], 1),
        "tokens": usage["prompt_tokens"] + usage["completion_tokens"], "llm_calls": usage["calls"],
        "latency_llm_s": usage["latency_s"], "cost_usd_priced_tiers": usage["cost_usd_priced_tiers"],
        "has_unpriced_tiers": usage["has_unpriced_tiers"], "tiers": {k: v["calls"] for k, v in usage["by_tier"].items()},
        "files_modified": [f["path"] for f in report["files_modified"]], "error": report.get("error"),
    }


def summarize(results: list[dict], meta: dict) -> dict:
    n = len(results)
    stage_counts = {st: sum(1 for r in results if r["stages"][st]) for st in STAGES}
    cats: dict[str, dict] = {}
    for r in results:
        c = cats.setdefault(r["category"], {"n": 0, "patch_verified": 0})
        c["n"] += 1
        c["patch_verified"] += int(r["stages"]["patch_verified"])
    priced = [r["cost_usd_priced_tiers"] for r in results if r["cost_usd_priced_tiers"] is not None]
    return {
        "meta": meta, "n_tasks": n, "stages": stage_counts, "by_category": cats,
        "totals": {
            "tokens": sum(r["tokens"] for r in results), "llm_calls": sum(r["llm_calls"] for r in results),
            "llm_latency_s": round(sum(r["latency_llm_s"] for r in results), 1),
            "wall_s": round(sum(r["duration_s"] for r in results), 1),
            "cost_usd_priced_tiers": round(sum(priced), 6) if priced else None,
            "any_unpriced_tiers": any(r["has_unpriced_tiers"] for r in results),
            "mean_attempts": round(sum(r["attempts"] for r in results) / n, 2) if n else 0,
        },
        "results": results,
    }


def render_markdown(summary: dict) -> str:
    m, n = summary["meta"], summary["n_tasks"]
    lines = [f"# ReproBench results ({m.get('label', '')})", ""]
    if m.get("llm_backend") != "nebius":
        lines += [f"> **LLM backend: `{m.get('llm_backend')}` -- NOT Nemotron.** These numbers validate the "
                  "pipeline and benchmark only; they say nothing about model capability.", ""]
    lines += [f"- date: {m.get('date')}", f"- router mode: {m.get('router_mode')}", f"- sandbox: {m.get('sandbox')}",
              f"- models: {m.get('models')}", f"- tasks: {n}", ""]
    lines += ["| stage | passed |", "|---|---|"]
    for st in STAGES:
        lines.append(f"| {st.replace('_', ' ')} | {summary['stages'][st]}/{n} |")
    t = summary["totals"]
    cost = "n/a" if t["cost_usd_priced_tiers"] is None else f"${t['cost_usd_priced_tiers']:.4f}" + (" (priced tiers only)" if t["any_unpriced_tiers"] else "")
    lines += ["", f"tokens: {t['tokens']:,} · LLM calls: {t['llm_calls']} · LLM latency: {t['llm_latency_s']}s · "
              f"wall: {t['wall_s']}s · mean experiments/task: {t['mean_attempts']} · cost: {cost}", "",
              "| task | category | status | " + " | ".join(s.replace("_", " ") for s in STAGES) + " | experiments |",
              "|---|---|---|" + "---|" * len(STAGES) + "---|"]
    for r in summary["results"]:
        marks = " | ".join("✓" if r["stages"][s] else "✗" for s in STAGES)
        lines.append(f"| {r['id']} | {r['category']} | {r['status']} | {marks} | {r['attempts']} |")
    return "\n".join(lines) + "\n"


def run_benchmark(*, tasks: list[Task], settings: Settings, sandbox: Sandbox, backend_for: Callable[[Task], ChatBackend],
                  out_dir: Path, router_mode: str = "router", max_attempts: int = 6, tavily=None, label: str = "",
                  progress: Callable[[str], None] = print) -> dict:
    results = []
    backend_name = "unknown"
    for i, task in enumerate(tasks, 1):
        backend = backend_for(task)
        backend_name = backend.name
        progress(f"[{i}/{len(tasks)}] {task.id} ...")
        outcome = run_task(task, settings=settings, sandbox=sandbox, backend=backend, run_dir=out_dir / task.id,
                           router_mode=router_mode, max_attempts=max_attempts, tavily=tavily)
        r = score(task, outcome, sandbox)
        results.append(r)
        progress(f"    {r['status']:16} " + " ".join(("✓" if r["stages"][s] else "✗") for s in STAGES) + f"  ({r['duration_s']}s)")
    meta = {"label": label, "date": time.strftime("%Y-%m-%d %H:%M"), "llm_backend": backend_name, "router_mode": router_mode,
            "sandbox": sandbox.policy().kind, "models": settings.models if backend_name == "nebius" else "n/a (scripted)"}
    return summarize(results, meta)


def oracle_backend_for(task: Task) -> ChatBackend:
    return OracleBackend(task.spec)
