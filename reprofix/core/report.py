"""Evidence-backed report. Every verification line is computed from the final re-run and the real diff."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..agents.reviewer import review
from ..claims import report_card
from ..claims.check import DEFAULT_TOLERANCE_FRACTION, DEFAULT_TOLERANCE_PERCENT, DEFAULT_TOLERANCE_RELATIVE
from ..models import Verdict
from .patching import diff_hardcoding_findings, is_protected, make_diff

if TYPE_CHECKING:  # pragma: no cover
    from .orchestrator import Orchestrator

STATUS_TEXT = {
    "verified": "Verified: the project runs and the documented metric was reproduced within tolerance.",
    "executes": "Runs successfully. No expected metric was provided, so reproduction of a result was NOT checked.",
    "already_passing": "The project already ran and met its expectations; nothing was changed.",
    "partial": "Partially repaired: progress was made but the documented result was not reproduced.",
    "failed": "Not repaired: no change improved on the baseline.",
    "cancelled": "Cancelled before completion.",
    "error": "The run stopped because of an error.",
}


def patch_applies(orig: Path, diff: str) -> bool | None:
    if not diff.strip():
        return None
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "o"
        shutil.copytree(orig, dst)
        patch_bin = shutil.which("patch")
        if not patch_bin and os.name == "nt":
            for candidate in [
                r"C:\Program Files\Git\usr\bin\patch.exe",
                r"C:\Program Files (x86)\Git\usr\bin\patch.exe",
            ]:
                if os.path.exists(candidate):
                    patch_bin = candidate
                    break
        if patch_bin:
            try:
                cmd = [patch_bin, "-p1", "--dry-run", "-s"]
                if os.name == "nt":
                    cmd.insert(2, "--binary")
                r = subprocess.run(cmd, input=diff.encode("utf-8"), cwd=dst,
                                   capture_output=True, timeout=60)
                return r.returncode == 0
            except (OSError, subprocess.SubprocessError):
                pass
        git_bin = shutil.which("git")
        if git_bin:
            try:
                r = subprocess.run([git_bin, "apply", "--check", "-"], input=diff.encode("utf-8"), cwd=dst,
                                   capture_output=True, timeout=60)
                return r.returncode == 0
            except (OSError, subprocess.SubprocessError):
                pass
        return None


def _vsum(v: Verdict | None) -> dict | None:
    if v is None:
        return None
    return {"stage": v.stage, "verified": v.verified, "metric_value": v.metric_value, "gap": v.gap,
            "tests_ok": v.tests_ok, "reasons": v.reasons}


def _no_run_reason(v: Verdict | None) -> str:
    if v is not None and v.stage == "install_failure":
        return "dependency installation failed, so the program did not run"
    return "the program did not run"


def _n_claims(n: int) -> str:
    return f"{n} claim" if n == 1 else f"{n} claims"


def _claim_caveats(card: dict, o: "Orchestrator") -> list[str]:
    out: list[str] = []
    if not card["checked"]:
        if o.claims.had_text:
            out.append("The pasted paper text produced no claim: the patterns only read lines like 'accuracy of 76.4%' and "
                       "markdown tables with metric columns. Pass the number with --claim (or the claims field) instead.")
        return out
    if card["sources"]["extracted"]:
        n = card["sources"]["extracted"]
        out.append(f"{_n_claims(n)} {'was' if n == 1 else 'were'} read from the pasted text by fixed patterns, not understood "
                   f"by a model. Each is shown with its quote: a number can belong to a baseline or another model. "
                   f"Confirm {'it' if n == 1 else 'them'}.")
    head = o.claims.headline_claim
    if head is not None and head.source != "explicit" and not any(c.headline and c.source == "explicit" for c in o.claims.claims) \
            and o.req.metric is None:
        out.append(f"The repair loop targeted claim {card['headline']} ({head.metric} = {head.value:g}), chosen automatically as the first "
                   "claim in the authors' voice. Pass the claim you mean explicitly to target another one.")
    if head is None and o.req.metric is None:
        out.append("No claim was chosen as the repair target: every claim read from the text follows a comparison with other "
                   "methods. Pass the claim you mean explicitly (--claim, or the claims field) to repair toward it.")
    if any(c.tolerance is None for c in o.claims.claims):
        out.append(f"Claims without a stated tolerance use ±{DEFAULT_TOLERANCE_PERCENT:g} percentage point for percentages, "
                   f"±{DEFAULT_TOLERANCE_FRACTION:g} for fractions and ±{DEFAULT_TOLERANCE_RELATIVE * 100:g}% for other metrics. "
                   "That is a default chosen here, not something measured; a different seed or hardware can move a number by more.")
    regressed = [i["id"] for i in card["items"] if i["change"] == "regressed"]
    if regressed:
        out.append(f"{'Claim' if len(regressed) == 1 else 'Claims'} {', '.join(regressed)} "
                   f"{'was' if len(regressed) == 1 else 'were'} reproduced before the changes and "
                   f"{'is' if len(regressed) == 1 else 'are'} not after them.")
    if card["summary"]["not_measured"]:
        n = card["summary"]["not_measured"]
        out.append(f"{_n_claims(n)} could not be measured: the program's output has no matching "
                   "`name: number` line, or it did not finish. See the reason on each claim.")
    return out


def build_report(o: "Orchestrator", status: str, error: str | None, baseline: Verdict | None,
                 final: Verdict | None, stable: bool | None) -> dict:
    diff, files = make_diff(o.orig, o.work, o.kept_paths) if o.orig.exists() else ("", [])
    issues = []
    for node in o.graph.accepted_hypotheses():
        d = node["data"]
        issues.append({"category": d.get("category"), "statement": node["label"], "files": d.get("files", []),
                       "model_confidence": d.get("model_confidence"), "result": d.get("result", "")})
    metric = o.metric
    desc = f"{metric.name} ≈ {metric.expected} (±{metric.tolerance})" if metric and metric.expected is not None else "none"

    checks: list[dict] = []
    if final is not None:
        checks.append({"check": "Application executes", "passed": final.executes,
                       "detail": "; ".join(final.reasons[:1]) or final.stage})
        checks.append({"check": "Tests pass", "passed": final.tests_ok,
                       "detail": "not run: no tests found, or pytest is not a requirement of the repository" if final.tests_ok is None else ("pytest -q passed" if final.tests_ok else "pytest -q failed")})
        checks.append({"check": "Documented metric reproduced", "passed": final.metric_ok,
                       "detail": "no expected metric was given" if final.metric_ok is None else
                       next((r for r in final.reasons if "expected" in r or "not found" in r), "")})
        checks.append({"check": "Result is stable on a clean re-run", "passed": stable,
                       "detail": "n/a" if stable is None else ("re-run agreed" if stable else "re-run disagreed")})
    applies = patch_applies(o.orig, diff) if o.orig.exists() else None
    checks.append({"check": "Patch applies cleanly to the original repository", "passed": applies,
                   "detail": "no changes" if applies is None else "patch -p1 --dry-run"})
    checks.append({"check": "No protected files (tests, README) modified",
                   "passed": not any(is_protected(f["path"], o.req.protected_paths) for f in files), "detail": ""})
    hard = diff_hardcoding_findings(diff, metric.expected if metric else None, metric.name if metric else "metric")
    checks.append({"check": "Expected value not hardcoded in the patch", "passed": not hard, "detail": hard[0] if hard else ""})

    rv = None
    if diff.strip() and status in {"verified", "executes", "partial"}:
        rv = review(o.router, diff, [i["statement"] for i in issues], desc)

    caveats: list[str] = []
    pol = o.sandbox.policy()
    if not pol.isolated:
        caveats.append("Sandbox backend is DEV-ONLY and does not isolate the filesystem; do not run untrusted repositories with it.")
    if o.backend.name != "nebius":
        caveats.append(f"LLM backend is '{o.backend.name}', NOT NVIDIA Nemotron on Nebius. Results validate the pipeline only and say nothing about model quality.")
    if o.metric_source.startswith("inferred"):
        caveats.append("The expected metric was inferred from the README by pattern matching; confirm it is the right number.")
    if metric is None or metric.expected is None:
        caveats.append("No expected metric: 'verified' can only mean the project executes.")
    claims_card = report_card(o.claims, o.baseline_rec, o.adopted_rec, baseline_reason=_no_run_reason(baseline),
                              final_reason=_no_run_reason(final))
    if claims_card.get("checked") and o.metric_source == "user":      # the person named the metric: no paper claim was the repair target
        claims_card["headline"] = None
        for it in claims_card["items"]:
            it["headline"] = False
    hw = o.hardware or {}
    if claims_card.get("checked") and hw:
        claims_card["measured_on"] = hw.get("summary")
    if hw.get("requested") and hw.get("status") != "detected":
        caveats.append(f"GPU was requested (REPROFIX_SANDBOX_GPU=1) but the probe says: {hw.get('summary')}. {hw.get('detail', '')} "
                       "The run used the CPU or failed for that reason.")
    caveats += _claim_caveats(claims_card, o)
    known = o.issue_log.to_dict(enabled=o.issues is not None)
    if known["results"]:
        n_found = len(known["results"])
        caveats.append(f"{n_found} possibly related {'issue was' if n_found == 1 else 'issues were'} found by searching for words from the failure. "
                       "The model read them as untrusted text; ReproFix did not verify that any of them describes this failure.")
    for e in known["errors"]:
        caveats.append("Known-issues search: " + e)
    usage = o.usage.summary()
    if usage["has_unpriced_tiers"]:
        caveats.append("Cost covers only tiers with a configured price; unpriced tiers are shown as n/a.")
    if o.tavily and o.tavily.errors:
        caveats.append(f"{len(o.tavily.errors)} Tavily request(s) failed; external evidence may be incomplete.")

    return {
        "version": 1,
        "status": status,
        "status_text": STATUS_TEXT.get(status, status),
        "error": error,
        "stop_reason": o.stop_reason,
        "goal": o.req.goal,
        "repository": {"source": o.req.repo_url or o.req.local_path, "name": (o.req.repo_url or o.req.local_path or "").rstrip("/").split("/")[-1].removesuffix(".git"),
                       "commit": o.provenance.get("commit") or None},
        "environment": {k: o.analysis.get(k) for k in ("framework", "task", "model", "dataset", "run_command", "run_command_source")},
        "metric": ({"name": metric.name, "expected": metric.expected, "tolerance": metric.tolerance, "source": o.metric_source}
                   if metric else None),
        "results": {"baseline": _vsum(baseline), "final": _vsum(final),
                    "progression": [{"experiment": e["n"], "kept": e["kept"], "stage": e["after"]["stage"],
                                     "metric_value": e["after"]["metric_value"], "reason": e["reason"]} for e in o.experiments]},
        "claims": claims_card,
        "hardware": hw,
        "known_issues": known,
        "issues": issues,
        "experiments": [{k: v for k, v in e.items() if k != "diff"} for e in o.experiments],
        "files_modified": files,
        "diff": diff,
        "verification": checks,
        "review": rv,
        "graph": o.graph.to_dict(),
        "executions": [{"id": r.id, "phase": r.phase, "argv": r.argv, "exit_code": r.exit_code, "duration_s": r.duration_s,
                        "timed_out": r.timed_out, "network": r.network, "metric_value": r.metric_value} for r in o.execs],
        "usage": usage,
        "routing": {"mode": o.req.router_mode, "decisions": o.router.decisions, "unavailable_tiers": sorted(o.router.unavailable)},
        "llm_backend": o.backend.name,
        "sandbox": pol.to_dict(),
        "tavily": {"enabled": bool(o.tavily and o.tavily.available), "calls": o.tavily.calls if o.tavily else 0},
        "caveats": caveats,
        "timing": {"started": o.t0, "duration_s": round(time.time() - o.t0, 2)},
    }
