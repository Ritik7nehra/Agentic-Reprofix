"""The ReproFix loop: reproduce -> diagnose -> repair -> verify, with every step recorded as evidence."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from ..agents.analyzer import analyze
from ..agents.common import AgentError
from ..agents.diagnoser import Diagnosis, Hypothesis, choose_hypothesis, diagnose
from ..agents.planner import plan as make_plan
from ..agents.repairer import repair
from ..agents.reviewer import review
from ..agents.tools import ToolBox
from ..claims import ClaimSet, collect as collect_claims, metric_for
from ..config import Settings
from ..inference import AuthError, ChatBackend, ModelRouter, ModelUnavailable, TavilyClient, UsageLog
from ..inference.tavily import EXTERNAL_HINTS
from ..models import ExecRecord, MetricSpec, Observation, RunRequest, Verdict
from ..sandbox.base import DEPS_DIR, ExecResult, Sandbox
from ..util import UnsafeCommand, parse_command
from .evidence import EvidenceGraph
from .issues import IssueFinder, IssueLog
from .patching import PatchError, apply_edits, literal_hardcoding_findings, make_diff, revert
from .repo import RepoError, acquire, scan_repo
from .verify import assess_progress, make_observation, make_verdict, parse_metric

Emit = Callable[[str, dict], None]

STEP_LABELS = [
    ("inspect", "Inspect repository"), ("deps", "Parse dependencies"), ("entry", "Identify entry points"),
    ("env", "Build environment"), ("baseline", "Run baseline"), ("analyze", "Analyze failures"),
    ("hypothesize", "Generate hypotheses"), ("test", "Test fixes"), ("verify", "Verify results"),
]


class Cancelled(Exception):
    pass


class Orchestrator:
    def __init__(self, *, settings: Settings, request: RunRequest, run_dir: Path, sandbox: Sandbox,
                 backend: ChatBackend, emit: Emit | None = None, cancel: threading.Event | None = None,
                 tavily: TavilyClient | None = None, issues: IssueFinder | None = None):
        self.s = settings
        self.req = request
        self.run_dir = run_dir.resolve()  # absolute: sandboxes build PYTHONPATH/mount paths from it
        self.work = self.run_dir / "work"
        self.orig = self.run_dir / "orig"
        self.sandbox = sandbox
        self.backend = backend
        self._emit = emit or (lambda kind, data: None)
        self.cancel = cancel or threading.Event()
        self.tavily = tavily
        self.issues = issues                      # known-issues search (GitHub issues + web); None = off
        self.issue_log = IssueLog()
        self.usage = UsageLog(prices=settings.prices)
        self.router = ModelRouter(backend, self.usage, request.router_mode)
        self.graph = EvidenceGraph()
        self.execs: list[ExecRecord] = []
        self.kept_paths: list[str] = []
        self.experiments: list[dict] = []
        self.metric: MetricSpec | None = None
        self.metric_source = "none"
        self.claims = ClaimSet(collected=not (request.paper_text.strip() or request.claims))      # numbers from the paper (explicit or extracted); see reprofix/claims
        self.hardware: dict = {}                  # what the run phase can see (nvidia-smi record), see sandbox/hardware.py
        self.baseline_rec: ExecRecord | None = None   # the baseline run of the main command
        self.adopted_rec: ExecRecord | None = None    # the run of the state ReproFix ends with (kept patches, or the final re-run)
        self.argv: list[str] = []
        self.analysis: dict = {}
        self.scan = None
        self.provenance: dict = {}   # where the code came from (source, commit); recorded in the report
        self._install_state: tuple[str, ExecResult | None] | None = None
        self._hyp_nodes: dict[tuple[int, str], str] = {}
        self.stop_reason = ""
        self._n_exec = 0
        self.t0 = time.time()
        self.max_attempts = max(1, min(request.max_attempts, settings.max_attempts_ceiling))
        self.deadline = self.t0 + min(request.max_run_seconds, settings.max_run_seconds_ceiling)
        self.cmd_timeout = max(5, min(request.command_timeout_s, settings.max_command_timeout_ceiling))

    # ------------------------------------------------------------------ helpers
    def emit(self, kind: str, data: dict | None = None) -> None:
        self._emit(kind, data or {})

    def step(self, sid: str, status: str, detail: str = "") -> None:
        label = dict(STEP_LABELS)[sid]
        self.emit("step", {"id": sid, "label": label, "status": status, "detail": detail})

    def _publish_graph(self) -> None:
        self.emit("graph", self.graph.to_dict())

    def _check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    def _budget_left(self) -> str | None:
        if time.time() > self.deadline:
            return "time budget exhausted"
        if self.usage.total_tokens >= self.req.max_total_tokens:
            return "token budget exhausted"
        return None

    def _record(self, phase: str, res: ExecResult) -> ExecRecord:
        self._n_exec += 1
        rec = ExecRecord(
            id=f"e{self._n_exec}", phase=phase, argv=res.argv, exit_code=res.exit_code, duration_s=round(res.duration_s, 2),
            timed_out=res.timed_out, network=res.network, stdout=res.stdout, stderr=res.stderr, truncated=res.truncated,
            metric_value=parse_metric(res.stdout, self.metric) if phase != "install" else None,
        )
        self.execs.append(rec)
        self.emit("exec", rec.model_dump())
        return rec

    # ------------------------------------------------------------------ environment
    def _req_files(self) -> list[str]:
        root = self.work / "requirements.txt"
        if root.is_file():
            return ["requirements.txt"]
        return sorted(p.name for p in self.work.glob("requirements*.txt"))

    def _req_hash(self, files: list[str]) -> str:
        h = hashlib.sha256()
        for f in files:
            h.update(f.encode())
            h.update((self.work / f).read_bytes())
        return h.hexdigest()

    def _ensure_env(self, phase: str = "experiment") -> ExecResult | None:
        """Install requirements if they changed. Returns the install result when relevant, else None."""
        files = self._req_files()
        if not files:
            shutil.rmtree(self.work / DEPS_DIR, ignore_errors=True)
            self._install_state = None
            return None
        key = self._req_hash(files)
        if self._install_state and self._install_state[0] == key:
            res = self._install_state[1]
            return res if (res is not None and res.exit_code != 0) else None
        shutil.rmtree(self.work / DEPS_DIR, ignore_errors=True)
        self.emit("exec.start", {"id": f"e{self._n_exec + 1}", "phase": "install", "argv": ["pip", "install", "-r", *files]})
        res = self.sandbox.install(self.work, files, self.cmd_timeout)
        self._record("install", res)
        self._install_state = (key, res)
        return res if res.exit_code != 0 or res.timed_out else None

    def _has_tests(self) -> bool:
        return (self.work / "tests").is_dir() or any(self.work.glob("test_*.py"))

    def _execute(self, phase: str) -> tuple[Verdict, dict]:
        """Run install (if needed) + main command + tests, and compute the verdict."""
        self._check()
        failed_install = self._ensure_env(phase)
        if failed_install is not None:
            v = make_verdict(install=failed_install, run=None, tests=None, metric=self.metric, workdir=self.work)
            return v, {"install": failed_install, "run": None, "tests": None}
        self.emit("exec.start", {"id": f"e{self._n_exec + 1}", "phase": phase, "argv": self.argv})
        run = self.sandbox.run(self.work, self.argv, self.cmd_timeout)
        run_rec = self._record(phase, run)
        tests = None
        if self._has_tests():
            targv = ["pytest", "-q", "-p", "no:cacheprovider"]
            self.emit("exec.start", {"id": f"e{self._n_exec + 1}", "phase": "tests", "argv": targv})
            tests = self.sandbox.run(self.work, targv, min(self.cmd_timeout, 300))
            self._record("tests", tests)
        v = make_verdict(install=None, run=run, tests=tests, metric=self.metric, workdir=self.work)
        return v, {"install": None, "run": run, "tests": tests, "run_rec": run_rec}

    # ------------------------------------------------------------------ main
    def run(self) -> dict:
        status = "error"
        error: str | None = None
        baseline: Verdict | None = None
        final: Verdict | None = None
        stable: bool | None = None
        try:
            self.emit("run.started", {
                "request": self.req.model_dump(), "sandbox": self.sandbox.policy().to_dict(),
                "llm_backend": self.backend.name, "router_mode": self.req.router_mode,
                "tavily": bool(self.tavily and self.tavily.available),
            })
            baseline, final, stable, status = self._pipeline()
        except Cancelled:
            status, error = "cancelled", "cancelled by user"
        except AuthError as exc:
            status, error = "error", f"LLM authentication failed: {exc}"
        except ModelUnavailable as exc:
            status, error = "error", f"no usable model: {exc}"
        except (RepoError, UnsafeCommand) as exc:
            status, error = "error", str(exc)
        except Exception as exc:  # last resort: surface it, don't hide it
            status, error = "error", f"{type(exc).__name__}: {exc}"
        report = self._report(status, error, baseline, final, stable)
        (self.run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        self.emit("usage", self.usage.summary())
        self.emit("report", report)
        self.emit("run.finished", {"status": report["status"]})
        return report

    def _pipeline(self) -> tuple[Verdict, Verdict, bool | None, str]:
        # 1. acquire + analyse
        self.step("inspect", "running")
        info = acquire(repo_url=self.req.repo_url, local_path=self.req.local_path, settings=self.s,
                       work=self.work, orig=self.orig)
        self.provenance = dict(info)
        self.scan = scan_repo(self.work)
        self.emit("acquired", info)
        self.analysis = analyze(self.router, self.scan, self.work, self.req)
        self.claims = collect_claims(self.req)
        if self.claims or self.claims.had_text:
            self.emit("claims", {"claims": [{"id": i, **c.model_dump()} for i, c in zip(self.claims.ids, self.claims.claims)],
                                 "headline": self.claims.ids[self.claims.headline] if self.claims.headline is not None else None,
                                 "skipped": self.claims.skipped})
        self._resolve_metric()
        self.emit("analysis", {"analysis": self.analysis, "scan": self.scan.to_dict(), "metric": self.metric.model_dump() if self.metric else None,
                               "metric_source": self.metric_source})
        self.step("inspect", "done", f"{self.analysis.get('framework')} · {self.analysis.get('task') or 'task unknown'}")
        self.step("deps", "done", ", ".join(self.scan.requirements) or "no requirements file")
        cmd = self.analysis.get("run_command")
        if not cmd:
            raise UnsafeCommand("could not determine how to run the project; provide an explicit command")
        self.argv = parse_command(cmd)
        self.step("entry", "done", f"{cmd}  (from {self.analysis['run_command_source']})")
        desc = f"{self.metric.name} ≈ {self.metric.expected}" if self.metric and self.metric.expected is not None else "none specified"
        plan = make_plan(self.router, self.analysis, self.req.goal, desc)
        self.emit("plan", plan)
        self._check()

        # 2. baseline
        self.hardware = self._probe_hardware()
        self.step("env", "running")
        self.step("baseline", "running")
        verdict, parts = self._execute("baseline")
        baseline = verdict
        self.baseline_rec = self.adopted_rec = parts.get("run_rec")
        self.step("env", "done" if verdict.stage != "install_failure" else "failed")
        self.step("baseline", "done", verdict.stage)
        self.emit("verdict", {"label": "baseline", "verdict": verdict.model_dump()})
        sym = self.graph.add("symptom", self._symptom_label(verdict, parts), status="open", stage=verdict.stage,
                             reasons=verdict.reasons)
        self._publish_graph()
        if verdict.verified:
            self.step("analyze", "skipped", "already passing")
            self.step("hypothesize", "skipped")
            self.step("test", "skipped")
            return baseline, self._final_check(verdict), None, "already_passing"

        # 3. repair loop
        rejected: list[dict] = []
        kept: list[str] = []
        consecutive_rejections = 0
        agent_errors = 0
        stop_reason = ""
        self.step("analyze", "running")
        for attempt in range(1, self.max_attempts + 1):
            self._check()
            why = self._budget_left()
            if why:
                stop_reason = why
                break
            obs = make_observation(verdict, install=parts["install"], run=parts["run"], tests=parts["tests"],
                                   metric=self.metric, workdir=self.work)
            toolbox = ToolBox(self.work, self.tavily)
            extra = ""
            if self.tavily and self.tavily.available and EXTERNAL_HINTS.search(obs.summary + " " + obs.log_tail[-800:]):
                q = f"{obs.exception_type or ''} {obs.exception_message or obs.summary}".strip()
                extra = toolbox.run({"tool": "search_web", "query": q})
            extra = "\n\n".join(x for x in (extra, self._known_issues(obs, toolbox, attempt)) if x)
            escalate = min(consecutive_rejections, 2)
            self.step("hypothesize", "running", f"attempt {attempt}")
            try:
                diag = diagnose(
                    self.router, toolbox, request=self.req, analysis=self.analysis, observation=obs,
                    file_list=self.scan.files, plan=plan, rejected=rejected, kept=kept,
                    log_text=obs.log_tail + "\n" + "\n".join(r.stdout + r.stderr for r in self.execs[-3:]),
                    readme=self.scan.readme_text, extra_context=extra, escalate=escalate,
                )
            except (AgentError, ) as exc:
                agent_errors += 1
                self.emit("attempt.failed", {"attempt": attempt, "stage": "diagnose", "error": str(exc)})
                if agent_errors >= 2:
                    stop_reason = f"diagnosis kept failing: {exc}"
                    break
                continue
            agent_errors = 0
            self._record_diagnosis(sym, diag, obs, attempt)
            chosen = choose_hypothesis(diag, rejected)
            self.step("analyze", "done")
            if chosen is None:
                stop_reason = "the model only re-proposed hypotheses that experiments already rejected"
                break
            self.emit("hypothesis.selected", {"attempt": attempt, "id": chosen.id, "statement": chosen.statement})
            hyp_node = self._hyp_nodes[(attempt, chosen.id)]

            self.step("test", "running", chosen.statement[:80])
            try:
                edits, meta = repair(self.router, workdir=self.work, request=self.req, hypothesis=chosen, observation=obs,
                                     requirement_files=self._req_files(), escalate=escalate, protected=self.req.protected_paths)
                findings = literal_hardcoding_findings(edits, self.metric.expected if self.metric else None,
                                                       self.metric.name if self.metric else "metric")
                if findings:
                    raise PatchError("patch hardcodes the expected metric: " + "; ".join(findings))
                self.emit("edit.proposed", {"attempt": attempt, "edits": [e.to_dict() for e in edits], **meta})
                applied = apply_edits(self.work, edits, self.req.protected_paths)
            except (PatchError, AgentError) as exc:
                self._reject(hyp_node, chosen, rejected, f"no valid patch: {exc}")
                consecutive_rejections += 1
                self.graph.set_status(hyp_node, "rejected")
                self.graph.update_data(hyp_node, result=f"no valid patch: {exc}")
                self.emit("attempt.failed", {"attempt": attempt, "stage": "repair", "error": str(exc)})
                self._publish_graph()
                continue

            exp = self.graph.add("experiment", f"Experiment {attempt}: {meta['rationale'] or chosen.statement}"[:200],
                                 status="info", files=applied.changed, edits=[e.to_dict() for e in edits])
            self.graph.link(hyp_node, exp, "tested_by")
            new_verdict, new_parts = self._execute("experiment")
            ok, reason = assess_progress(verdict, new_verdict)
            res_label = self._result_label(new_verdict, new_parts)
            res = self.graph.add("result", res_label, status="verified" if new_verdict.verified else ("confirmed" if ok else "rejected"),
                                 stage=new_verdict.stage, reasons=new_verdict.reasons, kept=ok, why=reason)
            self.graph.link(exp, res, "produced")
            diff, _ = make_diff(self.orig, self.work, applied.changed)
            self.experiments.append({"n": attempt, "hypothesis": chosen.to_dict(), "kept": ok, "reason": reason,
                                     "files": applied.changed, "rationale": meta["rationale"],
                                     "before": verdict.model_dump(), "after": new_verdict.model_dump(), "diff": diff})
            self.emit("experiment", self.experiments[-1])
            self.emit("verdict", {"label": f"experiment {attempt}", "verdict": new_verdict.model_dump()})
            if ok:
                for p in applied.changed:
                    if p not in self.kept_paths:
                        self.kept_paths.append(p)
                kept.append(chosen.statement)
                self.graph.set_status(hyp_node, "confirmed")
                self.graph.update_data(hyp_node, result=reason)
                consecutive_rejections = 0
                verdict, parts = new_verdict, new_parts
                self.adopted_rec = parts.get("run_rec")          # what the claims are measured on if this is where we stop
                if not verdict.verified:
                    sym = self.graph.add("symptom", self._symptom_label(verdict, parts), status="open",
                                         stage=verdict.stage, reasons=verdict.reasons)
                    self.graph.link(res, sym, "remaining_symptom")
                    self.step("analyze", "running", "continuing investigation")
            else:
                revert(self.work, applied)  # requirement hash changes back -> _ensure_env reinstalls next run
                self._reject(hyp_node, chosen, rejected, reason)
                self.graph.set_status(hyp_node, "rejected")
                self.graph.update_data(hyp_node, result=reason)
                consecutive_rejections += 1
            self._publish_graph()
            if verdict.verified:
                break
        else:
            stop_reason = stop_reason or "attempt limit reached"

        self.step("test", "done")
        self.step("hypothesize", "done")
        # 4. final verification from a clean re-run
        self.step("verify", "running")
        final = self._final_check(verdict) if verdict.verified else verdict
        stable = None
        if verdict.verified:
            stable = final.verified
        self.stop_reason = stop_reason
        if baseline.stage == "install_failure" and final.stage != "install_failure":
            self.step("env", "done", "dependency problem repaired")   # it was marked failed after the baseline
        self.step("verify", "done" if final.verified else "failed", stop_reason)
        self._publish_graph()
        if final.verified and (stable is not False):
            return baseline, final, stable, "verified" if (self.metric and self.metric.expected is not None) else "executes"
        better = final.stage != "install_failure" and (
            baseline.stage == "install_failure" or (baseline.stage == "crash" and final.stage != "crash")
            or (baseline.gap is not None and final.gap is not None and final.gap < baseline.gap))
        return baseline, final, stable, "partial" if better else "failed"

    def _probe_hardware(self) -> dict:
        try:
            rec = self.sandbox.probe_hardware(self.work)
        except Exception as exc:                  # never stop a run for this
            from ..sandbox import hardware
            rec = hardware.record(requested=False, status="probe_failed", probed_in=None, detail=f"{type(exc).__name__}: {str(exc)[:160]}")
        self.emit("hardware", rec)
        return rec

    def _known_issues(self, obs: Observation, toolbox: ToolBox, attempt: int) -> str:
        """Search for the failure that was just observed. The results are shown in the report and handed to the diagnoser as
        untrusted text; a URL becomes citable evidence only because this search returned it."""
        if self.issues is None:
            return ""
        found = self.issues.find(repo_url=self.req.repo_url, observation=obs, metric=self.metric)
        if found.searched:
            self.issue_log.add(found, attempt)
            self.emit("known_issues", {"attempt": attempt, "query": found.query, "searched": found.searched,
                                       "results": [r.to_dict() for r in found.results], "errors": found.errors})
        if not found.results:
            return ""
        toolbox.web_results.extend(found.as_web_evidence())
        return found.as_context()

    def _final_check(self, verdict: Verdict) -> Verdict:
        v, parts = self._execute("final")
        self.adopted_rec = parts.get("run_rec")
        self.emit("verdict", {"label": "final re-run", "verdict": v.model_dump()})
        return v

    # ------------------------------------------------------------------ bookkeeping
    def _resolve_metric(self) -> None:
        if self.req.metric is not None:
            self.metric, self.metric_source = self.req.metric, "user"
            return
        headline = self.claims.headline_claim
        if headline is not None:
            self.metric = metric_for(headline)
            self.metric_source = (f"paper claim {self.claims.ids[self.claims.headline]}: "
                                  + (headline.quote or f"{headline.metric} = {headline.value:g}")[:160])
            return
        em = self.analysis.get("expected_metric")
        if em:
            name = {"acc": "accuracy"}.get(em["name"], em["name"])
            self.metric = MetricSpec(name=name, expected=float(em["value"]))
            self.metric_source = "inferred from README: " + em["source"]

    def _symptom_label(self, v: Verdict, parts: dict) -> str:
        if v.stage == "install_failure":
            return "Dependency installation fails"
        if v.stage == "crash":
            o = make_observation(v, install=None, run=parts["run"], tests=parts["tests"], metric=self.metric, workdir=self.work)
            return f"Crash: {o.summary}"[:160]
        return ("Result differs from expected: " + "; ".join(v.reasons))[:200]

    def _result_label(self, v: Verdict, parts: dict) -> str:
        if v.verified:
            return "Verified: " + "; ".join(v.reasons)
        return self._symptom_label(v, parts)

    def _reject(self, node: str, h: Hypothesis, rejected: list[dict], result: str) -> None:
        rejected.append({"statement": h.statement, "result": result})

    def _record_diagnosis(self, sym: str, diag: Diagnosis, obs: Observation, attempt: int) -> None:
        for h in diag.hypotheses:
            node = self.graph.add("hypothesis", h.statement, status="untested", category=h.category,
                                  model_confidence=h.model_confidence, files=h.files, test=h.test, attempt=attempt)
            self._hyp_nodes[(attempt, h.id)] = node
            self.graph.link(sym, node, "explained_by")
            for ev in h.evidence:
                label = (ev.get("quote") or ev.get("url") or ev.get("note") or "")[:140]
                ev_node = self.graph.add("evidence", label, status="verified" if ev["verified"] else "unverified", **{
                    k: v for k, v in ev.items() if k != "verified"})
                self.graph.link(node, ev_node, "supported_by")
        self.emit("diagnosis", {"attempt": attempt, "summary": diag.summary, "recommended": diag.recommended,
                                "hypotheses": [h.to_dict() for h in diag.hypotheses], "tool_calls": diag.tool_calls,
                                "grounding_retry": diag.grounding_retry,
                                "web_results": [w.to_dict() for w in diag.web_results],
                                "routing": self.router.decisions[-3:]})
        self._publish_graph()

    # ------------------------------------------------------------------ report
    def _report(self, status: str, error: str | None, baseline: Verdict | None, final: Verdict | None,
                stable: bool | None) -> dict:
        from .report import build_report
        return build_report(self, status, error, baseline, final, stable)
