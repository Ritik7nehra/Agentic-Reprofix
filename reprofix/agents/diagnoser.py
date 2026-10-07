"""Diagnoser (Nemotron Super/Ultra): inspect the repo with read-only tools, then propose ranked hypotheses.

Evidence is checked by code, not trusted: a quoted code/log snippet must really occur in the cited
file/log, and an external URL must really have been returned by a search in this session.
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..inference import ModelRouter, WebEvidence
from ..models import Observation, RunRequest
from ..util import normalize_ws, safe_join
from .common import AgentError, call_json, fence, system_prompt
from .tools import ToolBox

CATEGORIES = ["dependency", "architecture", "data", "preprocessing", "training", "evaluation",
              "configuration", "checkpoint", "code", "device", "environment", "other"]

PROMPT = """
You are the diagnosis agent of ReproFix. A machine-learning repository was run in a sandbox and it failed
or did not reproduce its documented result. Find the root cause.

Work like an engineer: read the failure, form competing hypotheses, and inspect the code with the tools
before concluding. Prefer a hypothesis that explains ALL of the symptoms. Do not repeat a hypothesis that was
already tested and rejected. When a stack trace points at a line, read that file. When nothing crashed (the
result is simply wrong), compare the code against what the README/config says it should do.

{tools}

Each turn, reply with ONE of:
  {{"action": "inspect", "calls": [ ...up to 4 tool calls... ]}}
  {{"action": "conclude", "summary": "...", "recommended": "h1",
    "hypotheses": [
      {{"id": "h1", "statement": "one precise sentence naming the faulty code/config",
        "category": "{categories}",
        "model_confidence": "low|medium|high",
        "files": ["path/of/file/to/change"],
        "evidence": [
          {{"kind": "code", "file": "model.py", "quote": "text copied EXACTLY from that file", "note": "why it matters"}},
          {{"kind": "execution", "quote": "text copied EXACTLY from the log", "note": "..."}},
          {{"kind": "external", "url": "a URL returned by search_web", "note": "..."}}],
        "test": "the change that would confirm or refute this"}}]}}

Rules for evidence: quotes must be copied verbatim (they will be checked against the files and logs; unverifiable
evidence is discarded). Never invent URLs. Give 1-4 hypotheses, most likely first.
"""


@dataclass
class Hypothesis:
    id: str
    statement: str
    category: str
    model_confidence: str
    files: list[str]
    evidence: list[dict]
    test: str = ""

    @property
    def verified_evidence(self) -> list[dict]:
        return [e for e in self.evidence if e.get("verified")]

    def to_dict(self) -> dict:
        return {"id": self.id, "statement": self.statement, "category": self.category,
                "model_confidence": self.model_confidence, "files": self.files, "evidence": self.evidence, "test": self.test}


@dataclass
class Diagnosis:
    summary: str
    hypotheses: list[Hypothesis]
    recommended: str | None
    tool_calls: list[dict] = field(default_factory=list)
    web_results: list[WebEvidence] = field(default_factory=list)
    grounding_retry: bool = False


def _verify_evidence(ev: dict, workdir: Path, log_text: str, urls: set[str]) -> dict:
    kind = ev.get("kind")
    out = {"kind": kind, "note": str(ev.get("note", ""))[:300], "verified": False}
    quote = str(ev.get("quote", ""))[:400]
    if kind == "code":
        out.update(file=str(ev.get("file", "")), quote=quote)
        try:
            text = safe_join(workdir, out["file"]).read_text(encoding="utf-8", errors="replace")
            nq = normalize_ws(quote)
            out["verified"] = bool(nq) and nq in normalize_ws(text)
            if out["verified"]:  # best-effort line number for the UI
                first = next((ln for ln in quote.splitlines() if ln.strip()), "").strip()
                for i, ln in enumerate(text.splitlines(), 1):
                    if first and first in ln:
                        out["line"] = i
                        break
        except (ValueError, OSError):
            pass
    elif kind == "execution":
        out.update(quote=quote)
        nq = normalize_ws(quote)
        out["verified"] = bool(nq) and nq in normalize_ws(log_text)
    elif kind == "external":
        url = str(ev.get("url", ""))
        out.update(url=url)
        out["verified"] = url in urls
    else:
        out["kind"] = "other"
    return out


def _parse(obj: dict, workdir: Path, log_text: str, urls: set[str]) -> Diagnosis:
    hyps: list[Hypothesis] = []
    for i, h in enumerate(obj.get("hypotheses", [])[:4], 1):
        if not isinstance(h, dict) or not isinstance(h.get("statement"), str):
            continue
        cat = h.get("category") if h.get("category") in CATEGORIES else "other"
        conf = h.get("model_confidence") if h.get("model_confidence") in ("low", "medium", "high") else "low"
        files = [f for f in h.get("files", []) if isinstance(f, str)][:4]
        evidence = [_verify_evidence(e, workdir, log_text, urls) for e in h.get("evidence", []) if isinstance(e, dict)][:5]
        hyps.append(Hypothesis(id=str(h.get("id") or f"h{i}"), statement=h["statement"][:400], category=cat,
                               model_confidence=conf, files=files, evidence=evidence, test=str(h.get("test", ""))[:300]))
    return Diagnosis(summary=str(obj.get("summary", ""))[:600], hypotheses=hyps, recommended=obj.get("recommended"))


def diagnose(router: ModelRouter, toolbox: ToolBox, *, request: RunRequest, analysis: dict, observation: Observation,
             file_list: list[str], plan: dict, rejected: list[dict], kept: list[str], log_text: str,
             readme: str, escalate: int, extra_context: str = "", max_tool_steps: int = 5) -> Diagnosis:
    behavioral = observation.kind in ("metric_gap", "test_failure")
    purpose = "diagnose_behavioral" if behavioral else "diagnose"
    system = system_prompt("diagnose", PROMPT.format(tools=toolbox.describe(), categories="|".join(CATEGORIES)))
    ctx = [
        f"User goal: {request.goal}",
        f"Project: framework={analysis.get('framework')}, task={analysis.get('task')}, model={analysis.get('model')}, dataset={analysis.get('dataset')}",
        f"Command: {analysis.get('run_command')}",
        f"Failure kind: {observation.kind}",
        f"Failure summary: {observation.summary}",
    ]
    if observation.location:
        ctx.append(f"Deepest in-repo frame: {observation.location}")
    if observation.metric_value is not None and observation.expected is not None:
        ctx.append(f"Measured {observation.metric_value:.4f}; documented/expected {observation.expected:.4f}")
    if plan.get("suspect_areas"):
        ctx.append("Planner's ranked suspects (advisory): " + json.dumps(plan["suspect_areas"])[:1200])
    if kept:
        ctx.append("Fixes already applied and kept (do not redo): " + "; ".join(kept))
    if rejected:
        ctx.append("Hypotheses already TESTED AND REJECTED by experiment (do not repeat):\n" + "\n".join(
            f"- {r['statement']} -> {r['result']}" for r in rejected))
    user = "\n".join(ctx) + "\n\n" + fence("log tail", observation.log_tail, 8000) + "\n" + \
        fence("file listing", "\n".join(file_list[:120]), 3000) + "\n" + fence("README", readme[:5000] or "(none)", 5000)
    if extra_context:
        user += "\n" + fence("search results: known issues and web search (GitHub, Tavily)", extra_context, 6000)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    obj: dict | None = None
    for step in range(max_tool_steps + 1):
        force = step == max_tool_steps
        if force:
            messages.append({"role": "user", "content": "You have used all inspection steps. Conclude now."})
        obj, _ = call_json(router, purpose, messages, escalate=escalate)
        if obj.get("action") == "inspect" and not force and isinstance(obj.get("calls"), list):
            outputs = [f"### call {json.dumps(c)}\n{toolbox.run(c)}" for c in obj["calls"][:4] if isinstance(c, dict)]
            messages += [{"role": "assistant", "content": json.dumps(obj)[:3000]},
                         {"role": "user", "content": fence("tool results", "\n\n".join(outputs), 14000)}]
            continue
        if obj.get("action") == "conclude" or "hypotheses" in obj:
            break
    else:  # pragma: no cover
        raise AgentError("diagnoser never concluded")
    assert obj is not None

    urls = {w.url for w in toolbox.web_results}
    diag = _parse(obj, toolbox.workdir, log_text, urls)
    chosen = next((h for h in diag.hypotheses if h.id == diag.recommended), diag.hypotheses[0] if diag.hypotheses else None)
    if chosen and not chosen.verified_evidence:
        # One grounding round: ask for verbatim evidence for the leading hypothesis.
        diag.grounding_retry = True
        messages += [{"role": "assistant", "content": json.dumps(obj)[:3000]},
                     {"role": "user", "content": "None of your evidence for the leading hypothesis could be verified "
                      "(quotes must be copied verbatim from files/logs; URLs must come from search_web). "
                      "Use read_file/grep if needed, then conclude again with verifiable evidence."}]
        for _ in range(3):
            obj2, _ = call_json(router, purpose, messages, escalate=escalate)
            if obj2.get("action") == "inspect" and isinstance(obj2.get("calls"), list):
                outputs = [f"### call {json.dumps(c)}\n{toolbox.run(c)}" for c in obj2["calls"][:4] if isinstance(c, dict)]
                messages += [{"role": "assistant", "content": json.dumps(obj2)[:3000]},
                             {"role": "user", "content": fence("tool results", "\n\n".join(outputs), 14000)}]
                continue
            urls = {w.url for w in toolbox.web_results}
            redo = _parse(obj2, toolbox.workdir, log_text, urls)
            if redo.hypotheses:
                redo.grounding_retry = True
                diag = redo
            break
    diag.tool_calls = list(toolbox.calls)
    diag.web_results = list(toolbox.web_results)
    if not diag.hypotheses:
        raise AgentError("diagnoser returned no usable hypotheses")
    return diag


def choose_hypothesis(diag: Diagnosis, rejected: list[dict]) -> Hypothesis | None:
    """Pick the recommended hypothesis unless it is essentially a repeat of a rejected one."""
    order = sorted(diag.hypotheses, key=lambda h: (h.id != diag.recommended, {"high": 0, "medium": 1, "low": 2}[h.model_confidence]))
    for h in order:
        repeat = any(difflib.SequenceMatcher(None, h.statement.lower(), r["statement"].lower()).ratio() > 0.85 for r in rejected)
        if not repeat:
            return h
    return None
