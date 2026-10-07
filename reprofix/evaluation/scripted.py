"""Scripted LLM backends: deterministic stand-ins for Nemotron used to test the pipeline.

IMPORTANT: results produced with these backends say NOTHING about model quality. They validate the
sandbox, verification, patching, evidence graph and benchmark harness. Reports carry
`llm_backend: "scripted"` and a caveat so they cannot be mistaken for Nemotron results.
"""
from __future__ import annotations

import json
from typing import Callable

from ..inference.base import LLMResponse


class ScriptedBackend:
    """Replies come from a per-purpose queue of strings or callables(messages) -> str."""

    name = "scripted"

    def __init__(self, script: dict[str, list[str | Callable[[list[dict]], str]]] | None = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[dict] = []

    def complete(self, *, tier: str, messages: list[dict], purpose: str, max_tokens: int = 0,
                 temperature: float = 0.0) -> LLMResponse:
        self.calls.append({"tier": tier, "purpose": purpose, "messages": messages})
        queue = self.script.get(purpose)
        if not queue:
            raise AssertionError(f"scripted backend has no reply left for purpose {purpose!r}")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        text = item(messages) if callable(item) else item
        return LLMResponse(text=text, model=f"scripted-{tier}", tier=tier, purpose=purpose)


class OracleBackend(ScriptedBackend):
    """Answers every agent call from a benchmark task's reference fix.

    A task with `stages` (the multi-fault demo) is replayed one fix edit per stage. Any other task is a single fault, so
    its whole reference fix is replayed at once, however many edits that takes (replaying only the first edit of a
    two-edit fix leaves the repository half-fixed, which is a defect of the oracle, not a result about anything).
    """

    def __init__(self, task: dict):
        super().__init__()
        self.task = task
        self.repair_calls = 0

    def _stage(self) -> dict:
        fixes = self.task["fix"]
        stages = self.task.get("stages")
        gt = self.task["ground_truth"]
        if not stages:
            return {"edits": list(fixes), "category": gt["category"], "statement": gt["description"]}
        i = min(self.repair_calls, len(fixes) - 1)
        info = stages[i] if i < len(stages) else {"category": gt["category"], "statement": gt["description"]}
        return {"edits": [fixes[i]], "category": info["category"], "statement": info["statement"]}

    def complete(self, *, tier: str, messages: list[dict], purpose: str, max_tokens: int = 0,
                 temperature: float = 0.0) -> LLMResponse:
        self.calls.append({"tier": tier, "purpose": purpose})
        t = self.task
        if purpose == "summarize":
            body = {"framework": "numpy", "task": "image classification", "model": "MLP", "dataset": "synthetic",
                    "run_command": t["command"], "expected_metric": {"name": "val_accuracy", "value": t["metric"]["expected"],
                                                                      "source": "README"}, "notes": "scripted"}
        elif purpose == "plan":
            body = {"suspect_areas": [{"area": t["ground_truth"]["category"], "why": "scripted", "check": "scripted"}],
                    "first_step": "scripted"}
        elif purpose in ("diagnose", "diagnose_behavioral"):
            st = self._stage()
            e = st["edits"][0]
            quote = next((ln.strip() for ln in e["search"].splitlines() if ln.strip()), e["search"].strip())
            body = {"action": "conclude", "summary": "scripted diagnosis", "recommended": "h1", "hypotheses": [{
                "id": "h1", "statement": st["statement"], "category": st["category"], "model_confidence": "high",
                "files": list(dict.fromkeys(x["path"] for x in st["edits"])),
                "evidence": [{"kind": "code", "file": e["path"], "quote": quote, "note": "scripted"}],
                "test": "apply the reference fix and rerun"}]}
        elif purpose == "repair":
            st = self._stage()
            self.repair_calls += 1
            body = {"edits": st["edits"], "rationale": st["statement"], "expected_effect": "failure resolved"}
        elif purpose == "review":
            body = {"summary": "scripted review", "concerns": [], "hardcoding_suspected": False}
        else:
            body = {}
        return LLMResponse(text=json.dumps(body), model=f"oracle-{tier}", tier=tier, purpose=purpose)
