"""Patch reviewer (Nemotron Super): advisory second opinion on the final diff. Never gates verification."""
from __future__ import annotations

from ..inference import ModelRouter
from .common import call_json, fence, system_prompt

PROMPT = """
You review a patch produced by an automated repair agent for an ML repository. Look for: changes unrelated to the
stated fixes, hardcoded metric values, suppressed errors, test or data tampering, and anything that could make a
reproduced metric untrustworthy.

Return JSON: {"summary": "one sentence", "concerns": ["..."], "hardcoding_suspected": false}
"""


def review(router: ModelRouter, diff: str, issues: list[str], metric_desc: str) -> dict:
    if not diff.strip():
        return {"summary": "no changes to review", "concerns": [], "hardcoding_suspected": False}
    user = (f"Metric: {metric_desc}\nIssues the agent says it fixed:\n" + "\n".join(f"- {i}" for i in issues)
            + "\n\n" + fence("diff", diff, 12000))
    try:
        obj, _ = call_json(router, "review", [
            {"role": "system", "content": system_prompt("review", PROMPT)},
            {"role": "user", "content": user},
        ], max_tokens=2048)
    except Exception as exc:
        return {"summary": "review unavailable", "concerns": [], "hardcoding_suspected": False,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"summary": str(obj.get("summary", ""))[:300],
            "concerns": [str(c)[:300] for c in obj.get("concerns", [])][:8],
            "hardcoding_suspected": bool(obj.get("hardcoding_suspected", False))}
