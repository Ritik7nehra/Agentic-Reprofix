"""Planner (Nemotron Super): rank where to look first. Advisory -- the pipeline does not depend on it."""
from __future__ import annotations

from ..inference import ModelRouter
from .common import call_json, fence, system_prompt

PROMPT = """
You plan the investigation for ReproFix. Given a project summary and the user's goal, list the most likely
areas that could make this ML repository fail or miss its documented result, ranked by likelihood, and say
what to check first. Be specific to this project. Do not invent facts about the code.

Return JSON:
{"suspect_areas": [{"area": "dependency|architecture|data|preprocessing|training|evaluation|configuration|checkpoint|code|environment",
                    "why": "one sentence", "check": "what to inspect or run"}],
 "first_step": "one sentence"}
"""


def plan(router: ModelRouter, analysis: dict, goal: str, metric_desc: str) -> dict:
    user = (
        f"Goal: {goal}\nMetric to reproduce: {metric_desc}\n\n"
        + fence("analysis", str({k: v for k, v in analysis.items() if k != "llm_error"}), 4000)
    )
    try:
        obj, _ = call_json(router, "plan", [
            {"role": "system", "content": system_prompt("plan", PROMPT)},
            {"role": "user", "content": user},
        ], max_tokens=2048)
    except Exception as exc:
        return {"suspect_areas": [], "first_step": "", "error": f"{type(exc).__name__}: {exc}"}
    areas = [a for a in obj.get("suspect_areas", []) if isinstance(a, dict)][:6]
    return {"suspect_areas": areas, "first_step": str(obj.get("first_step", ""))[:300]}
