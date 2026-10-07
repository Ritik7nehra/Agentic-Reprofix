"""Tiered model routing across Nemotron Nano / Super / Ultra.

Policy v1 (a hypothesis to be measured with `reprofix bench run --router-mode router|super-only|ultra-only`
and `reprofix bench compare`, not a claim of optimality):
  * nano  : cheap, high-volume work -- repo summarisation, log digestion, classification
  * super : default for planning, diagnosis of crashes, code repair, patch review
  * ultra : diagnosis when there is no traceback (wrong-result bugs) and escalation after a
            hypothesis was rejected by experiment
Modes `super-only` / `ultra-only` exist so the router can be compared with fixed baselines.
"""
from __future__ import annotations

from ..config import TIERS
from ..models import RouterMode
from .base import ChatBackend, LLMResponse, ModelUnavailable, UsageLog

DEFAULT_TIER = {
    "summarize": "nano",
    "digest": "nano",
    "classify": "nano",
    "plan": "super",
    "diagnose": "super",
    "diagnose_behavioral": "ultra",
    "repair": "super",
    "review": "super",
}


class ModelRouter:
    def __init__(self, backend: ChatBackend, usage: UsageLog, mode: RouterMode = "router"):
        self.backend = backend
        self.usage = usage
        self.mode = mode
        self.unavailable: set[str] = set()
        self.decisions: list[dict] = []

    def pick(self, purpose: str, escalate: int = 0) -> str:
        if self.mode == "ultra-only":
            return "ultra"
        if self.mode == "super-only":
            return "super"
        base = DEFAULT_TIER.get(purpose, "super")
        idx = min(TIERS.index(base) + max(escalate, 0), len(TIERS) - 1)
        return TIERS[idx]

    def _chain(self, tier: str) -> list[str]:
        i = TIERS.index(tier)
        # prefer stepping down (cheaper) from the requested tier, then up as a last resort
        order = [tier] + list(reversed(TIERS[:i])) + list(TIERS[i + 1 :])
        return [t for t in order if t not in self.unavailable]

    def complete(self, purpose: str, messages: list[dict], *, escalate: int = 0,
                 max_tokens: int = 4096, temperature: float = 0.2) -> LLMResponse:
        wanted = self.pick(purpose, escalate)
        last_exc: Exception | None = None
        for tier in self._chain(wanted):
            try:
                resp = self.backend.complete(tier=tier, messages=messages, purpose=purpose,
                                             max_tokens=max_tokens, temperature=temperature)
            except ModelUnavailable as exc:
                self.unavailable.add(tier)
                last_exc = exc
                continue
            if tier != wanted:
                resp.requested_tier = wanted
            self.usage.add(resp)
            self.decisions.append({"purpose": purpose, "wanted": wanted, "used": tier, "model": resp.model})
            return resp
        raise ModelUnavailable(f"no Nemotron tier is available for '{purpose}': {last_exc}")
