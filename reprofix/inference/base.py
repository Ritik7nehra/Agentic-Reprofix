"""LLM backend interface + usage accounting (measured numbers only)."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Protocol

from ..config import TIERS


class ModelUnavailable(RuntimeError):
    """The requested model ID is not served (404 / model-not-found)."""


class AuthError(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    model: str
    tier: str
    purpose: str = ""
    reasoning: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    requested_tier: str | None = None  # set when a fallback changed the tier

    @property
    def fell_back(self) -> bool:
        return self.requested_tier is not None and self.requested_tier != self.tier


class ChatBackend(Protocol):
    name: str  # "nebius" | "scripted"

    def complete(
        self, *, tier: str, messages: list[dict], purpose: str, max_tokens: int, temperature: float
    ) -> LLMResponse: ...


@dataclass
class UsageLog:
    """Thread-safe record of every LLM call. Costs are computed only where a price is known."""

    prices: dict[str, tuple[float, float] | None]
    calls: list[LLMResponse] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, resp: LLMResponse) -> None:
        with self._lock:
            self.calls.append(resp)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum((c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in self.calls)

    def summary(self) -> dict:
        with self._lock:
            calls = list(self.calls)
        by_tier: dict[str, dict] = {}
        for t in TIERS:
            tc = [c for c in calls if c.tier == t]
            if not tc:
                continue
            pt = sum(c.prompt_tokens or 0 for c in tc)
            ct = sum(c.completion_tokens or 0 for c in tc)
            unknown = sum(1 for c in tc if c.prompt_tokens is None or c.completion_tokens is None)
            price = self.prices.get(t)
            # Unknown usage is not zero usage: with no usage data at all, cost is n/a, not $0.
            cost = None if (price is None or unknown == len(tc)) else round(pt / 1e6 * price[0] + ct / 1e6 * price[1], 8)
            by_tier[t] = {
                "calls": len(tc), "prompt_tokens": pt, "completion_tokens": ct,
                "latency_s": round(sum(c.latency_s for c in tc), 3),
                "calls_without_usage": unknown,
                "cost_usd": cost, "priced": price is not None,
            }
        priced = [v["cost_usd"] for v in by_tier.values() if v["cost_usd"] is not None]
        return {
            "calls": len(calls),
            "prompt_tokens": sum(v["prompt_tokens"] for v in by_tier.values()),
            "completion_tokens": sum(v["completion_tokens"] for v in by_tier.values()),
            "latency_s": round(sum(v["latency_s"] for v in by_tier.values()), 3),
            "cost_usd_priced_tiers": round(sum(priced), 8) if priced else None,
            "has_unpriced_tiers": any(not v["priced"] for v in by_tier.values()),
            "fallbacks": sum(1 for c in calls if c.fell_back),
            "by_tier": by_tier,
            "by_purpose": _count_by(calls, "purpose"),
        }


def _count_by(calls: list[LLMResponse], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in calls:
        key = getattr(c, attr) or "unknown"
        out[key] = out.get(key, 0) + 1
    return out
