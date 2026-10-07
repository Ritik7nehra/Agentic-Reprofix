"""Nebius Token Factory client (OpenAI-compatible /chat/completions) over httpx."""
from __future__ import annotations

import re
import time

import httpx

from ..config import Settings
from .base import AuthError, LLMError, LLMResponse, ModelUnavailable

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


def normalize_model_id(model_id: str) -> str:
    """Case- and punctuation-insensitive key: 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B' == 'nvidia/nvidia-nemotron-3-nano-30b-a3b'."""
    return re.sub(r"[^a-z0-9]", "", model_id.lower())


class NebiusClient:
    name = "nebius"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None,
                 timeout: float = 180.0, max_retries: int = 3, sleep=time.sleep):
        if not settings.nebius_api_key and transport is None:
            raise AuthError("NEBIUS_API_KEY is not set")
        self.settings = settings
        base = settings.nebius_base_url
        self.base_url = base if base.endswith("/") else base + "/"
        self.max_retries = max_retries
        self._sleep = sleep
        self._resolved: dict[str, str] = {}       # tier -> served spelling, learned from /models after a 404
        self._catalog: list[str] | None = None
        self._http = httpx.Client(
            transport=transport, timeout=timeout,
            headers={"Authorization": f"Bearer {settings.nebius_api_key}", "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._http.close()

    def list_models(self) -> list[str]:
        r = self._http.get(self.base_url + "models")
        if r.status_code in (401, 403):
            raise AuthError(f"Token Factory rejected the API key (HTTP {r.status_code})")
        r.raise_for_status()
        return [m.get("id", "") for m in r.json().get("data", [])]

    def _resolve_model(self, tier: str) -> str | None:
        """After a 404, look the configured ID up in /models ignoring case and punctuation. Once per tier."""
        configured = self.settings.models[tier]
        if tier in self._resolved:
            return None
        try:
            if self._catalog is None:
                self._catalog = self.list_models()
        except Exception:                           # auth/network problems surface through the normal path
            return None
        for served in self._catalog:
            if served != configured and normalize_model_id(served) == normalize_model_id(configured):
                self._resolved[tier] = served
                return served
        return None

    def complete(self, *, tier: str, messages: list[dict], purpose: str = "",
                 max_tokens: int = 4096, temperature: float = 0.2) -> LLMResponse:
        model = self._resolved.get(tier, self.settings.models[tier])
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        body.update(self.settings.extra_body)
        last_err = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                r = self._http.post(self.base_url + "chat/completions", json=body)
            except httpx.TransportError as exc:
                last_err = f"transport error: {exc}"
                if attempt < self.max_retries:
                    self._sleep(min(2 ** attempt, 20))
                    continue
                raise LLMError(last_err) from exc
            latency = time.time() - t0
            if r.status_code in (401, 403):
                raise AuthError(f"Token Factory rejected the API key (HTTP {r.status_code})")
            if r.status_code == 404 or (r.status_code == 400 and "model" in r.text.lower() and "not" in r.text.lower()):
                alt = self._resolve_model(tier)
                if alt:
                    model, body["model"] = alt, alt
                    continue
                raise ModelUnavailable(f"{model} ({tier}) not available: HTTP {r.status_code} {r.text[:200]}")
            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                try:
                    wait = float(r.headers.get("retry-after", ""))
                except ValueError:
                    wait = min(2 ** attempt, 20)
                last_err = f"HTTP {r.status_code}"
                self._sleep(min(wait, 30))
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            try:
                data = r.json()
                msg = data["choices"][0]["message"]
            except (ValueError, KeyError, IndexError) as exc:
                raise LLMError(f"unexpected response shape: {r.text[:300]}") from exc
            usage = data.get("usage") or {}
            return LLMResponse(
                text=msg.get("content") or "",
                reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
                model=data.get("model") or model, tier=tier, purpose=purpose,
                prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                latency_s=latency,
            )
        raise LLMError(f"gave up after retries: {last_err}")
