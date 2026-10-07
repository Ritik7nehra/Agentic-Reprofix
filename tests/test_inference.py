import json

import httpx
import pytest

from reprofix.config import Settings
from reprofix.inference import ModelRouter, TavilyClient, UsageLog
from reprofix.inference.base import AuthError, LLMError, LLMResponse, ModelUnavailable
from reprofix.inference.client import NebiusClient
from reprofix.inference.tavily import WebEvidence, sanitize_query

OK = {"model": "served/model", "choices": [{"message": {"content": "hello", "reasoning_content": "thinking"}}],
      "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def client(handler, **kw):
    s = Settings(nebius_api_key="k-test", **kw.pop("settings", {}))
    sleeps: list[float] = []
    c = NebiusClient(s, transport=httpx.MockTransport(handler), sleep=sleeps.append, **kw)
    return c, sleeps


def msgs():
    return [{"role": "user", "content": "hi"}]


def test_request_shape_and_response_parsing():
    seen = {}

    def handler(req: httpx.Request):
        seen.update(url=str(req.url), auth=req.headers["authorization"], body=json.loads(req.content))
        return httpx.Response(200, json=OK)

    c, _ = client(handler, settings={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    r = c.complete(tier="super", messages=msgs(), purpose="plan", max_tokens=123, temperature=0.0)
    assert seen["url"].endswith("/v1/chat/completions") and seen["auth"] == "Bearer k-test"
    assert seen["body"]["model"] == "nvidia/nemotron-3-super-120b-a12b"
    assert seen["body"]["max_tokens"] == 123 and seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert (r.text, r.reasoning, r.model, r.tier, r.prompt_tokens, r.completion_tokens) == ("hello", "thinking", "served/model", "super", 11, 7)


def test_missing_usage_is_none_not_zero():
    c, _ = client(lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]}))
    r = c.complete(tier="nano", messages=msgs())
    assert r.prompt_tokens is None and r.completion_tokens is None


def test_base_url_gets_a_trailing_slash():
    s = Settings(nebius_api_key="k", nebius_base_url="https://example.test/v1")
    assert NebiusClient(s, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=OK))).base_url == "https://example.test/v1/"


def test_retries_429_and_honours_retry_after():
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "3"}) if len(calls) < 3 else httpx.Response(200, json=OK)

    c, sleeps = client(handler)
    assert c.complete(tier="super", messages=msgs()).text == "hello"
    assert len(calls) == 3 and sleeps == [3.0, 3.0]


def test_backoff_without_retry_after_and_give_up():
    c, sleeps = client(lambda req: httpx.Response(503, text="busy"), max_retries=2)
    with pytest.raises(LLMError, match="503"):
        c.complete(tier="super", messages=msgs())
    assert sleeps == [1, 2]                                   # exponential, and no sleep after the last attempt


def test_transport_errors_are_retried_then_raised():
    n = []

    def handler(req):
        n.append(1)
        raise httpx.ConnectError("boom")

    c, sleeps = client(handler, max_retries=2)
    with pytest.raises(LLMError, match="transport error"):
        c.complete(tier="super", messages=msgs())
    assert len(n) == 3 and len(sleeps) == 2


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_not_retried(status):
    n = []
    c, sleeps = client(lambda req: (n.append(1), httpx.Response(status, text="no"))[1])
    with pytest.raises(AuthError):
        c.complete(tier="super", messages=msgs())
    assert len(n) == 1 and sleeps == []


@pytest.mark.parametrize("resp", [
    httpx.Response(404, text="not found"),
    httpx.Response(400, json={"error": {"message": "The model `x` does not exist or is not available"}}),
])
def test_unknown_model_is_reported_as_model_unavailable(resp):
    c, _ = client(lambda req: resp)
    with pytest.raises(ModelUnavailable):
        c.complete(tier="ultra", messages=msgs())


def test_other_client_errors_and_bad_payloads_raise_llm_error_without_retry():
    n = []
    c, _ = client(lambda req: (n.append(1), httpx.Response(400, text="bad temperature"))[1])
    with pytest.raises(LLMError, match="HTTP 400"):
        c.complete(tier="super", messages=msgs())
    assert len(n) == 1
    c2, _ = client(lambda req: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(LLMError, match="unexpected response"):
        c2.complete(tier="super", messages=msgs())


def test_no_key_means_no_client():
    with pytest.raises(AuthError):
        NebiusClient(Settings(nebius_api_key=""))


def test_list_models_and_auth_failure():
    c, _ = client(lambda req: httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]}))
    assert c.list_models() == ["a", "b"]
    bad, _ = client(lambda req: httpx.Response(401))
    with pytest.raises(AuthError):
        bad.list_models()


# --------------------------------------------------------------------------- router
class Fake:
    name = "fake"

    def __init__(self, unavailable=()):
        self.unavailable = set(unavailable)
        self.calls = []

    def complete(self, *, tier, messages, purpose, max_tokens, temperature):
        self.calls.append(tier)
        if tier in self.unavailable:
            raise ModelUnavailable(tier)
        return LLMResponse(text="{}", model=f"m-{tier}", tier=tier, purpose=purpose, prompt_tokens=10, completion_tokens=5)


def router(mode="router", unavailable=(), prices=None):
    b = Fake(unavailable)
    return ModelRouter(b, UsageLog(prices=prices or Settings().prices), mode), b


@pytest.mark.parametrize("purpose,tier", [
    ("summarize", "nano"), ("digest", "nano"), ("classify", "nano"), ("plan", "super"), ("diagnose", "super"),
    ("diagnose_behavioral", "ultra"), ("repair", "super"), ("review", "super"), ("something-new", "super"),
])
def test_default_policy(purpose, tier):
    assert router()[0].pick(purpose) == tier


def test_escalation_moves_up_and_is_capped():
    r, _ = router()
    assert [r.pick("summarize", e) for e in (0, 1, 2, 5)] == ["nano", "super", "ultra", "ultra"]
    assert r.pick("diagnose", 1) == "ultra" and r.pick("diagnose", -3) == "super"


def test_fixed_baseline_modes_ignore_purpose_and_escalation():
    assert {router("super-only")[0].pick(p, e) for p in ("summarize", "diagnose_behavioral") for e in (0, 2)} == {"super"}
    assert {router("ultra-only")[0].pick(p, e) for p in ("summarize", "plan") for e in (0, 2)} == {"ultra"}


def test_fallback_steps_down_first_and_remembers_the_outage():
    r, b = router(unavailable={"ultra"})
    resp = r.complete("diagnose_behavioral", msgs())
    assert resp.tier == "super" and resp.requested_tier == "ultra" and resp.fell_back
    assert b.calls == ["ultra", "super"]
    r.complete("diagnose_behavioral", msgs())
    assert b.calls == ["ultra", "super", "super"]             # ultra is not retried once known to be missing
    assert r.decisions[-1] == {"purpose": "diagnose_behavioral", "wanted": "ultra", "used": "super", "model": "m-super"}


def test_fallback_steps_up_when_nothing_cheaper_exists():
    r, _ = router(unavailable={"nano"})
    assert r.complete("summarize", msgs()).tier == "super"


def test_every_tier_unavailable_is_an_error():
    r, _ = router(unavailable={"nano", "super", "ultra"})
    with pytest.raises(ModelUnavailable, match="no Nemotron tier"):
        r.complete("plan", msgs())


def test_usage_log_costs_use_the_configured_prices_per_tier():
    r, _ = router()
    for p in ("summarize", "plan", "diagnose_behavioral"):
        r.complete(p, msgs())
    u = r.usage.summary()
    assert u["calls"] == 3 and u["has_unpriced_tiers"] is False and u["by_purpose"]["plan"] == 1
    assert u["by_tier"]["nano"]["cost_usd"] == pytest.approx(10 / 1e6 * 0.06 + 5 / 1e6 * 0.24)
    assert u["by_tier"]["super"]["cost_usd"] == pytest.approx(10 / 1e6 * 0.30 + 5 / 1e6 * 0.90)
    assert u["by_tier"]["ultra"]["cost_usd"] == pytest.approx(10 / 1e6 * 1.00 + 5 / 1e6 * 3.00)
    assert u["cost_usd_priced_tiers"] == pytest.approx(sum(t["cost_usd"] for t in u["by_tier"].values()))


def test_a_tier_without_a_price_reports_n_a_instead_of_zero():
    r, _ = router(prices={**Settings().prices, "nano": None})
    for p in ("summarize", "plan"):
        r.complete(p, msgs())
    u = r.usage.summary()
    assert u["by_tier"]["nano"]["cost_usd"] is None and u["by_tier"]["nano"]["priced"] is False and u["has_unpriced_tiers"] is True
    assert u["cost_usd_priced_tiers"] == pytest.approx(u["by_tier"]["super"]["cost_usd"])      # only the priced tier is summed


def test_usage_log_reports_cost_as_unknown_when_no_call_reported_usage():
    log = UsageLog(prices=Settings().prices)
    log.add(LLMResponse(text="", model="m", tier="super"))
    u = log.summary()
    assert u["by_tier"]["super"]["cost_usd"] is None and u["by_tier"]["super"]["calls_without_usage"] == 1
    assert u["cost_usd_priced_tiers"] is None and log.total_tokens == 0


# --------------------------------------------------------------------------- tavily
@pytest.mark.parametrize("raw,must_have,must_not", [
    ("ImportError in /home/alice/work/secret_project/train.py", "train.py", "alice"),
    ("fail C:\\Users\\bob\\proj\\model.py", "model.py", "bob"),
    ("token " + "a" * 40 + " leaked", "leaked", "a" * 40),
    ("Authorization: Bearer sk-abc123 failed", "failed", "sk-abc123"),
])
def test_sanitize_query_removes_private_details(raw, must_have, must_not):
    out = sanitize_query(raw)
    assert must_have in out and must_not not in out


def test_sanitize_query_is_length_capped():
    assert len(sanitize_query("word " * 500)) <= 300


def tavily(handler, key="tvly-test"):
    return TavilyClient(key, transport=httpx.MockTransport(handler))


def test_tavily_request_and_parsing():
    seen = {}

    def handler(req):
        seen.update(body=json.loads(req.content), auth=req.headers["authorization"], url=str(req.url))
        return httpx.Response(200, json={"results": [{"title": "T", "url": "https://x.dev/a", "content": "c" * 900, "score": 0.9}]})

    t = tavily(handler)
    out = t.search("ResolutionImpossible in /srv/app/requirements.txt", max_results=50, include_domains=["pypi.org"])
    assert seen["url"] == "https://api.tavily.com/search" and seen["auth"] == "Bearer tvly-test"
    assert seen["body"]["max_results"] == 8 and seen["body"]["include_domains"] == ["pypi.org"]
    assert "/srv/app" not in seen["body"]["query"]                       # sanitized before leaving the machine
    assert out == [WebEvidence(title="T", url="https://x.dev/a", snippet="c" * 500, score=0.9)] and t.calls == 1


def test_tavily_without_key_or_query_never_calls_out():
    n = []
    t = tavily(lambda req: (n.append(1), httpx.Response(200, json={}))[1], key="")
    assert t.search("anything") == [] and not t.available and n == [] and t.calls == 0
    assert tavily(lambda req: httpx.Response(200, json={})).search("   ") == []


@pytest.mark.parametrize("resp", [httpx.Response(500, text="oops"), httpx.Response(200, text="not json")])
def test_tavily_failures_are_recorded_not_raised(resp):
    t = tavily(lambda req: resp)
    assert t.search("CUDA mismatch") == [] and t.calls == 1 and len(t.errors) == 1


# --------------------------------------------------------------------------- model-id spelling
NANO_CATALOG_SPELLING = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def spelling_server(served=(NANO_CATALOG_SPELLING,)):
    seen = {"chat": [], "models": 0}

    def handler(req: httpx.Request):
        if req.url.path.endswith("/models"):
            seen["models"] += 1
            return httpx.Response(200, json={"data": [{"id": m} for m in served]})
        model = json.loads(req.content)["model"]
        seen["chat"].append(model)
        return httpx.Response(200, json=OK) if model in served else httpx.Response(404, text="model not found")

    return handler, seen


def test_a_differently_cased_model_id_is_resolved_against_the_models_list_once():
    handler, seen = spelling_server()
    c, _ = client(handler)
    assert c.complete(tier="nano", messages=msgs()).text == "hello"
    assert c.complete(tier="nano", messages=msgs()).text == "hello"
    assert seen["chat"] == ["nvidia/nvidia-nemotron-3-nano-30b-a3b", NANO_CATALOG_SPELLING, NANO_CATALOG_SPELLING]
    assert seen["models"] == 1                                   # the catalog is fetched once, the answer is remembered


def test_resolution_only_accepts_the_same_id_modulo_case_and_punctuation():
    handler, seen = spelling_server(served=("nvidia/Nemotron-3_5-Lightning", "meta/other-model"))
    c, _ = client(handler)
    with pytest.raises(ModelUnavailable):                        # a *different* model is never substituted silently
        c.complete(tier="nano", messages=msgs())
    assert seen["chat"] == ["nvidia/nvidia-nemotron-3-nano-30b-a3b"]


def test_resolution_failure_paths_still_raise_model_unavailable():
    def handler(req):
        return httpx.Response(401) if req.url.path.endswith("/models") else httpx.Response(404, text="model not found")

    c, _ = client(handler)
    with pytest.raises(ModelUnavailable):
        c.complete(tier="super", messages=msgs())


def test_normalize_model_id():
    from reprofix.inference.client import normalize_model_id
    assert normalize_model_id("nvidia/Nemotron-3_5-Lightning") == normalize_model_id("NVIDIA/nemotron-3.5-lightning")
    assert normalize_model_id("a/b-1") != normalize_model_id("a/b-2")
