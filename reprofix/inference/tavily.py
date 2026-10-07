"""Tavily web search for external evidence (REST: POST https://api.tavily.com/search)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

import httpx

TAVILY_URL = "https://api.tavily.com/search"

# Error text that is unlikely to be answerable from the repository alone.
EXTERNAL_HINTS = re.compile(
    r"CUDA|CUBLAS|cuDNN|NCCL|ResolutionImpossible|No matching distribution|undefined symbol|"
    r"DLL load failed|incompatible|conflicting dependencies|Segmentation fault|GLIBC",
    re.I,
)


@dataclass
class WebEvidence:
    title: str
    url: str
    snippet: str
    score: float | None = None

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "score": self.score}


def sanitize_query(q: str) -> str:
    """Strip anything private before it leaves the machine: paths -> basenames, long tokens dropped."""
    q = re.sub(r"(?:/[\w.\-]+){2,}", lambda m: PurePosixPath(m.group(0)).name, q)
    q = re.sub(r"[A-Za-z]:\\(?:[^\\\s]+\\)*([^\\\s]+)", r"\1", q)
    q = re.sub(r"\b[A-Za-z0-9_\-]{32,}\b", "", q)
    q = re.sub(r"(?i)bearer\s+\S+", "", q)
    return re.sub(r"\s+", " ", q).strip()[:300]


class TavilyClient:
    def __init__(self, api_key: str, transport: httpx.BaseTransport | None = None, timeout: float = 30.0):
        self.api_key = api_key
        self.calls = 0
        self.errors: list[str] = []
        self._http = httpx.Client(transport=transport, timeout=timeout)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5, search_depth: str = "basic",
               include_domains: list[str] | None = None) -> list[WebEvidence]:
        query = sanitize_query(query)
        if not query or not self.available:
            return []
        body: dict = {"query": query, "search_depth": search_depth, "max_results": max(1, min(max_results, 8)),
                      "include_answer": False}
        if include_domains:
            body["include_domains"] = include_domains
        self.calls += 1
        try:
            r = self._http.post(TAVILY_URL, json=body, headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            results = r.json().get("results", [])
        except (httpx.HTTPError, ValueError) as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return []
        out = []
        for item in results:
            out.append(WebEvidence(
                title=str(item.get("title", ""))[:200], url=str(item.get("url", "")),
                snippet=str(item.get("content", ""))[:500], score=item.get("score"),
            ))
        return out
