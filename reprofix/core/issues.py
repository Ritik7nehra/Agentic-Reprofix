"""Known issues: has somebody already reported this failure?

When a run is not already passing, ReproFix searches (1) the GitHub issues of the repository it was given and (2) the web
(Tavily, limited to a short list of code-help sites) for the failure it just observed. The results are

  * shown in the report as "possibly related, not verified", and
  * handed to the diagnoser as UNTRUSTED evidence, fenced like any other repository or web text.

What this module does not do: decide that an issue describes this failure (the ranking is word overlap and says nothing about
causes), read an issue's comments, follow links, or use a GitHub token (unauthenticated search is rate limited to about ten
requests a minute; when GitHub says stop, the search stops for the rest of the run). The only text that leaves the machine is a
query made of identifiers taken from the failure, with paths, long tokens and search operators removed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from ..inference.tavily import TavilyClient, WebEvidence, sanitize_query
from ..models import MetricSpec, Observation

GITHUB_API = "https://api.github.com"
GITHUB_HOST = "api.github.com"
WEB_DOMAINS = ["github.com", "stackoverflow.com", "discuss.pytorch.org", "discuss.huggingface.co", "forums.developer.nvidia.com"]
MAX_RESULTS = 5
MAX_QUERY_TERMS = 8
SNIPPET_CHARS = 300

_REPO_URL = re.compile(r"^https://github\.com/([\w.\-]+)/([\w.\-]+?)(?:\.git)?/?$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP = {"the", "and", "for", "not", "with", "from", "that", "this", "was", "are", "has", "have", "found", "line", "file", "error",
         "errors", "exception", "traceback", "most", "recent", "call", "last", "named", "got", "expected", "but", "none", "true",
         "false", "can", "cannot", "could", "should", "would", "but", "you", "your", "its", "all", "any", "one", "more", "than", "use",
         "using", "run", "running", "failed", "failure", "pip", "install", "installing", "python"}


# --------------------------------------------------------------------------- query and ranking
_SECRET_PREFIX = ("akia", "asia", "ghp_", "gho_", "ghs_", "ghu_", "github_pat_", "sk_", "sk-", "xox", "aiza", "hf_", "nvapi", "tvly")


def _looks_secret(word: str) -> bool:
    """A token that could be a key or password: a known key prefix, or a long mix of letters and digits. Never put in a query."""
    low = word.lower()
    if low.startswith(_SECRET_PREFIX):
        return True
    return len(word) >= 16 and any(c.isdigit() for c in word) and any(c.isalpha() for c in word)


def query_terms(obs: Observation, metric: MetricSpec | None = None) -> list[str]:
    """Identifiers to search for, in order of usefulness. Only [A-Za-z0-9_] words survive: no quotes, no `repo:`/`user:`
    qualifiers, no paths (a search term built from untrusted logs must not be able to change what is searched)."""
    raw: list[str] = []
    if obs.exception_type:
        raw.append(obs.exception_type.split(".")[-1])
    text = sanitize_query(" ".join(x for x in (obs.exception_message or "", obs.summary if obs.kind != "metric_gap" else "") if x))
    raw += _IDENT.findall(text)
    if obs.kind == "metric_gap":
        raw += [metric.name if metric else "accuracy", "reproduce", "lower"]
        raw = [w for part in raw for w in re.split(r"[^A-Za-z0-9]+", part) if w]
    if obs.kind == "install_failure":
        raw.append("requirements")
    seen: set[str] = set()
    out: list[str] = []
    for w in raw:
        w = re.sub(r"[^A-Za-z0-9_]", "", w)
        key = w.lower()
        if len(w) < 3 or key in _STOP or key in seen or key.isdigit() or _looks_secret(w):
            continue
        seen.add(key)
        out.append(w)
        if len(out) >= MAX_QUERY_TERMS:
            break
    return out


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) >= 3}


def rank(terms: list[str], title: str, snippet: str, exc_type: str | None) -> tuple[float, list[str]]:
    """(overlap score in 0..1, the query terms found in the title or snippet). Deterministic; says nothing about causes."""
    if not terms:
        return 0.0, []
    have = _words(title + " " + snippet)
    matched = [t for t in terms if t.lower() in have]
    score = len(matched) / len(terms)
    if exc_type and exc_type.split(".")[-1].lower() in have:
        score = min(1.0, score + 0.25)
    return round(score, 3), matched


def relevant(matched: list[str], terms: list[str], exc_type: str | None) -> bool:
    """Keep a result only when it shares at least two of the query's words, or the exception's name."""
    if len(matched) >= 2:
        return True
    return bool(exc_type) and exc_type.split(".")[-1].lower() in {m.lower() for m in matched}


# --------------------------------------------------------------------------- data
@dataclass
class Issue:
    title: str
    url: str
    source: str                      # "github-issues" | "web"
    snippet: str = ""
    state: str | None = None         # GitHub: open | closed
    comments: int | None = None
    updated: str | None = None
    score: float = 0.0
    matched: list[str] = field(default_factory=list)
    query: str = ""

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "source": self.source, "snippet": self.snippet, "state": self.state,
                "comments": self.comments, "updated": self.updated, "score": self.score, "matched_terms": self.matched,
                "query": self.query}


@dataclass
class IssueSearch:
    """What one search round found."""

    query: str = ""
    results: list[Issue] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)       # which sources were actually asked
    cached: bool = False                                     # the same query was answered earlier in this run

    def as_context(self) -> str:
        if not self.results:
            return ""
        return "\n\n".join(
            f"[{i + 1}] {r.title}\nURL: {r.url}\nsource: {r.source}" + (f", {r.state}" if r.state else "")
            + (f", {r.comments} comments" if r.comments is not None else "") + f"\n{r.snippet}"
            for i, r in enumerate(self.results))

    def as_web_evidence(self) -> list[WebEvidence]:
        return [WebEvidence(title=r.title, url=r.url, snippet=r.snippet, score=r.score) for r in self.results]


# --------------------------------------------------------------------------- finder
def github_slug(repo_url: str | None) -> tuple[str, str] | None:
    m = _REPO_URL.match(repo_url or "")
    return (m.group(1), m.group(2)) if m else None


class IssueFinder:
    def __init__(self, tavily: TavilyClient | None = None, *, transport: httpx.BaseTransport | None = None,
                 timeout: float = 15.0, github: bool = True, max_results: int = MAX_RESULTS):
        self.tavily = tavily
        self.github = github
        self.max_results = max_results
        self.errors: list[str] = []
        self.github_calls = 0
        self._github_stopped = False
        self._cache: dict[str, IssueSearch] = {}
        # no redirects: a response must come from api.github.com itself
        self._http = httpx.Client(transport=transport, timeout=timeout, follow_redirects=False, headers={
            "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "reprofix-known-issues"})

    def close(self) -> None:
        self._http.close()

    def sources_for(self, repo_url: str | None) -> list[str]:
        out = []
        if self.github and not self._github_stopped and github_slug(repo_url):
            out.append("github-issues")
        if self.tavily and self.tavily.available:
            out.append("web")
        return out

    def find(self, *, repo_url: str | None, observation: Observation, metric: MetricSpec | None = None) -> IssueSearch:
        terms = query_terms(observation, metric)
        res = IssueSearch(query=" ".join(terms))
        if len(terms) < 2:
            return res                                         # one bare word finds noise, not issues
        if res.query in self._cache:                           # same failure as an earlier round: no new request
            c = self._cache[res.query]
            return IssueSearch(query=c.query, results=c.results, cached=True)
        found: list[Issue] = []
        slug = github_slug(repo_url)
        if self.github and slug and not self._github_stopped:
            res.searched.append("github-issues")
            found += self._github(slug, terms, res)
        if self.tavily and self.tavily.available:
            res.searched.append("web")
            found += self._web(terms, res)
        exc = observation.exception_type
        keep: dict[str, Issue] = {}
        for it in found:
            it.score, it.matched = rank(terms, it.title, it.snippet, exc)
            it.query = res.query
            if not relevant(it.matched, terms, exc):
                continue
            if it.url not in keep or keep[it.url].score < it.score:
                keep[it.url] = it
        ordered = sorted(keep.values(), key=lambda i: (-i.score, i.source != "github-issues", i.url))
        res.results = ordered[: self.max_results]
        if not res.errors:                                      # a failed search is worth asking again next round
            self._cache[res.query] = res
        return res

    # ------------------------------------------------------------------ GitHub issues of the repository itself
    def _github(self, slug: tuple[str, str], terms: list[str], res: IssueSearch) -> list[Issue]:
        owner, repo = slug
        q = f"{' '.join(terms)} repo:{owner}/{repo} is:issue"
        self.github_calls += 1
        try:
            r = self._http.get(f"{GITHUB_API}/search/issues", params={"q": q, "per_page": self.max_results * 2})
        except httpx.HTTPError as exc:
            self._note(res, f"GitHub issue search failed: {type(exc).__name__}")
            return []
        if r.status_code in (403, 429):
            self._github_stopped = True
            self._note(res, "GitHub refused the search (rate limit: unauthenticated search allows about 10 requests a minute); "
                            "issue search is skipped for the rest of this run")
            return []
        if r.status_code != 200:
            self._note(res, f"GitHub issue search returned HTTP {r.status_code}")
            return []
        try:
            data = r.json()
        except ValueError:
            self._note(res, "GitHub issue search returned something that is not JSON")
            return []
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            self._note(res, "GitHub issue search returned a response in an unexpected shape")
            return []
        out: list[Issue] = []
        for it in items:
            if not isinstance(it, dict) or "pull_request" in it:
                continue
            url = str(it.get("html_url", ""))
            if urlparse(url).hostname != "github.com":
                continue
            out.append(Issue(title=str(it.get("title", ""))[:200], url=url, source="github-issues",
                             snippet=re.sub(r"\s+", " ", str(it.get("body") or ""))[:SNIPPET_CHARS],
                             state=it.get("state") if it.get("state") in ("open", "closed") else None,
                             comments=it.get("comments") if isinstance(it.get("comments"), int) else None,
                             updated=str(it.get("updated_at") or "")[:10] or None))
        return out

    # ------------------------------------------------------------------ the web, restricted to code-help sites
    def _web(self, terms: list[str], res: IssueSearch) -> list[Issue]:
        assert self.tavily is not None
        before = len(self.tavily.errors)
        results = self.tavily.search(" ".join(terms), max_results=self.max_results, include_domains=WEB_DOMAINS)
        if len(self.tavily.errors) > before:
            self._note(res, "web search failed: " + self.tavily.errors[-1][:120])
        out = []
        for w in results:
            if urlparse(w.url).scheme not in ("http", "https"):
                continue
            out.append(Issue(title=w.title, url=w.url, source="web", snippet=re.sub(r"\s+", " ", w.snippet)[:SNIPPET_CHARS]))
        return out

    def _note(self, res: IssueSearch, msg: str) -> None:
        res.errors.append(msg)
        self.errors.append(msg)


# --------------------------------------------------------------------------- accumulation across attempts, for the report
class IssueLog:
    """Merges the rounds of one run (the same issue can turn up for several failures) and keeps what was asked."""

    def __init__(self) -> None:
        self.rounds: list[dict] = []
        self.by_url: dict[str, Issue] = {}
        self.errors: list[str] = []

    def add(self, search: IssueSearch, attempt: int) -> None:
        self.rounds.append({"attempt": attempt, "query": search.query, "searched": search.searched, "found": len(search.results)})
        for e in search.errors:
            if e not in self.errors:
                self.errors.append(e)
        for r in search.results:
            if r.url not in self.by_url or self.by_url[r.url].score < r.score:
                self.by_url[r.url] = r

    def to_dict(self, enabled: bool) -> dict:
        results = sorted(self.by_url.values(), key=lambda i: (-i.score, i.url))
        return {"enabled": enabled, "rounds": self.rounds, "results": [r.to_dict() for r in results[:8]], "errors": self.errors,
                "note": "Possibly related, not verified: found by searching for words from the failure. ReproFix does not "
                        "check that any of them describes this failure."}
