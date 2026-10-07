"""A small in-memory stand-in for the parts of the GitHub REST API that ReproFix uses.

It is NOT GitHub: it encodes what the documentation says (status codes, required fields, async forks) so the
pull-request code can be tested without network access or a token. Nothing here proves behaviour against the
real service.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from reprofix.core.patching import make_diff
from reprofix.core.pullrequest import GitHubClient, blob_sha

GOOD_TOKEN = "ghp_test_token_0123456789"


@dataclass
class Repo:
    owner: str
    name: str
    files: dict[str, bytes]
    base_sha: str
    default_branch: str = "main"
    push: bool = False
    archived: bool = False
    ready_after: int = 0                      # polls of the base commit that 404 before the repo is "ready" (forks)
    branches: dict[str, dict[str, bytes]] = field(default_factory=dict)
    polls: int = 0

    @property
    def full(self) -> str:
        return f"{self.owner}/{self.name}"


class FakeGitHub:
    def __init__(self, *, user: str = "me"):
        self.user = user
        self.repos: dict[str, Repo] = {}
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict | None] = []
        self.prs: list[dict] = []
        self.fork_ready_after = 0
        self.rate_limited = False
        self.fail: dict[tuple[str, str], tuple[int, str]] = {}   # (method, path regex) -> (status, message)

    def add_repo(self, repo: Repo) -> Repo:
        self.repos[repo.full] = repo
        return repo

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # ------------------------------------------------------------------ helpers
    def calls(self) -> list[str]:
        return [f"{r.method} {r.url.path}" for r in self.requests]

    def _json(self, status: int, data, headers: dict | None = None) -> httpx.Response:
        return httpx.Response(status, json=data, headers=headers or {})

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content) if request.content else None
        self.bodies.append(body)
        path, method = request.url.path, request.method
        if request.headers.get("authorization") != f"Bearer {GOOD_TOKEN}":
            return self._json(401, {"message": "Bad credentials"})
        if self.rate_limited:
            return self._json(403, {"message": "API rate limit exceeded"}, {"x-ratelimit-remaining": "0"})
        for (m, pat), (status, msg) in self.fail.items():
            if m == method and re.fullmatch(pat, path):
                return self._json(status, {"message": msg})
        m = re.fullmatch(r"/repos/([^/]+)/([^/]+)(/.*)?", path)
        if not m:
            return self._json(404, {"message": "Not Found"})
        full, rest = f"{m.group(1)}/{m.group(2)}", m.group(3) or ""
        repo = self.repos.get(full)
        if repo is None:
            return self._json(404, {"message": "Not Found"})
        if rest == "" and method == "GET":
            return self._json(200, {"full_name": full, "default_branch": repo.default_branch, "archived": repo.archived,
                                    "permissions": {"push": repo.push, "pull": True}})
        if rest == "/forks" and method == "POST":
            fork = self.repos.setdefault(f"{self.user}/{repo.name}", Repo(
                owner=self.user, name=repo.name, files=dict(repo.files), base_sha=repo.base_sha,
                default_branch=repo.default_branch, push=True, ready_after=self.fork_ready_after))
            return self._json(202, {"full_name": fork.full, "name": fork.name, "owner": {"login": fork.owner}})
        cm = re.fullmatch(r"/git/commits/([0-9a-f]+)", rest)
        if cm and method == "GET":
            repo.polls += 1
            if cm.group(1) != repo.base_sha or repo.polls <= repo.ready_after:
                return self._json(404, {"message": "Not Found"})
            return self._json(200, {"sha": repo.base_sha})
        if rest == "/git/refs" and method == "POST":
            ref = body["ref"].removeprefix("refs/heads/")
            if ref in repo.branches:
                return self._json(422, {"message": "Reference already exists"})
            if body["sha"] != repo.base_sha:
                return self._json(422, {"message": "Object does not exist"})
            repo.branches[ref] = dict(repo.files)
            return self._json(201, {"ref": body["ref"]})
        fm = re.fullmatch(r"/contents/(.+)", rest)
        if fm:
            fpath = fm.group(1)
            branch = request.url.params.get("ref") if method == "GET" else (body or {}).get("branch")
            files = repo.branches.get(branch or "", repo.files)
            if method == "GET":
                if fpath not in files:
                    return self._json(404, {"message": "Not Found"})
                return self._json(200, {"type": "file", "path": fpath, "sha": blob_sha(files[fpath])})
            if method == "PUT":
                if branch not in repo.branches:
                    return self._json(404, {"message": "Branch not found"})
                if fpath in files and body.get("sha") != blob_sha(files[fpath]):
                    return self._json(409, {"message": f"{fpath} does not match"})
                created = fpath not in files
                files[fpath] = base64.b64decode(body["content"])
                return self._json(201 if created else 200, {"content": {"path": fpath}})
        if rest == "/pulls" and method == "POST":
            if body["head"].split(":")[-1] not in {b for r in self.repos.values() for b in r.branches}:
                return self._json(422, {"message": "Validation Failed: head invalid"})
            number = len(self.prs) + 1
            pr = {"number": number, "html_url": f"https://github.com/{full}/pull/{number}", **body}
            self.prs.append(pr)
            return self._json(201, pr)
        return self._json(404, {"message": "Not Found"})


# --------------------------------------------------------------------------- case builders
BASE_SHA = "a" * 40


def make_case(tmp_path: Path, *, source: str = "https://github.com/acme/widget", status: str = "verified", commit: str | None = BASE_SHA):
    orig, work = tmp_path / "orig", tmp_path / "work"
    orig.mkdir()
    (orig / "train.py").write_text("a = 1\nb = a // 2\nprint(b)\n", newline="")
    (orig / "requirements.txt").write_text("numpy==2.1.0\nnumpy<2\n", newline="")
    (orig / "README.md").write_text("docs\n", newline="")
    shutil.copytree(orig, work)
    (work / "train.py").write_text("a = 1\nb = a / 2\nprint(b)\n", newline="")
    (work / "requirements.txt").write_text("numpy>=1.26\n", newline="")
    (work / "helper.py").write_text("X = 1\n", newline="")
    diff, files = make_diff(orig, work, ["train.py", "requirements.txt", "helper.py"])
    report = {
        "status": status, "status_text": "Verified: the project runs and the documented metric was reproduced within tolerance.",
        "repository": {"source": source, "name": "widget", "commit": commit}, "diff": diff, "files_modified": files,
        "metric": {"name": "val_accuracy", "expected": 0.881, "tolerance": 0.03, "source": "user"},
        "results": {"baseline": {"stage": "install_failure", "metric_value": None},
                    "final": {"stage": "verified", "metric_value": 0.8815}},
        "issues": [{"category": "dependency", "statement": "requirements.txt pins numpy twice @maintainer", "files": ["requirements.txt"]},
                   {"category": "code", "statement": "floor division in train.py", "files": ["train.py"]}],
        "verification": [{"check": "Application executes", "passed": True, "detail": "exit 0"},
                         {"check": "Tests pass", "passed": None, "detail": "not run"},
                         {"check": "Expected value not hardcoded in the patch", "passed": False, "detail": "line 3"}],
        "caveats": ["LLM backend is 'scripted', NOT NVIDIA Nemotron on Nebius."], "llm_backend": "scripted",
    }
    return report, orig, work


def upstream(gh: FakeGitHub, orig: Path, *, push: bool) -> Repo:
    files = {p.relative_to(orig).as_posix(): p.read_bytes() for p in orig.rglob("*") if p.is_file()}
    return gh.add_repo(Repo(owner="acme", name="widget", files=files, base_sha=BASE_SHA, push=push))


def client(gh: FakeGitHub, sleeps: list | None = None, token: str = GOOD_TOKEN) -> GitHubClient:
    return GitHubClient(token, transport=gh.transport(), sleep=(sleeps.append if sleeps is not None else (lambda s: None)))
