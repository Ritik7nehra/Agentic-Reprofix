"""Open a GitHub pull request from a finished run.

Two steps, kept apart on purpose:

  * `plan_from_report` is pure and offline: it rebuilds the new file contents by applying the stored diff to
    the original snapshot (so the pull request contains exactly what the user reviewed, not whatever the
    repository's own code may have written into the workspace) and drafts a title and body from the report.
  * `open_pull_request` talks to the GitHub REST API with a token the caller supplies. It is only ever called
    after an explicit confirmation, and the token is never stored, logged or echoed.

Flow: confirm the base commit exists, branch from it (in a fork when the token has no push access), commit each
changed file, open the pull request. Contents API commits are serial, one per file.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx

GITHUB_API = "https://api.github.com"
# GitHub versions its REST API by date and supports each version for a long time after a newer one ships.
GITHUB_API_VERSION = "2022-11-28"
ALLOWED_STATUSES = {"verified", "executes", "partial"}
MAX_TITLE = 120
MAX_BODY = 20_000
MAX_FILE_BYTES = 1_000_000


class PullRequestError(RuntimeError):
    """kind: 'unavailable' (nothing to open a PR for), 'invalid' (bad input), 'github' (the API said no), 'conflict'."""

    def __init__(self, message: str, kind: str = "invalid"):
        super().__init__(message)
        self.kind = kind


# ----------------------------------------------------------------------------- plan (offline)
@dataclass
class ChangedFile:
    path: str
    content: bytes
    is_new: bool
    orig_blob_sha: str | None  # git blob SHA-1 of the original bytes, to confirm GitHub holds the same file

    def to_dict(self) -> dict:
        return {"path": self.path, "status": "added" if self.is_new else "modified", "bytes": len(self.content)}


@dataclass
class PullRequestPlan:
    owner: str
    repo: str
    base_sha: str
    run_id: str
    status: str
    title: str
    body: str
    files: list[ChangedFile] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    def to_dict(self) -> dict:
        return {"target": self.full_name, "base_commit": self.base_sha, "status": self.status, "title": self.title,
                "body": self.body, "files": [f.to_dict() for f in self.files]}


def parse_github_repo(url: str) -> tuple[str, str]:
    p = urlparse((url or "").strip())
    if p.scheme != "https" or (p.hostname or "").lower() != "github.com":
        raise PullRequestError("pull requests can only be opened against repositories on github.com", "unavailable")
    m = re.fullmatch(r"/([\w.\-]+)/([\w.\-]+?)(?:\.git)?/?", p.path)
    if not m or m.group(1) in {".", ".."} or m.group(2) in {".", ".."}:
        raise PullRequestError("the repository URL does not look like https://github.com/<owner>/<repo>", "unavailable")
    return m.group(1), m.group(2)


def blob_sha(data: bytes) -> str:
    """The SHA-1 git assigns to a file's contents."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


_DIFF_HEAD = re.compile(r"^diff --git a/(.+) b/\1$", re.M)


def _safe_rel(path: str) -> str:
    p = path.replace("\\", "/")
    if not p or p.startswith("/") or any(part in {"", ".", ".."} for part in p.split("/")) or p.split("/")[0] in {".git", ".deps", ".home"}:
        raise PullRequestError(f"refusing unsafe path in patch: {path!r}")
    if p.startswith(".github/"):
        raise PullRequestError("the patch edits .github/ (CI configuration); ReproFix will not open a pull request that does")
    return p


def files_from_diff(orig_dir: Path, diff: str) -> list[ChangedFile]:
    """Apply `diff` to a scratch copy of the touched original files and return their new contents."""
    heads = list(_DIFF_HEAD.finditer(diff))
    if not heads:
        raise PullRequestError("the run produced no patch", "unavailable")
    blocks = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(diff)
        blocks.append((_safe_rel(m.group(1)), diff[m.start():end]))
    changed: list[ChangedFile] = []
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        originals: dict[str, bytes | None] = {}
        for rel, block in blocks:
            if "deleted file mode" in block:
                raise PullRequestError(f"{rel}: deleting files is not supported")
            src = orig_dir / rel
            is_new = "new file mode" in block
            if is_new and src.exists():
                raise PullRequestError(f"{rel}: the patch creates a file that already exists in the snapshot")
            if not is_new:
                if not src.is_file():
                    raise PullRequestError(f"{rel}: the original snapshot no longer has this file")
                originals[rel] = src.read_bytes()
                (scratch / rel).parent.mkdir(parents=True, exist_ok=True)
                (scratch / rel).write_bytes(originals[rel])
            else:
                originals[rel] = None
        patch_bin = shutil.which("patch")
        if not patch_bin and os.name == "nt":
            for candidate in [
                r"C:\Program Files\Git\usr\bin\patch.exe",
                r"C:\Program Files (x86)\Git\usr\bin\patch.exe",
            ]:
                if os.path.exists(candidate):
                    patch_bin = candidate
                    break
        patch_cmd = [patch_bin or "patch", "-p1", "-s", "--no-backup-if-mismatch"]
        if os.name == "nt":
            patch_cmd.insert(2, "--binary")
        try:
            r = subprocess.run(patch_cmd, input=diff.encode("utf-8"), cwd=scratch,
                               capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            raise PullRequestError(f"could not run `patch` to rebuild the changed files: {exc}") from exc
        if r.returncode != 0:
            msg = (r.stdout + r.stderr).decode("utf-8", "replace").strip()[:200]
            raise PullRequestError("the stored diff no longer applies to the original snapshot: " + msg)
        for rel, _ in blocks:
            out = scratch / rel
            if not out.is_file():
                raise PullRequestError(f"{rel}: not produced by the patch")
            data = out.read_bytes()
            if len(data) > MAX_FILE_BYTES:
                raise PullRequestError(f"{rel}: file is larger than {MAX_FILE_BYTES // 1000} KB")
            old = originals[rel]
            changed.append(ChangedFile(path=rel, content=data, is_new=old is None,
                                       orig_blob_sha=None if old is None else blob_sha(old)))
    return changed


def _clean(text: object, limit: int) -> str:
    """Model- and log-derived text for a title/body line: one line, no @-mentions, bounded."""
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = re.sub(r"@(?=\w)", "@​", s)  # do not ping people from generated text
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def default_title(report: dict) -> str:
    issues = report.get("issues") or []
    if issues:
        return _clean("ReproFix: " + str(issues[0].get("statement") or ""), MAX_TITLE)
    name = (report.get("repository") or {}).get("name") or "repository"
    return _clean(f"ReproFix: repair {name}", MAX_TITLE)


def _num(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def build_body(report: dict, run_id: str) -> str:
    res = report.get("results") or {}
    base, final = res.get("baseline") or {}, res.get("final") or {}
    metric = report.get("metric")
    out = ["## Summary", "", _clean(report.get("status_text"), 400), ""]
    if metric and metric.get("expected") is not None:
        out += ["### Result", "", "| | stage | " + _clean(metric["name"], 60) + " |", "|---|---|---|",
                f"| before | {base.get('stage', 'n/a')} | {_num(base.get('metric_value'))} |",
                f"| after | {final.get('stage', 'n/a')} | {_num(final.get('metric_value'))} |",
                f"| documented | | {_num(metric['expected'])} (tolerance ±{metric.get('tolerance')}) |", ""]
    out += ["### Changes", ""]
    for f in report.get("files_modified") or []:
        out.append(f"- `{f['path']}` ({f['status']}, +{f['added']} −{f['removed']})")
    issues = report.get("issues") or []
    if issues:
        out += ["", "### Diagnosis (written by the model; not verified)", ""]
        for i in issues:
            files = ", ".join(f"`{x}`" for x in i.get("files", []))
            out.append(f"- **{_clean(i.get('category'), 30)}**: {_clean(i.get('statement'), 300)}" + (f" ({files})" if files else ""))
    out += ["", "### Verification (computed by running the project)", ""]
    for c in report.get("verification") or []:
        mark = {True: "✅", False: "❌", None: "➖"}[c.get("passed")]
        out.append(f"- {mark} {_clean(c.get('check'), 80)}" + (f": {_clean(c.get('detail'), 200)}" if c.get("detail") else ""))
    caveats = report.get("caveats") or []
    if caveats:
        out += ["", "### Caveats", ""] + [f"- {_clean(c, 300)}" for c in caveats]
    out += ["", "---", f"Opened by ReproFix from run `{run_id}` using the `{_clean(report.get('llm_backend'), 30)}` model backend. "
            "An automated agent wrote this change: please review the diff before merging."]
    text = "\n".join(out)
    return text if len(text) <= MAX_BODY else text[: MAX_BODY - 1] + "…"


def plan_from_report(report: dict, orig_dir: Path, run_id: str) -> PullRequestPlan:
    """Raises PullRequestError(kind='unavailable') with a user-facing reason when a pull request makes no sense."""
    repo = report.get("repository") or {}
    owner, name = parse_github_repo(repo.get("source") or "")
    if report.get("status") not in ALLOWED_STATUSES:
        raise PullRequestError(f"a pull request is only offered for runs that ended verified, executes or partial (this one: {report.get('status')})", "unavailable")
    if not (report.get("diff") or "").strip():
        raise PullRequestError("the run produced no patch", "unavailable")
    sha = repo.get("commit") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise PullRequestError("this run did not record which commit it started from (it predates pull-request support)", "unavailable")
    if not orig_dir.is_dir():
        raise PullRequestError("the run's original snapshot has been removed from disk", "unavailable")
    files = files_from_diff(orig_dir, report["diff"])
    return PullRequestPlan(owner=owner, repo=name, base_sha=sha, run_id=run_id, status=report["status"],
                           title=default_title(report), body=build_body(report, run_id), files=files)


# ----------------------------------------------------------------------------- GitHub (network)
@dataclass
class PullRequestResult:
    url: str
    number: int
    branch: str
    head_repo: str
    base_repo: str
    base_branch: str
    mode: str  # "branch" (pushed to the repository itself) or "fork"
    commits: int

    def to_dict(self) -> dict:
        return {"url": self.url, "number": self.number, "branch": self.branch, "head_repo": self.head_repo,
                "base_repo": self.base_repo, "base_branch": self.base_branch, "mode": self.mode, "commits": self.commits}


class GitHubClient:
    def __init__(self, token: str, *, transport: httpx.BaseTransport | None = None, timeout: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep):
        if not token or not token.strip():
            raise PullRequestError("a GitHub token is required to open a pull request")
        self._token = token.strip()
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=GITHUB_API, transport=transport, timeout=timeout, follow_redirects=False,
            headers={"Authorization": f"Bearer {self._token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": GITHUB_API_VERSION, "User-Agent": "reprofix"})

    def __repr__(self) -> str:  # never show the token
        return "GitHubClient(token=***)"

    def close(self) -> None:
        self._http.close()

    # -- plumbing
    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***")

    def _request(self, method: str, path: str, *, what: str, ok: tuple[int, ...] = (200,), json: dict | None = None,
                 params: dict | None = None, allow: tuple[int, ...] = ()) -> httpx.Response:
        try:
            r = self._http.request(method, path, json=json, params=params)
        except httpx.HTTPError as exc:
            raise PullRequestError(f"could not reach GitHub while {what} ({type(exc).__name__})", "github") from None
        if r.status_code in ok or r.status_code in allow:
            return r
        raise self._error(r, what)

    def _error(self, r: httpx.Response, what: str) -> PullRequestError:
        try:
            detail = str(r.json().get("message", ""))
        except (ValueError, AttributeError):
            detail = ""
        detail = self._redact(detail)[:300]
        code = r.status_code
        if code == 401:
            hint = "GitHub rejected the token (expired, revoked or mistyped)"
        elif code == 403 and r.headers.get("x-ratelimit-remaining") == "0":
            hint = "GitHub's rate limit was reached; try again later"
        elif code == 403:
            hint = "the token is not allowed to do this (needs Contents and Pull requests write access, or `public_repo` for a classic token)"
        elif code == 404:
            hint = "GitHub could not find it, or the token cannot see it"
        elif code in (301, 302, 307, 308):
            hint = "the repository has moved; update its URL"
        elif code == 409:
            hint = "GitHub reports a conflict"
        elif code == 422:
            hint = "GitHub rejected the request as invalid"
        else:
            hint = f"GitHub answered HTTP {code}"
        return PullRequestError(f"{what}: {hint}" + (f" ({detail})" if detail else ""), "github")

    # -- steps
    def get_repo(self, owner: str, repo: str) -> dict:
        r = self._request("GET", f"/repos/{owner}/{repo}", what=f"looking up {owner}/{repo}")
        data = r.json()
        if data.get("archived"):
            raise PullRequestError(f"{owner}/{repo} is archived and cannot receive pull requests")
        return data

    def _fork(self, owner: str, repo: str) -> tuple[str, str]:
        r = self._request("POST", f"/repos/{owner}/{repo}/forks", json={"default_branch_only": True},
                          what=f"forking {owner}/{repo}", ok=(202, 200, 201))
        d = r.json()
        return d["owner"]["login"], d["name"]

    def _wait_for_commit(self, full: str, sha: str, attempts: int = 24, delay: float = 5.0) -> None:
        for i in range(attempts):
            r = self._request("GET", f"/repos/{full}/git/commits/{sha}", what=f"waiting for {full}", allow=(404, 409))
            if r.status_code == 200:
                return
            self._sleep(delay)
        raise PullRequestError(f"the fork {full} was created but its contents are not available yet; try again in a minute", "github")

    def _create_branch(self, full: str, name: str, sha: str) -> str:
        for n in range(1, 6):
            candidate = name if n == 1 else f"{name}-{n}"
            r = self._request("POST", f"/repos/{full}/git/refs", json={"ref": f"refs/heads/{candidate}", "sha": sha},
                              what=f"creating branch {candidate} in {full}", ok=(201,), allow=(409, 422))
            if r.status_code == 201:
                return candidate
            msg = ""
            try:
                msg = str(r.json().get("message", ""))
            except ValueError:
                pass
            if "already exists" in msg.lower() or r.status_code == 409:
                continue
            raise PullRequestError(f"GitHub could not create the branch at commit {sha[:10]} in {full}: {self._redact(msg)[:200] or 'HTTP ' + str(r.status_code)}. "
                                   "If this is a fork, sync it with the original repository and try again.", "github")
        raise PullRequestError(f"could not find a free branch name starting with {name}", "conflict")

    def _commit_file(self, full: str, branch: str, f: ChangedFile) -> None:
        got = self._request("GET", f"/repos/{full}/contents/{f.path}", params={"ref": branch},
                            what=f"reading {f.path}", allow=(404,))
        existing_sha = None
        if got.status_code == 200:
            info = got.json()
            if isinstance(info, list) or info.get("type") != "file":
                raise PullRequestError(f"{f.path} is not a regular file in {full}")
            if f.is_new:
                raise PullRequestError(f"{f.path} already exists in {full}; the run was not based on this commit")
            existing_sha = info["sha"]
            if f.orig_blob_sha and existing_sha != f.orig_blob_sha:
                raise PullRequestError(f"{f.path} on GitHub differs from the copy the run worked on (for example through line-ending "
                                       "conversion); not overwriting it")
        elif not f.is_new:
            raise PullRequestError(f"{f.path} does not exist at commit {branch} in {full}")
        body = {"message": f"ReproFix: {'add' if f.is_new else 'update'} {f.path}",
                "content": base64.b64encode(f.content).decode("ascii"), "branch": branch}
        if existing_sha:
            body["sha"] = existing_sha
        self._request("PUT", f"/repos/{full}/contents/{f.path}", json=body, what=f"committing {f.path}", ok=(200, 201))


def open_pull_request(client: GitHubClient, plan: PullRequestPlan, *, title: str | None = None, body: str | None = None,
                      draft: bool = False) -> PullRequestResult:
    title = _clean(title, MAX_TITLE) if title is not None else plan.title
    if not title:
        raise PullRequestError("the pull request needs a title")
    body = plan.body if body is None else body[:MAX_BODY]
    upstream = client.get_repo(plan.owner, plan.repo)
    base_branch = upstream["default_branch"]
    can_push = bool((upstream.get("permissions") or {}).get("push"))
    if can_push:
        target, head_owner, mode = plan.full_name, plan.owner, "branch"
    else:
        head_owner, fork_name = client._fork(plan.owner, plan.repo)
        target, mode = f"{head_owner}/{fork_name}", "fork"
    client._wait_for_commit(target, plan.base_sha)
    branch = client._create_branch(target, f"reprofix/{plan.run_id}", plan.base_sha)
    for f in plan.files:
        client._commit_file(target, branch, f)
    pr = {"title": title, "head": branch if mode == "branch" else f"{head_owner}:{branch}", "base": base_branch,
          "body": body, "draft": bool(draft)}
    if mode == "fork":
        pr["maintainer_can_modify"] = True
    r = client._request("POST", f"/repos/{plan.full_name}/pulls", json=pr, what="opening the pull request", ok=(201,))
    d = r.json()
    return PullRequestResult(url=d["html_url"], number=d["number"], branch=branch, head_repo=target, base_repo=plan.full_name,
                             base_branch=base_branch, mode=mode, commits=len(plan.files))
