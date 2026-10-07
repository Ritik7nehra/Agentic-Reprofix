"""Pull-request feature: planning from a report (offline) and the GitHub flow against an in-memory fake API.

The fake encodes GitHub's documented behaviour; nothing here has been run against the real service.
"""
from __future__ import annotations

import base64

import httpx
import pytest

from reprofix.core.patching import make_diff
from reprofix.core.pullrequest import (
    GitHubClient, PullRequestError, blob_sha, build_body, default_title, files_from_diff, open_pull_request,
    parse_github_repo, plan_from_report,
)
from tests.fake_github import BASE_SHA, GOOD_TOKEN, FakeGitHub, client, make_case, upstream

# --------------------------------------------------------------------------- offline pieces
def test_blob_sha_matches_what_git_computes():
    assert blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


@pytest.mark.parametrize("url", ["https://github.com/acme/widget", "https://github.com/acme/widget.git", "https://github.com/a-b/c.d/"])
def test_parse_github_repo_accepts_github_urls(url):
    owner, repo = parse_github_repo(url)
    assert owner and repo and not repo.endswith(".git")


@pytest.mark.parametrize("url", ["", "/tmp/some/dir", "http://github.com/a/b", "https://gitlab.com/a/b", "https://github.com/a",
                                 "https://github.com/a/b/tree/main", "https://github.com/../b"])
def test_parse_github_repo_rejects_everything_else(url):
    with pytest.raises(PullRequestError) as e:
        parse_github_repo(url)
    assert e.value.kind == "unavailable"


def test_files_from_diff_rebuilds_contents_from_the_snapshot_not_the_workspace(tmp_path):
    report, orig, work = make_case(tmp_path)
    (work / "train.py").write_text("# the repository's own code rewrote this file while it ran\n")   # must not leak into the PR
    files = {f.path: f for f in files_from_diff(orig, report["diff"])}
    assert set(files) == {"train.py", "requirements.txt", "helper.py"}
    assert files["train.py"].content == b"a = 1\nb = a / 2\nprint(b)\n"
    assert files["requirements.txt"].content == b"numpy>=1.26\n"
    assert files["helper.py"].is_new and files["helper.py"].orig_blob_sha is None
    assert files["train.py"].orig_blob_sha == blob_sha((orig / "train.py").read_bytes())


def test_files_from_diff_keeps_a_missing_trailing_newline(tmp_path):
    orig, work = tmp_path / "o", tmp_path / "w"
    orig.mkdir(); work.mkdir()
    (orig / "x.txt").write_bytes(b"one\ntwo")
    (work / "x.txt").write_bytes(b"one\nthree")
    diff, _ = make_diff(orig, work, ["x.txt"])
    (f,) = files_from_diff(orig, diff)
    assert f.content == b"one\nthree"


def test_files_from_diff_refuses_unsafe_or_unsupported_changes(tmp_path):
    report, orig, _ = make_case(tmp_path)
    for bad in ("../evil.py", ".github/workflows/ci.yml", ".git/config", "/etc/passwd"):
        evil = f"diff --git a/{bad} b/{bad}\nnew file mode 100644\n--- /dev/null\n+++ b/{bad}\n@@ -0,0 +1 @@\n+x\n"
        with pytest.raises(PullRequestError):
            files_from_diff(orig, evil)
    deletion = "diff --git a/train.py b/train.py\ndeleted file mode 100644\n--- a/train.py\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-a = 1\n-b = a // 2\n-print(b)\n"
    with pytest.raises(PullRequestError, match="deleting"):
        files_from_diff(orig, deletion)
    with pytest.raises(PullRequestError, match="no patch"):
        files_from_diff(orig, "")


def test_files_from_diff_detects_a_snapshot_that_no_longer_matches(tmp_path):
    report, orig, _ = make_case(tmp_path)
    (orig / "train.py").write_text("something else entirely\n")
    with pytest.raises(PullRequestError, match="no longer applies"):
        files_from_diff(orig, report["diff"])


def test_title_and_body_are_derived_from_the_report_and_neutralise_mentions(tmp_path):
    report, _, _ = make_case(tmp_path)
    title = default_title(report)
    assert title.startswith("ReproFix: requirements.txt pins numpy twice") and "@maintainer" not in title and "@​maintainer" in title
    assert "\n" not in title and len(title) <= 120
    body = build_body(report, "run123")
    assert "Verified: the project runs" in body
    assert "| after | verified | 0.8815 |" in body and "| before | install_failure | n/a |" in body
    assert "✅ Application executes" in body and "➖ Tests pass" in body and "❌ Expected value not hardcoded" in body
    assert "written by the model; not verified" in body and "NOT NVIDIA Nemotron" in body and "run123" in body and "scripted" in body
    assert "@maintainer" not in body


@pytest.mark.parametrize("change,reason", [
    ({"source": "/home/me/proj"}, "github.com"),
    ({"status": "failed"}, "only offered"),
    ({"status": "error"}, "only offered"),
    ({"commit": None}, "predates"),
    ({"commit": "main"}, "predates"),
])
def test_plan_is_unavailable_when_a_pull_request_makes_no_sense(tmp_path, change, reason):
    kwargs = {k: v for k, v in change.items() if k in {"source", "status", "commit"}}
    report, orig, _ = make_case(tmp_path, **kwargs)
    with pytest.raises(PullRequestError, match=reason) as e:
        plan_from_report(report, orig, "r1")
    assert e.value.kind == "unavailable"


def test_plan_is_unavailable_without_a_patch_or_snapshot(tmp_path):
    report, orig, _ = make_case(tmp_path)
    with pytest.raises(PullRequestError, match="no patch"):
        plan_from_report({**report, "diff": ""}, orig, "r1")
    with pytest.raises(PullRequestError, match="snapshot has been removed"):
        plan_from_report(report, tmp_path / "gone", "r1")


def test_plan_describes_the_target_without_exposing_file_contents(tmp_path):
    report, orig, _ = make_case(tmp_path)
    plan = plan_from_report(report, orig, "r1")
    d = plan.to_dict()
    assert d["target"] == "acme/widget" and d["base_commit"] == BASE_SHA and d["status"] == "verified"
    assert {f["path"]: f["status"] for f in d["files"]} == {"train.py": "modified", "requirements.txt": "modified", "helper.py": "added"}
    assert "content" not in str(d["files"])


# --------------------------------------------------------------------------- GitHub flow
def test_branch_mode_commits_each_file_then_opens_the_pull_request(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    up = upstream(gh, orig, push=True)
    plan = plan_from_report(report, orig, "run123")
    res = open_pull_request(client(gh), plan)
    assert res.mode == "branch" and res.url == "https://github.com/acme/widget/pull/1" and res.commits == 3
    assert res.branch == "reprofix/run123" and res.head_repo == "acme/widget" and res.base_branch == "main"
    calls = gh.calls()
    assert calls[0] == "GET /repos/acme/widget" and "POST /repos/acme/widget/forks" not in calls
    assert calls.index("POST /repos/acme/widget/git/refs") < calls.index("POST /repos/acme/widget/pulls") == len(calls) - 1
    assert sum(c.startswith("PUT ") for c in calls) == 3
    # the committed bytes are the rebuilt files, base64 encoded, with the blob SHA only for files that already existed
    branch = up.branches["reprofix/run123"]
    assert branch["train.py"] == b"a = 1\nb = a / 2\nprint(b)\n" and branch["helper.py"] == b"X = 1\n"
    assert up.files["train.py"] == b"a = 1\nb = a // 2\nprint(b)\n"            # the default branch itself is untouched
    puts = {r.url.path.split("/contents/")[1]: b for r, b in zip(gh.requests, gh.bodies) if r.method == "PUT"}
    assert "sha" in puts["train.py"] and "sha" not in puts["helper.py"]
    assert base64.b64decode(puts["helper.py"]["content"]) == b"X = 1\n" and puts["train.py"]["branch"] == "reprofix/run123"
    pr = gh.prs[0]
    assert pr["head"] == "reprofix/run123" and pr["base"] == "main" and pr["title"] == plan.title and "Verification" in pr["body"]
    assert "maintainer_can_modify" not in pr and pr["draft"] is False


def test_fork_mode_waits_for_the_fork_and_opens_a_cross_repository_pull_request(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    gh.fork_ready_after = 3
    upstream(gh, orig, push=False)
    sleeps: list = []
    res = open_pull_request(client(gh, sleeps), plan_from_report(report, orig, "run9"), draft=True)
    assert res.mode == "fork" and res.head_repo == "me/widget" and res.base_repo == "acme/widget"
    assert "POST /repos/acme/widget/forks" in gh.calls() and len(sleeps) == 3
    pr = gh.prs[0]
    assert pr["head"] == "me:reprofix/run9" and pr["maintainer_can_modify"] is True and pr["draft"] is True
    assert "reprofix/run9" in gh.repos["me/widget"].branches and "reprofix/run9" not in gh.repos["acme/widget"].branches


def test_a_fork_that_never_becomes_ready_gives_a_clear_error(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    gh.fork_ready_after = 10_000
    upstream(gh, orig, push=False)
    sleeps: list = []
    with pytest.raises(PullRequestError, match="not available yet") as e:
        open_pull_request(client(gh, sleeps), plan_from_report(report, orig, "r"))
    assert e.value.kind == "github" and len(sleeps) == 24
    assert not any(c.startswith("PUT ") for c in gh.calls())


def test_a_taken_branch_name_gets_a_numeric_suffix(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    up = upstream(gh, orig, push=True)
    up.branches["reprofix/r1"] = {}
    up.branches["reprofix/r1-2"] = {}
    res = open_pull_request(client(gh), plan_from_report(report, orig, "r1"))
    assert res.branch == "reprofix/r1-3"


def test_it_refuses_to_overwrite_a_file_that_differs_from_the_runs_snapshot(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    up = upstream(gh, orig, push=True)
    up.files["train.py"] = b"a = 1\r\nb = a // 2\r\nprint(b)\r\n"          # e.g. CRLF on GitHub, LF in the clone
    with pytest.raises(PullRequestError, match="differs from the copy the run worked on"):
        open_pull_request(client(gh), plan_from_report(report, orig, "r"))
    assert not gh.prs


def test_it_refuses_to_create_a_file_that_already_exists_upstream(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    up = upstream(gh, orig, push=True)
    up.files["helper.py"] = b"X = 0\n"
    with pytest.raises(PullRequestError, match="already exists"):
        open_pull_request(client(gh), plan_from_report(report, orig, "r"))


def test_an_archived_repository_is_refused_before_anything_is_written(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    upstream(gh, orig, push=True).archived = True
    with pytest.raises(PullRequestError, match="archived"):
        open_pull_request(client(gh), plan_from_report(report, orig, "r"))
    assert gh.calls() == ["GET /repos/acme/widget"]


def test_a_missing_repository_and_a_vanished_base_commit_are_reported(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    with pytest.raises(PullRequestError, match="could not find it"):
        open_pull_request(client(gh), plan_from_report(report, orig, "r"))
    up = upstream(gh, orig, push=True)
    up.base_sha = "b" * 40                                                     # upstream history was rewritten
    with pytest.raises(PullRequestError, match="not available yet"):
        open_pull_request(client(gh, []), plan_from_report(report, orig, "r"))


def test_errors_never_contain_the_token_and_the_client_never_prints_it(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    upstream(gh, orig, push=True)
    secret = "ghp_wrong_but_secret_value"
    c = client(gh, token=secret)
    assert secret not in repr(c) and secret not in str(c)
    with pytest.raises(PullRequestError, match="rejected the token") as e:
        open_pull_request(c, plan_from_report(report, orig, "r"))
    assert secret not in str(e.value) and e.value.kind == "github"
    # a hostile or careless API message that echoes the token is redacted too
    gh2 = FakeGitHub()
    upstream(gh2, orig, push=True)
    gh2.fail[("GET", r"/repos/acme/widget")] = (500, f"oops {GOOD_TOKEN}")
    with pytest.raises(PullRequestError) as e2:
        open_pull_request(client(gh2), plan_from_report(report, orig, "r"))
    assert GOOD_TOKEN not in str(e2.value) and "***" in str(e2.value)


def test_a_redirect_is_never_followed_so_the_token_is_not_sent_to_another_host(tmp_path):
    report, orig, _ = make_case(tmp_path)
    plan = plan_from_report(report, orig, "r")
    seen = []

    def redirect(request):
        seen.append((request.url.host, request.headers.get("authorization")))
        return httpx.Response(301, headers={"location": "https://evil.example/collect"})
    with pytest.raises(PullRequestError, match="has moved") as e:
        open_pull_request(GitHubClient(GOOD_TOKEN, transport=httpx.MockTransport(redirect)), plan)
    assert [h for h, _ in seen] == ["api.github.com"]                       # no second request, to evil.example or anywhere
    assert GOOD_TOKEN not in str(e.value)


def test_a_token_is_required():
    with pytest.raises(PullRequestError, match="token is required"):
        GitHubClient("   ")


def test_rate_limit_network_failure_and_existing_pull_request_have_distinct_messages(tmp_path):
    report, orig, _ = make_case(tmp_path)
    plan = plan_from_report(report, orig, "r")
    gh = FakeGitHub()
    upstream(gh, orig, push=True)
    gh.rate_limited = True
    with pytest.raises(PullRequestError, match="rate limit"):
        open_pull_request(client(gh), plan)

    def boom(request):
        raise httpx.ConnectError("no route to host")
    with pytest.raises(PullRequestError, match="could not reach GitHub") as e:
        open_pull_request(GitHubClient(GOOD_TOKEN, transport=httpx.MockTransport(boom)), plan)
    assert GOOD_TOKEN not in str(e.value)

    gh3 = FakeGitHub()
    upstream(gh3, orig, push=True)
    gh3.fail[("POST", r"/repos/acme/widget/pulls")] = (422, "A pull request already exists for acme:reprofix/r.")
    with pytest.raises(PullRequestError, match="already exists for acme"):
        open_pull_request(client(gh3), plan)


def test_a_custom_title_is_cleaned_and_an_empty_one_is_refused(tmp_path):
    report, orig, _ = make_case(tmp_path)
    gh = FakeGitHub()
    upstream(gh, orig, push=True)
    open_pull_request(client(gh), plan_from_report(report, orig, "r"), title="Fix\nthe   thing @bob", body="custom body")
    assert gh.prs[0]["title"] == "Fix the thing @​bob" and gh.prs[0]["body"] == "custom body"
    with pytest.raises(PullRequestError, match="needs a title"):
        open_pull_request(client(FakeGitHub()), plan_from_report(report, orig, "r"), title="   ")
