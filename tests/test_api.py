import time

import pytest
from fastapi.testclient import TestClient

from conftest import BUG_LINE, FIX_LINE, NETWORK, hypothesis_reply, repair_reply, scripted
from reprofix.api.app import create_app
from reprofix.api.store import Store
from reprofix.sandbox.local_backend import LocalSandbox

H1 = ("h1", "floor division truncates the accuracy to zero")
TERMINAL = {"verified", "executes", "already_passing", "partial", "failed", "cancelled", "error", "interrupted"}


def make_app(settings, **kw):
    backend = kw.pop("backend", lambda req: scripted([hypothesis_reply(*H1, BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)]))
    return create_app(settings, backend_factory=backend, sandbox_factory=lambda: LocalSandbox(settings))


@pytest.fixture
def client(settings):
    with TestClient(make_app(settings)) as c:
        yield c


def body(tiny_repo, **over):
    d = {"local_path": str(tiny_repo), "command": "python train.py", "metric": {"name": "val_accuracy", "expected": 0.9, "tolerance": 0.02}}
    d.update(over)
    return d


def wait(client, run_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in TERMINAL:
            return run
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_health_reports_what_is_and_is_not_configured(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["llm"]["configured"] is False and h["tavily"]["configured"] is False
    assert h["sandbox"]["available"] and h["sandbox"]["filesystem_isolated"] is False and h["sandbox"]["kind"] == "local"
    assert h["llm"]["models"]["ultra"] == "nvidia/Nemotron-3-Ultra-550b-a55b" and h["auth_required"] is False
    assert h["limits"]["allowed_git_hosts"] == ["github.com"]


@pytest.mark.parametrize("payload,fragment", [
    ({}, "exactly one"),
    ({"repo_url": "https://github.com/o/r", "local_path": "/tmp"}, "exactly one"),
    ({"repo_url": "https://evil.example/o/r"}, "not allowed"),
    ({"repo_url": "http://github.com/o/r"}, "https"),
    ({"repo_url": "https://github.com/o/r", "command": "bash run.sh"}, "must start with"),
    ({"repo_url": "https://github.com/o/r", "command": "python a.py; rm -rf /"}, "metacharacters"),
])
def test_invalid_requests_are_rejected_before_any_work(client, payload, fragment):
    r = client.post("/api/runs", json=payload)
    assert r.status_code == 400 and fragment in r.json()["error"]
    assert client.get("/api/runs").json() == []


def test_schema_violations_are_422(client):
    assert client.post("/api/runs", json={"repo_url": "https://github.com/o/r", "router_mode": "turbo"}).status_code == 422
    assert client.post("/api/runs", json={"repo_url": "https://github.com/o/r", "metric": {"regex": "no group"}}).status_code == 422


def test_local_paths_are_refused_unless_the_server_allows_them(settings, tiny_repo):
    settings.allow_local_paths = False
    with TestClient(make_app(settings)) as c:
        r = c.post("/api/runs", json=body(tiny_repo))
        assert r.status_code == 400 and "disabled" in r.json()["error"]


def test_client_supplied_limits_are_clamped_to_server_ceilings(settings, tiny_repo):
    settings.max_attempts_ceiling = 3
    with TestClient(make_app(settings)) as c:
        rid = c.post("/api/runs", json=body(tiny_repo, max_attempts=999, command_timeout_s=99999, max_total_tokens=10**12)).json()["id"]
        req = c.get(f"/api/runs/{rid}").json()["request"]
        assert req["max_attempts"] == 3 and req["command_timeout_s"] <= settings.max_command_timeout_ceiling
        assert req["max_total_tokens"] <= 2_000_000
        wait(c, rid)


def test_a_run_end_to_end_through_the_api(client, tiny_repo):
    r = client.post("/api/runs", json=body(tiny_repo))
    assert r.status_code == 202
    rid = r.json()["id"]
    run = wait(client, rid)
    assert run["status"] == "verified" and run["report"]["results"]["final"]["metric_value"] == 0.9
    assert any(x["id"] == rid for x in client.get("/api/runs").json())

    g = client.get(f"/api/runs/{rid}/graph").json()
    assert {n["type"] for n in g["nodes"]} >= {"symptom", "hypothesis", "evidence", "experiment", "result"}

    p = client.get(f"/api/runs/{rid}/patch")
    assert p.status_code == 200 and f"+{FIX_LINE}" in p.text
    assert f"reprofix-{rid}.patch" in p.headers["content-disposition"] and p.headers["content-type"].startswith("text/plain")


def test_event_stream_replays_in_order_and_ends(client, tiny_repo):
    rid = client.post("/api/runs", json=body(tiny_repo)).json()["id"]
    wait(client, rid)
    r = client.get(f"/api/runs/{rid}/events")
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [(blk.split("\n")[0], blk) for blk in r.text.strip().split("\n\n")]
    kinds = [b.split("event: ")[1].split("\n")[0] for _, b in events]
    assert kinds[0] == "run.started" and kinds[-1] == "end" and kinds[-2] == "run.finished"
    ids = [int(l.removeprefix("id: ")) for l in r.text.splitlines() if l.startswith("id: ")]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_event_stream_resumes_after_last_event_id(client, tiny_repo):
    rid = client.post("/api/runs", json=body(tiny_repo)).json()["id"]
    wait(client, rid)
    full = [int(l.removeprefix("id: ")) for l in client.get(f"/api/runs/{rid}/events").text.splitlines() if l.startswith("id: ")]
    mid = full[len(full) // 2]
    resumed = [int(l.removeprefix("id: ")) for l in
               client.get(f"/api/runs/{rid}/events", headers={"Last-Event-ID": str(mid)}).text.splitlines() if l.startswith("id: ")]
    assert resumed == [i for i in full if i > mid]
    via_query = [int(l.removeprefix("id: ")) for l in client.get(f"/api/runs/{rid}/events?after={mid}").text.splitlines() if l.startswith("id: ")]
    assert via_query == resumed
    assert client.get(f"/api/runs/{rid}/events", headers={"Last-Event-ID": "garbage"}).status_code == 200   # not a 500


@pytest.mark.parametrize("path", ["", "/graph", "/patch", "/events", "/cancel"])
def test_unknown_run_ids_are_404(client, path):
    method = client.post if path == "/cancel" else client.get
    assert method(f"/api/runs/doesnotexist{path}").status_code == 404


def test_cancel_a_queued_or_running_run(settings, tiny_repo):
    import threading
    gate = threading.Event()

    def slow_backend(req):
        gate.wait(10)                                    # holds the job at the first model call so cancel arrives mid-run
        return scripted([hypothesis_reply(*H1, BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])

    with TestClient(make_app(settings, backend=slow_backend)) as c:
        rid = c.post("/api/runs", json=body(tiny_repo)).json()["id"]
        time.sleep(0.3)
        assert c.post(f"/api/runs/{rid}/cancel").json() == {"cancelled": True}
        gate.set()
        run = wait(c, rid)
        assert run["status"] == "cancelled"
        assert c.post(f"/api/runs/{rid}/cancel").json() == {"cancelled": False}      # nothing left to cancel


def test_a_crashing_backend_becomes_an_error_run_not_a_hung_one(settings, tiny_repo):
    def broken(req):
        raise RuntimeError("backend factory exploded")

    with TestClient(make_app(settings, backend=broken)) as c:
        rid = c.post("/api/runs", json=body(tiny_repo)).json()["id"]
        run = wait(c, rid)
        assert run["status"] == "error"
        assert "exploded" in c.get(f"/api/runs/{rid}/events").text


def test_missing_api_key_gives_a_clear_error_run(settings, tiny_repo):
    app = create_app(settings, sandbox_factory=lambda: LocalSandbox(settings))      # default backend factory = real Nebius client
    with TestClient(app) as c:
        rid = c.post("/api/runs", json=body(tiny_repo)).json()["id"]
        run = wait(c, rid)
        assert run["status"] == "error" and "NEBIUS_API_KEY" in c.get(f"/api/runs/{rid}/events").text


def test_runs_left_running_by_a_crash_are_marked_interrupted_on_restart(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    st = Store(settings.data_dir / "reprofix.db")
    st.create_run("abc", {"goal": "x"})
    st.set_status("abc", "running")
    st.close()
    with TestClient(make_app(settings)) as c:
        assert c.get("/api/runs/abc").json()["status"] == "interrupted"


# --------------------------------------------------------------------------- optional bearer token
def test_api_token_protects_everything_but_health_and_static_files(settings, tiny_repo):
    settings.api_token = "s3cret"
    with TestClient(make_app(settings)) as c:
        assert c.get("/api/health").json()["auth_required"] is True
        assert c.get("/api/runs").status_code == 401
        assert c.post("/api/runs", json=body(tiny_repo)).status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/api/runs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/api/runs?token=s3cret").status_code == 200                    # EventSource cannot set headers
        rid = c.post("/api/runs", json=body(tiny_repo), headers={"Authorization": "Bearer s3cret"}).json()["id"]
        assert c.get(f"/api/runs/{rid}/events").status_code == 401
        assert c.get(f"/api/runs/{rid}/patch").status_code == 401
        wait_headers = {"Authorization": "Bearer s3cret"}
        t0 = time.time()
        while time.time() - t0 < 60 and c.get(f"/api/runs/{rid}", headers=wait_headers).json()["status"] not in TERMINAL:
            time.sleep(0.1)


# --------------------------------------------------------------------------- demo + benchmark endpoints
@pytest.mark.skipif(not NETWORK, reason="the demo's first fault is a dependency problem, which needs real pip; set REPROFIX_TEST_NETWORK=1")
def test_demo_runs_offline_and_is_labelled_as_scripted(settings):
    with TestClient(make_app(settings)) as c:
        assert c.get("/api/health").json()["demo_available"] is True
        rid = c.post("/api/demo").json()["id"]
        run = wait(c, rid, timeout=180)
        assert run["status"] == "verified" and run["report"]["llm_backend"] == "scripted"
        assert any("NOT NVIDIA Nemotron" in x for x in run["report"]["caveats"])
        assert run["label"].startswith("demo")


def test_benchmark_endpoint_lists_tasks(client):
    b = client.get("/api/benchmark").json()
    assert b["n_tasks"] >= 30 and isinstance(b["results"], list)


def test_static_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ReproFix" in r.text
    assert client.get("/app.js").status_code == 200


# --------------------------------------------------------------------------- pull requests (GitHub is an in-memory fake)
import shutil  # noqa: E402
import threading  # noqa: E402

from tests.fake_github import GOOD_TOKEN, FakeGitHub, make_case, upstream  # noqa: E402


def pr_app(settings, gh):
    return create_app(settings, backend_factory=lambda req: None, sandbox_factory=lambda: LocalSandbox(settings),
                      github_transport=gh.transport())


def seed_run(c, settings, tmp_path, run_id="seeded1", *, request=None, **case):
    """A finished run with a report and an original snapshot on disk, as the server would have left it."""
    (tmp_path / "case").mkdir(exist_ok=True)
    report, orig, _ = make_case(tmp_path / "case", **case)
    c.app.state.store.create_run(run_id, request or {"repo_url": "https://github.com/acme/widget"}, None)
    c.app.state.store.save_report(run_id, report)
    shutil.copytree(orig, settings.data_dir / "runs" / run_id / "orig")
    return run_id, orig


def confirm(**over):
    return {"token": GOOD_TOKEN, "confirm": True, **over}


def test_pull_request_preview_needs_no_network_and_shows_exactly_what_would_be_sent(settings, tmp_path):
    gh = FakeGitHub()
    with TestClient(pr_app(settings, gh)) as c:
        rid, _ = seed_run(c, settings, tmp_path)
        d = c.get(f"/api/runs/{rid}/pull-request").json()
        assert d["available"] is True and d["target"] == "acme/widget" and d["existing"] is None
        assert {f["path"] for f in d["files"]} == {"train.py", "requirements.txt", "helper.py"}
        assert d["title"].startswith("ReproFix: ") and "Verification" in d["body"] and len(d["base_commit"]) == 40
        assert gh.requests == []
        assert c.get("/api/health").json()["pull_requests"] is True


@pytest.mark.parametrize("case,reason", [
    ({"status": "failed"}, "only offered"),
    ({"commit": None}, "predates"),
])
def test_pull_request_preview_explains_why_it_is_unavailable(settings, tmp_path, case, reason):
    with TestClient(pr_app(settings, FakeGitHub())) as c:
        rid, _ = seed_run(c, settings, tmp_path, **case)
        d = c.get(f"/api/runs/{rid}/pull-request").json()
        assert d["available"] is False and reason in d["reason"]


def test_preview_for_a_run_started_from_a_local_path_or_still_running_is_unavailable(settings, tmp_path):
    with TestClient(pr_app(settings, FakeGitHub())) as c:
        rid, _ = seed_run(c, settings, tmp_path, "local1", source="/home/me/project")
        assert "github.com" in c.get(f"/api/runs/{rid}/pull-request").json()["reason"]
        c.app.state.store.create_run("live1", {"repo_url": "https://github.com/acme/widget"}, None)
        assert "not finished" in c.get("/api/runs/live1/pull-request").json()["reason"]
        assert c.get("/api/runs/nope/pull-request").status_code == 404


def test_creating_a_pull_request_requires_explicit_confirmation_and_a_token(settings, tmp_path):
    gh = FakeGitHub()
    with TestClient(pr_app(settings, gh)) as c:
        rid, orig = seed_run(c, settings, tmp_path)
        upstream(gh, orig, push=True)
        r = c.post(f"/api/runs/{rid}/pull-request", json={"token": GOOD_TOKEN})
        assert r.status_code == 400 and "confirm" in r.json()["error"]
        r = c.post(f"/api/runs/{rid}/pull-request", json={"confirm": True})
        assert r.status_code == 400 and "token" in r.json()["error"]
        r = c.post(f"/api/runs/{rid}/pull-request", json={"confirm": True, "token": "   "})
        assert r.status_code == 400
        assert gh.requests == [] and c.get(f"/api/runs/{rid}/pull-request").json()["existing"] is None


def test_a_pull_request_is_opened_once_recorded_and_the_token_is_never_stored(settings, tmp_path):
    gh = FakeGitHub()
    with TestClient(pr_app(settings, gh)) as c:
        rid, orig = seed_run(c, settings, tmp_path)
        upstream(gh, orig, push=True)
        r = c.post(f"/api/runs/{rid}/pull-request", json=confirm(title="Fix training", draft=True))
        assert r.status_code == 201, r.text
        res = r.json()
        assert res["url"] == "https://github.com/acme/widget/pull/1" and res["mode"] == "branch" and res["commits"] == 3
        assert gh.prs[0]["title"] == "Fix training" and gh.prs[0]["draft"] is True
        assert GOOD_TOKEN not in r.text
        # second attempt: refused, with the link to the first
        again = c.post(f"/api/runs/{rid}/pull-request", json=confirm())
        assert again.status_code == 409 and res["url"] in again.json()["error"] and len(gh.prs) == 1
        assert c.get(f"/api/runs/{rid}/pull-request").json()["existing"]["url"] == res["url"]
        db = (settings.data_dir / "reprofix.db")
        blob = b"".join(p.read_bytes() for p in settings.data_dir.glob("reprofix.db*"))
        assert db.exists() and GOOD_TOKEN.encode() not in blob


def test_the_fork_flow_is_reachable_through_the_api(settings, tmp_path):
    gh = FakeGitHub()
    with TestClient(pr_app(settings, gh)) as c:
        rid, orig = seed_run(c, settings, tmp_path)
        upstream(gh, orig, push=False)
        r = c.post(f"/api/runs/{rid}/pull-request", json=confirm())
        assert r.status_code == 201 and r.json()["mode"] == "fork" and r.json()["head_repo"] == "me/widget"


def test_github_failures_become_502_with_a_clear_message_and_can_be_retried(settings, tmp_path):
    gh = FakeGitHub()
    with TestClient(pr_app(settings, gh)) as c:
        rid, orig = seed_run(c, settings, tmp_path)
        r = c.post(f"/api/runs/{rid}/pull-request", json={"token": "ghp_not_the_right_one", "confirm": True})
        assert r.status_code == 502 and "rejected the token" in r.json()["error"] and "ghp_not_the_right_one" not in r.text
        assert c.get(f"/api/runs/{rid}/pull-request").json()["existing"] is None
        upstream(gh, orig, push=True)
        assert c.post(f"/api/runs/{rid}/pull-request", json=confirm()).status_code == 201


def test_unavailable_runs_cannot_be_posted_to(settings, tmp_path):
    with TestClient(pr_app(settings, FakeGitHub())) as c:
        rid, _ = seed_run(c, settings, tmp_path, status="failed")
        r = c.post(f"/api/runs/{rid}/pull-request", json=confirm())
        assert r.status_code == 400 and "only offered" in r.json()["error"]
        assert c.post("/api/runs/nope/pull-request", json=confirm()).status_code == 404


def test_pull_request_endpoints_honour_the_api_token_and_can_be_disabled(settings, tmp_path):
    settings.api_token = "secret-api-token"
    gh = FakeGitHub()
    with TestClient(pr_app(settings, gh)) as c:
        rid, orig = seed_run(c, settings, tmp_path)
        upstream(gh, orig, push=True)
        assert c.get(f"/api/runs/{rid}/pull-request").status_code == 401
        assert c.post(f"/api/runs/{rid}/pull-request", json=confirm()).status_code == 401
        hdr = {"Authorization": "Bearer secret-api-token"}
        assert c.get(f"/api/runs/{rid}/pull-request", headers=hdr).json()["available"] is True
        assert c.post(f"/api/runs/{rid}/pull-request", json=confirm(), headers=hdr).status_code == 201


def test_pull_requests_can_be_switched_off_for_a_public_instance(settings, tmp_path):
    settings.allow_pull_requests = False
    with TestClient(pr_app(settings, FakeGitHub())) as c:
        rid, _ = seed_run(c, settings, tmp_path)
        assert c.get("/api/health").json()["pull_requests"] is False
        d = c.get(f"/api/runs/{rid}/pull-request").json()
        assert d["available"] is False and "disabled" in d["reason"]
        assert c.post(f"/api/runs/{rid}/pull-request", json=confirm()).status_code == 403


def test_a_double_submit_opens_exactly_one_pull_request(settings, tmp_path):
    gh = FakeGitHub()
    with TestClient(pr_app(settings, gh)) as c:
        rid, orig = seed_run(c, settings, tmp_path)
        upstream(gh, orig, push=True)
        codes: list[int] = []
        go = threading.Barrier(2)

        def post():
            go.wait()
            codes.append(c.post(f"/api/runs/{rid}/pull-request", json=confirm()).status_code)
        threads = [threading.Thread(target=post) for _ in range(2)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert sorted(codes) == [201, 409] and len(gh.prs) == 1
