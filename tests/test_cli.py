import json
import shutil
from pathlib import Path

import pytest

from conftest import NETWORK
from reprofix.cli import main


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for k in ("NEBIUS_API_KEY", "TAVILY_API_KEY", "REPROFIX_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("REPROFIX_ENV_FILE", "")          # a developer's real .env must not leak into tests
    monkeypatch.setenv("REPROFIX_SANDBOX", "local")
    monkeypatch.setenv("REPROFIX_ALLOW_UNSAFE_LOCAL", "1")
    monkeypatch.setenv("REPROFIX_DATA_DIR", str(tmp_path / "data"))


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == "0.1.0"


def test_bench_list_shows_every_task(capsys):
    assert main(["bench", "list"]) == 0
    out = capsys.readouterr().out
    assert out.count("\n") >= 30 and "eval_floor_division" in out and "dep_conflicting_pins" in out
    assert "torch_double_softmax" in out and "device_cuda_config" in out and out.rstrip().endswith("tasks")


def test_doctor_without_a_key_names_the_problem_and_fails(capsys):
    assert main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "NEBIUS_API_KEY is not set" in out and "[FAIL]" in out
    assert "local backend is development-only" in out and "problem(s) found" in out
    assert "Traceback" not in out


def test_run_without_a_key_explains_how_to_continue(capsys, tiny_repo):
    with pytest.raises(SystemExit) as e:
        main(["run", "--path", str(tiny_repo), "--expected", "0.9"])
    assert "NEBIUS_API_KEY" in str(e.value) and "reprofix demo" in str(e.value)


def test_repo_and_path_are_mutually_exclusive():
    with pytest.raises(SystemExit) as e:
        main(["run", "--repo", "https://github.com/o/r", "--path", "/tmp"])
    assert e.value.code == 2


@pytest.mark.skipif(not NETWORK, reason="the demo runs a real pip install; set REPROFIX_TEST_NETWORK=1")
def test_offline_demo_is_labelled_and_verifies(capsys, tmp_path):
    assert main(["demo", "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "NOT Nemotron" in out and "verified" in out.lower()


@pytest.mark.skipif(not NETWORK, reason="needs real pip; set REPROFIX_TEST_NETWORK=1")
def test_bench_run_with_the_oracle_writes_labelled_results_without_touching_the_repo(capsys, tmp_path, monkeypatch):
    import reprofix.cli as cli
    bench = tmp_path / "benchmark"
    bench.mkdir()
    try:
        (bench / "tasks").symlink_to(cli.BENCH_DIR / "tasks", target_is_directory=True)
    except OSError:
        shutil.copytree(cli.BENCH_DIR / "tasks", bench / "tasks")
    repo_results = cli.REPO_ROOT / "benchmark" / "results"
    committed = {p.name: p.read_bytes() for p in repo_results.glob("*")} if repo_results.is_dir() else {}
    monkeypatch.setattr(cli, "BENCH_DIR", bench)
    assert main(["bench", "run", "--llm", "oracle", "--only", "eval_floor_division", "--out", str(tmp_path / "work")]) == 0
    (saved,) = (bench / "results").glob("*.json")
    assert saved.name == "oracle-router-subset.json"                       # a subset run cannot masquerade as the full run
    data = json.loads(saved.read_text())
    assert data["meta"]["llm_backend"] == "scripted" and data["n_tasks"] == 1
    assert "NOT Nemotron" in saved.with_suffix(".md").read_text()
    # the repository's own results directory (which holds the committed oracle run) is exactly as it was
    after = {p.name: p.read_bytes() for p in repo_results.glob("*")} if repo_results.is_dir() else {}
    assert after == committed


def test_doctor_with_a_key_reports_each_tier_against_the_served_catalog(capsys, monkeypatch):
    from reprofix.inference.client import NebiusClient
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    monkeypatch.setattr(NebiusClient, "list_models", lambda self: [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "nvidia/nemotron-3-super-120b-a12b", "nvidia/Nemotron-3_5-Lightning"])
    assert main(["doctor"]) == 1                                            # ultra is not served in this fake catalog
    out = capsys.readouterr().out
    assert "NEBIUS_API_KEY is set" in out and "models" in out
    assert "nano: served as 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B'" in out and "resolves that spelling automatically" in out
    assert "super: nvidia/nemotron-3-super-120b-a12b is served" in out
    assert "ultra: nvidia/Nemotron-3-Ultra-550b-a55b is NOT in the catalog" in out and "Nemotron-3_5-Lightning" in out
    assert "Set NEBIUS_MODEL_ULTRA" in out


def test_doctor_with_a_rejected_key_says_so(capsys, monkeypatch):
    from reprofix.inference.base import AuthError
    from reprofix.inference.client import NebiusClient
    monkeypatch.setenv("NEBIUS_API_KEY", "bad")
    monkeypatch.setattr(NebiusClient, "list_models", lambda self: (_ for _ in ()).throw(AuthError("Token Factory rejected the API key (HTTP 401)")))
    assert main(["doctor"]) == 1
    assert "rejected the API key" in capsys.readouterr().out


# --------------------------------------------------------------------------- reprofix pr (GitHub is an in-memory fake)
import json as _json  # noqa: E402

from tests.fake_github import GOOD_TOKEN, FakeGitHub, make_case, upstream  # noqa: E402


def pr_run_dir(tmp_path, **case):
    import shutil
    (tmp_path / "case").mkdir()
    report, orig, _ = make_case(tmp_path / "case", **case)
    run = tmp_path / "runs" / "run-20261003-120000"
    run.mkdir(parents=True)
    (run / "report.json").write_text(_json.dumps(report))
    shutil.copytree(orig, run / "orig")
    return run, orig


def test_pr_dry_run_shows_the_plan_and_sends_nothing(tmp_path, capsys, monkeypatch):
    run, _ = pr_run_dir(tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert main(["pr", str(run), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "acme/widget" in out and "train.py (modified)" in out and "helper.py (added)" in out
    assert "Verification (computed by running the project)" in out and "nothing was sent to GitHub" in out


def test_pr_without_a_token_stops_with_instructions_before_any_request(tmp_path, monkeypatch):
    run, _ = pr_run_dir(tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit) as e:
        main(["pr", str(run), "--yes"])
    assert "GITHUB_TOKEN" in str(e.value)


def test_pr_asks_before_acting_and_declining_sends_nothing(tmp_path, capsys, monkeypatch):
    import reprofix.core.pullrequest as P
    gh = FakeGitHub()
    run, orig = pr_run_dir(tmp_path)
    upstream(gh, orig, push=True)
    monkeypatch.setenv("GITHUB_TOKEN", GOOD_TOKEN)
    monkeypatch.setattr(P, "GitHubClient", lambda token: (_ for _ in ()).throw(AssertionError("must not be created")))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert main(["pr", str(run)]) == 1
    assert "aborted" in capsys.readouterr().out and gh.requests == []


def test_pr_opens_the_pull_request_and_prints_its_url(tmp_path, capsys, monkeypatch):
    import reprofix.core.pullrequest as P
    gh = FakeGitHub()
    run, orig = pr_run_dir(tmp_path)
    upstream(gh, orig, push=True)
    monkeypatch.setenv("GITHUB_TOKEN", GOOD_TOKEN)
    real = P.GitHubClient
    monkeypatch.setattr(P, "GitHubClient", lambda token: real(token, transport=gh.transport()))
    assert main(["pr", str(run), "--yes", "--draft", "--title", "Custom title"]) == 0
    out = capsys.readouterr().out
    assert "https://github.com/acme/widget/pull/1" in out and "reprofix/run-20261003-120000" in out and GOOD_TOKEN not in out
    assert gh.prs[0]["title"] == "Custom title" and gh.prs[0]["draft"] is True


def test_pr_reports_github_errors_without_a_traceback_or_the_token(tmp_path, monkeypatch):
    import reprofix.core.pullrequest as P
    gh = FakeGitHub()
    run, _ = pr_run_dir(tmp_path)                       # the repository is not known to the fake: 404
    monkeypatch.setenv("GITHUB_TOKEN", GOOD_TOKEN)
    real = P.GitHubClient
    monkeypatch.setattr(P, "GitHubClient", lambda token: real(token, transport=gh.transport()))
    with pytest.raises(SystemExit) as e:
        main(["pr", str(run), "--yes"])
    assert "pull request failed" in str(e.value) and GOOD_TOKEN not in str(e.value)


def test_pr_explains_when_a_run_cannot_produce_one(tmp_path):
    run, _ = pr_run_dir(tmp_path, status="failed")
    with pytest.raises(SystemExit, match="cannot open a pull request: a pull request is only offered"):
        main(["pr", str(run), "--dry-run"])
    with pytest.raises(SystemExit, match="no report.json"):
        main(["pr", str(tmp_path / "nowhere"), "--dry-run"])


def test_bench_list_and_compare(capsys):
    assert main(["bench", "list"]) == 0
    out = capsys.readouterr().out
    assert "eval_floor_division" in out and "30 tasks" in out

    oracle_json = Path(__file__).resolve().parents[1] / "benchmark" / "results" / "oracle-router.json"
    assert main(["bench", "compare", str(oracle_json)]) == 0
    out_cmp = capsys.readouterr().out
    assert "oracle-router" in out_cmp and "30/30" in out_cmp
