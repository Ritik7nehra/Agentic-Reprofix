"""The evidence graph must tell the story of every run, not just the runs that were fixed: a passing project, a run whose
diagnosis never came back, and a repaired project all start at the repository and end in an outcome. Plus the log clean-up
(terminal colour codes) that went with it."""
from __future__ import annotations

import json

import pytest

from conftest import BUG_LINE, FIX_LINE, TRAIN_PY, hypothesis_reply, repair_reply, scripted
from reprofix.core.evidence import EvidenceGraph
from reprofix.core.orchestrator import Orchestrator
from reprofix.models import MetricSpec, RunRequest
from reprofix.sandbox.local_backend import LocalSandbox
from reprofix.util import strip_ansi

METRIC = MetricSpec(name="val_accuracy", expected=0.9, tolerance=0.02)


def _run(settings, repo, tmp_path, backend):
    request = RunRequest(local_path=str(repo), command="python train.py", metric=METRIC, max_attempts=3)
    return Orchestrator(settings=settings, request=request, run_dir=tmp_path / "run", sandbox=LocalSandbox(settings), backend=backend,
                        emit=lambda k, d: None).run()


def _by_type(rep, type_):
    return [n for n in rep["graph"]["nodes"] if n["type"] == type_]


def _spine(rep):
    """Follow the edges from the repository node; every node of the graph must be reachable, and nothing may be orphaned."""
    nodes = {n["id"]: n for n in rep["graph"]["nodes"]}
    out: dict[str, list[str]] = {}
    for e in rep["graph"]["edges"]:
        assert e["from"] in nodes and e["to"] in nodes, e
        out.setdefault(e["from"], []).append(e["to"])
    (repo,) = [n for n in nodes.values() if n["type"] == "repo"]
    seen, todo = set(), [repo["id"]]
    while todo:
        cur = todo.pop()
        if cur not in seen:
            seen.add(cur)
            todo.extend(out.get(cur, []))
    return repo, nodes, seen


def test_a_repaired_run_goes_from_the_repository_to_a_verified_outcome(settings, tiny_repo, tmp_path):
    rep = _run(settings, tiny_repo, tmp_path, scripted([hypothesis_reply("h1", "floor division", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)]))
    assert rep["status"] == "verified"
    repo, nodes, seen = _spine(rep)
    assert seen == set(nodes), "every node is connected to the repository"
    assert repo["label"] == "src_repo"                                # the name alone: the node's kind already says "Repository"
    assert repo["status"] == "info" and repo["data"]["command"] == "python train.py" and repo["data"]["command_source"] == "user"
    (outcome,) = _by_type(rep, "outcome")
    assert outcome["status"] == "verified" and "0.9" in outcome["label"]
    (symptom,) = _by_type(rep, "symptom")
    assert symptom["status"] == "resolved", "a symptom the final run no longer shows is resolved, not left open"
    assert not [n for n in nodes.values() if n["status"] == "open"]


def test_a_project_that_already_passes_shows_a_result_not_an_open_symptom(settings, tiny_repo, tmp_path):
    (tiny_repo / "train.py").write_text(TRAIN_PY.replace(BUG_LINE, FIX_LINE))
    rep = _run(settings, tiny_repo, tmp_path, scripted([], []))
    assert rep["status"] == "already_passing"
    repo, nodes, seen = _spine(rep)
    assert seen == set(nodes)
    assert sorted(n["type"] for n in nodes.values()) == ["outcome", "repo", "result"]
    assert not _by_type(rep, "symptom") and not [n for n in nodes.values() if n["status"] == "open"]
    (res,) = _by_type(rep, "result")
    assert res["status"] == "verified" and "reproduced" in res["label"]
    assert _by_type(rep, "outcome")[0]["status"] == "verified"


def test_a_run_whose_diagnosis_never_comes_back_shows_the_failures_and_why_it_stopped(settings, tiny_repo, tmp_path):
    rep = _run(settings, tiny_repo, tmp_path, scripted([json.dumps({"answer": "?"})], []))
    assert rep["status"] == "failed"
    repo, nodes, seen = _spine(rep)
    assert seen == set(nodes)
    failures = _by_type(rep, "failure")
    assert failures and all(f["status"] == "rejected" and "never concluded" in f["label"] for f in failures)
    (outcome,) = _by_type(rep, "outcome")
    assert outcome["status"] == "rejected" and outcome["label"].startswith("Stopped:")
    (symptom,) = _by_type(rep, "symptom")
    assert symptom["status"] == "open", "the problem was never fixed, so it stays open"
    assert not _by_type(rep, "hypothesis") and not _by_type(rep, "experiment")      # nothing is invented for a diagnosis that did not happen


def test_the_graph_rejects_unknown_kinds_and_statuses():
    g = EvidenceGraph()
    for kind in ("repo", "failure", "outcome"):
        g.add(kind, "x", status="info")
    with pytest.raises(AssertionError):
        g.add("mystery", "x")
    with pytest.raises(AssertionError):
        g.add("symptom", "x", status="fixed-ish")
    s = g.add("symptom", "x")
    g.resolve_open_symptoms()
    assert g.nodes[s]["status"] == "resolved"


def test_resolve_only_touches_open_symptoms():
    g = EvidenceGraph()
    sym = g.add("symptom", "a", status="open")
    hyp = g.add("hypothesis", "b", status="rejected")
    res = g.add("result", "c", status="open")
    g.resolve_open_symptoms()
    assert (g.nodes[sym]["status"], g.nodes[hyp]["status"], g.nodes[res]["status"]) == ("resolved", "rejected", "open")


# ----------------------------------------------------------------------------- terminal colour codes in logs
@pytest.mark.parametrize("raw,clean", [
    ("\x1b[31mERROR\x1b[0m failed", "ERROR failed"),
    ("\x1b[1;32mPASSED\x1b[0m", "PASSED"),
    ("\x1b[2K\rprogress 50%", "\rprogress 50%"),
    ("plain text, 100% sure [ok]", "plain text, 100% sure [ok]"),
    ("", ""),
])
def test_strip_ansi_removes_colour_and_cursor_codes_and_nothing_else(raw, clean):
    assert strip_ansi(raw) == clean


def test_sandbox_output_has_no_colour_codes_but_keeps_the_text(settings, tmp_path):
    work = tmp_path / "colour"
    work.mkdir()
    (work / "train.py").write_text('import sys\nprint("\\x1b[31mval_accuracy: 0.9000\\x1b[0m")\nprint("\\x1b[33mwarn\\x1b[0m", file=sys.stderr)\n')
    res = LocalSandbox(settings).run(work, ["python", "train.py"], 30)
    assert res.exit_code == 0
    assert "\x1b" not in res.stdout and "\x1b" not in res.stderr
    assert "val_accuracy: 0.9000" in res.stdout and "warn" in res.stderr
