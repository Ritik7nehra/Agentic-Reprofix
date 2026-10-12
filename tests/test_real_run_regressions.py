"""Defects seen in the first real run (a RAG application repository, real Nemotron on Nebius): the program that was run was
`python -m venv venv`, the documented value was read out of "+2 points", the diagnoser used up its turns without ever concluding,
and the progress steps showed green for work that never happened. Each is pinned here without a model or a network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import BUG_LINE, hypothesis_reply, scripted
from reprofix.agents.analyzer import analyze
from reprofix.agents.common import AgentError
from reprofix.agents.diagnoser import MAX_CONCLUDE_RETRIES, diagnose
from reprofix.agents.tools import ToolBox
from reprofix.core.orchestrator import Orchestrator
from reprofix.core.repo import scan_repo
from reprofix.evaluation.scripted import ScriptedBackend
from reprofix.inference import ModelRouter
from reprofix.inference.base import UsageLog
from reprofix.models import MetricSpec, Observation, RunRequest
from reprofix.sandbox.local_backend import LocalSandbox
from reprofix.util import UnsafeCommand, is_setup_command, parse_command, strip_venv_prefix

README = """# Papermind

| Question | Claim | Page |
|---|---|:---:|
| "What were the results?" | 28.4 BLEU on English-to-German (+2 points over recurrent); 41.0 BLEU on English-to-French | 2 |

## Setup

```powershell
python -m venv venv
.\\venv\\Scripts\\activate
pip install -r requirements.txt
```

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```powershell
.\\venv\\Scripts\\streamlit run app.py
.\\venv\\Scripts\\python test_rag.py
.\\venv\\Scripts\\python cli.py --mock
venv/bin/python3 cli.py --mock
```
"""


@pytest.fixture
def rag_repo(tmp_path: Path) -> Path:
    r = tmp_path / "rag"
    r.mkdir()
    (r / "README.md").write_text(README, encoding="utf-8")
    main = "if __name__ == '__main__':\n    print('ok')\n"
    for name in ("app.py", "test_rag.py", "cli.py"):
        (r / name).write_text(main)
    (r / "requirements.txt").write_text("numpy\n")
    return r


# ----------------------------------------------------------------------------- which program is run
def test_setup_commands_are_recognised_and_venv_paths_are_stripped():
    assert is_setup_command(parse_command("python -m venv venv"))
    assert is_setup_command(parse_command("python3 -m pip install -r requirements.txt"))
    assert is_setup_command(parse_command("python -m ensurepip"))
    assert not is_setup_command(parse_command("python -m pytest -q"))
    assert not is_setup_command(parse_command("python train.py --venv x"))
    assert strip_venv_prefix(r".\venv\Scripts\python test_rag.py") == "python test_rag.py"
    assert strip_venv_prefix("venv/bin/python3 cli.py --mock") == "python3 cli.py --mock"
    assert strip_venv_prefix(r"venv\Scripts\python.exe -m pytest") == "python -m pytest"
    assert strip_venv_prefix("python train.py") == "python train.py"
    assert strip_venv_prefix(r".\venv\Scripts\streamlit run app.py") == r".\venv\Scripts\streamlit run app.py"   # not an allowed interpreter


def test_readme_setup_lines_are_not_taken_as_the_program_to_run(rag_repo):
    s = scan_repo(rag_repo)
    assert not any("venv" in c.split()[:3] or " pip " in c for c in s.readme_commands)
    assert s.readme_commands == ["python test_rag.py", "python cli.py --mock", "python3 cli.py --mock"]    # in README order, no duplicates


def _router(script: dict | None = None) -> ModelRouter:
    return ModelRouter(ScriptedBackend(script or {"summarize": ["not json"]}), UsageLog(prices={}))


def test_analyzer_runs_the_project_not_the_environment_setup(rag_repo):
    s = scan_repo(rag_repo)
    a = analyze(_router(), s, rag_repo, RunRequest(local_path=str(rag_repo)))
    assert a["run_command"] == "python test_rag.py" and a["run_command_source"] == "readme"


def test_a_model_proposed_setup_command_is_ignored_but_an_explicit_one_is_kept(rag_repo):
    (rag_repo / "README.md").write_text("# nothing to run here\n")
    s = scan_repo(rag_repo)
    reply = json.dumps({"run_command": "python -m pip install -r requirements.txt"})
    a = analyze(_router({"summarize": [reply]}), s, rag_repo, RunRequest(local_path=str(rag_repo)))
    assert a["run_command"] == "python app.py" and a["run_command_source"] == "entry-point"
    a = analyze(_router({"summarize": [reply]}), s, rag_repo,
                RunRequest(local_path=str(rag_repo), command="python -m venv elsewhere"))
    assert (a["run_command"], a["run_command_source"]) == ("python -m venv elsewhere", "user")      # the person's own command is never replaced


def test_an_unsafe_explicit_command_still_raises(rag_repo):
    with pytest.raises(UnsafeCommand):
        analyze(_router(), scan_repo(rag_repo), rag_repo, RunRequest(local_path=str(rag_repo), command="bash run.sh"))


# ----------------------------------------------------------------------------- which number is documented
def claims(text: str, tmp_path: Path) -> list[tuple[str, float]]:
    d = tmp_path / "c"
    d.mkdir(exist_ok=True)
    (d / "README.md").write_text(text, encoding="utf-8")
    return [(c["metric_hint"], c["value"]) for c in scan_repo(d).readme_claims]


def test_a_difference_is_not_read_as_the_documented_value(rag_repo, tmp_path):
    got = claims("We get 28.4 BLEU on English-to-German (+2 points over recurrent models).", tmp_path)
    assert got == [("bleu", 28.4)]                                    # not ("bleu", 2.0)
    assert scan_repo(rag_repo).readme_claims[0]["value"] == 28.4      # the first number of the real README, in reading order


@pytest.mark.parametrize("text", [
    "BLEU improved by +2 points over the baseline.",
    "accuracy gain of +1.5%",
    "accuracy: -0.3 relative to last year",
    "BLEU 2x better",
    "accuracy ±1.2",
    "f1 − 0.04",
])
def test_deltas_signs_and_multipliers_make_no_claim(text, tmp_path):
    assert claims(text, tmp_path) == []


@pytest.mark.parametrize("text, expected", [
    ("Top-1 accuracy 76.1", [("top-1", 76.1)]),                       # the 1 in "Top-1" is part of the name
    ("It reaches 91% accuracy.", [("accuracy", 0.91)]),
    ("We report 0.87 f1 and accuracy: 0.9", [("f1", 0.87), ("accuracy", 0.9)]),
    ("In 2019 accuracy was measured.", []),                           # a year is not a value
    ("accuracy 87.5% accuracy", [("accuracy", 0.875)]),               # the same number is not listed twice
])
def test_number_first_phrasing_without_false_matches(text, expected, tmp_path):
    assert claims(text, tmp_path) == expected


# ----------------------------------------------------------------------------- the diagnoser must conclude or say why not
def _diag(tmp_path: Path, replies: list, max_tool_steps: int = 5):
    (tmp_path / "train.py").write_text("x = 1\n" + BUG_LINE + "\n")
    backend = ScriptedBackend({"diagnose": list(replies)})
    obs = Observation(kind="crash", summary="boom", exception_type="ValueError", exception_message="boom", signature="s", log_tail="boom")
    kw = dict(request=RunRequest(local_path=str(tmp_path)), analysis={}, observation=obs, file_list=["train.py"], plan={}, rejected=[],
              kept=[], log_text="boom", readme="", escalate=0, max_tool_steps=max_tool_steps)
    try:
        out = diagnose(ModelRouter(backend, UsageLog(prices={})), ToolBox(tmp_path, None), **kw)
    except AgentError as exc:
        out = exc
    return out, backend


INSPECT = json.dumps({"action": "inspect", "calls": [{"tool": "list_dir", "path": "."}]})
CONCLUDE = hypothesis_reply("h1", "integer division", BUG_LINE)


def test_a_model_that_keeps_inspecting_is_told_to_conclude_and_then_does(tmp_path):
    out, backend = _diag(tmp_path, [INSPECT] * 6 + [CONCLUDE])       # 5 allowed rounds, one ignored reminder, then a conclusion
    assert not isinstance(out, Exception) and out.hypotheses[0].id == "h1"
    assert len(backend.calls) == 7
    last_user = [m["content"] for m in backend.calls[5]["messages"] if m["role"] == "user"][-1]
    assert "Do not inspect again" in last_user                         # the reminder names exactly what to send
    second = [m["content"] for m in backend.calls[1]["messages"] if m["role"] == "user"][-1]
    assert "Inspection steps left: 4" in second                         # and the model is told how many rounds remain


def test_an_unknown_reply_shape_gets_a_correction_instead_of_a_repeat_of_the_same_prompt(tmp_path):
    out, backend = _diag(tmp_path, [json.dumps({"answer": "it is the division"}), CONCLUDE])
    assert not isinstance(out, Exception) and out.hypotheses
    assert backend.calls[1]["messages"] != backend.calls[0]["messages"]
    assert "not understood" in backend.calls[1]["messages"][-1]["content"]


def test_a_model_that_never_concludes_is_cut_off_with_a_clear_error_and_a_bounded_cost(tmp_path):
    out, backend = _diag(tmp_path, [INSPECT])                          # every reply is another inspect
    assert isinstance(out, AgentError)
    assert "never concluded" in str(out) and "'inspect'" in str(out) and "reminders" in str(out)
    assert len(backend.calls) == 5 + 1 + MAX_CONCLUDE_RETRIES          # the budget is exact: no unbounded loop on the paid tier


def test_inspect_calls_are_not_made_once_the_model_has_been_told_to_stop(tmp_path):
    out, backend = _diag(tmp_path, [INSPECT] * 2 + [CONCLUDE], max_tool_steps=1)
    assert not isinstance(out, Exception)
    assert len(backend.calls) == 3                                      # 1 round, a reminder that is ignored, a second reminder that works


# ----------------------------------------------------------------------------- progress steps tell the truth
def test_steps_show_what_did_not_happen_when_no_diagnosis_came_back(settings, tiny_repo, tmp_path):
    garbage = [json.dumps({"answer": "?"})]
    b = scripted(garbage, [])
    events: list[tuple[str, dict]] = []
    request = RunRequest(local_path=str(tiny_repo), command="python train.py", max_attempts=4,
                         metric=MetricSpec(name="val_accuracy", expected=0.9, tolerance=0.02))
    orch = Orchestrator(settings=settings, request=request, run_dir=tmp_path / "run", sandbox=LocalSandbox(settings), backend=b,
                        emit=lambda k, d: events.append((k, d)))
    rep = orch.run()
    assert rep["status"] != "verified" and "diagnosis kept failing" in rep["stop_reason"] and "never concluded" in rep["stop_reason"]
    last = {}
    for kind, d in events:
        if kind == "step":
            last[d["id"]] = d
    assert last["analyze"]["status"] == "failed" and "never concluded" in last["analyze"]["detail"]
    assert last["hypothesize"]["status"] == "failed"
    assert last["test"]["status"] == "skipped" and last["test"]["detail"] == "no fix was tested"
    assert last["verify"]["status"] == "failed"
    assert not [k for k, d in events if k == "step" and d["status"] == "running" and d["id"] in ("analyze", "hypothesize", "test")
                and (d is last[d["id"]])]                                # nothing is left spinning at the end
    assert len(b.calls) < 4 * (5 + 1 + MAX_CONCLUDE_RETRIES) + 4        # two failed attempts, then the run stops


def test_steps_still_end_green_after_a_normal_fix(settings, tiny_repo, tmp_path):
    from conftest import FIX_LINE, repair_reply
    b = scripted([hypothesis_reply("h1", "floor division", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    events: list[tuple[str, dict]] = []
    request = RunRequest(local_path=str(tiny_repo), command="python train.py", max_attempts=4,
                         metric=MetricSpec(name="val_accuracy", expected=0.9, tolerance=0.02))
    rep = Orchestrator(settings=settings, request=request, run_dir=tmp_path / "run", sandbox=LocalSandbox(settings), backend=b,
                       emit=lambda k, d: events.append((k, d))).run()
    assert rep["status"] == "verified"
    last = {d["id"]: d["status"] for k, d in events if k == "step"}
    assert (last["analyze"], last["hypothesize"], last["test"], last["verify"]) == ("done", "done", "done", "done")
