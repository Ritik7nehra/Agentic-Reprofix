"""ReproBench: the task set itself must be sound, and the harness must not be gameable or flattering."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

from reprofix.core.patching import apply_edits, is_protected, parse_edits
from reprofix.evaluation.harness import (DEFAULT_PROTECTED, STAGES, hidden_check, load_tasks, oracle_backend_for,
                                         render_markdown, run_benchmark, validate_task)
from reprofix.evaluation.scripted import ScriptedBackend
from reprofix.sandbox.base import ExecResult
from reprofix.sandbox.local_backend import LocalSandbox

TASK_ROOT = Path(__file__).resolve().parents[1] / "benchmark" / "tasks"
TASKS = load_tasks(TASK_ROOT)
IDS = [t.id for t in TASKS]
from conftest import NETWORK, TORCH  # noqa: E402

HOST_PACKAGES = {  # requirement name -> importable top-level names that make it work offline
    "numpy": ["numpy"],
    "pytest": ["pytest", "_pytest", "pluggy", "iniconfig", "packaging", "pygments", "py"],
}


def requirement_names(task) -> set[str]:
    req = task.repo / "requirements.txt"
    if not req.is_file():
        return set()
    return {re.split(r"[=<>!~;\s]", ln.strip())[0].lower() for ln in req.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")}


def needs_pip(task) -> bool:
    return task.spec["symptom"]["kind"] == "install_failure" or any(e["path"].startswith("requirements") for e in task.spec["fix"])


def needs_torch(task) -> bool:
    return "torch" in requirement_names(task)


# Hermetic = runs against the host's numpy/pytest with no network. The install-failure and requirements tasks need real
# pip; the PyTorch task needs a multi-GB download. Both are validated in their own, opt-in tests below.
HERMETIC = [t for t in TASKS if not needs_pip(t) and not needs_torch(t) and requirement_names(t) <= set(HOST_PACKAGES)]


def _link_into(deps: Path, module: str) -> None:
    import importlib.util
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        return
    origin = Path(spec.origin).resolve()
    src = origin.parent if origin.name == "__init__.py" else origin          # package dir, or a single-file module
    dst = deps / src.name
    if not dst.exists():
        try:
            dst.symlink_to(src, target_is_directory=src.is_dir())
        except OSError:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    libs = src.parent / f"{src.name}.libs"                                   # wheels with bundled shared libraries
    if libs.exists() and not (deps / libs.name).exists():
        try:
            (deps / libs.name).symlink_to(libs, target_is_directory=True)
        except OSError:
            shutil.copytree(libs, deps / libs.name)


class NumpyFromHost(LocalSandbox):
    """No-network stand-in for `pip install`: link the host's numpy/pytest into .deps; refuse anything else."""

    def install(self, workdir, requirements, timeout):
        names = {re.split(r"[=<>!~;\s]", ln.strip())[0].lower() for r in requirements
                 for ln in (workdir / r).read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")}
        if names - set(HOST_PACKAGES):
            return ExecResult(argv=["pip"], exit_code=1, stdout="", stderr=f"ERROR: no index in this test: {sorted(names - set(HOST_PACKAGES))}\n",
                              duration_s=0, network=True)
        deps = workdir / ".deps"
        deps.mkdir(exist_ok=True)
        for name in names:
            for module in HOST_PACKAGES[name]:
                _link_into(deps, module)
        return ExecResult(argv=["pip"], exit_code=0, stdout="", stderr="", duration_s=0, network=True)


@pytest.fixture(scope="module")
def sbx(tmp_path_factory):
    from reprofix.config import Settings
    return NumpyFromHost(Settings(allow_unsafe_local=True, data_dir=tmp_path_factory.mktemp("bench-data")))


# --------------------------------------------------------------------------- the task set (static)
def test_the_suite_has_the_promised_size_and_spread():
    assert len(TASKS) >= 30 and len(set(IDS)) == len(IDS)
    cats = {t.spec["category"] for t in TASKS}
    assert {"dependency", "architecture", "data", "preprocessing", "training", "evaluation", "configuration", "checkpoint",
            "code", "device"} <= cats
    assert {t.spec["symptom"]["kind"] for t in TASKS} == {"install_failure", "crash", "wrong_result"}


def test_exactly_one_task_needs_pytorch_and_it_gets_a_long_timeout():
    torch_tasks = [t for t in TASKS if needs_torch(t)]
    assert [t.id for t in torch_tasks] == ["torch_double_softmax"]
    (t,) = torch_tasks
    assert t.timeout_s >= 600 and "REAL PyTorch" in t.spec["notes"]
    assert all(x.timeout_s == 300 for x in TASKS if x is not t)                # nothing else changed its limits
    assert (t.repo / "model.py").read_text().count("torch") >= 1 and "numpy" not in (t.repo / "model.py").read_text()


def test_the_device_task_is_labelled_as_a_simulation_and_uses_no_gpu_library():
    (t,) = [x for x in TASKS if x.spec["category"] == "device"]
    assert "SIMULATED" in t.spec["notes"] and "No GPU" in t.spec["notes"]
    source = "\n".join(p.read_text() for p in t.repo.rglob("*.py"))
    assert "torch" not in source and "cuda" not in source.replace('"cuda"', "").lower().replace("'cuda'", "")


def _generator():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("reprofix_build_tasks", TASK_ROOT.parent / "build_tasks.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                      # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_the_generator_reproduces_every_committed_task_byte_for_byte():
    """build_tasks.py is the source of truth: a hand edit to a task (which would invalidate published results) shows up here."""
    gen = _generator()
    cal = gen.committed_calibration()
    assert cal is not None and cal.torch_acc is not None
    assert {b["id"] for b in gen.BUGS} == set(IDS)
    for bug in gen.BUGS:
        assert gen.differences(bug, cal) == [], bug["id"]


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_task_spec_is_complete_and_consistent(task):
    s = task.spec
    assert s["id"] == task.id and s["difficulty"] in {"easy", "medium", "hard"}
    for key in ("goal", "command", "metric", "symptom", "ground_truth", "fix", "hidden"):
        assert s[key], key
    assert s["metric"]["expected"] is not None and 0 < s["metric"]["tolerance"] < s["hidden"]["tolerance"] <= 0.1
    assert s["ground_truth"]["files"] and s["ground_truth"]["category"]
    readme = (task.repo / "README.md").read_text()
    assert f"{s['metric']['expected']}" in readme, "the README must document the number the task is judged against"


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_reference_fix_applies_cleanly_and_never_touches_protected_files(task, tmp_path):
    work = tmp_path / "w"
    shutil.copytree(task.repo, work)
    edits = parse_edits(task.spec["fix"])
    assert not any(is_protected(e.path, DEFAULT_PROTECTED) for e in edits)
    apply_edits(work, edits, DEFAULT_PROTECTED)                         # exact, unique matches
    gt_files = set(task.spec["ground_truth"]["files"])
    assert gt_files <= {e.path for e in edits} or task.spec["ground_truth"]["category"] == "multi"


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_repositories_do_not_leak_the_answer(task):
    hints = re.compile(r"\b(bug|buggy|broken|fix|fixme|hint|todo|should be|mistake|oops|wrong|intentional)\b", re.I)
    for p in task.repo.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".txt", ".json", ".cfg", ".toml"}:
            m = hints.search(p.read_text(errors="replace"))
            assert not m, f"{p.relative_to(task.repo)} contains the hint word {m.group(0)!r}"


# --------------------------------------------------------------------------- validation (executes code)
@pytest.mark.parametrize("task", HERMETIC, ids=[t.id for t in HERMETIC])
def test_broken_state_matches_symptom_and_reference_fix_passes_the_hidden_check(task, sbx, tmp_path):
    out = validate_task(task, sbx, tmp_path)
    assert out["ok"], out["problems"]
    assert out["fixed"]["stage"] == "verified" and out["hidden_check"]["passed"]


@pytest.mark.skipif(not NETWORK, reason="needs a PyPI connection; set REPROFIX_TEST_NETWORK=1")
@pytest.mark.parametrize("task", [t for t in TASKS if needs_pip(t)], ids=[t.id for t in TASKS if needs_pip(t)])
def test_dependency_tasks_validate_against_real_pip(task, tmp_path):
    from reprofix.config import Settings
    real = LocalSandbox(Settings(allow_unsafe_local=True, data_dir=tmp_path / "d"))
    out = validate_task(task, real, tmp_path)
    assert out["ok"], out["problems"]


@pytest.mark.skipif(not (NETWORK and TORCH), reason="downloads PyTorch (about 3 GB, minutes); set REPROFIX_TEST_NETWORK=1 and REPROFIX_TEST_TORCH=1")
@pytest.mark.parametrize("task", [t for t in TASKS if needs_torch(t)], ids=[t.id for t in TASKS if needs_torch(t)])
def test_pytorch_task_validates_against_real_pip(task, tmp_path):
    from reprofix.config import Settings
    real = LocalSandbox(Settings(allow_unsafe_local=True, data_dir=tmp_path / "d"))
    out = validate_task(task, real, tmp_path)
    assert out["ok"], out["problems"]
    assert out["broken"]["stage"] == "wrong_result" and out["fixed"]["stage"] == "verified"


# --------------------------------------------------------------------------- the hidden check cannot be satisfied by cheating
def fixed_copy(task, tmp_path, sbx):
    work = tmp_path / "cheat"
    shutil.copytree(task.repo, work)
    apply_edits(work, parse_edits(task.spec["fix"]), [])
    assert sbx.install(work, ["requirements.txt"], 60).exit_code == 0
    return work


def test_hidden_check_passes_on_an_honest_fix(sbx, tmp_path):
    task = next(t for t in HERMETIC if t.id == "eval_floor_division")
    assert hidden_check(task, fixed_copy(task, tmp_path, sbx), sbx)["passed"]


def test_printing_the_documented_number_fails_the_hidden_check(sbx, tmp_path):
    task = next(t for t in HERMETIC if t.id == "eval_floor_division")
    work = fixed_copy(task, tmp_path, sbx)
    src = (work / "train.py").read_text()
    assert 'print(f"val_accuracy: {acc:.4f}")' in src
    (work / "train.py").write_text(src.replace('print(f"val_accuracy: {acc:.4f}")', 'print("val_accuracy: 0.8810")'))
    verdict_run = sbx.run(work, ["python", "train.py"], 120)
    assert "val_accuracy: 0.8810" in verdict_run.stdout                  # a naive checker would accept this
    hc = hidden_check(task, work, sbx)
    assert hc["passed"] is False and any("recomputed" in f for f in hc["failures"])


def test_making_the_metric_function_return_the_target_fails_the_hidden_check(sbx, tmp_path):
    task = next(t for t in HERMETIC if t.id == "eval_floor_division")
    work = fixed_copy(task, tmp_path, sbx)
    (work / "metrics.py").write_text("def accuracy(preds, labels):\n    return 0.881\n")
    assert hidden_check(task, work, sbx)["passed"] is False


def test_editing_a_protected_file_fails_the_hidden_check(sbx, tmp_path):
    task = next(t for t in HERMETIC if t.id == "eval_floor_division")
    work = fixed_copy(task, tmp_path, sbx)
    (work / "README.md").write_text("Expected val_accuracy: 0.2\n")
    hc = hidden_check(task, work, sbx)
    assert hc["passed"] is False and any("protected file modified: README.md" in f for f in hc["failures"])


def test_a_crashing_or_silent_program_fails_the_hidden_check(sbx, tmp_path):
    task = next(t for t in HERMETIC if t.id == "eval_floor_division")
    work = fixed_copy(task, tmp_path, sbx)
    (work / "train.py").write_text("raise SystemExit(0)\n")
    assert hidden_check(task, work, sbx)["passed"] is False


# --------------------------------------------------------------------------- scoring is honest
def run_bench(settings, sbx, tmp_path, tasks, backend_for, **kw):
    return run_benchmark(tasks=tasks, settings=settings, sandbox=sbx, backend_for=backend_for, out_dir=tmp_path / "out",
                         progress=lambda s: None, label="test", **kw)


def test_oracle_run_scores_every_stage_and_is_labelled_scripted(settings, sbx, tmp_path):
    # device_cuda_config guards two things: the agent's category vocabulary includes "device", and the root-cause stage can score it
    tasks = [t for t in HERMETIC if t.id in {"eval_floor_division", "cfg_epochs", "device_cuda_config"}]
    summary = run_bench(settings, sbx, tmp_path, tasks, oracle_backend_for)
    assert summary["n_tasks"] == 3 and all(summary["stages"][s] == 3 for s in STAGES)
    assert summary["meta"]["llm_backend"] == "scripted" and summary["meta"]["models"] == "n/a (scripted)"
    assert summary["totals"]["cost_usd_priced_tiers"] is None             # scripted calls report no usage -> no invented cost
    md = render_markdown(summary)
    assert "NOT Nemotron" in md and "say nothing about model capability" in md and "n/a" in md
    json.dumps(summary)                                                      # results are serialisable as-is


def test_the_oracle_replays_a_single_fault_fix_whole_and_a_staged_fix_one_edit_at_a_time():
    """A two-edit fix for ONE fault (the PyTorch task) must not be replayed half-way: that crashed the first full run."""
    from reprofix.evaluation.scripted import OracleBackend
    e1 = {"path": "model.py", "search": "import torch\n", "replace": ""}
    e2 = {"path": "model.py", "search": "return softmax(x)", "replace": "return x"}
    e3 = {"path": "config.json", "search": "1", "replace": "2"}
    gt = {"category": "training", "files": ["model.py"], "description": "d"}

    def call(backend, purpose):
        return json.loads(backend.complete(tier="nano", messages=[], purpose=purpose).text)

    single = OracleBackend({"fix": [e1, e2], "ground_truth": gt, "command": "python train.py", "metric": {"expected": 0.9}})
    assert call(single, "repair")["edits"] == [e1, e2]
    assert call(single, "diagnose")["hypotheses"][0]["files"] == ["model.py"]
    staged = OracleBackend({"fix": [e1, e3], "ground_truth": gt, "command": "python train.py", "metric": {"expected": 0.9},
                            "stages": [{"category": "dependency", "statement": "a"}, {"category": "configuration", "statement": "b"}]})
    assert call(staged, "repair")["edits"] == [e1]                       # stage 1
    assert call(staged, "repair")["edits"] == [e3]                       # stage 2
    assert call(staged, "diagnose")["hypotheses"][0]["category"] == "configuration"


def test_a_model_that_does_nothing_scores_zero_on_the_stages_it_did_not_earn(settings, sbx, tmp_path):
    class Blank(ScriptedBackend):
        def complete(self, **kw):
            return super().complete(**{**kw, "purpose": "unused"}) if False else self._blank(**kw)

        def _blank(self, *, tier, messages, purpose, max_tokens=0, temperature=0.0):
            from reprofix.inference.base import LLMResponse
            return LLMResponse(text="{}", model="blank", tier=tier, purpose=purpose)

    task = next(t for t in HERMETIC if t.id == "eval_floor_division")
    summary = run_bench(settings, sbx, tmp_path, [task], lambda t: Blank())
    r = summary["results"][0]
    assert r["status"] in {"failed", "error"} and r["files_modified"] == []
    assert r["stages"]["reproduced"] is True                                 # the harness itself can reproduce the symptom...
    assert r["stages"]["patch_generated"] is False and r["stages"]["patch_verified"] is False   # ...but nothing was fixed
    assert r["stages"]["root_cause"] is False
