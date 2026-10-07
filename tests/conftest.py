"""Shared fixtures. Nothing here needs network access, GPUs, Docker or an API key."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if os.name == "nt":
    for extra in [r"C:\Program Files\Git\usr\bin", r"C:\Program Files (x86)\Git\usr\bin"]:
        if os.path.exists(extra) and extra not in os.environ.get("PATH", ""):
            os.environ["PATH"] = extra + os.pathsep + os.environ.get("PATH", "")

from reprofix.config import Settings  # noqa: E402
from reprofix.evaluation.scripted import ScriptedBackend  # noqa: E402

# Tests that run a real `pip install` are skipped unless REPROFIX_TEST_NETWORK=1.
NETWORK = os.environ.get("REPROFIX_TEST_NETWORK") == "1"
# The PyTorch benchmark task downloads about 3 GB and unpacks about 5 GB: opt in separately (needs NETWORK too).
TORCH = os.environ.get("REPROFIX_TEST_TORCH") == "1"

TRAIN_PY = '''\
preds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
labels = [1] * 10
correct = sum(p == l for p, l in zip(preds, labels))
accuracy = correct // len(preds)
print(f"val_accuracy: {accuracy:.4f}")
'''
BUG_LINE = "accuracy = correct // len(preds)"
FIX_LINE = "accuracy = correct / len(preds)"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", sandbox="local", allow_unsafe_local=True, allow_local_paths=True)


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A one-file project whose reported accuracy is wrong (0.0 instead of the documented 0.9)."""
    repo = tmp_path / "src_repo"
    repo.mkdir()
    (repo / "train.py").write_text(TRAIN_PY)
    (repo / "README.md").write_text("# toy\n\nRun `python train.py`. Expected val_accuracy: 0.90\n")
    return repo


def hypothesis_reply(hid: str, statement: str, quote: str, *, category: str = "code", evidence: list | None = None) -> str:
    return json.dumps({
        "action": "conclude", "summary": "scripted", "recommended": hid,
        "hypotheses": [{
            "id": hid, "statement": statement, "category": category, "model_confidence": "high", "files": ["train.py"],
            "evidence": evidence if evidence is not None else [{"kind": "code", "file": "train.py", "quote": quote, "note": "n"}],
            "test": "rerun",
        }],
    })


def repair_reply(search: str, replace: str, rationale: str = "fix") -> str:
    return json.dumps({"edits": [{"path": "train.py", "search": search, "replace": replace}],
                       "rationale": rationale, "expected_effect": "metric moves"})


def scripted(diagnoses: list[str], repairs: list[str]) -> ScriptedBackend:
    """Backend whose agents answer from queues. A wrong-result bug has no traceback, so the pipeline
    asks for 'diagnose_behavioral'; both purposes get the same queue so tests do not depend on that."""
    return ScriptedBackend({
        "summarize": [json.dumps({"framework": "numpy", "task": "toy", "run_command": "python train.py", "expected_metric": None})],
        "plan": [json.dumps({"suspect_areas": [], "first_step": "run it"})],
        "diagnose": list(diagnoses), "diagnose_behavioral": list(diagnoses),
        "repair": list(repairs),
        "review": [json.dumps({"summary": "ok", "concerns": [], "hardcoding_suspected": False})],
    })
