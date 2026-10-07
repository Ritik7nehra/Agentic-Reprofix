import json
import os
import shutil
import subprocess

import pytest

from reprofix.core.patching import (Edit, PatchError, apply_edits, diff_hardcoding_findings, is_protected,
                                    literal_hardcoding_findings, make_diff, parse_edits, revert)

PROTECTED = ["tests/**", "test_*.py", "**/test_*.py", "README*", "LICENSE*"]


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "work"
    (w / "pkg").mkdir(parents=True)
    (w / "train.py").write_text("lr = 0.1\nepochs = 3\nprint(lr)\n")
    (w / "pkg" / "m.py").write_text("x = 1\nx = 1\n")
    (w / "tests").mkdir()
    (w / "tests" / "test_a.py").write_text("def test_a(): pass\n")
    (w / "README.md").write_text("docs\n")
    return w


@pytest.mark.parametrize("raw", [None, [], {}, [1], [{"path": 3}], [{"path": "a"}], [{"path": "a", "search": "", "replace": "x"}],
                                 [{"path": "a", "search": "x", "replace": "x"}], [{"path": "a", "search": "x"}]])
def test_parse_edits_rejects_malformed(raw):
    with pytest.raises(PatchError):
        parse_edits(raw)


def test_parse_edits_ok():
    edits = parse_edits([{"path": "a.py", "search": "x", "replace": "y"}, {"path": "b.py", "create": "print(1)\n"}])
    assert [e.describe() for e in edits] == ["edit a.py", "create b.py"]


@pytest.mark.parametrize("path,expected", [
    ("tests/test_a.py", True), ("tests/sub/x.py", True), ("test_x.py", True), ("pkg/test_y.py", True), ("README.md", True),
    ("LICENSE", True), ("./tests/a.py", True), ("train.py", False), ("pkg/m.py", False), ("mytests/a.py", False),
    ("tests_helper.py", False),
])
def test_is_protected(path, expected):
    assert is_protected(path, PROTECTED) is expected


def test_is_protected_does_not_eat_dotfile_names():
    # regression: lstrip("./") used to turn ".env" into "env" and miss the pattern
    assert is_protected(".env", [".env"]) is True
    assert is_protected("./.env", [".env"]) is True
    assert is_protected("env", [".env"]) is False


def test_apply_exact_replace_and_revert(work):
    applied = apply_edits(work, [Edit("train.py", search="lr = 0.1", replace="lr = 0.01")], PROTECTED)
    assert (work / "train.py").read_text().startswith("lr = 0.01")
    assert applied.changed == ["train.py"]
    revert(work, applied)
    assert (work / "train.py").read_text() == "lr = 0.1\nepochs = 3\nprint(lr)\n"


def test_search_must_match_exactly_once(work):
    with pytest.raises(PatchError, match="not found"):
        apply_edits(work, [Edit("train.py", search="lr = 0.2", replace="lr = 1")], PROTECTED)
    with pytest.raises(PatchError, match="2 places"):
        apply_edits(work, [Edit("pkg/m.py", search="x = 1", replace="x = 2")], PROTECTED)
    with pytest.raises(PatchError, match="not found"):  # whitespace differences are not forgiven
        apply_edits(work, [Edit("train.py", search="lr  =  0.1", replace="lr = 1")], PROTECTED)


def test_not_found_message_hints_the_closest_line(work):
    with pytest.raises(PatchError, match="Closest line"):
        apply_edits(work, [Edit("train.py", search="epochs = 4", replace="epochs = 5")], PROTECTED)


def test_failed_patch_leaves_nothing_changed(work):
    before = {p: p.read_text() for p in work.rglob("*.py")}
    edits = [Edit("train.py", search="lr = 0.1", replace="lr = 9"), Edit("pkg/m.py", search="nope", replace="y"),
             Edit("new.py", create="print(1)\n")]
    with pytest.raises(PatchError):
        apply_edits(work, edits, PROTECTED)
    assert {p: p.read_text() for p in work.rglob("*.py")} == before
    assert not (work / "new.py").exists()


def test_create_new_file_and_refuse_overwrite(work):
    applied = apply_edits(work, [Edit("pkg/new.py", create="y = 2\n")], PROTECTED)
    assert (work / "pkg" / "new.py").read_text() == "y = 2\n"
    revert(work, applied)
    assert not (work / "pkg" / "new.py").exists()
    with pytest.raises(PatchError, match="already exists"):
        apply_edits(work, [Edit("train.py", create="boom")], PROTECTED)


@pytest.mark.parametrize("path", ["tests/test_a.py", "README.md", "./tests/test_a.py", "../work/tests/test_a.py", "tests/new_test.py"])
def test_protected_files_cannot_be_edited(work, path):
    edit = Edit(path, create="x") if path.endswith("new_test.py") else Edit(path, search="def test_a(): pass", replace="def test_a(): assert 0")
    with pytest.raises(PatchError):
        apply_edits(work, [edit], PROTECTED)
    assert (work / "tests" / "test_a.py").read_text() == "def test_a(): pass\n"


@pytest.mark.parametrize("path", ["../escape.py", "/etc/cron.d/x", "pkg/../../escape.py", ".git/config", ".deps/numpy/__init__.py", "__pycache__/x.py"])
def test_paths_cannot_escape_or_touch_internals(work, path):
    with pytest.raises((PatchError, ValueError)):
        apply_edits(work, [Edit(path, create="x = 1\n")], PROTECTED)
    assert not (work.parent / "escape.py").exists()


def test_symlinks_cannot_be_used_to_write_outside(work, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n")
    try:
        (work / "link.py").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported or permitted on this system")
    with pytest.raises((PatchError, ValueError)):
        apply_edits(work, [Edit("link.py", search="x = 1", replace="x = 2")], PROTECTED)
    assert outside.read_text() == "x = 1\n"


def test_too_many_files_in_one_patch(work):
    edits = [Edit(f"f{i}.py", create="x\n") for i in range(6)]
    with pytest.raises(PatchError, match="too many"):
        apply_edits(work, edits, PROTECTED)


def test_non_utf8_files_are_refused(work):
    (work / "blob.py").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(PatchError, match="UTF-8"):
        apply_edits(work, [Edit("blob.py", search="bad", replace="good")], PROTECTED)


# --------------------------------------------------------------------------- hardcoding guard
@pytest.mark.parametrize("replace", [
    "acc = 0.9", "print('val_accuracy:', 0.90)", "val_accuracy = 0.900", "accuracy = 90", "metric = 90.0",
])
def test_hardcoded_expected_value_is_flagged(replace):
    assert literal_hardcoding_findings([Edit("train.py", search="x", replace=replace)], 0.9, "val_accuracy")


@pytest.mark.parametrize("replace", [
    "accuracy = correct / len(preds)", "momentum = 0.9", "lr = 0.09", "epochs = 90", "accuracy = 0.95",
    "print(f'val_accuracy: {acc:.4f}')",
])
def test_legitimate_edits_are_not_flagged(replace):
    assert literal_hardcoding_findings([Edit("train.py", search="x", replace=replace)], 0.9, "val_accuracy") == []


def test_hardcoding_guard_checks_new_files_and_is_off_without_expected_value():
    assert literal_hardcoding_findings([Edit("a.py", create="print('val_accuracy: 0.9')\n")], 0.9, "val_accuracy")
    assert literal_hardcoding_findings([Edit("a.py", search="x", replace="acc = 0.9")], None, "val_accuracy") == []


def test_report_check_uses_the_same_line_level_rule():
    legit = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-momentum = 0.5\n+momentum = 0.9\n"
    cheat = "diff --git a/t.py b/t.py\n--- a/t.py\n+++ b/t.py\n@@ -1 +1 @@\n-acc = compute()\n+acc = 0.9\n"
    assert diff_hardcoding_findings(legit, 0.9, "val_accuracy") == []
    assert diff_hardcoding_findings(cheat, 0.9, "val_accuracy")


# --------------------------------------------------------------------------- diffs
def test_make_diff_is_a_real_unified_diff_that_applies(work, tmp_path):
    orig = tmp_path / "orig"
    shutil.copytree(work, orig)
    apply_edits(work, [Edit("train.py", search="lr = 0.1", replace="lr = 0.01"), Edit("pkg/new.py", create="y = 2\n")], PROTECTED)
    diff, files = make_diff(orig, work, ["train.py", "pkg/new.py", "pkg/m.py"])
    assert {f["path"]: (f["status"], f["added"], f["removed"]) for f in files} == {
        "train.py": ("modified", 1, 1), "pkg/new.py": ("added", 1, 0)}   # unchanged m.py is omitted
    assert "diff --git a/train.py b/train.py" in diff and "+lr = 0.01" in diff and "new file mode" in diff
    (tmp_path / "p.diff").write_bytes(diff.encode("utf-8"))
    scratch = tmp_path / "scratch"
    shutil.copytree(orig, scratch)
    cmd = ["patch", "-p1", "-s", "-i", str(tmp_path / "p.diff")]
    if shutil.which("patch") and os.name == "nt":
        cmd.insert(2, "--binary")
    r = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (scratch / "train.py").read_text() == (work / "train.py").read_text()
    assert (scratch / "pkg" / "new.py").read_text() == "y = 2\n"


def test_make_diff_handles_missing_trailing_newline(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "f.py").write_text("x = 1")
    (b / "f.py").write_text("x = 2")
    diff, _ = make_diff(a, b, ["f.py"])
    assert "\\ No newline at end of file" in diff


def test_make_diff_skips_binary_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "w.bin").write_bytes(b"\x00\x01")
    (b / "w.bin").write_bytes(b"\x00\x02")
    assert make_diff(a, b, ["w.bin"]) == ("", [])
