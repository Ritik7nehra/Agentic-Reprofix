from pathlib import Path

import pytest

from reprofix.core.verify import assess_progress, extract_failure, make_signature, make_verdict, parse_metric
from reprofix.models import MetricSpec, Verdict
from reprofix.sandbox.base import ExecResult


def res(exit_code=0, stdout="", stderr="", timed_out=False):
    return ExecResult(argv=["python", "train.py"], exit_code=None if timed_out else exit_code, stdout=stdout,
                      stderr=stderr, duration_s=0.1, timed_out=timed_out, network=False, truncated=False)


TB = '''Traceback (most recent call last):
  File "/work/train.py", line 12, in <module>
    main()
  File "/work/model.py", line 30, in forward
    return self.fc(x)
  File "/work/.deps/torch/nn/linear.py", line 114, in forward
    raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (32x784 and 100x10)")
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x784 and 100x10)
'''


def test_extract_failure_uses_deepest_in_repo_frame_and_ignores_deps(tmp_path):
    et, em, loc = extract_failure("", TB, tmp_path)
    assert et == "RuntimeError" and "shapes cannot be multiplied" in em
    assert loc == "model.py:30"  # .deps frame is skipped, deepest repo frame wins


def test_extract_failure_prefers_the_last_traceback_and_final_exception(tmp_path):
    chained = ('Traceback (most recent call last):\n  File "/work/a.py", line 1, in <module>\n    x()\nKeyError: \'a\'\n\n'
               'During handling of the above exception, another exception occurred:\n\n'
               'Traceback (most recent call last):\n  File "/work/b.py", line 7, in <module>\n    y()\nValueError: bad\n')
    assert extract_failure("", chained, tmp_path) == ("ValueError", "bad", "b.py:7")


def test_extract_failure_reports_pip_errors(tmp_path):
    err = "Collecting foo\nERROR: Could not find a version that satisfies the requirement foo==9.9 (from versions: 1.0)\n"
    et, em, loc = extract_failure("", err, tmp_path)
    assert et == "PipInstallError" and "foo==9.9" in em and loc is None


def test_signature_ignores_numbers_and_line_numbers():
    a = make_signature("KeyError", "missing key 3", "train.py:10")
    b = make_signature("KeyError", "missing key 47", "train.py:99")
    assert a == b
    assert a != make_signature("KeyError", "missing key 3", "other.py:10")


SPEC = MetricSpec(name="val_accuracy", expected=0.88, tolerance=0.03)


@pytest.mark.parametrize("out,value", [
    ("epoch 1\nval_accuracy: 0.8123\n", 0.8123), ("VAL_ACCURACY = 0.9", 0.9), ("val_accuracy: 1e-2", 0.01),
    ("val_accuracy: 0.5\nval_accuracy: 0.7\n", 0.7),  # last occurrence wins
    ("val_accuracy: -0.25", -0.25), ("no metric here", None), ("val_accuracy: nan", None),
])
def test_parse_metric(out, value):
    assert parse_metric(out, SPEC) == value


def test_parse_metric_custom_regex_and_no_spec():
    assert parse_metric("top1=71.5%", MetricSpec(name="top1", regex=r"top1=([\d.]+)%")) == 71.5
    assert parse_metric("val_accuracy: 0.9", None) is None


def test_metric_regex_needs_a_capture_group():
    with pytest.raises(ValueError):
        MetricSpec(name="x", regex=r"x=\d+")


def verdict(run, install=None, tests=None, metric=SPEC, tmp=Path(".")):
    return make_verdict(install=install, run=run, tests=tests, metric=metric, workdir=tmp)


def test_verdict_stages(tmp_path):
    assert verdict(None, install=res(1, stderr="ERROR: no matching distribution")).stage == "install_failure"
    assert verdict(res(1, stderr=TB), tmp=tmp_path).stage == "crash"
    assert verdict(res(timed_out=True), tmp=tmp_path).stage == "crash"
    wrong = verdict(res(stdout="val_accuracy: 0.50"))
    assert (wrong.stage, wrong.verified, wrong.metric_ok, wrong.executes) == ("wrong_result", False, False, True)
    assert wrong.gap == pytest.approx(0.38)
    ok = verdict(res(stdout="val_accuracy: 0.8815"))
    assert (ok.stage, ok.verified, ok.metric_ok) == ("verified", True, True)


def test_tolerance_band_is_inclusive_and_symmetric():
    assert verdict(res(stdout="val_accuracy: 0.85")).metric_ok is True   # 0.88 - 0.03
    assert verdict(res(stdout="val_accuracy: 0.91")).metric_ok is True   # 0.88 + 0.03
    assert verdict(res(stdout="val_accuracy: 0.8499")).metric_ok is False
    assert verdict(res(stdout="val_accuracy: 0.9101")).metric_ok is False  # too good is also a discrepancy


def test_missing_metric_line_is_not_a_pass():
    v = verdict(res(stdout="training done"))
    assert v.stage == "wrong_result" and v.metric_ok is False and any("not found" in r for r in v.reasons)


def test_without_expected_value_a_clean_run_only_means_it_executes():
    v = verdict(res(stdout="anything"), metric=MetricSpec(name="val_accuracy"))
    assert v.verified and v.metric_ok is None


def test_failing_tests_block_verification_but_pytest_exit_5_does_not():
    good = res(stdout="val_accuracy: 0.88")
    assert verdict(good, tests=res(1)).stage == "wrong_result"
    assert verdict(good, tests=res(1)).tests_ok is False
    assert verdict(good, tests=res(0)).verified
    assert verdict(good, tests=res(5)).tests_ok is None  # "no tests collected"


def test_missing_pytest_is_not_counted_as_a_test_failure():
    # a repo with tests/ but no pytest in its requirements: `python -S -m pytest` cannot even start
    good = res(stdout="val_accuracy: 0.88")
    no_pytest = res(1, stderr="/usr/bin/python3: No module named pytest")
    v = verdict(good, tests=no_pytest)
    assert v.verified and v.tests_ok is None and any("pytest is not a requirement" in r for r in v.reasons)
    # ...but a genuine failure, or a different import error inside the tests, still blocks verification
    assert verdict(good, tests=res(1, stderr="E   assert 0")).tests_ok is False
    assert verdict(good, tests=res(1, stderr="ModuleNotFoundError: No module named 'numpy'")).tests_ok is False


def V(stage, gap=None, tests_ok=None, sig=""):
    return Verdict(stage=stage, verified=stage == "verified", executes=stage in ("wrong_result", "verified"),
                   gap=gap, tests_ok=tests_ok, signature=sig)


@pytest.mark.parametrize("prev,new,ok", [
    (V("install_failure", sig="a"), V("crash", sig="b"), True),
    (V("crash", sig="a"), V("wrong_result", gap=0.4), True),
    (V("wrong_result", gap=0.4), V("verified", gap=0.01), True),
    (V("verified"), V("wrong_result", gap=0.3), False),               # regression
    (V("crash", sig="a"), V("install_failure", sig="b"), False),
    (V("crash", sig="a"), V("crash", sig="a"), False),                # same failure persists
    (V("crash", sig="a"), V("crash", sig="b"), True),                 # a different failure = progress
    (V("wrong_result", gap=0.40), V("wrong_result", gap=0.30), True),
    (V("wrong_result", gap=0.40), V("wrong_result", gap=0.398), False),  # below the improvement threshold
    (V("wrong_result", gap=0.40), V("wrong_result", gap=0.45), False),
    (V("wrong_result", gap=0.2, tests_ok=False), V("wrong_result", gap=0.2, tests_ok=True), True),
])
def test_assess_progress(prev, new, ok):
    assert assess_progress(prev, new)[0] is ok
