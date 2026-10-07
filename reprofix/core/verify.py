"""Deterministic verification. No LLM decides whether something is fixed."""
from __future__ import annotations

import re
from pathlib import Path

from ..claims.check import claim_missing_reason, claim_value
from ..models import MetricSpec, Observation, Verdict
from ..sandbox.base import ExecResult
from ..util import tail_lines

# |0.85 - 0.88| is 0.030000000000000027 in binary floating point; a value exactly on the edge of the
# documented band must count as inside it.
TOLERANCE_EPS = 1e-9

STAGE_ORDER = {"install_failure": 0, "crash": 1, "wrong_result": 2, "verified": 3}

_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
_EXC_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)?)(?::\s*(.*))?$")


def pytest_not_installed(tests: ExecResult | None) -> bool:
    """`python -S -m pytest` fails like this when the repo's own requirements do not include pytest.
    That says nothing about the repo's tests, so it must not count as a test failure."""
    return bool(tests) and tests.exit_code not in (0, 5) and bool(re.search(r"No module named '?pytest'?", tests.stderr))


def _rel(path: str, workdir: Path) -> str | None:
    """Map a traceback path to a repo-relative path; None if it is outside the repo or in .deps."""
    p = path.replace("\\", "/")
    for prefix in (workdir.resolve().as_posix() + "/", "/work/"):
        if p.startswith(prefix):
            rel = p[len(prefix):]
            return None if rel.startswith((".deps/", ".home/")) else rel
    if not p.startswith("/") and not re.match(r"^[A-Za-z]:", p) and not p.startswith("<"):
        return p
    return None


def extract_failure(stdout: str, stderr: str, workdir: Path) -> tuple[str | None, str | None, str | None]:
    """(exception_type, message, 'path:line' of deepest in-repo frame) from a Python traceback."""
    text = stderr if "Traceback (most recent call last)" in stderr else (stderr + "\n" + stdout)
    idx = text.rfind("Traceback (most recent call last):")
    if idx == -1:
        # Not a traceback: surface pip-style errors or the last stderr line.
        for ln in reversed(stderr.splitlines()):
            if ln.strip().startswith("ERROR:"):
                return "PipInstallError", ln.strip()[len("ERROR:"):].strip(), None
        last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), None)
        return (None, last, None)
    tb = text[idx:]
    location = None
    for m in _FRAME.finditer(tb):
        rel = _rel(m.group(1), workdir)
        if rel:
            location = f"{rel}:{m.group(2)}"
    exc_type = exc_msg = None
    for ln in tb.splitlines()[1:]:
        if ln and not ln[0].isspace() and not ln.startswith(("During handling", "The above exception", "Traceback", "^")):
            m = _EXC_LINE.match(ln)
            if m:
                exc_type, exc_msg = m.group(1), (m.group(2) or "")
    return exc_type, exc_msg, location


def make_signature(exc_type: str | None, exc_msg: str | None, location: str | None) -> str:
    msg = re.sub(r"\d+", "N", (exc_msg or "")[:100])
    loc = re.sub(r":\d+$", "", location or "")  # line numbers shift as code is edited
    return f"{exc_type or 'Failure'}|{loc}|{msg}"


def parse_metric(stdout: str, spec: MetricSpec | None) -> float | None:
    if spec is None:
        return None
    if spec.claim is not None:       # a paper claim: read by the same matcher that fills the report card
        return claim_value(stdout, spec.claim)
    matches = spec.pattern().findall(stdout)
    if not matches:
        return None
    try:
        return float(matches[-1] if isinstance(matches[-1], str) else matches[-1][0])
    except ValueError:
        return None


def make_verdict(*, install: ExecResult | None, run: ExecResult | None, tests: ExecResult | None,
                 metric: MetricSpec | None, workdir: Path) -> Verdict:
    reasons: list[str] = []
    if install is not None and (install.exit_code != 0 or install.timed_out):
        et, em, loc = extract_failure(install.stdout, install.stderr, workdir)
        reasons.append("dependency installation failed" + (" (timeout)" if install.timed_out else ""))
        return Verdict(stage="install_failure", verified=False, executes=False, reasons=reasons,
                       signature=make_signature(et, em, loc))
    assert run is not None
    if run.exit_code != 0 or run.timed_out:
        et, em, loc = extract_failure(run.stdout, run.stderr, workdir)
        reasons.append("command timed out" if run.timed_out else f"command exited with code {run.exit_code}")
        return Verdict(stage="crash", verified=False, executes=False, reasons=reasons,
                       signature=make_signature("Timeout" if run.timed_out else et, em, loc))
    value = parse_metric(run.stdout, metric)
    metric_ok: bool | None = None
    gap: float | None = None
    if metric is not None and metric.expected is not None:
        if value is None:
            metric_ok = False
            reasons.append(f"metric '{metric.name}' not found in output"
                           + (f": {claim_missing_reason(run.stdout, metric.claim)}" if metric.claim is not None else ""))
        else:
            gap = abs(value - metric.expected)
            metric_ok = gap <= metric.tolerance + TOLERANCE_EPS
            reasons.append(
                f"{metric.name}={value:.4f} vs expected {metric.expected:.4f} (|gap|={gap:.4f}, tolerance {metric.tolerance})"
            )
    tests_ok: bool | None = None
    if pytest_not_installed(tests):
        reasons.append("tests not run: pytest is not a requirement of this repository")
    elif tests is not None and tests.exit_code != 5:  # 5 = pytest collected nothing
        tests_ok = tests.exit_code == 0 and not tests.timed_out
        reasons.append("tests passed" if tests_ok else "tests failed")
    verified = metric_ok in (True, None) and tests_ok in (True, None)
    sig = ""
    if not verified:
        sig = make_signature("WrongResult", "metric" if metric_ok is False else "tests", None)
    return Verdict(stage="verified" if verified else "wrong_result", verified=verified, executes=True,
                   metric_value=value, metric_ok=metric_ok, tests_ok=tests_ok, gap=gap, signature=sig, reasons=reasons)


def assess_progress(prev: Verdict, new: Verdict, min_gap_improvement: float = 0.005) -> tuple[bool, str]:
    """Did an experiment move us forward? Used to keep or revert a patch."""
    a, b = STAGE_ORDER[prev.stage], STAGE_ORDER[new.stage]
    if b > a:
        return True, f"advanced from {prev.stage} to {new.stage}"
    if b < a:
        return False, f"regressed from {prev.stage} to {new.stage}"
    if new.stage == "verified":
        return True, "still verified"
    if new.stage == "wrong_result":
        if prev.gap is not None and new.gap is not None and prev.gap - new.gap >= min_gap_improvement:
            return True, f"metric gap shrank from {prev.gap:.4f} to {new.gap:.4f}"
        if prev.tests_ok is False and new.tests_ok is True:
            return True, "tests now pass"
        return False, "result did not improve"
    # same failing stage (install_failure / crash)
    if new.signature != prev.signature:
        return True, "the previous failure is gone; a different one remains"
    return False, "same failure persists"


def make_observation(verdict: Verdict, *, install: ExecResult | None, run: ExecResult | None,
                     tests: ExecResult | None, metric: MetricSpec | None, workdir: Path) -> Observation:
    expected = metric.expected if metric else None
    if verdict.stage == "install_failure":
        assert install is not None
        et, em, loc = extract_failure(install.stdout, install.stderr, workdir)
        return Observation(kind="install_failure", summary="pip could not install the requirements",
                           exception_type=et, exception_message=em, signature=verdict.signature,
                           log_tail=tail_lines(install.stderr or install.stdout, 40))
    assert run is not None
    if verdict.stage == "crash":
        et, em, loc = extract_failure(run.stdout, run.stderr, workdir)
        kind = "timeout" if run.timed_out else "crash"
        return Observation(kind=kind, summary=f"{et or 'failure'}: {em or ''}".strip(), exception_type=et,
                           exception_message=em, location=loc, signature=verdict.signature,
                           log_tail=tail_lines((run.stderr or "") + "\n" + (run.stdout or ""), 60))
    if verdict.metric_ok is False:
        return Observation(kind="metric_gap", summary="; ".join(verdict.reasons), metric_value=verdict.metric_value,
                           expected=expected, signature=verdict.signature, log_tail=tail_lines(run.stdout, 40))
    out = tests.stdout + "\n" + tests.stderr if tests else ""
    return Observation(kind="test_failure", summary="; ".join(verdict.reasons), signature=verdict.signature,
                       metric_value=verdict.metric_value, expected=expected, log_tail=tail_lines(out, 60))
