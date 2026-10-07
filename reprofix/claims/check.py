"""Compare a paper's numbers with what the program actually printed.

No model is involved. A claim ("test accuracy 76.4%") is matched to a line of the program's stdout ("test_acc: 0.764") by
name, the printed number is read on the claim's scale, and the difference is compared with a tolerance. When the match is not
clear (two outputs fit, none fits, the program crashed) the answer is "not measured" with the reason, never a guess.

Rules, all of them visible in the report:
  * only `name: number` and `name = number` pairs in stdout count (what a training script prints); tables are not parsed;
  * the LAST value printed under a name is the measurement (the end of training, not epoch 1), and `nan`, `inf`, `N/A` or a
    number too large for a float count as values: a run that ends in `accuracy: nan` measured nothing;
  * a printed name must contain every word of the claimed one, and may add only words that do not make it another quantity
    (`cifar10_accuracy` is accuracy; `accuracy_gap`, `worst_group_accuracy` and `accuracy@5` are not);
  * a run that exited non-zero or timed out measures nothing, even if it printed numbers before it died;
  * a number followed by % is a percentage; otherwise a number up to 1 is read as a fraction and one between 1 and 100 as a
    percentage (so 0.764 and 76.4 both mean 76.4%) for accuracy-like metrics. Loss, perplexity and the like are compared as printed;
  * a claim that names no split ignores outputs labelled "train", and prefers an unlabelled output over a labelled one.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ..models import PaperClaim
from .names import extras_ok, is_bounded, tokens

TOLERANCE_EPS = 1e-9
MAX_LINES = 5000          # the last lines of stdout; a training log can be long
MAX_LINE_CHARS = 400      # longer lines are cut: the regexes below must stay linear on adversarial output
MAX_NAME_WORDS = 5

DEFAULT_TOLERANCE_PERCENT = 1.0     # percentage points
DEFAULT_TOLERANCE_FRACTION = 0.01
DEFAULT_TOLERANCE_RELATIVE = 0.01   # raw metrics (loss, perplexity): 1% of the claimed value

# Atomic: a number is read whole or not at all. (Without the atomic group the pattern backtracks into "12.5 hours" -> 12. and
# takes quadratic time on a long run of digits.)
NUM = r"(?>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
# `name: 0.91`, `name = 91.2%`; also `name: nan` / `inf` / `N/A`, which are recorded as "not a number" so that they can be the last
# value printed under a name (a diverged run must not be read as the epoch before it).
_VALUE = re.compile(rf"[:=]\s*(?:(?P<num>{NUM})\s*(?P<pct>%)?(?![\w/]|\.\d)|(?P<bad>[+-]?(?:nan|inf(?:inity)?|n/a|none|null))(?!\w))", re.I)
_WORD = re.compile(r"^[A-Za-z_][\w\-./@]*$")
_STRIP = " \t,;:()[]{}<>\"'|"


@dataclass
class Output:
    """One `name: number` pair found in the program's output."""

    name: str
    toks: frozenset[str]
    qualifier: str | None
    value: float
    pct: bool
    line_no: int
    text: str

    def key(self) -> tuple[frozenset[str], str | None]:
        return self.toks, self.qualifier


@dataclass
class Measurement:
    status: str                     # "found" | "missing" | "ambiguous"
    output: Output | None = None
    candidates: tuple[str, ...] = ()
    reason: str = ""


# --------------------------------------------------------------------------- reading the output
def _trailing_name(segment: str) -> str:
    words: list[str] = []
    for raw in reversed(segment.split()):
        w = raw.strip(_STRIP)
        if not w or not _WORD.match(w):
            break
        words.append(w)
        if len(words) >= MAX_NAME_WORDS:
            break
    return " ".join(reversed(words))


def parse_outputs(stdout: str) -> list[Output]:
    outs: list[Output] = []
    lines = stdout.splitlines()
    first = max(0, len(lines) - MAX_LINES)
    for i, line in enumerate(lines[first:], start=first + 1):
        line = line[:MAX_LINE_CHARS]
        prev = 0
        for m in _VALUE.finditer(line):
            name = _trailing_name(line[prev:m.start()])
            prev = m.end()
            if not name:
                continue
            try:
                value = float(m.group("num")) if m.group("num") is not None else math.nan
            except ValueError:
                continue
            toks, qual = tokens(name)
            if not toks:
                continue
            outs.append(Output(name=name, toks=toks, qualifier=qual, value=value, pct=bool(m.group("pct")), line_no=i,
                               text=line.strip()[:200]))
    return outs


# --------------------------------------------------------------------------- scales
def resolve_unit(claim: PaperClaim) -> str:
    if claim.unit:
        return claim.unit
    toks, _ = tokens(claim.metric)
    if is_bounded(toks):
        if 0 <= claim.value <= 1:
            return "fraction"
        if 1 < claim.value <= 100:
            return "percent"
    return "raw"


def default_tolerance(unit: str, value: float) -> float:
    if unit == "percent":
        return DEFAULT_TOLERANCE_PERCENT
    if unit == "fraction":
        return DEFAULT_TOLERANCE_FRACTION
    return max(abs(value) * DEFAULT_TOLERANCE_RELATIVE, 1e-12)


def comparison(claim: PaperClaim) -> tuple[float, float, str]:
    """(claimed value, tolerance, unit) on the scale the comparison happens on: a fraction for percent and fraction
    claims, the number itself for raw ones. `unit` is the claim's own unit, used when showing numbers."""
    unit = resolve_unit(claim)
    tol_written = claim.tolerance if claim.tolerance is not None else default_tolerance(unit, claim.value)
    if unit == "percent":
        return claim.value / 100.0, tol_written / 100.0, unit
    return claim.value, tol_written, unit


def read_value(o: Output, unit: str) -> tuple[float, str]:
    """The printed number on the comparison scale, and how it was read ("fraction" | "percent" | "as printed")."""
    if unit == "raw":
        return o.value, "as printed"
    if o.pct:
        return o.value / 100.0, "percent"
    if 0 <= o.value <= 1:
        return o.value, "fraction"
    if 1 < o.value <= 100:
        return o.value / 100.0, "percent"
    return o.value, "as printed"


def to_claim_unit(x: float, unit: str) -> float:
    return x * 100.0 if unit == "percent" else x


# --------------------------------------------------------------------------- matching
def measure(claim: PaperClaim, outputs: list[Output]) -> Measurement:
    ctoks, cqual = tokens(claim.metric)
    cqual = claim.qualifier or cqual
    if not ctoks:
        return Measurement("missing", reason=f"'{claim.metric}' is not a metric name ReproFix can match")
    cands = [o for o in outputs if ctoks <= o.toks and extras_ok(ctoks, o.toks) and (o.qualifier != "train" or cqual == "train")]
    if cqual:
        same = [o for o in cands if o.qualifier == cqual]
        cands = same or [o for o in cands if o.qualifier is None]
    else:
        plain = [o for o in cands if o.qualifier is None]
        cands = plain or cands
    if not cands:
        seen = sorted({o.name for o in outputs})[:8]
        hint = f" (the output has: {', '.join(seen)})" if seen else " (the output has no `name: number` lines)"
        return Measurement("missing", reason=f"no output line is named like '{claim.metric}'" + hint)
    exact = [o for o in cands if o.toks == ctoks]
    cands = exact or cands
    names = sorted({o.name.lower() for o in cands})
    if len({o.key() for o in cands}) > 1:
        return Measurement("ambiguous", candidates=tuple(names),
                           reason=f"several outputs fit '{claim.metric}': {', '.join(names)}. Name the one you mean "
                                  f"(e.g. --claim {names[0]}={claim.value:g})")
    last = cands[-1]
    if not math.isfinite(last.value):                  # nan, inf, N/A, 1e999: the last value counts, and it is not a result
        return Measurement("missing", candidates=tuple(names),
                           reason=f"the last value printed for '{last.name}' is not a finite number (line {last.line_no}: "
                                  f"{last.text[:80]!r}), so no result was read")
    return Measurement("found", output=last, candidates=tuple(names))


def claim_value(stdout: str, claim: PaperClaim) -> float | None:
    """The value the repair loop compares with the claim (comparison scale), or None when it cannot be measured."""
    m = measure(claim, parse_outputs(stdout))
    if m.status != "found" or m.output is None:
        return None
    return read_value(m.output, resolve_unit(claim))[0]


def claim_missing_reason(stdout: str, claim: PaperClaim) -> str:
    m = measure(claim, parse_outputs(stdout))
    return m.reason or "not found in the output"


# --------------------------------------------------------------------------- judging one run
def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 6)


def check_run(claim: PaperClaim, rec: Any, *, no_run_reason: str = "the program did not run") -> dict:
    """`rec` is an ExecRecord (or anything with exit_code / timed_out / stdout / id) or None."""
    base: dict = {"verdict": "not_measured", "measured": None, "delta": None, "output": None, "exec": None,
                  "read_as": None, "reason": ""}
    if rec is None:
        return {**base, "reason": no_run_reason}
    base["exec"] = rec.id
    if rec.timed_out:
        return {**base, "reason": "the command timed out, so its output is not a finished result"}
    if rec.exit_code != 0:
        return {**base, "reason": f"the command exited with code {rec.exit_code}, so its output is not a finished result"}
    m = measure(claim, parse_outputs(rec.stdout))
    if m.status != "found" or m.output is None:
        return {**base, "reason": m.reason}
    claimed, tol, unit = comparison(claim)
    value, read_as = read_value(m.output, unit)
    delta = value - claimed
    ok = abs(delta) <= tol + TOLERANCE_EPS
    return {**base, "verdict": "reproduced" if ok else "not_reproduced", "measured": _r(to_claim_unit(value, unit)),
            "delta": _r(to_claim_unit(delta, unit)), "output": m.output.text, "read_as": read_as,
            "line": m.output.line_no}


def _change(before: str, after: str) -> str:
    if after == "reproduced":
        return "reproduced" if before == "reproduced" else "fixed"
    if after == "not_reproduced":
        return "regressed" if before == "reproduced" else "not_reproduced"
    return "not_measured"


def check_claims(claims: list[PaperClaim], ids: list[str], baseline: Any, final: Any, *,
                 baseline_reason: str = "the program did not run",
                 final_reason: str = "the program did not run") -> list[dict]:
    items = []
    for cid, c in zip(ids, claims):
        unit = resolve_unit(c)
        tol = c.tolerance if c.tolerance is not None else default_tolerance(unit, c.value)
        b = check_run(c, baseline, no_run_reason=baseline_reason)
        f = check_run(c, final, no_run_reason=final_reason)
        items.append({"id": cid, "metric": c.metric, "qualifier": c.qualifier or tokens(c.metric)[1], "claimed": c.value,
                      "unit": unit, "tolerance": _r(tol), "quote": c.quote, "source": c.source, "headline": c.headline,
                      "note": c.note, "baseline": b, "final": f, "verdict": f["verdict"],
                      "change": _change(b["verdict"], f["verdict"])})
    return items


def summarize(items: list[dict]) -> dict:
    n = len(items)
    return {
        "total": n,
        "reproduced_before": sum(1 for i in items if i["baseline"]["verdict"] == "reproduced"),
        "reproduced_after": sum(1 for i in items if i["final"]["verdict"] == "reproduced"),
        "fixed": sum(1 for i in items if i["change"] == "fixed"),
        "regressed": sum(1 for i in items if i["change"] == "regressed"),
        "not_reproduced": sum(1 for i in items if i["final"]["verdict"] == "not_reproduced"),
        "not_measured": sum(1 for i in items if i["final"]["verdict"] == "not_measured"),
    }
