"""Shared data models."""
from __future__ import annotations

import math
import re
import time
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PaperClaim(BaseModel):
    """One number a paper says the code produces, e.g. "test accuracy 76.4%".

    Claims are matched against the program's own printed output by rules (reprofix/claims), never by a model.
    """

    metric: str = Field(min_length=1, max_length=60)             # as written: "accuracy", "Top-1 accuracy", "BLEU"
    value: float                                                  # as written: 76.4 for "76.4%", 0.764 for "0.764"
    unit: Literal["percent", "fraction", "raw"] | None = None   # None: decided from the metric and the value
    tolerance: float | None = None                                # in the unit of `value` (points for percent); None: default
    qualifier: Literal["test", "val", "eval", "train"] | None = None
    quote: str = Field(default="", max_length=400)               # the sentence or table row it came from
    source: str = Field(default="explicit", max_length=80)       # "explicit" | "paper text, line 12"
    headline: bool = False                                        # the repair loop optimises this claim
    note: str = Field(default="", max_length=240)

    @field_validator("value", "tolerance")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("must be a finite number")
        return v

    @field_validator("tolerance")
    @classmethod
    def _tol_positive(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("tolerance cannot be negative")
        return v


class MetricSpec(BaseModel):
    """What 'reproduced' means. Metrics are parsed from stdout by regex -- never by an LLM."""

    name: str = "val_accuracy"
    regex: str | None = None  # one capture group; default is built from `name`
    expected: float | None = None
    tolerance: float = 0.02  # absolute; "reproduced" means |value - expected| <= tolerance
    # Set when the metric comes from a paper claim: the value is then read with the same matcher that fills the report card
    # (reprofix/claims/check.py), on the scale the claim is compared on (a fraction for percent/fraction claims).
    claim: PaperClaim | None = None

    @field_validator("regex")
    @classmethod
    def _regex_ok(cls, v: str | None) -> str | None:
        if v is None:
            return v
        compiled = re.compile(v)
        if compiled.groups < 1:
            raise ValueError("metric regex needs one capture group")
        return v

    def pattern(self) -> re.Pattern[str]:
        if self.regex:
            return re.compile(self.regex)
        return re.compile(rf"{re.escape(self.name)}\s*[:=]\s*(-?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", re.I)


RouterMode = Literal["router", "super-only", "ultra-only"]


class RunRequest(BaseModel):
    repo_url: str | None = None
    local_path: str | None = None
    goal: str = "Reproduce the documented experiment and determine what is wrong."
    command: str | None = None
    metric: MetricSpec | None = None
    max_attempts: int = 6
    router_mode: RouterMode = "router"
    install: bool = True
    # Optional paper claims. `paper_text` is text pasted from the paper or README (never sent to a model: only rule-based
    # patterns read it); `claims` are numbers the person typed in. Both feed the report card and the headline claim.
    paper_text: str = Field(default="", max_length=200_000)
    claims: list[PaperClaim] = Field(default_factory=list, max_length=30)
    command_timeout_s: int = 600
    max_total_tokens: int = 400_000
    max_run_seconds: int = 1800
    protected_paths: list[str] = Field(default_factory=lambda: ["tests/**", "test_*.py", "**/test_*.py", "README*", "LICENSE*"])


class ExecRecord(BaseModel):
    """One command execution, as shown in the UI and report."""

    id: str
    phase: Literal["install", "baseline", "experiment", "tests", "final"]
    argv: list[str]
    exit_code: int | None
    duration_s: float
    timed_out: bool = False
    network: bool = False
    stdout: str = ""
    stderr: str = ""
    truncated: bool = False
    metric_value: float | None = None
    started_at: float = Field(default_factory=time.time)


class Observation(BaseModel):
    """Digest of a failing state handed to the diagnoser."""

    kind: Literal["install_failure", "crash", "metric_gap", "test_failure", "timeout"]
    summary: str
    exception_type: str | None = None
    exception_message: str | None = None
    location: str | None = None  # "path:line" of the deepest in-repo frame
    signature: str = ""
    log_tail: str = ""
    metric_value: float | None = None
    expected: float | None = None


class Verdict(BaseModel):
    stage: Literal["install_failure", "crash", "wrong_result", "verified"]
    verified: bool
    executes: bool
    metric_value: float | None = None
    metric_ok: bool | None = None  # None = no metric spec
    tests_ok: bool | None = None  # None = no tests found
    gap: float | None = None
    signature: str = ""
    reasons: list[str] = Field(default_factory=list)


class Event(BaseModel):
    seq: int
    ts: float
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)
