"""Paper claims: collect the numbers a paper states, measure them in the program's output, and build the report card.

    claims = collect(request)                    # explicit claims + rule-based extraction from request.paper_text
    spec   = metric_for(claims.headline_claim)   # the claim the repair loop optimises, as a MetricSpec
    card   = report_card(claims, baseline_exec, final_exec)

No model is involved anywhere in this package, and the paper text is never sent to one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import MetricSpec, PaperClaim, RunRequest
from .check import NUM, check_claims, comparison, resolve_unit, summarize
from .extract import extract_claims, pick_headline
from .names import tokens

METHOD = ("Numbers are read from the paper text by fixed patterns (no model), and compared with `name: number` lines in the "
          "program's own output. The tolerance is a chosen default unless the claim gives one.")


@dataclass
class ClaimSet:
    claims: list[PaperClaim] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    headline: int | None = None
    skipped: list[str] = field(default_factory=list)
    n_explicit: int = 0
    n_extracted: int = 0
    had_text: bool = False
    collected: bool = True           # False: the run ended before claims were read (so "no claims" would be a wrong thing to say)

    @property
    def headline_claim(self) -> PaperClaim | None:
        return self.claims[self.headline] if self.headline is not None else None

    def __bool__(self) -> bool:
        return bool(self.claims)


def _key(c: PaperClaim) -> tuple:
    toks, qual = tokens(c.metric)
    return toks, c.qualifier or qual, c.value


def collect(req: RunRequest) -> ClaimSet:
    explicit = [c.model_copy(update={"source": "explicit"}) for c in req.claims]
    extracted: list[PaperClaim] = []
    skipped: list[str] = []
    if req.paper_text.strip():
        extracted, skipped = extract_claims(req.paper_text)
        have = {_key(c) for c in explicit}
        extracted = [c for c in extracted if _key(c) not in have]
    claims = explicit + extracted
    cs = ClaimSet(skipped=skipped, n_explicit=len(explicit), n_extracted=len(extracted), had_text=bool(req.paper_text.strip()),
                  collected=True)
    if not claims:
        return cs
    cs.headline = pick_headline(claims)
    first: dict[tuple, int] = {}
    notes: list[str] = []
    for i, c in enumerate(claims):                  # one printed number can match only one of two claims that share a name
        toks, qual = tokens(c.metric)
        k = (toks, c.qualifier or qual)
        j = first.setdefault(k, i)
        same = j != i and claims[j].value != c.value
        notes.append((c.note + "; " if c.note else "") + f"same metric name as c{j + 1} with a different value: the program's output "
                     "can match only one of them" if same else c.note)
    claims = [c.model_copy(update={"headline": i == cs.headline, "unit": resolve_unit(c), "note": notes[i]}) for i, c in enumerate(claims)]
    cs.claims = claims
    cs.ids = [f"c{i + 1}" for i in range(len(claims))]
    return cs


def metric_for(claim: PaperClaim) -> MetricSpec:
    """The claim as the repair loop's target: compared on the claim's own scale, read by the same matcher as the report card."""
    expected, tol, _ = comparison(claim)
    return MetricSpec(name=claim.metric[:60], expected=expected, tolerance=tol, claim=claim)


def report_card(cs: ClaimSet, baseline, final, *, baseline_reason: str = "the program did not run",
                final_reason: str = "the program did not run") -> dict:
    if not cs:
        reason = ("no claim could be read from the pasted text" if cs.had_text
                  else "no paper text or claims were given, so no paper result was checked")
        if not cs.collected:
            reason = "the run ended before the claims were read, so nothing was checked"
        return {"checked": False, "reason": reason, "skipped": cs.skipped, "method": METHOD}
    items = check_claims(cs.claims, cs.ids, baseline, final, baseline_reason=baseline_reason, final_reason=final_reason)
    return {"checked": True, "method": METHOD, "headline": cs.ids[cs.headline] if cs.headline is not None else None,
            "items": items, "summary": summarize(items), "skipped": cs.skipped,
            "sources": {"explicit": cs.n_explicit, "extracted": cs.n_extracted}}


# --------------------------------------------------------------------------- command-line / form syntax
_CLAIM_ARG = re.compile(rf"^\s*(?P<metric>[^=:]*?[A-Za-z][^=:]*?)\s*[=:]\s*(?P<val>{NUM})\s*(?P<pct>%)?\s*"
                        rf"(?:(?:±|\+-|\+/-)\s*(?P<tol>{NUM})\s*(?P<tpct>%)?)?\s*$")


def parse_claim(text: str) -> PaperClaim:
    """`accuracy=76.4%`, `val_acc=0.912`, `bleu=27.3`, `accuracy=76.4%±0.5`. A % sign makes it a percentage, and then the
    tolerance is in percentage points."""
    m = _CLAIM_ARG.match(text)
    if not m:
        raise ValueError(f"cannot read a claim from {text!r}: write it like accuracy=76.4%  or  val_acc=0.912  or  bleu=27.3±0.3")
    unit = "percent" if m.group("pct") else None
    return PaperClaim(metric=m.group("metric").strip(), value=float(m.group("val")), unit=unit,
                      tolerance=float(m.group("tol")) if m.group("tol") else None, source="explicit")


__all__ = ["ClaimSet", "collect", "metric_for", "report_card", "parse_claim", "extract_claims"]
