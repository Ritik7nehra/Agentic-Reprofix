"""Find "the paper says the code reaches X" statements in pasted text, with patterns. No model reads the text.

What it recognises (and nothing else):
  * `accuracy: 91.2%`, `val_acc = 0.912`, `test accuracy of 76.4%`, `BLEU 27.3`   (metric word, optional connector, number)
  * `76.4% top-1 accuracy`, `27.3 BLEU`                                           (number, metric word)
  * markdown tables whose column headers are metric names (one row labelled "ours"/"proposed", or a single row), and
    two-column `| Accuracy | 91.2% |` tables.
It does NOT read PDFs, LaTeX, images, figures or free-form prose beyond those patterns, and it can pick up a number that
belongs to a baseline. Every extracted claim therefore carries its quote and the person is expected to confirm it.
"""
from __future__ import annotations

import re

from ..models import PaperClaim
from .check import NUM
from .names import DISTINCT, is_metric_word, tokens

MAX_CLAIMS = 20
MAX_LINES = 4000
MAX_LINE_CHARS = 1500

_Q = r"(?:test|testing|val|valid|validation|dev|eval|evaluation|train|training|held[\s\-]?out|final|overall|best|mean|average|avg)"
_M = (r"(?:top[\s\-]?[15](?:[\s\-]+(?:accuracy|acc))?|accuracy|acc|f[\s\-]?1(?:[\s\-]+score)?|f[\s\-]score|auroc|roc[\s\-]auc|auc|"
      r"miou|iou|bleu|rouge(?:[\s\-]?(?:l|1|2))?|exact[\s\-]match|perplexity|ppl|loss|mse|rmse|mae|r2|r²|precision|recall|"
      r"error[\s\-]rate|wer|cer|dice|ndcg|(?-i:mAP))")
_UNIT_AFTER = (r"(?!\s*(?:epochs?|steps?|iterations?|iters?|seconds?|secs?|ms|hours?|hrs?|minutes?|mins?|layers?|samples?|"
               r"images?|gb|mb|k|m|x|×|times)(?![A-Za-z]))")

# metric word, optional connector, number:  "accuracy of 76.4%", "val acc = 0.91"
_A = re.compile(
    rf"(?<![A-Za-z0-9])(?P<qual>(?:{_Q}[\s\-]+)*)(?P<metric>{_M})(?![A-Za-z0-9])"
    rf"(?:\s*\(\s*%\s*\))?(?:\s+(?:score|rate|value))?"
    rf"\s*(?P<conn>(?:of|is|was|are|were|reaches|reached|achieves|achieved|attains|attained|reports|reported|gets|got|yields|at|=|:|≈|~|\())?\s*"
    rf"(?P<val>{NUM})(?!,\d)\s*(?P<pct>%)?" + _UNIT_AFTER, re.I)
# number, metric word:  "76.4% top-1 accuracy", "27.3 BLEU"
_B = re.compile(
    rf"(?<![A-Za-z0-9.\-])(?P<val>{NUM})(?!,\d)\s*(?P<pct>%)?\s*(?P<qual>(?:{_Q}[\s\-]+)*)(?P<metric>{_M})(?![A-Za-z0-9])", re.I)
_SCORE_LIKE = {"bleu", "f1", "rouge", "rougel", "rouge1", "rouge2", "em", "miou", "dice", "ndcg", "map"}

_COMPARATIVE = re.compile(r"\b(baselines?|compared\s+(?:to|with)|than|vs\.?|versus|prior|previous(?:ly)?|outperform\w*|"
                          r"improv\w*\s+(?:over|upon)|surpass\w*|beats?|reported\s+by)\b", re.I)
# A margin is a difference between two results ("outperforms X by 2.1% accuracy", "a +2.1% accuracy gain"), not a result: the
# program will not print it, so reading it as a claim would report a failure that is not one.
_MARGIN_NOUNS = (r"improvements?|gains?|increases?|increased|reductions?|boosts?|drops?|decreases?|decreased|gaps?|margins?|"
                 r"differences?|degradations?|lifts?|jumps?|declines?|regressions?|penalt(?:y|ies)|deficits?|uplifts?|losses|loss|"
                 r"changes?|deltas?")
_MARGIN_BEFORE = re.compile(rf"(?:\bby|\b(?:{_MARGIN_NOUNS})(?:\s+(?:in|of|by))?(?:\s+\w+)?|\+)\s*$", re.I)
_MARGIN_AFTER = re.compile(rf"\s*(?:{_MARGIN_NOUNS})\b", re.I)
# More shapes of "not a result": a bound ("less than 1% accuracy loss", "within 2%", "at least 90%"), the other end of a range
# ("74-76%", "74 to 76%"), a number followed by a comparison ("2.3% lower than"), a value of the metric at a threshold ("recall at 10").
_BOUND_BEFORE = re.compile(r"(?:\b(?:less|fewer|more|greater|higher|lower|better|worse|smaller|larger)\s+than|\bwithin|\bbelow|\babove|"
                           r"\bover|\bunder|\bat\s+(?:most|least)|\bup\s+to|\bexceed\w*|\bbeyond|\bnearly|\balmost)\s*$", re.I)
_RANGE_BEFORE = re.compile(r"\d\s*%?\s*(?:-|–|—|to)\s*$", re.I)
_RANGE_AFTER = re.compile(r"\s*%?\s*(?:-|–|—|to)\s*[-+]?\.?\d", re.I)
_CMP_AFTER = re.compile(r"\s*(?:percent(?:age)?\s+points?|pp|pts?|points?)?\s*(?:higher|lower|better|worse|more|less|above|below|"
                        r"greater|smaller|larger|over|under)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(we|our|ours|my|proposed|this\s+(?:paper|work|repo|repository|implementation|code))\b", re.I)
_OURS_ROW = re.compile(r"\b(ours?|proposed|this\s+(?:work|paper|repo(?:sitory)?|implementation))\b", re.I)
_DELTA_HEADER = re.compile(r"[Δ∆]|\b(?:delta|diff\w*|gain|improv\w*|change|gap|std\w*|var\w*|margin)\b|±|\+/-", re.I)
_ABLATION_ROW = re.compile(r"w/o|w/\s|\bwithout\b|\bablat\w*|\bonly\b|\bno\b", re.I)
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_CELL_NUM = re.compile(rf"^\**\s*({NUM})\s*(%)?")


def _qualifier(text: str) -> str | None:
    q = tokens(text)[1]
    return q if q in ("test", "val", "eval", "train") else None


def _clean_metric(text: str) -> str:
    t = re.sub(r"\(\s*%\s*\)", "", text)
    return re.sub(r"\s+", " ", t.replace("_", " ")).strip(" :-|*`")[:60]


def _sentence(line: str, start: int, end: int) -> tuple[str, int]:
    """(the sentence of `line` that contains [start, end) cut to a quotable length, the offset where it starts)."""
    pos = 0
    for part in re.split(r"(?<=[.!?])\s+", line):
        i = line.find(part, pos)
        if i == -1:
            continue
        pos = i + len(part)
        if i <= start < pos:
            return part.strip()[:300], i
    return line.strip()[:300], 0


def _unit(pct: bool, header_pct: bool = False) -> str | None:
    return "percent" if (pct or header_pct) else None


def _claim(*, metric: str, value: float, pct: bool, quote: str, source: str, qualifier: str | None, header_pct: bool = False,
           note: str = "") -> PaperClaim | None:
    metric = _clean_metric(metric)
    if not metric:
        return None
    try:
        return PaperClaim(metric=metric, value=value, unit=_unit(pct, header_pct), qualifier=qualifier, quote=quote.strip()[:400],
                          source=source, note=note)
    except ValueError:
        return None


# --------------------------------------------------------------------------- prose
def _prose_claims(line: str, line_no: int, skipped: list[str] | None = None) -> list[PaperClaim]:
    norm = line.replace("_", " ").replace("*", " ").replace("`", " ")     # 1:1, so offsets stay valid
    found: list[tuple[int, int, PaperClaim]] = []
    taken: list[tuple[int, int]] = []
    for pat, kind in ((_A, "A"), (_B, "B")):
        for m in pat.finditer(norm):
            s, e = m.span("val")
            if any(s < b and a < e for a, b in taken):
                continue
            metric = m.group("metric")
            toks, _ = tokens(metric)
            pct = bool(m.group("pct"))
            if kind == "A" and not (m.group("conn") or pct or "." in m.group("val")):
                continue                                # "accuracy 76" with nothing else: too weak
            if kind == "B" and not (pct or "." in m.group("val") or toks & _SCORE_LIKE):
                continue                                # "5 loss", "3 precision": not a result
            try:
                value = float(m.group("val"))
            except ValueError:
                continue
            quote, sent_start = _sentence(line, m.start(), m.end())
            origin = m.start() if kind == "A" else s                       # where the phrase around the number begins
            why = None
            if _MARGIN_BEFORE.search(norm[max(sent_start, origin - 40):origin]) or (kind == "B" and _MARGIN_AFTER.match(norm, m.end())):
                why = "a margin between results"
            elif _BOUND_BEFORE.search(norm[max(sent_start, s - 30):s]):
                why = "a bound, not a result"
            elif _RANGE_BEFORE.search(norm[max(sent_start, s - 12):s]) or _RANGE_AFTER.match(norm, e + (1 if pct else 0)):
                why = "one end of a range"
            elif kind == "A" and _CMP_AFTER.match(norm, m.end()):
                why = "a difference from something else"
            elif kind == "A" and (m.group("conn") or "").lower() == "at" and not pct:
                why = "a threshold or position (such as 'recall at 10'), not a result"
            elif kind == "A" and re.search(r"\d\s*$", norm[max(0, m.start() - 8):m.start()]):
                why = "a value at a threshold (such as 'mAP at 0.5 IoU'), not a result"
            if why:
                taken.append((s, e))
                if skipped is not None:
                    skipped.append(f"line {line_no}: {m.group('val')}{'%' if pct else ''} {metric} reads as {why} ({quote[:90]!r}), "
                                   "so it was not read as a claim")
                continue
            if kind == "A":                  # "balanced accuracy of 0.9" is not "accuracy": keep the word that changes the quantity
                prev = re.search(r"([A-Za-z][A-Za-z\-]*)\s+$", norm[max(0, m.start() - 30):m.start()])
                if prev and any(part in DISTINCT for part in re.split(r"-+", prev.group(1).lower())):
                    metric = f"{prev.group(1)} {metric}"
            note = ""
            if _COMPARATIVE.search(norm, sent_start, m.start()):
                # a number that comes AFTER "than", "vs", "baseline", "outperforms" in its sentence is usually somebody else's
                note = "comes after a comparison with other methods: check it is the number this repository should produce"
            claim = _claim(metric=metric, value=value, pct=pct, quote=quote, source=f"paper text, line {line_no}",
                           qualifier=_qualifier(m.group("qual") + " " + metric), note=note)
            if claim is not None:
                found.append((m.start(), m.end(), claim))
                taken.append((s, e))
    found.sort(key=lambda t: t[0])
    return [c for _, _, c in found]


# --------------------------------------------------------------------------- tables
def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _cell_number(cell: str) -> tuple[float, bool] | None:
    m = _CELL_NUM.match(cell.replace("`", "").strip())
    if not m:
        return None
    try:
        return float(m.group(1)), bool(m.group(2))
    except ValueError:
        return None


def _table_claims(rows: list[tuple[int, str]], skipped: list[str]) -> list[PaperClaim]:
    """`rows` are (line number, text) of one markdown table, header first."""
    if len(rows) < 3 or not _TABLE_SEP.match(rows[1][1]):
        return []
    header = _cells(rows[0][1])
    data = [(n, _cells(t)) for n, t in rows[2:]]
    out: list[PaperClaim] = []
    metric_cols = [j for j, h in enumerate(header) if j > 0 and is_metric_word(tokens(h.replace("*", ""))[0])]
    delta_cols = [j for j in metric_cols if _DELTA_HEADER.search(header[j])]
    for j in delta_cols:
        skipped.append(f"table at line {rows[0][0]}: column {header[j].strip()!r} is a difference or a spread, not a result, so it was not read")
    metric_cols = [j for j in metric_cols if j not in delta_cols]
    if metric_cols:
        ours = [(n, r) for n, r in data if r and _OURS_ROW.search(r[0])]
        if len(ours) > 1:                                      # "Ours", "Ours w/o aug", "Ours (small)": not one result
            full = [(n, r) for n, r in ours if not _ABLATION_ROW.search(r[0])]
            if len(full) == 1:
                ours = full
        if len(ours) > 1:
            skipped.append(f"table at line {rows[0][0]} has {len(ours)} rows labelled 'ours' ({', '.join(repr(r[0].strip()) for _, r in ours[:4])}): "
                           "pass the row you mean as an explicit claim")
            return []
        chosen = ours or (data if len(data) == 1 else [])
        if not chosen:
            skipped.append(f"table at line {rows[0][0]} has {len(data)} rows and none is labelled 'ours' or 'proposed': "
                           "pass the row you mean as an explicit claim")
            return []
        for n, r in chosen:
            for j in metric_cols:
                if j >= len(r):
                    continue
                num = _cell_number(r[j])
                if num is None:
                    continue
                hdr = header[j].replace("*", "")
                c = _claim(metric=hdr, value=num[0], pct=num[1], header_pct="%" in hdr, qualifier=_qualifier(hdr),
                           quote=f"{header[0] or 'row'}: {r[0]} | {hdr}: {r[j]}  (table at line {rows[0][0]})",
                           source=f"paper text, table at line {n}")
                if c is not None:
                    out.append(c)
        return out
    # metric per row:  | Accuracy | 91.2% |
    for n, r in data:
        if len(r) < 2 or not is_metric_word(tokens(r[0].replace("*", ""))[0]):
            continue
        nums = [x for x in (_cell_number(c) for c in r[1:]) if x is not None]
        if len(nums) != 1:
            continue
        c = _claim(metric=r[0].replace("*", ""), value=nums[0][0], pct=nums[0][1], header_pct="%" in r[0],
                   qualifier=_qualifier(r[0]), quote=" | ".join(r), source=f"paper text, table at line {n}")
        if c is not None:
            out.append(c)
    return out


# --------------------------------------------------------------------------- entry points
def extract_claims(text: str) -> tuple[list[PaperClaim], list[str]]:
    """(claims in document order, notes about things that were seen but not extracted)."""
    lines = [ln[:MAX_LINE_CHARS] for ln in text.splitlines()[:MAX_LINES]]
    skipped: list[str] = []
    claims: list[tuple[int, PaperClaim]] = []
    i = 0
    while i < len(lines):
        if _TABLE_LINE.match(lines[i]):
            j = i
            while j < len(lines) and _TABLE_LINE.match(lines[j]):
                j += 1
            for c in _table_claims([(k + 1, lines[k]) for k in range(i, j)], skipped):
                claims.append((i, c))
            i = j
            continue
        for c in _prose_claims(lines[i], i + 1, skipped):
            claims.append((i, c))
        i += 1
    seen: set[tuple] = set()
    out: list[PaperClaim] = []
    for _, c in claims:
        toks, qual = tokens(c.metric)
        key = (toks, c.qualifier or qual, c.value)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    if len(out) > MAX_CLAIMS:
        skipped.append(f"{len(out) - MAX_CLAIMS} more {'claim' if len(out) - MAX_CLAIMS == 1 else 'claims'} beyond the first {MAX_CLAIMS} were not kept")
        out = out[:MAX_CLAIMS]
    return out, skipped


def pick_headline(claims: list[PaperClaim]) -> int | None:
    """Index of the claim the repair loop should optimise, or None. Explicit `headline` wins; then the first explicit claim;
    then the first extracted claim whose sentence is in the authors' voice and does not compare with other methods; then the
    first extracted claim without a comparison note. None when every claim follows a comparison with other methods."""
    if not claims:
        return None
    for i, c in enumerate(claims):
        if c.headline:
            return i
    for i, c in enumerate(claims):
        if c.source == "explicit":
            return i
    for i, c in enumerate(claims):
        if not c.note and (_FIRST_PERSON.search(c.quote or "") or _OURS_ROW.search(c.quote or "")):
            return i
    for i, c in enumerate(claims):
        if not c.note:
            return i
    return None                                   # every claim follows a comparison: do not guess which one is the repository's
