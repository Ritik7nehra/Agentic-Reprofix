"""Metric names: turn "Top-1 accuracy", "val_acc" or "test set F1 score" into comparable token sets.

Everything here is plain string handling. The point is to decide whether a line the program printed
("val_acc: 0.764") is the same quantity as a number the paper states ("top-1 accuracy 76.4%"), and to say "I cannot tell"
when it is not clear, instead of guessing.
"""
from __future__ import annotations

import re

QUALIFIERS = {"test": "test", "testing": "test", "val": "val", "valid": "val", "validation": "val", "dev": "val",
              "eval": "eval", "evaluation": "eval", "train": "train", "training": "train"}
# words that carry no information about WHICH quantity it is
FILLER = {"score", "rate", "value", "final", "best", "overall", "mean", "average", "avg", "the", "of", "metric",
          "held", "out", "heldout", "set", "data", "dataset", "split", "on", "per", "total", "model"}
ALIASES = {"acc": "accuracy", "ppl": "perplexity", "auroc": "auc", "rocauc": "auc", "err": "error"}
PHRASES = [
    (re.compile(r"top[\s\-_]?1(?![0-9])"), " top1 "), (re.compile(r"top[\s\-_]?5(?![0-9])"), " top5 "),
    (re.compile(r"exact[\s\-_]match"), " em "), (re.compile(r"roc[\s\-_]auc"), " auc "),
    (re.compile(r"r²|r[\s\-_]?squared"), " r2 "), (re.compile(r"\bf[\s\-_]?1\b"), " f1 "),
    (re.compile(r"\bf[\s\-_]score\b"), " f1 "), (re.compile(r"error[\s\-_]rate"), " error "),
    (re.compile(r"rouge[\s\-_]?l\b"), " rougel "), (re.compile(r"rouge[\s\-_]?1\b"), " rouge1 "),
    (re.compile(r"rouge[\s\-_]?2\b"), " rouge2 "), (re.compile(r"mean[\s\-_]average[\s\-_]precision"), " map "),
]
# quantities that are bounded scores (a fraction in 0..1, or a percentage in 0..100)
BOUNDED = {"accuracy", "top1", "top5", "f1", "auc", "precision", "recall", "map", "miou", "iou", "bleu", "rouge", "rougel",
           "rouge1", "rouge2", "em", "dice", "ndcg"}
# anything in here counts as "a metric" when it heads a table column or a row
VOCAB = BOUNDED | {"perplexity", "loss", "mse", "rmse", "mae", "r2", "error", "wer", "cer"}


def tokens(name: str) -> tuple[frozenset[str], str | None]:
    """(content tokens, qualifier). Qualifier is the split the name mentions: test / val / eval / train / None."""
    low = name.lower()
    for pat, rep in PHRASES:
        low = pat.sub(rep, low)
    words = re.sub(r"[^a-z0-9]+", " ", low).split()
    qualifier: str | None = None
    out: set[str] = set()
    for w in words:
        if w in QUALIFIERS:
            qualifier = qualifier or QUALIFIERS[w]
            continue
        if w in FILLER:
            continue
        out.add(ALIASES.get(w, w))
    if out & {"top1", "top5"}:
        out.discard("accuracy")                    # "top1_acc", "top-1 accuracy" and "top1" are one quantity
    return frozenset(out), qualifier


# Words that make a name a DIFFERENT quantity from the plain metric: "accuracy_gap", "worst_group_accuracy", "balanced accuracy",
# "loss_scale", "accuracy@5". A printed name that has one of these on top of the claimed name is not the claimed metric.
# Words that do not change the quantity (a dataset name, "epoch") are not listed, so "cifar10_accuracy" still counts as accuracy.
DISTINCT = {
    "gap", "std", "stdev", "stddev", "var", "variance", "delta", "diff", "difference", "change", "gain", "improvement", "drop",
    "margin", "ratio", "count", "num", "worst", "group", "groups", "target", "expected", "baseline", "ref", "reference", "zero",
    "shot", "few", "weight", "weights", "scale", "scaling", "ema", "min", "max", "step", "steps", "lr", "norm", "grad", "gradient",
    "threshold", "thresh", "balanced", "macro", "micro", "weighted", "pixel", "class", "classwise", "sample", "token", "word", "char",
    "character", "relative", "absolute", "normalized", "normalised", "adjusted", "calibrated", "calibration", "ece", "entropy",
    "confidence", "conf", "prior", "pseudo", "noisy", "adversarial", "robust", "ood", "corrupted", "instance", "frame", "sentence",
}


def extras_ok(claimed: frozenset[str], printed: frozenset[str]) -> bool:
    """May a printed name that contains every word of the claimed name stand for it? Only if the extra words do not make it a
    different quantity (a number such as the 5 of accuracy@5, or a word in DISTINCT)."""
    return not any(w.isdigit() or w in DISTINCT for w in printed - claimed)


def is_bounded(toks: frozenset[str]) -> bool:
    return bool(toks & BOUNDED)


def is_metric_word(toks: frozenset[str]) -> bool:
    return bool(toks & VOCAB)
