"""Paper claims: extraction from text, matching against program output, and the report card.

Nothing here involves a model: the point of the claims package is that a number is judged by rules and a re-run, not by a
model's say-so. The end-to-end tests use the scripted backend, which is NOT Nemotron, so they validate the pipeline only.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from conftest import BUG_LINE, FIX_LINE, TRAIN_PY, hypothesis_reply, repair_reply, scripted
from reprofix.claims import collect, metric_for, parse_claim, report_card
from reprofix.claims.check import (check_run, claim_missing_reason, claim_value, comparison, default_tolerance, measure,
                                   parse_outputs, read_value, resolve_unit)
from reprofix.claims.extract import extract_claims, pick_headline
from reprofix.claims.names import tokens
from reprofix.core.orchestrator import Orchestrator
from reprofix.core.verify import make_verdict, parse_metric
from reprofix.models import MetricSpec, PaperClaim, RunRequest
from reprofix.sandbox import ExecResult
from reprofix.sandbox.local_backend import LocalSandbox


def rec(stdout: str, code: int = 0, timed_out: bool = False, id: str = "e1") -> SimpleNamespace:
    return SimpleNamespace(id=id, exit_code=code, timed_out=timed_out, stdout=stdout)


# --------------------------------------------------------------------------- names
@pytest.mark.parametrize("name,toks,qual", [
    ("Top-1 accuracy", {"top1"}, None), ("top1_acc", {"top1"}, None), ("val_acc", {"accuracy"}, "val"),
    ("test set F1 score", {"f1"}, "test"), ("Exact Match", {"em"}, None), ("ROUGE-L", {"rougel"}, None),
    ("R²", {"r2"}, None), ("Test Accuracy (%)", {"accuracy"}, "test"), ("train_loss", {"loss"}, "train"),
])
def test_metric_names_are_reduced_to_comparable_tokens(name, toks, qual):
    assert tokens(name) == (frozenset(toks), qual)


# --------------------------------------------------------------------------- extraction
PAPER = """# Foo-Net
We achieve a top-1 accuracy of 76.4% on ImageNet, outperforming ResNet (accuracy of 74.1%) by 2.3 points.
Our model reaches 27.3 BLEU on WMT14 en-de.
The accuracy improved from 70.1% to 76.4% after tuning.
Training takes 90 epochs; accuracy 90 epochs is not a claim.

| Model | Top-1 (%) | Top-5 (%) |
|---|---|---|
| ResNet-50 | 76.1 | 92.9 |
| Foo-Net (ours) | **78.3** | 94.0 |

| Model | Acc |
|---|---|
| A | 10 |
| B | 20 |

| Metric | Value |
|---|---|
| Test accuracy | 91.2% |
| Epochs | 90 |

Final val_acc = 0.912 and loss: 0.31.
"""


def test_extraction_reads_prose_and_tables_and_says_what_it_skipped():
    claims, skipped = extract_claims(PAPER)
    got = [(c.metric, c.value, c.unit, c.qualifier) for c in claims]
    assert ("top-1 accuracy", 76.4, "percent", None) in got
    assert ("BLEU", 27.3, None, None) in got
    assert ("Top-1", 78.3, "percent", None) in got and ("Top-5", 94.0, "percent", None) in got      # only the 'ours' row
    assert not any(v in (76.1, 92.9) for _, v, _, _ in got)                                          # ResNet-50 row is not read
    assert ("Test accuracy", 91.2, "percent", "test") in got
    assert ("acc", 0.912, None, "val") in got and ("loss", 0.31, None, None) in got
    assert not any(v in (70.1, 90) for _, v, _, _ in got)                                           # "improved from 70.1% to 76.4%"
    assert len(skipped) == 1 and "2 rows" in skipped[0] and "ours" in skipped[0]                   # the unlabeled two-row table


def test_a_number_after_a_comparison_word_is_flagged_as_possibly_somebody_elses():
    claims, _ = extract_claims("We achieve a top-1 accuracy of 76.4%, outperforming ResNet (accuracy of 74.1%) by 2.3 points.")
    own, other = claims
    assert own.value == 76.4 and own.note == ""
    assert other.value == 74.1 and "comparison" in other.note
    assert pick_headline(claims) == 0


def test_weak_patterns_are_not_claims():
    for text in ["accuracy 90 epochs", "5 loss functions", "accuracy improved from 70.1% to 76.4%", "the accuracy by 3.2%",
                 "precision 4 layers", "map 5 elements"]:
        assert extract_claims(text)[0] == [], text


@pytest.mark.parametrize("text,metric,value,unit,qual", [
    ("accuracy: 91.2%", "accuracy", 91.2, "percent", None),
    ("**Test accuracy**: 0.912", "accuracy", 0.912, None, "test"),            # the split is kept apart from the name
    ("val_acc = 0.912", "acc", 0.912, None, "val"),
    ("reaches 0.91 F1", "F1", 0.91, None, None),
    ("a perplexity of 18.4 on WikiText", "perplexity", 18.4, None, None),
    ("mAP of 45.2", "mAP", 45.2, None, None),
])
def test_extraction_patterns(text, metric, value, unit, qual):
    claims, _ = extract_claims(text)
    assert [(c.metric, c.value, c.unit, c.qualifier) for c in claims] == [(metric, value, unit, qual)]
    assert claims[0].quote and claims[0].source.startswith("paper text, line 1")


def test_two_column_metric_table_and_single_row_table():
    claims, skipped = extract_claims("| Metric | Value |\n|---|---|\n| Accuracy | 91.2% |\n| Epochs | 90 |\n\n"
                                     "| Model | Accuracy |\n|---|---|\n| Mine | 88.0 |\n")
    assert [(c.metric, c.value) for c in claims] == [("Accuracy", 91.2), ("Accuracy", 88.0)] and skipped == []


def test_duplicates_are_dropped_and_the_count_is_capped():
    claims, _ = extract_claims("accuracy of 91.2%.\nOnce more: accuracy of 91.2%.")
    assert len(claims) == 1
    many = "\n".join(f"accuracy of {i}.5%" for i in range(40))
    claims, skipped = extract_claims(many)
    assert len(claims) == 20 and "beyond the first 20" in skipped[-1]


def test_extraction_is_linear_on_hostile_text():
    hostile = ("accuracy " * 20000 + "\n" + "a" * 300_000 + "\n" + "| " * 50000 + "\n")[:200_000]
    t0 = time.time()
    extract_claims(hostile)
    assert time.time() - t0 < 5


# --------------------------------------------------------------------------- reading program output
def test_output_pairs_are_read_the_way_training_scripts_print_them():
    out = parse_outputs("Epoch 3/10 - loss: 0.52 - val_accuracy: 0.81\nlr=0.01 batch_size=32\nTest Accuracy: 76.4%\n")
    assert [(o.name, o.value, o.pct) for o in out] == [
        ("loss", 0.52, False), ("val_accuracy", 0.81, False), ("lr", 0.01, False), ("batch_size", 32.0, False),
        ("Test Accuracy", 76.4, True)]
    assert [o.line_no for o in out] == [1, 1, 2, 2, 3]


def test_output_parsing_is_bounded():
    t0 = time.time()
    parse_outputs(("a" * 200_000 + ": 1\n") + "x: 1\n" * 10_000)
    assert time.time() - t0 < 5
    many = "\n".join(f"step{i}_loss: {i}" for i in range(20_000))
    assert len(parse_outputs(many)) <= 5000                                   # only the last lines are read


def test_the_last_value_printed_is_the_measurement():
    c = PaperClaim(metric="accuracy", value=90.0, unit="percent")
    m = measure(c, parse_outputs("val_accuracy: 0.10\nval_accuracy: 0.50\nval_accuracy: 0.90\n"))
    assert m.status == "found" and m.output.value == 0.90 and m.output.line_no == 3


def test_matching_refuses_to_guess_between_two_outputs():
    c = PaperClaim(metric="accuracy", value=90.0, unit="percent")
    m = measure(c, parse_outputs("val_accuracy: 0.8\ntest_accuracy: 0.9\n"))
    assert m.status == "ambiguous" and set(m.candidates) == {"val_accuracy", "test_accuracy"} and "--claim" in m.reason
    # naming the split resolves it
    q = PaperClaim(metric="accuracy", value=90.0, unit="percent", qualifier="test")
    assert measure(q, parse_outputs("val_accuracy: 0.8\ntest_accuracy: 0.9\n")).output.name == "test_accuracy"


def test_unqualified_claims_ignore_train_outputs_and_prefer_an_unlabelled_output():
    c = PaperClaim(metric="accuracy", value=90.0, unit="percent")
    assert measure(c, parse_outputs("train_accuracy: 0.99\n")).status == "missing"
    assert measure(c, parse_outputs("train_accuracy: 0.99\naccuracy: 0.9\nval_accuracy: 0.8\n")).output.name == "accuracy"
    t = PaperClaim(metric="accuracy", value=99.0, unit="percent", qualifier="train")
    assert measure(t, parse_outputs("train_accuracy: 0.99\n")).output.name == "train_accuracy"


def test_a_qualified_claim_accepts_an_unlabelled_output_only_when_nothing_labelled_fits():
    c = PaperClaim(metric="accuracy", value=90.0, unit="percent", qualifier="test")
    assert measure(c, parse_outputs("accuracy: 0.9\n")).output.name == "accuracy"
    assert measure(c, parse_outputs("accuracy: 0.9\nval_accuracy: 0.8\n")).output.name == "accuracy"
    assert measure(c, parse_outputs("val_accuracy: 0.8\n")).status == "missing"


def test_metric_families_match_across_spellings():
    for claim_name, printed in [("Top-1 accuracy", "top1_acc: 0.76"), ("acc", "accuracy: 0.76"), ("F1 score", "f1: 0.76"),
                                ("top-1", "val top1 accuracy: 0.76")]:
        assert measure(PaperClaim(metric=claim_name, value=76.0, unit="percent"), parse_outputs(printed)).status == "found", claim_name
    m = measure(PaperClaim(metric="f1", value=76.0, unit="percent"), parse_outputs("macro_f1: 0.7\nmicro_f1: 0.8\n"))
    assert m.status == "missing" and "macro_f1" in m.reason and "micro_f1" in m.reason     # a different quantity, not "f1"
    assert measure(PaperClaim(metric="f1", value=76.0, unit="percent"), parse_outputs("f1_a: 0.7\nf1_b: 0.8\n")).status == "ambiguous"


# --------------------------------------------------------------------------- scales and tolerance
@pytest.mark.parametrize("printed", ["val_accuracy: 0.764", "val_accuracy: 76.4", "val_accuracy: 76.4%"])
def test_percentages_and_fractions_are_the_same_number(printed):
    c = PaperClaim(metric="accuracy", value=76.4, unit="percent")
    r = check_run(c, rec(printed + "\n"))
    assert r["verdict"] == "reproduced" and r["measured"] == pytest.approx(76.4) and r["delta"] == pytest.approx(0, abs=1e-6)


def test_a_fraction_claim_is_compared_with_a_percentage_output_too():
    r = check_run(PaperClaim(metric="accuracy", value=0.764), rec("accuracy: 76.4\n"))
    assert r["verdict"] == "reproduced" and r["read_as"] == "percent"


def test_default_tolerances_and_the_edge_of_the_band():
    assert default_tolerance("percent", 76.4) == 1.0 and default_tolerance("fraction", 0.7) == 0.01
    assert default_tolerance("raw", 200.0) == pytest.approx(2.0)
    c = PaperClaim(metric="accuracy", value=76.4, unit="percent")
    assert check_run(c, rec("accuracy: 0.754\n"))["verdict"] == "reproduced"          # exactly 1.0 point below: inside the band
    assert check_run(c, rec("accuracy: 0.753\n"))["verdict"] == "not_reproduced"      # 1.1 points below
    own = PaperClaim(metric="accuracy", value=76.4, unit="percent", tolerance=0.1)
    assert check_run(own, rec("accuracy: 0.764\n"))["verdict"] == "reproduced"
    assert check_run(own, rec("accuracy: 0.766\n"))["verdict"] == "not_reproduced"


def test_a_raw_metric_is_compared_as_printed_with_a_relative_tolerance():
    c = PaperClaim(metric="perplexity", value=18.4)
    assert resolve_unit(c) == "raw"
    assert check_run(c, rec("perplexity: 18.5\n"))["verdict"] == "reproduced"         # within 1% (0.184)
    r = check_run(c, rec("perplexity: 21.0\n"))
    assert r["verdict"] == "not_reproduced" and r["measured"] == 21.0 and r["delta"] == pytest.approx(2.6)


def test_units_are_decided_from_the_metric_and_the_value():
    assert resolve_unit(PaperClaim(metric="accuracy", value=0.9)) == "fraction"
    assert resolve_unit(PaperClaim(metric="accuracy", value=90.0)) == "percent"
    assert resolve_unit(PaperClaim(metric="BLEU", value=27.3)) == "percent"          # BLEU is printed as 27.3 or 0.273
    assert resolve_unit(PaperClaim(metric="loss", value=0.31)) == "raw"
    assert resolve_unit(PaperClaim(metric="accuracy", value=250.0)) == "raw"


# --------------------------------------------------------------------------- a run that did not finish measures nothing
@pytest.mark.parametrize("r,why", [(rec("accuracy: 0.99\n", code=1), "exited with code 1"),
                                   (rec("accuracy: 0.99\n", timed_out=True), "timed out"),
                                   (None, "did not run"), (rec("hello\n"), "no output line is named like")])
def test_not_measured_always_says_why(r, why):
    out = check_run(PaperClaim(metric="accuracy", value=90.0, unit="percent"), r)
    assert out["verdict"] == "not_measured" and out["measured"] is None and why in out["reason"]


# --------------------------------------------------------------------------- explicit claims and the headline
def test_claim_syntax():
    c = parse_claim("accuracy=76.4%")
    assert (c.metric, c.value, c.unit, c.tolerance) == ("accuracy", 76.4, "percent", None)
    c = parse_claim("val_acc: 0.912")
    assert (c.metric, c.value, c.unit) == ("val_acc", 0.912, None)
    c = parse_claim("bleu=27.3±0.3")
    assert (c.metric, c.value, c.tolerance) == ("bleu", 27.3, 0.3)
    assert parse_claim("Top-1 accuracy = 76.4% +- 0.5%").tolerance == 0.5
    for bad in ["76.4", "accuracy=", "=0.5", "accuracy=abc", ""]:
        with pytest.raises(ValueError):
            parse_claim(bad)


def test_collect_merges_explicit_and_extracted_and_picks_a_headline():
    req = RunRequest(repo_url="https://github.com/a/b", paper_text="We reach an accuracy of 80.0% on the test set.",
                     claims=[PaperClaim(metric="accuracy", value=80.0, unit="percent"), PaperClaim(metric="bleu", value=27.3)])
    cs = collect(req)
    assert [c.metric for c in cs.claims] == ["accuracy", "bleu"]                      # the extracted duplicate of the first is dropped
    assert cs.ids == ["c1", "c2"] and cs.headline == 0 and cs.claims[0].headline and not cs.claims[1].headline
    assert cs.n_explicit == 2 and cs.n_extracted == 0
    flagged = collect(RunRequest(repo_url="https://github.com/a/b", claims=[PaperClaim(metric="a", value=1.0),
                                                                           PaperClaim(metric="b", value=2.0, headline=True)]))
    assert flagged.headline == 1


def test_no_claims_gives_a_card_that_says_why():
    assert report_card(collect(RunRequest(repo_url="https://github.com/a/b")), None, None) == {
        "checked": False, "reason": "no paper text or claims were given, so no paper result was checked", "skipped": [],
        "method": report_card(collect(RunRequest(repo_url="https://github.com/a/b")), None, None)["method"]}
    card = report_card(collect(RunRequest(repo_url="https://github.com/a/b", paper_text="nothing numeric here")), None, None)
    assert card["checked"] is False and "no claim could be read" in card["reason"]


# --------------------------------------------------------------------------- the repair loop reads claims with the same matcher
def test_the_headline_claim_becomes_the_metric_the_loop_optimises():
    claim = PaperClaim(metric="Top-1 accuracy", value=76.4, unit="percent", quote="we reach 76.4%")
    spec = metric_for(claim)
    assert spec.expected == pytest.approx(0.764) and spec.tolerance == pytest.approx(0.01) and spec.claim == claim
    assert parse_metric("top1_acc: 76.4\n", spec) == pytest.approx(0.764)             # percent output, fraction scale
    assert parse_metric("top1_acc: 0.764\n", spec) == pytest.approx(0.764)
    assert parse_metric("nothing\n", spec) is None


def test_the_verdict_explains_an_ambiguous_claim_metric(tmp_path):
    spec = metric_for(PaperClaim(metric="accuracy", value=90.0, unit="percent"))
    run = ExecResult(argv=["python", "t.py"], exit_code=0, stdout="val_accuracy: 0.8\ntest_accuracy: 0.9\n", stderr="", duration_s=0.1)
    v = make_verdict(install=None, run=run, tests=None, metric=spec, workdir=tmp_path)
    assert not v.verified and v.metric_value is None
    assert "several outputs fit 'accuracy'" in v.reasons[0]


def test_hardcoding_the_claimed_number_is_still_refused_for_a_claim_metric():
    from reprofix.core.patching import Edit, literal_hardcoding_findings
    spec = metric_for(PaperClaim(metric="accuracy", value=76.4, unit="percent"))
    edit = Edit(path="train.py", search="x", replace='print("accuracy: 76.4")')
    assert literal_hardcoding_findings([edit], spec.expected, spec.name)


# --------------------------------------------------------------------------- end to end (scripted backend, NOT Nemotron)
PAPER_TEXT = "Our model reaches a validation accuracy of 90.0% on the toy task."


def run_with_claims(settings, repo, tmp_path, backend, *, attempts=4, **req):
    req.setdefault("command", "python train.py")
    request = RunRequest(local_path=str(repo), max_attempts=attempts, **req)
    events: list[tuple[str, dict]] = []
    orch = Orchestrator(settings=settings, request=request, run_dir=tmp_path / "run", sandbox=LocalSandbox(settings),
                        backend=backend, emit=lambda k, d: events.append((k, d)))
    return orch, orch.run(), events


def item(rep, cid="c1"):
    return next(i for i in rep["claims"]["items"] if i["id"] == cid)


def test_a_paper_claim_is_measured_before_and_after_the_repair_and_drives_it(settings, tiny_repo, tmp_path):
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    orch, rep, events = run_with_claims(settings, tiny_repo, tmp_path, b, paper_text=PAPER_TEXT)
    assert rep["status"] == "verified"
    assert rep["metric"]["source"].startswith("paper claim c1: Our model reaches a validation accuracy of 90.0%")
    card = rep["claims"]
    assert card["checked"] and card["headline"] == "c1" and card["sources"] == {"explicit": 0, "extracted": 1}
    c1 = item(rep)
    assert (c1["claimed"], c1["unit"], c1["qualifier"], c1["headline"]) == (90.0, "percent", "val", True)
    assert c1["baseline"]["verdict"] == "not_reproduced" and c1["baseline"]["measured"] == 0.0 and c1["baseline"]["delta"] == -90.0
    assert c1["final"]["verdict"] == "reproduced" and c1["final"]["measured"] == pytest.approx(90.0)
    assert c1["change"] == "fixed" and c1["verdict"] == "reproduced"
    assert c1["final"]["output"] == "val_accuracy: 0.9000"
    assert card["summary"] == {"total": 1, "reproduced_before": 0, "reproduced_after": 1, "fixed": 1, "regressed": 0,
                               "not_reproduced": 0, "not_measured": 0}
    # the proof is a real execution that the report lists, and the final one is the clean re-run
    execs = {e["id"]: e for e in rep["executions"]}
    assert execs[c1["baseline"]["exec"]]["phase"] == "baseline" and execs[c1["final"]["exec"]]["phase"] == "final"
    assert any("1 claim was read from the pasted text" in c for c in rep["caveats"])
    assert any(k == "claims" and d["headline"] == "c1" for k, d in events)


def test_the_paper_text_is_never_sent_to_a_model(settings, tiny_repo, tmp_path):
    marker = "SENTINEL-PAPER-TEXT-42"
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    run_with_claims(settings, tiny_repo, tmp_path, b, paper_text=f"{PAPER_TEXT}\n{marker}")
    assert b.calls and marker not in json.dumps([c["messages"] for c in b.calls])


def test_explicit_claims_choose_the_headline_and_an_unmeasurable_claim_says_why(settings, tiny_repo, tmp_path):
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    claims = [PaperClaim(metric="val_accuracy", value=0.9), PaperClaim(metric="bleu", value=27.3)]
    _, rep, _ = run_with_claims(settings, tiny_repo, tmp_path, b, claims=claims)
    assert rep["status"] == "verified" and rep["claims"]["headline"] == "c1"
    assert item(rep)["change"] == "fixed"
    bleu = item(rep, "c2")
    assert bleu["final"]["verdict"] == "not_measured"
    assert "no output line is named like 'bleu' (the output has: val_accuracy)" in bleu["final"]["reason"]
    assert bleu["change"] == "not_measured" and rep["claims"]["summary"]["not_measured"] == 1
    assert any("could not be measured" in c for c in rep["caveats"])
    assert not any("chosen automatically" in c for c in rep["caveats"])               # the person chose it


def test_an_explicit_metric_still_beats_a_claim_for_the_loop_but_the_card_is_filled(settings, tiny_repo, tmp_path):
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    own = MetricSpec(name="val_accuracy", expected=0.9, tolerance=0.02)
    _, rep, _ = run_with_claims(settings, tiny_repo, tmp_path, b, metric=own, paper_text=PAPER_TEXT)
    assert rep["metric"]["source"] == "user" and rep["status"] == "verified"
    assert item(rep)["change"] == "fixed"
    assert not any("targeted claim" in c for c in rep["caveats"])


def test_a_reverted_experiment_is_not_what_the_card_measures(settings, tiny_repo, tmp_path):
    """The workspace is put back after a useless experiment; the claim must be judged on the state that remains."""
    useless = BUG_LINE.replace("correct //", "(correct) //")
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, useless)])
    orch, rep, _ = run_with_claims(settings, tiny_repo, tmp_path, b, paper_text=PAPER_TEXT, attempts=3)
    assert rep["status"] == "failed" and rep["files_modified"] == []
    c1 = item(rep)
    assert c1["final"]["verdict"] == "not_reproduced" and c1["change"] == "not_reproduced"
    assert c1["final"]["exec"] == c1["baseline"]["exec"]                              # the baseline run, not the reverted experiment's
    assert any(e["phase"] == "experiment" for e in rep["executions"])


def test_a_baseline_that_crashes_has_nothing_to_measure_but_the_repair_can(settings, tiny_repo, tmp_path):
    (tiny_repo / "train.py").write_text(TRAIN_PY.replace("preds = [", "print(undefined_name)\npreds = [", 1))
    crash_line = "print(undefined_name)\n"
    b = scripted([hypothesis_reply("h1", "an undefined name is printed", "print(undefined_name)"),
                  hypothesis_reply("h2", "floor division truncates the accuracy to zero", BUG_LINE)],
                 [repair_reply(crash_line, ""), repair_reply(BUG_LINE, FIX_LINE)])
    _, rep, _ = run_with_claims(settings, tiny_repo, tmp_path, b, paper_text=PAPER_TEXT, attempts=4)
    c1 = item(rep)
    assert c1["baseline"]["verdict"] == "not_measured" and "exited with code 1" in c1["baseline"]["reason"]
    assert c1["final"]["verdict"] == "reproduced" and c1["change"] == "fixed" and rep["status"] == "verified"


def test_a_project_that_already_reproduces_the_claim_is_reported_as_unchanged(settings, tiny_repo, tmp_path):
    (tiny_repo / "train.py").write_text(TRAIN_PY.replace(BUG_LINE, FIX_LINE))
    _, rep, _ = run_with_claims(settings, tiny_repo, tmp_path, scripted([], []), paper_text=PAPER_TEXT)
    c1 = item(rep)
    assert rep["status"] == "already_passing" and c1["baseline"]["verdict"] == c1["final"]["verdict"] == "reproduced"
    assert c1["change"] == "reproduced" and rep["claims"]["summary"]["fixed"] == 0


def test_a_claim_the_repair_cannot_reach_is_reported_as_not_reproduced(settings, tiny_repo, tmp_path):
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    claims = [PaperClaim(metric="val_accuracy", value=0.9), PaperClaim(metric="val_accuracy", value=0.99, qualifier="val")]
    _, rep, _ = run_with_claims(settings, tiny_repo, tmp_path, b, claims=claims)
    assert rep["status"] == "verified"                                                  # the headline (0.9) is reproduced ...
    c2 = item(rep, "c2")
    assert c2["final"]["verdict"] == "not_reproduced" and c2["final"]["delta"] == pytest.approx(-0.09)   # ... the 0.99 claim is not
    assert rep["claims"]["summary"]["reproduced_after"] == 1 and rep["claims"]["summary"]["not_reproduced"] == 1


def test_text_without_claims_is_reported_honestly_and_the_run_continues_as_before(settings, tiny_repo, tmp_path):
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    _, rep, _ = run_with_claims(settings, tiny_repo, tmp_path, b, paper_text="We study toy problems.",
                                metric=MetricSpec(name="val_accuracy", expected=0.9, tolerance=0.02))
    assert rep["claims"]["checked"] is False and "no claim could be read" in rep["claims"]["reason"]
    assert any("produced no claim" in c for c in rep["caveats"]) and rep["status"] == "verified"


def test_runs_without_any_claims_are_unaffected(settings, tiny_repo, tmp_path):
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    _, rep, events = run_with_claims(settings, tiny_repo, tmp_path, b, metric=MetricSpec(name="val_accuracy", expected=0.9, tolerance=0.02))
    assert rep["claims"]["checked"] is False and not any(k == "claims" for k, _ in events)
    assert rep["status"] == "verified" and not any("claim" in c.lower() for c in rep["caveats"])


# --------------------------------------------------------------------------- API and CLI
def test_api_runs_a_paper_claim_and_serves_the_card(settings, tiny_repo):
    from fastapi.testclient import TestClient
    from test_api import body, make_app, wait
    with TestClient(make_app(settings)) as client:
        payload = body(tiny_repo, paper_text=PAPER_TEXT, claims=[{"metric": "val_accuracy", "value": 90.0, "unit": "percent"}])
        payload.pop("metric")
        r = client.post("/api/runs", json=payload)
        assert r.status_code == 202, r.text
        run = wait(client, r.json()["id"])
        assert run["status"] == "verified"
        card = run["report"]["claims"]
        assert card["checked"] and card["summary"]["fixed"] == 1 and card["headline"] == "c1"
        assert [i["source"] for i in card["items"]] == ["explicit"]                      # the pasted sentence duplicates it
        assert run["request"]["paper_text"] == PAPER_TEXT


@pytest.mark.parametrize("extra", [
    {"paper_text": "x" * 200_001},
    {"claims": [{"metric": "accuracy", "value": "not-a-number"}]},
    {"claims": [{"metric": "accuracy", "value": 1.0, "tolerance": -1}]},
    {"claims": [{"metric": "", "value": 1.0}]},
    {"claims": [{"metric": "accuracy", "value": 1.0}] * 31},
])
def test_api_rejects_malformed_claims_before_any_work(settings, tiny_repo, extra):
    from fastapi.testclient import TestClient
    from test_api import body, make_app
    with TestClient(make_app(settings)) as client:
        assert client.post("/api/runs", json=body(tiny_repo, **extra)).status_code in (400, 422)


def test_claims_command_shows_what_would_be_read_and_runs_nothing(tmp_path, capsys, monkeypatch):
    from reprofix.cli import main
    monkeypatch.setenv("REPROFIX_ENV_FILE", "")
    paper = tmp_path / "paper.txt"
    paper.write_text(PAPER)
    assert main(["claims", str(paper)]) == 0
    out = capsys.readouterr().out
    assert "*c1 top-1 accuracy = 76.4%" in out and "comes after a comparison" in out and "none is labelled 'ours'" in out
    assert "pattern matches" in out
    empty = tmp_path / "empty.txt"
    empty.write_text("nothing here")
    assert main(["claims", str(empty)]) == 1 and "no claims found" in capsys.readouterr().out


def test_run_rejects_a_claim_it_cannot_read(monkeypatch):
    from reprofix.cli import main
    monkeypatch.setenv("REPROFIX_ENV_FILE", "")
    monkeypatch.setenv("NEBIUS_API_KEY", "k")
    with pytest.raises(SystemExit) as e:
        main(["run", "--path", ".", "--claim", "accuracy"])
    assert "cannot read a claim" in str(e.value)


@pytest.mark.parametrize("text", [
    "It outperforms the baseline by 2.1% accuracy.",
    "A +2.1% accuracy gain over the baseline.",
    "We see a 2.1% accuracy improvement.",
])
def test_a_margin_between_results_is_not_a_claim_and_is_listed_as_skipped(text):
    claims, skipped = extract_claims(text)
    assert claims == []
    assert len(skipped) == 1 and "margin" in skipped[0] and "line 1" in skipped[0]


def test_a_gain_stated_after_the_metric_word_is_never_read_as_a_result():
    assert extract_claims("an accuracy gain of 2.1% over ResNet")[0] == []


def test_a_margin_does_not_hide_the_result_in_the_same_sentence():
    claims, skipped = extract_claims("We reach 76.4% top-1 accuracy, outperforming ResNet by 2.3% accuracy.")
    assert [c.value for c in claims] == [76.4]
    assert len(skipped) == 1 and "2.3%" in skipped[0]


# --------------------------------------------------------------------------- found by independent review
def _measure(claim_metric, out, value=76.4):
    return measure(PaperClaim(metric=claim_metric, value=value, unit="percent"), parse_outputs(out))


@pytest.mark.parametrize("out", ["accuracy: 0.764\naccuracy: nan\n", "accuracy: 0.764\naccuracy: inf\n", "accuracy: 0.764\naccuracy: N/A\n",
                                 "accuracy: 0.764\naccuracy: -inf\n", "accuracy: 0.764\naccuracy: 1e999\n", "accuracy: 0.764\naccuracy = None\n"])
def test_a_run_that_ends_in_a_non_number_measured_nothing_even_if_an_earlier_epoch_looked_right(out):
    m = _measure("accuracy", out)
    assert m.status == "missing" and "not a finite number" in m.reason and "line 2" in m.reason
    from reprofix.claims.check import claim_value
    assert claim_value(out, PaperClaim(metric="accuracy", value=76.4, unit="percent")) is None       # so the repair loop sees no metric either


def test_a_later_good_value_after_a_nan_is_still_the_measurement():
    m = _measure("accuracy", "accuracy: nan\naccuracy: 0.764\n")
    assert m.status == "found" and m.output.value == 0.764


@pytest.mark.parametrize("printed", ["accuracy_gap", "accuracy_std", "worst_group_accuracy", "target_accuracy", "expected accuracy",
                                     "accuracy@5", "zero_shot_accuracy", "balanced_accuracy", "macro_accuracy", "ema_accuracy"])
def test_a_derived_or_different_quantity_is_not_the_claimed_metric(printed):
    m = _measure("accuracy", f"{printed}: 0.764\n")
    assert m.status == "missing" and printed.split("@")[0].split()[0] in m.reason


@pytest.mark.parametrize("printed", ["cifar10_accuracy", "test_set_accuracy", "final_test_accuracy", "imagenet val acc", "accuracy"])
def test_a_dataset_name_or_split_on_the_printed_name_does_not_change_the_quantity(printed):
    assert _measure("accuracy", f"{printed}: 0.764\n").status == "found"


def test_loss_is_not_matched_to_loss_scale_or_loss_weight():
    for printed in ("loss_scale: 65536", "loss_weight: 0.31", "aux_loss_weight: 0.5"):
        assert measure(PaperClaim(metric="loss", value=0.31), parse_outputs(printed + "\n")).status == "missing", printed
    assert measure(PaperClaim(metric="loss", value=0.31), parse_outputs("test_loss: 0.31\n")).status == "found"


def test_a_qualified_claim_is_not_matched_to_the_plain_metric():
    assert _measure("balanced accuracy", "accuracy: 0.764\n").status == "missing"
    assert _measure("balanced accuracy", "balanced_accuracy: 0.764\n").status == "found"
    assert _measure("macro F1", "f1: 0.764\n").status == "missing"
    assert _measure("macro F1", "macro_f1: 0.764\n").status == "found"


@pytest.mark.parametrize("text", [
    "less than 1% accuracy loss", "with <1% accuracy degradation", "a 1.2% accuracy lift", "a 1.2% accuracy jump", "a 1.2% accuracy decline",
    "accuracy is 2.3% lower than the baseline", "accuracy is 2.3% higher than the baseline", "an increase in accuracy of 2.3%",
    "an improvement in accuracy of 2.3%", "within 2.3% accuracy of the baseline", "at least 90% accuracy", "above 90% accuracy",
    "more than 90% accuracy",
])
def test_margins_bounds_and_comparisons_are_never_read_as_results(text):
    claims, _ = extract_claims(text)
    assert claims == [], (text, [(c.metric, c.value) for c in claims])


@pytest.mark.parametrize("text", ["accuracy of 74-76%", "accuracy of 74 to 76%", "accuracy of 0.74-0.76", "accuracy of 74–76%", "74-76% accuracy"])
def test_a_range_is_not_read_as_its_first_number(text):
    claims, skipped = extract_claims(text)
    assert [c.value for c in claims if c.value in (74.0, 0.74)] == [], text
    assert skipped and "range" in skipped[0] or not claims


@pytest.mark.parametrize("text", ["recall at 10 of 0.9", "mAP at 0.5 IoU is 45.1", "precision at 5 is 0.8"])
def test_a_value_at_a_threshold_is_not_read_as_a_result(text):
    claims, _ = extract_claims(text)
    assert not any(c.value in (10.0, 0.5, 5.0, 45.1) for c in claims), (text, [(c.metric, c.value) for c in claims])


def test_a_modified_metric_keeps_its_modifier_so_it_is_not_taken_for_the_plain_one():
    claims, _ = extract_claims("We reach a balanced accuracy of 0.9 and a worst-group accuracy of 71.2%.")
    assert [(c.metric, c.value) for c in claims] == [("balanced accuracy", 0.9), ("worst-group accuracy", 71.2)]


def test_numbers_are_read_whole_or_not_at_all():
    for text in ["accuracy of 12.5 hours", "accuracy of 90K", "accuracy of 76,4%"]:
        assert extract_claims(text)[0] == [], text
    assert [c.value for c in extract_claims("We reach an accuracy of 76.4.")[0]] == [76.4]


def test_a_long_run_of_digits_cannot_stall_the_readers():
    import time
    t = time.time()
    extract_claims(("9" * 1499 + "\n") * 130)
    parse_outputs("accuracy: " + "9" * 1400 + "\n" + ("x: " + "1" * 1400 + "\n") * 200)
    assert time.time() - t < 2.0


def test_a_fraction_such_as_45_of_50_is_not_read_as_45():
    assert parse_outputs("accuracy: 45/50 (90.0%)\n") == []


def test_the_headline_is_never_a_number_that_follows_a_comparison():
    from reprofix.claims.extract import pick_headline
    claims, _ = extract_claims("We outperform the baseline (accuracy of 70%) with an accuracy of 76.4%.")
    assert all(c.note for c in claims) and pick_headline(claims) is None
    own, _ = extract_claims("We reach an accuracy of 76.4%, outperforming ResNet (accuracy of 74.1%).")
    assert pick_headline(own) == 0


def test_when_every_claim_follows_a_comparison_nothing_is_the_repair_target_and_the_report_says_so(settings, tiny_repo, tmp_path):
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    _, rep, _ = run_with_claims(settings, tiny_repo, tmp_path, b, paper_text="We outperform the baseline (accuracy of 70%) here.",
                                metric=None)
    assert rep["claims"]["headline"] is None and not any(i["headline"] for i in rep["claims"]["items"])
    assert any("No claim was chosen as the repair target" in c for c in rep["caveats"])


TABLE_DELTA = "| Model | Acc | Δ Acc |\n|---|---|---|\n| ResNet | 70.1 | 0.0 |\n| Ours | 76.4 | +6.3 |\n"
TABLE_ABLATION = "| Model | Top-1 (%) |\n|---|---|\n| ResNet | 70.1 |\n| Ours | 76.4 |\n| Ours w/o aug | 74.0 |\n| Ours (small) | 72.0 |\n"


def test_a_difference_column_is_not_a_result():
    claims, skipped = extract_claims(TABLE_DELTA)
    assert [(c.metric, c.value) for c in claims] == [("Acc", 76.4)]
    assert any("Δ Acc" in s and "difference" in s for s in skipped)


def test_several_rows_labelled_ours_are_not_guessed_between():
    claims, skipped = extract_claims(TABLE_ABLATION)
    assert claims == [] and any("3 rows labelled 'ours'" in s for s in skipped)
    one, _ = extract_claims("| Model | Top-1 (%) |\n|---|---|\n| Ours | 76.4 |\n| Ours w/o aug | 74.0 |\n")
    assert [c.value for c in one] == [76.4]                        # the ablation is not a second "ours"


def test_two_claims_with_one_name_and_different_values_say_so():
    cs = collect(RunRequest(local_path="/x", claims=[PaperClaim(metric="accuracy", value=70.0, unit="percent"),
                                                     PaperClaim(metric="accuracy", value=76.4, unit="percent")]))
    assert cs.claims[0].note == "" and "same metric name as c1 with a different value" in cs.claims[1].note


def test_a_run_that_ends_before_reading_the_claims_does_not_claim_none_were_given():
    from reprofix.claims import ClaimSet, report_card
    card = report_card(ClaimSet(collected=False), None, None)
    assert card["checked"] is False and "ended before the claims were read" in card["reason"]
    assert "no paper text or claims were given" in report_card(ClaimSet(), None, None)["reason"]


def test_naming_the_metric_yourself_means_no_claim_is_starred_as_the_repair_target(settings, tiny_repo, tmp_path):
    b = scripted([hypothesis_reply("h1", "floor division truncates the accuracy to zero", BUG_LINE)], [repair_reply(BUG_LINE, FIX_LINE)])
    _, rep, _ = run_with_claims(settings, tiny_repo, tmp_path, b, paper_text=PAPER_TEXT,
                                metric=MetricSpec(name="val_accuracy", expected=0.9, tolerance=0.02))
    assert rep["claims"]["headline"] is None and not any(i["headline"] for i in rep["claims"]["items"])
    assert rep["metric"]["source"] == "user"
