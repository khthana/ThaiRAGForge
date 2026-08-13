"""The ColBERT pilot's continuation rule, pinned on every branch.

The rule decides whether ~3x more GPU time is spent, and it was frozen before the
artifact existed. A rule that lives only in prose is a rule that gets re-read
favourably once a number is on the table, so each branch is asserted here --
including the two that are easy to get backwards:

* STOP is evaluated **before** significance. At n=30 per entity type a 0.05 loss
  can fail to reach significance, and "we could not resolve it" is not a reason
  to spend three more chunkers on it.
* "clears" is the prediction's own wording, *ties or beats* -- so a loss that is
  not significant clears, and only a significant loss fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))

pytest.importorskip("pyarrow")

import colbert_pilot  # noqa: E402
from colbert_pilot import STOP_MARGIN, clears, decide, rider_for  # noqa: E402


def cell(label, diff, significant):
    return {"label": label, "diff": diff, "significant": significant,
            "queries": ["q"]}


def both(person, program):
    return [cell("`person` vs bm25", *person), cell("`program` vs dense", *program)]


# ------------------------------------------------------------------- clears
def test_a_win_clears_whether_or_not_it_is_significant():
    assert clears(0.0400, significant=True)
    assert clears(0.0400, significant=False)


def test_an_exact_tie_clears():
    assert clears(0.0, significant=False)


def test_an_unresolved_loss_clears_because_the_prediction_says_ties_or_beats():
    assert clears(-0.0200, significant=False)


def test_a_significant_loss_does_not_clear():
    assert not clears(-0.0200, significant=True)


# ------------------------------------------------------------------- decide
def test_both_cells_clearing_continues():
    verdict, why = decide(both((0.0120, False), (0.0310, True)))
    assert verdict == "CONTINUE"
    assert "clear" in why


def test_one_significant_loss_inside_the_margin_narrows():
    verdict, why = decide(both((-0.0300, True), (0.0100, False)))
    assert verdict == "NARROW"
    assert "person" in why


def test_a_loss_beyond_the_margin_stops_even_when_not_significant():
    """The branch the rule exists for: a 0.05 loss at n=30 need not reach
    significance, and an unresolved loss that large is still not worth 3x the GPU."""
    verdict, why = decide(both((-0.0900, False), (0.0500, True)))
    assert verdict == "STOP"
    assert "person" in why


def test_stop_overrides_a_second_cell_that_clears_easily():
    assert decide(both((-0.3000, True), (0.2000, True)))[0] == "STOP"


def test_the_margin_is_exclusive_so_exactly_the_margin_is_not_a_stop():
    """A boundary that decides GPU spend should not be ambiguous: `> 0.05` stops,
    `== 0.05` does not."""
    assert decide(both((-STOP_MARGIN, True), (0.01, False)))[0] == "NARROW"
    assert decide(both((-STOP_MARGIN - 1e-9, True), (0.01, False)))[0] == "STOP"


def test_a_significant_loss_on_program_alone_still_blocks_a_continue():
    """The prediction is a conjunction -- an aggregate win cannot buy one cell."""
    assert decide(both((0.1000, True), (-0.0400, True)))[0] == "NARROW"


# ------------------------------------------------- the unmeasured cell (NaN)
# Found by running `--smoke`, not by reading the rule: the 8-query slice left
# `person` with 0 paired queries, the bootstrap returned NaN, and the verdict
# came out CONTINUE. Every comparison against NaN is False, so `diff >= 0` was
# False and `not significant` was True -- an unmeasured cell read as a *tie*,
# which is the one reading a conjunction cannot survive.
def test_an_unmeasured_cell_is_invalid_not_a_tie():
    assert decide(both((float("nan"), False), (0.0310, True)))[0] == "INVALID"


def test_invalid_beats_even_a_stop_because_the_rule_is_not_applicable():
    """Order matters: with one cell unmeasured the conjunction has no truth value,
    so no branch of the rule may be reported -- not even the cheap one."""
    verdict, why = decide(both((float("nan"), False), (-0.9000, True)))
    assert verdict == "INVALID"
    assert "person" in why and "program" not in why


def test_an_unmeasured_cell_would_otherwise_have_cleared():
    """States the defect the INVALID branch exists for, so a future refactor that
    drops the branch fails loudly instead of silently restoring the old reading."""
    assert clears(float("nan"), significant=False)


def test_both_cells_unmeasured_names_both():
    verdict, why = decide(both((float("nan"), False), (float("nan"), False)))
    assert verdict == "INVALID"
    assert "person" in why and "program" in why


# ------------------------------------------------------------- the length rider
# `DECISION_RULE`'s 512/48 fallback fires "only if the losing cell's truncation is
# materially above the corpus rate" -- and choosing what counts as material after
# the gap is on the table is the favourable re-reading the frozen rule exists to
# prevent. It is therefore answered as an arithmetic bound, and these pin the gate
# rather than the arithmetic: the bound itself needs the real tokenizer.
@pytest.fixture()
def stub_bound(monkeypatch):
    """Replace the tokenizing half so the gate can be tested without a model."""
    def _set(bound):
        monkeypatch.setattr(colbert_pilot, "truncation_rider",
                            lambda *a, **k: {"recall_damage_bound": bound})
    return _set


def test_the_rider_does_not_run_on_a_continue(stub_bound):
    """A CONTINUE has no losing cell, so asking whether truncation explains a loss
    is answering a question the rule does not ask."""
    stub_bound(1.0)
    assert rider_for("CONTINUE", both((0.0120, False), (0.0310, True)),
                     [], {}, 300) is None


def test_the_rider_runs_on_a_stop_and_picks_the_worst_cell(stub_bound):
    stub_bound(0.0)
    r = rider_for("STOP", both((-0.0100, True), (-0.3331, True)), [], {}, 300)
    assert r is not None and "program" in r["cell"]
    assert r["gap"] == pytest.approx(0.3331)


def test_the_rider_runs_on_a_narrow_too(stub_bound):
    stub_bound(0.0)
    assert rider_for("NARROW", both((0.1000, True), (-0.0400, True)),
                     [], {}, 300) is not None


def test_a_bound_below_the_gap_cannot_explain_it(stub_bound):
    stub_bound(0.0500)
    assert rider_for("STOP", both((0.0308, False), (-0.3331, True)),
                     [], {}, 300)["fires"] is False


def test_a_bound_reaching_the_gap_fires(stub_bound):
    """`>=`, not `>`: a bound that exactly accounts for the gap is enough to make
    truncation a live explanation, and the fallback is the conservative answer."""
    stub_bound(0.3331)
    assert rider_for("STOP", both((0.0308, False), (-0.3331, True)),
                     [], {}, 300)["fires"] is True


def test_the_margin_is_the_one_this_rule_was_justified_against():
    """0.05 is justified in the docstring against `power_analysis.md` (every
    observed chunker-pair diff |d| <= 0.0230, MDEs 0.0302-0.0532) and the measured
    bar spreads (0.0283 / 0.0346). Changing it silently would decouple the number
    from its justification."""
    assert STOP_MARGIN == 0.05
