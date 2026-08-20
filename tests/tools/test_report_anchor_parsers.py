"""The three cross-artifact anchors that read a report instead of freezing it.

All three were literals until 2026-08-18, when rebuild #4 legitimately moved
every number they anchored on and two scripts exited 1 against reports they in
fact agreed with. Reading the artifact fixes that, but a parser is only an
anchor while it keeps finding its counterpart -- a silently-failing one is
strictly worse than a literal, because it turns a wrong answer into no answer
and the check into a vacuous pass.

So every parser is pinned in BOTH directions: it finds the right figure in a
report shaped like the real one, and it returns None (never a plausible wrong
number) when the report moves. The scoping test for `parse_hybrid_anchors` is
the one that matters most -- routing_eval.md really does carry two sections
whose rows are labelled identically, and the dense one comes first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))

from miss_depth_profile import parse_ceiling_anchors  # noqa: E402
from qdrant_routed_check import parse_f200  # noqa: E402
from routed_fetch_depth_test import parse_hybrid_anchors  # noqa: E402


# --------------------------------------------------------------------------
# oracle_union_ceiling.md -> (hybrid-only unfound, all-arm unfound)
# --------------------------------------------------------------------------

CEILING = """\
# Oracle-union ceiling

## 2. สรุป

- คู่ทั้งหมด: **1046** · ระบบที่ดีที่สุดเจอ **509** (48.7%) · union ของทุกระบบเจอ **873** (83.5%)

## 6. ถ้า union ข้าม retriever ด้วย

| arm | queries | pairs | macro | micro | ไม่มีใครเจอ |
|---|---|---|---|---|---|
| hybrid only | 61 | 700 | 0.8948 | 0.8595 | 173 |
| hybrid + dense + BM25 (+4) | 76 | 760 | 0.9418 | 0.9130 | 91 |
"""


def test_ceiling_anchors_parse_both_figures():
    assert parse_ceiling_anchors(CEILING) == (173, 91)


def test_ceiling_hybrid_figure_is_the_subtraction_not_the_total():
    """§2 never states the hybrid count; it states total and union."""
    exp_hyb, _ = parse_ceiling_anchors(CEILING)
    assert exp_hyb == 1046 - 873
    assert exp_hyb not in (1046, 873)


def test_ceiling_anchors_handle_thousands_separators():
    text = CEILING.replace("**1046**", "**1,046**").replace("| 91 |", "| 1,091 |")
    assert parse_ceiling_anchors(text) == (173, 1091)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        # §2 summary line reworded -> hybrid figure unavailable, table still parses
        ("union ของทุกระบบเจอ", (None, 91)),
        # §6 row label changed -> all-arm figure unavailable, summary still parses
        ("hybrid + dense + BM25", (173, None)),
    ],
)
def test_ceiling_anchors_return_none_when_the_report_moves(mutation, expected):
    assert parse_ceiling_anchors(CEILING.replace(mutation, "RENAMED")) == expected


def test_ceiling_anchors_on_empty_text():
    """A missing report must not look like a report that says zero."""
    assert parse_ceiling_anchors("") == (None, None)


# --------------------------------------------------------------------------
# routing_eval.md -> {'routed', 'unrouted'}, scoped to the HYBRID section
# --------------------------------------------------------------------------

ROUTING_EVAL = """\
# Routing eval

## 3. Routed system vs single-combo baselines -- dense

| metric | arm | value | Δ |
|---|---|---|---|
| recall@10 | best single combo = semantic+qwen3_0.6b | 0.5673 | +0.0000 |
| recall@10 | routed (shipped) | 0.6173 | +0.0500 |

## 3. Routed system vs single-combo baselines -- hybrid

| metric | arm | value | Δ |
|---|---|---|---|
| recall@10 | best single combo = fixed_size+bge_m3 | 0.6229 | +0.0000 |
| recall@10 | routed (shipped) | 0.6811 | +0.0582 |
| mrr | routed (shipped) | 0.8412 | +0.0080 |

## 4. Paired bootstrap, routed vs baseline -- hybrid

| metric | arm | value | Δ |
|---|---|---|---|
| recall@10 | routed (shipped) | 0.9999 | +0.9999 |
"""


def test_hybrid_anchors_take_the_hybrid_section_not_the_dense_one():
    """The dense section comes first and its rows are labelled identically."""
    assert parse_hybrid_anchors(ROUTING_EVAL) == {
        "routed": 0.6811,
        "unrouted": 0.6229,
    }


def test_hybrid_anchors_stop_at_the_next_heading():
    """§4 repeats `recall@10 | routed (shipped)` with a different number."""
    assert parse_hybrid_anchors(ROUTING_EVAL)["routed"] != 0.9999


def test_hybrid_anchors_read_recall_not_whichever_metric_comes_last():
    """`mrr | routed (shipped)` sits below the recall row in the same section."""
    assert parse_hybrid_anchors(ROUTING_EVAL)["routed"] != 0.8412


def test_hybrid_anchors_none_when_the_section_is_renamed():
    text = ROUTING_EVAL.replace(
        "## 3. Routed system vs single-combo baselines -- hybrid", "## 3. Renamed")
    assert parse_hybrid_anchors(text) == {"routed": None, "unrouted": None}


def test_hybrid_anchors_none_when_the_row_label_is_renamed():
    text = ROUTING_EVAL.replace("| routed (shipped) |", "| routed (v2) |")
    out = parse_hybrid_anchors(text)
    assert out["routed"] is None
    assert out["unrouted"] == 0.6229  # the other half still anchors


def test_hybrid_anchors_on_empty_text():
    assert parse_hybrid_anchors("") == {"routed": None, "unrouted": None}


# --------------------------------------------------------------------------
# routed_fetch_depth_test.md -> the F=200 routed macro recall@10
# --------------------------------------------------------------------------

FETCH_DEPTH = """\
# Routed fetch depth

## 1. ระบบที่ ship จริง (routed)

| F | top-10 เหมือน k=n เป๊ะ | recall@10 | Δ | MRR | Δ | nDCG@10 | Δ |
|---|---|---|---|---|---|---|---|
| 20 | 16/106 (15.1%) | 0.6392 | -0.0419 | 0.8711 | +0.0093 | 0.7053 | -0.0420 |
| 200 | 89/106 (84.0%) | 0.6815 | +0.0005 | 0.8594 | -0.0024 | 0.7451 | -0.0022 |
| 1,000 | 98/106 (92.5%) | 0.6784 | -0.0027 | 0.8618 | +0.0000 | 0.7452 | -0.0021 |
| 5,000 | 100/106 (94.3%) | 0.6792 | -0.0019 | 0.8618 | +0.0000 | 0.7464 | -0.0009 |
"""


def test_f200_takes_the_row_whose_depth_cell_is_exactly_200():
    assert parse_f200(FETCH_DEPTH) == 0.6815


def test_f200_does_not_match_a_depth_that_merely_contains_200():
    """`2,000` and `1,200` must not stand in for `200`."""
    text = FETCH_DEPTH.replace("| 200 |", "| 1,200 |").replace(
        "| 5,000 |", "| 2,000 |")
    assert parse_f200(text) is None


def test_f200_none_when_the_row_is_gone():
    text = "\n".join(l for l in FETCH_DEPTH.splitlines() if not l.startswith("| 200 "))
    assert parse_f200(text) is None


def test_f200_none_on_empty_text():
    assert parse_f200("") is None
