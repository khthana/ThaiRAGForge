"""The seven cross-artifact anchors the 2026-08-20 reranker refresh exposed.

Every one was a frozen literal until that day, and rebuild #4 moved every number
they anchored on. Four of them (`reranker_model_comparison.PUBLISHED`) were the
dangerous kind: `head`, the denominator of that report's own "captured ceiling"
column, read from the same dict, so a stale literal produced a *silently* wrong
percentage on a fresh run rather than a red check.

A parser is only an anchor while it keeps finding its counterpart, so each is
pinned in BOTH directions: it finds the right figure in a report shaped like the
real one, and it returns None/{} -- never a plausible wrong number -- when the
report moves. Three column-choice tests carry most of the weight, because in each
case an adjacent column is a real, plausible number answering a different
question:

  * `parse_miss_depth_delivered` must take *delivered*, not *in pool*: a pool
    figure legitimately exceeds the qrels ceiling, so anchoring on it gives a
    check that cannot fail.
  * `parse_pool_source_truncate` must take the real arm, not the *oracle* beside it.
  * `parse_routed_oracle` must take the *routed* delivered column, not the
    *unrouted* one two cells to its right.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))

from reranker_pool_source_test import parse_miss_depth_delivered  # noqa: E402
from reranker_rrf_signal_test import parse_pool_source_truncate  # noqa: E402
from reranker_rrf_routed_test import (  # noqa: E402
    parse_pool_source_oracle,
    parse_routed_arms,
    parse_routed_oracle,
    parse_routing_eval_routed,
    parse_rrf_signal_arms,
)

# --------------------------------------------------------------------------
# miss_depth_profile.md section 2 -> {P: delivered}
# The two "in pool" columns are deliberately larger than the delivered ones.
# --------------------------------------------------------------------------
MISS_DEPTH = """\
# miss depth

## 1. how deep

| rank | pairs | % |
|---|---|---|
| 11-50 | 71 | 78.0% |

## 2. pool depth

| P | single in pool | single **delivered** | all arms in pool | all arms **delivered** |
|---|---|---|---|---|
| 10 | 0.6229 | **0.6229** | 0.9418 | **0.8605** |
| 50 | 0.8896 | **0.8268** | 0.9837 | **0.8783** |
| 1000 | 0.9798 | **0.8738** | 0.9990 | **0.8846** |

## 3. by type
"""


def test_miss_depth_takes_the_delivered_column_not_the_pool_one():
    got = parse_miss_depth_delivered(MISS_DEPTH)
    assert got[50] == 0.8268
    # 0.8896 is the in-pool figure on the same row and exceeds the qrels
    # ceiling: anchoring on it would make the caller's check unfailable.
    assert 0.8896 not in got.values()


def test_miss_depth_reads_every_depth():
    assert set(parse_miss_depth_delivered(MISS_DEPTH)) == {10, 50, 1000}


def test_miss_depth_stops_at_the_next_section():
    """Section 1's rows have a different width and must not be absorbed."""
    assert 11 not in parse_miss_depth_delivered(MISS_DEPTH)


def test_miss_depth_empty_when_the_section_is_renamed():
    assert parse_miss_depth_delivered(MISS_DEPTH.replace("## 2.", "## 2b.")) == {}


def test_miss_depth_empty_on_empty_text():
    assert parse_miss_depth_delivered("") == {}


# --------------------------------------------------------------------------
# reranker_pool_source_test.md -> truncate-and-replace, per pool source
# Both tables are identically shaped; only the caption tells them apart.
# --------------------------------------------------------------------------
POOL_SOURCE = """\
# pool source

**pool จาก `dense`**

| P | ใน pool | oracle | จริง | จับได้ | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| 20 | 0.6085 | 0.5875 | **0.5068** | 3% | 0.7397 | 0.5813 |
| 50 | 0.7371 | 0.6798 | **0.5086** | 3% | 0.6739 | 0.5532 |

**pool จาก `hybrid`**

| P | ใน pool | oracle | จริง | จับได้ | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| 20 | 0.7720 | 0.7510 | **0.6464** | 18% | 0.7764 | 0.6913 |
| 50 | 0.8896 | 0.8268 | **0.6182** | -2% | 0.7233 | 0.6465 |

## next
"""


def test_pool_source_truncate_selects_the_named_pool():
    """The two tables are the same shape -- a parser that ignores the caption
    silently answers about the other retriever."""
    assert parse_pool_source_truncate(POOL_SOURCE, "hybrid")[50]["recall@10"] == 0.6182
    assert parse_pool_source_truncate(POOL_SOURCE, "dense")[50]["recall@10"] == 0.5086


def test_pool_source_truncate_takes_the_real_arm_not_the_oracle():
    got = parse_pool_source_truncate(POOL_SOURCE, "hybrid")[50]
    assert got["recall@10"] == 0.6182
    assert got["recall@10"] != 0.8268  # the oracle column, one cell left


def test_pool_source_truncate_carries_the_other_metrics():
    got = parse_pool_source_truncate(POOL_SOURCE, "hybrid")[50]
    assert got["mrr"] == 0.7233 and got["ndcg@10"] == 0.6465


def test_pool_source_truncate_empty_for_an_unknown_pool():
    assert parse_pool_source_truncate(POOL_SOURCE, "bm25") == {}


def test_pool_source_truncate_empty_on_empty_text():
    assert parse_pool_source_truncate("", "hybrid") == {}


def test_pool_source_oracle_takes_delivered_and_holds():
    delivered, holds = parse_pool_source_oracle(POOL_SOURCE, "hybrid", 50)
    assert (delivered, holds) == (0.8268, 0.8896)


def test_pool_source_oracle_none_for_a_depth_that_is_absent():
    assert parse_pool_source_oracle(POOL_SOURCE, "hybrid", 200) == (None, None)


def test_pool_source_oracle_none_when_the_caption_moves():
    text = POOL_SOURCE.replace("**pool จาก `hybrid`**", "**hybrid pool**")
    assert parse_pool_source_oracle(text, "hybrid", 50) == (None, None)


# --------------------------------------------------------------------------
# reranker_rrf_routed_test.md -> the 2x2 arms and the routed oracle
# --------------------------------------------------------------------------
ROUTED = """\
# rrf4 routed

## ตาราง 2×2

| arm | routing | rrf4 | recall@10 | MRR | nDCG@10 | ดึง / ส่ง |
|---|---|---|---|---|---|---|
| A | ไม่มี | ไม่มี | 0.6229 | 0.8478 | 0.6961 | 10 / 10 |
| B | ไม่มี | มี | 0.6622 | 0.8257 | 0.7163 | 50 / 10 |
| C | hard | ไม่มี | 0.6811 | 0.8618 | 0.7473 | 10 / 10 |
| **D** | hard | มี | **0.6713** | 0.8665 | 0.7402 | 50 / 10 |
| D′ (oracle w=0.10) | hard | มี | 0.6867 | 0.8767 | 0.7551 | 50 / 10 |

## มีอะไรให้ได้อยู่จริงมั้ย

| P | routed: pool มี | routed: oracle ส่งมอบ | เหนือ arm C | unrouted: oracle ส่งมอบ |
|---|---|---|---|---|
| 20 | 0.8135 | **0.7891** | +0.1080 | 0.7510 |
| 50 | 0.9054 | **0.8331** | +0.1520 | 0.8268 |

## self-check
"""


def test_routed_arms_reads_all_four():
    assert parse_routed_arms(ROUTED) == {
        "A": 0.6229, "B": 0.6622, "C": 0.6811, "D": 0.6713}


def test_routed_arms_skips_the_oracle_row():
    """D-prime is a bound, not an arm; its label is not a bare letter."""
    assert 0.6867 not in parse_routed_arms(ROUTED).values()


def test_routed_arms_empty_when_the_heading_moves():
    text = ROUTED.replace("## ตาราง 2×2", "## arms")
    assert parse_routed_arms(text) == {}


def test_routed_arms_empty_on_empty_text():
    assert parse_routed_arms("") == {}


def test_routed_oracle_takes_the_routed_column_not_the_unrouted_one():
    delivered, holds = parse_routed_oracle(ROUTED, 50)
    assert (delivered, holds) == (0.8331, 0.9054)
    # 0.8268 is the UNROUTED delivered oracle on the same row -- a real,
    # plausible number that answers a different question.
    assert delivered != 0.8268


def test_routed_oracle_none_for_an_absent_depth():
    assert parse_routed_oracle(ROUTED, 100) == (None, None)


def test_routed_oracle_none_when_the_heading_moves():
    text = ROUTED.replace("## มีอะไร", "## เพดาน")
    assert parse_routed_oracle(text, 50) == (None, None)


# --------------------------------------------------------------------------
# routing_eval.md -> the routed arm, per retriever
# The two sections are identically shaped and the DENSE one comes first.
# --------------------------------------------------------------------------
ROUTING_EVAL = """\
# routing eval

## 3. Routed system vs single-combo baselines -- dense

| metric | arm | value | n |
|---|---|---|---|
| recall@10 | best single | 0.5673 | 106 |
| recall@10 | routed (shipped) | 0.6173 | 106 |
| recall@10 | routed (loo) | 0.5989 | 106 |

## 3. Routed system vs single-combo baselines -- hybrid

| metric | arm | value | n |
|---|---|---|---|
| recall@10 | best single | 0.6229 | 106 |
| recall@10 | routed (shipped) | 0.6811 | 106 |
| recall@10 | routed (loo) | 0.6794 | 106 |
"""


def test_routing_eval_selects_the_hybrid_section():
    """The dense section comes first and its rows are labelled identically."""
    assert parse_routing_eval_routed(ROUTING_EVAL, "hybrid", "recall@10") == 0.6811
    assert parse_routing_eval_routed(ROUTING_EVAL, "dense", "recall@10") == 0.6173


def test_routing_eval_none_for_an_unknown_retriever():
    assert parse_routing_eval_routed(ROUTING_EVAL, "bm25", "recall@10") is None


def test_routing_eval_none_for_an_unmeasured_metric():
    """Never the neighbouring row's number for a metric the table lacks."""
    assert parse_routing_eval_routed(ROUTING_EVAL, "hybrid", "map") is None


def test_routing_eval_none_when_the_heading_is_renamed():
    text = ROUTING_EVAL.replace("## 3. Routed system vs single-combo baselines",
                                "## 3. Routing")
    assert parse_routing_eval_routed(text, "hybrid", "recall@10") is None


def test_routing_eval_none_on_empty_text():
    assert parse_routing_eval_routed("", "hybrid", "recall@10") is None


# --------------------------------------------------------------------------
# reranker_rrf_signal_test.md -> the deployable arms
# --------------------------------------------------------------------------
RRF_SIGNAL = """\
# rrf4 signal

## arm ที่ deploy ได้ — w เลือกแบบ leave-one-out

| arm | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| hybrid (shipped) | 0.6229 | 0.8478 | 0.6961 |
| dense | 0.5041 | 0.7939 | 0.5982 |
| truncate-and-replace (w=1.00) | 0.6182 | 0.7233 | 0.6465 |
| **rrf4 (loo)** | **0.6622** | 0.8257 | 0.7163 |

## next
"""


def test_rrf_signal_arms_reads_the_two_the_caller_anchors_on():
    got = parse_rrf_signal_arms(RRF_SIGNAL)
    assert got["hybrid (shipped)"] == 0.6229
    assert got["rrf4 (loo)"] == 0.6622


def test_rrf_signal_arms_empty_when_the_heading_moves():
    """The first version scanned the WHOLE file for 4-cell rows, so it returned
    a full dict from a report whose section had been renamed -- unscoped, it
    could not fail. Scoping it is what makes this assertion meaningful."""
    text = RRF_SIGNAL.replace("## arm ที่ deploy ได้", "## deployable arms")
    assert parse_rrf_signal_arms(text) == {}


def test_rrf_signal_arms_ignores_a_four_cell_table_elsewhere():
    """A new 4-cell table outside the section must not overwrite an arm."""
    text = RRF_SIGNAL + chr(10).join([
        "", "## cost", "", "| arm | ms | x | y |", "|---|---|---|---|",
        "| rrf4 (loo) | 486 | 1 | 2 |", ""])
    assert parse_rrf_signal_arms(text)["rrf4 (loo)"] == 0.6622


def test_rrf_signal_arms_empty_on_empty_text():
    assert parse_rrf_signal_arms("") == {}


# --------------------------------------------------------------------------
# Each parser must still find the LIVE artifact -- an anchor that stops
# matching the real report is a vacuous pass, not a check.
# --------------------------------------------------------------------------
RESULTS = REPO / "data" / "results"


def _live(name: str) -> str:
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} not built in this checkout")
    return p.read_text(encoding="utf-8")


def test_live_miss_depth():
    got = parse_miss_depth_delivered(_live("miss_depth_profile.md"))
    assert 50 in got and 0.0 < got[50] <= 1.0


def test_live_pool_source():
    text = _live("reranker_pool_source_test.md")
    assert parse_pool_source_truncate(text, "hybrid")
    delivered, holds = parse_pool_source_oracle(text, "hybrid", 50)
    assert delivered is not None and holds is not None and holds >= delivered


def test_live_routed():
    text = _live("reranker_rrf_routed_test.md")
    arms = parse_routed_arms(text)
    assert set(arms) == {"A", "B", "C", "D"}
    delivered, holds = parse_routed_oracle(text, 50)
    assert delivered is not None and holds >= delivered


def test_live_routing_eval():
    text = _live("routing_eval.md")
    hyb = parse_routing_eval_routed(text, "hybrid", "recall@10")
    den = parse_routing_eval_routed(text, "dense", "recall@10")
    assert hyb is not None and den is not None and hyb != den


def test_live_rrf_signal():
    got = parse_rrf_signal_arms(_live("reranker_rrf_signal_test.md"))
    assert "rrf4 (loo)" in got and "hybrid (shipped)" in got
