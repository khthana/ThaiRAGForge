"""Both directions of the pilot-baseline self-checks, without touching results.

All five checks PASS on live data, so their failing branches are unexercised
there -- the same reason `test_audit_g1_prompt_fit.py` exists. Two of them are
load-bearing for whether the pilot is scored honestly at all:

* **S2/S5** guard the `program` bar. It is `max(qwen3_0.6b, argmax)` precisely
  so a ColBERT win cannot come from the comparator being weak at whichever
  chunker the pilot happened to run on; a `max` that could sit below the named
  embedder would silently lower the bar.
* **S3/S4** guard the denominator. A missing combo does not crash -- it just
  averages over fewer queries, which moves a bar without saying so.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "eval"))
from colbert_pilot_baselines import (  # noqa: E402
    _PREDICTION_EMBEDDER,
    _PUBLISHED,
    across_chunkers,
    self_checks,
)

# Read the anchors from the module rather than retyping them. `_PUBLISHED` is a
# CODE-PATH anchor that has to be re-pointed after an index rebuild (it moved
# 0.6066 -> 0.6034 at rebuild #4), and a fixture carrying its own copy would
# turn every such re-point into two unrelated test failures -- the fixture
# drifting from the constant it exists to exercise.
_BM25_PERSON = _PUBLISHED[("bm25", "person")]
_DENSE_PROGRAM = _PUBLISHED[(_PREDICTION_EMBEDDER, "program")]

CHUNKERS = ["fixed_size", "recursive", "semantic", "sentence"]
EMBEDDERS = [_PREDICTION_EMBEDDER, "bge_m3"]


def _cells(values: dict[tuple[str, str], dict[str, float]], nq: int = 30):
    """values[(chunker, embedder)][etype] -> the flat recall every query gets."""
    out = {}
    for key, by_type in values.items():
        out[key] = {t: {f"q{i}": v for i in range(nq)} for t, v in by_type.items()}
    return out


def _good():
    """A run shaped like the real one: the named embedder is the argmax."""
    bm25 = _cells({(c, "-"): {"person": _BM25_PERSON} for c in CHUNKERS})
    dense = _cells({
        (c, e): {"program": _DENSE_PROGRAM if e == _PREDICTION_EMBEDDER else 0.5}
        for c in CHUNKERS
        for e in EMBEDDERS
    })
    return bm25, dense


def _verdicts(bm25, dense, chunkers=None, embedders=None):
    checks = self_checks(bm25, dense, chunkers or CHUNKERS, embedders or EMBEDDERS)
    return {name.split()[0]: ok for name, ok, _ in checks}


def test_a_clean_run_passes_every_check():
    bm25, dense = _good()
    assert all(_verdicts(bm25, dense).values())


def test_s1_fails_when_bm25_no_longer_reproduces_the_published_aggregate():
    """If the anchor drifts, this script is not measuring what the prediction
    was registered against, whatever else it reports."""
    bm25, dense = _good()
    bm25 = _cells({(c, "-"): {"person": 0.79} for c in CHUNKERS})
    v = _verdicts(bm25, dense)
    assert v["S1"] is False
    assert v["S2"] is True, "only the BM25 anchor moved"


def test_s2_fails_when_the_dense_anchor_drifts():
    bm25, dense = _good()
    dense = _cells({
        (c, e): {"program": 0.55 if e == _PREDICTION_EMBEDDER else 0.5}
        for c in CHUNKERS
        for e in EMBEDDERS
    })
    assert _verdicts(bm25, dense)["S2"] is False


def test_s3_catches_a_chunker_scored_on_fewer_queries():
    """A short combo lowers a bar by averaging over less, and never crashes."""
    bm25, dense = _good()
    bm25 = deepcopy(bm25)
    bm25[("semantic", "-")]["person"] = {"q0": _BM25_PERSON}
    assert _verdicts(bm25, dense)["S3"] is False


def test_s4_catches_a_missing_embedder_at_one_chunker():
    bm25, dense = _good()
    dense = deepcopy(dense)
    del dense[("semantic", "bge_m3")]
    assert _verdicts(bm25, dense)["S4"] is False


def test_s5_fails_if_the_bar_could_sit_below_the_named_embedder():
    """The `max` is what makes the bar conservative; a NaN slipping into the
    comparison is enough to defeat it, so the check is stated over the real
    numbers rather than assumed from the expression."""
    bm25, dense = _good()
    dense = deepcopy(dense)
    dense[("recursive", "bge_m3")]["program"] = {"q0": float("nan")}
    dense[("recursive", _PREDICTION_EMBEDDER)]["program"] = {"q0": 0.9}
    assert _verdicts(bm25, dense)["S5"] is False


def test_across_chunkers_averages_per_query_before_averaging_chunkers():
    """The published convention: mean over queries of the per-query mean across
    chunkers -- not the mean of four per-chunker means, which differs whenever
    a chunker is missing a query."""
    dense = {
        ("recursive", "e"): {"program": {"q0": 1.0, "q1": 0.0}},
        ("sentence", "e"): {"program": {"q0": 0.0}},
    }
    # q0 -> mean(1.0, 0.0) = 0.5 ; q1 -> mean(0.0) = 0.0 ; overall 0.25
    assert abs(across_chunkers(dense, "e", "program") - 0.25) < 1e-12
