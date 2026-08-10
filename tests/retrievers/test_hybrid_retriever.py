"""Cycle G — HybridRetriever fuses Dense + BM25 (RRF default, weighted option)."""
from __future__ import annotations

import inspect

import numpy as np

from rag_lab.config import StrategySpec
from rag_lab.factory import build_retriever
from rag_lab.retrievers import DenseRetriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.schema import Chunk, Index, Query


def _index():
    chunks = [
        Chunk(chunk_id="c0", resolution_id="r0", text="t0", chunk_index=0, page=1),
        Chunk(chunk_id="c1", resolution_id="r1", text="t1", chunk_index=1, page=1),
        Chunk(chunk_id="c2", resolution_id="r2", text="t2", chunk_index=2, page=1),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    lexical = [["x"], ["ค่าธรรมเนียม"], ["y"]]
    return Index(chunks=chunks, embeddings=embeddings, meta={}, lexical=lexical)


def _query():
    return Query(text="ค่าธรรมเนียม", vector=np.array([1.0, 0.0]), tokens=["ค่าธรรมเนียม"])


def test_rrf_fuses_dense_and_lexical_signals():
    index = _index()
    q = _query()

    dense_order = [r.chunk_id for r in DenseRetriever().retrieve(q, index, 3)]
    hybrid_order = [
        r.chunk_id for r in build_retriever(StrategySpec(type="hybrid")).retrieve(q, index, 3)
    ]

    # c0 is top under dense (aligned vector) and stays top under hybrid
    assert hybrid_order[0] == "c0"
    # dense alone ranks c2 above c1; the lexical (BM25) hit on "ค่าธรรมเนียม" lifts
    # c1 above c2 under hybrid — proof both signals are fused
    assert dense_order.index("c2") < dense_order.index("c1")
    assert hybrid_order.index("c1") < hybrid_order.index("c2")


def test_weighted_all_dense_matches_dense_order():
    index = _index()
    q = _query()
    dense_order = [r.chunk_id for r in DenseRetriever().retrieve(q, index, 3)]

    weighted = build_retriever(
        StrategySpec(
            type="hybrid",
            params={"method": "weighted", "dense_weight": 1.0, "bm25_weight": 0.0},
        )
    )
    assert [r.chunk_id for r in weighted.retrieve(q, index, 3)] == dense_order


def _hybrid(**params):
    return build_retriever(StrategySpec(type="hybrid", params=params))


def test_weighted_at_a_mid_blend_fuses_both_signals():
    # The degenerate all-dense case above only proves the dense term is wired.
    # A real blend has to show the BM25 term changing the order too: c1 wins
    # the lexical match but is last under dense, so at 0.5/0.5 it must sit
    # above c2 (as under RRF) while the dense-aligned c0 stays on top.
    index, q = _index(), _query()
    order = [r.chunk_id for r in _hybrid(method="weighted").retrieve(q, index, 3)]
    assert order[0] == "c0"
    assert order.index("c1") < order.index("c2")


def test_weighted_weights_actually_move_the_ranking():
    index, q = _index(), _query()
    bm25_heavy = [
        r.chunk_id
        for r in _hybrid(method="weighted", dense_weight=0.1, bm25_weight=0.9).retrieve(
            q, index, 3
        )
    ]
    # tilted far enough toward lexical, the one BM25-matching chunk takes the
    # top slot away from the dense-aligned one -- i.e. the weight is load-
    # bearing, not just present in the signature
    assert bm25_heavy[0] == "c1"


def test_rrf_default_weights_are_rank_order_identical_to_unweighted_rrf():
    # Load-bearing regression guard: dense_weight/bm25_weight were added to the
    # rrf branch so an alpha sweep isolates the weight instead of also switching
    # rank fusion for score fusion. That is only sound if the 0.5/0.5 default
    # reproduces plain RRF exactly -- every published hybrid number depends on
    # it. A uniform 0.5x factor cannot reorder, so this must hold identically.
    index, q = _index(), _query()
    unweighted = [
        r.chunk_id for r in _hybrid(dense_weight=1.0, bm25_weight=1.0).retrieve(q, index, 3)
    ]
    assert [r.chunk_id for r in _hybrid().retrieve(q, index, 3)] == unweighted


def test_rrf_weights_actually_move_the_ranking():
    index, q = _index(), _query()
    bm25_heavy = [
        r.chunk_id
        for r in _hybrid(dense_weight=0.05, bm25_weight=0.95).retrieve(q, index, 3)
    ]
    assert bm25_heavy[0] == "c1"


def _truncation_index():
    """8 chunks with deliberately crossed dense and BM25 orders.

    dense puts c0 > c1 > c2 > ... (cosine against [1, 0]); BM25 puts
    c2 > c1 > c3 > the rest (term frequency of "ก"). c2 is therefore the chunk
    that depends on being reachable in *both* arms, which is what truncation
    breaks.

    Eight chunks rather than four because `BM25Okapi` uses the classic
    `log((N-df+0.5)/(df+0.5))` IDF, which goes **negative** for a term carried by
    most of the collection -- at N=4, df=3 the arm ranks the matching chunks
    *below* the non-matching ones and the fixture tests the opposite of what it
    reads as. At N=8, df=3 the IDF is positive and the order is the intended one.
    """
    chunks = [
        Chunk(chunk_id=f"c{i}", resolution_id=f"r{i}", text=f"t{i}", chunk_index=i, page=1)
        for i in range(8)
    ]
    embeddings = np.array(
        [[1.0, 0.0], [3.0, 1.0], [2.0, 1.0], [1.0, 1.0],
         [1.0, 2.0], [1.0, 3.0], [1.0, 5.0], [0.0, 1.0]]
    )
    lexical = [["z"], ["ก", "ก", "z", "z"], ["ก", "ก", "ก"], ["ก", "z", "z", "z"]]
    lexical += [["z"]] * 4
    return Index(chunks=chunks, embeddings=embeddings, meta={}, lexical=lexical)


def _truncation_query():
    return Query(text="ก", vector=np.array([1.0, 0.0]), tokens=["ก"])


def test_fetch_depth_none_is_identical_to_fetching_the_whole_corpus():
    # The anchor guarding every hybrid number this project has published: the
    # default must be exactly the old k=n path, so adding the knob cannot move a
    # published figure. Also pins that a depth past n is clamped, not an error.
    index, q = _index(), _query()
    full = [r.chunk_id for r in _hybrid().retrieve(q, index, 3)]
    assert [r.chunk_id for r in _hybrid(fetch_depth=3).retrieve(q, index, 3)] == full
    assert [r.chunk_id for r in _hybrid(fetch_depth=999).retrieve(q, index, 3)] == full


def test_the_class_default_stays_none_so_only_callers_opt_into_a_cut():
    # The 2026-08-09 ship decision, pinned. Measured against the shipped hard
    # router, F=200 costs nothing significant (recall@10 +0.0005, MRR -0.0024,
    # nDCG@10 -0.0022, all Holm-adj 1.0000) and saves 0.718s/query -- so the
    # Streamlit UI sets it, per query, via StrategySpec params. What it must
    # NOT become is this default: eval scripts construct HybridRetriever()
    # directly, and a cut here would silently re-rank 17 of 106 Gold queries
    # while every published table still said k=n.
    # See data/results/routed_fetch_depth_test.md.
    assert inspect.signature(HybridRetriever).parameters["fetch_depth"].default is None


def test_truncating_the_fetch_drops_a_chunk_out_of_reach_of_one_arm():
    # The measured claim in tools/eval/hybrid_fetch_depth_sweep.py, in miniature:
    # truncation is not an optimisation, it changes the ranking. c2 leads at full
    # depth on the strength of two arms; cut to 2 it keeps only its BM25 term
    # (dense rank 3 is past the cut) and falls to last.
    index, q = _truncation_index(), _truncation_query()
    assert [r.chunk_id for r in DenseRetriever().retrieve(q, index, 4)] == [
        "c0", "c1", "c2", "c3",
    ]
    assert [r.chunk_id for r in _hybrid().retrieve(q, index, 4)][0] == "c2"

    cut = [r.chunk_id for r in _hybrid(fetch_depth=2).retrieve(q, index, 4)]
    assert cut[0] == "c1"
    assert cut.index("c2") > cut.index("c0")
    # c3 is past both cuts, so it is not merely demoted -- it is unreachable,
    # and a k of 4 cannot be filled from a fetch of 2.
    assert "c3" not in cut and len(cut) == 3


def test_truncated_ties_still_break_dense_first():
    # Under truncation the fusion dict is filled dense-top-F first, then the
    # BM25-only remainder, so equal scores keep dense ahead. The sweep replicates
    # this order in numpy; if the retriever ever stopped doing it, the sweep's
    # agreement figures would silently measure the wrong thing.
    index, q = _truncation_index(), _truncation_query()
    cut = _hybrid(fetch_depth=2).retrieve(q, index, 4)
    scores = {r.chunk_id: r.score for r in cut}
    assert scores["c0"] == scores["c2"]  # dense-rank-1-only vs BM25-rank-1-only
    assert [r.chunk_id for r in cut].index("c0") < [r.chunk_id for r in cut].index("c2")
