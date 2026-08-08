"""Cycle G — HybridRetriever fuses Dense + BM25 (RRF default, weighted option)."""
from __future__ import annotations

import numpy as np

from rag_lab.config import StrategySpec
from rag_lab.factory import build_retriever
from rag_lab.retrievers import DenseRetriever
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
