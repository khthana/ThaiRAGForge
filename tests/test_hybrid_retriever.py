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
