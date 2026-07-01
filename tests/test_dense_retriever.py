"""Cycle 7 — DenseRetriever ranks by cosine, honours k, and is NaN-safe."""
from __future__ import annotations

import numpy as np

from rag_lab.retrievers import DenseRetriever
from rag_lab.schema import Chunk, Index


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}", resolution_id=f"r{i}", text=f"t{i}", chunk_index=0, page=1
    )


def _index(embeddings: np.ndarray) -> Index:
    return Index(
        chunks=[_chunk(i) for i in range(len(embeddings))],
        embeddings=embeddings,
        meta={},
    )


def test_ranks_by_cosine_and_limits_to_k():
    index = _index(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))
    ranked = DenseRetriever().retrieve(np.array([1.0, 0.0]), index, k=2)

    assert [r.chunk_id for r in ranked] == ["c0", "c1"]
    assert [r.rank for r in ranked] == [1, 2]
    assert ranked[0].score == 1.0  # identical direction → cosine 1
    assert ranked[0].score >= ranked[1].score


def test_k_larger_than_corpus_returns_all():
    index = _index(np.array([[1.0, 0.0], [0.0, 1.0]]))
    ranked = DenseRetriever().retrieve(np.array([1.0, 1.0]), index, k=10)
    assert len(ranked) == 2


def test_zero_embedding_scores_zero_not_nan():
    index = _index(np.array([[0.0, 0.0], [1.0, 0.0]]))
    ranked = DenseRetriever().retrieve(np.array([1.0, 0.0]), index, k=2)
    by_id = {r.chunk_id: r.score for r in ranked}
    assert by_id["c0"] == 0.0
