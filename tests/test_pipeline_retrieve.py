"""pipeline.retrieve()'s reranker orchestration: when a reranker is given, the
underlying retriever must be queried with the *widened* pool size (not the
final k), and the returned RetrievalResult must report the final k, not the
pool size."""
from __future__ import annotations

import numpy as np

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.pipeline import build_index, retrieve
from rag_lab.retrievers import DenseRetriever
from rag_lab.schema import Resolution

from tests.fakes import BagOfWordsEmbedder, FakeReranker


def _index(n: int):
    resolutions = [
        Resolution(resolution_id=f"r{i}", source_path=f"r{i}.md", raw_text=f"เอกสาร {i} เรื่อง ทดสอบ")
        for i in range(n)
    ]
    embedder = BagOfWordsEmbedder()
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
    return build_index(resolutions, chunker, embedder), embedder


def test_reranker_widens_the_retriever_pool_size():
    index, embedder = _index(20)
    reranker = FakeReranker()

    retrieve(
        "ทดสอบ", index, embedder, DenseRetriever(), k=5,
        reranker=reranker, rerank_pool_size=15,
    )

    [(_, n_candidates, k)] = reranker.calls
    assert n_candidates == 15  # retriever was asked for the widened pool
    assert k == 5  # reranker was asked to truncate to the final k


def test_result_has_exactly_k_items_after_reranking():
    index, embedder = _index(20)
    reranker = FakeReranker()

    result = retrieve(
        "ทดสอบ", index, embedder, DenseRetriever(), k=5,
        reranker=reranker, rerank_pool_size=15,
    )

    assert len(result.results) == 5


def test_top_k_reports_final_k_not_pool_size():
    index, embedder = _index(20)
    reranker = FakeReranker()

    result = retrieve(
        "ทดสอบ", index, embedder, DenseRetriever(), k=5,
        reranker=reranker, rerank_pool_size=15,
    )

    assert result.top_k == 5


def test_reranker_name_is_recorded_on_the_result():
    index, embedder = _index(5)
    reranker = FakeReranker()

    result = retrieve("ทดสอบ", index, embedder, DenseRetriever(), k=3, reranker=reranker)

    assert result.reranker == "fake_reranker"


def test_no_reranker_leaves_reranker_field_none():
    index, embedder = _index(5)

    result = retrieve("ทดสอบ", index, embedder, DenseRetriever(), k=3)

    assert result.reranker is None


def test_no_rerank_pool_size_falls_back_to_k_for_the_retriever():
    index, embedder = _index(5)
    reranker = FakeReranker()

    retrieve("ทดสอบ", index, embedder, DenseRetriever(), k=3, reranker=reranker)

    [(_, n_candidates, k)] = reranker.calls
    assert n_candidates == 3
    assert k == 3
