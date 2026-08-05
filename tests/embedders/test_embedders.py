"""Cycle 11 — embed_query seam: BaseEmbedder.embed stays passage-only (abstract);
embed_query defaults to a single-item embed() call so existing embedders (hashing,
bge-m3) need no changes, while embedders with asymmetric passage/query encodings
(e5) can override it.
"""
from __future__ import annotations

import numpy as np

from rag_lab.embedders.base import BaseEmbedder


class _SpyEmbedder(BaseEmbedder):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "spy"

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.array([[float(len(t))] for t in texts], dtype=np.float32)


def test_embed_query_default_delegates_to_single_item_embed_batch():
    embedder = _SpyEmbedder()

    vector = embedder.embed_query("hello")

    assert embedder.calls == [["hello"]]
    assert vector.tolist() == [5.0]


class _DualSpyEmbedder(BaseEmbedder):
    """Distinguishes embed() (passages, at build time) from embed_query()
    (queries, at retrieve time) so a test can prove which one pipeline.retrieve
    calls for the query text."""

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    @property
    def model_id(self) -> str:
        return "dual-spy"

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embed_calls.append(list(texts))
        return np.ones((len(texts), 1), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        self.embed_query_calls.append(text)
        return np.ones(1, dtype=np.float32)


def test_retrieve_embeds_the_query_via_embed_query_not_embed():
    from rag_lab.chunkers import FixedSizeChunker
    from rag_lab.pipeline import build_index, retrieve
    from rag_lab.retrievers import DenseRetriever
    from rag_lab.schema import Resolution

    resolutions = [Resolution(resolution_id="r1", source_path="r1.md", raw_text="ทดสอบ")]
    embedder = _DualSpyEmbedder()
    index = build_index(resolutions, FixedSizeChunker(chunk_size=100), embedder)
    embedder.embed_calls.clear()  # only the query-time call matters below

    retrieve("คำถาม", index, embedder, DenseRetriever(), k=1)

    assert embedder.embed_query_calls == ["คำถาม"]
    assert embedder.embed_calls == []  # the query must not go through embed() directly
