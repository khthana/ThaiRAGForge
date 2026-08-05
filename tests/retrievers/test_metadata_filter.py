"""Cycle E — MetadataFilter restricts candidates before any retriever ranks.

Tested in isolation with hand-set chunk.metadata (no loader / no build / no #6).
"""
from __future__ import annotations

import numpy as np

from rag_lab.retrievers.filters import MetadataFilter
from rag_lab.schema import Chunk, Index


def _index():
    years = ["2568", "2569", "2569"]
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            resolution_id=f"r{i}",
            text=f"t{i}",
            chunk_index=0,
            page=1,
            metadata={"year": y},
        )
        for i, y in enumerate(years)
    ]
    embeddings = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    lexical = [["a"], ["b"], ["c"]]
    return Index(chunks=chunks, embeddings=embeddings, meta={}, lexical=lexical)


def test_filter_keeps_only_matching_chunks_and_stays_aligned():
    filtered = MetadataFilter({"year": "2569"}).apply(_index())

    assert [c.chunk_id for c in filtered.chunks] == ["c1", "c2"]
    assert np.array_equal(filtered.embeddings, np.array([[2.0, 0.0], [3.0, 0.0]]))
    assert filtered.lexical == [["b"], ["c"]]


def test_filter_with_no_matches_returns_empty_index():
    filtered = MetadataFilter({"year": "9999"}).apply(_index())
    assert filtered.chunks == []
