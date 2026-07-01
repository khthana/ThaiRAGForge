"""Cycle A — Index.select slices chunks, embeddings and lexical together."""
from __future__ import annotations

import numpy as np

from rag_lab.schema import Chunk, Index


def _chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"c{i}", resolution_id="r", text=f"t{i}", chunk_index=i, page=1)


def test_select_keeps_all_arrays_aligned():
    chunks = [_chunk(i) for i in range(3)]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    lexical = [["a"], ["b"], ["c"]]
    index = Index(chunks=chunks, embeddings=embeddings, meta={}, lexical=lexical)

    sub = index.select([2, 0])

    assert [c.chunk_id for c in sub.chunks] == ["c2", "c0"]
    assert np.array_equal(sub.embeddings, np.array([[1.0, 1.0], [1.0, 0.0]]))
    assert sub.lexical == [["c"], ["a"]]

    # honest alignment: each surviving chunk's embedding row still equals the row
    # it had in the original index
    original_row = {c.chunk_id: embeddings[i] for i, c in enumerate(chunks)}
    for i, c in enumerate(sub.chunks):
        assert np.array_equal(sub.embeddings[i], original_row[c.chunk_id])


def test_select_tolerates_missing_lexical():
    index = Index(chunks=[_chunk(0), _chunk(1)], embeddings=np.zeros((2, 2)), meta={})
    sub = index.select([1])
    assert [c.chunk_id for c in sub.chunks] == ["c1"]
    assert sub.lexical is None
