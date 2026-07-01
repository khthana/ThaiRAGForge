"""Cycle D — the lexical (BM25) index is built and persisted with the artifact."""
from __future__ import annotations

import numpy as np

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.embedders import HashingEmbedder
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.pipeline import build_index
from rag_lab.schema import Resolution


def _index():
    resolutions = [
        Resolution(
            resolution_id="r1",
            source_path="r1.md",
            raw_text="ค่าธรรมเนียม การศึกษา ภาคเรียน",
        )
    ]
    return build_index(resolutions, FixedSizeChunker(chunk_size=200), HashingEmbedder())


def test_build_index_populates_lexical_tokens():
    index = _index()
    assert index.lexical is not None
    assert len(index.lexical) == len(index.chunks)
    assert any("ค่าธรรมเนียม" in tokens for tokens in index.lexical)


def test_artifact_store_round_trips_lexical(tmp_path):
    index = _index()
    store = ArtifactStore()
    store.save(index, tmp_path)
    loaded = store.load(tmp_path)

    assert loaded.lexical == index.lexical
    assert np.array_equal(loaded.embeddings, index.embeddings)
