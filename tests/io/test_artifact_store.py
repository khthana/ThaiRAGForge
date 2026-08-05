"""Cycle 5 — an Index artifact survives a save→load round-trip unchanged."""
from __future__ import annotations

import numpy as np

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.pipeline import build_index
from rag_lab.schema import Resolution

from tests.fakes import BagOfWordsEmbedder


def _built_index():
    resolutions = [
        Resolution(
            resolution_id="r1",
            source_path="r1.md",
            raw_text="## Page 1\nฟู บาร์ บาซ\n\n## Page 2\nอีก หน้า หนึ่ง",
        ),
        Resolution(resolution_id="r2", source_path="r2.md", raw_text="สั้น"),
    ]
    return build_index(
        resolutions, FixedSizeChunker(chunk_size=8, chunk_overlap=0), BagOfWordsEmbedder()
    )


def test_save_load_round_trip(tmp_path):
    original = _built_index()

    store = ArtifactStore()
    store.save(original, tmp_path)
    loaded = store.load(tmp_path)

    assert [c.model_dump() for c in loaded.chunks] == [
        c.model_dump() for c in original.chunks
    ]
    assert np.array_equal(loaded.embeddings, original.embeddings)
    assert loaded.meta == original.meta
