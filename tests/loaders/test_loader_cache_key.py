"""Cycle 5 — the Index cache key distinguishes Resolutions that share raw_text
but differ in metadata (so MetadataLoader vs NERLoader don't collide)."""
from __future__ import annotations

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.pipeline import build_index
from rag_lab.schema import Resolution

from tests.fakes import BagOfWordsEmbedder


def _res(metadata):
    return [
        Resolution(
            resolution_id="r1",
            source_path="r1.md",
            raw_text="เนื้อหา เหมือนกัน ทุกประการ",
            metadata=metadata,
        )
    ]


def test_same_text_different_metadata_uses_different_cache_entry(tmp_path):
    embedder = BagOfWordsEmbedder()
    chunker = FixedSizeChunker(chunk_size=100)

    build_index(_res({}), chunker, embedder, cache_dir=tmp_path)
    assert embedder.call_count == 1

    build_index(
        _res({"entities": [{"text": "x", "tag": "PERSON"}]}),
        chunker,
        embedder,
        cache_dir=tmp_path,
    )
    # different metadata → cache miss → re-embed (no silent collision)
    assert embedder.call_count == 2
