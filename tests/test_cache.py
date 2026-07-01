"""Cycle 6 — build_index caches on (docset, chunker params, embedder).

Uses the embedder spy: a cache hit must not re-invoke embed(); a changed
parameter must miss the cache and re-embed.
"""
from __future__ import annotations

import numpy as np

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.pipeline import build_index
from rag_lab.schema import Resolution

from tests.fakes import BagOfWordsEmbedder


def _resolutions():
    return [
        Resolution(resolution_id="r1", source_path="r1.md", raw_text="ฟู บาร์ บาซ ก ข ค"),
        Resolution(resolution_id="r2", source_path="r2.md", raw_text="อีก หนึ่ง เอกสาร"),
    ]


def test_rebuild_same_config_hits_cache_without_reembedding(tmp_path):
    embedder = BagOfWordsEmbedder()
    chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=0)

    first = build_index(_resolutions(), chunker, embedder, cache_dir=tmp_path)
    assert embedder.call_count == 1

    second = build_index(_resolutions(), chunker, embedder, cache_dir=tmp_path)
    assert embedder.call_count == 1  # cache hit — not re-invoked

    assert [c.model_dump() for c in second.chunks] == [c.model_dump() for c in first.chunks]
    assert np.array_equal(second.embeddings, first.embeddings)


def test_changing_chunk_size_misses_cache_and_reembeds(tmp_path):
    embedder = BagOfWordsEmbedder()

    build_index(_resolutions(), FixedSizeChunker(chunk_size=20), embedder, cache_dir=tmp_path)
    assert embedder.call_count == 1

    build_index(_resolutions(), FixedSizeChunker(chunk_size=30), embedder, cache_dir=tmp_path)
    assert embedder.call_count == 2  # different params → cache miss
