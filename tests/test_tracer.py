"""Cycle 1 — the walking-skeleton tracer bullet.

Proves the whole indexing→retrieval path is wired end-to-end with deterministic
components: load resolutions → chunk → embed → build Index → retrieve. Asserts
real behaviour (the chunk mentioning the query term ranks first), not just shape.
"""
from __future__ import annotations

from rag_lab.schema import Resolution
from rag_lab.chunkers import FixedSizeChunker
from rag_lab.retrievers import DenseRetriever
from rag_lab.pipeline import build_index, retrieve

from tests.fakes import BagOfWordsEmbedder


def test_query_ranks_matching_resolution_first():
    resolutions = [
        Resolution(
            resolution_id="r1",
            source_path="r1.md",
            raw_text="เรื่อง การลด ค่าธรรมเนียม การศึกษา ภาคเรียน",
        ),
        Resolution(
            resolution_id="r2",
            source_path="r2.md",
            raw_text="เรื่อง การปรับปรุง หลักสูตร วิศวกรรม คอมพิวเตอร์",
        ),
    ]
    embedder = BagOfWordsEmbedder()
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)

    index = build_index(resolutions, chunker, embedder)
    result = retrieve("ค่าธรรมเนียม", index, embedder, DenseRetriever(), k=3)

    top = result.results[0]
    assert top.resolution_id == "r1"
    assert top.page is not None
    # scores are sorted descending
    assert [r.score for r in result.results] == sorted(
        (r.score for r in result.results), reverse=True
    )
