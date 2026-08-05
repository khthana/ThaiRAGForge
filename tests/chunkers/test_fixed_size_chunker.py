"""Cycles 2–4 — FixedSizeChunker behaviour."""
from __future__ import annotations

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.schema import Resolution


def _res(text: str, rid: str = "r1") -> Resolution:
    return Resolution(resolution_id=rid, source_path=f"{rid}.md", raw_text=text)


def test_short_text_is_a_single_chunk():
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk(_res("สั้นมาก"))
    assert len(chunks) == 1
    assert chunks[0].text == "สั้นมาก"


def test_long_text_splits_into_overlapping_chunks():
    text = "0123456789" * 5  # 50 chars
    chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(_res(text))

    # step = 15 → starts at 0,15,30,45 → 4 chunks
    assert len(chunks) == 4
    # consecutive chunks overlap by chunk_overlap characters
    assert chunks[0].text[-5:] == chunks[1].text[:5]
    # reassembling non-overlapping prefixes reconstructs the original text
    assert chunks[0].text == text[0:20]
    assert chunks[1].text == text[15:35]


def test_every_chunk_carries_source_resolution_id_and_sequential_index():
    text = "0123456789" * 5
    chunks = FixedSizeChunker(chunk_size=20, chunk_overlap=0).chunk(_res(text, rid="abc"))

    assert all(c.resolution_id == "abc" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id == f"abc:{c.chunk_index}" for c in chunks)
