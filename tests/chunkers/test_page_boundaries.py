"""Cycle 3 — `## Page` markers are hard chunk boundaries and set the page number.

The OCR corpus marks page breaks with `## Page N` lines. A chunk must never span
two pages, and each chunk records which page it came from.
"""
from __future__ import annotations

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.schema import Resolution


def _res(text: str) -> Resolution:
    return Resolution(resolution_id="r1", source_path="r1.md", raw_text=text)


def test_does_not_chunk_across_page_boundary():
    text = "## Page 1\nAAAA BBBB\n\n## Page 2\nCCCC DDDD"
    chunks = FixedSizeChunker(chunk_size=100, chunk_overlap=0).chunk(_res(text))

    assert len(chunks) == 2
    assert chunks[0].page == 1
    assert "AAAA" in chunks[0].text and "CCCC" not in chunks[0].text
    assert chunks[1].page == 2
    assert "CCCC" in chunks[1].text and "AAAA" not in chunks[1].text


def test_page_number_recorded_on_every_chunk_of_a_long_page():
    text = "## Page 1\nshort\n\n## Page 2\n" + ("X" * 45)
    chunks = FixedSizeChunker(chunk_size=20, chunk_overlap=0).chunk(_res(text))

    page2 = [c for c in chunks if c.page == 2]
    assert len(page2) >= 2  # the long page split into several chunks
    assert all(set(c.text) == {"X"} for c in page2)  # none bled into page 1
