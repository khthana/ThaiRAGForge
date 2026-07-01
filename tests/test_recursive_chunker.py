"""Cycles 1–3 — RecursiveChunker (langchain, Thai separators)."""
from __future__ import annotations

from rag_lab.config import StrategySpec
from rag_lab.factory import build_chunker
from rag_lab.schema import Resolution


def _res(text: str) -> Resolution:
    return Resolution(resolution_id="r1", source_path="r1.md", raw_text=text)


def test_recursive_is_registered_and_produces_chunks():
    chunker = build_chunker(
        StrategySpec(type="recursive", params={"chunk_size": 50, "chunk_overlap": 0})
    )
    chunks = chunker.chunk(_res("ย่อหน้าแรก\n\nย่อหน้าที่สอง\n\nย่อหน้าที่สาม"))

    assert len(chunks) >= 1
    assert all(c.resolution_id == "r1" for c in chunks)
    assert all(c.page == 1 for c in chunks)


def test_recursive_respects_page_boundary():
    text = "## Page 1\nอัลฟ่า เบต้า\n\n## Page 2\nแกมม่า เดลต้า"
    chunks = build_chunker(StrategySpec(type="recursive", params={"chunk_size": 100})).chunk(_res(text))

    assert {c.page for c in chunks} == {1, 2}
    assert all("แกมม่า" not in c.text for c in chunks if c.page == 1)
    assert all("อัลฟ่า" not in c.text for c in chunks if c.page == 2)


def test_recursive_splits_at_paragraph_separators_within_size():
    text = "\n\n".join(["ก" * 40, "ข" * 40, "ค" * 40])
    chunks = build_chunker(
        StrategySpec(type="recursive", params={"chunk_size": 45, "chunk_overlap": 0})
    ).chunk(_res(text))

    assert all(len(c.text) <= 45 for c in chunks)
    # each 40-char paragraph fits under 45 → splits at the \n\n separators
    assert [set(c.text) for c in chunks] == [{"ก"}, {"ข"}, {"ค"}]
