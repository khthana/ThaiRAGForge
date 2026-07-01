"""Cycles 4–5 — SentenceChunker (pythainlp crfcut, group whole sentences)."""
from __future__ import annotations

from pythainlp.tokenize import sent_tokenize

from rag_lab.config import StrategySpec
from rag_lab.factory import build_chunker
from rag_lab.schema import Resolution

_BODY = (
    "ที่ประชุมมีมติอนุมัติหลักสูตรวิศวกรรม "
    "วันนี้อากาศดีมากจริงๆ "
    "นักศึกษาลงทะเบียนเรียบร้อยแล้วครับ"
)


def _res(text: str) -> Resolution:
    return Resolution(resolution_id="r1", source_path="r1.md", raw_text=text)


def test_groups_whole_sentences_without_splitting():
    sentences = sent_tokenize(_BODY, engine="crfcut")
    n = len(sentences)
    assert n >= 2  # sanity: the tokenizer found multiple sentences

    # tiny target → one whole sentence per chunk (never split mid-sentence)
    small = build_chunker(StrategySpec(type="sentence", params={"chunk_size": 1})).chunk(_res(_BODY))
    assert len(small) == n
    for sentence, chunk in zip(sentences, small):
        assert chunk.text.strip() == sentence.strip()

    # huge target → all sentences grouped into a single chunk
    big = build_chunker(
        StrategySpec(type="sentence", params={"chunk_size": 100_000})
    ).chunk(_res(_BODY))
    assert len(big) == 1


def test_sentence_respects_page_boundary():
    text = "## Page 1\nประโยคแรกของหน้าหนึ่ง\n\n## Page 2\nประโยคของหน้าสอง"
    chunks = build_chunker(
        StrategySpec(type="sentence", params={"chunk_size": 1000})
    ).chunk(_res(text))
    assert {c.page for c in chunks} == {1, 2}
