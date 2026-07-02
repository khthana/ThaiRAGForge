"""Cycle 18 — SemanticChunker: adjacent-sentence embedding similarity decides
breakpoints, using a FIXED embedding model independent of the Embedder axis
under test. Letting breakpoints shift with whichever Embedder a combo happens
to use would confound the two axes (ADR-0001: cross-embedder scores aren't
comparable either — same root reason).

No test here loads the real bge-m3 model: a fake embedder is injected. Two
different fakes are used depending on what's being tested — see the class
docstrings below for why.
"""
from __future__ import annotations

import numpy as np

from rag_lab.chunkers.semantic import SemanticChunker
from rag_lab.config import StrategySpec
from rag_lab.factory import build_chunker
from rag_lab.schema import Resolution

from tests.fakes import BagOfWordsEmbedder

_BODY = (
    "ที่ประชุมมีมติอนุมัติหลักสูตรวิศวกรรม "
    "วันนี้อากาศดีมากจริงๆ "
    "นักศึกษาลงทะเบียนเรียบร้อยแล้วครับ"
)


def _res(text: str, resolution_id: str = "r1") -> Resolution:
    return Resolution(resolution_id=resolution_id, source_path=f"{resolution_id}.md", raw_text=text)


class _PresetVectorEmbedder:
    """Returns exact, hand-picked vectors in call order. Only safe for a
    SINGLE _split() call (one page => one embed() batch of a known size) —
    a second page with a different sentence count would break the
    len(texts) == len(vectors) assumption. Use BagOfWordsEmbedder (content-
    derived, any batch size) for multi-page tests instead."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    @property
    def model_id(self) -> str:
        return "fake-preset"

    def embed(self, texts: list[str]) -> np.ndarray:
        assert len(texts) == len(self.vectors)
        return np.array(self.vectors, dtype=np.float32)


def test_splits_where_adjacent_sentence_similarity_drops_below_threshold():
    # 3 real crfcut sentences; vectors chosen so 1~2 are similar (cos ~0.99)
    # and 2~3 are not (cos ~0.11), straddling threshold=0.5.
    embedder = _PresetVectorEmbedder([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    chunker = SemanticChunker(breakpoint_threshold=0.5, embedder=embedder)

    chunks = chunker.chunk(_res(_BODY))

    assert len(chunks) == 2
    assert chunks[0].text == "ที่ประชุมมีมติอนุมัติหลักสูตรวิศวกรรม วันนี้อากาศดีมากจริงๆ "
    assert chunks[1].text == "นักศึกษาลงทะเบียนเรียบร้อยแล้วครับ"


def test_page_boundary_is_a_hard_break_even_for_identical_content():
    # threshold=0.0 merges everything *within* a page (bag-of-words cosine of
    # non-empty text is never negative), but a page boundary must still force
    # a split — even between two pages with the exact same sentence content.
    text = "## Page 1\nประโยคหนึ่ง ประโยคสอง\n\n## Page 2\nประโยคหนึ่ง ประโยคสอง"
    chunker = SemanticChunker(breakpoint_threshold=0.0, embedder=BagOfWordsEmbedder())

    chunks = chunker.chunk(_res(text))

    assert len(chunks) == 2
    assert {c.page for c in chunks} == {1, 2}


def test_resolution_id_is_stable_across_all_produced_chunks():
    chunker = SemanticChunker(breakpoint_threshold=0.5, embedder=BagOfWordsEmbedder())

    chunks = chunker.chunk(_res(_BODY, resolution_id="2569/1/example"))

    assert chunks  # sanity: something was produced
    assert all(c.resolution_id == "2569/1/example" for c in chunks)


def test_params_report_the_fixed_model_not_the_injected_embedders_id():
    """The discriminating test for fixedness: params() must not drift to
    reporting self._embedder.model_id, or an injected test double would
    silently change this chunker's cache identity."""
    fake = BagOfWordsEmbedder()
    assert fake.model_id != "BAAI/bge-m3"  # sanity: the fake really differs

    chunker = SemanticChunker(breakpoint_threshold=0.42, engine="crfcut", embedder=fake)

    assert chunker.params() == {
        "type": "semantic",
        "breakpoint_threshold": 0.42,
        "engine": "crfcut",
        "embedding_model": "BAAI/bge-m3",
    }


def test_selectable_via_config_with_fixed_model_and_no_load_at_construction():
    """Construction-only: never call .chunk() here. build_chunker's default
    embedder is a real, lazily-loaded LocalSTEmbedder(bge-m3) — calling
    .chunk() would download/load a ~2 GB model in the default test suite."""
    chunker = build_chunker(
        StrategySpec(type="semantic", params={"breakpoint_threshold": 0.6})
    )

    assert isinstance(chunker, SemanticChunker)
    assert chunker.params() == {
        "type": "semantic",
        "breakpoint_threshold": 0.6,
        "engine": "crfcut",
        "embedding_model": "BAAI/bge-m3",
    }
