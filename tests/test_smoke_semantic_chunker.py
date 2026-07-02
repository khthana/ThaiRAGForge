"""Cycle 19 — heavy smoke: SemanticChunker with its real, fixed bge-m3 model
loads and runs on Thai OCR-shaped text. This is a load-and-run sanity check
only — it does NOT claim the breakpoints are semantically "correct" (there's
no cheap falsifiable claim to make about that; chunk quality is eyeball-only
per the PRD). The actual breakpoint-placement guarantee is
test_semantic_chunker.py's fake-embedder unit test.

Skipped by default (loads a ~2 GB model). Enable with:

    RAG_LAB_SMOKE=1 pytest tests/test_smoke_semantic_chunker.py
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_LAB_SMOKE") != "1",
    reason="set RAG_LAB_SMOKE=1 to run the heavy semantic-chunker smoke test",
)


def test_real_semantic_chunker_produces_chunks_with_stable_resolution_id():
    from rag_lab.chunkers.semantic import SemanticChunker
    from rag_lab.schema import Resolution

    resolution = Resolution(
        resolution_id="fee",
        source_path="fee.md",
        raw_text=(
            "## Page 1\n"
            "เรื่อง การลดค่าธรรมเนียมการศึกษาให้นักศึกษาในช่วงการระบาดของโควิด-19 "
            "ที่ประชุมมีมติเห็นชอบให้ดำเนินการตามเสนอ\n\n"
            "## Page 2\n"
            "เรื่อง การปรับปรุงหลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิศวกรรมคอมพิวเตอร์ "
            "ที่ประชุมมีมติอนุมัติหลักสูตรปรับปรุงดังกล่าว"
        ),
    )

    chunks = SemanticChunker().chunk(resolution)

    assert len(chunks) >= 1
    assert all(c.resolution_id == "fee" for c in chunks)
    assert {c.page for c in chunks} <= {1, 2}
