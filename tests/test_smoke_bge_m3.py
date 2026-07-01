"""Cycle 10 — heavy smoke: a real bge-m3 build+retrieve returns the right
resolution for a semantic query. This is the AC's demoable path.

Skipped by default (downloads/loads a ~2 GB model). Enable with:

    RAG_LAB_SMOKE=1 pytest tests/test_smoke_bge_m3.py
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_LAB_SMOKE") != "1",
    reason="set RAG_LAB_SMOKE=1 to run the heavy bge-m3 smoke test",
)


def test_real_bge_m3_ranks_relevant_resolution_first():
    from rag_lab.chunkers import FixedSizeChunker
    from rag_lab.embedders.local_st_embedder import LocalSTEmbedder
    from rag_lab.pipeline import build_index, retrieve
    from rag_lab.retrievers import DenseRetriever
    from rag_lab.schema import Resolution

    resolutions = [
        Resolution(
            resolution_id="fee",
            source_path="fee.md",
            raw_text="เรื่อง การลดค่าธรรมเนียมการศึกษาให้นักศึกษาในช่วงการระบาดของโควิด-19",
        ),
        Resolution(
            resolution_id="curriculum",
            source_path="curriculum.md",
            raw_text="เรื่อง การปรับปรุงหลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิศวกรรมคอมพิวเตอร์",
        ),
    ]
    embedder = LocalSTEmbedder()
    index = build_index(resolutions, FixedSizeChunker(chunk_size=256), embedder)
    result = retrieve("ขอลดค่าเทอมช่วงโควิด", index, embedder, DenseRetriever(), k=2)

    assert result.results[0].resolution_id == "fee"
