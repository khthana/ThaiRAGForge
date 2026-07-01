"""Cycle 9 — PlainLoader reads a .md resolution and derives basic metadata.

Corpus layout: <year พ.ศ.>/ครั้งที่ N/<เรื่อง>.md. The loader pulls year/session
from the path and title from the filename, and preserves `## Page` markers in
raw_text so the chunker can segment on them.
"""
from __future__ import annotations

from rag_lab.loaders import PlainLoader


def _write(tmp_path, body: str):
    doc = tmp_path / "2569" / "ครั้งที่ 3" / "เรื่อง การลดค่าธรรมเนียม.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(body, encoding="utf-8")
    return doc


def test_parses_path_metadata_and_preserves_page_markers(tmp_path):
    doc = _write(tmp_path, "# Document: x.pdf\n\n## Page 1\n\nเนื้อหา หน้าหนึ่ง\n")

    res = PlainLoader().load(str(doc))

    assert res.year == "2569"
    assert res.session == "3"
    assert res.title == "เรื่อง การลดค่าธรรมเนียม"
    assert res.source_path == str(doc)
    assert "## Page 1" in res.raw_text


def test_resolution_id_is_non_empty_and_stable(tmp_path):
    doc = _write(tmp_path, "## Page 1\nx\n")

    first = PlainLoader().load(str(doc))
    second = PlainLoader().load(str(doc))

    assert first.resolution_id
    assert first.resolution_id == second.resolution_id
