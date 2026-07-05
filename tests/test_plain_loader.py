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


def test_special_session_keeps_s_suffix(tmp_path):
    doc = tmp_path / "2566" / "ครั้งที่ 3s" / "เรื่อง วาระพิเศษ.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("## Page 1\nx\n", encoding="utf-8")

    res = PlainLoader().load(str(doc))

    # วาระพิเศษ must not collide with regular session 3 of the same year
    assert res.session == "3s"
    assert res.resolution_id.startswith("2566/3s/")


def test_manifest_title_overrides_truncated_filename(tmp_path):
    import json

    doc = _write(tmp_path, "## Page 1\nx\n")
    full_title = "เรื่อง การลดค่าธรรมเนียม ฉบับเต็มที่ยาวเกินกว่าชื่อไฟล์จะเก็บได้"
    (doc.parent / "meeting_manifest.json").write_text(
        json.dumps([{"file": doc.name, "title": full_title, "url": None}],
                   ensure_ascii=False),
        encoding="utf-8",
    )

    res = PlainLoader().load(str(doc))

    assert res.title == full_title
    assert res.resolution_id == f"2569/3/{full_title}"
