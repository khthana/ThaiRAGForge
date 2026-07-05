"""Cycles 1–3 — MetadataLoader: path fields, source_url, header cleaning."""
from __future__ import annotations

from rag_lab.config import StrategySpec
from rag_lab.factory import build_loader


def _write(tmp_path, body: str, with_link: bool = True):
    d = tmp_path / "2569" / "ครั้งที่ 3"
    d.mkdir(parents=True)
    doc = d / "เรื่อง ค่าธรรมเนียม.md"
    doc.write_text(body, encoding="utf-8")
    if with_link:
        (d / "เรื่อง ค่าธรรมเนียม_LINK.txt").write_text(
            "https://drive.google.com/file/d/ABC/view", encoding="utf-8"
        )
    return doc


def test_metadata_loader_parses_fields_and_source_url(tmp_path):
    doc = _write(tmp_path, "# Document: x.pdf\n\n## Page 1\n\nเนื้อหา หน้าหนึ่ง")

    res = build_loader(StrategySpec(type="metadata")).load(str(doc))

    assert res.year == "2569"
    assert res.session == "3"
    assert res.title == "เรื่อง ค่าธรรมเนียม"
    assert res.source_url == "https://drive.google.com/file/d/ABC/view"
    # mirrored into metadata so it propagates onto chunks and is filterable
    assert res.metadata["year"] == "2569"
    assert res.metadata["session"] == "3"
    assert res.metadata["source_url"].startswith("https://")


def test_metadata_loader_strips_document_header_but_keeps_pages(tmp_path):
    doc = _write(tmp_path, "# Document: foo.pdf\n\n## Page 1\n\nเนื้อหา")
    res = build_loader(StrategySpec(type="metadata")).load(str(doc))

    assert "# Document:" not in res.raw_text
    assert "## Page 1" in res.raw_text  # page markers preserved for the chunker


def test_metadata_loader_without_link_has_no_source_url(tmp_path):
    doc = _write(tmp_path, "## Page 1\nเนื้อหา", with_link=False)
    res = build_loader(StrategySpec(type="metadata")).load(str(doc))
    assert res.source_url is None


def test_metadata_loader_prefers_manifest_over_link_file(tmp_path):
    import json

    doc = _write(tmp_path, "## Page 1\nเนื้อหา")  # _LINK.txt has .../d/ABC/view
    (doc.parent / "meeting_manifest.json").write_text(
        json.dumps([{"file": doc.name, "title": "เรื่อง ค่าธรรมเนียม (ฉบับเต็ม)",
                     "url": "https://drive.google.com/file/d/XYZ/view"}],
                   ensure_ascii=False),
        encoding="utf-8",
    )

    res = build_loader(StrategySpec(type="metadata")).load(str(doc))

    assert res.title == "เรื่อง ค่าธรรมเนียม (ฉบับเต็ม)"
    assert res.source_url == "https://drive.google.com/file/d/XYZ/view"
    assert res.metadata["title"] == "เรื่อง ค่าธรรมเนียม (ฉบับเต็ม)"
