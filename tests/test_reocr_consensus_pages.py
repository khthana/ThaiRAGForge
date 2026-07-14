"""Pure-logic tests for reocr_consensus_pages.py (Phase 1 of the consensus
re-OCR pipeline): resolving a corpus-relative path back to its source PDF,
and grouping consensus-flagged pages by unique (pdf, page) pair so
split-document siblings that flag the same page aren't re-OCR'd twice. No
Ollama/poppler calls -- those are exercised manually, not by this suite (same
convention as ocr_pdf_to_md.py / llm_ocr_scan.py, neither of which is
unit-tested for their I/O-heavy main loop).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep" / "consensus_review"))
import logic  # noqa: E402
import reocr_consensus_pages as reocr  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _write_md(corpus_root: Path, relpath: str, header_line: str) -> None:
    p = corpus_root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header_line + "\n\n## Page 1\n\nเนื้อหา\n", encoding="utf-8")


class TestResolveSrcDir:
    def test_direct_match(self, tmp_path):
        _touch(tmp_path / "2567" / "ครั้งที่ 9" / "placeholder.pdf")
        result = reocr.resolve_src_dir(tmp_path, "2567", "ครั้งที่ 9")
        assert result == tmp_path / "2567" / "ครั้งที่ 9"

    def test_special_session_falls_back_to_underscore_year(self, tmp_path):
        _touch(tmp_path / "2566" / "ครั้งที่ 2_2566" / "placeholder.pdf")
        result = reocr.resolve_src_dir(tmp_path, "2566", "ครั้งที่ 2s")
        assert result == tmp_path / "2566" / "ครั้งที่ 2_2566"

    def test_special_session_falls_back_to_waraphiset_prefix(self, tmp_path):
        _touch(tmp_path / "2564" / "วาระพิเศษ ครั้งที่ 1_2564" / "placeholder.pdf")
        result = reocr.resolve_src_dir(tmp_path, "2564", "ครั้งที่ 1s")
        assert result == tmp_path / "2564" / "วาระพิเศษ ครั้งที่ 1_2564"

    def test_no_match_returns_none(self, tmp_path):
        assert reocr.resolve_src_dir(tmp_path, "2599", "ครั้งที่ 1") is None


class TestResolvePdfPath:
    def test_plain_file_matches_same_stem_pdf(self, tmp_path):
        _touch(tmp_path / "2567" / "ครั้งที่ 9" / "เอกสาร ก.pdf")
        result = reocr.resolve_pdf_path(tmp_path, "2567\\ครั้งที่ 9\\เอกสาร ก.md")
        assert result == tmp_path / "2567" / "ครั้งที่ 9" / "เอกสาร ก.pdf"

    def test_split_piece_collapses_to_shared_pre_split_pdf(self, tmp_path):
        _touch(tmp_path / "2567" / "ครั้งที่ 9" / "เอกสาร ข.pdf")
        result = reocr.resolve_pdf_path(tmp_path, "2567\\ครั้งที่ 9\\เอกสาร ข__2.md")
        assert result == tmp_path / "2567" / "ครั้งที่ 9" / "เอกสาร ข.pdf"

    def test_missing_pdf_returns_none(self, tmp_path):
        (tmp_path / "2567" / "ครั้งที่ 9").mkdir(parents=True)
        result = reocr.resolve_pdf_path(tmp_path, "2567\\ครั้งที่ 9\\เอกสาร ก.md")
        assert result is None

    def test_missing_session_dir_returns_none(self, tmp_path):
        result = reocr.resolve_pdf_path(tmp_path, "2567\\ครั้งที่ 9\\เอกสาร ก.md")
        assert result is None

    def test_strips_corpus_only_dedup_suffix(self, tmp_path):
        # Two same-titled corpus docs get " (2)" appended to disambiguate --
        # that suffix exists only in the corpus, never on the source drive.
        _touch(tmp_path / "2567" / "ครั้งที่ 10" / "เอกสาร ก.pdf")
        result = reocr.resolve_pdf_path(tmp_path, "2567\\ครั้งที่ 10\\เอกสาร ก (2).md")
        assert result == tmp_path / "2567" / "ครั้งที่ 10" / "เอกสาร ก.pdf"

    def test_falls_back_to_header_derived_filename(self, tmp_path, monkeypatch):
        # The source PDF still carries an agenda-item prefix the corpus
        # filename dropped -- only the file's own "# Document:" header
        # records the true original name.
        corpus_root = tmp_path / "corpus"
        src_root = tmp_path / "src"
        monkeypatch.setattr(reocr, "CORPUS_ROOT", corpus_root)
        _write_md(
            corpus_root, "2567\\ครั้งที่ 10\\เอกสาร ก.md",
            "# Document: 10-2567 4.5-2 เอกสาร ก.pdf",
        )
        _touch(src_root / "2567" / "ครั้งที่ 10" / "4.5-2 เอกสาร ก.pdf")
        result = reocr.resolve_pdf_path(src_root, "2567\\ครั้งที่ 10\\เอกสาร ก.md")
        assert result == src_root / "2567" / "ครั้งที่ 10" / "4.5-2 เอกสาร ก.pdf"

    def test_header_filename_also_missing_returns_none(self, tmp_path, monkeypatch):
        corpus_root = tmp_path / "corpus"
        src_root = tmp_path / "src"
        monkeypatch.setattr(reocr, "CORPUS_ROOT", corpus_root)
        _write_md(
            corpus_root, "2567\\ครั้งที่ 10\\เอกสาร ก.md",
            "# Document: 4.14-9 completely different name.pdf",
        )
        (src_root / "2567" / "ครั้งที่ 10").mkdir(parents=True)
        result = reocr.resolve_pdf_path(src_root, "2567\\ครั้งที่ 10\\เอกสาร ก.md")
        assert result is None


class TestParseDocumentHeader:
    def test_plain_header_no_prefix_or_suffix(self, tmp_path):
        _write_md(tmp_path, "a.md", "# Document: เอกสาร ก.pdf")
        assert reocr.parse_document_header(tmp_path / "a.md") == "เอกสาร ก.pdf"

    def test_strips_meeting_year_prefix(self, tmp_path):
        _write_md(tmp_path, "a.md", "# Document: 10-2567 4.5-2 เอกสาร ก.pdf")
        assert reocr.parse_document_header(tmp_path / "a.md") == "4.5-2 เอกสาร ก.pdf"

    def test_strips_split_piece_suffix(self, tmp_path):
        _write_md(tmp_path, "a.md", "# Document: เอกสาร ก.pdf (ส่วนที่ 3/28)")
        assert reocr.parse_document_header(tmp_path / "a.md") == "เอกสาร ก.pdf"

    def test_missing_file_returns_none(self, tmp_path):
        assert reocr.parse_document_header(tmp_path / "missing.md") is None

    def test_non_matching_header_returns_none(self, tmp_path):
        _write_md(tmp_path, "a.md", "# Something else entirely")
        assert reocr.parse_document_header(tmp_path / "a.md") is None


class TestPageLabelToInt:
    def test_plain_page_number(self):
        assert reocr.page_label_to_int("Page 12") == 12

    def test_sub_chunk_label_resolves_to_the_physical_page(self):
        assert reocr.page_label_to_int("Page 7.1") == 7
        assert reocr.page_label_to_int("Page 7.2") == 7

    def test_non_page_label_returns_none(self):
        assert reocr.page_label_to_int("chunk1") is None

    def test_malformed_label_returns_none(self):
        assert reocr.page_label_to_int("Page abc") is None


def _unlimited_pages(pdf_path):
    return 9999


class TestBuildWorkItems:
    def test_dedups_split_siblings_flagging_the_same_page(self, tmp_path):
        _touch(tmp_path / "2567" / "ครั้งที่ 9" / "เอกสาร ข.pdf")
        entries = [
            logic.FileEntry(year="2567", file="2567\\ครั้งที่ 9\\เอกสาร ข__1.md",
                             pages=[logic.PageEntry(page="Page 2")]),
            logic.FileEntry(year="2567", file="2567\\ครั้งที่ 9\\เอกสาร ข__2.md",
                             pages=[logic.PageEntry(page="Page 2")]),
        ]
        items, unresolved = reocr.build_work_items(entries, tmp_path, page_count_fn=_unlimited_pages)
        assert unresolved == []
        assert len(items) == 1
        assert items[0].page == 2
        assert items[0].files == (
            "2567\\ครั้งที่ 9\\เอกสาร ข__1.md",
            "2567\\ครั้งที่ 9\\เอกสาร ข__2.md",
        )

    def test_distinct_pages_produce_distinct_work_items(self, tmp_path):
        _touch(tmp_path / "2567" / "ครั้งที่ 9" / "เอกสาร ก.pdf")
        entries = [
            logic.FileEntry(year="2567", file="2567\\ครั้งที่ 9\\เอกสาร ก.md",
                             pages=[logic.PageEntry(page="Page 1"), logic.PageEntry(page="Page 3")]),
        ]
        items, unresolved = reocr.build_work_items(entries, tmp_path, page_count_fn=_unlimited_pages)
        assert unresolved == []
        assert [item.page for item in items] == [1, 3]

    def test_unresolvable_file_is_reported_not_silently_dropped(self, tmp_path):
        entries = [
            logic.FileEntry(year="2567", file="2567\\ครั้งที่ 9\\เอกสาร ก.md",
                             pages=[logic.PageEntry(page="Page 1")]),
        ]
        items, unresolved = reocr.build_work_items(entries, tmp_path, page_count_fn=_unlimited_pages)
        assert items == []
        assert unresolved == ["2567\\ครั้งที่ 9\\เอกสาร ก.md"]

    def test_flagged_page_beyond_resolved_pdfs_page_count_is_reported_unresolved(self, tmp_path):
        # A fallback match in resolve_pdf_path can land on a different
        # document that happens to share the same title -- if it's too
        # short to contain the flagged page, that's the wrong file.
        _touch(tmp_path / "2567" / "ครั้งที่ 9" / "เอกสาร ก.pdf")
        entries = [
            logic.FileEntry(year="2567", file="2567\\ครั้งที่ 9\\เอกสาร ก.md",
                             pages=[logic.PageEntry(page="Page 23")]),
        ]
        items, unresolved = reocr.build_work_items(entries, tmp_path, page_count_fn=lambda p: 10)
        assert items == []
        assert unresolved == ["2567\\ครั้งที่ 9\\เอกสาร ก.md"]

    def test_page_count_error_is_reported_unresolved_not_raised(self, tmp_path):
        _touch(tmp_path / "2567" / "ครั้งที่ 9" / "เอกสาร ก.pdf")
        entries = [
            logic.FileEntry(year="2567", file="2567\\ครั้งที่ 9\\เอกสาร ก.md",
                             pages=[logic.PageEntry(page="Page 1")]),
        ]

        def _boom(pdf_path):
            raise RuntimeError("not a valid PDF")

        items, unresolved = reocr.build_work_items(entries, tmp_path, page_count_fn=_boom)
        assert items == []
        assert unresolved == ["2567\\ครั้งที่ 9\\เอกสาร ก.md"]
