"""Pure-logic tests for scan_entity_candidates.py: pattern construction and
candidate extraction/cleanup. No corpus I/O -- `scan_corpus`'s file-walking
loop is exercised manually against the real corpus, same convention as the
rest of tools/corpus_prep/ (see test_reocr_consensus_pages.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep"))
import scan_entity_candidates as scan  # noqa: E402


class TestExtractCandidates:
    def test_finds_a_simple_prefixed_candidate(self):
        text = "คณะวิศวกรรมศาสตร์ มีความประสงค์เสนอ"
        assert scan.extract_candidates(text, ("คณะ", "วิทยาลัย")) == ["คณะวิศวกรรมศาสตร์"]

    def test_stops_at_the_first_space_even_for_a_multi_word_entity(self):
        # "คณะสถาปัตยกรรม ศิลปะและการออกแบบ" is a real multi-word faculty name,
        # but single-token capture can't tell that apart from the next
        # sentence word starting -- this is exactly why multi-word names are
        # hand-confirmed once into data/entity_dictionaries/faculties.json
        # rather than algorithmically recovered on every scan.
        text = "คณะสถาปัตยกรรม ศิลปะและการออกแบบ มีความประสงค์"
        result = scan.extract_candidates(text, ("คณะ",))
        assert result == ["คณะสถาปัตยกรรม"]

    def test_excludes_committee_not_faculty(self):
        text = "คณะกรรมการสภาวิชาการ ครั้งที่ ๙/๒๕๖๗"
        assert scan.extract_candidates(text, ("คณะ",)) == []

    def test_excludes_subcommittee_not_faculty(self):
        text = "คณะอนุกรรมการกลั่นกรองระดับปริญญาตรี พิจารณาแล้ว"
        assert scan.extract_candidates(text, ("คณะ",)) == []

    def test_does_not_match_the_tail_of_มหาวิทยาลัย(self):
        text = "มหาวิทยาลัยมหิดล เสนอความร่วมมือ"
        assert scan.extract_candidates(text, ("วิทยาลัย",)) == []

    def test_strips_pdf_extension_noise(self):
        text = "# Document: 9-2567 คณะวิศวกรรมศาสตร์.pdf\n\nเนื้อหา"
        assert scan.extract_candidates(text, ("คณะ",)) == ["คณะวิศวกรรมศาสตร์"]

    def test_strips_trailing_markdown_backtick(self):
        text = "ดูไฟล์ `คณะวิศวกรรมศาสตร์.md`"
        result = scan.extract_candidates(text, ("คณะ",))
        assert result == ["คณะวิศวกรรมศาสตร์"]

    def test_finds_วิทยาเขต_prefix(self):
        text = "วิทยาเขตชุมพรเขตรอุดมศักดิ์ จำนวน ๓ หลักสูตร"
        assert scan.extract_candidates(text, ("วิทยาเขต",)) == ["วิทยาเขตชุมพรเขตรอุดมศักดิ์"]

    def test_finds_multiple_distinct_candidates_in_order(self):
        text = "คณะวิศวกรรมศาสตร์ และ วิทยาลัยวิศวกรรมสังคีต ร่วมกันเสนอ"
        result = scan.extract_candidates(text, ("คณะ", "วิทยาลัย"))
        assert result == ["คณะวิศวกรรมศาสตร์", "วิทยาลัยวิศวกรรมสังคีต"]

    def test_too_short_candidate_is_dropped(self):
        # a bare prefix with nothing following (e.g. end of line) shouldn't
        # survive as a 2-3 char "candidate"
        text = "คณะ ๑๑/๒๕๖๔"
        assert scan.extract_candidates(text, ("คณะ",)) == []


class TestScanCorpus:
    def test_counts_across_multiple_files_skips_dup(self, tmp_path):
        (tmp_path / "a.md").write_text("คณะวิศวกรรมศาสตร์ เสนอ", encoding="utf-8")
        (tmp_path / "b.md").write_text("คณะวิศวกรรมศาสตร์ อีกครั้ง", encoding="utf-8")
        (tmp_path / "c.md.dup").write_text("คณะวิศวกรรมศาสตร์ ไม่ควรนับ", encoding="utf-8")
        counter = scan.scan_corpus(tmp_path, ("คณะ",))
        assert counter["คณะวิศวกรรมศาสตร์"] == 2

    def test_skips_document_header_line(self, tmp_path):
        (tmp_path / "a.md").write_text(
            "# Document: 9-2567 คณะครุศาสตร์.pdf\n\nคณะวิศวกรรมศาสตร์ เสนอ",
            encoding="utf-8",
        )
        counter = scan.scan_corpus(tmp_path, ("คณะ",))
        assert counter["คณะครุศาสตร์"] == 0
        assert counter["คณะวิศวกรรมศาสตร์"] == 1
