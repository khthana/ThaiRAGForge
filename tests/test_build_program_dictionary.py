"""Pure-logic tests for build_program_dictionary.py: title parsing and
dictionary assembly. No corpus I/O -- collect_program_counts's manifest-walk
is exercised manually against the real corpus, same convention as the rest
of tools/corpus_prep/ (see test_reocr_consensus_pages.py).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep"))
import build_program_dictionary as prog  # noqa: E402


class TestExtractProgramFromTitle:
    def test_standard_degree_and_field(self):
        tail = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมเคมี (หลักสูตรนานาชาติ)"
        assert prog.extract_program_from_title(tail) == ("วิศวกรรมศาสตรบัณฑิต", "วิศวกรรมเคมี")

    def test_field_runs_to_end_of_string_with_no_trailing_paren(self):
        tail = "หลักสูตรวิทยาศาสตรมหาบัณฑิต สาขาวิชาเทคโนโลยีพอลิเมอร์"
        assert prog.extract_program_from_title(tail) == (
            "วิทยาศาสตรมหาบัณฑิต",
            "เทคโนโลยีพอลิเมอร์",
        )

    def test_associate_degree_with_attached_ศาสตร์_suffix(self):
        tail = "หลักสูตรอนุปริญญาวิศวกรรมศาสตร์ สาขาวิชาวิศวกรรมไฟฟ้าและอิเล็กทรอนิกส์ (การปรับปรุง)"
        assert prog.extract_program_from_title(tail) == (
            "อนุปริญญาวิศวกรรมศาสตร์",
            "วิศวกรรมไฟฟ้าและอิเล็กทรอนิกส์",
        )

    def test_bare_associate_degree_with_separate_field(self):
        tail = "หลักสูตรอนุปริญญา สาขาวิชาวิศวกรรมแมคคาทรอนิกส์ (หลักสูตรปรับปรุง พ.ศ. ๒๕๖๗)"
        assert prog.extract_program_from_title(tail) == (
            "อนุปริญญา",
            "วิศวกรรมแมคคาทรอนิกส์",
        )

    def test_direct_degree_with_no_separate_field(self):
        tail = "หลักสูตรบริหารธุรกิจบัณฑิต (หลักสูตรนานาชาติ) (การปรับปรุงแก้ไขหลักสูตร ฉบับปี พ.ศ. ๒๕๖๕)"
        assert prog.extract_program_from_title(tail) == ("บริหารธุรกิจบัณฑิต", None)

    def test_garbled_title_with_space_before_degree_word_does_not_match(self):
        # a space between "หลักสูตร" and the next word means this is not a
        # degree-program mention in this corpus's convention (either OCR
        # corruption or an elective-course title, both out of scope)
        tail = "หลักสูตร เทพธิราช วิศวกรรมสูง (การปรับกลุ่มวิศวกรรมที่ ๑๒ มกราคม ๒๕๖๕"
        assert prog.extract_program_from_title(tail) is None

    def test_bare_บัณฑิต_with_no_preceding_degree_word_does_not_match(self):
        tail = "หลักสูตรบัณฑิต สประยุกต์และปรุงแก้ไขหลัก การสูตร ฉบับปี"
        assert prog.extract_program_from_title(tail) is None

    def test_no_หลักสูตร_at_all_returns_none(self):
        assert prog.extract_program_from_title("เรื่อง รับรองรายงานการประชุม") is None

    def test_trims_trailing_explanatory_clause_with_no_paren_boundary(self):
        tail = (
            "หลักสูตรวิศวกรรมศาสตรมหาบัณฑิต สาขาวิชาวิศวกรรมยานยนต์และระบบขนส่งขั้นสูง "
            "จำนวนการรับ ๑๕ คน"
        )
        assert prog.extract_program_from_title(tail) == (
            "วิศวกรรมศาสตรมหาบัณฑิต",
            "วิศวกรรมยานยนต์และระบบขนส่งขั้นสูง",
        )

    def test_strips_a_duplicated_หลักสูตร_word_in_the_degree_capture(self):
        # a real title-recovery typo seen in the corpus: "หลักสูตรหลักสูตร..."
        tail = "หลักสูตรหลักสูตรบริหารธุรกิจบัณฑิต สาขาวิชาการเป็นผู้ประกอบการระดับโลก (หลักสูตรนานาชาติ)"
        assert prog.extract_program_from_title(tail) == (
            "บริหารธุรกิจบัณฑิต",
            "การเป็นผู้ประกอบการระดับโลก",
        )


class TestCanonicalName:
    def test_with_field(self):
        assert (
            prog.canonical_name("วิศวกรรมศาสตรบัณฑิต", "วิศวกรรมเคมี")
            == "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมเคมี"
        )

    def test_without_field(self):
        assert prog.canonical_name("บริหารธุรกิจบัณฑิต", None) == "หลักสูตรบริหารธุรกิจบัณฑิต"


class TestBuildDictionary:
    def test_sorts_by_count_descending_then_name(self):
        counts = Counter(
            {
                ("วิศวกรรมศาสตรบัณฑิต", "วิศวกรรมเคมี"): 3,
                ("วิทยาศาสตรบัณฑิต", "ฟิสิกส์"): 5,
            }
        )
        entries = prog.build_dictionary(counts)
        assert [e["canonical"] for e in entries] == [
            "หลักสูตรวิทยาศาสตรบัณฑิต สาขาวิชาฟิสิกส์",
            "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมเคมี",
        ]
        assert entries[0]["count"] == 5
        assert entries[0]["prefix_type"] == "หลักสูตร"
        assert entries[0]["degree"] == "วิทยาศาสตรบัณฑิต"
        assert entries[0]["field"] == "ฟิสิกส์"

    def test_merges_whitespace_only_variants_and_sums_counts(self):
        counts = Counter(
            {
                ("ปรัชญาดุษฎีบัณฑิต", "นวัตกรรมการบริหารการศึกษาเพื่อความก้าวหน้าของมนุษยชาติ"): 3,
                ("ปรัชญาดุษฎีบัณฑิต", "นวัตกรรมการบริหารการศึกษาเพื่อความก้าวหน้าของ มนุษยชาติ"): 1,
                ("ปรัชญาดุษฎีบัณฑิต", "นวัตกรรมการบริหารการศึกษาเพื่อความก้าวหน้า ของมนุษยชาติ"): 1,
            }
        )
        entries = prog.build_dictionary(counts)
        assert len(entries) == 1
        assert entries[0]["count"] == 5
        # the most-frequent spelling wins as the canonical one
        assert entries[0]["field"] == "นวัตกรรมการบริหารการศึกษาเพื่อความก้าวหน้าของมนุษยชาติ"
