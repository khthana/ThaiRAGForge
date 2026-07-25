"""CourseLoader tags each Resolution with metadata['courses']: the 8-digit
course codes (รายวิชา) it mentions, distinguished from 8-digit student ID
numbers by the "รหัสนักศึกษา" label anchor, not a digit-range guess (see
course_loader.py's module docstring for why the corpus doesn't support a
digit-range heuristic)."""
from __future__ import annotations

from rag_lab.config import StrategySpec
from rag_lab.factory import build_loader
from rag_lab.loaders.course_loader import match_courses, match_courses_by_name

_FAKE_DICT = [
    {"code": "01416522", "canonical": "BIG DATA"},
    {"code": "13016226", "canonical": "COMPILER CONSTRUCTION"},
]


class TestMatchCourses:
    def test_matches_arabic_digit_code_followed_by_english_name(self):
        text = "01276764 AUGMENTED REALITY 3 (3-0-6)"
        assert match_courses(text) == ["01276764"]

    def test_matches_thai_numeral_code_inline(self):
        text = "ลงทะเบียนเพิ่มรายวิชา ๐๒๖๘๖๐๐๑ COMPUTER-AIDED ARCHITECTURAL DESIGN"
        assert match_courses(text) == ["02686001"]

    def test_excludes_student_id_immediately_preceded_by_the_label(self):
        text = "นายปฏิพัทธ์ สุขสังขาร รหัสนักศึกษา ๖๖๐๒๐๕๗๔"
        assert match_courses(text) == []

    def test_excludes_a_bare_8_digit_number_with_nothing_name_like_after_it(self):
        text = "เอกสารเลขที่ 12345678 สิ้นสุด"
        assert match_courses(text) == []

    def test_course_code_and_student_id_both_present_only_course_code_kept(self):
        text = "๓) Mr. Rayane Mounsi รหัสนักศึกษา ๖๗๐๒๐๔๖๐ ลงทะเบียนเพิ่มรายวิชา ๐๒๖๘๖๐๐๑ COMPUTER-AIDED"
        assert match_courses(text) == ["02686001"]

    def test_finds_multiple_distinct_codes_deduped_and_sorted(self):
        text = "01006301 CO-OPERATIVE EDUCATION or 13016006 CO-OPERATIVE EDUCATION"
        assert match_courses(text) == ["01006301", "13016006"]

    def test_no_match_for_unrelated_text(self):
        text = "สภาสถาบันมีมติเห็นชอบตามที่เสนอ"
        assert match_courses(text) == []


class TestMatchCoursesByName:
    def test_matches_name_with_spaces_around_it(self):
        assert match_courses_by_name("วิชา BIG DATA เปิดเทอมไหน", _FAKE_DICT) == ["01416522"]

    def test_matches_name_with_no_space_at_the_thai_latin_seam(self):
        # \b never fires between a Thai char (\w under Unicode) and a Latin
        # one with no space at the seam -- this is the case that motivated
        # the manual neighbor-character boundary check instead of \b.
        assert match_courses_by_name("วิชาBIG DATAเปิดเทอมไหน", _FAKE_DICT) == ["01416522"]

    def test_does_not_match_a_longer_word_containing_the_name(self):
        assert match_courses_by_name("อยากรู้เรื่อง BIG DATABASE หน่อย", _FAKE_DICT) == []

    def test_case_insensitive(self):
        assert match_courses_by_name("big data คืออะไร", _FAKE_DICT) == ["01416522"]

    def test_no_match_for_unrelated_query(self):
        assert match_courses_by_name("ขอความเห็นชอบหลักสูตรใหม่", _FAKE_DICT) == []

    def test_multiple_names_in_one_query(self):
        got = match_courses_by_name("BIG DATA กับ COMPILER CONSTRUCTION ต่างกันยังไง", _FAKE_DICT)
        assert got == ["01416522", "13016226"]


class TestCourseLoaderIntegration:
    def test_tags_metadata_with_courses(self, tmp_path):
        d = tmp_path / "2569" / "ครั้งที่ 1"
        d.mkdir(parents=True)
        doc = d / "a.md"
        doc.write_text("## Page 1\n01276764 AUGMENTED REALITY 3 (3-0-6)", encoding="utf-8")

        res = build_loader(StrategySpec(type="course")).load(str(doc))

        assert res.metadata["courses"] == ["01276764"]

    def test_no_mention_gives_empty_list(self, tmp_path):
        d = tmp_path / "2569" / "ครั้งที่ 1"
        d.mkdir(parents=True)
        doc = d / "a.md"
        doc.write_text("## Page 1\nสภาสถาบันมีมติเห็นชอบตามที่เสนอ", encoding="utf-8")

        res = build_loader(StrategySpec(type="course")).load(str(doc))

        assert res.metadata["courses"] == []
