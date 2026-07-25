"""strip_mapping_tables — remove Curriculum/SKILL Mapping checkbox grids
(zero retrieval value, often the largest structural block in a
curriculum-revision document) while leaving unrelated tables untouched.

strip_course_comparison_tables — simplify (not remove) old/new
course-comparison tables: keep code + short title per course (credit-tuple
dropped, not searchable content), drop the long English description prose
that makes these tables the corpus's single largest chunks.
"""
from __future__ import annotations

from rag_lab.loaders.common import strip_course_comparison_tables, strip_mapping_tables


def test_removes_table_immediately_following_heading():
    text = (
        "ก่อนหน้า\n\n"
        "Curriculum Mapping\n\n"
        "<table><tr><td>1</td><td>0</td></tr></table>\n\n"
        "หลังจากนั้น"
    )

    out = strip_mapping_tables(text)

    assert "Curriculum Mapping" not in out
    assert "<table>" not in out
    assert "ก่อนหน้า" in out
    assert "หลังจากนั้น" in out


def test_chains_tables_split_across_a_page_break():
    text = (
        "SKILL MAPPING\n\n"
        "<table><tr><td>a</td></tr></table>\n\n"
        "---\n\n## Page 2\n\n"
        "<table><tr><td>b</td></tr></table>\n\n"
        "เนื้อหาจริงหลังตาราง"
    )

    out = strip_mapping_tables(text)

    assert "<table>" not in out
    assert "เนื้อหาจริงหลังตาราง" in out


def test_does_not_swallow_an_unrelated_table_far_after_the_heading():
    # regression: a flat char-count window previously blanked unrelated
    # content (e.g. an OCR repetition-loop) that happened to start within the
    # window of an earlier, unrelated Mapping heading on the same page.
    text = (
        "Curriculum Mapping\n\n"
        "<table><tr><td>1</td></tr></table>\n\n"
        + "เนื้อหาปกติอีกมากมายอยู่ระหว่างกลาง " * 30  # > _MAPPING_CHAIN_GAP (500 chars)
        + "\n\n<table><tr><td>compliance; security; compliance; security;</td></tr></table>"
    )

    out = strip_mapping_tables(text)

    assert "compliance; security;" in out


def test_falls_back_to_flat_window_when_no_table_follows():
    # no <table> anywhere -- must fall back to the flat 8000-char window
    # rather than removing nothing (a malformed/never-closing tag case)
    text = "Curriculum Mapping\n\n" + ("x" * 50) + "\n" + ("y" * 9000)

    out = strip_mapping_tables(text)

    assert "Curriculum Mapping" not in out
    assert "x" * 50 not in out
    assert "y" * 100 in out  # beyond the 8000-char fallback window, untouched


def test_leaves_text_without_any_mapping_heading_unchanged():
    text = "เนื้อหาปกติ ไม่มีตาราง mapping ใดๆ\n\n<table><tr><td>a</td></tr></table>"

    assert strip_mapping_tables(text) == text


_COURSE_TABLE = (
    "<table>"
    "<tr><td>รหัส/หน่วยกิต</td><th colspan=\"2\">เปลี่ยนเป็น</th></tr>"
    "<tr><td>20626105<br/>2 (1-2-3)</td><td colspan=\"2\">Medical Biochemistry</td></tr>"
    "<tr><td rowspan=\"3\"></td><td colspan=\"2\">"
    "Long English description prose that goes on for a while about biochemistry"
    " pathways and enzymes and metabolism and cell signalling and so on."
    "</td></tr>"
    "</table>"
)


def test_compacts_a_course_comparison_table_to_code_title():
    text = "ก่อนหน้า\n\n" + _COURSE_TABLE + "\n\nหลังจากนั้น"

    out = strip_course_comparison_tables(text)

    assert "20626105" in out
    assert "(1-2-3)" not in out  # credit-tuple dropped: not searchable content
    assert "Medical Biochemistry" in out
    assert "Long English description prose" not in out
    assert "ก่อนหน้า" in out
    assert "หลังจากนั้น" in out


def test_compacted_code_and_title_satisfy_match_courses_plausibility_check():
    # the whole point of dropping the credit-tuple: match_courses's
    # "followed by a letter" check never matched CODE<br/>credit-tuple in
    # the raw HTML, but CODE (whitespace) Title does -- for free, no
    # match_courses regex change needed, as long as this runs first
    from rag_lab.loaders.course_loader import match_courses

    out = strip_course_comparison_tables(_COURSE_TABLE)

    assert match_courses(out) == ["20626105"]


def test_dedupes_a_code_repeated_by_rowspan_reconstruction():
    # a real corpus document re-emits the same code+title header row on
    # every rowspan-continuation row (a table-extraction artifact, not OCR)
    repeated = _COURSE_TABLE.replace("</table>", "") + (
        "<tr><td>20626105<br/>2 (1-2-3)</td><td colspan=\"2\">Medical Biochemistry</td></tr>"
        "<tr><td rowspan=\"3\"></td><td colspan=\"2\">A different fragment of the same description.</td></tr>"
        "</table>"
    )

    out = strip_course_comparison_tables(repeated)

    assert out.count("20626105") == 1


def test_leaves_a_table_without_any_marker_untouched():
    text = "<table><tr><td>unrelated</td><td>content</td></tr></table>"

    assert strip_course_comparison_tables(text) == text


def test_leaves_a_marker_matching_table_untouched_when_no_course_code_present():
    # regression: "เปลี่ยนเป็น" ("changed to") is ordinary Thai prose, not a
    # table-specific marker on its own -- a real MoA/joint-degree fee table
    # matched it with no course codes or credit-tuples anywhere in the table
    text = (
        "<table><tr><td>หลักสูตร</td><th colspan=\"2\">เปลี่ยนเป็น</th></tr>"
        "<tr><td>มหาวิทยาลัยกรุงเทพ</td><td colspan=\"2\">"
        "ค่าธรรมเนียมการศึกษาใหม่ ๖๗,๕๕๘ บาท ต่อภาคการศึกษา"
        "</td></tr></table>"
    )

    assert strip_course_comparison_tables(text) == text
