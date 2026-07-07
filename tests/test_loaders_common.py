"""strip_mapping_tables — remove Curriculum/SKILL Mapping checkbox grids
(zero retrieval value, often the largest structural block in a
curriculum-revision document) while leaving unrelated tables untouched.
"""
from __future__ import annotations

from rag_lab.loaders.common import strip_mapping_tables


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
