"""Pins the discriminator the `match_programs` absorption audit rests on.

The audit's whole §1 split turns on `inserted_chars` telling "OCR misread a
character" apart from "this is a different program". If that function drifts,
the report keeps producing plausible counts of the wrong thing -- the failure
mode this project keeps meeting. See docs/program-matcher-absorption.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))
sys.path.insert(0, str(REPO / "src"))

from audit_program_matcher_absorption import (  # noqa: E402
    inserted_chars,
    match_programs_detailed,
    strip_window_slack,
)
from rag_lab.loaders.program_loader import _WINDOW_SLACK, match_programs  # noqa: E402


def test_identical_strings_insert_nothing():
    assert inserted_chars("หลักสูตรแพทยศาสตรบัณฑิต", "หลักสูตรแพทยศาสตรบัณฑิต") == 0


def test_the_motivating_absorption_is_counted_as_an_insertion():
    """`ทันต` (4 chars) turns medicine into dentistry -- a different program.

    This is the pair that produced the finding, and it must land well clear of
    the 1-char OCR band or the audit's two buckets are not separable.
    """
    assert inserted_chars("หลักสูตรแพทยศาสตรบัณฑิต",
                          "หลักสูตรทันตแพทยศาสตรบัณฑิต") == 4


def test_the_windows_own_tail_is_not_an_insertion():
    """The defect the first run of this audit published, pinned in both halves.

    `_bounded_span_for` reads `len(canonical) + _WINDOW_SLACK` characters, so a
    perfect match still arrives carrying up to 4 characters of whatever followed
    the name. Counting those put 6,807 matches in a "4 inserted chars" bucket --
    a mode sitting exactly on the slack constant -- and produced a headline of
    "99.3% of matches absorb a different name" that was measuring the window.
    """
    canonical = "หลักสูตรแพทยศาสตรบัณฑิต"
    assert inserted_chars(canonical, canonical + " (กา") == 0
    assert strip_window_slack(canonical, canonical + " (กา") == canonical

    # ... and the trim is bounded by the slack, so a genuinely longer name is
    # still counted rather than trimmed away wholesale.
    long_tail = canonical + "x" * (_WINDOW_SLACK + 3)
    assert inserted_chars(canonical, long_tail) == 3


def test_a_single_ocr_misread_is_one_character():
    """The other side of the cut, pinned so the floor stays meaningful."""
    canonical = "หลักสูตรแพทยศาสตรบัณฑิต"
    misread = canonical[:5] + "x" + canonical[6:]
    assert inserted_chars(canonical, misread) == 1


def test_instrumented_matcher_agrees_with_the_shipped_one():
    """S1 in miniature: the audit transcribes `match_programs`' loop to expose
    the span it accepted, and a measurement of a slightly different function
    would measure nothing. Uses a real canonical so the prefix-group path runs.
    """
    from rag_lab.loaders.program_loader import load_dictionary

    canonical = load_dictionary()[0]["canonical"]
    text = f"ที่ประชุมเห็นชอบ{canonical} ตามที่เสนอ"
    detailed = sorted({h["canonical"] for h in match_programs_detailed(text)})
    assert detailed == match_programs(text)
    assert detailed, "fixture must actually match something, or this is vacuous"
