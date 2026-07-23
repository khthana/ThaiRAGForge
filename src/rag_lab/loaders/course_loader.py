"""Tags each Resolution with which course code(s) (รายวิชา) it mentions,
stored in metadata['courses'] -- same convention as ProgramLoader's
metadata['programs'].

Course codes in this corpus are exactly 8 digits (Arabic or Thai numerals),
e.g. "01276764 AUGMENTED REALITY 3 (3-0-6)" or, inline,
"...ลงทะเบียนเพิ่มรายวิชา ๐๒๖๘๖๐๐๑ COMPUTER-AIDED...". Unlike programs, a
course code is a self-describing key -- no fuzzy matching or pre-built
dictionary is needed; the matched digit string IS the canonical value.

The one real ambiguity is student ID numbers, which are *also* 8 digits
(e.g. "รหัสนักศึกษา ๖๘๐๕๖๑๓๐"). A prefix-range heuristic (student IDs start
with a Buddhist-era year like 65-69) does NOT reliably discriminate here --
grepping the corpus turned up real course codes prefixed 01, 02, 06, 09, 11,
13, 14, 21, 90, 91, and 96, so course-code prefixes are not confined to a
range clear of 65-69. What IS a clean, deterministic signal, confirmed
against every student-ID occurrence found in the corpus: it is always
immediately preceded by the literal label "รหัสนักศึกษา" with nothing but a
space in between. So this module excludes on that label, not on digit
range. As a secondary plausibility check (same "structural anchor"
philosophy as person_loader/program_loader), a match is also required to be
followed by an uppercase Latin letter after optional whitespace -- every
course-code occurrence found in this corpus is immediately followed by the
course's English-transliterated title (even where Thai prose surrounds it,
e.g. "...รายวิชา ๐๒๖๘๖๐๐๑ COMPUTER-AIDED..."), so this is a precise anchor,
not a loose "any letter follows" check -- the latter would pass almost any
8-digit number followed by ordinary Thai prose and defeat the point of a
plausibility check.

This pattern is empirically derived from a partial corpus sample, not a
documented spec, and there is no pre-built ground-truth dictionary (unlike
programs.json) to check false positives against. Run
tools/corpus_prep/tag_courses.py's coverage report -- which samples matched
codes with context, not just unmatched files -- before trusting this as a
retrieval filter.
"""
from __future__ import annotations

import re
from typing import Any

from pythainlp.util import thai_digit_to_arabic_digit

from rag_lab.loaders.base import BaseLoader
from rag_lab.loaders.common import (
    make_resolution_id,
    parse_path,
    read_text,
    strip_document_header,
    strip_mapping_tables,
)
from rag_lab.registries import loader_registry
from rag_lab.schema import Resolution

_COURSE_CODE = re.compile(r"\d{8}")
_STUDENT_ID_LABEL = re.compile(r"รหัสนักศึกษา")
_LOOKBACK = 20  # chars before a match to check for the student-ID label
_FOLLOWED_BY_NAME = re.compile(r"\s*[A-Z]")  # plausibility check: an English course title follows


def match_courses(text: str) -> list[str]:
    """Every course code (8-digit string) found in `text`, deduped and
    sorted. Digits are normalized to Arabic first (thai_digit_to_arabic_digit
    only converts digit characters, leaving Thai script -- including the
    "รหัสนักศึกษา" label -- untouched), so both Arabic- and Thai-numeral
    course codes are caught by one pattern."""
    normalized = thai_digit_to_arabic_digit(text)
    found: set[str] = set()
    for m in _COURSE_CODE.finditer(normalized):
        before = normalized[max(0, m.start() - _LOOKBACK) : m.start()]
        if _STUDENT_ID_LABEL.search(before):
            continue
        after = normalized[m.end() : m.end() + 5]
        if not _FOLLOWED_BY_NAME.match(after):
            continue
        found.add(m.group(0))
    return sorted(found)


@loader_registry.register("course")
class CourseLoader(BaseLoader):
    """Tags each Resolution with the course code(s) (รายวิชา) it mentions,
    stored in metadata['courses']."""

    def load(self, path: str) -> Resolution:
        text = strip_mapping_tables(strip_document_header(read_text(path)))
        year, session, title = parse_path(path)
        metadata: dict[str, Any] = {
            "year": year,
            "session": session,
            "title": title,
            "courses": match_courses(text),
        }
        return Resolution(
            resolution_id=make_resolution_id(path, year, session, title),
            source_path=str(path),
            raw_text=text,
            year=year,
            session=session,
            title=title,
            metadata=metadata,
        )
