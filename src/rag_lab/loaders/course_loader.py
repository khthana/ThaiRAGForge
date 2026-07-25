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

`match_courses_by_name` is a separate, query-side-only matcher (mirrors
person_loader's no-title dictionary fallback): a real user names a course
by its title, not its 8-digit code, so detect_entities needs a name->code
path too. Built from data/entity_dictionaries/courses.json
(tools/corpus_prep/build_course_dictionary.py), which extracts the English
title that structurally follows every validated course-code occurrence
(Thai titles are extracted far less reliably -- confirmed by direct visual
review -- and are excluded entirely, Latin/English only). Course names
collide far more than program names (generic titles like "GENERAL PHYSICS
1" and sequential-section electives like "TECHNOLOGY MANAGEMENT IN DAILY
LIFE" x57 are genuinely reused across many different codes in this corpus),
so the dictionary only contains names that are unique to exactly one code --
gated on actual uniqueness, not name length or word count, since short
names like "BIG DATA" or "ROUTE SURVEY" are perfectly good anchors once
confirmed unique. Matching is near-exact substring (not program_loader's
SequenceMatcher fuzzy ratio): query text is human-typed, not OCR'd, so
exact containment is both sufficient and safer against false positives on
short queries. Word-boundary checks only look at immediate neighbor
characters (not regex `\b`), because `\b` never fires between a Thai
character and a Latin one (Thai is `\w` under Unicode) when a query has no
space at that seam, e.g. "...วิชาBIG DATAเปิด...".
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
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

_NAME_DICT_PATH = Path(__file__).resolve().parents[3] / "data" / "entity_dictionaries" / "courses.json"
_ALNUM = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


@lru_cache(maxsize=1)
def load_name_dictionary() -> list[dict[str, str]]:
    return json.loads(_NAME_DICT_PATH.read_text(encoding="utf-8"))


def match_courses_by_name(query: str, dictionary: list[dict[str, str]] | None = None) -> list[str]:
    """Every course code whose unique canonical name is found in `query` as
    a near-exact, case-insensitive substring. Query-side only -- corpus
    tagging (CourseLoader) stays on match_courses/code matching, unchanged."""
    dictionary = dictionary if dictionary is not None else load_name_dictionary()
    found: set[str] = set()
    for entry in dictionary:
        name = entry["canonical"]
        for m in re.finditer(re.escape(name), query, re.IGNORECASE):
            before = query[m.start() - 1] if m.start() > 0 else ""
            after = query[m.end()] if m.end() < len(query) else ""
            if before in _ALNUM or after in _ALNUM:
                continue  # e.g. "BIG DATABASE" must not match "BIG DATA"
            found.add(entry["code"])
            break
    return sorted(found)


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
