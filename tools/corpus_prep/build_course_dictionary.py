"""Builds data/entity_dictionaries/courses.json: a code<->name dictionary
for query-side course-name matching (mirrors build_program_dictionary.py's
role for programs.json).

Source: every occurrence of a course code already validated by
course_loader.match_courses (so รหัสนักศึกษา/student-ID false positives are
excluded upstream, not re-derived here) is scanned for the English title
that structurally always follows it (course_loader's module docstring).
Thai titles are extracted far less reliably -- when no Thai title sits in
the same text window, the extractor grabs whatever Thai text happens to be
nearest instead (usually a professor's name/qualifications), not the
course's actual Thai name. Confirmed by direct visual review (50-sample
artifact) and dropped entirely per user decision -- this dictionary is
Latin/English names only.

Distinctiveness, not token count, is the primary inclusion gate: a course
name is only useful as a query-matching anchor if it resolves to exactly
one code. Generic/reused titles ("GENERAL PHYSICS 1", "THESIS",
sequential-section electives like "TECHNOLOGY MANAGEMENT IN DAILY LIFE"
x57) are real, frequent collisions in this corpus, not extraction bugs --
gating on name uniqueness (not "looks distinctive" heuristics like word
count) is what correctly keeps "BIG DATA" and "ROUTE SURVEY" (short but
unique) while dropping the generics.

A single-word minimum-length guard (>=2 words) is applied on top, as a
secondary filter: single-word "names" in the extracted data are dominated
by OCR/parsing fragments ("RADIUMICROELECTRIC", "DAMENTA", "back") that
happen not to collide with anything simply because nothing else was
truncated the same way, plus a handful of genuinely single generic English
words ("TRAINING", "PROCESS") that are unique in this corpus snapshot by
luck but would false-positive on any unrelated query containing that word.
Uniqueness alone doesn't catch either case.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from pythainlp.util import thai_digit_to_arabic_digit  # noqa: E402
from rag_lab.loaders.common import iter_corpus_files, strip_document_header, strip_mapping_tables  # noqa: E402
from rag_lab.loaders.course_loader import match_courses  # noqa: E402

CORPUS_ROOT = REPO / "academic_resolutions"
OUT_PATH = REPO / "data" / "entity_dictionaries" / "courses.json"

_CODE = re.compile(r"\d{8}")
_PREREQ = re.compile(r"PREREQUISITE|Prerequisite|วิชาบังคับก่อน")
_CREDIT_TUPLE = re.compile(r"\d\s*\(\s*\d+\s*-\s*\d+\s*-\s*\d+\s*\)")
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9 ,\-/&]{2,70}")
_WINDOW_CAP = 200
_HTML_TAG = re.compile(r"<[^>]*>?")

_STRIP_TOKENS = re.compile(
    r"^\s*(จำนวน|หน่วยกิต|หรือ|และ|:|-|,)+\s*|\s*(จำนวน|หน่วยกิต|หรือ|และ|:|-|,)+\s*$"
)
_WS = re.compile(r"\s+")
_MARKUP_NOISE_RE = re.compile(
    r"^(TD|TR|TABLE|PAGE|ROWSPAN|COLSPAN)([ 0-9]*(ROWSPAN|COLSPAN|TD|TR|PAGE))*\s*[0-9]*$",
    re.IGNORECASE,
)


def _clean(s: str) -> str:
    s = _HTML_TAG.sub(" ", s)
    s = _STRIP_TOKENS.sub("", s).strip()
    s = _WS.sub(" ", s)
    return s.strip(" -,:")


def _extract_latin_name(text: str, code_end: int, next_code_pos: int | None, next_prereq_pos: int | None) -> str | None:
    bounds = [code_end + _WINDOW_CAP]
    if next_code_pos is not None:
        bounds.append(next_code_pos)
    if next_prereq_pos is not None:
        bounds.append(next_prereq_pos)
    window = _HTML_TAG.sub(" ", text[code_end : min(bounds)])
    m = _CREDIT_TUPLE.search(window)
    segments = (window[: m.start()], window[m.end() : m.end() + 80]) if m else (window[:60], "")
    for segment in segments:
        lm = _LATIN_RUN.search(segment)
        if lm:
            cleaned = _clean(lm.group(0))
            if len(cleaned) >= 4 and not _MARKUP_NOISE_RE.match(cleaned):
                return cleaned
    return None


def extract_all() -> dict[str, set[str]]:
    names_by_code: dict[str, set[str]] = defaultdict(set)
    for f in iter_corpus_files(CORPUS_ROOT):
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")
        text = strip_mapping_tables(strip_document_header(text))
        valid_codes = set(match_courses(text))
        if not valid_codes:
            continue
        normalized = thai_digit_to_arabic_digit(text)
        code_positions = [(m.start(), m.group(0)) for m in _CODE.finditer(normalized) if m.group(0) in valid_codes]
        prereq_positions = [m.start() for m in _PREREQ.finditer(normalized)]
        for i, (start, code) in enumerate(code_positions):
            end = start + len(code)
            next_code_pos = code_positions[i + 1][0] if i + 1 < len(code_positions) else None
            next_prereq_pos = next((p for p in prereq_positions if p > end), None)
            name = _extract_latin_name(normalized, end, next_code_pos, next_prereq_pos)
            if name:
                names_by_code[code].add(name)
    return names_by_code


def build_dictionary(names_by_code: dict[str, set[str]]) -> list[dict[str, str]]:
    # Case-insensitive collision groups across the WHOLE corpus, so a name
    # is only "unique" if it belongs to exactly one code no matter how many
    # of that code's own case-variant spellings it has.
    name_to_codes: dict[str, set[str]] = defaultdict(set)
    name_display: dict[str, str] = {}
    for code, names in names_by_code.items():
        for name in names:
            key = name.upper()
            name_to_codes[key].add(code)
            # Prefer an all-caps display form (the corpus's dominant style for
            # official course titles); otherwise keep the first seen spelling.
            if key not in name_display or name == name.upper():
                name_display[key] = name

    entries = []
    for key, codes in name_to_codes.items():
        if len(codes) != 1:
            continue  # ambiguous name -- dropped, not guessable which code it means
        if len(key.split()) < 2:
            continue  # single-word: usually a fragment, or too generic to trust
        (code,) = codes
        entries.append({"code": code, "canonical": name_display[key]})
    entries.sort(key=lambda e: (e["code"], e["canonical"]))
    return entries


def main() -> None:
    names_by_code = extract_all()
    entries = build_dictionary(names_by_code)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")

    codes_with_entry = {e["code"] for e in entries}
    print(f"codes with >=1 extracted name: {len(names_by_code)}")
    print(f"codes with >=1 unique (matchable) name: {len(codes_with_entry)}")
    print(f"total matchable (name, code) entries: {len(entries)}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
