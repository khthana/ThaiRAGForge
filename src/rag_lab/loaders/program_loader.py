"""Tags each Resolution with which canonical program(s) (หลักสูตร) it
mentions, stored in metadata['programs'] -- same convention as
FacultyLoader's metadata['faculties'].

Matches against data/entity_dictionaries/programs.json, generated from
meeting_manifest.json titles by tools/corpus_prep/build_program_dictionary.py
(see that script's docstring for why titles, not body-text scanning, are
the dictionary's source).

Programs differ from faculties in one important way that changes the
matching algorithm: many canonical program names are literal prefixes of
other canonical names in the SAME dictionary (e.g. "...สาขาวิชาวิศวกรรม
ไฟฟ้า" is a prefix of "...สาขาวิชาวิศวกรรมไฟฟ้าและคอมพิวเตอร์" -- two
different, both real, programs). FacultyLoader's algorithm -- bound the
comparison window to len(candidate)+slack, take the first candidate to
clear the threshold -- breaks here: comparing the short candidate against a
window truncated to its own length hides the extra text that would reveal
the mention is actually the longer program, so the short candidate can win
with a near-perfect ratio purely because the window was sized for it. Fixed
by bounding the match span to where the mention actually ends in the text
(next "(", newline, or a length cap) *before* comparing anything, and
keeping the best-scoring candidate across the whole group instead of the
first one to clear the threshold.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

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

_DICT_PATH = Path(__file__).resolve().parents[3] / "data" / "entity_dictionaries" / "programs.json"

# Picked to match FacultyLoader's threshold: high enough an unrelated
# candidate never wins, low enough to tolerate a handful of OCR-misread
# characters across a much longer (avg ~60 char) canonical name.
_MATCH_THRESHOLD = 0.82

# A mention's real extent in body text ends at the first "(" (a qualifier
# like "(หลักสูตรนานาชาติ)") or newline; capped so a document with neither
# nearby doesn't hand the matcher an unbounded runaway span. Longest known
# canonical is 102 chars (see data/entity_dictionaries/programs.json); cap
# comfortably past that.
_SPAN_CAP = 120


@lru_cache(maxsize=1)
def load_dictionary() -> list[dict[str, Any]]:
    return json.loads(_DICT_PATH.read_text(encoding="utf-8"))


def _by_prefix(dictionary: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for entry in dictionary:
        grouped.setdefault(entry["prefix_type"], []).append(entry["canonical"])
    return grouped


def _build_scan_pattern(prefixes: tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(re.escape(p) for p in prefixes))


def _bounded_span(text: str, pos: int) -> str:
    """The text from `pos` up to wherever the mention actually ends --
    never the full remainder of the document."""
    end = len(text)
    for stopper in ("(", "\n"):
        idx = text.find(stopper, pos)
        if idx != -1:
            end = min(end, idx)
    end = min(end, pos + _SPAN_CAP)
    return text[pos:end].strip()


def match_programs(text: str, dictionary: list[dict[str, Any]] | None = None) -> list[str]:
    """Every canonical program name found in `text`, deduped and sorted.

    Unlike FacultyLoader's match_faculties, this takes the best-scoring
    candidate in each prefix group against a span bounded to where the
    mention actually ends -- see module docstring for why "first candidate
    to clear the threshold" is unsafe here."""
    dictionary = dictionary if dictionary is not None else load_dictionary()
    grouped = _by_prefix(dictionary)
    pattern = _build_scan_pattern(tuple(grouped))
    found: set[str] = set()
    for m in pattern.finditer(text):
        span = _bounded_span(text, m.start())
        best_canonical, best_ratio = None, 0.0
        for canonical in grouped[m.group(0)]:
            ratio = SequenceMatcher(None, canonical, span).ratio()
            if ratio > best_ratio:
                best_canonical, best_ratio = canonical, ratio
        if best_canonical is not None and best_ratio >= _MATCH_THRESHOLD:
            found.add(best_canonical)
    return sorted(found)


@loader_registry.register("program")
class ProgramLoader(BaseLoader):
    """Tags each Resolution with the canonical program (หลักสูตร) name(s) it
    mentions, stored in metadata['programs']."""

    def load(self, path: str) -> Resolution:
        text = strip_mapping_tables(strip_document_header(read_text(path)))
        year, session, title = parse_path(path)
        metadata: dict[str, Any] = {
            "year": year,
            "session": session,
            "title": title,
            "programs": match_programs(text),
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
