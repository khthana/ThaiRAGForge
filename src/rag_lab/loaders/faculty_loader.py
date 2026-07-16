"""Tags each Resolution with which canonical faculty/unit(s) it mentions,
stored in metadata['faculties'] for filtering (same convention as
NERLoader's metadata['entities']).

Matches against the hand-confirmed dictionary at
data/entity_dictionaries/faculties.json rather than running NER: generic Thai
NER does not reliably catch organizational units on this corpus (see
tools/corpus_prep/scan_entity_candidates.py's module docstring), and the unit
list is a closed, small vocabulary anyway -- exact/fuzzy string matching
against a confirmed list is both simpler and more precise than a model here.

Fuzzy (not just exact-substring) matching is needed because the corpus has
known OCR corruption (see docs/llm-ocr-scan-log.md) that can misread a
character inside a faculty name without touching its คณะ/วิทยาลัย/สถาบัน/
วิทยาเขต prefix -- an exact match would silently miss those mentions.
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

_DICT_PATH = Path(__file__).resolve().parents[3] / "data" / "entity_dictionaries" / "faculties.json"

# Same false-positive as tools/corpus_prep/scan_entity_candidates.py: a
# prefix search for "วิทยาลัย" alone matches the tail of "มหาวิทยาลัย"
# (university) unless excluded, which would tag every OTHER university
# mentioned in the corpus (MOUs, curriculum benchmarking) as if it were an
# internal unit. Duplicated here rather than imported -- tools/corpus_prep is
# read-only diagnostic tooling outside this package's boundary, not something
# src/rag_lab imports from (see loaders/common.py's strip_mapping_tables for
# the same convention).
_NOT_PRECEDED_BY_MAHA = "(?<!มหา)"

# Picked by checking real corpus samples: high enough that an unrelated
# candidate sharing the same prefix never wins, low enough to tolerate a
# handful of OCR-misread characters in a ~15-30 char faculty name.
_MATCH_THRESHOLD = 0.82

# Extra chars of slack past the canonical name's own length, to absorb OCR
# insertions without growing the window so large it starts overlapping the
# next sentence.
_WINDOW_SLACK = 4


@lru_cache(maxsize=1)
def load_dictionary() -> list[dict[str, str]]:
    return json.loads(_DICT_PATH.read_text(encoding="utf-8"))


def _by_prefix(dictionary: list[dict[str, Any]]) -> dict[str, list[tuple[str, str]]]:
    """prefix_type -> [(name_to_match_against, canonical), ...], where
    name_to_match_against is the canonical name itself plus any of its
    `aliases` (e.g. a prior official name, still used in older documents
    from before a rename -- not an OCR typo, so it needs its own dictionary
    entry rather than relying on the fuzzy-match tolerance)."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    for entry in dictionary:
        canonical = entry["canonical"]
        names = [canonical, *entry.get("aliases", [])]
        bucket = grouped.setdefault(entry["prefix_type"], [])
        bucket.extend((name, canonical) for name in names)
    return grouped


def _build_scan_pattern(prefixes: tuple[str, ...]) -> re.Pattern:
    alternation = "|".join(re.escape(p) for p in prefixes)
    return re.compile(rf"{_NOT_PRECEDED_BY_MAHA}(?:{alternation})")


def match_faculties(text: str, dictionary: list[dict[str, Any]] | None = None) -> list[str]:
    """Every canonical faculty/unit name found in `text`, deduped and sorted.
    A match against an alias is still reported under its canonical name --
    callers never see the alias spelling.

    For each prefix occurrence (คณะ/วิทยาลัย/สถาบัน/วิทยาเขต), only the
    names sharing that same prefix_type are compared -- a คณะ occurrence
    can never match a วิทยาลัย candidate, keeping the fuzzy comparison both
    cheap and unambiguous."""
    dictionary = dictionary if dictionary is not None else load_dictionary()
    grouped = _by_prefix(dictionary)
    pattern = _build_scan_pattern(tuple(grouped))
    found: set[str] = set()
    for m in pattern.finditer(text):
        pos = m.start()
        for name, canonical in grouped[m.group(0)]:
            window = text[pos : pos + len(name) + _WINDOW_SLACK]
            if SequenceMatcher(None, name, window).ratio() >= _MATCH_THRESHOLD:
                found.add(canonical)
                break
    return sorted(found)


@loader_registry.register("faculty")
class FacultyLoader(BaseLoader):
    """Tags each Resolution with the canonical faculty/unit name(s) it
    mentions, stored in metadata['faculties']."""

    def load(self, path: str) -> Resolution:
        text = strip_mapping_tables(strip_document_header(read_text(path)))
        year, session, title = parse_path(path)
        metadata: dict[str, Any] = {
            "year": year,
            "session": session,
            "title": title,
            "faculties": match_faculties(text),
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
