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
clear the threshold -- breaks here if applied naively: comparing the short
candidate against a window truncated to its own length hides the extra text
that would reveal the mention is actually the longer program, so the short
candidate can win with a near-perfect ratio purely because the window was
sized for it. Fixed by giving *each* candidate in a prefix group its own
window sized to that candidate's own length+slack (so a long candidate's
comparison always includes the trailing text that would distinguish it from
a short prefix) and keeping the best-scoring candidate across the whole
group, never stopping at the first to clear the threshold.

Originally (see git history) the per-candidate window was further capped at
the next "(" or newline in the source text, reasoning that a mention's real
extent in *corpus* prose always ends at one of those. That broke query-side
detection (Gold-set eval, 2026-07-25): a natural-language question has no
such marker after the program name, so the old single bounded-span dragged
the entire rest of the question into the comparison and diluted the ratio
below threshold (measured: 7/30 Gold program queries detected). The
structural-marker cap is gone; the per-candidate length+slack window alone
turned out to be sufficient discriminator for the prefix-collision case too
(verified by re-running tag_programs.py's coverage check, see docstring on
match_programs), and it works unmodified on both corpus text and queries.
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

# Same convention as FacultyLoader._WINDOW_SLACK: extra chars past a
# candidate's own length, to absorb OCR-misread/extra characters without
# dragging in unrelated trailing text (the next sentence in corpus prose, or
# the rest of the question in a query).
_WINDOW_SLACK = 4

# Absolute ceiling on any one candidate's window, independent of its own
# length -- a safety net, not the primary bound (see _bounded_span_for).
# Longest known canonical is 102 chars (see data/entity_dictionaries/
# programs.json); cap comfortably past that.
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


def _bounded_span_for(text: str, pos: int, candidate_len: int) -> str:
    """The text from `pos`, windowed to `candidate_len` + slack -- sized per
    candidate so a long candidate's window always reaches far enough to
    include the text that would distinguish it from a shorter prefix (see
    module docstring). No longer capped at the next "(" or newline: that
    only ever helped corpus text (which reliably has one nearby) and
    actively hurt query text (which doesn't) without changing which
    candidate wins in the corpus case, since the length+slack window is
    already tight enough to exclude the next sentence."""
    end = min(len(text), pos + candidate_len + _WINDOW_SLACK, pos + _SPAN_CAP)
    return text[pos:end].strip()


def match_programs(text: str, dictionary: list[dict[str, Any]] | None = None) -> list[str]:
    """Every canonical program name found in `text`, deduped and sorted.

    Unlike FacultyLoader's match_faculties, this keeps the best-scoring
    candidate across the whole prefix group instead of stopping at the
    first to clear the threshold -- see module docstring for why "first to
    clear" is unsafe when candidates share a prefix."""
    dictionary = dictionary if dictionary is not None else load_dictionary()
    grouped = _by_prefix(dictionary)
    pattern = _build_scan_pattern(tuple(grouped))
    found: set[str] = set()
    for m in pattern.finditer(text):
        best_canonical, best_ratio = None, 0.0
        for canonical in grouped[m.group(0)]:
            span = _bounded_span_for(text, m.start(), len(canonical))
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
