"""Tags each Resolution with which titled academic person(s) it mentions,
stored in metadata['people'] -- same convention as FacultyLoader's
metadata['faculties'] and ProgramLoader's metadata['programs'].

Rule-based, not model-based: an academic rank (ผศ./รศ./ศ./ดร., abbreviated
or spelled out in full) directly precedes a person's name with no space in
this corpus's convention -- a strong, cheap, deterministic anchor. This is
the same reasoning that moved faculties/programs off generic NER (see
faculty_loader.py's docstring: it fragments this corpus's own institution
name into multiple ORG spans). Deliberately scoped to titled academic
personnel only, the priority use case (searching by "ผศ.ดร.X") -- an
untitled person mention (a student's name, someone referred to without an
academic rank) is a different, harder, open-vocabulary problem this module
does not attempt; a rank-based pattern has zero recall for it by
construction. Model-based NER (see ner_loader.py) is the complementary tool
for that gap if it's ever revisited, not a competitor to this one -- the
two catch different things.

Person names are open-vocabulary (unlike the ~20 faculties or ~253
programs), so there is no hand-confirmed dictionary here: every match is
accepted and reported as-is, deduped per document. Cross-document identity
resolution (the same person spelled slightly differently across mentions,
or promoted from ผศ. to รศ. between meetings) is out of scope for this
first pass.
"""
from __future__ import annotations

import json
import re
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

_PEOPLE_DICT_PATH = Path(__file__).resolve().parents[3] / "data" / "entity_dictionaries" / "people.json"

# Spelled-out rank -> its abbreviated form. Prose tends to spell ranks out
# ("รองศาสตราจารย์ ดร.คมสัน มาลีสี"); committee-member tables use the
# abbreviation ("ผศ.ดร.พิชชา..."). Normalizing both to the abbreviated
# spelling means the same person tagged from either phrasing lands on the
# same metadata string.
_RANK_NORMALIZE = {
    "ผู้ช่วยศาสตราจารย์": "ผศ.",
    "รองศาสตราจารย์": "รศ.",
    "ศาสตราจารย์": "ศ.",
}

# Longest-alternative-first: "ผศ.ดร." must be tried before bare "ผศ." (and
# likewise for the spelled-out + ดร. combination) so a title with ดร.
# attached doesn't get matched as the shorter rank alone, leaving "ดร."
# dangling to be misparsed as the start of the name.
_TITLE = (
    r"(?:ผู้ช่วยศาสตราจารย์|รองศาสตราจารย์|ศาสตราจารย์)\s*ดร\.|"
    r"(?:ผศ\.|รศ\.|ศ\.)ดร\.|"
    r"ผู้ช่วยศาสตราจารย์|รองศาสตราจารย์|ศาสตราจารย์|"
    r"ผศ\.|รศ\.|ศ\.|ดร\."
)

# Thai consonants/vowels/tone-marks, deliberately excluding U+0E3F (the Baht
# currency sign, which falls inside the naive U+0E01-U+0E4F range and would
# otherwise slip into a name token from a nearby stipend/price in a table)
# and U+0E4F (a paragraph-marker punctuation glyph). Capped at 18 chars --
# comfortably above the longest legitimate given name/surname observed in
# this corpus (14) -- so an OCR-dropped space between a real surname and the
# next word (e.g. "...พันธุ์เป็นผู้เชี่ยวชาญ" with no space before "เป็น")
# leaks in fewer extra characters than an unbounded token would. This is a
# damage-limiting cap, not a fix: when the source text is missing the space
# entirely, no length heuristic recovers the true boundary.
_THAI_CHAR = r"[ก-ฮะ-ฺเ-๎]"
_NAME_TOKEN = rf"{_THAI_CHAR}{{2,18}}"

# The given name and surname are separated by a real space in prose, but by
# an OCR-to-markdown-table linebreak in committee-member tables (see e.g.
# "ผศ.ดร.วุฒิชัย<br/>ชาติพัฒนาบันท์").
_SEP = r"(?:\s+|<br\s*/?>\s*)"

_TITLED_NAME = re.compile(rf"({_TITLE})({_NAME_TOKEN}){_SEP}({_NAME_TOKEN})")

# Common function/pronoun words that are valid "Thai character runs" and so
# would otherwise pass as a plausible given name -- seen in practice from
# procedural text that mentions a rank generically rather than naming
# someone (e.g. "...ตำแหน่ง ศ. นั้น จะได้ทรงพระกรุณาโปรดเกล้าฯ แต่งตั้ง...",
# about the appointment process itself, not a specific ศาสตราจารย์).
_NOT_A_NAME = {"นั้น", "นี้", "ที่", "จะ", "ได้", "เป็น", "ว่า", "ซึ่ง", "อัน", "ให้", "แต่"}


def _normalize_title(raw: str) -> str:
    has_dr = "ดร." in raw
    rank = raw.replace("ดร.", "").strip()
    abbr_rank = _RANK_NORMALIZE.get(rank, rank)
    return f"{abbr_rank}ดร." if has_dr else abbr_rank


def match_people(text: str) -> list[dict[str, str]]:
    """Every titled academic person mentioned in `text`: rank normalized to
    its abbreviated form, given name, and surname -- deduped (the same
    person is routinely mentioned more than once: a committee list, then
    again in prose) and sorted for determinism."""
    found: set[tuple[str, str, str]] = set()
    for m in _TITLED_NAME.finditer(text):
        given = m.group(2)
        if given in _NOT_A_NAME:
            continue
        found.add((_normalize_title(m.group(1)), given, m.group(3)))
    return [
        {
            "title": title,
            "given_name": given,
            "surname": surname,
            "full_name": f"{title}{given} {surname}",
        }
        for title, given, surname in sorted(found)
    ]


@lru_cache(maxsize=1)
def load_people_dictionary() -> list[dict[str, Any]]:
    return json.loads(_PEOPLE_DICT_PATH.read_text(encoding="utf-8"))


def match_people_by_dictionary(
    query: str, dictionary: list[dict[str, Any]] | None = None
) -> list[dict[str, str]]:
    """Untitled, dictionary-based fallback for query-side entity detection
    only (see router.detect_entities) -- corpus tagging stays on
    match_people's title-anchored regex above, unchanged. A user typing a
    search query usually won't include an academic rank the way this
    corpus's own documents consistently do, so this scans people.json's
    canonical (given, surname) pairs -- and their known OCR-variant aliases
    -- for a substring match against `query` directly, with no title
    required. Returns the same dict shape as match_people (minus 'title',
    which a caller keying on given_name+surname doesn't need)."""
    dictionary = dictionary if dictionary is not None else load_people_dictionary()
    found: dict[tuple[str, str], dict[str, str]] = {}
    for entry in dictionary:
        candidates = [(entry["canonical_given"], entry["canonical_surname"])]
        candidates += [(a["given"], a["surname"]) for a in entry.get("aliases", [])]
        for given, surname in candidates:
            if given and surname and f"{given} {surname}" in query:
                key = (entry["canonical_given"], entry["canonical_surname"])
                found[key] = {
                    "given_name": entry["canonical_given"],
                    "surname": entry["canonical_surname"],
                    "full_name": entry["canonical_full_name"],
                }
    return sorted(found.values(), key=lambda d: (d["given_name"], d["surname"]))


@loader_registry.register("person")
class PersonLoader(BaseLoader):
    """Tags each Resolution with the titled academic person(s) it mentions,
    stored in metadata['people']."""

    def load(self, path: str) -> Resolution:
        text = strip_mapping_tables(strip_document_header(read_text(path)))
        year, session, title = parse_path(path)
        metadata: dict[str, Any] = {
            "year": year,
            "session": session,
            "title": title,
            "people": match_people(text),
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
