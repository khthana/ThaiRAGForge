"""Tags each Resolution with which titled academic person(s) it mentions,
stored in metadata['people'] -- same convention as FacultyLoader's
metadata['faculties'] and ProgramLoader's metadata['programs'].

Rule-based, not model-based: an academic rank (ผศ./รศ./ศ./ดร., abbreviated
or spelled out in full; also the bare "อ." abbreviation for "อาจารย์" --
spelled out in full it is far too generic a word to use as an anchor, see
_TITLE) directly precedes a person's name with no space in
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
#
# "อ." (bare, abbreviated "อาจารย์"/Instructor -- the base rank below
# ผศ./รศ./ศ., what most part-time/special instructors are cited with) is
# included; the spelled-out "อาจารย์" is deliberately NOT -- it's
# overwhelmingly used as a generic category noun in this corpus's own
# procedural prose ("อาจารย์พิเศษ", "อาจารย์ผู้สอนในรายวิชา", "อาจารย์ประจำ
# หลักสูตร"), and every one of those reads as a syntactically valid "title +
# 2-token name" to this regex -- confirmed empirically (corpus-wide sample)
# before adding "อ." alone: 1,415 raw matches, essentially all genuine names,
# no observed collision with "อ." as an abbreviation for "อำเภอ" (district).
_TITLE = (
    r"(?:ผู้ช่วยศาสตราจารย์|รองศาสตราจารย์|ศาสตราจารย์)\s*ดร\.|"
    r"(?:ผศ\.|รศ\.|ศ\.)ดร\.|"
    r"ผู้ช่วยศาสตราจารย์|รองศาสตราจารย์|ศาสตราจารย์|"
    r"ผศ\.|รศ\.|ศ\.|ดร\.|อ\."
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

# A second, narrower shape: title+given in one <td> cell, surname alone in
# the immediately following <td> cell -- joined by literal `</td><td>`
# markup, no whitespace/<br/> between them, so _TITLED_NAME/_SEP above never
# bridges it (confirmed 0/4 people matched on a real corpus table before
# this was added: academic_resolutions/2564/ครั้งที่ 1/
# รับรองรายงานการประชุม.md, concentrated in this "รับรองรายงานการประชุม"
# meeting-minutes rank-correction document type -- 8 genuine matches across
# 4 files/4 distinct people in a full-corpus scan).
#
# Deliberately NOT a blanket "any adjacent cell" bridge: a naive version of
# that produced a confirmed false positive in a different, OCR-corrupted
# document type ("อาจารย์พิเศษสอนเกินร้อยละ 50" teaching-load reports) --
# `ผศ.ดร.อำภาพรรณ` followed by a garbled cell reading
# "อาจารย์ผู้สอน ภายในไม่เพียงพอ" (extra corrupted prose, not a clean
# label or a surname) would otherwise be tagged as a fake person. Two
# guards fix it, both grounded in the real corpus scan, not assumed:
# (1) the following cell's content must be the surname candidate AND
# NOTHING ELSE (anchored start/end within the cell) -- the false-positive
# cell always had trailing text after the token, every genuine surname
# cell had none; (2) the candidate must be >=6 Thai chars -- the shortest
# genuine surname found is 6 chars ("มิตะถา"), and a known OCR-garbage
# fragment from the same corrupted document type ("มเชี่", 5 chars) sits
# right below that line. This is a damage-limiting heuristic, not a
# guarantee: a same-length OCR fragment could still slip through, which is
# why every match this pattern contributes should be spot-checked, not
# just the ones it changes vs. the old behavior.
_MIN_CROSS_CELL_SURNAME = 6
_TITLED_NAME_CROSS_CELL = re.compile(
    rf"<td[^>]*>({_TITLE})({_NAME_TOKEN})</td>\s*<td[^>]*>\s*"
    rf"({_THAI_CHAR}{{{_MIN_CROSS_CELL_SURNAME},18}})\s*</td>"
)

# Common function/pronoun words that are valid "Thai character runs" and so
# would otherwise pass as a plausible given name -- seen in practice from
# procedural text that mentions a rank generically rather than naming
# someone (e.g. "...ตำแหน่ง ศ. นั้น จะได้ทรงพระกรุณาโปรดเกล้าฯ แต่งตั้ง...",
# about the appointment process itself, not a specific ศาสตราจารย์). Also
# covers "อ." false positives found in the same corpus-wide check ("อ.ผู้สอน
# ตรวจสอบ...", "อ.ผู้รับ เหตุ...") -- generic role nouns following the bare
# instructor title, not a name.
_NOT_A_NAME = {
    "นั้น", "นี้", "ที่", "จะ", "ได้", "เป็น", "ว่า", "ซึ่ง", "อัน", "ให้", "แต่",
    "ผู้สอน", "ผู้รับ",
}


def _normalize_title(raw: str) -> str:
    has_dr = "ดร." in raw
    rank = raw.replace("ดร.", "").strip()
    abbr_rank = _RANK_NORMALIZE.get(rank, rank)
    return f"{abbr_rank}ดร." if has_dr else abbr_rank


def _person(title: str, given: str, surname: str) -> dict[str, str]:
    return {
        "title": title,
        "given_name": given,
        "surname": surname,
        "full_name": f"{title}{given} {surname}",
    }


def find_people(text: str) -> list[tuple[int, int, dict[str, str]]]:
    """Every person *occurrence* in `text` as (start, end, person), in the
    order they appear -- the same matches match_people reports, before it
    dedupes and sorts them away.

    Exists because a relation needs position: "the person named immediately
    before this สังกัด marker" (tools/corpus_prep/build_relation_graph.py)
    cannot be answered from a deduped, alphabetically sorted set. Kept as the
    one implementation both callers share rather than letting the graph
    builder re-derive the patterns -- two copies of _TITLE would drift, and
    the fixes those patterns carry (the bare "อ." rank, the cross-cell split
    name) were each found the hard way.
    """
    hits: list[tuple[int, int, dict[str, str]]] = []
    for pattern in (_TITLED_NAME, _TITLED_NAME_CROSS_CELL):
        for m in pattern.finditer(text):
            given = m.group(2)
            if given in _NOT_A_NAME:
                continue
            hits.append(
                (m.start(), m.end(), _person(_normalize_title(m.group(1)), given, m.group(3)))
            )
    return sorted(hits, key=lambda h: h[0])


def match_people(text: str) -> list[dict[str, str]]:
    """Every titled academic person mentioned in `text`: rank normalized to
    its abbreviated form, given name, and surname -- deduped (the same
    person is routinely mentioned more than once: a committee list, then
    again in prose) and sorted for determinism."""
    found = {
        (p["title"], p["given_name"], p["surname"]) for _, _, p in find_people(text)
    }
    return [_person(*key) for key in sorted(found)]


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
