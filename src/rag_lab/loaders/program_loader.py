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

# The degree ladder, longest first -- `ดุษฎีบัณฑิต` and `มหาบัณฑิต` both END in
# `บัณฑิต`, so a shortest-first scan would read every doctorate as a bachelor's.
_DEGREES = ("ดุษฎีบัณฑิต", "มหาบัณฑิต", "อนุปริญญา", "บัณฑิต")

# OCR wraps long programme names mid-token ("ครุศาสตร์อุตสาหกรรม บัณฑิต", or a
# <br/> inside one), and a guard that read those as "no degree" would go blind
# exactly where the corpus is messiest.
_DEGREE_NOISE = re.compile(r"(<br\s*/?>|\s|​)+")

# Canonical names are "หลักสูตร{degree} สาขาวิชา{field}"; the marker is what
# splits the two halves, and the degree guard below has to compare them apart.
_FIELD_MARKER = "สาขาวิชา"

# The scan anchor every canonical starts with, and so the left edge of the head
# noun -- see _head_of for why the LAST one in a span is the right one.
_ANCHOR = "หลักสูตร"

# A head noun is pure Thai letters. Dropping everything else in one pass is what
# makes `<br/>`, `</td><td>`, `**`, `:` and the agenda numbering `๑)` (Thai
# numerals sit at U+0E50+, outside this range) invisible to the head comparison
# -- measured: those four artifacts were every false positive the head test had.
_THAI_LETTERS_ONLY = re.compile(r"[^ก-๎]+")

# Two cuts, each picked from its own measured distribution rather than by taste,
# and each sitting in a sparse region so it is not load-bearing (the full
# histograms are in docs/program-matcher-absorption.md §6).
#
# The two halves need DIFFERENT tests, and the asymmetry is structural, not a
# fudge: `_bounded_span_for` sizes the window from the match position, so the
# head noun (immediately after the anchor) is always covered in full while the
# subject sits at the tail and is routinely cut short. So truncation must be
# forgiven in the subject and must NOT be forgiven in the head -- extra material
# in front of a head noun is a different formal degree name
# (ทันตแพทยศาสตรบัณฑิต is not แพทยศาสตรบัณฑิต): over the cached walk the head
# test contradicts on 319 mentions and 60 of them are exactly that shape.
_HEAD_THRESHOLD = 0.90
# Longest-common-substring coverage of the shorter subject: 1.0 means one
# subject is a contiguous truncation or extension of the other, i.e. the same
# name cut by the window. A plain ratio cannot express that.
_SUBJECT_COVER = 0.80


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


def degree_level(text: str) -> str | None:
    """The degree level `text` positively shows, or None if it shows none.

    None is *undecidable*, never "no degree" -- callers must not treat it as a
    mismatch ([[feedback_undefined_is_not_zero]])."""
    flat = _DEGREE_NOISE.sub("", text)
    for degree in _DEGREES:
        if degree in flat:
            return degree
    return None


def _field_of(text: str) -> str | None:
    """Everything after `สาขาวิชา`, whitespace/`<br/>` flattened, or None if the
    marker is absent -- undecidable, not empty ([[feedback_undefined_is_not_zero]])."""
    i = text.find(_FIELD_MARKER)
    return _DEGREE_NOISE.sub("", text[i + len(_FIELD_MARKER):]) if i >= 0 else None


def _field_agrees(canonical: str, span: str) -> bool:
    """Does `span` support `canonical`'s subject? False when either side has no
    subject to compare -- this gates a *re-selection*, so no evidence must mean no
    rescue, never a free pass."""
    a, b = _field_of(canonical), _field_of(span)
    if a is None or b is None:
        return False
    return SequenceMatcher(None, a, b).ratio() >= _MATCH_THRESHOLD


def _head_of(text: str) -> str | None:
    """The degree-naming half of a programme name: between the LAST `หลักสูตร`
    and the end of its degree token, non-Thai characters dropped.

    Delimited at *both* ends on purpose, and both ends were found by measurement.
    The left edge takes the last anchor because the scanner re-anchors on a
    heading (`หลักสูตร ๑) หลักสูตรวิทยาศาสตรบัณฑิต...`), which is the same name
    found twice, not a longer one -- reading from the first anchor reported 149
    such mentions as contradictions. The right edge stops at the degree token
    because a canonical with no `สาขาวิชา` lets the window overrun into the
    following sentence (`...บัณฑิตหล`), which is text the mention does not own.

    None when no degree token is present -- undecidable, never "no head"
    ([[feedback_undefined_is_not_zero]]); a span whose head the window cut mid
    token has no evidence to offer and must not be read as a contradiction."""
    i = text.find(_FIELD_MARKER)
    head = _THAI_LETTERS_ONLY.sub("", text if i < 0 else text[:i])
    if _ANCHOR in head:
        head = head[head.rindex(_ANCHOR) + len(_ANCHOR):]
    for degree in _DEGREES:  # longest-first, same reason as degree_level
        at = head.find(degree)
        if at >= 0:
            return head[: at + len(degree)]
    return None


def _head_contradicted(canonical: str, span: str) -> bool:
    """Positive evidence that the span names a different *degree* name than
    `canonical` does. Missing evidence on either side is not evidence."""
    a, b = _head_of(canonical), _head_of(span)
    if a is None or b is None:
        return False
    return SequenceMatcher(None, a, b).ratio() < _HEAD_THRESHOLD


def _subject_contradicted(canonical: str, span: str) -> bool:
    """Positive evidence that the span names a different *subject*.

    Scored by how much of the shorter subject the two share contiguously, not by
    overall similarity: the window truncates the subject routinely, and a
    truncation is the same name continuing, so `วิศวกรรมดนตรีและ` must not
    contradict `วิศวกรรมดนตรีและสื่อประสม` (ratio 0.78, coverage 1.00) while
    `วิศวกรรมเซมิคอนด` must contradict `วิศวกรรมเคมี` (coverage 0.75).

    Coverage alone is not enough to *convict*, though, and the second clause was
    added after the first walk: coverage is blind to a difference spread over the
    string rather than concentrated at one end, so a one-character OCR/dictionary
    variant (`วิศวกรรมเล็กทรอนิกส์` vs `วิศวกรรมอิเล็กทรอนิกส์`, coverage 0.55)
    reads as a contradiction while `_field_agrees` -- this project's own settled
    test for the same relation -- puts it at ratio 0.95. One pair must not be
    simultaneously agreeing and contradicting, so a drop needs BOTH tests against
    it; 133 of 569 subject drops sat in exactly that band."""
    a, b = _field_of(canonical), _field_of(span)
    if not a or not b:
        return False
    shared = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b)).size
    return (shared / min(len(a), len(b)) < _SUBJECT_COVER
            and SequenceMatcher(None, a, b).ratio() < _MATCH_THRESHOLD)


def match_programs(text: str, dictionary: list[dict[str, Any]] | None = None) -> list[str]:
    """Every canonical program name found in `text`, deduped and sorted.

    Unlike FacultyLoader's match_faculties, this keeps the best-scoring
    candidate across the whole prefix group instead of stopping at the
    first to clear the threshold -- see module docstring for why "first to
    clear" is unsafe when candidates share a prefix.

    When the winner's degree level *contradicts* its span's, the degree becomes a
    **filter on the candidate set rather than a veto on the winner**: re-select the
    best-scoring candidate that both sits at the span's own degree and agrees on the
    subject, and if none exists leave the mention untagged -- the "matches nothing"
    exit this matcher was measured to be missing
    (docs/program-matcher-absorption.md). The ratio alone cannot do it:
    `บัณฑิต` -> `มหาบัณฑิต` is one token in a ~50-char name, so the bachelor's of a
    subject scores ~0.96 against the master's text and can edge it out on
    window-tail noise; that swap was 35.7% of all absorptions.

    All three rules here were walked over the whole corpus before this one was
    picked, and two were rejected by their own numbers. A plain **fallthrough** to
    the next-best candidate re-tagged 11 mentions, only 3 of them the right
    programme (`...ดุษฎีบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า` became `...โยธา`) -- eliminating
    the winner in a prefix group leaves candidates that are by construction
    *different programmes*, so it trades a wrong degree for a wrong subject, and a
    query names the subject. A plain **reject** then dropped 340 distinct tags and
    stripped 71 files bare -- but of the 752 mentions where the guard fires, **354
    (47%) had a same-degree, same-subject candidate sitting in the dictionary
    already**: the shipped matcher had not run out of options, it had *ranked* them
    wrong, and rejecting throws that away. Selection keeps both halves: same
    removals as reject, 134 tags recovered, files losing every tag 71 -> 44. The
    subject test is what separates this from the failed fallthrough -- the 213
    mentions with a same-degree candidate but no subject agreement are exactly the
    ...ไฟฟ้า -> ...โยธา family, and they stay untagged.

    Both halves of the test fire only on positive evidence from both sides, so OCR
    that garbles a degree token, or a name with no `สาขาวิชา` to compare, leaves the
    mention with the behaviour it had ([[feedback_undefined_is_not_zero]]).

    A second guard closes the half the degree filter is structurally blind to:
    **same-degree, different programme**, where the degree agrees so the rule above
    never fires. It compares the two halves of the name *separately* (`_head_of`,
    `_field_of`), which is the whole mechanism -- the ratio is computed over the
    concatenation, so a disagreement confined to one half is diluted by the other
    half agreeing, and `หลักสูตรทันตแพทยศาสตรบัณฑิต` scores 0.88 against
    `หลักสูตรแพทยศาสตรบัณฑิต` on the strength of a shared suffix. Split apart, the
    head noun scores the same 0.88 with nothing to hide behind, and the subject
    comparison is scored by contiguous coverage so the window's own truncation is
    not mistaken for a different name. Neither cut is load-bearing: over the 9,134
    live matches carrying a head on both sides, 8,124 sit at exactly 1.00 and 157
    land in the band below the cut against 599 just above it -- a dense mass of
    agreement separated from a thin tail, not a slice through the middle of one.
    Those figures are re-derived from the cache by the audit's §3b rather than
    typed here; this docstring quotes them, it does not own them."""
    dictionary = dictionary if dictionary is not None else load_dictionary()
    grouped = _by_prefix(dictionary)
    pattern = _build_scan_pattern(tuple(grouped))
    found: set[str] = set()
    for m in pattern.finditer(text):
        scored: list[tuple[float, str, str]] = []
        for canonical in grouped[m.group(0)]:
            span = _bounded_span_for(text, m.start(), len(canonical))
            scored.append((SequenceMatcher(None, canonical, span).ratio(), canonical, span))
        qualified = [c for c in scored if c[0] >= _MATCH_THRESHOLD]
        if not qualified:
            continue
        best_ratio, best_canonical, best_span = max(qualified, key=lambda c: c[0])
        span_degree = degree_level(best_span)
        canonical_degree = degree_level(best_canonical)
        if (
            span_degree is not None
            and canonical_degree is not None
            and span_degree != canonical_degree
        ):
            # The winner names a degree the text contradicts. Re-select among the
            # candidates the text's own degree admits, best ratio first, and require
            # the subject to agree -- see docstring for why degree alone is not enough.
            for _ratio, canonical, span in sorted(qualified, reverse=True):
                if degree_level(canonical) == span_degree and _field_agrees(canonical, span):
                    found.add(canonical)
                    break
            continue
        if _head_contradicted(best_canonical, best_span) or _subject_contradicted(
            best_canonical, best_span
        ):
            # Same degree, different programme: the half the degree filter cannot
            # see. Re-selection is filtered the same way -- a candidate is only
            # admissible if nothing about it is contradicted and its subject is
            # positively supported. The degree condition is not redundant with the
            # branch above: this branch is reached precisely when the *winner's*
            # degree is uncontradicted, which says nothing about the runner-up's,
            # and without it 6 rescues in the first walk crossed บัณฑิต/มหาบัณฑิต
            # -- a cross-subject rescue silently undoing the settled degree rule.
            for _ratio, canonical, span in sorted(qualified, reverse=True):
                if (
                    (span_degree is None or degree_level(canonical) in (None, span_degree))
                    and not _head_contradicted(canonical, span)
                    and not _subject_contradicted(canonical, span)
                    and _field_agrees(canonical, span)
                ):
                    found.add(canonical)
                    break
            continue
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


# --------------------------------------------------------------------------
# Bare-field matching, and the programme identity it needs first.
#
# `match_programs` matches a full canonical name ("หลักสูตร<degree>
# สาขาวิชา<field>"). A person searching types the field alone --
# "วิศวกรรมคอมพิวเตอร์" -- which matches nothing and routes to `unmatched`.
# That gap is invisible to the Gold set, whose 30 program queries all name a
# full canonical, so nothing here can be validated by the published eval; it
# is a deployment fix, measured only by its blast radius being zero.
#
# Resolving a field to programmes needs one prior step, because a field does
# not identify a programme: 40 of 200 fields carry more than one entry. Most of
# that is real (a bachelor's, a master's and a PhD in one field are three
# programmes), but some of it is ONE programme recorded under two names, and
# those must not be returned twice to a caller that counts.
#
# Two such causes exist in this corpus, both KOSEN, both verified against the
# manifest titles the dictionary was generated from:
#
#   * the degree title was renamed in 2568, so the same programme appears as
#     `หลักสูตรอนุปริญญา สาขาวิชา<field>` (2565-2567) and
#     `หลักสูตรอนุปริญญาวิศวกรรมศาสตร์ สาขาวิชา<field>` (2568-2569). 2569/3
#     amends "ฉบับปี พ.ศ. ๒๕๖๗", i.e. the curriculum 2567/2 approved under the
#     older name -- so it is one programme, not two;
#   * one 2566 title dropped `วิศวกรรม` from the field itself
#     (`สาขาวิชาแมคคาทรอนิกส์` for `สาขาวิชาวิศวกรรมแมคคาทรอนิกส์`), which
#     mints a third entry for the same programme.
#
# Both are collapsed by rule rather than by an explicit list of the seven
# entries: a hard-coded list would silently stop covering the corpus the next
# time a manifest is added, which is this project's recurring failure shape.
# The rules are deliberately narrow -- one degree name extending the other, or
# one field name extending the other -- because the pair of signals is what
# distinguishes a rename from two real programmes. Disjoint publication years
# alone does NOT: a master's and a PhD in one field routinely fail to co-occur
# in a six-year window, and testing on that alone flags three such pairs as
# renames.
#
# ALL surface forms are kept in the group. Collapsing must never delete a
# spelling: a document titled with the retired name has to keep matching, and
# for containment scoring more spellings is strictly better.

_DEGREE_FAMILY_ANCHOR = "อนุปริญญา"


def _one_extends_the_other(a: str, b: str) -> bool:
    a, b = (a or ""), (b or "")
    return bool(a) and bool(b) and a != b and (a.startswith(b) or b.startswith(a))


def _same_programme(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Are these two dictionary entries one programme under two names?

    Requires the two entries to agree on one of {field, degree} and for the
    other to be an extension of its partner. Never merges two entries that
    differ on both, and never merges two entries that differ on neither
    (they would be the same canonical).
    """
    fa, fb = a.get("field") or "", b.get("field") or ""
    da, db = a.get("degree") or "", b.get("degree") or ""
    if fa == fb and _one_extends_the_other(da, db):
        # a renamed degree title. Narrowed to the associate-degree family: a
        # bachelor's is not a renamed master's, and "บัณฑิต" is a suffix of
        # "มหาบัณฑิต", so an unrestricted rule here would merge degree levels
        # -- the exact error the 2026-08-11 degree guard exists to prevent.
        return _DEGREE_FAMILY_ANCHOR in da and _DEGREE_FAMILY_ANCHOR in db
    # NOTE: there is deliberately NO "same degree, one field extends the other"
    # rule. It looks symmetric with the one above and it is not: a longer field
    # name is normally a DIFFERENT programme, which is the prefix-group problem
    # this module's own docstring opens with. Tried and rejected on its output,
    # not on reasoning -- it collapsed 28 entries, among them
    # `วิศวกรรมไฟฟ้า` with `วิศวกรรมไฟฟ้าสื่อสารและเครือข่าย`, `ภาษาญี่ปุ่น`
    # with `ภาษาญี่ปุ่นธุรกิจ`, and `วิศวกรรมอุตสาหการ` with three siblings.
    # The one real instance it was written for is a single 2566 title that
    # dropped `วิศวกรรม` from `สาขาวิชาวิศวกรรมแมคคาทรอนิกส์`; that is a typo in
    # one manifest, costing one count=1 entry, and it is not worth a rule that
    # cannot tell it from a real programme.
    return False


def programme_groups(
    dictionary: list[dict[str, Any]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Dictionary entries grouped into one list per real programme.

    Order is stable (dictionary order) so a caller may treat the first entry of
    a group as its representative without the choice drifting between runs.
    """
    entries = list(load_dictionary() if dictionary is None else dictionary)
    parent = list(range(len(entries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if _same_programme(entries[i], entries[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for i, e in enumerate(entries):
        grouped.setdefault(find(i), []).append(e)
    return [grouped[k] for k in sorted(grouped)]


def match_programs_by_field(
    text: str, dictionary: list[dict[str, Any]] | None = None
) -> list[str]:
    """Every canonical whose FIELD appears in `text`, deduped and sorted.

    For a query naming a bare field this returns every programme offering it --
    all of them, never a guess at which degree level was meant. A field maps to
    up to four entries here, and picking one would be the degree-swap error
    `match_programs`' own guard was added to prevent.

    Grouping does not reduce what is returned; it exists so a caller can report
    how many *programmes* were matched without counting a renamed one twice.
    Use `programme_groups` for that count.

    A field is matched as a standalone phrase, so `แมคคาทรอนิกส์` does not fire
    inside `วิศวกรรมแมคคาทรอนิกส์` and mint a spurious second programme; the
    longest field wins where several match at the same position.
    """
    from rag_lab.text_match import collapse_ws, contains_phrase

    entries = list(load_dictionary() if dictionary is None else dictionary)
    hay = collapse_ws(text)
    fields = {e["field"] for e in entries if e.get("field")}
    hit_fields = {f for f in fields if contains_phrase(hay, f)}
    # drop a field that is only matching as part of a longer matched field
    hit_fields = {
        f for f in hit_fields if not any(f != o and f in o for o in hit_fields)
    }
    return sorted(
        {e["canonical"] for e in entries if e.get("field") in hit_fields}
    )
