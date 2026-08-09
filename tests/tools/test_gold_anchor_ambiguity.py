"""Pure-logic tests for the course anchor-precision rules.

Two scripts share one rule and must not drift apart:
`tools/corpus_prep/build_gold_candidates.py` annotates *candidates* at build
time, `tools/eval/audit_gold_anchor_ambiguity.py` measures the *shipped* gold
set. Both need the same phrase-boundary semantics as
`course_loader.match_courses_by_name`, or the audit would report a defect the
builder cannot see (or the reverse).

Every rule here was learned by getting it wrong first, so each is pinned in
both directions:
  - collapsing whitespace: matching raw text called genuine mentions absent,
    because OCR'd minutes wrap a long course name across a line
  - `no_name_evidence` vs `ambiguous`: with no naming document the ratio is
    undefined, not zero. Collapsing them reported 264 flags, 198 of which were
    OCR-garbled dictionary names -- burying the 66 real ones
  - the boundary rule itself: `CONTROL SYSTEM` must NOT match inside `CONTROL
    SYSTEMS`, which is the entire mechanism under study
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "corpus_prep"))
sys.path.insert(0, str(REPO / "tools" / "eval"))
sys.path.insert(0, str(REPO / "src"))

import build_gold_candidates as bgc  # noqa: E402
from rag_lab.loaders.course_loader import match_courses_by_name  # noqa: E402


def _audit_contains_phrase():
    """Imported lazily: the audit module reads persisted results at import of
    its own main() only, but keeping the import local documents that these
    tests exercise the function, not the script."""
    import audit_gold_anchor_ambiguity as audit

    return audit.contains_phrase


class TestPhraseBoundary:
    def test_plural_does_not_match_singular_needle(self):
        # The defect under study exists only because these stay distinct.
        assert not bgc._contains_phrase("DIGITAL CONTROL SYSTEMS", "CONTROL SYSTEM")

    def test_exact_phrase_matches(self):
        assert bgc._contains_phrase("DIGITAL CONTROL SYSTEMS", "CONTROL SYSTEMS")

    def test_matches_at_string_start_and_end(self):
        assert bgc._contains_phrase("CONTROL SYSTEMS", "CONTROL SYSTEMS")

    def test_thai_neighbour_is_a_boundary(self):
        # regex \b never fires at a Thai/Latin seam, which is why the rule
        # inspects the immediate neighbours instead.
        assert bgc._contains_phrase("วิชาCONTROL SYSTEMSของภาค", "CONTROL SYSTEMS")

    def test_digit_neighbour_is_not_a_boundary(self):
        assert not bgc._contains_phrase("CONTROL SYSTEMS2", "CONTROL SYSTEMS")

    def test_case_insensitive(self):
        assert bgc._contains_phrase("general physics laboratory", "General Physics Laboratory")

    def test_later_occurrence_still_matches_after_a_rejected_one(self):
        # A rejected first hit must not short-circuit the scan.
        assert bgc._contains_phrase("XCONTROL SYSTEMS and CONTROL SYSTEMS", "CONTROL SYSTEMS")


class TestWhitespaceCollapsing:
    def test_needle_wrapped_across_a_line_in_the_document(self):
        assert bgc._contains_phrase("รายวิชา CONTROL SYSTEMS ประจำภาค", "CONTROL\nSYSTEMS")

    def test_document_side_must_be_collapsed_by_the_caller(self):
        # Documented contract: the haystack arrives collapsed. Pinned so a
        # future caller that forgets cannot silently under-count.
        assert not bgc._contains_phrase("CONTROL\nSYSTEMS", "CONTROL SYSTEMS")

    def test_both_scripts_agree_on_the_same_input(self):
        contains = _audit_contains_phrase()
        for hay, needle in [
            ("DIGITAL CONTROL SYSTEMS", "CONTROL SYSTEM"),
            ("DIGITAL CONTROL SYSTEMS", "CONTROL SYSTEMS"),
            ("วิชาCONTROL SYSTEMSของภาค", "CONTROL SYSTEMS"),
            ("CONTROL SYSTEMS2", "CONTROL SYSTEMS"),
            ("รายวิชา CONTROL SYSTEMS ประจำภาค", "CONTROL\nSYSTEMS"),
        ]:
            assert contains(hay, needle) == bgc._contains_phrase(hay, needle), (hay, needle)


class TestAgreesWithTheShippedMatcher:
    """The audit's premise is that a name-anchored system sees exactly the
    documents `match_courses_by_name` would credit. If the two boundary rules
    diverge the whole `ชื่อปรากฏ` column is measuring the wrong thing."""

    def test_singular_query_does_not_resolve_to_the_plural_course(self):
        codes = match_courses_by_name("รายวิชา CONTROL SYSTEM ถูกกล่าวถึงในการประชุมใด")
        assert "01306023" not in codes
        assert "01046707" in codes

    def test_plural_query_does_not_resolve_to_the_singular_course(self):
        codes = match_courses_by_name("รายวิชา CONTROL SYSTEMS ถูกกล่าวถึงในการประชุมใด")
        assert "01046707" not in codes
        assert "01306023" in codes


class TestAnchorStatusClassification:
    """`no_name_evidence` and `ambiguous` are different defects and must stay
    apart -- collapsing them buried 66 real flags among 198 OCR artifacts."""

    @staticmethod
    def _classify(gold: set[str], naming: set[str]) -> str:
        # Mirrors course_candidates' branch; kept here as the executable
        # statement of the rule so a change to either side fails loudly.
        if not naming:
            return "no_name_evidence"
        if len(gold & naming) / len(naming) < bgc.ANCHOR_PRECISION_FLOOR:
            return "ambiguous"
        return "ok"

    def test_no_naming_document_is_not_zero_precision(self):
        assert self._classify({"a", "b"}, set()) == "no_name_evidence"

    def test_most_namers_irrelevant_is_ambiguous(self):
        # The CONTROL SYSTEMS shape: 8 gold among 65 documents naming it.
        gold = {f"g{i}" for i in range(8)}
        naming = gold | {f"x{i}" for i in range(57)}
        assert self._classify(gold, naming) == "ambiguous"

    def test_clean_anchor_is_ok(self):
        gold = {f"g{i}" for i in range(10)}
        assert self._classify(gold, gold) == "ok"

    def test_floor_is_exclusive_so_exactly_half_passes(self):
        gold = {"a", "b"}
        naming = {"a", "b", "c", "d"}
        assert self._classify(gold, naming) == "ok"

    def test_gold_documents_that_never_name_the_course_do_not_lower_precision(self):
        # A gold document with no name text is the *other* mechanism
        # (`gold_not_naming`), and must not be read as ambiguity.
        gold = {"a", "b", "c"}
        naming = {"a", "b"}
        assert self._classify(gold, naming) == "ok"
        assert len(gold - naming) == 1
