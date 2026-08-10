"""Pure-logic tests for the simple relation graph's extraction and voting.

Every rule pinned here is one the graph would be *wrong but plausible* without
-- an edge to the wrong faculty reads exactly like an edge to the right one:

  - the faculty must sit where `สังกัด` ends. Without that anchor a 60-char
    window happily matches a faculty named later in the same sentence and
    attributes it to whoever happened to be standing before the marker
  - a name too far from the marker yields *no* edge rather than a guessed one,
    and the distance is recorded either way so the window can be justified from
    the corpus instead of asserted
  - a name that appears *after* the marker is measured (`d_after`) but never
    used, so the report can price the direction choice
  - `no_evidence` is undefined, not zero -- the bucket collapse that buried 66
    real flags under 198 artifacts in the gold-anchor-ambiguity work
  - a source naming two faculties abstains; it cannot say which owns the
    programme, and letting it vote for both is how a tie becomes a majority
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "corpus_prep"))
sys.path.insert(0, str(REPO / "src"))

import build_relation_graph as brg  # noqa: E402

FAC_A = "คณะวิศวกรรมศาสตร์"
FAC_B = "คณะวิทยาศาสตร์"
NAME = "ผศ.ดร.สมชาย ใจดี"
NAME2 = "รศ.ดร.สมหญิง รักเรียน"


class TestAffiliationAnchor:
    def test_extracts_the_faculty_written_at_the_marker(self):
        rows = brg.affiliations_in(f"{NAME} (สังกัด{FAC_A})")
        assert [(r["person"], r["faculty"]) for r in rows] == [(NAME, FAC_A)]

    def test_ignores_a_faculty_that_is_merely_nearby(self):
        # the marker is followed by prose, not by a faculty name; the faculty
        # further along the line must not be pulled back to it
        text = f"{NAME} สังกัดหน่วยงานที่ระบุไว้ใน {FAC_A} ตามประกาศ"
        assert brg.affiliations_in(text) == []

    def test_a_marker_with_no_faculty_at_all_yields_nothing(self):
        assert brg.affiliations_in(f"{NAME} สังกัดเดิม") == []

    def test_punctuation_between_marker_and_faculty_is_tolerated(self):
        rows = brg.affiliations_in(f"{NAME} สังกัด : {FAC_B}")
        assert [r["faculty"] for r in rows] == [FAC_B]


class TestPersonWindow:
    def test_a_distant_name_gives_an_edge_with_no_person(self):
        text = f"{NAME}" + ("ก" * (brg.PERSON_WINDOW + 40)) + f" สังกัด{FAC_A}"
        (row,) = brg.affiliations_in(text)
        assert "person" not in row
        assert row["faculty"] == FAC_A
        assert row["d_before"] > brg.PERSON_WINDOW

    def test_the_nearest_preceding_name_wins(self):
        rows = brg.affiliations_in(f"{NAME} และ {NAME2} สังกัด{FAC_A}")
        assert [r["person"] for r in rows] == [NAME2]

    def test_a_name_after_the_marker_is_measured_but_never_used(self):
        (row,) = brg.affiliations_in(f"สังกัด{FAC_A} {NAME}")
        assert "person" not in row
        assert row["d_before"] is None
        assert row["d_after"] is not None


class TestClassifyIsThreeWay:
    def test_no_evidence_is_undefined_not_a_low_share(self):
        got = brg.classify(Counter(), min_votes=1)
        assert got["status"] == "no_evidence"
        assert got["share"] is None and got["faculty"] is None

    def test_conflicting_evidence_is_ambiguous_not_no_evidence(self):
        got = brg.classify(Counter({FAC_A: 2, FAC_B: 2}), min_votes=1)
        assert got["status"] == "ambiguous"
        assert got["share"] == 0.5

    def test_a_lone_vote_is_ambiguous_when_a_repeat_is_demanded(self):
        assert brg.classify(Counter({FAC_A: 1}), min_votes=2)["status"] == "ambiguous"
        assert brg.classify(Counter({FAC_A: 1}), min_votes=1)["status"] == "resolved"

    def test_no_evidence_carries_no_faculty_so_S2_can_tell_the_buckets_apart(self):
        # S2 gates on exactly this shape; if `classify` ever returned share=0.0
        # here, "undefined" would silently become "worst" and the gate is the
        # only thing that would notice.
        got = brg.classify(Counter(), min_votes=2)
        assert (got["total"], got["share"], got["faculty"]) == (0, None, None)

    def test_a_clear_majority_resolves(self):
        got = brg.classify(Counter({FAC_A: 9, FAC_B: 1}), min_votes=2)
        assert (got["status"], got["faculty"], got["votes"], got["total"]) == (
            "resolved", FAC_A, 9, 10,
        )


class TestAffiliationContext:
    """The marker's *meaning* is the point, not its count: if it is written
    only for cross-faculty appointments then A' is a biased sample of people,
    and that has to be measured rather than assumed."""

    def _raw(self, doc_faculty, marked):
        return {
            "files": {
                "f.md": {
                    "faculties": [doc_faculty],
                    "programs": [],
                    "people": [NAME],
                    "affiliations": [{"faculty": marked, "person": NAME,
                                      "d_before": 2, "d_after": None}],
                }
            },
            "titles": {},
        }

    def test_counts_a_cross_faculty_marking(self):
        got = brg.affiliation_context(self._raw(FAC_A, FAC_B))
        assert (got["same"], got["different"], got["unknown"]) == (0, 1, 0)

    def test_counts_a_same_faculty_marking(self):
        got = brg.affiliation_context(self._raw(FAC_A, FAC_A))
        assert (got["same"], got["different"], got["unknown"]) == (1, 0, 0)

    def test_an_undeterminable_owner_is_its_own_bucket(self):
        raw = self._raw(FAC_A, FAC_B)
        raw["files"]["f.md"]["faculties"] = [FAC_A, FAC_B]  # document names two
        got = brg.affiliation_context(raw)
        assert (got["same"], got["different"], got["unknown"]) == (0, 0, 1)

    def test_the_title_outranks_the_body_as_the_owning_faculty(self):
        raw = self._raw(FAC_B, FAC_B)
        raw["titles"] = {"f.md": {"title": "t", "programs": [], "faculties": [FAC_A]}}
        assert brg.affiliation_context(raw)["different"] == 1


class TestVotingAbstains:
    def test_a_source_naming_two_faculties_does_not_vote(self):
        tagsets = {
            "a": {"programs": ["P"], "faculties": [FAC_A, FAC_B]},
            "b": {"programs": ["P"], "faculties": [FAC_A]},
        }
        assert brg.one_to_one_votes(tagsets) == {"P": Counter({FAC_A: 1})}

    def test_a_source_naming_two_programs_does_not_vote(self):
        tagsets = {"a": {"programs": ["P", "Q"], "faculties": [FAC_A]}}
        assert brg.one_to_one_votes(tagsets) == {}
