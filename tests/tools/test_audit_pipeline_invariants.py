"""E0's attribution rules, pinned in both directions.

E0 asks whether a persisted result can be attributed to exactly one built index.
It says PASS today, and a PASS is only evidence if each of the three rules that
produce it is exercised -- rule 1 (`recorded`) in particular fires on nothing
currently on disk, because the ~24k persisted results predate the field. Pinning
it here is what keeps it from being a rule that silently never runs.

The negatives matter as much: `ambiguous` is the only outcome that fails the
check, so the two lookalikes that must *not* fail it (`no built index`, `no
candidate fits`) are pinned too.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "tools/eval/audit_pipeline_invariants.py"
_spec = importlib.util.spec_from_file_location("audit_pipeline_invariants", _SRC)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

FULL = Path("data/index/chunker_compare_full/plain__fixed_size__local__ceea7536")
SMOKE = Path("data/index/chunker_compare_smoke/plain__fixed_size__local__ceea7536")
SOLO = Path("data/index/entity_tags_full/entity_tags__semantic__local__e4fe19d6")

NAME = FULL.name  # the same combo id names both the full index and the fixture
SOLO_NAME = SOLO.name


@pytest.fixture
def attributor() -> audit.IndexAttributor:
    return audit.IndexAttributor(
        {
            FULL: {"2568/1/ก", "2568/1/ข", "2568/2/ค"},
            SMOKE: {"2568/1/ก"},  # a subset, exactly as the real fixture is
            SOLO: {"2568/1/ก"},
        }
    )


def _result(ids: list[str], index_dir: str | None = None) -> dict:
    data = {"results": [{"resolution_id": r} for r in ids]}
    if index_dir is not None:
        data["index_dir"] = index_dir
    return data


def test_ambiguity_is_real_in_the_fixture(attributor):
    """The premise the other tests rest on: one combo id, two index roots."""
    assert set(attributor.ambiguous_names) == {NAME}
    assert sorted(attributor.ambiguous_names[NAME]) == sorted([FULL, SMOKE])


def test_recorded_index_dir_wins_over_every_other_rule(attributor):
    """Rule 1. The cited ids fit *both* candidates here, so elimination would
    give up -- a recorded index_dir is what makes it decidable at all."""
    data = _result(["2568/1/ก"], index_dir=str(SMOKE))

    assert attributor.attribute(data, NAME) == (SMOKE, "recorded")


def test_recorded_index_dir_is_matched_after_resolving_the_path(attributor):
    """An absolute path recorded by the writer must match the relative path the
    audit scans, or rule 1 silently degrades to rule 2/3 and never fires."""
    data = _result(["2568/1/ก"], index_dir=str(SMOKE.resolve()))

    assert attributor.attribute(data, NAME) == (SMOKE, "recorded")


def test_recorded_index_dir_that_no_longer_exists_falls_through(attributor):
    """A deleted index must not be attributed to on the strength of a stale
    record; the weaker rules take over and stay honest about what they can see."""
    data = _result(["2568/1/ก", "2568/2/ค"], index_dir="data/index/deleted/combo")

    assert attributor.attribute(data, NAME) == (FULL, "elimination")


def test_unique_name_needs_no_evidence(attributor):
    """Rule 2. Note the cited id is not one this index holds -- attribution and
    id-correctness are separate questions, and E3a is what asks the second."""
    assert attributor.attribute(_result(["nope"]), SOLO_NAME) == (SOLO, "unique name")


def test_elimination_rules_out_the_candidate_missing_a_cited_id(attributor):
    """Rule 3, and the one that resolves all 7,268 real ambiguous results: a
    result citing an id the smoke fixture does not hold cannot have come from it."""
    assert attributor.attribute(_result(["2568/1/ก", "2568/2/ค"]), NAME) == (
        FULL,
        "elimination",
    )


def test_a_result_that_fits_both_candidates_is_the_only_failure(attributor):
    """The smoke fixture is a subset of the full index, so a result citing only
    shared ids fits both -- unattributable, and the sole thing E0 fails on."""
    assert attributor.attribute(_result(["2568/1/ก"]), NAME) == (None, "ambiguous")


def test_no_built_index_is_classified_not_failed(attributor):
    """The 8 superseded combos were deleted on purpose; their 848 results name an
    index that is gone. That is not an ambiguity and must not turn E0 red."""
    assert attributor.attribute(_result(["2568/1/ก"]), "plain__gone__local__00000000") == (
        None,
        "no built index",
    )


def test_no_candidate_fits_is_classified_not_failed(attributor):
    """Drift: the result cites an id no candidate holds. That is E3a's finding,
    and double-reporting it as an attribution failure would hide which is which."""
    assert attributor.attribute(_result(["2568/9/ฮ"]), NAME) == (None, "no candidate fits")


# --- the published report's own freshness -----------------------------------
#
# A bare run prints and vanishes, which is how the 2026-08-11 probe run left a
# report on disk claiming 26 pass / 2 warn / 0 fail while the prose -- correctly
# -- said 27/1/0. The run now reads the published summary back and says whether
# it still matches. Pinned both ways: a parser that always returns None would
# make the notice permanently silent, which is the failure it exists to prevent.

def test_published_counts_parses_the_report_header(tmp_path, monkeypatch):
    p = tmp_path / "pipeline-invariant-audit.md"
    p.write_text("# Pipeline invariant audit\n\nRun 2026-08-12 00:28 UTC. "
                 "27 pass / 1 warn / 0 fail.\n", encoding="utf-8")
    monkeypatch.setattr(audit, "PUBLISHED_REPORT", p)
    assert audit._published_counts() == (27, 1, 0)


def test_published_counts_survives_a_missing_or_unreadable_report(tmp_path, monkeypatch):
    """Absent is not zero: the caller must be able to tell 'no artifact' from
    'an artifact that disagrees', or a fresh clone reads as stale."""
    monkeypatch.setattr(audit, "PUBLISHED_REPORT", tmp_path / "nope.md")
    assert audit._published_counts() is None
    p = tmp_path / "empty.md"
    p.write_text("# Pipeline invariant audit\n", encoding="utf-8")
    monkeypatch.setattr(audit, "PUBLISHED_REPORT", p)
    assert audit._published_counts() is None
