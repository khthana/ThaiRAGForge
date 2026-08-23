"""Tests for `tools/eval/report_currency.py`.

The check this file exists to protect is ATTRIBUTION. The first version of the
script compared every report against the newest build across all index roots,
which called both `gold_entity_*_73det_report.md` stale while the index they were
scored on (`entity_tags_full`) had not moved -- a different root had. An
always-red check is one nobody reads, so `test_attributed_*` below fails on that
implementation by construction.

The second thing pinned here is that the exclusion buckets cannot go vacuous: a
`_`-prefixed snapshot directory and a RETIRED_REPORTS name are excluded from the
date test, and an exemption naming a file that does not exist is reported BROKEN
rather than passing silently.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))
rc = importlib.import_module("report_currency")


UTC = dt.timezone.utc


def _stamp(path: Path, when: dt.datetime) -> None:
    ts = when.timestamp()
    import os
    os.utime(path, (ts, ts))


@pytest.fixture()
def fake_tree(tmp_path, monkeypatch):
    """Two index roots built on different days, and reports to classify."""
    results = tmp_path / "results"
    index = tmp_path / "index"
    (results / "_pre_refresh").mkdir(parents=True)
    monkeypatch.setattr(rc, "RESULTS", results)
    monkeypatch.setattr(rc, "INDEX_ROOT", index)
    monkeypatch.setattr(rc, "OUT", results / "report_currency.md")

    for root, day in (("old_root", 10), ("new_root", 20)):
        d = index / root / f"{root}__combo"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(
            {"timestamp": dt.datetime(2026, 8, day, 12, 0, tzinfo=UTC).isoformat()}),
            encoding="utf-8")
    return results, index


def _write(results: Path, name: str, body: str, day: int) -> Path:
    p = results / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    _stamp(p, dt.datetime(2026, 8, day, 12, 0, tzinfo=UTC))
    return p


def test_builds_are_read_per_root(fake_tree):
    builds = rc.builds_by_root()
    assert set(builds) == {"old_root", "new_root"}
    assert builds["old_root"][0].day == 10
    assert builds["new_root"][0].day == 20


def test_attributed_report_is_judged_against_the_root_it_names(fake_tree):
    """The real regression: newer than ITS root, older than the newest root."""
    results, _ = fake_tree
    r = _write(results, "scored_on_old.md", "computed over `old_root` rows", day=15)
    when, who, attributed = rc.cutoff_for(r, rc.builds_by_root())
    assert attributed is True
    assert who.startswith("old_root")
    # Judged against the global newest (day 20) this would be stale; it is not.
    assert rc.classify(rc.builds_by_root())["current"][0][0] == "scored_on_old.md"


def test_report_naming_no_root_is_screened_against_the_global_newest(fake_tree):
    results, _ = fake_tree
    r = _write(results, "anonymous.md", "no index root is named here", day=15)
    when, who, attributed = rc.cutoff_for(r, rc.builds_by_root())
    assert attributed is False
    assert who.startswith("new_root")
    stale = rc.classify(rc.builds_by_root())["stale"]
    assert [row[0] for row in stale] == ["anonymous.md"]
    assert stale[0][4] is False, "a screened row must not claim to be attributed"


def test_report_newer_than_every_build_is_current(fake_tree):
    results, _ = fake_tree
    _write(results, "fresh.md", "no root named", day=25)
    buckets = rc.classify(rc.builds_by_root())
    assert [row[0] for row in buckets["current"]] == ["fresh.md"]
    assert buckets["stale"] == []


def test_snapshot_directory_and_retired_report_are_excluded_and_printed(fake_tree, monkeypatch):
    results, _ = fake_tree
    _write(results, "_pre_refresh/old_table.md", "no root", day=1)
    monkeypatch.setattr(rc, "RETIRED_REPORTS", {"superseded.md": "reason"})
    _write(results, "superseded.md", "no root", day=1)
    buckets = rc.classify(rc.builds_by_root())
    assert sorted(row[0] for row in buckets["retired"]) == [
        "_pre_refresh/old_table.md", "superseded.md"]
    assert buckets["stale"] == [], "an excluded report must not also be reported stale"

    body = rc.render(rc.builds_by_root(), buckets, [])
    assert "superseded.md" in body and "_pre_refresh/old_table.md" in body, (
        "the excluded bucket must be printed, or it can quietly absorb a live report")


def test_declined_and_corpus_independent_keep_their_reasons(fake_tree, monkeypatch):
    results, _ = fake_tree
    monkeypatch.setattr(rc, "NOT_WORTH_REFRESHING", {"closed.md": "the axis is closed"})
    monkeypatch.setattr(rc, "CORPUS_INDEPENDENT", {"handwritten.md": "no corpus is read"})
    _write(results, "closed.md", "no root", day=1)
    _write(results, "handwritten.md", "no root", day=1)
    buckets = rc.classify(rc.builds_by_root())
    assert [row[0] for row in buckets["declined"]] == ["closed.md"]
    assert [row[0] for row in buckets["corpus_independent"]] == ["handwritten.md"]
    body = rc.render(rc.builds_by_root(), buckets, [])
    assert "the axis is closed" in body and "no corpus is read" in body


def test_an_exemption_naming_a_missing_file_is_reported_broken(fake_tree, monkeypatch):
    results, _ = fake_tree
    monkeypatch.setattr(rc, "NOT_WORTH_REFRESHING", {"gone.md": "was deleted"})
    monkeypatch.setattr(rc, "CORPUS_INDEPENDENT", {})
    assert rc.dead_entries() == ["gone.md"]
    body = rc.render(rc.builds_by_root(), rc.classify(rc.builds_by_root()), ["gone.md"])
    assert "BROKEN" in body and "gone.md" in body


def test_the_report_states_no_four_decimal_figure(fake_tree):
    """It lands in D2's haystack, so a published value here becomes a false alibi."""
    import re
    results, _ = fake_tree
    _write(results, "anonymous.md", "no root", day=15)
    body = rc.render(rc.builds_by_root(), rc.classify(rc.builds_by_root()), [])
    assert not re.search(r"(?<![\d.])\d\.\d{4}(?!\d)", body)


def test_the_live_exemption_tables_name_files_that_exist():
    """The real tables, not a fixture -- D1c's rule applied to this script."""
    importlib.reload(rc)
    assert rc.dead_entries() == []


def test_retired_reports_is_imported_not_re_listed():
    """Two copies of a retirement rule diverge; this one must be the same object."""
    from audit_doc_claims import RETIRED_REPORTS as canonical
    importlib.reload(rc)
    assert rc.RETIRED_REPORTS is canonical
