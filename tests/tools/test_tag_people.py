"""Pure-logic tests for tag_people.py: aggregation and report rendering over
a precomputed relpath->people mapping. No corpus I/O -- tag_corpus's
file-walking loop is exercised manually against the real corpus, same
convention as the rest of tools/corpus_prep/ (see test_tag_faculties.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep"))
import tag_people as tag  # noqa: E402

_KOMSAN = {
    "title": "รศ.ดร.",
    "given_name": "คมสัน",
    "surname": "มาลีสี",
    "full_name": "รศ.ดร.คมสัน มาลีสี",
}
_PICHA = {
    "title": "ผศ.ดร.",
    "given_name": "พิชชา",
    "surname": "ประสิทธิ์มีบุญ",
    "full_name": "ผศ.ดร.พิชชา ประสิทธิ์มีบุญ",
}


class TestSummarize:
    def test_counts_and_unmatched(self):
        tags = {
            "a.md": [_KOMSAN],
            "b.md": [_KOMSAN, _PICHA],
            "c.md": [],
        }
        summary = tag.summarize(tags)
        assert summary["total"] == 3
        assert summary["counts"][_KOMSAN["full_name"]] == 2
        assert summary["counts"][_PICHA["full_name"]] == 1
        assert summary["unmatched"] == ["c.md"]

    def test_empty_corpus(self):
        summary = tag.summarize({})
        assert summary == {"total": 0, "counts": {}, "unmatched": []}


class TestRenderReport:
    def test_report_includes_counts(self):
        summary = tag.summarize({"a.md": [_KOMSAN], "b.md": []})
        report = tag.render_report(summary)
        assert "Total live files: 2" in report
        assert "Files matched to >=1 titled person: 1 (50%)" in report
        assert "Files matched to 0 people: 1" in report
        assert "Distinct people matched: 1" in report
        assert "รศ.ดร.คมสัน มาลีสี" in report

    def test_report_on_empty_corpus_does_not_crash(self):
        report = tag.render_report(tag.summarize({}))
        assert "Total live files: 0" in report
        assert "n/a" in report
