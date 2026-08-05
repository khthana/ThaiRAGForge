"""Pure-logic tests for tag_faculties.py: aggregation and report rendering
over a precomputed relpath->faculties mapping. No corpus I/O -- tag_corpus's
file-walking loop is exercised manually against the real corpus, same
convention as the rest of tools/corpus_prep/ (see test_reocr_consensus_pages.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep"))
import tag_faculties as tag  # noqa: E402


class TestSummarize:
    def test_counts_and_unmatched(self):
        tags = {
            "a.md": ["คณะวิศวกรรมศาสตร์"],
            "b.md": ["คณะวิศวกรรมศาสตร์", "คณะวิทยาศาสตร์"],
            "c.md": [],
        }
        summary = tag.summarize(tags)
        assert summary["total"] == 3
        assert summary["counts"]["คณะวิศวกรรมศาสตร์"] == 2
        assert summary["counts"]["คณะวิทยาศาสตร์"] == 1
        assert summary["unmatched"] == ["c.md"]

    def test_empty_corpus(self):
        summary = tag.summarize({})
        assert summary == {"total": 0, "counts": {}, "unmatched": []}


class TestRenderReport:
    def test_report_includes_counts_and_unmatched_sample(self):
        summary = tag.summarize(
            {
                "a.md": ["คณะวิศวกรรมศาสตร์"],
                "b.md": [],
            }
        )
        report = tag.render_report(summary)
        assert "Total live files: 2" in report
        assert "Files matched to >=1 faculty: 1 (50%)" in report
        assert "Files matched to 0 faculties: 1" in report
        assert "คณะวิศวกรรมศาสตร์" in report
        assert "- b.md" in report

    def test_report_on_empty_corpus_does_not_crash(self):
        report = tag.render_report(tag.summarize({}))
        assert "Total live files: 0" in report
        assert "n/a" in report
