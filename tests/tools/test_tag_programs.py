"""Pure-logic tests for tag_programs.py: aggregation and report rendering
over a precomputed relpath->programs mapping. No corpus I/O -- tag_corpus's
file-walking loop is exercised manually against the real corpus, same
convention as the rest of tools/corpus_prep/ (see test_tag_faculties.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep"))
import tag_programs as tag  # noqa: E402


class TestSummarize:
    def test_counts_and_unmatched(self):
        tags = {
            "a.md": ["หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า"],
            "b.md": [
                "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า",
                "หลักสูตรบริหารธุรกิจบัณฑิต",
            ],
            "c.md": [],
        }
        summary = tag.summarize(tags)
        assert summary["total"] == 3
        assert summary["counts"]["หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า"] == 2
        assert summary["counts"]["หลักสูตรบริหารธุรกิจบัณฑิต"] == 1
        assert summary["unmatched"] == ["c.md"]

    def test_empty_corpus(self):
        summary = tag.summarize({})
        assert summary == {"total": 0, "counts": {}, "unmatched": []}


class TestRenderReport:
    def test_report_includes_counts_and_unmatched_sample(self):
        summary = tag.summarize(
            {
                "a.md": ["หลักสูตรบริหารธุรกิจบัณฑิต"],
                "b.md": [],
            }
        )
        report = tag.render_report(summary, dict_size=253)
        assert "Total live files: 2" in report
        assert "Files matched to >=1 program: 1 (50%)" in report
        assert "Files matched to 0 programs: 1" in report
        assert "Distinct programs matched: 1 / 253" in report
        assert "หลักสูตรบริหารธุรกิจบัณฑิต" in report
        assert "- b.md" in report

    def test_report_on_empty_corpus_does_not_crash(self):
        report = tag.render_report(tag.summarize({}), dict_size=253)
        assert "Total live files: 0" in report
        assert "n/a" in report
