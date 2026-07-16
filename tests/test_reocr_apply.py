"""Pure-logic tests for reocr_apply.py (Phase 3 of the consensus re-OCR
pipeline): deciding whether an adjudicated page should be applied, rewriting
a single '## Page N' section in place, and the per-file write/backup
behavior. No real corpus or Ollama calls -- same convention as
test_reocr_adjudicate.py, leaving main()'s I/O-heavy loop itself untested.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep" / "consensus_review"))
import reocr_apply as apply_mod  # noqa: E402
import logic as review_logic  # noqa: E402


def _both_new_record(pdf="a.pdf", page=1, files=("a.md",)) -> dict:
    return {
        "pdf": pdf, "page": page, "files": list(files), "old_text_source": files[0],
        "diverging_siblings": [],
        "verdicts": {
            "phi4:latest": {"verdict": "new", "reason": "r1", "error": None, "elapsed_s": 1.0},
            "gemma4:e4b": {"verdict": "new", "reason": "r2", "error": None, "elapsed_s": 1.0},
        },
        "timestamp": "t",
    }


def _disputed_record(pdf="a.pdf", page=2, files=("a.md",)) -> dict:
    return {
        "pdf": pdf, "page": page, "files": list(files), "old_text_source": files[0],
        "diverging_siblings": [],
        "verdicts": {
            "phi4:latest": {"verdict": "old", "reason": "r1", "error": None, "elapsed_s": 1.0},
            "gemma4:e4b": {"verdict": "new", "reason": "r2", "error": None, "elapsed_s": 1.0},
        },
        "timestamp": "t",
    }


def _decision(verdict: str, note: str = ""):
    return review_logic.ReocrReviewDecision(pdf="a.pdf", page=2, verdict=verdict, note=note, timestamp="t")


class TestDecideAction:
    def test_both_models_new_auto_applies_with_no_decisions_needed(self):
        decision = apply_mod.decide_action(_both_new_record(), decisions={})
        assert decision.action == apply_mod.ACTION_APPLY
        assert "both models" in decision.reason

    def test_disputed_with_no_human_decision_skips_as_pending(self):
        decision = apply_mod.decide_action(_disputed_record(), decisions={})
        assert decision.action == apply_mod.ACTION_SKIP
        assert decision.reason == "awaiting human review"

    def test_disputed_with_apply_new_decision_applies(self):
        decisions = {("a.pdf", 2): _decision(review_logic.REOCR_VERDICT_APPLY_NEW, note="looks right")}
        decision = apply_mod.decide_action(_disputed_record(), decisions)
        assert decision.action == apply_mod.ACTION_APPLY
        assert "looks right" in decision.reason

    def test_disputed_with_keep_old_decision_skips(self):
        decisions = {("a.pdf", 2): _decision(review_logic.REOCR_VERDICT_KEEP_OLD)}
        decision = apply_mod.decide_action(_disputed_record(), decisions)
        assert decision.action == apply_mod.ACTION_SKIP
        assert "keep-old" in decision.reason

    def test_disputed_with_defer_decision_skips(self):
        decisions = {("a.pdf", 2): _decision(review_logic.REOCR_VERDICT_DEFER)}
        decision = apply_mod.decide_action(_disputed_record(), decisions)
        assert decision.action == apply_mod.ACTION_SKIP
        assert "defer" in decision.reason


class TestReplacePageText:
    def test_replaces_body_of_single_page_file(self):
        text = "## Page 1\n\nold garbled text\n"
        result = apply_mod.replace_page_text(text, 1, "fresh clean text")
        assert result == "## Page 1\n\nfresh clean text\n\n"

    def test_replaces_only_the_target_page_in_a_multi_page_file(self):
        text = "## Page 1\n\npage one\n\n## Page 2\n\npage two\n\n## Page 3\n\npage three\n"
        result = apply_mod.replace_page_text(text, 2, "NEW PAGE TWO")
        assert "page one" in result
        assert "page three" in result
        assert "page two" not in result
        assert "NEW PAGE TWO" in result
        # header ordering and page 1/3 bodies are untouched
        assert result.index("## Page 1") < result.index("NEW PAGE TWO") < result.index("## Page 3")

    def test_missing_header_returns_none(self):
        text = "## Page 1\n\nonly page\n"
        assert apply_mod.replace_page_text(text, 5, "new") is None

    def test_ambiguous_duplicate_header_returns_none(self):
        text = "## Page 1\n\nfirst copy\n\n## Page 1\n\nsecond copy\n"
        assert apply_mod.replace_page_text(text, 1, "new") is None

    def test_reapplying_the_same_text_is_idempotent(self):
        text = "## Page 1\n\nold text\n\n## Page 2\n\nother page\n"
        once = apply_mod.replace_page_text(text, 1, "new text")
        twice = apply_mod.replace_page_text(once, 1, "new text")
        assert once == twice

    def test_occurrence_selects_the_requested_duplicate(self):
        text = "## Page 1\n\nfirst copy\n\n## Page 1\n\nsecond copy\n\n## Page 2\n\nunrelated\n"
        result = apply_mod.replace_page_text(text, 1, "fixed", occurrence=2)
        assert "first copy" in result
        assert "second copy" not in result
        assert "fixed" in result
        assert "unrelated" in result

    def test_occurrence_out_of_range_returns_none(self):
        text = "## Page 1\n\nfirst copy\n\n## Page 1\n\nsecond copy\n"
        assert apply_mod.replace_page_text(text, 1, "fixed", occurrence=3) is None

    def test_occurrence_none_still_requires_exactly_one_match(self):
        text = "## Page 1\n\nfirst copy\n\n## Page 1\n\nsecond copy\n"
        assert apply_mod.replace_page_text(text, 1, "fixed") is None


class TestLoadPageOccurrenceOverrides:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        assert apply_mod.load_page_occurrence_overrides(tmp_path / "missing.json") == {}

    def test_parses_existing_file(self, tmp_path):
        path = tmp_path / "overrides.json"
        path.write_text('{"a.md": {"4": 2}}', encoding="utf-8")
        assert apply_mod.load_page_occurrence_overrides(path) == {"a.md": {"4": 2}}


class TestApplyToFile:
    def test_dry_run_does_not_write_or_back_up(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("## Page 1\n\nold\n", encoding="utf-8")
        result = apply_mod.apply_to_file(f, 1, "new", apply=False)
        assert result.status == "written"
        assert f.read_text(encoding="utf-8") == "## Page 1\n\nold\n"
        assert not (tmp_path / "doc.md.pre_reocr.bak").exists()

    def test_apply_writes_new_text_and_creates_backup_of_original(self, tmp_path):
        f = tmp_path / "doc.md"
        original = "## Page 1\n\nold\n"
        f.write_text(original, encoding="utf-8")
        result = apply_mod.apply_to_file(f, 1, "new text", apply=True)
        assert result.status == "written"
        assert "new text" in f.read_text(encoding="utf-8")
        backup = tmp_path / "doc.md.pre_reocr.bak"
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == original

    def test_apply_does_not_overwrite_an_existing_backup_on_a_later_change(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("## Page 1\n\noriginal\n", encoding="utf-8")
        apply_mod.apply_to_file(f, 1, "first fix", apply=True)
        backup_after_first = (tmp_path / "doc.md.pre_reocr.bak").read_text(encoding="utf-8")

        apply_mod.apply_to_file(f, 1, "second fix", apply=True)
        backup_after_second = (tmp_path / "doc.md.pre_reocr.bak").read_text(encoding="utf-8")
        assert backup_after_first == backup_after_second
        assert "original" in backup_after_second

    def test_already_applied_text_is_unchanged_not_rewritten(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("## Page 1\n\noriginal\n", encoding="utf-8")
        apply_mod.apply_to_file(f, 1, "fixed text", apply=True)

        result = apply_mod.apply_to_file(f, 1, "fixed text", apply=True)
        assert result.status == "unchanged"

    def test_missing_header_is_reported_without_writing(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("## Page 1\n\nonly page\n", encoding="utf-8")
        result = apply_mod.apply_to_file(f, 9, "new", apply=True)
        assert result.status == "missing_header"
        assert not (tmp_path / "doc.md.pre_reocr.bak").exists()

    def test_occurrence_resolves_a_duplicate_header_that_would_otherwise_be_ambiguous(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("## Page 1\n\nboilerplate\n\n## Page 1\n\nreal content\n", encoding="utf-8")
        result = apply_mod.apply_to_file(f, 1, "fixed content", apply=True, occurrence=2)
        assert result.status == "written"
        text = f.read_text(encoding="utf-8")
        assert "boilerplate" in text
        assert "fixed content" in text
        assert "real content" not in text
