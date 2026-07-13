"""Ticket 1 (tickets.md) -- pure-logic tests for the consensus review app.

No Streamlit, no real corpus: everything runs against small synthetic
fixtures. See tools/corpus_prep/consensus_review/SPEC.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep" / "consensus_review"))
import logic  # noqa: E402

_CONSENSUS_FIXTURE = """\
# Consensus flags -- both models agree (priority review list)

Auto-generated from `full_review_<year>.md` files.

Total files with at least one consensus page: 3
Total consensus pages across the corpus: 4

## [2567] 2567\\ครั้งที่ 9\\เอกสาร ก.md  (2 consensus page(s))
### Page 1
- **[phi4:latest]** sentence is garbled and incoherent
  > บางข้อความที่พัง
- **[gemma4:e4b]** garbled words and unclear transitions
  > บางข้อความ
### Page 3
- **[phi4:latest]** nonsense insertion mid-sentence
  > อีกข้อความ

## [2567] 2567\\ครั้งที่ 9\\เอกสาร ข__1.md  (1 consensus page(s))
### Page 2
- **[phi4:latest]** broken words
  > span หนึ่ง
- **[gemma4:e4b]** garbled
"""


def _write_fixture(tmp_path) -> Path:
    path = tmp_path / "consensus_priority.md"
    path.write_text(_CONSENSUS_FIXTURE, encoding="utf-8")
    return path


def test_parse_consensus_priority_preserves_file_order(tmp_path):
    entries = logic.parse_consensus_priority(_write_fixture(tmp_path))

    assert [e.file for e in entries] == [
        "2567\\ครั้งที่ 9\\เอกสาร ก.md",
        "2567\\ครั้งที่ 9\\เอกสาร ข__1.md",
    ]
    assert entries[0].year == "2567"


def test_parse_consensus_priority_captures_pages_and_model_flags(tmp_path):
    entries = logic.parse_consensus_priority(_write_fixture(tmp_path))
    doc_a = entries[0]

    assert [p.page for p in doc_a.pages] == ["Page 1", "Page 3"]

    page1 = doc_a.pages[0]
    assert set(page1.models) == {"phi4:latest", "gemma4:e4b"}
    assert page1.models["phi4:latest"].reason == "sentence is garbled and incoherent"
    assert page1.models["phi4:latest"].span == "บางข้อความที่พัง"
    assert page1.models["gemma4:e4b"].span == "บางข้อความ"


def test_parse_consensus_priority_handles_model_line_with_no_span(tmp_path):
    entries = logic.parse_consensus_priority(_write_fixture(tmp_path))
    doc_b = entries[1]
    page2 = doc_b.pages[0]

    assert page2.models["phi4:latest"].span == "span หนึ่ง"
    assert page2.models["gemma4:e4b"].reason == "garbled"
    assert page2.models["gemma4:e4b"].span == ""


def test_load_page_markdown_returns_the_matching_page_body(tmp_path):
    doc_dir = tmp_path / "2567" / "ครั้งที่ 9"
    doc_dir.mkdir(parents=True)
    (doc_dir / "เอกสาร ก.md").write_text(
        "## Page 1\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Page 3\nข้อความปกติ",
        encoding="utf-8",
    )

    body = logic.load_page_markdown(tmp_path, "2567\\ครั้งที่ 9\\เอกสาร ก.md", "Page 1")

    assert body is not None
    assert "| a | b |" in body
    assert "1 | 2" in body


def test_load_page_markdown_returns_none_for_missing_page(tmp_path):
    doc_dir = tmp_path / "2567" / "ครั้งที่ 9"
    doc_dir.mkdir(parents=True)
    (doc_dir / "เอกสาร ก.md").write_text("## Page 1\ntext", encoding="utf-8")

    assert logic.load_page_markdown(tmp_path, "2567\\ครั้งที่ 9\\เอกสาร ก.md", "Page 99") is None


def test_load_page_markdown_returns_none_for_missing_file(tmp_path):
    assert logic.load_page_markdown(tmp_path, "2567\\ครั้งที่ 9\\ไม่มีไฟล์นี้.md", "Page 1") is None


def test_is_split_piece_true_only_for_dunder_n_filenames(tmp_path):
    entries = logic.parse_consensus_priority(_write_fixture(tmp_path))
    doc_a, doc_b = entries

    assert logic.is_split_piece(doc_a.file) is False
    assert logic.is_split_piece(doc_b.file) is True


def test_consensus_siblings_empty_when_not_a_split_piece_or_no_sibling_flagged(tmp_path):
    entries = logic.parse_consensus_priority(_write_fixture(tmp_path))
    doc_a, doc_b = entries

    # doc_a is not a split piece at all.
    assert logic.consensus_siblings(entries, doc_a) == []

    # doc_b (__1) is a split piece, but no sibling __N piece is also
    # consensus-flagged in this small fixture -- still a split piece
    # (is_split_piece is True), just with no consensus sibling to list.
    assert logic.is_split_piece(doc_b.file) is True
    assert logic.consensus_siblings(entries, doc_b) == []


def test_consensus_siblings_finds_sibling_when_present(tmp_path):
    fixture = _CONSENSUS_FIXTURE + (
        "\n## [2567] 2567\\ครั้งที่ 9\\เอกสาร ข__2.md  (1 consensus page(s))\n"
        "### Page 1\n"
        "- **[phi4:latest]** garbled\n"
        "- **[gemma4:e4b]** garbled\n"
    )
    path = tmp_path / "consensus_priority.md"
    path.write_text(fixture, encoding="utf-8")
    entries = logic.parse_consensus_priority(path)

    doc_b1 = next(e for e in entries if e.file.endswith("ข__1.md"))
    siblings = logic.consensus_siblings(entries, doc_b1)

    assert siblings == ["2567\\ครั้งที่ 9\\เอกสาร ข__2.md"]


# --- Ticket 2 (tickets.md): decision log --------------------------------


def test_append_decision_writes_one_json_line(tmp_path):
    log_path = tmp_path / "review_decisions.jsonl"

    logic.append_decision(
        log_path, "2567", "2567\\ครั้งที่ 9\\เอกสาร ก.md", logic.VERDICT_REOCR,
        note="หน้า 3 เท่านั้น", timestamp="2026-07-13T00:00:00+00:00",
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record == {
        "year": "2567",
        "file": "2567\\ครั้งที่ 9\\เอกสาร ก.md",
        "verdict": logic.VERDICT_REOCR,
        "note": "หน้า 3 เท่านั้น",
        "timestamp": "2026-07-13T00:00:00+00:00",
    }


def test_append_decision_appends_without_removing_previous_lines(tmp_path):
    log_path = tmp_path / "review_decisions.jsonl"

    logic.append_decision(log_path, "2567", "ก.md", logic.VERDICT_UNSURE, timestamp="t1")
    logic.append_decision(log_path, "2567", "ก.md", logic.VERDICT_REOCR, timestamp="t2")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_load_decisions_returns_empty_list_when_file_missing(tmp_path):
    assert logic.load_decisions(tmp_path / "does_not_exist.jsonl") == []


def test_load_decisions_parses_each_line_in_order(tmp_path):
    log_path = tmp_path / "review_decisions.jsonl"
    logic.append_decision(log_path, "2567", "ก.md", logic.VERDICT_UNSURE, timestamp="t1")
    logic.append_decision(log_path, "2567", "ข.md", logic.VERDICT_FALSE_POSITIVE, timestamp="t2")

    decisions = logic.load_decisions(log_path)

    assert [d.file for d in decisions] == ["ก.md", "ข.md"]
    assert [d.verdict for d in decisions] == [logic.VERDICT_UNSURE, logic.VERDICT_FALSE_POSITIVE]


def test_resolve_decisions_latest_record_per_file_wins(tmp_path):
    log_path = tmp_path / "review_decisions.jsonl"
    logic.append_decision(log_path, "2567", "ก.md", logic.VERDICT_UNSURE, timestamp="t1")
    logic.append_decision(log_path, "2567", "ข.md", logic.VERDICT_REOCR, timestamp="t2")
    # Changed my mind about ก.md -- this later record should win, and the
    # earlier "ไม่แน่ใจ" line above must still be present in the raw log
    # (append-only, nothing rewritten in place).
    logic.append_decision(log_path, "2567", "ก.md", logic.VERDICT_REOCR, timestamp="t3")

    raw = logic.load_decisions(log_path)
    assert len(raw) == 3  # nothing was removed

    resolved = logic.resolve_decisions(raw)
    assert resolved["ก.md"].verdict == logic.VERDICT_REOCR
    assert resolved["ข.md"].verdict == logic.VERDICT_REOCR
    assert len(resolved) == 2
