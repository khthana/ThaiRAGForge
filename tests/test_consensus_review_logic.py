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


def _write_manifest(corpus_root, year, session, entries):
    import json
    d = corpus_root / year / session
    d.mkdir(parents=True, exist_ok=True)
    (d / "meeting_manifest.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def test_meeting_info_returns_year_session_title_url_from_manifest(tmp_path):
    _write_manifest(tmp_path, "2567", "ครั้งที่ 9", [
        {"file": "เอกสาร ก.md", "title": "เรื่อง ทดสอบ", "url": "https://drive.google.com/x"},
    ])
    info = logic.meeting_info(tmp_path, "2567\\ครั้งที่ 9\\เอกสาร ก.md")
    assert info == {
        "year": "2567", "session": "ครั้งที่ 9",
        "title": "เรื่อง ทดสอบ", "url": "https://drive.google.com/x",
    }


def test_meeting_info_none_when_manifest_missing(tmp_path):
    assert logic.meeting_info(tmp_path, "2567\\ครั้งที่ 9\\เอกสาร ก.md") is None


def test_meeting_info_none_title_url_when_filename_not_in_manifest(tmp_path):
    _write_manifest(tmp_path, "2567", "ครั้งที่ 9", [
        {"file": "อีกไฟล์.md", "title": "อื่น", "url": "https://x"},
    ])
    info = logic.meeting_info(tmp_path, "2567\\ครั้งที่ 9\\เอกสาร ก.md")
    assert info == {"year": "2567", "session": "ครั้งที่ 9", "title": None, "url": None}


def test_meeting_info_none_when_path_has_no_year_session_prefix(tmp_path):
    assert logic.meeting_info(tmp_path, "เอกสาร ก.md") is None


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


# --- Ticket 3 (tickets.md): re-OCR worklist -------------------------------


def _resolved(*decisions: logic.Decision) -> dict[str, logic.Decision]:
    return logic.resolve_decisions(list(decisions))


def test_generate_worklist_exact_content_for_a_small_fixture(tmp_path):
    """The ticket's own wording ('assert exact worklist contents') calls for
    a hand-written expected string, not a comparison against the function's
    own output -- that would be tautological."""
    resolved = _resolved(
        logic.Decision(year="2567", file="2567\\ครั้งที่ 9\\ก.md", verdict=logic.VERDICT_REOCR),
        logic.Decision(year="2567", file="2567\\ครั้งที่ 9\\ข.md", verdict=logic.VERDICT_FALSE_POSITIVE),
        logic.Decision(year="2567", file="2567\\ครั้งที่ 9\\ค.md", verdict=logic.VERDICT_REOCR),
    )

    content = logic.generate_worklist(resolved)

    assert content == "2567\\ครั้งที่ 9\\ก.md\n2567\\ครั้งที่ 9\\ค.md\n"


def test_generate_worklist_lists_only_reocr_verdicts_sorted(tmp_path):
    resolved = _resolved(
        logic.Decision(year="2567", file="ข.md", verdict=logic.VERDICT_REOCR),
        logic.Decision(year="2567", file="ก.md", verdict=logic.VERDICT_FALSE_POSITIVE),
        logic.Decision(year="2567", file="ค.md", verdict=logic.VERDICT_REOCR),
        logic.Decision(year="2567", file="ง.md", verdict=logic.VERDICT_UNSURE),
    )

    content = logic.generate_worklist(resolved)

    assert "ก.md" not in content
    assert "ง.md" not in content
    ix, cx = content.index("ข.md"), content.index("ค.md")
    assert ix < cx  # sorted: ข before ค (codepoint / Thai alphabetical order)


def test_generate_worklist_empty_when_no_reocr_decisions(tmp_path):
    resolved = _resolved(
        logic.Decision(year="2567", file="ก.md", verdict=logic.VERDICT_FALSE_POSITIVE),
    )

    assert logic.generate_worklist(resolved) == ""


def test_write_worklist_writes_generate_worklists_exact_content(tmp_path):
    resolved = _resolved(
        logic.Decision(year="2567", file="ก.md", verdict=logic.VERDICT_REOCR),
    )
    worklist_path = tmp_path / "reocr_worklist.md"

    count = logic.write_worklist(worklist_path, resolved)

    assert count == 1
    assert worklist_path.read_text(encoding="utf-8") == logic.generate_worklist(resolved)


def test_write_worklist_regenerates_in_full_not_appends(tmp_path):
    worklist_path = tmp_path / "reocr_worklist.md"

    logic.write_worklist(worklist_path, _resolved(
        logic.Decision(year="2567", file="ก.md", verdict=logic.VERDICT_REOCR),
        logic.Decision(year="2567", file="ข.md", verdict=logic.VERDICT_REOCR),
    ))
    # Reviewer changed their mind about ก.md -- regenerating must fully
    # replace the file's contents, not append a second copy.
    logic.write_worklist(worklist_path, _resolved(
        logic.Decision(year="2567", file="ข.md", verdict=logic.VERDICT_REOCR),
    ))

    content = worklist_path.read_text(encoding="utf-8")
    assert "ก.md" not in content
    assert content.count("ข.md") == 1


# --- In-text span highlighting --------------------------------------------


def test_highlight_spans_wraps_an_exact_match():
    body = "ก่อนหน้า บางข้อความที่พัง หลังจากนั้น"

    highlighted = logic.highlight_spans(body, ["บางข้อความที่พัง"])

    assert highlighted == "ก่อนหน้า <mark>บางข้อความที่พัง</mark> หลังจากนั้น"


def test_highlight_spans_wraps_multiple_distinct_spans():
    body = "หนึ่ง สอง สาม"

    highlighted = logic.highlight_spans(body, ["หนึ่ง", "สาม"])

    assert highlighted == "<mark>หนึ่ง</mark> สอง <mark>สาม</mark>"


def test_highlight_spans_skips_a_span_not_found_verbatim():
    """The model may have paraphrased or trimmed the span slightly -- when
    it doesn't match verbatim, silently skip it rather than erroring or
    corrupting the body."""
    body = "เนื้อหาปกติ ไม่มีอะไรผิด"

    highlighted = logic.highlight_spans(body, ["ข้อความที่ไม่มีอยู่จริง"])

    assert highlighted == body


def test_highlight_spans_handles_empty_and_blank_spans():
    body = "เนื้อหาปกติ"

    assert logic.highlight_spans(body, []) == body
    assert logic.highlight_spans(body, [""]) == body
    assert logic.highlight_spans(body, ["   "]) == body


# --- Old-vs-new re-OCR adjudication review -------------------------------


def _adjudication_record(pdf="a.pdf", page=1, phi="new", gemma="new", **extra) -> dict:
    record = {
        "pdf": pdf,
        "page": page,
        "files": [f"2567\\ครั้งที่ 9\\{Path(pdf).stem}.md"],
        "old_text_source": f"2567\\ครั้งที่ 9\\{Path(pdf).stem}.md",
        "diverging_siblings": [],
        "verdicts": {
            "phi4:latest": {"verdict": phi, "reason": "x", "error": None, "elapsed_s": 1.0},
            "gemma4:e4b": {"verdict": gemma, "reason": "y", "error": None, "elapsed_s": 1.0},
        },
        "timestamp": "t",
    }
    record.update(extra)
    return record


def test_load_jsonl_returns_empty_list_for_missing_file(tmp_path):
    assert logic.load_jsonl(tmp_path / "missing.jsonl") == []


def test_load_jsonl_parses_records_and_skips_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert logic.load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_needs_reocr_review_false_when_both_models_say_new():
    record = _adjudication_record(phi="new", gemma="new")
    assert logic.needs_reocr_review(record) is False


def test_needs_reocr_review_true_on_disagreement():
    record = _adjudication_record(phi="old", gemma="new")
    assert logic.needs_reocr_review(record) is True


def test_needs_reocr_review_true_when_both_say_old():
    record = _adjudication_record(phi="old", gemma="old")
    assert logic.needs_reocr_review(record) is True


def test_needs_reocr_review_true_when_both_bad():
    record = _adjudication_record(phi="both_bad", gemma="both_bad")
    assert logic.needs_reocr_review(record) is True


def test_staged_text_by_key_maps_pdf_page_pairs():
    staging = [
        {"pdf": "a.pdf", "page": 1, "files": ["a.md"], "new_text": "new1", "timestamp": "t"},
        {"pdf": "a.pdf", "page": 2, "files": ["a.md"], "new_text": "new2", "timestamp": "t"},
    ]
    result = logic.staged_text_by_key(staging)
    assert result == {("a.pdf", 1): "new1", ("a.pdf", 2): "new2"}


def test_load_old_text_reads_the_page_from_the_corpus(tmp_path):
    doc_dir = tmp_path / "2567" / "ครั้งที่ 9"
    doc_dir.mkdir(parents=True)
    (doc_dir / "a.md").write_text("## Page 1\n\nเนื้อหาเดิม\n", encoding="utf-8")
    record = _adjudication_record(pdf="a.pdf", page=1)
    record["old_text_source"] = "2567\\ครั้งที่ 9\\a.md"

    assert logic.load_old_text(tmp_path, record) == "เนื้อหาเดิม"


def test_load_old_text_missing_page_returns_none(tmp_path):
    doc_dir = tmp_path / "2567" / "ครั้งที่ 9"
    doc_dir.mkdir(parents=True)
    (doc_dir / "a.md").write_text("## Page 1\n\nเนื้อหาเดิม\n", encoding="utf-8")
    record = _adjudication_record(pdf="a.pdf", page=5)
    record["old_text_source"] = "2567\\ครั้งที่ 9\\a.md"

    assert logic.load_old_text(tmp_path, record) is None


def test_reocr_review_decision_round_trips_through_resolve(tmp_path):
    log_path = tmp_path / "reocr_review_decisions.jsonl"
    logic.append_reocr_review_decision(log_path, "a.pdf", 1, logic.REOCR_VERDICT_APPLY_NEW)

    decisions = logic.load_reocr_review_decisions(log_path)
    assert len(decisions) == 1
    assert decisions[0].pdf == "a.pdf"
    assert decisions[0].page == 1

    resolved = logic.resolve_reocr_review_decisions(decisions)
    assert resolved[("a.pdf", 1)].verdict == logic.REOCR_VERDICT_APPLY_NEW


def test_reocr_review_decision_latest_record_wins(tmp_path):
    log_path = tmp_path / "reocr_review_decisions.jsonl"
    logic.append_reocr_review_decision(log_path, "a.pdf", 1, logic.REOCR_VERDICT_DEFER)
    logic.append_reocr_review_decision(log_path, "a.pdf", 1, logic.REOCR_VERDICT_KEEP_OLD)

    resolved = logic.resolve_reocr_review_decisions(logic.load_reocr_review_decisions(log_path))
    assert resolved[("a.pdf", 1)].verdict == logic.REOCR_VERDICT_KEEP_OLD
    assert len(logic.load_reocr_review_decisions(log_path)) == 2  # append-only, both kept
