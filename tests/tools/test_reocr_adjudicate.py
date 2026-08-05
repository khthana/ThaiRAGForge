"""Pure-logic tests for reocr_adjudicate.py (Phase 2 of the consensus re-OCR
pipeline): reading Phase 1's staging JSONL, resolving the current corpus text
for a staged page (and flagging sibling divergence), resume bookkeeping, and
the model-response retry/validation logic. No real Ollama calls -- those are
monkeypatched, same convention as the rest of this pipeline leaving the
I/O-heavy main loop itself untested.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "corpus_prep"))
import reocr_adjudicate as adjudicate  # noqa: E402


def _write_staging(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_corpus_page(corpus_root: Path, relpath: str, page: int, body: str) -> None:
    p = corpus_root / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"## Page {page}\n\n{body}\n", encoding="utf-8")


class TestLoadStagedPages:
    def test_parses_records_in_order(self, tmp_path):
        staging = tmp_path / "staging.jsonl"
        _write_staging(staging, [
            {"pdf": "a.pdf", "page": 1, "files": ["a.md"], "new_text": "new1", "timestamp": "t1"},
            {"pdf": "a.pdf", "page": 2, "files": ["a.md"], "new_text": "new2", "timestamp": "t2"},
        ])
        pages = adjudicate.load_staged_pages(staging)
        assert [p.page for p in pages] == [1, 2]
        assert pages[0].files == ("a.md",)
        assert pages[0].new_text == "new1"

    def test_skips_blank_lines(self, tmp_path):
        staging = tmp_path / "staging.jsonl"
        staging.write_text(
            json.dumps({"pdf": "a.pdf", "page": 1, "files": ["a.md"], "new_text": "x", "timestamp": "t"})
            + "\n\n",
            encoding="utf-8",
        )
        assert len(adjudicate.load_staged_pages(staging)) == 1


class TestLoadFullPageText:
    def test_plain_unchunked_page(self, tmp_path):
        _write_corpus_page(tmp_path, "2567/ครั้งที่ 9/เอกสาร ก.md", 2, "เนื้อหาหน้า 2")
        assert adjudicate.load_full_page_text(tmp_path, "2567/ครั้งที่ 9/เอกสาร ก.md", 2) == "เนื้อหาหน้า 2"

    def test_reassembles_a_page_split_ns_own_sub_chunks(self, tmp_path):
        # split_pages() breaks a page whose body exceeds PAGE_CHAR_BUDGET
        # (6000 chars) into "Page N.1", "Page N.2", ... -- there is no plain
        # "Page N" label to look up once that happens, so load_full_page_text
        # has to recognise and rejoin them. A single oversized "## Page 2"
        # body is what actually triggers this in the real pipeline (not a
        # literal "## Page 2.1" header, which split_pages' regex wouldn't
        # even recognise as a page marker).
        oversized_body = "ก" * 7000
        _write_corpus_page(tmp_path, "2567/ครั้งที่ 9/เอกสาร ก.md", 2, oversized_body)
        assert adjudicate.load_full_page_text(tmp_path, "2567/ครั้งที่ 9/เอกสาร ก.md", 2) == oversized_body

    def test_missing_file_returns_none(self, tmp_path):
        assert adjudicate.load_full_page_text(tmp_path, "2567/ครั้งที่ 9/ไม่มีไฟล์.md", 1) is None

    def test_missing_page_in_existing_file_returns_none(self, tmp_path):
        _write_corpus_page(tmp_path, "2567/ครั้งที่ 9/เอกสาร ก.md", 1, "หน้า 1")
        assert adjudicate.load_full_page_text(tmp_path, "2567/ครั้งที่ 9/เอกสาร ก.md", 5) is None


class TestResolveOldText:
    def test_returns_the_first_files_text_when_no_divergence(self, tmp_path):
        _write_corpus_page(tmp_path, "2567/ครั้งที่ 9/เอกสาร ข__1.md", 2, "เนื้อหาเดียวกัน")
        _write_corpus_page(tmp_path, "2567/ครั้งที่ 9/เอกสาร ข__2.md", 2, "เนื้อหาเดียวกัน")
        staged = adjudicate.StagedPage(
            pdf="x.pdf", page=2,
            files=("2567/ครั้งที่ 9/เอกสาร ข__1.md", "2567/ครั้งที่ 9/เอกสาร ข__2.md"),
            new_text="new", timestamp="t",
        )
        old_text, diverging = adjudicate.resolve_old_text(tmp_path, staged)
        assert old_text == "เนื้อหาเดียวกัน"
        assert diverging == []

    def test_flags_a_sibling_whose_text_diverges(self, tmp_path):
        _write_corpus_page(tmp_path, "2567/ครั้งที่ 9/เอกสาร ข__1.md", 2, "ต้นฉบับ")
        _write_corpus_page(tmp_path, "2567/ครั้งที่ 9/เอกสาร ข__2.md", 2, "ข้อความอื่น")
        staged = adjudicate.StagedPage(
            pdf="x.pdf", page=2,
            files=("2567/ครั้งที่ 9/เอกสาร ข__1.md", "2567/ครั้งที่ 9/เอกสาร ข__2.md"),
            new_text="new", timestamp="t",
        )
        old_text, diverging = adjudicate.resolve_old_text(tmp_path, staged)
        assert old_text == "ต้นฉบับ"
        assert diverging == ["2567/ครั้งที่ 9/เอกสาร ข__2.md"]

    def test_missing_page_in_corpus_returns_none(self, tmp_path):
        (tmp_path / "2567" / "ครั้งที่ 9").mkdir(parents=True)
        staged = adjudicate.StagedPage(
            pdf="x.pdf", page=5, files=("2567/ครั้งที่ 9/ไม่มีไฟล์.md",),
            new_text="new", timestamp="t",
        )
        old_text, diverging = adjudicate.resolve_old_text(tmp_path, staged)
        assert old_text is None
        assert diverging == []


class TestLoadDoneKeys:
    def test_no_file_yet_returns_empty_set(self, tmp_path):
        assert adjudicate.load_done_keys(tmp_path / "missing.jsonl") == set()

    def test_returns_pdf_page_keys_already_recorded(self, tmp_path):
        adj = tmp_path / "adjudication.jsonl"
        _write_staging(adj, [
            {"pdf": "a.pdf", "page": 1, "files": ["a.md"], "old_text_source": "a.md",
             "diverging_siblings": [], "verdicts": {}, "timestamp": "t"},
        ])
        assert adjudicate.load_done_keys(adj) == {("a.pdf", 1)}


class TestCallCompareModel:
    def test_valid_response_on_first_attempt(self, monkeypatch):
        def fake_chat(**kwargs):
            return {"message": {"content": json.dumps({"verdict": "new", "reason": "clearer"})}}

        monkeypatch.setattr(adjudicate.ollama, "chat", fake_chat)
        result = adjudicate.call_compare_model("phi4:latest", "old text", "new text")
        assert result["verdict"] == "new"
        assert result["reason"] == "clearer"
        assert result["error"] is None

    def test_invalid_verdict_retries_then_succeeds(self, monkeypatch):
        calls = []

        def fake_chat(**kwargs):
            calls.append(kwargs["options"]["temperature"])
            if len(calls) == 1:
                return {"message": {"content": json.dumps({"verdict": "maybe", "reason": "?"})}}
            return {"message": {"content": json.dumps({"verdict": "both_ok", "reason": "same"})}}

        monkeypatch.setattr(adjudicate.ollama, "chat", fake_chat)
        result = adjudicate.call_compare_model("phi4:latest", "old", "new", retries=2)
        assert result["verdict"] == "both_ok"
        assert calls == [0.0, 0.3]

    def test_exhausted_retries_returns_none_verdict_with_error(self, monkeypatch):
        def fake_chat(**kwargs):
            return {"message": {"content": "not json"}}

        monkeypatch.setattr(adjudicate.ollama, "chat", fake_chat)
        result = adjudicate.call_compare_model("phi4:latest", "old", "new", retries=2)
        assert result["verdict"] is None
        assert result["error"]


class TestAppendResult:
    def test_round_trips_through_load_done_keys(self, tmp_path):
        adj = tmp_path / "adjudication.jsonl"
        result = adjudicate.AdjudicationResult(
            pdf="a.pdf", page=3, files=("a.md",), old_text_source="a.md",
            diverging_siblings=(), verdicts={"phi4:latest": {"verdict": "new", "reason": "x",
                                                              "error": None, "elapsed_s": 1.0}},
            timestamp="t",
        )
        adjudicate.append_result(adj, result)
        assert adjudicate.load_done_keys(adj) == {("a.pdf", 3)}
