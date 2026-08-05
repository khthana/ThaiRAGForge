"""Streamlit-driven smoke tests for pages/1_reocr_diff_review.py (the
old-vs-new Phase 3 decision UI). Follows the same AppTest pattern as
test_consensus_review_app.py: drive the real widgets against small fixture
JSONL + fixture corpus, never the real files.
"""
from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

_PAGE = str(
    Path(__file__).resolve().parents[1]
    / "tools"
    / "corpus_prep"
    / "consensus_review"
    / "pages"
    / "1_reocr_diff_review.py"
)


def _adjudication_line(pdf, page, phi, gemma, files=None, old_text_source=None, diverging=None) -> str:
    files = files or [f"2567\\ครั้งที่ 9\\{Path(pdf).stem}.md"]
    record = {
        "pdf": pdf,
        "page": page,
        "files": files,
        "old_text_source": old_text_source or files[0],
        "diverging_siblings": diverging or [],
        "verdicts": {
            "phi4:latest": {"verdict": phi, "reason": f"phi says {phi}", "error": None, "elapsed_s": 1.0},
            "gemma4:e4b": {"verdict": gemma, "reason": f"gemma says {gemma}", "error": None, "elapsed_s": 1.0},
        },
        "timestamp": "t",
    }
    return json.dumps(record, ensure_ascii=False)


def _staging_line(pdf, page, new_text, files=None) -> str:
    files = files or [f"2567\\ครั้งที่ 9\\{Path(pdf).stem}.md"]
    return json.dumps(
        {"pdf": pdf, "page": page, "files": files, "new_text": new_text, "timestamp": "t"},
        ensure_ascii=False,
    )


def _run_with_fixture(tmp_path, adjudication_lines, staging_lines, docs: dict[str, str]):
    corpus_root = tmp_path / "academic_resolutions"
    doc_dir = corpus_root / "2567" / "ครั้งที่ 9"
    doc_dir.mkdir(parents=True)
    for name, content in docs.items():
        (doc_dir / name).write_text(content, encoding="utf-8")

    adjudication_file = tmp_path / "reocr_adjudication.jsonl"
    adjudication_file.write_text("\n".join(adjudication_lines) + "\n", encoding="utf-8")
    staging_file = tmp_path / "reocr_pages_staging.jsonl"
    staging_file.write_text("\n".join(staging_lines) + "\n", encoding="utf-8")
    decisions_file = tmp_path / "reocr_review_decisions.jsonl"

    at = AppTest.from_file(_PAGE)
    at.session_state["reocr_corpus_root"] = str(corpus_root)
    at.session_state["reocr_adjudication_file"] = str(adjudication_file)
    at.session_state["reocr_staging_file"] = str(staging_file)
    at.session_state["reocr_decisions_file"] = str(decisions_file)
    at.run(timeout=30)
    return at, decisions_file


def test_page_excludes_clean_new_new_agreement_from_the_review_queue(tmp_path):
    lines = [
        _adjudication_line("a.pdf", 1, "new", "new"),
        _adjudication_line("b.pdf", 1, "old", "new"),
    ]
    staging = [
        _staging_line("a.pdf", 1, "new text a"),
        _staging_line("b.pdf", 1, "new text b"),
    ]
    docs = {"a.md": "## Page 1\nold text a", "b.md": "## Page 1\nold text b"}
    at, _ = _run_with_fixture(tmp_path, lines, staging, docs)

    assert not at.exception
    assert any("b.pdf" in el.value for el in at.subheader)
    assert not any("a.pdf" in el.value for el in at.subheader)
    assert any("ต้องรีวิว 1 หน้า" in el.value for el in at.sidebar.caption)


def test_page_shows_old_and_new_text_side_by_side(tmp_path):
    lines = [_adjudication_line("a.pdf", 1, "old", "new")]
    staging = [_staging_line("a.pdf", 1, "ข้อความใหม่จาก re-OCR")]
    docs = {"a.md": "## Page 1\nข้อความเดิมในคอร์ปัส"}
    at, _ = _run_with_fixture(tmp_path, lines, staging, docs)

    all_markdown = " ".join(el.value for el in at.markdown)
    assert "ข้อความเดิมในคอร์ปัส" in all_markdown
    assert "ข้อความใหม่จาก re-OCR" in all_markdown


def test_verdict_button_writes_decision_and_advances(tmp_path):
    lines = [
        _adjudication_line("a.pdf", 1, "old", "new"),
        _adjudication_line("b.pdf", 1, "both_bad", "both_bad"),
    ]
    staging = [
        _staging_line("a.pdf", 1, "new a"),
        _staging_line("b.pdf", 1, "new b"),
    ]
    docs = {"a.md": "## Page 1\nold a", "b.md": "## Page 1\nold b"}
    at, decisions_file = _run_with_fixture(tmp_path, lines, staging, docs)

    assert any("a.pdf" in el.value for el in at.subheader)
    at.button(key="reocr_verdict_apply_new_a.pdf_1").click().run(timeout=30)

    assert not at.exception
    lines_written = decisions_file.read_text(encoding="utf-8").splitlines()
    assert len(lines_written) == 1
    record = json.loads(lines_written[0])
    assert record["pdf"] == "a.pdf"
    assert record["page"] == 1
    assert record["verdict"] == "ใช้ข้อความใหม่ (new)"

    # Decided page drops out of the default (undecided-only) view.
    assert any("b.pdf" in el.value for el in at.subheader)
    assert any("ตัดสินแล้ว 1/2" in el.value for el in at.sidebar.caption)


def test_defer_verdict_available_for_both_bad(tmp_path):
    """The both_bad/both_bad case the user explicitly said to defer -- make
    sure the defer verdict button actually exists and works, not just
    apply/keep."""
    lines = [_adjudication_line("a.pdf", 1, "both_bad", "both_bad")]
    staging = [_staging_line("a.pdf", 1, "new a")]
    docs = {"a.md": "## Page 1\nold a"}
    at, decisions_file = _run_with_fixture(tmp_path, lines, staging, docs)

    at.button(key="reocr_verdict_defer_a.pdf_1").click().run(timeout=30)

    assert not at.exception
    record = json.loads(decisions_file.read_text(encoding="utf-8").splitlines()[0])
    assert record["verdict"] == "รอไว้ก่อน (ทั้งคู่ยังพัง/ไม่แน่ใจ)"


def test_all_pages_clean_shows_success_and_stops(tmp_path):
    lines = [_adjudication_line("a.pdf", 1, "new", "new")]
    staging = [_staging_line("a.pdf", 1, "new a")]
    docs = {"a.md": "## Page 1\nold a"}
    at, _ = _run_with_fixture(tmp_path, lines, staging, docs)

    assert not at.exception
    assert any("ไม่มีหน้าที่ต้องรีวิว" in el.value for el in at.success)
