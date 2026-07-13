"""Tickets 1-2 (tickets.md) -- Streamlit-driven smoke tests for review_app.py.

Follows the AppTest pattern already established by
tests/test_streamlit_build_run.py: drive the real widgets, not the logic
module directly, against a small fixture corpus + fixture
consensus_priority.md + fixture decision log (never the real corpus/files).
Session state for `corpus_root` / `consensus_file` / `decisions_file` is
seeded *before* the first `.run()` so the app never touches the real paths,
even transiently.
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
    / "review_app.py"
)

_ONE_FILE_CONSENSUS_MD = (
    "# Consensus flags -- both models agree (priority review list)\n\n"
    "Total files with at least one consensus page: 1\n"
    "Total consensus pages across the corpus: 2\n\n"
    "## [2567] 2567\\ครั้งที่ 9\\เอกสาร ก.md  (2 consensus page(s))\n"
    "### Page 1\n"
    "- **[phi4:latest]** table looks garbled\n"
    "  > 1 | 2\n"
    "### Page 3\n"
    "- **[gemma4:e4b]** garbled words\n"
)

_TWO_FILE_CONSENSUS_MD = (
    "## [2567] 2567\\ครั้งที่ 9\\เอกสาร ก.md  (1 consensus page(s))\n"
    "### Page 1\n"
    "- **[phi4:latest]** garbled\n"
    "- **[gemma4:e4b]** garbled\n\n"
    "## [2567] 2567\\ครั้งที่ 9\\เอกสาร ข.md  (1 consensus page(s))\n"
    "### Page 1\n"
    "- **[phi4:latest]** garbled\n"
    "- **[gemma4:e4b]** garbled\n"
)


def _write_fixture(tmp_path, docs: dict[str, str], consensus_md: str) -> tuple[Path, Path]:
    corpus_root = tmp_path / "academic_resolutions"
    doc_dir = corpus_root / "2567" / "ครั้งที่ 9"
    doc_dir.mkdir(parents=True)
    for name, content in docs.items():
        (doc_dir / name).write_text(content, encoding="utf-8")

    consensus_file = tmp_path / "consensus_priority.md"
    consensus_file.write_text(consensus_md, encoding="utf-8")
    return corpus_root, consensus_file


def _run_with_fixture(tmp_path, docs: dict[str, str], consensus_md: str) -> tuple[AppTest, Path]:
    corpus_root, consensus_file = _write_fixture(tmp_path, docs, consensus_md)
    decisions_file = tmp_path / "review_decisions.jsonl"

    at = AppTest.from_file(_PAGE)
    # Seed session state *before* the first run so the app reads the fixture
    # paths from the very first script execution -- it never touches the
    # real corpus_root / consensus_priority.md / review_decisions.jsonl
    # defaults, even transiently.
    at.session_state["corpus_root"] = str(corpus_root)
    at.session_state["consensus_file"] = str(consensus_file)
    at.session_state["decisions_file"] = str(decisions_file)
    at.run(timeout=30)
    return at, decisions_file


def test_review_app_renders_the_first_file_and_page_content(tmp_path):
    docs = {
        "เอกสาร ก.md": "## Page 1\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Page 3\nข้อความปกติ",
    }
    at, _ = _run_with_fixture(tmp_path, docs, _ONE_FILE_CONSENSUS_MD)

    assert not at.exception
    assert any("เอกสาร ก.md" in el.value for el in at.subheader)
    all_markdown = " ".join(el.value for el in at.markdown)
    assert "table looks garbled" in all_markdown
    assert "| a | b |" in all_markdown


def test_review_app_next_button_advances_to_next_file(tmp_path):
    docs = {
        "เอกสาร ก.md": "## Page 1\nข้อความ ก",
        "เอกสาร ข.md": "## Page 1\nข้อความ ข",
    }
    at, _ = _run_with_fixture(tmp_path, docs, _TWO_FILE_CONSENSUS_MD)

    assert any("เอกสาร ก.md" in el.value for el in at.subheader)

    at.button(key="next_button").click().run(timeout=30)

    assert not at.exception
    assert any("เอกสาร ข.md" in el.value for el in at.subheader)


def test_review_app_badges_split_piece_with_no_consensus_sibling(tmp_path):
    """A file matching the __N split-piece convention should be badged even
    when no sibling piece is also consensus-flagged (the badge trigger is
    "this file is a split piece," not "a sibling happens to be flagged
    too") -- see tools/corpus_prep/consensus_review/SPEC.md user story 5."""
    docs = {"เอกสาร ข__1.md": "## Page 1\nข้อความ"}
    consensus_md = (
        "## [2567] 2567\\ครั้งที่ 9\\เอกสาร ข__1.md  (1 consensus page(s))\n"
        "### Page 1\n"
        "- **[phi4:latest]** garbled\n"
        "- **[gemma4:e4b]** garbled\n"
    )
    at, _ = _run_with_fixture(tmp_path, docs, consensus_md)

    assert not at.exception
    assert any("ตัดเป็นหลายชิ้น" in el.value for el in at.warning)


# --- Ticket 2 (tickets.md): verdict recording ----------------------------


def test_review_app_verdict_button_writes_decision_and_updates_progress(tmp_path):
    docs = {
        "เอกสาร ก.md": "## Page 1\nข้อความ ก",
        "เอกสาร ข.md": "## Page 1\nข้อความ ข",
    }
    at, decisions_file = _run_with_fixture(tmp_path, docs, _TWO_FILE_CONSENSUS_MD)

    assert any("ตัดสินแล้ว 0/2" in el.value for el in at.sidebar.caption)
    assert any("เอกสาร ก.md" in el.value for el in at.subheader)

    at.button(key="verdict_reocr_2567\\ครั้งที่ 9\\เอกสาร ก.md").click().run(timeout=30)

    assert not at.exception
    lines = decisions_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["file"] == "2567\\ครั้งที่ 9\\เอกสาร ก.md"
    assert record["verdict"] == "ควร re-OCR"

    # The decided file drops out of the default (undecided-only) view, so the
    # remaining file is now shown automatically, and progress moved to 1/2.
    assert any("ตัดสินแล้ว 1/2" in el.value for el in at.sidebar.caption)
    assert any("เอกสาร ข.md" in el.value for el in at.subheader)


def test_review_app_show_all_toggle_reveals_decided_files_and_allows_revision(tmp_path):
    docs = {
        "เอกสาร ก.md": "## Page 1\nข้อความ ก",
        "เอกสาร ข.md": "## Page 1\nข้อความ ข",
    }
    at, decisions_file = _run_with_fixture(tmp_path, docs, _TWO_FILE_CONSENSUS_MD)

    at.button(key="verdict_unsure_2567\\ครั้งที่ 9\\เอกสาร ก.md").click().run(timeout=30)
    assert any("เอกสาร ข.md" in el.value for el in at.subheader)

    at.sidebar.checkbox(key="show_all").set_value(True).run(timeout=30)

    # "All files" view includes the decided one again, in original order.
    assert any("เอกสาร ก.md" in el.value for el in at.subheader)
    assert any("ตัดสินไปแล้ว: ไม่แน่ใจ" in el.value for el in at.info)

    # Changing the verdict appends a new record rather than rewriting the
    # old one -- the log keeps both lines, and the resolved state is the
    # latest.
    at.button(key="verdict_reocr_2567\\ครั้งที่ 9\\เอกสาร ก.md").click().run(timeout=30)

    lines = decisions_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["verdict"] == "ไม่แน่ใจ"
    assert json.loads(lines[1])["verdict"] == "ควร re-OCR"
    assert any("ตัดสินไปแล้ว: ควร re-OCR" in el.value for el in at.info)
