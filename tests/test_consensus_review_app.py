"""Ticket 1 (tickets.md) -- Streamlit-driven smoke test for review_app.py.

Follows the AppTest pattern already established by
tests/test_streamlit_build_run.py: drive the real widgets, not the logic
module directly, against a small fixture corpus + fixture
consensus_priority.md (never the real corpus).
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

_PAGE = str(
    Path(__file__).resolve().parents[1]
    / "tools"
    / "corpus_prep"
    / "consensus_review"
    / "review_app.py"
)


def _write_fixture(tmp_path) -> tuple[Path, Path]:
    corpus_root = tmp_path / "academic_resolutions"
    doc_dir = corpus_root / "2567" / "ครั้งที่ 9"
    doc_dir.mkdir(parents=True)
    (doc_dir / "เอกสาร ก.md").write_text(
        "## Page 1\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Page 3\nข้อความปกติ",
        encoding="utf-8",
    )

    consensus_file = tmp_path / "consensus_priority.md"
    consensus_file.write_text(
        "# Consensus flags -- both models agree (priority review list)\n\n"
        "Total files with at least one consensus page: 1\n"
        "Total consensus pages across the corpus: 2\n\n"
        "## [2567] 2567\\ครั้งที่ 9\\เอกสาร ก.md  (2 consensus page(s))\n"
        "### Page 1\n"
        "- **[phi4:latest]** table looks garbled\n"
        "  > 1 | 2\n"
        "### Page 3\n"
        "- **[gemma4:e4b]** garbled words\n",
        encoding="utf-8",
    )
    return corpus_root, consensus_file


def _run_with_fixture(tmp_path):
    corpus_root, consensus_file = _write_fixture(tmp_path)
    at = AppTest.from_file(_PAGE)
    at.run(timeout=30)
    at.sidebar.text_input(key="corpus_root").set_value(str(corpus_root))
    at.sidebar.text_input(key="consensus_file").set_value(str(consensus_file))
    at.run(timeout=30)
    return at


def test_review_app_renders_the_first_file_and_page_content(tmp_path):
    at = _run_with_fixture(tmp_path)

    assert not at.exception
    assert any("เอกสาร ก.md" in el.value for el in at.subheader)
    all_markdown = " ".join(el.value for el in at.markdown)
    assert "table looks garbled" in all_markdown
    assert "| a | b |" in all_markdown


def test_review_app_next_button_advances_to_next_file(tmp_path):
    corpus_root = tmp_path / "academic_resolutions"
    doc_dir = corpus_root / "2567" / "ครั้งที่ 9"
    doc_dir.mkdir(parents=True)
    (doc_dir / "เอกสาร ก.md").write_text("## Page 1\nข้อความ ก", encoding="utf-8")
    (doc_dir / "เอกสาร ข.md").write_text("## Page 1\nข้อความ ข", encoding="utf-8")

    consensus_file = tmp_path / "consensus_priority.md"
    consensus_file.write_text(
        "## [2567] 2567\\ครั้งที่ 9\\เอกสาร ก.md  (1 consensus page(s))\n"
        "### Page 1\n"
        "- **[phi4:latest]** garbled\n"
        "- **[gemma4:e4b]** garbled\n\n"
        "## [2567] 2567\\ครั้งที่ 9\\เอกสาร ข.md  (1 consensus page(s))\n"
        "### Page 1\n"
        "- **[phi4:latest]** garbled\n"
        "- **[gemma4:e4b]** garbled\n",
        encoding="utf-8",
    )

    at = AppTest.from_file(_PAGE)
    at.run(timeout=30)
    at.sidebar.text_input(key="corpus_root").set_value(str(corpus_root))
    at.sidebar.text_input(key="consensus_file").set_value(str(consensus_file))
    at.run(timeout=30)

    assert any("เอกสาร ก.md" in el.value for el in at.subheader)

    at.button(key="next_button").click().run(timeout=30)

    assert not at.exception
    assert any("เอกสาร ข.md" in el.value for el in at.subheader)
