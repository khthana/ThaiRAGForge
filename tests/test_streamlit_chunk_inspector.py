"""Headless smoke test (streamlit.testing.v1.AppTest) for the chunk-inspector
page: builds two tiny real combos (different chunkers, same corpus) via
rag_lab.runner directly (not through any UI), points the page at that
output_dir, and drives the actual tab widgets."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from rag_lab.config import CorpusSpec, ExperimentConfig, StrategySpec
from rag_lab.runner import run_experiment

_PAGE = str(Path(__file__).resolve().parents[1] / "app" / "pages" / "2_chunk_inspector.py")


def _write_corpus(tmp_path):
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "เรื่อง ค่าธรรมเนียม.md").write_text(
        "## Page 1\n" + ("ค่าธรรมเนียม การศึกษา ภาคเรียน ที่หนึ่ง. " * 40),
        encoding="utf-8",
    )
    return tmp_path / "corpus"


def _build_two_chunkers(tmp_path) -> Path:
    corpus = _write_corpus(tmp_path)
    output_dir = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="chunk_inspector_smoke",
        corpus=CorpusSpec(input_dir=str(corpus), subset="dev", limit=5),
        output_dir=str(output_dir),
        loaders=[StrategySpec(type="plain")],
        chunkers=[
            StrategySpec(type="fixed_size", params={"chunk_size": 150, "chunk_overlap": 0}),
            StrategySpec(type="recursive", params={"chunk_size": 150, "chunk_overlap": 0}),
        ],
        embedders=[StrategySpec(type="hashing")],
    )
    run_experiment(config)
    return output_dir


def test_chunk_inspector_discovers_combos_and_renders_both_tabs(tmp_path):
    output_dir = _build_two_chunkers(tmp_path)

    at = AppTest.from_file(_PAGE)
    at.run(timeout=30)
    assert not at.exception

    at.sidebar.text_input(key="base_dir").set_value(str(output_dir))
    at.run(timeout=30)
    assert not at.exception

    caption_text = " ".join(c.value for c in at.caption)
    assert "fixed_size" in caption_text
    assert "recursive" in caption_text


def test_compare_tab_shows_chunks_for_a_picked_resolution(tmp_path):
    output_dir = _build_two_chunkers(tmp_path)

    at = AppTest.from_file(_PAGE)
    at.run(timeout=30)
    at.sidebar.text_input(key="base_dir").set_value(str(output_dir))
    at.run(timeout=30)
    assert not at.exception

    assert len(at.selectbox(key="doc_pick").options) >= 1
    at.selectbox(key="doc_pick").select_index(0).run(timeout=30)
    assert not at.exception
    # Each chunker's column renders a chunk-count caption ("N chunks").
    assert any("chunks" in c.value for c in at.caption)


def test_oversized_tab_lists_and_shows_full_text_of_a_large_chunk(tmp_path):
    output_dir = _build_two_chunkers(tmp_path)

    at = AppTest.from_file(_PAGE)
    at.run(timeout=30)
    at.sidebar.text_input(key="base_dir").set_value(str(output_dir))
    at.run(timeout=30)

    at.selectbox(key="oversized_chunker").select("fixed_size").run(timeout=30)
    assert not at.exception
    at.number_input(key="oversized_threshold").set_value(100).run(timeout=30)
    assert not at.exception

    assert len(at.selectbox(key="oversized_pick").options) >= 1
    at.selectbox(key="oversized_pick").select_index(0).run(timeout=30)
    assert not at.exception
    assert len(at.text_area(key="oversized_text").value) > 50
