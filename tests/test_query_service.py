"""Mode B core (Streamlit-free): query built indices and compare side-by-side."""
from __future__ import annotations

from pathlib import Path

from rag_lab.config import ExperimentConfig, StrategySpec
from rag_lab.query_service import discover_indices, query_indices
from rag_lab.results import load_retrieval_result
from rag_lab.runner import run_experiment


def _build_two_indices(tmp_path):
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "เรื่อง ค่าธรรมเนียม.md").write_text(
        "## Page 1\nค่าธรรมเนียม การศึกษา ภาคเรียน", encoding="utf-8"
    )
    (corpus / "เรื่อง หลักสูตร.md").write_text(
        "## Page 1\nหลักสูตร วิศวกรรม คอมพิวเตอร์", encoding="utf-8"
    )
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[
            StrategySpec(type="fixed_size", params={"chunk_size": 100}),
            StrategySpec(type="fixed_size", params={"chunk_size": 50}),
        ],
        embedders=[StrategySpec(type="hashing")],
    )
    run_experiment(config)
    return out


def test_query_indices_returns_ranked_results_per_combo(tmp_path):
    out = _build_two_indices(tmp_path)
    dirs = [info.dir for info in discover_indices(out)]
    assert len(dirs) == 2

    combos = query_indices("ค่าธรรมเนียม", dirs, StrategySpec(type="dense"), k=3)

    assert len(combos) == 2
    for cr in combos:
        top = cr.result.results[0]
        assert top.resolution_id and top.page is not None
        assert "ค่าธรรมเนียม" in top.resolution_id  # relevant resolution ranks first


def test_combination_id_is_build_combo_id_plus_retriever(tmp_path):
    # the join key #9 needs: persisted combination_id must equal the build combo_id
    out = _build_two_indices(tmp_path)
    infos = discover_indices(out)
    combos = query_indices("x", [i.dir for i in infos], StrategySpec(type="dense"), k=2)
    for info, cr in zip(infos, combos):
        assert cr.result.combination_id == f"{info.combo_id}__dense"


def test_query_persists_retrieval_result_and_round_trips(tmp_path):
    out = _build_two_indices(tmp_path)
    dirs = [i.dir for i in discover_indices(out)]
    results_dir = tmp_path / "results"

    query_indices("ค่าธรรมเนียม", dirs, StrategySpec(type="dense"), k=2, results_dir=results_dir)

    files = sorted(Path(results_dir).glob("*.json"))
    assert len(files) == 2
    loaded = load_retrieval_result(files[0])
    assert loaded.query == "ค่าธรรมเนียม"
    assert loaded.combination_id.endswith("__dense")
    assert len(loaded.results) >= 1


def test_discover_indices_skips_dirs_without_manifest(tmp_path):
    out = _build_two_indices(tmp_path)
    junk = Path(out) / "junk"
    junk.mkdir()
    (junk / "chunks.parquet").write_text("x", encoding="utf-8")

    infos = discover_indices(out)
    assert len(infos) == 2
    assert all(Path(i.dir).name != "junk" for i in infos)


def test_query_uses_each_index_own_embedder(tmp_path):
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "a.md").write_text("## Page 1\nalpha beta gamma", encoding="utf-8")
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 100})],
        embedders=[
            StrategySpec(type="hashing", params={"dim": 128}),
            StrategySpec(type="hashing", params={"dim": 256}),
        ],
    )
    run_experiment(config)
    dirs = [i.dir for i in discover_indices(out)]
    assert len(dirs) == 2

    # each index has a different embedding dim; if the query weren't embedded with
    # the matching per-index embedder, the dot product would raise a shape error
    combos = query_indices("alpha", dirs, StrategySpec(type="dense"), k=1)
    assert len(combos) == 2
    assert all(len(cr.result.results) >= 1 for cr in combos)
