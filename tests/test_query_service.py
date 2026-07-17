"""Mode B core (Streamlit-free): query built indices and compare side-by-side."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag_lab.config import ExperimentConfig, StrategySpec
from rag_lab.query_service import discover_indices, query_indices, resolve_index, route_query
from rag_lab.results import load_retrieval_result
from rag_lab.router import RouteTarget
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


def test_query_indices_with_bm25_and_hybrid(tmp_path):
    out = _build_two_indices(tmp_path)
    dirs = [i.dir for i in discover_indices(out)]

    for retriever in ("bm25", "hybrid"):
        combos = query_indices("ค่าธรรมเนียม", dirs, StrategySpec(type=retriever), k=3)
        assert len(combos) == 2
        for cr in combos:
            assert "ค่าธรรมเนียม" in cr.result.results[0].resolution_id, retriever


def test_query_indices_metadata_filter_narrows_by_year(tmp_path):
    for year, name in [("2569", "a"), ("2568", "b")]:
        d = tmp_path / "corpus" / year / "ครั้งที่ 1"
        d.mkdir(parents=True)
        (d / f"{name}.md").write_text("## Page 1\nค่าธรรมเนียม การศึกษา", encoding="utf-8")
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 200})],
        embedders=[StrategySpec(type="hashing")],
    )
    run_experiment(config)
    dirs = [i.dir for i in discover_indices(out)]

    combos = query_indices(
        "ค่าธรรมเนียม", dirs, StrategySpec(type="dense"), k=10,
        filter_criteria={"year": "2569"},
    )
    for cr in combos:
        assert cr.result.results  # something survived the filter
        assert all(r.resolution_id.startswith("2569/") for r in cr.result.results)


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


def _build_routing_indices(tmp_path):
    """Two cheap (no real-model-download) combos distinguishable by chunker
    type only, standing in for the real person/program/unmatched routes so
    router-orchestration logic can be tested without loading e5/bge-m3/
    ConGen-PhayaThaiBERT."""
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "เรื่อง ค่าธรรมเนียม.md").write_text(
        "## Page 1\nค่าธรรมเนียม การศึกษา ภาคเรียน", encoding="utf-8"
    )
    (corpus / "เรื่อง หลักสูตร.md").write_text(
        "## Page 1\nหลักสูตร วิศวกรรม คอมพิวเตอร์ สาขาวิชา", encoding="utf-8"
    )
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[
            StrategySpec(type="fixed_size", params={"chunk_size": 100}),
            StrategySpec(type="recursive", params={"chunk_size": 100}),
        ],
        embedders=[StrategySpec(type="hashing")],
    )
    run_experiment(config)
    return discover_indices(out)


def test_resolve_index_finds_the_one_matching_combo(tmp_path):
    indices = _build_routing_indices(tmp_path)
    target = RouteTarget(chunker_type="recursive", embedder_type="hashing")
    found = resolve_index(target, indices)
    assert found.chunker.type == "recursive"
    assert found.embedder.type == "hashing"


def test_resolve_index_raises_when_no_index_matches():
    with pytest.raises(LookupError):
        resolve_index(RouteTarget(chunker_type="semantic", embedder_type="local"), [])


def test_route_query_person_query_goes_to_person_route(tmp_path):
    indices = _build_routing_indices(tmp_path)
    route_combo = {
        "person": RouteTarget(chunker_type="recursive", embedder_type="hashing"),
        "program": RouteTarget(chunker_type="fixed_size", embedder_type="hashing"),
        "unmatched": RouteTarget(chunker_type="fixed_size", embedder_type="hashing"),
    }
    result = route_query(
        "ผศ.ดร.สมชาย ใจดี มีประวัติเป็นกรรมการหลักสูตรใดบ้าง",
        indices, StrategySpec(type="dense"), k=3, route_combo=route_combo,
    )
    recursive_combo_id = next(i.combo_id for i in indices if i.chunker.type == "recursive")
    assert result.combination_id == f"{recursive_combo_id}__dense"


def test_route_query_unmatched_default_strategy_queries_one_index(tmp_path):
    indices = _build_routing_indices(tmp_path)
    route_combo = {
        "person": RouteTarget(chunker_type="recursive", embedder_type="hashing"),
        "program": RouteTarget(chunker_type="fixed_size", embedder_type="hashing"),
        "unmatched": RouteTarget(chunker_type="fixed_size", embedder_type="hashing"),
    }
    result = route_query(
        "ในการประชุมครั้งนี้ มีการพิจารณาเรื่องค่าธรรมเนียมการศึกษาในกรณีใดบ้าง",
        indices, StrategySpec(type="dense"), k=3,
        unmatched_strategy="default", route_combo=route_combo,
    )
    fixed_combo_id = next(i.combo_id for i in indices if i.chunker.type == "fixed_size")
    assert result.combination_id == f"{fixed_combo_id}__dense"


def test_route_query_unmatched_rrf_strategy_merges_multiple_indices(tmp_path):
    indices = _build_routing_indices(tmp_path)
    route_combo = {
        "person": RouteTarget(chunker_type="recursive", embedder_type="hashing"),
        "program": RouteTarget(chunker_type="fixed_size", embedder_type="hashing"),
        "unmatched": RouteTarget(chunker_type="fixed_size", embedder_type="hashing"),
    }
    result = route_query(
        "ในการประชุมครั้งนี้ มีการพิจารณาเรื่องค่าธรรมเนียมการศึกษาในกรณีใดบ้าง",
        indices, StrategySpec(type="dense"), k=3,
        unmatched_strategy="rrf", route_combo=route_combo,
    )
    assert result.combination_id == "routed__rrf__unmatched"
    assert result.results  # merged ranking is non-empty
