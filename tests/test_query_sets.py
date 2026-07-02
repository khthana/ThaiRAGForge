"""Cycle 16 — Silver/Gold query sets (CONTEXT.md): Silver is auto-generated from
each Resolution's title (relevant to itself); Gold is hand-written and loaded
from a file.
"""
from __future__ import annotations

from pathlib import Path

from rag_lab.config import ExperimentConfig, StrategySpec
from rag_lab.loaders import PlainLoader
from rag_lab.metrics import evaluate
from rag_lab.query_sets import (
    QuerySetEntry,
    build_silver_query_set,
    load_gold_query_set,
    run_query_set,
)
from rag_lab.query_service import discover_indices
from rag_lab.results import load_retrieval_result
from rag_lab.runner import run_experiment
from rag_lab.schema import Resolution


def test_silver_query_set_uses_each_resolutions_title_as_a_query_for_itself():
    resolutions = [
        Resolution(
            resolution_id="2569/1/fee",
            source_path="fee.md",
            raw_text="...",
            title="การลดค่าธรรมเนียมการศึกษา",
        ),
    ]

    entries = build_silver_query_set(resolutions)

    assert len(entries) == 1
    assert entries[0].query == "การลดค่าธรรมเนียมการศึกษา"
    assert entries[0].relevant_resolution_ids == ["2569/1/fee"]


def test_silver_query_set_skips_resolutions_without_a_title():
    resolutions = [
        Resolution(resolution_id="untitled", source_path="x.md", raw_text="...", title=None),
    ]

    assert build_silver_query_set(resolutions) == []


def test_load_gold_query_set_from_yaml_file(tmp_path):
    path = tmp_path / "gold.yaml"
    path.write_text(
        """
- query: ขอลดค่าเทอมช่วงโควิด
  relevant_resolution_ids: ["2569/1/fee"]
- query: ปรับหลักสูตรวิศวะคอม
  relevant_resolution_ids: ["2569/1/curriculum", "2568/2/curriculum-amendment"]
""",
        encoding="utf-8",
    )

    entries = load_gold_query_set(path)

    assert len(entries) == 2
    assert entries[0].query == "ขอลดค่าเทอมช่วงโควิด"
    assert entries[0].relevant_resolution_ids == ["2569/1/fee"]
    assert entries[1].relevant_resolution_ids == [
        "2569/1/curriculum",
        "2568/2/curriculum-amendment",
    ]


def test_run_query_set_persists_one_retrieval_result_per_query_per_index(tmp_path):
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "เรื่อง ค่าธรรมเนียม.md").write_text(
        "## Page 1\nค่าธรรมเนียม การศึกษา ภาคเรียน", encoding="utf-8"
    )
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 100})],
        embedders=[StrategySpec(type="hashing")],
    )
    run_experiment(config)
    index_dirs = [i.dir for i in discover_indices(out)]

    query_set = [
        QuerySetEntry(query="ค่าธรรมเนียม", relevant_resolution_ids=["ignored"]),
        QuerySetEntry(query="หลักสูตร", relevant_resolution_ids=["ignored"]),
    ]
    results_dir = tmp_path / "results"

    run_query_set(
        query_set, index_dirs, StrategySpec(type="dense"), k=3, results_dir=results_dir
    )

    files = sorted(Path(results_dir).glob("*.json"))
    assert len(files) == 2  # 2 queries x 1 index dir


def test_silver_set_run_through_evaluate_scores_a_real_hit(tmp_path):
    """End-to-end: real retriever output (real `rank` values, real
    combination_id/query join key) persisted by run_query_set and re-loaded
    from disk, scored by evaluate(). Every other test here uses synthetic
    RetrievalResults, so this is the only one that would catch a mismatch
    between what retrievers actually persist and what evaluate() assumes."""
    corpus_dir = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "เรื่อง ค่าธรรมเนียม.md").write_text(
        "## Page 1\nเรื่อง ค่าธรรมเนียม การศึกษา ภาคเรียน", encoding="utf-8"
    )
    (corpus_dir / "เรื่อง หลักสูตร.md").write_text(
        "## Page 1\nเรื่อง หลักสูตร วิศวกรรม คอมพิวเตอร์", encoding="utf-8"
    )
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 100})],
        embedders=[StrategySpec(type="hashing")],
    )
    run_experiment(config)
    index_dirs = [i.dir for i in discover_indices(out)]

    resolutions = [
        PlainLoader().load(str(p)) for p in sorted(corpus_dir.rglob("*.md"))
    ]
    query_set = build_silver_query_set(resolutions)
    results_dir = tmp_path / "results"

    run_query_set(
        query_set, index_dirs, StrategySpec(type="dense"), k=3, results_dir=results_dir
    )

    persisted = [load_retrieval_result(p) for p in Path(results_dir).glob("*.json")]
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    scores = evaluate(persisted, qrels, k=3)

    assert len(scores) == 1  # one build combination
    combo_scores = next(iter(scores.values()))
    assert combo_scores["recall@3"] == 1.0
    assert combo_scores["mrr"] == 1.0
