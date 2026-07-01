"""Cycles 6–7 — invariants shared by all chunkers + YAML selectability."""
from __future__ import annotations

from rag_lab.config import ExperimentConfig, StrategySpec
from rag_lab.factory import build_chunker
from rag_lab.runner import run_experiment
from rag_lab.schema import Resolution


def test_resolution_id_is_stable_across_chunkers():
    res = Resolution(
        resolution_id="r1",
        source_path="r1.md",
        raw_text="## Page 1\nประโยคหนึ่งนะครับ ประโยคสองนะครับ\n\n## Page 2\nอีกหน้าหนึ่งครับ",
    )
    for chunker_type in ("fixed_size", "recursive", "sentence"):
        chunks = build_chunker(
            StrategySpec(type=chunker_type, params={"chunk_size": 60})
        ).chunk(res)
        assert chunks, chunker_type
        assert all(c.resolution_id == "r1" for c in chunks), chunker_type


def test_recursive_and_sentence_are_selectable_via_yaml(tmp_path):
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "a.md").write_text(
        "## Page 1\nที่ประชุมมีมติอนุมัติ วันนี้อากาศดีมาก", encoding="utf-8"
    )
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[
            StrategySpec(type="recursive", params={"chunk_size": 100}),
            StrategySpec(type="sentence", params={"chunk_size": 100}),
        ],
        embedders=[StrategySpec(type="hashing")],
    )
    results = run_experiment(config)
    assert len(results) == 2
    assert all(r.status == "ok" for r in results)
