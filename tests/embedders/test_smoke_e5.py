"""Cycle 14 — heavy smoke: a real multilingual-e5-large build+query, round-tripped
through the manifest (query_service), not a hand-constructed embedder passed
straight to pipeline.retrieve(). That distinction matters: query_service
rebuilds the embedder from the persisted {type, params} StrategySpec via
factory.build_embedder, so this is the only test proving the query: prefix
survives that reconstruction (a bug here would silently embed queries as
passages — no crash, just degraded ranking).

Skipped by default (downloads/loads a ~2.2 GB model). Enable with:

    RAG_LAB_SMOKE=1 pytest tests/test_smoke_e5.py
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_LAB_SMOKE") != "1",
    reason="set RAG_LAB_SMOKE=1 to run the heavy e5-large smoke test",
)


def test_real_e5_via_query_service_ranks_relevant_resolution_first(tmp_path):
    from rag_lab.config import ExperimentConfig, StrategySpec
    from rag_lab.query_service import discover_indices, query_indices
    from rag_lab.runner import run_experiment

    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "ค่าธรรมเนียม.md").write_text(
        "## Page 1\nเรื่อง การลดค่าธรรมเนียมการศึกษาให้นักศึกษาในช่วงการระบาดของโควิด-19",
        encoding="utf-8",
    )
    (corpus / "หลักสูตร.md").write_text(
        "## Page 1\nเรื่อง การปรับปรุงหลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิศวกรรมคอมพิวเตอร์",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    config = ExperimentConfig(
        experiment_name="e5-smoke",
        corpus={"input_dir": (tmp_path / "corpus").as_posix()},
        output_dir=out.as_posix(),
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 256})],
        embedders=[StrategySpec(type="e5")],
    )

    results = run_experiment(config)
    assert all(r.status == "ok" for r in results), results

    dirs = [i.dir for i in discover_indices(out)]
    combos = query_indices("ขอลดค่าเทอมช่วงโควิด", dirs, StrategySpec(type="dense"), k=2)

    assert "ค่าธรรมเนียม" in combos[0].result.results[0].resolution_id
