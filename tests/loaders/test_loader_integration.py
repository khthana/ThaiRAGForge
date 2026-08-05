"""Cycles 6–7 — loaders selectable via YAML; their metadata reaches chunks."""
from __future__ import annotations

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.config import ExperimentConfig, StrategySpec
from rag_lab.embedders import HashingEmbedder
from rag_lab.factory import build_loader
from rag_lab.pipeline import build_index
from rag_lab.retrievers.filters import MetadataFilter
from rag_lab.runner import run_experiment


def _corpus(tmp_path):
    d = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    d.mkdir(parents=True)
    doc = d / "เรื่อง แต่งตั้ง.md"
    doc.write_text(
        "# Document: x.pdf\n\n## Page 1\nที่ประชุมมีมติแต่งตั้ง "
        "รองศาสตราจารย์ ดร.คมสัน มาลีสี",
        encoding="utf-8",
    )
    (d / "เรื่อง แต่งตั้ง_LINK.txt").write_text("https://drive.google.com/x", encoding="utf-8")
    return tmp_path / "corpus", doc


def test_metadata_and_ner_loaders_selectable_via_yaml(tmp_path):
    corpus, _ = _corpus(tmp_path)
    config = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": corpus.as_posix()},
        output_dir=(tmp_path / "out").as_posix(),
        loaders=[StrategySpec(type="metadata"), StrategySpec(type="ner")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 200})],
        embedders=[StrategySpec(type="hashing")],
    )
    results = run_experiment(config)
    assert len(results) == 2  # loader axis: metadata + ner
    assert all(r.status == "ok" for r in results)


def test_metadata_loader_source_url_reaches_chunks_and_filter(tmp_path):
    _, doc = _corpus(tmp_path)
    res = build_loader(StrategySpec(type="metadata")).load(str(doc))
    index = build_index([res], FixedSizeChunker(chunk_size=300), HashingEmbedder())

    assert all(c.metadata.get("source_url") == "https://drive.google.com/x" for c in index.chunks)
    assert MetadataFilter({"year": "2569"}).apply(index).chunks
