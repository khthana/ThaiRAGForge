"""Cycle 4 — NERLoader stores Thai named entities in metadata['entities']."""
from __future__ import annotations

from rag_lab.config import StrategySpec
from rag_lab.factory import build_loader


def test_ner_loader_extracts_entities(tmp_path):
    d = tmp_path / "2569" / "ครั้งที่ 1"
    d.mkdir(parents=True)
    doc = d / "a.md"
    doc.write_text(
        "## Page 1\nรองศาสตราจารย์ ดร.คมสัน มาลีสี อธิการบดี "
        "สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง",
        encoding="utf-8",
    )

    res = build_loader(StrategySpec(type="ner")).load(str(doc))

    entities = res.metadata["entities"]
    assert len(entities) >= 1
    assert all("text" in e and "tag" in e for e in entities)
    tags = {e["tag"] for e in entities}
    assert "PERSON" in tags or "ORGANIZATION" in tags
