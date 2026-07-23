"""Cycle 2 — ExperimentConfig.from_yaml parses YAML and applies defaults."""
from __future__ import annotations

import yaml

from rag_lab.config import ExperimentConfig, StrategySpec


def test_from_yaml_parses_and_applies_defaults(tmp_path):
    text = """
experiment_name: exp1
corpus:
  input_dir: /data
output_dir: /out
loaders:
  - type: plain
chunkers:
  - type: fixed_size
    params: {chunk_size: 256}
embedders:
  - type: hashing
"""
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")

    cfg = ExperimentConfig.from_yaml(path)

    assert cfg.experiment_name == "exp1"
    assert cfg.chunkers[0].params["chunk_size"] == 256
    # defaults
    assert cfg.run_mode == "cartesian"
    assert cfg.seed == 42
    assert cfg.corpus.subset == "dev"
    assert [r.type for r in cfg.retrievers] == ["dense"]
    assert cfg.rerankers == []


def test_to_yaml_round_trips_through_from_yaml(tmp_path):
    cfg = ExperimentConfig(
        experiment_name="ทดสอบ",
        corpus={"input_dir": "academic_resolutions/2569", "subset": "full"},
        output_dir="data/index/ทดสอบ",
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 256})],
        embedders=[StrategySpec(type="hashing")],
    )
    path = tmp_path / "roundtrip.yaml"

    cfg.to_yaml(path)
    loaded = ExperimentConfig.from_yaml(path)

    assert loaded == cfg


def test_to_yaml_string_round_trips_and_includes_chunker_params():
    cfg = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": "x"},
        output_dir="out",
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 256})],
        embedders=[StrategySpec(type="hashing")],
    )

    text = cfg.to_yaml_string()

    assert "chunk_size: 256" in text
    assert ExperimentConfig.model_validate(yaml.safe_load(text)) == cfg


def test_rerankers_round_trip_through_yaml(tmp_path):
    cfg = ExperimentConfig(
        experiment_name="e",
        corpus={"input_dir": "x"},
        output_dir="out",
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size")],
        embedders=[StrategySpec(type="hashing")],
        rerankers=[StrategySpec(type="cross_encoder", params={"model_name": "m"})],
    )
    path = tmp_path / "rerankers.yaml"

    cfg.to_yaml(path)
    loaded = ExperimentConfig.from_yaml(path)

    assert loaded == cfg
    assert loaded.rerankers[0].params["model_name"] == "m"


def test_to_yaml_writes_thai_text_literally_not_as_unicode_escapes(tmp_path):
    cfg = ExperimentConfig(
        experiment_name="ทดสอบ",
        corpus={"input_dir": "academic_resolutions/2569"},
        output_dir="data/index/ทดสอบ",
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size")],
        embedders=[StrategySpec(type="hashing")],
    )
    path = tmp_path / "thai.yaml"

    cfg.to_yaml(path)

    text = path.read_text(encoding="utf-8")
    assert "ทดสอบ" in text
    assert "\\u" not in text
