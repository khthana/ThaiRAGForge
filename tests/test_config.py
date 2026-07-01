"""Cycle 2 — ExperimentConfig.from_yaml parses YAML and applies defaults."""
from __future__ import annotations

from rag_lab.config import ExperimentConfig


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
