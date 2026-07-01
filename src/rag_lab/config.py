"""Experiment configuration — YAML is the source of truth (grill decision).

A run is fully described by an ExperimentConfig; the UI (later) only edits/launches
these, and every run snapshots the resolved config into its manifest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class StrategySpec(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class CorpusSpec(BaseModel):
    input_dir: str
    subset: Literal["dev", "full"] = "dev"
    limit: int | None = None  # dev-subset size; None = all


class ExperimentConfig(BaseModel):
    experiment_name: str
    corpus: CorpusSpec
    output_dir: str
    loaders: list[StrategySpec]
    chunkers: list[StrategySpec]
    embedders: list[StrategySpec]
    retrievers: list[StrategySpec] = Field(
        default_factory=lambda: [StrategySpec(type="dense")]
    )
    run_mode: Literal["cartesian", "paired"] = "cartesian"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
