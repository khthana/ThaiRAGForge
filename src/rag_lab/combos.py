"""Enumerate the index-build combinations for a run.

Build combos are loader × chunker × embedder only — the retriever is query-time
(ADR-0001), not part of building an Index artifact.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass

from rag_lab.config import ExperimentConfig, StrategySpec


@dataclass
class BuildCombo:
    loader: StrategySpec
    chunker: StrategySpec
    embedder: StrategySpec

    @property
    def id(self) -> str:
        payload = json.dumps(
            {
                "loader": self.loader.model_dump(),
                "chunker": self.chunker.model_dump(),
                "embedder": self.embedder.model_dump(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        short = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
        return f"{self.loader.type}__{self.chunker.type}__{self.embedder.type}__{short}"


def enumerate_build_combos(config: ExperimentConfig) -> list[BuildCombo]:
    if config.run_mode == "cartesian":
        return [
            BuildCombo(loader, chunker, embedder)
            for loader, chunker, embedder in itertools.product(
                config.loaders, config.chunkers, config.embedders
            )
        ]
    if config.run_mode == "paired":
        lengths = {len(config.loaders), len(config.chunkers), len(config.embedders)}
        if len(lengths) != 1:
            raise ValueError(
                "paired run_mode requires loaders, chunkers and embedders "
                "of equal length"
            )
        return [
            BuildCombo(loader, chunker, embedder)
            for loader, chunker, embedder in zip(
                config.loaders, config.chunkers, config.embedders
            )
        ]
    raise ValueError(f"unknown run_mode: {config.run_mode!r}")
