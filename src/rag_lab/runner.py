"""Batch runner: build + cache one Index artifact per combination.

Reproducible (fixed seed, manifest per combo) and error-isolated (a failing
combo is recorded and the batch continues). Build combos are loader × chunker ×
embedder; the retriever axis is query-time and not built here (ADR-0001).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag_lab.combos import enumerate_build_combos
from rag_lab.config import ExperimentConfig
from rag_lab.factory import build_chunker, build_embedder, build_loader
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.manifest import build_manifest, write_manifest
from rag_lab.pipeline import build_index


@dataclass
class ComboResult:
    combo_id: str
    status: str  # "ok" | "error"
    n_chunks: int | None = None
    error: str | None = None


def _discover_paths(config: ExperimentConfig) -> list[str]:
    paths = sorted(Path(config.corpus.input_dir).rglob("*.md"))
    if config.corpus.subset == "dev" and config.corpus.limit:
        paths = paths[: config.corpus.limit]
    return [str(p) for p in paths]


def run_experiment(config: ExperimentConfig) -> list[ComboResult]:
    random.seed(config.seed)
    np.random.seed(config.seed)

    paths = _discover_paths(config)
    store = ArtifactStore()
    out = Path(config.output_dir)
    results: list[ComboResult] = []

    for combo in tqdm(enumerate_build_combos(config), desc="combos"):
        try:
            loader = build_loader(combo.loader)
            resolutions = [loader.load(p) for p in paths]
            chunker = build_chunker(combo.chunker)
            embedder = build_embedder(combo.embedder)

            index = build_index(resolutions, chunker, embedder)

            combo_dir = out / combo.id
            store.save(index, combo_dir)
            write_manifest(combo_dir, build_manifest(config, combo, resolutions))
            results.append(ComboResult(combo.id, "ok", n_chunks=len(index.chunks)))
        except Exception as exc:  # error isolation — one bad combo must not kill the batch
            results.append(ComboResult(combo.id, "error", error=str(exc)))
    return results
