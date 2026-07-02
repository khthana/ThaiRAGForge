"""Batch runner: build + cache one Index artifact per combination.

Reproducible (fixed seed, manifest per combo) and error-isolated (a failing
combo is recorded and the batch continues). Build combos are loader × chunker ×
embedder; the retriever axis is query-time and not built here (ADR-0001).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
    timings: dict[str, float] = field(default_factory=dict)
    chunk_size_stats: dict[str, float] = field(default_factory=dict)


def _discover_paths(config: ExperimentConfig) -> list[str]:
    paths = sorted(Path(config.corpus.input_dir).rglob("*.md"))
    if config.corpus.subset == "dev" and config.corpus.limit:
        paths = paths[: config.corpus.limit]
    return [str(p) for p in paths]


def run_experiment(
    config: ExperimentConfig,
    on_combo_done: Callable[[ComboResult], None] | None = None,
) -> list[ComboResult]:
    random.seed(config.seed)
    np.random.seed(config.seed)

    paths = _discover_paths(config)
    store = ArtifactStore()
    out = Path(config.output_dir)
    results: list[ComboResult] = []

    for combo in tqdm(enumerate_build_combos(config), desc="combos"):
        try:
            loader = build_loader(combo.loader)
            t0 = time.perf_counter()
            resolutions = [loader.load(p) for p in paths]
            load_seconds = time.perf_counter() - t0
            chunker = build_chunker(combo.chunker)
            embedder = build_embedder(combo.embedder)

            index = build_index(resolutions, chunker, embedder)

            combo_dir = out / combo.id
            store.save(index, combo_dir)
            write_manifest(combo_dir, build_manifest(config, combo, resolutions))
            timings = {"load_seconds": load_seconds, **index.meta.get("timings", {})}
            lengths = [len(c.text) for c in index.chunks]
            chunk_size_stats = (
                {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)}
                if lengths
                else {}
            )
            result = ComboResult(
                combo.id,
                "ok",
                n_chunks=len(index.chunks),
                timings=timings,
                chunk_size_stats=chunk_size_stats,
            )
        except Exception as exc:  # error isolation — one bad combo must not kill the batch
            result = ComboResult(combo.id, "error", error=str(exc))
        results.append(result)
        if on_combo_done is not None:
            on_combo_done(result)
    return results
