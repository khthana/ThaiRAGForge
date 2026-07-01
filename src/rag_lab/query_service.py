"""Mode B core: query one or more built Index artifacts and compare results.

Streamlit-free so it is unit-testable; the Streamlit app is a thin shell over
this. The query is embedded with the *same* embedder that built each index
(reconstructed from that index's manifest), because cross-embedder scores are
not comparable (ADR-0001).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_lab.config import StrategySpec
from rag_lab.factory import build_embedder, build_retriever
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.pipeline import retrieve
from rag_lab.results import save_retrieval_result
from rag_lab.schema import RetrievalResult


@dataclass
class IndexInfo:
    combo_id: str
    dir: str
    loader: StrategySpec
    chunker: StrategySpec
    embedder: StrategySpec


@dataclass
class ComboRetrieval:
    combo_id: str
    index_dir: str
    result: RetrievalResult


def _read_manifest(index_dir: str | Path) -> dict:
    return json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))


def discover_indices(output_dir: str | Path) -> list[IndexInfo]:
    """List built indices under output_dir. Only directories with a manifest.json
    are queryable (the manifest is what lets us reconstruct the embedder)."""
    infos: list[IndexInfo] = []
    for d in sorted(Path(output_dir).iterdir()):
        manifest_path = d / "manifest.json"
        if not (d.is_dir() and manifest_path.exists()):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        combo = manifest["combo"]
        infos.append(
            IndexInfo(
                combo_id=manifest["combo_id"],
                dir=str(d),
                loader=StrategySpec.model_validate(combo["loader"]),
                chunker=StrategySpec.model_validate(combo["chunker"]),
                embedder=StrategySpec.model_validate(combo["embedder"]),
            )
        )
    return infos


def query_indices(
    query: str,
    index_dirs: list[str],
    retriever_spec: StrategySpec,
    k: int,
    results_dir: str | Path | None = None,
) -> list[ComboRetrieval]:
    store = ArtifactStore()
    retriever = build_retriever(retriever_spec)

    out: list[ComboRetrieval] = []
    for index_dir in index_dirs:
        manifest = _read_manifest(index_dir)
        embedder = build_embedder(
            StrategySpec.model_validate(manifest["combo"]["embedder"])
        )
        index = store.load(index_dir)

        combination_id = f"{manifest['combo_id']}__{retriever.name}"
        result: RetrievalResult = retrieve(
            query, index, embedder, retriever, k, combination_id=combination_id
        )
        if results_dir is not None:
            save_retrieval_result(result, results_dir)
        out.append(
            ComboRetrieval(
                combo_id=manifest["combo_id"], index_dir=str(index_dir), result=result
            )
        )
    return out
