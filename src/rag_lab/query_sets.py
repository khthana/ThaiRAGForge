"""Silver and Gold query sets (CONTEXT.md): Query -> relevant resolution_ids.

Silver is free (derived from the corpus itself) but "easy" — title wording
overlaps the document. Gold is hand-written, harder, and truer to real use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_lab.config import StrategySpec
from rag_lab.factory import build_embedder, build_retriever
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.pipeline import retrieve as pipeline_retrieve
from rag_lab.results import save_retrieval_result
from rag_lab.schema import Resolution


@dataclass
class QuerySetEntry:
    query: str
    relevant_resolution_ids: list[str]


def build_silver_query_set(resolutions: list[Resolution]) -> list[QuerySetEntry]:
    return [
        QuerySetEntry(query=r.title, relevant_resolution_ids=[r.resolution_id])
        for r in resolutions
        if r.title
    ]


def load_gold_query_set(path: str | Path) -> list[QuerySetEntry]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        QuerySetEntry(
            query=entry["query"],
            relevant_resolution_ids=entry["relevant_resolution_ids"],
        )
        for entry in data
    ]


def run_query_set(
    query_set: list[QuerySetEntry],
    index_dirs: list[str],
    retriever_spec: StrategySpec,
    k: int,
    results_dir: str | Path,
) -> None:
    """Retrieve every query in the set against every index and persist a
    RetrievalResult per (query, index) pair, so eval can score them later
    without re-indexing (results.py is the shared writer for #4 and here).

    Loads each index's embedder/Index/retriever once and loops queries against
    the already-loaded objects -- unlike query_service.query_indices (built for
    Mode B's one-query-at-a-time UI use), reloading a ~500MB Index and rebuilding
    the embedder on every query is fine for one interactive query but makes a
    query-set run (hundreds+ queries) take hours instead of minutes."""
    store = ArtifactStore()
    retriever = build_retriever(retriever_spec)

    for index_dir in index_dirs:
        manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
        embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
        index = store.load(index_dir)
        combination_id = f"{manifest['combo_id']}__{retriever.name}"

        for entry in query_set:
            result = pipeline_retrieve(
                entry.query, index, embedder, retriever, k, combination_id=combination_id
            )
            save_retrieval_result(result, results_dir)
