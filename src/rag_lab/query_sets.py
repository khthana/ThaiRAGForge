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
from rag_lab.factory import build_embedder, build_reranker, build_retriever
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.pipeline import retrieve as pipeline_retrieve
from rag_lab.query_service import check_entity_tags_loader
from rag_lab.results import save_retrieval_result
from rag_lab.retrievers.filters import EntityFilter
from rag_lab.router import detect_entities
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
    reranker_spec: StrategySpec | None = None,
    rerank_pool_size: int | None = None,
    entity_boost: bool = False,
) -> None:
    """Retrieve every query in the set against every index and persist a
    RetrievalResult per (query, index) pair, so eval can score them later
    without re-indexing (results.py is the shared writer for #4 and here).

    Loads each index's embedder/Index/retriever once and loops queries against
    the already-loaded objects -- unlike query_service.query_indices (built for
    Mode B's one-query-at-a-time UI use), reloading a ~500MB Index and rebuilding
    the embedder on every query is fine for one interactive query but makes a
    query-set run (hundreds+ queries) take hours instead of minutes. A reranker,
    if given, is built once here too and applied per query inside pipeline.retrieve.

    entity_boost mirrors query_indices' keyword/filter-boost: entities are
    detected per query (detection depends on entry.query, not the index), and
    an index not built with the entity_tags loader degrades gracefully --
    narrowing is skipped for that index rather than failing the whole run."""
    store = ArtifactStore()
    retriever = build_retriever(retriever_spec)
    reranker = build_reranker(reranker_spec) if reranker_spec is not None else None

    for index_dir in index_dirs:
        manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
        embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
        index = store.load(index_dir)
        is_entity_tagged = manifest["combo"]["loader"]["type"] == "entity_tags"
        combination_id = f"{manifest['combo_id']}__{retriever.name}"
        if reranker is not None:
            combination_id = f"{combination_id}__{reranker.name}"

        for entry in query_set:
            query_index = index
            query_combination_id = combination_id
            if entity_boost and is_entity_tagged:
                detected = detect_entities(entry.query)
                if detected:
                    query_index = EntityFilter(detected).apply(index)
                    query_combination_id = f"{combination_id}__entity_boost"
            result = pipeline_retrieve(
                entry.query, query_index, embedder, retriever, k,
                reranker=reranker, rerank_pool_size=rerank_pool_size,
                combination_id=query_combination_id,
            )
            save_retrieval_result(result, results_dir)


def run_entity_lookup_query_set(
    query_set: list[QuerySetEntry],
    index_dirs: list[str],
    results_dir: str | Path,
) -> None:
    """Exhaustive-lookup analogue of run_query_set: retrieves every query
    against every index using EntityLookupRetriever, persisting one
    RetrievalResult per (query, index) pair. Hard-fails (via
    check_entity_tags_loader) if any index_dir wasn't built with the
    entity_tags loader, rather than silently producing empty results.

    Scoring caveat: entity_lookup's ranks are arbitrary corpus order, not a
    relevance ranking. Call metrics.evaluate(..., k=<a value >= the largest
    per-query result-set size>) so recall@k/precision@k reduce to plain set
    recall/precision; mrr/ndcg remain computable but aren't meaningful here."""
    store = ArtifactStore()
    retriever = build_retriever(StrategySpec(type="entity_lookup"))

    for index_dir in index_dirs:
        manifest = json.loads((Path(index_dir) / "manifest.json").read_text(encoding="utf-8"))
        check_entity_tags_loader(manifest, index_dir)
        embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
        index = store.load(index_dir)
        combination_id = f"{manifest['combo_id']}__entity_lookup"

        for entry in query_set:
            result = pipeline_retrieve(
                entry.query, index, embedder, retriever, k=0, combination_id=combination_id,
            )
            save_retrieval_result(result, results_dir)
