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
from rag_lab.factory import build_embedder, build_reranker, build_retriever
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.pipeline import retrieve
from rag_lab.results import save_retrieval_result
from rag_lab.retrievers.filters import EntityFilter, MetadataFilter
from rag_lab.router import (
    ROUTE_COMBO,
    ROUTE_UNMATCHED,
    RouteTarget,
    classify_query,
    detect_entities,
    rrf_merge,
)
from rag_lab.schema import RetrievalResult

ENTITY_TAGS_LOADER = "entity_tags"


def check_entity_tags_loader(manifest: dict, index_dir: str | Path) -> None:
    """Loud guard for entity_lookup: metadata['people']/['programs']/
    ['courses'] only exist on chunks from an index built with
    loaders.entity_loader.EntityTagLoader -- a missing key is
    indistinguishable from a genuinely empty match (see EntityFilter), so
    pointing entity_lookup at the wrong index must fail loudly, not
    silently return nothing."""
    loader_type = manifest["combo"]["loader"]["type"]
    if loader_type != ENTITY_TAGS_LOADER:
        raise LookupError(
            f"{index_dir} was built with loader {loader_type!r}, not "
            f"{ENTITY_TAGS_LOADER!r} -- entity_lookup needs "
            "metadata['people']/['programs']/['courses'] on every chunk."
        )


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
    filter_criteria: dict | None = None,
    reranker_spec: StrategySpec | None = None,
    rerank_pool_size: int | None = None,
    entity_boost: bool = False,
) -> list[ComboRetrieval]:
    store = ArtifactStore()
    retriever = build_retriever(retriever_spec)
    reranker = build_reranker(reranker_spec) if reranker_spec is not None else None
    detected = detect_entities(query) if entity_boost else {}

    out: list[ComboRetrieval] = []
    for index_dir in index_dirs:
        manifest = _read_manifest(index_dir)
        embedder = build_embedder(
            StrategySpec.model_validate(manifest["combo"]["embedder"])
        )
        index = store.load(index_dir)
        if filter_criteria:
            index = MetadataFilter(filter_criteria).apply(index)
        # query_indices compares potentially-heterogeneous combos side by
        # side, so an index not built with entity_tags must degrade
        # gracefully (skip narrowing, keep comparing) rather than hard-fail
        # the whole comparison the way entity_lookup does for a single,
        # deliberately-chosen index.
        applied_boost = bool(detected) and manifest["combo"]["loader"]["type"] == ENTITY_TAGS_LOADER
        if applied_boost:
            index = EntityFilter(detected).apply(index)

        combination_id = f"{manifest['combo_id']}__{retriever.name}"
        if reranker is not None:
            combination_id = f"{combination_id}__{reranker.name}"
        if applied_boost:
            combination_id = f"{combination_id}__entity_boost"
        result: RetrievalResult = retrieve(
            query, index, embedder, retriever, k,
            reranker=reranker, rerank_pool_size=rerank_pool_size,
            combination_id=combination_id,
        )
        if results_dir is not None:
            save_retrieval_result(result, results_dir)
        out.append(
            ComboRetrieval(
                combo_id=manifest["combo_id"], index_dir=str(index_dir), result=result
            )
        )
    return out


def entity_lookup(
    query: str,
    index_dirs: list[str],
    results_dir: str | Path | None = None,
) -> list[ComboRetrieval]:
    """Exhaustive entity-lookup mode: returns every matching Resolution for
    a query naming a known person/program/course, bypassing top-k ranking
    entirely. A separate top-level function from query_indices/route_query
    (not an extension of either) -- route_query picks one pre-designated
    index and ranks it; query_indices compares heterogeneous combos and
    degrades gracefully on an untagged index; this always runs against
    caller-specified entity_tags-loader index dirs and hard-fails loudly if
    one isn't (see check_entity_tags_loader)."""
    store = ArtifactStore()
    retriever = build_retriever(StrategySpec(type="entity_lookup"))

    out: list[ComboRetrieval] = []
    for index_dir in index_dirs:
        manifest = _read_manifest(index_dir)
        check_entity_tags_loader(manifest, index_dir)
        embedder = build_embedder(
            StrategySpec.model_validate(manifest["combo"]["embedder"])
        )
        index = store.load(index_dir)
        combination_id = f"{manifest['combo_id']}__entity_lookup"
        result: RetrievalResult = retrieve(
            query, index, embedder, retriever, k=0, combination_id=combination_id,
        )
        if results_dir is not None:
            save_retrieval_result(result, results_dir)
        out.append(
            ComboRetrieval(
                combo_id=manifest["combo_id"], index_dir=str(index_dir), result=result
            )
        )
    return out


def resolve_index(target: RouteTarget, indices: list[IndexInfo]) -> IndexInfo:
    """The one built IndexInfo matching a RouteTarget's chunker/embedder
    identity. Raises if none (or more than one, an ambiguous build) match --
    a route silently falling back to the wrong index is worse than a loud
    error at query time."""
    matches = [
        i for i in indices
        if i.chunker.type == target.chunker_type
        and i.embedder.type == target.embedder_type
        and (
            target.embedder_model_name is None
            or i.embedder.params.get("model_name") == target.embedder_model_name
        )
    ]
    if not matches:
        raise LookupError(f"no built index matches route target {target!r}")
    if len(matches) > 1:
        raise LookupError(f"route target {target!r} matches {len(matches)} built indices, expected 1")
    return matches[0]


def route_query(
    query: str,
    indices: list[IndexInfo],
    retriever_spec: StrategySpec,
    k: int,
    results_dir: str | Path | None = None,
    unmatched_strategy: str = "default",
    route_combo: dict[str, RouteTarget] = ROUTE_COMBO,
) -> RetrievalResult:
    """Classify `query` (router.classify_query) and retrieve against the
    matching route's index only -- rather than every built combo like
    query_indices, which is for side-by-side comparison, not routing.

    unmatched_strategy controls what happens when the query doesn't match
    the person or program pattern:
    - "default" (recommended default; see tools/eval/routing_eval.py):
      query just the unmatched route's index. Offline validation against the
      Gold set found this statistically indistinguishable from RRF-merging
      on recall@10 (t=0.59), so the extra cost of querying 3 indices isn't
      justified unless top-of-list ranking quality specifically matters.
    - "rrf": query the unmatched, person, and program routes' indices and
      combine with Reciprocal Rank Fusion (router.rrf_merge). The same
      validation found this improves MRR (+15%) and ndcg@10 (+12%) -- worth
      it for a UI that only surfaces the top few results.
    """
    route = classify_query(query)

    if route == ROUTE_UNMATCHED and unmatched_strategy == "rrf":
        targets = [route_combo[ROUTE_UNMATCHED], route_combo["person"], route_combo["program"]]
        chosen = [resolve_index(t, indices) for t in targets]
        retrievals = query_indices(query, [i.dir for i in chosen], retriever_spec, k)
        merged = rrf_merge(
            [cr.result for cr in retrievals], top_k=k, combination_id=f"routed__rrf__{route}",
        )
        if results_dir is not None:
            save_retrieval_result(merged, results_dir)
        return merged

    chosen = resolve_index(route_combo[route], indices)
    [retrieval] = query_indices(query, [chosen.dir], retriever_spec, k, results_dir=results_dir)
    return retrieval.result
