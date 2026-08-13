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
    route_targets,
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

    # A retriever serving from an external store (Qdrant) reads no Index rows,
    # so two things follow from one fact. It does not need `embeddings.npy`
    # (~234MB per collection -- skipping it is the point of an engine-served
    # path), and a row-level narrowing CANNOT reach it: filtering the
    # in-process Index would leave the engine returning the whole collection,
    # i.e. an unfiltered answer presented as a filtered one. Refuse loudly
    # instead. `Query.filters` is the engine-side route for that, already
    # supported for `resolution_id_in` and not yet plumbed through this call.
    reads_rows = getattr(retriever, "reads_index_rows", True)
    if not reads_rows and (filter_criteria or entity_boost):
        raise ValueError(
            f"retriever {retriever.name!r} serves from an external store and cannot be "
            f"narrowed by an in-process filter/entity boost -- the narrowing would be "
            f"silently ignored. Use a row-reading retriever, or express the constraint "
            f"as a Query.filters kind the retriever supports."
        )

    out: list[ComboRetrieval] = []
    for index_dir in index_dirs:
        manifest = _read_manifest(index_dir)
        embedder = build_embedder(
            StrategySpec.model_validate(manifest["combo"]["embedder"])
        )
        index = store.load(index_dir, with_embeddings=reads_rows)
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
    route_combo: dict[str, RouteTarget] | None = None,
) -> RetrievalResult:
    """Classify `query` (router.classify_query) and retrieve against the
    matching route's index only -- rather than every built combo like
    query_indices, which is for side-by-side comparison, not routing.

    `route_combo` defaults to the map measured for `retriever_spec`'s type
    (router.route_targets). It has to depend on the retriever: the best combo
    per route is retriever-dependent -- person peaks at semantic+qwen3 under
    dense but sentence+bge_m3 under hybrid -- and until 2026-08-08 a single
    flat dict served whichever retriever it happened to be picked under. Pass
    `route_combo` explicitly to override. See data/results/routing_eval.md §2.

    unmatched_strategy controls what happens when the query matches no route
    pattern at all. Since the 2026-08-08 course/faculty routes, that is
    **0/106 queries** on the Gold set, so neither branch below is exercised
    by any current eval and both are unmeasured:
    - "default" (recommended): query just the unmatched route's index.
    - "rrf": fan out to the unmatched, person and program routes and combine
      with Reciprocal Rank Fusion (router.rrf_merge). Note the target list is
      hardcoded and was never extended to course/faculty.
    The earlier numbers quoted here ("indistinguishable on recall@10, t=0.59;
    RRF +15% MRR") came from the retired 252-query/3-route routing_eval and
    no longer have a script that reproduces them -- do not cite them.
    """
    route = classify_query(query)
    if route_combo is None:
        route_combo = route_targets(retriever_spec.type)

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
