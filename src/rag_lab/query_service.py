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
from rag_lab.factory import (
    build_embedder_cached,
    build_reranker,
    build_retriever,
    build_retriever_cached,
)
from rag_lab.io.index_cache import load_index_cached
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

#: Retrievers that score a lexical arm in process, i.e. the ones for which the
#: `BM25Okapi` memoised on the Index is worth ~1.0s of warm-up. `qdrant_hybrid`
#: is deliberately absent: its sparse arm is scored by the engine, from weights
#: precomputed at ingest.
_LEXICAL_RETRIEVERS = {"bm25", "hybrid"}


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
    # Cached: this is the third construction a served query pays for, and it
    # was the largest remaining one on the engine path -- 327 ms of a 433 ms
    # query spent rebuilding a Qdrant client and re-parsing a 78k-term
    # vocabulary the previous query had already parsed
    # (`data/results/serving_concurrency.md` section 4).
    retriever = build_retriever_cached(retriever_spec)
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
        # Cached: the serving path hits the same two models over and over, and
        # a fresh construction reloads ~8.9 s of weights (78% of a served
        # query). Bounded at 2 -- see factory.build_embedder_cached for why
        # the eval path deliberately does NOT share this.
        embedder = build_embedder_cached(
            StrategySpec.model_validate(manifest["combo"]["embedder"])
        )
        # Cached: re-reading ~234MB of embeddings.npy and rebuilding 57k Chunk
        # objects costs ~1,159 ms, and throwing the Index away also throws away
        # the BM25Okapi memoised on it (~921 ms more). The cache re-stats the
        # artifacts on every hit, so a rebuilt index is never served from RAM.
        index = load_index_cached(index_dir, with_embeddings=reads_rows, store=store)
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
        # Cached: the serving path hits the same two models over and over, and
        # a fresh construction reloads ~8.9 s of weights (78% of a served
        # query). Bounded at 2 -- see factory.build_embedder_cached for why
        # the eval path deliberately does NOT share this.
        embedder = build_embedder_cached(
            StrategySpec.model_validate(manifest["combo"]["embedder"])
        )
        index = load_index_cached(index_dir, store=store)
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


def warm_serving_caches(
    indices: list[IndexInfo],
    retriever_type: str = "hybrid",
    *,
    with_rows: bool = True,
    probe: str = "อุ่นเครื่อง",
    probe_retrieval: bool = True,
    retriever_params: dict | None = None,
    route_combo: dict[str, RouteTarget] | None = None,
) -> dict:
    """Load everything the shipped routes will need, before a user asks.

    The two serving caches take a warm routed query from 12,329 ms to 447 ms
    (`data/results/serving_cost_profile.md`) -- but only for the SECOND caller
    on each route. The first still pays ~9.3 s of weight loading plus ~1.1 s of
    index read plus the ~1.0 s BM25 rebuild, and there are four routed indices
    and two embedders, so a fresh process has four such first callers. This
    front-loads them.

    Three details are load-bearing, each measured rather than assumed.

    1. **Building the embedder is not loading it.** `LocalSTEmbedder._load()`
       runs inside the first `embed()`, so constructing one costs 0.0 ms and
       warms nothing at all -- the trap that made the first cost decomposition
       price this whole cache at 10% instead of 78%
       ([[feedback_a_lazy_constructor_hides_the_cost_you_are_pricing]]). A one
       string `embed` is what actually pulls the weights onto the card.
    2. **The BM25 scorer rides on the Index, not on the cache.** Discarding an
       Index discards the memo, and rebuilding it is ~1.0 s of the first hybrid
       query -- so warming the rows without warming the scorer delivers less
       than half of what this function claims. Whether to warm it is **derived
       from `retriever_type`, not a separate flag**: a dense deployment would
       pay ~1.0 s per index for a structure it never scores with, and a caller
       who could ask for `hybrid` *and* "no scorer" would be asking for a state
       that cannot serve -- the probe retrieval in (4) would build it anyway.
    3. **Route targets are retriever-dependent** (`route_targets`), and the five
       routes resolve to four distinct index directories -- `faculty` and
       `unmatched` share one -- so this dedupes by directory rather than
       looping over routes. `route_combo` overrides the map, as in `route_query`.

    4. **Loading everything is still not warm.** With all four indices and both
       embedders resident, the first real query measured **1,240.8 ms** against
       ~430 for the ones after it -- and a single throwaway retrieval before
       them takes that first one to **488.6 ms** (2026-08-21, one process per
       arm, four routed queries). The residue is process-global CUDA/BLAS
       initialisation, not per-index: ONE probe retrieval fixed all four routes,
       which is why this does one rather than one per index. It is the same
       lesson as (1) one layer up -- a resource can be present and still not be
       initialised. End to end over those four queries: cold **30,550 ms**,
       warmed **1,634 ms** after a 29,642 ms warm-up (of which the probe is
       1,093).

       **Pass `retriever_params`.** The probe must exercise the path the
       deployment serves: left at the class defaults a `hybrid` probe fuses at
       `fetch_depth=None`, i.e. the whole corpus, which costs 2,052 ms against
       the shipped F=200's ~470 -- warming a slower code path than the one the
       user's query will take, and charging the difference to startup.

    `with_rows=False` is the engine-served shape: `query_indices` loads without
    `embeddings.npy` when the retriever reports `reads_index_rows is False`, and
    that flag is **part of the cache key**, so warming the full variant for an
    engine retriever would both waste ~234MB per index and warm an entry the
    serving path never asks for. It also implies no lexical warm-up -- the
    engine scores its own sparse arm. It does **not** imply skipping the probe:
    that was the original gating and it cost an engine-only process ~485 ms on
    its first real query, because the probe's job in (4) is process-global
    initialisation that has nothing to do with which rows are resident.

    Returns what was warmed and what it cost, so a caller can log it. Failures
    are collected per target rather than raised: a warm-up is an optimisation,
    and a deployment that refuses to start because one index directory is
    missing is worse than one that serves the other three.
    """
    import time

    from rag_lab.retrievers.bm25 import BM25Retriever

    store = ArtifactStore()
    targets = route_targets(retriever_type) if route_combo is None else route_combo
    seen: set[str] = set()
    first_pair = None
    warmed: list[dict] = []
    failures: list[dict] = []
    t_all = time.perf_counter()

    for route, target in targets.items():
        try:
            info = resolve_index(target, indices)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            failures.append({"route": route, "stage": "resolve", "error": str(exc)})
            continue
        if info.dir in seen:
            continue
        seen.add(info.dir)

        entry: dict = {"route": route, "dir": info.dir}
        t0 = time.perf_counter()
        try:
            manifest = _read_manifest(info.dir)
            spec = StrategySpec.model_validate(manifest["combo"]["embedder"])
            embedder = build_embedder_cached(spec)
            embedder.embed([probe])  # forces the lazy weight load -- see (1)
            entry["embedder_ms"] = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            index = load_index_cached(info.dir, with_embeddings=with_rows, store=store)
            entry["index_ms"] = (time.perf_counter() - t1) * 1000
            entry["n_chunks"] = len(index.chunks)

            if with_rows and _LEXICAL_RETRIEVERS & {retriever_type} and index.lexical is not None:
                t2 = time.perf_counter()
                BM25Retriever._scorer(index)  # memoised ON the Index -- see (2)
                entry["lexical_ms"] = (time.perf_counter() - t2) * 1000
        except Exception as exc:  # noqa: BLE001
            failures.append({"route": route, "dir": info.dir, "error": str(exc)})
            continue
        entry["total_ms"] = (time.perf_counter() - t0) * 1000
        warmed.append(entry)
        if first_pair is None:
            first_pair = (index, embedder)

    probe_ms = None
    # NOT gated on `with_rows`, and that was a measured defect rather than a
    # tidy-up (2026-08-21): the probe's job in (4) is process-global CUDA/BLAS
    # initialisation, which an engine-served deployment pays exactly like a
    # row-reading one. Gated on with_rows, an engine-only process got no probe
    # and its first real query cost ~485 ms against a ~159 ms steady state
    # (`data/results/serving_concurrency.md` section 3). The engine retriever
    # reads no Index rows, so a rows-less Index is a valid probe target, and a
    # probe that cannot run is collected as a failure rather than raised.
    if probe_retrieval and first_pair is not None:
        index, embedder = first_pair
        t3 = time.perf_counter()
        try:
            retrieve(
                probe,
                index,
                embedder,
                # Cached, so the probe leaves the retriever resident too --
                # the third construction the serving path pays for.
                build_retriever_cached(
                    StrategySpec(type=retriever_type, params=retriever_params or {})
                ),
                1,
                combination_id="warmup",
            )
            probe_ms = (time.perf_counter() - t3) * 1000
        except Exception as exc:  # noqa: BLE001 - an optimisation, never fatal
            # e.g. an engine retriever that needs a url it was not given.
            failures.append({"route": "(probe retrieval)", "error": str(exc)})

    return {
        "retriever_type": retriever_type,
        "with_rows": with_rows,
        "probe_ms": probe_ms,
        "warmed": warmed,
        "failures": failures,
        "total_ms": (time.perf_counter() - t_all) * 1000,
    }
