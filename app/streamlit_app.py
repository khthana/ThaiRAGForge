"""Mode B UI — Query & Compare. Thin Streamlit shell over rag_lab.query_service.

Run with:  streamlit run app/streamlit_app.py
All logic lives in the Streamlit-free core; this file only renders.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# make src/ importable when launched via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.query_service import discover_indices, query_indices, route_query  # noqa: E402
from rag_lab.router import classify_query, detect_entities, route_targets  # noqa: E402

st.set_page_config(page_title="RAG Lab — Query & Compare", layout="wide")
st.title("RAG Lab — Query & Compare (Mode B)")

def _discoverable_index_dirs() -> list[str]:
    """Subdirectories of data/index/ that actually contain built combos --
    lets the picker be a selectbox (no risk of a stale/mistyped path) while
    still covering every experiment output on disk, not just one hardcoded name."""
    root = Path("data/index")
    if not root.is_dir():
        return []
    found = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and discover_indices(d):
            found.append(str(d))
    return found


_CUSTOM_PATH = "Custom path..."
_dir_options = _discoverable_index_dirs()
# default to a dir with at least one non-toy embedder if one exists, so the
# picker doesn't silently land on a hashing-only dev/smoke index; with no
# discoverable dirs at all (fresh clone, nothing built yet) fall through to
# the custom-path text input -- same widget tree either way, just which one
# is pre-selected, so behavior doesn't depend on what happens to be on disk.
_default_choice = next(
    (d for d in _dir_options if any(i.embedder.type != "hashing" for i in discover_indices(d))),
    _dir_options[0] if _dir_options else _CUSTOM_PATH,
)
_all_choices = _dir_options + [_CUSTOM_PATH]
_choice = st.sidebar.selectbox(
    "Index output dir", _all_choices, index=_all_choices.index(_default_choice), key="output_dir_choice",
)
output_dir = (
    st.sidebar.text_input("Custom index dir", "", key="output_dir")
    if _choice == _CUSTOM_PATH
    else _choice
)

try:
    infos = discover_indices(output_dir)
except FileNotFoundError:
    infos = []

if not infos:
    st.warning(
        f"No built indices with a manifest under `{output_dir}`.\n\n"
        "Build some first, e.g.\n"
        "`PYTHONPATH=src python -m rag_lab.cli run --config config/experiments/dev_smoke.yaml`"
    )
    st.stop()

if all(i.embedder.type == "hashing" for i in infos):
    st.warning(
        "This index was built with the **hashing** embedder -- a fast placeholder "
        "with no real semantic understanding, meant for testing the pipeline, not "
        "for judging retrieval quality. Pick an index dir with a real embedder "
        "(`e5` or `local`) for results that mean anything, e.g. `data/index/chunker_compare_full`."
    )

by_id = {info.combo_id: info for info in infos}


def _param_summary(spec: StrategySpec) -> str:
    """Short, non-exhaustive disambiguator for the one param that's actually
    varied across combos in this repo's experiments (chunk_size) -- without
    it, e.g. fixed_size@512 and fixed_size@256 render as the same label."""
    chunk_size = spec.params.get("chunk_size")
    return f"[{chunk_size}]" if chunk_size is not None else ""


def _combo_label(combo_id: str) -> str:
    """combo_id alone collapses distinct local-embedder models (e.g. bge-m3
    vs ConGen-PhayaThaiBERT), or distinct chunker params (e.g. chunk_size),
    to the same visible prefix plus an opaque hash -- show the actual
    model_name / param so they're distinguishable."""
    info = by_id[combo_id]
    model = info.embedder.params.get("model_name")
    embedder_label = f"{info.embedder.type} ({model})" if model else info.embedder.type
    chunker_label = f"{info.chunker.type}{_param_summary(info.chunker)}"
    return f"{chunker_label} + {embedder_label}"


smart_routing = st.sidebar.checkbox(
    "Smart routing (route by query shape)", value=False, key="smart_routing",
    help=(
        "Classify the query as person-/program-/course-/faculty-/unmatched-shaped "
        "(src/rag_lab/router.py) and query only that route's best-performing "
        "combo, instead of comparing every selected combo side by side. Which "
        "combo each route reaches depends on the Retriever setting below "
        "(router.route_targets), so the required combos differ between dense and "
        "hybrid -- the caption under the result names the one actually used. "
        "Evidence for the targets: tools/eval/routing_eval.py -> "
        "data/results/routing_eval.md."
    ),
)
if not smart_routing:
    selected = st.sidebar.multiselect(
        "Combinations to compare", list(by_id), default=list(by_id)[:2],
        format_func=_combo_label, key="selected_combos",
    )
else:
    unmatched_strategy = st.sidebar.radio(
        "Unmatched-query fallback", ["default", "rrf"], index=0, key="unmatched_strategy",
        help=(
            "'default': query only the unmatched route's combo (cheaper). "
            "'rrf': also query the other routes' combos and merge with "
            "Reciprocal Rank Fusion. UNMEASURED -- since the 2026-08-08 route "
            "expansion no Gold query falls through to 'unmatched' (0/106), so "
            "no eval exercises this branch; its old numbers came from the "
            "retired 252-query/3-route eval and are withdrawn."
        ),
    )
retriever = st.sidebar.selectbox(
    "Retriever", ["dense", "bm25", "hybrid", "qdrant_hybrid"], index=0, key="retriever",
    help=(
        "'hybrid' fuses in-process numpy + rank_bm25. 'qdrant_hybrid' fuses the "
        "SAME two arms served by a Qdrant server (dense exact=True + precomputed "
        "BM25 sparse vectors), with the identical RRF -- data/results/"
        "qdrant_routed_check.md measured the served routed stack at 0.6827 "
        "against the published 0.6835, and the per-query Python scoring it "
        "replaces is ~0.4s. It needs a running container AND the collections "
        "ingested by tools/eval/qdrant_pilot_ingest.py; a collection is a copy "
        "of an index's rows, so an index rebuild stales it."
    ),
)
_ENGINE_RETRIEVERS = {"qdrant_hybrid"}
_FUSED_RETRIEVERS = {"hybrid", "qdrant_hybrid"}
_WHOLE_CORPUS = "ทั้งคลัง (k=n)"
qdrant_url = st.sidebar.text_input(
    "Qdrant URL", "http://localhost:6333", key="qdrant_url",
    disabled=retriever not in _ENGINE_RETRIEVERS,
)
fetch_depth = st.sidebar.selectbox(
    "Hybrid fetch depth", [200, 1000, _WHOLE_CORPUS], index=0, key="fetch_depth",
    disabled=retriever not in _FUSED_RETRIEVERS,
    help=(
        "How many candidates each arm returns before RRF fuses them. 'ทั้งคลัง' "
        "is what every published number was measured at -- RRF sees complete "
        "rankings, and a chunk past an arm's cut loses that arm's term outright "
        "rather than earning a small one, so this is a real change to the "
        "ranking, not a cache. 200 is the default here on measurement: against "
        "the shipped hard router it is 2.51x faster (p50 1193.9ms -> 475.6ms) "
        "for no significant quality cost on any metric (recall@10 +0.0005, MRR "
        "-0.0024, nDCG@10 -0.0022, all Holm-adj 1.0000, m=3; the CI rules out a "
        "loss worse than 0.0078). It does change the top-10 on 17 of 106 Gold "
        "queries, which is why eval code keeps k=n. See "
        "data/results/routed_fetch_depth_test.md."
    ),
)
_engine_depth_unavailable = retriever in _ENGINE_RETRIEVERS and fetch_depth == _WHOLE_CORPUS
if _engine_depth_unavailable:
    st.sidebar.warning(
        f"`{retriever}` has no whole-corpus setting: each arm is a `limit=` "
        "request to the server, so 'ทั้งคลัง' would mean fetching every point "
        "in the collection twice per query — the over-fetch the served path "
        "exists to remove. Pick 200 or 1000."
    )


def _retriever_spec() -> StrategySpec:
    """The retriever the UI queries with.

    `fetch_depth` is set HERE, not on HybridRetriever's constructor, and that
    split is deliberate: the class default stays `None` (= whole corpus) so
    every eval script and all ~24k persisted results keep reproducing their
    published numbers by construction. Only the interactive path, where 0.72s
    per query is what a person actually feels, opts into the cut.

    `qdrant_hybrid` is the one retriever with no whole-corpus setting: its arms
    are two `limit=` requests to a server, so "k=n" would mean asking the engine
    for all ~57k points per arm -- the exact over-fetch the served path exists to
    remove. Rather than silently substituting a depth, that combination is
    refused above (the caption beside the selectbox) and never reaches here.
    """
    if retriever in _ENGINE_RETRIEVERS:
        params: dict = {"url": qdrant_url.strip()}
        if fetch_depth != _WHOLE_CORPUS:
            params["fetch_depth"] = int(fetch_depth)
        return StrategySpec(type=retriever, params=params)
    if retriever != "hybrid" or fetch_depth == _WHOLE_CORPUS:
        return StrategySpec(type=retriever)
    return StrategySpec(type=retriever, params={"fetch_depth": int(fetch_depth)})
k = st.sidebar.slider("top-k", min_value=1, max_value=20, value=5, key="k")
_engine_narrowing = retriever in _ENGINE_RETRIEVERS
year_filter = st.sidebar.text_input(
    "Filter by year (พ.ศ., optional)", "", key="year_filter", disabled=_engine_narrowing,
    help=(
        "Disabled for an engine-served retriever: both this and the entity boost "
        "narrow the in-process Index, which cannot narrow what the server "
        "returns. query_service refuses the combination rather than answering an "
        "unfiltered query as if it were filtered."
        if _engine_narrowing else None
    ),
)
entity_boost = st.sidebar.checkbox(
    "Entity boost (narrow to detected person/program/course/faculty before ranking)",
    value=False, key="entity_boost", disabled=_engine_narrowing,
    help=(
        "Detects named people/programs/courses/faculties in the query "
        "(src/rag_lab/router.py's detect_entities) and, for any selected "
        "combo built with the entity_tags loader (e.g. data/index/"
        "entity_tags_full), narrows to only resolutions mentioning that "
        "entity before ranking with the chosen retriever -- fixes cases "
        "where the entity's name alone doesn't carry enough weight to rank "
        "the right resolutions into the top-k. No effect on a combo built "
        "with any other loader (narrowing is skipped for it, not an error)."
    ),
)
if _engine_narrowing:
    # A DISABLED Streamlit widget keeps whatever the session already held, so
    # switching to an engine retriever with a year typed (or the boost ticked)
    # would carry that value into query_indices and surface its refusal as a
    # traceback. Read the disabled state instead of the widget. Nothing is
    # silently dropped: both widgets are visibly disabled and say why.
    year_filter, entity_boost = "", False
query = st.text_input("Query (คำค้น)", key="query")


def _render_result(label: str, result) -> None:
    boosted = result.combination_id.endswith("__entity_boost")
    st.subheader(f"{label} (entity-boosted)" if boosted else label)
    for r in result.results:
        st.markdown(f"**#{r.rank}** · score `{r.score:.3f}` · p{r.page} · `{r.resolution_id}`")
        st.write(r.text[:300])
        st.divider()


def _show_detected_entities(q: str) -> None:
    detected = detect_entities(q)
    if detected:
        parts = [f"{kind}: {', '.join(values)}" for kind, values in detected.items()]
        st.caption("Detected entities — " + " · ".join(parts))
    elif entity_boost:
        st.caption("Detected entities — none (entity boost has no effect on this query)")


search_clicked = st.button(
    "Search", type="primary", key="search_button", disabled=_engine_depth_unavailable,
)

if search_clicked and query and smart_routing:
    route = classify_query(query)
    target = route_targets(retriever).get(route)
    st.info(f"Classified route: **{route}**")
    # Since 2026-08-08 the route -> index map is keyed by retriever (the best
    # target per route differs between dense and hybrid), so switching the
    # retriever silently changes which index a route reaches. Say so, or the
    # same query under two retrievers looks like an unexplained result change.
    if target is not None:
        st.caption(
            f"Route target for retriever `{retriever}`: "
            f"{target.chunker_type} + {target.embedder_model_name or target.embedder_type}"
        )
    try:
        result = route_query(
            query, infos, _retriever_spec(), k,
            results_dir="data/results/mode_b_routed",
            unmatched_strategy=unmatched_strategy,
        )
    except LookupError as e:
        st.error(
            f"{e}\n\nSmart routing needs specific combos built under this index "
            "dir (see the checkbox tooltip). Point 'Index output dir' at one "
            "that has them, or turn off smart routing to compare freely."
        )
    except FileNotFoundError as e:
        # qdrant_hybrid resolves its collection from the routed index's own
        # directory name, so an un-ingested route fails here and nowhere else.
        st.error(f"{e}")
    else:
        _render_result(result.combination_id, result)
elif search_clicked and query and not smart_routing and selected:
    dirs = [by_id[c].dir for c in selected]
    criteria = {"year": year_filter.strip()} if year_filter.strip() else None
    _show_detected_entities(query)
    combos = query_indices(
        query,
        dirs,
        _retriever_spec(),
        k,
        results_dir="data/results/mode_b",
        filter_criteria=criteria,
        entity_boost=entity_boost,
    )
    for col, cr in zip(st.columns(len(combos)), combos):
        with col:
            _render_result(_combo_label(cr.combo_id), cr.result)
