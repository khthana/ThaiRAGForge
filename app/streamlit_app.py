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
from rag_lab.router import classify_query  # noqa: E402

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
        "Classify the query as person-/program-/unmatched-shaped "
        "(src/rag_lab/router.py) and query only that route's best-performing "
        "combo, instead of comparing every selected combo side by side. "
        "Needs a fixed_size+bge-m3, semantic+bge-m3, and "
        "sentence+ConGen-PhayaThaiBERT combo built under this index dir -- "
        "see docs on tools/eval/gold_embedder_breakdown.py for how those "
        "were picked."
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
            "'rrf': also query the person/program combos and merge with "
            "Reciprocal Rank Fusion -- improves MRR/ndcg@10 on the Gold set "
            "but not recall@10 (see project_hybrid_routing memory)."
        ),
    )
retriever = st.sidebar.selectbox("Retriever", ["dense", "bm25", "hybrid"], index=0, key="retriever")
k = st.sidebar.slider("top-k", min_value=1, max_value=20, value=5, key="k")
year_filter = st.sidebar.text_input("Filter by year (พ.ศ., optional)", "", key="year_filter")
query = st.text_input("Query (คำค้น)", key="query")


def _render_result(label: str, result) -> None:
    st.subheader(label)
    for r in result.results:
        st.markdown(f"**#{r.rank}** · score `{r.score:.3f}` · p{r.page} · `{r.resolution_id}`")
        st.write(r.text[:300])
        st.divider()


search_clicked = st.button("Search", type="primary", key="search_button")

if search_clicked and query and smart_routing:
    route = classify_query(query)
    st.info(f"Classified route: **{route}**")
    try:
        result = route_query(
            query, infos, StrategySpec(type=retriever), k,
            results_dir="data/results/mode_b_routed",
            unmatched_strategy=unmatched_strategy,
        )
    except LookupError as e:
        st.error(
            f"{e}\n\nSmart routing needs specific combos built under this index "
            "dir (see the checkbox tooltip). Point 'Index output dir' at one "
            "that has them, or turn off smart routing to compare freely."
        )
    else:
        _render_result(result.combination_id, result)
elif search_clicked and query and not smart_routing and selected:
    dirs = [by_id[c].dir for c in selected]
    criteria = {"year": year_filter.strip()} if year_filter.strip() else None
    combos = query_indices(
        query,
        dirs,
        StrategySpec(type=retriever),
        k,
        results_dir="data/results/mode_b",
        filter_criteria=criteria,
    )
    for col, cr in zip(st.columns(len(combos)), combos):
        with col:
            _render_result(_combo_label(cr.combo_id), cr.result)
