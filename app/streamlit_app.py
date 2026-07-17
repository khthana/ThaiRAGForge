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

output_dir = st.sidebar.text_input("Index output dir", "data/index/dev_smoke", key="output_dir")
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

by_id = {info.combo_id: info for info in infos}

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
        "Combinations to compare", list(by_id), default=list(by_id)[:2], key="selected_combos",
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
            _render_result(cr.combo_id, cr.result)
