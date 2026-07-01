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
from rag_lab.query_service import discover_indices, query_indices  # noqa: E402

st.set_page_config(page_title="RAG Lab — Query & Compare", layout="wide")
st.title("RAG Lab — Query & Compare (Mode B)")

output_dir = st.sidebar.text_input("Index output dir", "data/index/dev_smoke")
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
selected = st.sidebar.multiselect(
    "Combinations to compare", list(by_id), default=list(by_id)[:2]
)
retriever = st.sidebar.selectbox("Retriever", ["dense", "bm25", "hybrid"], index=0)
k = st.sidebar.slider("top-k", min_value=1, max_value=20, value=5)
year_filter = st.sidebar.text_input("Filter by year (พ.ศ., optional)", "")
query = st.text_input("Query (คำค้น)")

if st.button("Search", type="primary") and query and selected:
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
            st.subheader(cr.combo_id)
            for r in cr.result.results:
                st.markdown(
                    f"**#{r.rank}** · score `{r.score:.3f}` · p{r.page} · "
                    f"`{r.resolution_id}`"
                )
                st.write(r.text[:300])
                st.divider()
