"""Chunk inspector — thin Streamlit shell for visually reviewing how each
chunker splits a real document, and triaging abnormally large chunks
(design agreed 2026-07-25: user's concern was that chunks live in a
parquet+numpy artifact that's hard to eyeball).

Reads chunks.parquet directly (never embeddings.npy): chunk text/chunk_id
are byte-identical across every embedder for the same chunker, since
chunking happens before embedding (verified against
plain__semantic__local__834c4336 vs plain__semantic__qwen3__a0f495a8 --
same chunk count, same ids, same text). So one representative combo per
chunker is enough, and the embeddings matrix -- the largest artifact on
disk, and irrelevant to this page's job -- is never loaded.

Run with:  streamlit run app/streamlit_app.py   (this page appears in the nav)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

# make src/ importable when launched via `streamlit run`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

st.set_page_config(page_title="RAG Lab — Chunk Inspector", layout="wide")
st.title("RAG Lab — Chunk Inspector")

_DEFAULT_BASE_DIR = "data/index/chunker_compare_full"
_COLUMNS = ["chunk_id", "resolution_id", "text", "chunk_index", "page"]

base_dir = st.sidebar.text_input(
    "Index directory to inspect (a built combo per chunker lives under here)",
    _DEFAULT_BASE_DIR,
    key="base_dir",
)


@st.cache_data(show_spinner=False)
def discover_representative_combos(base_dir_str: str) -> dict[str, str]:
    """chunker type -> one built combo directory (first match, alphabetical).
    Any embedder works -- chunk text is embedder-independent for a given
    chunker, see module docstring."""
    base = Path(base_dir_str)
    reps: dict[str, str] = {}
    if not base.is_dir():
        return reps
    for combo_dir in sorted(base.iterdir()):
        meta_path = combo_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunker_type = meta.get("chunker", {}).get("type")
        if chunker_type and chunker_type not in reps:
            reps[chunker_type] = str(combo_dir)
    return reps


@st.cache_data(show_spinner="Loading chunks…")
def load_chunks(combo_dir_str: str) -> pd.DataFrame:
    table = pq.read_table(Path(combo_dir_str) / "chunks.parquet", columns=_COLUMNS)
    df = table.to_pandas()
    df["length"] = df["text"].str.len()
    return df


combos = discover_representative_combos(base_dir)
if not combos:
    st.error(f"No built combo (with a meta.json) found under `{base_dir}`.")
    st.stop()

st.caption(
    "Representative combo per chunker (chunk text is embedder-independent, "
    "so any built embedder for that chunker works): "
    + " · ".join(f"**{k}** = `{Path(v).name}`" for k, v in sorted(combos.items()))
)

chunk_frames = {chunker: load_chunks(path) for chunker, path in combos.items()}

tab_compare, tab_oversized = st.tabs(["เปรียบเทียบเอกสาร", "Chunk ใหญ่ผิดปกติ"])

with tab_compare:
    st.subheader("How each chunker splits the same document")
    any_df = next(iter(chunk_frames.values()))
    all_resolution_ids = sorted(any_df["resolution_id"].unique())

    query = st.text_input(
        "Search resolution_id / title (resolution_id embeds the title, e.g. "
        "'2566/4s/เรื่อง ...')",
        key="doc_search",
    )
    if query:
        filtered = [r for r in all_resolution_ids if query.lower() in r.lower()]
    else:
        filtered = all_resolution_ids[:200]

    if not filtered:
        st.info("No matching resolution.")
    else:
        if not query:
            st.caption(f"Showing first 200 of {len(all_resolution_ids)} resolutions — search to narrow.")
        picked = st.selectbox(
            f"Resolution ({len(filtered)} match{'es' if len(filtered) != 1 else ''})",
            filtered,
            key="doc_pick",
        )
        cols = st.columns(len(chunk_frames))
        for col, (chunker, df) in zip(cols, sorted(chunk_frames.items())):
            with col:
                st.markdown(f"**{chunker}**")
                sub = df[df["resolution_id"] == picked].sort_values("chunk_index")
                st.caption(f"{len(sub)} chunks")
                for _, row in sub.iterrows():
                    with st.expander(f"#{row['chunk_index']} · {row['length']:,} chars · page {row['page']}"):
                        st.text(row["text"])

with tab_oversized:
    st.subheader("Abnormally large chunks")
    chunker_pick = st.selectbox("Chunker", sorted(chunk_frames), key="oversized_chunker")
    threshold = st.number_input(
        "Length threshold (chars)", min_value=100, value=3000, step=500, key="oversized_threshold"
    )

    df = chunk_frames[chunker_pick]
    over = df[df["length"] > threshold].sort_values("length", ascending=False)
    st.caption(f"{len(over)} of {len(df)} chunks in `{chunker_pick}` exceed {threshold:,} chars")

    if len(over):
        st.dataframe(
            over[["resolution_id", "chunk_index", "page", "length"]],
            width="stretch",
            height=min(400, 40 + 35 * len(over)),
        )
        label_to_text = {
            f"{row.length:,} chars — #{row.chunk_index} — {row.resolution_id}": row.text
            for row in over.itertuples()
        }
        picked_label = st.selectbox(
            "Pick a chunk to view its full text", list(label_to_text), key="oversized_pick",
        )
        st.text_area(
            "Full chunk text", label_to_text[picked_label], height=400, key="oversized_text"
        )
