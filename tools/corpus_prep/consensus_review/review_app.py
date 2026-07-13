"""Consensus Review App -- browse the consensus-flagged files from
`llm_ocr_scan.py` (both phi4:latest and gemma4:e4b independently flagged the
same page), rendered as real Markdown so tables and prose are actually
readable, instead of the raw OCR text or the short quoted span alone.

Read-only in this ticket: no verdict recording yet (see tickets.md ticket 2).

Run with:  streamlit run tools/corpus_prep/consensus_review/review_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# tools/corpus_prep/ has no __init__.py -- add this directory to sys.path
# directly and import the bare module name (same convention
# app/pages/1_build_run.py uses for src/: insert then plain import).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import logic  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CORPUS_ROOT = REPO / "academic_resolutions"
_DEFAULT_CONSENSUS_FILE = _DEFAULT_CORPUS_ROOT / "llm_ocr_scan" / "consensus_priority.md"

st.set_page_config(page_title="Consensus Review", layout="wide")
st.title("Consensus Review -- ตรวจ flag ที่ทั้ง 2 โมเดลเห็นตรงกัน")

corpus_root = Path(
    st.sidebar.text_input("Corpus root", str(_DEFAULT_CORPUS_ROOT), key="corpus_root")
)
consensus_file = Path(
    st.sidebar.text_input(
        "consensus_priority.md path", str(_DEFAULT_CONSENSUS_FILE), key="consensus_file"
    )
)

if not consensus_file.exists():
    st.error(f"ไม่พบไฟล์: {consensus_file}")
    st.stop()

entries = logic.parse_consensus_priority(consensus_file)

if not entries:
    st.info("consensus_priority.md ไม่มีรายการ")
    st.stop()

if "review_idx" not in st.session_state:
    st.session_state.review_idx = 0
st.session_state.review_idx = max(0, min(st.session_state.review_idx, len(entries) - 1))
idx = st.session_state.review_idx

st.caption(f"ไฟล์ {idx + 1}/{len(entries)} -- ปี {entries[idx].year}")

entry = entries[idx]
st.subheader(entry.file)

if logic.is_split_piece(entry.file):
    siblings = logic.consensus_siblings(entries, entry)
    if siblings:
        st.warning(
            "ส่วนหนึ่งของเอกสารที่ถูกตัดเป็นหลายชิ้น -- ชิ้นอื่นในลิสต์ consensus นี้: "
            + ", ".join(siblings)
        )
    else:
        st.warning(
            "ส่วนหนึ่งของเอกสารที่ถูกตัดเป็นหลายชิ้น -- ชิ้นอื่นไม่อยู่ในลิสต์ consensus "
            "นี้ (อาจไม่ถูก flag) แต่ควรเช็คไฟล์พี่น้องในคอร์ปัสด้วย"
        )

for page in entry.pages:
    st.markdown(f"#### {page.page}")
    for model, flag in page.models.items():
        line = f"**[{model}]** {flag.reason}"
        if flag.span:
            line += f"  \n> {flag.span}"
        st.markdown(line)

    body = logic.load_page_markdown(corpus_root, entry.file, page.page)
    if body is None:
        st.error("ไม่พบเนื้อหาหน้านี้ในคอร์ปัส (ไฟล์อาจถูกย้าย/แก้ไข)")
    else:
        st.markdown(body)
    st.divider()

col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("← ก่อนหน้า", disabled=idx == 0, key="prev_button"):
        st.session_state.review_idx = idx - 1
        st.rerun()
with col_next:
    if st.button("ถัดไป →", disabled=idx >= len(entries) - 1, key="next_button"):
        st.session_state.review_idx = idx + 1
        st.rerun()
