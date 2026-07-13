"""Consensus Review App -- browse the consensus-flagged files from
`llm_ocr_scan.py` (both phi4:latest and gemma4:e4b independently flagged the
same page), rendered as real Markdown so tables and prose are actually
readable, instead of the raw OCR text or the short quoted span alone.

Record a verdict per file ("ควร re-OCR" / "false positive" / "ไม่แน่ใจ") --
persisted to an append-only decision log, resumable across restarts. Worklist
generation (tickets.md ticket 3) is not implemented yet.

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
_DEFAULT_DECISIONS_FILE = _DEFAULT_CORPUS_ROOT / "llm_ocr_scan" / "review_decisions.jsonl"

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
decisions_file = Path(
    st.sidebar.text_input(
        "review_decisions.jsonl path", str(_DEFAULT_DECISIONS_FILE), key="decisions_file"
    )
)

if not consensus_file.exists():
    st.error(f"ไม่พบไฟล์: {consensus_file}")
    st.stop()

entries = logic.parse_consensus_priority(consensus_file)

if not entries:
    st.info("consensus_priority.md ไม่มีรายการ")
    st.stop()

resolved = logic.resolve_decisions(logic.load_decisions(decisions_file))
show_all = st.sidebar.checkbox(
    "แสดงไฟล์ที่ตัดสินแล้วด้วย (แก้ไข verdict ได้)", value=False, key="show_all"
)
st.sidebar.caption(f"ตัดสินแล้ว {len(resolved)}/{len(entries)}")

visible_entries = entries if show_all else [e for e in entries if e.file not in resolved]

if not visible_entries:
    st.success("ตรวจครบทุกไฟล์ที่ยังไม่ตัดสินแล้ว -- ติ๊ก \"แสดงไฟล์ที่ตัดสินแล้วด้วย\" เพื่อย้อนดู")
    st.stop()

if "review_idx" not in st.session_state:
    st.session_state.review_idx = 0
st.session_state.review_idx = max(0, min(st.session_state.review_idx, len(visible_entries) - 1))
idx = st.session_state.review_idx

entry = visible_entries[idx]

st.caption(
    f"ไฟล์ {idx + 1}/{len(visible_entries)} ({'ทั้งหมด' if show_all else 'ยังไม่ตัดสิน'})"
    f" -- ปี {entry.year}"
)
st.subheader(entry.file)

prior_decision = resolved.get(entry.file)
if prior_decision is not None:
    note_suffix = f" -- โน้ต: {prior_decision.note}" if prior_decision.note else ""
    st.info(f"ตัดสินไปแล้ว: {prior_decision.verdict}{note_suffix}")

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

note = st.text_input("โน้ต (ถ้ามี)", value="", key=f"note_{entry.file}")

col_reocr, col_fp, col_unsure = st.columns(3)
with col_reocr:
    if st.button(logic.VERDICT_REOCR, key=f"verdict_reocr_{entry.file}"):
        logic.append_decision(decisions_file, entry.year, entry.file, logic.VERDICT_REOCR, note=note)
        st.rerun()
with col_fp:
    if st.button(logic.VERDICT_FALSE_POSITIVE, key=f"verdict_fp_{entry.file}"):
        logic.append_decision(
            decisions_file, entry.year, entry.file, logic.VERDICT_FALSE_POSITIVE, note=note
        )
        st.rerun()
with col_unsure:
    if st.button(logic.VERDICT_UNSURE, key=f"verdict_unsure_{entry.file}"):
        logic.append_decision(decisions_file, entry.year, entry.file, logic.VERDICT_UNSURE, note=note)
        st.rerun()

col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("← ก่อนหน้า", disabled=idx == 0, key="prev_button"):
        st.session_state.review_idx = idx - 1
        st.rerun()
with col_next:
    if st.button("ถัดไป →", disabled=idx >= len(visible_entries) - 1, key="next_button"):
        st.session_state.review_idx = idx + 1
        st.rerun()
