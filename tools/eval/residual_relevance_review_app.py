"""Residual-relevance review app -- browse the 126-item blinded sample from
residual_relevance_sample.py in a browser instead of raw YAML, and record a
verdict per item with one click.

Fixes a real gap in the raw-YAML workflow: the sheet's persisted `snippet`
field is truncated to `--snippet` chars (600 by default) at build time, and a
fixed character cut sometimes lands before the sentence that would establish
relevance -- there is no way to tell "not relevant" from "would be relevant if
I could see the rest" from a truncated view. This app instead re-fetches each
candidate's full, untruncated chunk text live from the persisted retrieval
results (`residual_relevance_sample.build_text_index`, the same lookup the
sheet-builder truncates from) and writes that back into the sheet's `snippet`
field alongside the verdict -- so the on-disk sheet self-heals as items are
judged, and a later run of `--score` (or anyone reading the raw YAML) sees the
full text too.

The retrieving arm stays hidden (`sample_key.yaml` is never opened here) --
blinding is the whole point of the study design in residual_relevance_sample.py.

Run with:
    .venv/Scripts/streamlit.exe run tools/eval/residual_relevance_review_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import yaml

# tools/eval/ has no __init__.py -- add it to sys.path directly and import the
# bare module name, same convention consensus_review/review_app.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import residual_relevance_sample as rrs  # noqa: E402

_VERDICT_LABELS = {"y": "เกี่ยวข้อง (y)", "n": "ไม่เกี่ยวข้อง (n)", "?": "บอกไม่ได้ (?)"}
_JUDGED = {"y", "n", "?"}


def _verdict_of(item: dict) -> str:
    return str(item.get("verdict", "")).strip().lower()


@st.cache_data
def _load_text_index() -> dict[tuple[str, str], str]:
    return rrs.build_text_index(rrs.load_qrels())


def _load_sheet() -> list[dict]:
    return yaml.safe_load(rrs._SHEET.read_text(encoding="utf-8"))


def _save_sheet(sheet: list[dict]) -> None:
    rrs._SHEET.write_text(
        "# Residual-relevance review sheet. Fill `verdict` for every item:\n"
        "#   y = this document IS relevant to the query (the qrels missed it)\n"
        "#   n = not relevant\n"
        "#   ? = cannot tell from the snippet\n"
        "# `already_judged_relevant` shows up to 3 documents the qrels DO count,\n"
        "# as a calibration reference for what 'relevant' means for this query.\n"
        "# The retrieving arm is deliberately not shown -- it is in sample_key.yaml.\n\n"
        + yaml.safe_dump(sheet, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8")


st.set_page_config(page_title="Residual Relevance Review", layout="wide")
st.title("Residual Relevance Review -- เอกสารที่ qrels ไม่เคยตัดสิน")

if not rrs._SHEET.exists():
    st.error(f"ไม่พบไฟล์: {rrs._SHEET} -- รัน `residual_relevance_sample.py` ก่อน (ไม่ใส่ --score)")
    st.stop()

text_index = _load_text_index()
sheet = _load_sheet()

judged = sum(1 for it in sheet if _verdict_of(it) in _JUDGED)
show_all = st.sidebar.checkbox("แสดงรายการที่ตัดสินแล้วด้วย (แก้ไข verdict ได้)", value=False, key="show_all")
st.sidebar.progress(judged / len(sheet) if sheet else 0.0)
st.sidebar.caption(f"ตัดสินแล้ว {judged}/{len(sheet)}")

visible = sheet if show_all else [it for it in sheet if _verdict_of(it) not in _JUDGED]

if not visible:
    st.success('ตัดสินครบทุกรายการแล้ว -- ติ๊ก "แสดงรายการที่ตัดสินแล้วด้วย" เพื่อย้อนดู/แก้ไข')
    st.stop()

if "rr_idx" not in st.session_state:
    st.session_state.rr_idx = 0
if "rr_show_all_prev" not in st.session_state:
    st.session_state.rr_show_all_prev = show_all

# Mirror consensus_review/review_app.py's resync convention: `visible`'s
# membership changes shape across the show_all toggle, so resync to the same
# *item id* rather than reusing the old numeric position.
if show_all != st.session_state.rr_show_all_prev:
    current_id = st.session_state.get("rr_current_id")
    try:
        st.session_state.rr_idx = next(i for i, it in enumerate(visible) if it["id"] == current_id)
    except StopIteration:
        pass
    st.session_state.rr_show_all_prev = show_all

st.session_state.rr_idx = max(0, min(st.session_state.rr_idx, len(visible) - 1))
idx = st.session_state.rr_idx
item = visible[idx]
st.session_state.rr_current_id = item["id"]

st.caption(
    f"รายการ {idx + 1}/{len(visible)} ({'ทั้งหมด' if show_all else 'ยังไม่ตัดสิน'})"
    f" -- id {item['id']} -- ประเภท {item['entity_type']}"
)

prior = _verdict_of(item)
if prior in _JUDGED:
    st.info(f"ตัดสินไปแล้ว: {_VERDICT_LABELS[prior]}")

st.subheader("คำถาม")
st.write(item["query"])

st.subheader("เอกสารที่พิจารณา (candidate)")
st.caption(item["candidate"])
full_text = text_index.get((item["query"], item["candidate"]))
if full_text is None:
    st.warning("หาเนื้อหาเต็มไม่เจอในผลลัพธ์ที่บันทึกไว้ -- ใช้ snippet ที่ตัดไว้แทน (อาจไม่ครบ)")
    full_text = item.get("snippet", "")
st.markdown(full_text.replace("\n", "  \n"))

with st.expander("เอกสารที่ qrels ตัดสินว่าเกี่ยวข้องแล้ว (ใช้เทียบมาตรฐานว่า 'เกี่ยวข้อง' หมายถึงอะไรสำหรับคำถามนี้)"):
    for rid in item.get("already_judged_relevant", []):
        st.markdown(f"- {rid}")

col_y, col_n, col_u = st.columns(3)
for col, verdict in zip((col_y, col_n, col_u), ("y", "n", "?")):
    with col:
        if st.button(_VERDICT_LABELS[verdict], key=f"verdict_{verdict}_{item['id']}", use_container_width=True):
            for it in sheet:
                if it["id"] == item["id"]:
                    it["verdict"] = verdict
                    it["snippet"] = full_text  # self-heal the on-disk truncation
                    break
            _save_sheet(sheet)
            st.rerun()

col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("← ก่อนหน้า", disabled=idx == 0, key="prev_button"):
        st.session_state.rr_idx = idx - 1
        st.rerun()
with col_next:
    if st.button("ถัดไป →", disabled=idx >= len(visible) - 1, key="next_button"):
        st.session_state.rr_idx = idx + 1
        st.rerun()
