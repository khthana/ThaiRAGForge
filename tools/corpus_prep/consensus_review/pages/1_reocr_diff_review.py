"""Old-vs-new re-OCR review -- the Phase 3 decision UI for
reocr_consensus_pages.py (Phase 1: fresh re-OCR) + reocr_adjudicate.py
(Phase 2: dual-model old-vs-new adjudication).

Every page where both phi4:latest and gemma4:e4b independently verdict "new"
is a safe auto-apply candidate and doesn't show up here. Everything else
(disagreement, both prefer old, no real difference, both still bad) needs a
human decision, recorded to an append-only log -- same resumable pattern as
the file-level review_app.py this page sits alongside. Still staging-only:
nothing here writes back into the real corpus.

Run with:  streamlit run tools/corpus_prep/consensus_review/review_app.py
(this page appears in the sidebar nav automatically, next to the file-level
consensus review).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logic  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
_DEFAULT_CORPUS_ROOT = REPO / "academic_resolutions"
_DEFAULT_STAGING_FILE = _DEFAULT_CORPUS_ROOT / "llm_ocr_scan" / "reocr_pages_staging.jsonl"
_DEFAULT_ADJUDICATION_FILE = _DEFAULT_CORPUS_ROOT / "llm_ocr_scan" / "reocr_adjudication.jsonl"
_DEFAULT_DECISIONS_FILE = _DEFAULT_CORPUS_ROOT / "llm_ocr_scan" / "reocr_review_decisions.jsonl"


def _path_input(label: str, default: Path, key: str) -> Path:
    """A sidebar path override, defaulting to the real location but always
    overridable -- e.g. by tests, which seed `st.session_state[key]` before
    the first run so they never touch the real path, even transiently."""
    return Path(st.sidebar.text_input(label, str(default), key=key))


def _key(record: dict) -> tuple[str, int]:
    return (record["pdf"], record["page"])


st.set_page_config(page_title="Re-OCR Old vs New Review", layout="wide")
st.title("เทียบ Old vs New -- หน้าที่โมเดลยังไม่เห็นตรงกันว่า re-OCR ดีกว่า")

corpus_root = _path_input("Corpus root", _DEFAULT_CORPUS_ROOT, "reocr_corpus_root")
staging_file = _path_input(
    "reocr_pages_staging.jsonl path", _DEFAULT_STAGING_FILE, "reocr_staging_file"
)
adjudication_file = _path_input(
    "reocr_adjudication.jsonl path", _DEFAULT_ADJUDICATION_FILE, "reocr_adjudication_file"
)
decisions_file = _path_input(
    "reocr_review_decisions.jsonl path", _DEFAULT_DECISIONS_FILE, "reocr_decisions_file"
)

if not adjudication_file.exists():
    st.error(f"ไม่พบไฟล์: {adjudication_file}")
    st.stop()

all_records = logic.load_jsonl(adjudication_file)
staged_text = logic.staged_text_by_key(logic.load_jsonl(staging_file))
review_records = [r for r in all_records if logic.needs_reocr_review(r)]

st.sidebar.caption(
    f"ทั้งหมด {len(all_records)} หน้า -- ต้องรีวิว {len(review_records)} หน้า "
    f"({len(all_records) - len(review_records)} หน้าโมเดลเห็นตรงกันว่า new ดีกว่า อยู่นอกคิวนี้)"
)

if not review_records:
    st.success("ไม่มีหน้าที่ต้องรีวิว -- ทุกหน้าโมเดลเห็นตรงกันว่า new ดีกว่า (หรือยังไม่มีข้อมูล adjudication)")
    st.stop()

decisions = logic.resolve_reocr_review_decisions(logic.load_reocr_review_decisions(decisions_file))
show_all = st.sidebar.checkbox("แสดงหน้าที่ตัดสินแล้วด้วย (แก้ไข verdict ได้)", value=False, key="reocr_show_all")
st.sidebar.caption(f"ตัดสินแล้ว {len(decisions)}/{len(review_records)}")

visible_records = review_records if show_all else [r for r in review_records if _key(r) not in decisions]

if not visible_records:
    st.success("ตรวจครบทุกหน้าที่ยังไม่ตัดสินแล้ว -- ติ๊ก \"แสดงหน้าที่ตัดสินแล้วด้วย\" เพื่อย้อนดู")
    st.stop()

if "reocr_review_idx" not in st.session_state:
    st.session_state.reocr_review_idx = 0
if "reocr_review_show_all_prev" not in st.session_state:
    st.session_state.reocr_review_show_all_prev = show_all

# Same show_all resync-by-identity approach as review_app.py: the position
# in `visible_records` isn't meaningful across a show_all toggle (undecided-
# only subset <-> full list have different membership), so re-find the same
# (pdf, page) rather than reusing the old numeric index.
if show_all != st.session_state.reocr_review_show_all_prev:
    current_key = st.session_state.get("reocr_review_current_key")
    try:
        st.session_state.reocr_review_idx = next(
            i for i, r in enumerate(visible_records) if _key(r) == current_key
        )
    except StopIteration:
        pass  # page no longer visible in the new view -- keep the old position
    st.session_state.reocr_review_show_all_prev = show_all

st.session_state.reocr_review_idx = max(
    0, min(st.session_state.reocr_review_idx, len(visible_records) - 1)
)
idx = st.session_state.reocr_review_idx

record = visible_records[idx]
st.session_state.reocr_review_current_key = _key(record)

st.caption(
    f"หน้า {idx + 1}/{len(visible_records)} ({'ทั้งหมด' if show_all else 'ยังไม่ตัดสิน'})"
)
st.subheader(f"{Path(record['pdf']).name} -- หน้า {record['page']}")

info = logic.meeting_info(corpus_root, record["files"][0])
if info is None:
    st.warning("หาปี/ครั้งที่ประชุมไม่เจอจาก path ของไฟล์")
else:
    header = f"**ปี {info['year']} {info['session']}**"
    if info["url"]:
        header += f" -- [เปิดต้นฉบับ (PDF/Drive)]({info['url']})"
    else:
        header += " -- ไม่พบลิงก์ต้นฉบับใน meeting_manifest.json"
    st.markdown(header)
    if info["title"]:
        st.caption(info["title"])

prior_decision = decisions.get(_key(record))
if prior_decision is not None:
    note_suffix = f" -- โน้ต: {prior_decision.note}" if prior_decision.note else ""
    st.info(f"ตัดสินไปแล้ว: {prior_decision.verdict}{note_suffix}")

if len(record["files"]) > 1:
    st.caption("ไฟล์คอร์ปัสที่ใช้หน้านี้ร่วมกัน: " + ", ".join(record["files"]))
if record.get("diverging_siblings"):
    st.warning("เนื้อหาของไฟล์พี่น้องไม่ตรงกัน: " + ", ".join(record["diverging_siblings"]))

for model, v in record["verdicts"].items():
    st.markdown(f"**[{model}]** {v['verdict']} -- {v['reason']}")

old_text = logic.load_old_text(corpus_root, record)
new_text = staged_text.get(_key(record))

col_old, col_new = st.columns(2)
with col_old:
    st.markdown("#### เดิม (old)")
    if old_text is None:
        st.error("ไม่พบเนื้อหาเดิมในคอร์ปัส")
    else:
        # unsafe_allow_html: the real corpus renders tables as raw HTML
        # <table> tags, not pipe Markdown -- same precedent as review_app.py.
        st.markdown(old_text, unsafe_allow_html=True)
with col_new:
    st.markdown("#### ใหม่ (re-OCR)")
    if new_text is None:
        st.error("ไม่พบข้อความ re-OCR ใน staging")
    else:
        st.markdown(new_text, unsafe_allow_html=True)

note = st.text_input("โน้ต (ถ้ามี)", value="", key=f"reocr_note_{record['pdf']}_{record['page']}")

_VERDICT_KEY_SUFFIXES = {
    logic.REOCR_VERDICT_APPLY_NEW: "apply_new",
    logic.REOCR_VERDICT_KEEP_OLD: "keep_old",
    logic.REOCR_VERDICT_DEFER: "defer",
}
for col, verdict in zip(st.columns(3), logic.REOCR_REVIEW_VERDICTS):
    with col:
        button_key = f"reocr_verdict_{_VERDICT_KEY_SUFFIXES[verdict]}_{record['pdf']}_{record['page']}"
        if st.button(verdict, key=button_key):
            logic.append_reocr_review_decision(
                decisions_file, record["pdf"], record["page"], verdict, note=note
            )
            st.rerun()

col_prev, col_next = st.columns(2)
with col_prev:
    if st.button("← ก่อนหน้า", disabled=idx == 0, key="reocr_prev_button"):
        st.session_state.reocr_review_idx = idx - 1
        st.rerun()
with col_next:
    if st.button("ถัดไป →", disabled=idx >= len(visible_records) - 1, key="reocr_next_button"):
        st.session_state.reocr_review_idx = idx + 1
        st.rerun()
