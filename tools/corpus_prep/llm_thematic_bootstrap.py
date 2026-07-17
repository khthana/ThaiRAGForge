# -*- coding: utf-8 -*-
"""Freeform-LLM thematic-query bootstrap for the Gold query set.

The two query shapes in build_gold_candidates.py (program-history,
person-history, faculty-adjunct-aggregate) all have ground truth derivable
deterministically from entity dictionaries -- no LLM judgment needed. This
script covers the deliberately-deferred third shape: *thematic* queries that
don't anchor to one named entity ("what changed about international-program
tuition fees in this session", "which resolutions this session touched
admission quotas") and therefore need an LLM to both propose the theme and
judge which resolutions belong to it.

Letting an LLM invent *and* grade its own ground truth is a materially
different risk than the deterministic shapes: a wrong label here doesn't just
add noise, it can flip which chunker looks better (see
docs/chunker-embedder-comparison-log.md) by rewarding whichever retriever
agrees with the labeling model instead of whichever retriever finds
genuinely relevant text. Four things keep labels checkable rather than just
plausible (advisor guidance, 2026-07-17 session):

1. **Closed window** -- one council *session* (ครั้งที่ N ปี YYYY) is the
   candidate set for every theme proposed from it. A theme is scoped to that
   session's business ("...ในการประชุมครั้งนี้"), which makes "the relevant
   set is complete" true by construction -- there is no missing-but-relevant
   document outside the window to worry about, only false positives inside
   it.
2. **Grounded spans** -- every doc the model marks relevant must come with a
   verbatim quote from that doc's own text. `_verify_span` checks the quote
   actually appears (whitespace-normalized) before the label survives; an
   unverified label is dropped, never trusted. Same spirit as the
   re-OCR-diff verification in reocr_adjudicate.py.
3. **Body-based labeling** -- stage B reads full resolution bodies, not just
   titles, so the label is at least as informed as what a retriever is being
   asked to find. Title-only matching would just be lexical search running
   twice.
4. **Small themes** -- faculty_adjunct_aggregate already showed a large flat
   relevant set caps recall@10 well below 1.0 by construction (some sets
   have ~30 ids, k=10 can't recall past 10/30). A theme with a large member
   count is still recorded, but flagged in the report rather than treated as
   a clean win.

Two-stage design keeps LLM cost proportional to what's actually promising,
not to session size (some sessions run 60+ agenda items, most of them
routine and unrelated to each other):

- **Stage A (cheap, titles only)**: one call per session lists every
  event's title and asks the model to propose 0-3 candidate cross-cutting
  themes, each with a natural Thai query phrasing and a rough (recall-
  oriented, over-inclusion expected) list of which titles plausibly belong.
  Explicitly told NOT to propose single-entity themes -- those are already
  covered deterministically.
- **Stage B (bodies, only for shortlisted docs)**: for each theme, send the
  full body text of only the docs Stage A shortlisted for it, and ask for a
  true/false relevance call per doc plus a grounding span when true.
  Distinct call per theme (not batched across themes) so one theme's context
  doesn't bias another's judgment.

ADR-0004 curriculum-bundle splits (`__1.md`, `__2.md`, ...) share one
underlying meeting item and duplicate the same preamble text verbatim into
every piece -- collapsed to one representative per _event_key (same helper
concept as build_gold_candidates.py) before either stage, so the LLM judges
each real event once, not once per split piece. relevant_resolution_ids in
the output still expand back to every member file (every piece is a real
retrievable unit).

Uses the Claude API (model: claude-opus-4-8) via the `anthropic` SDK --
resolved by the user during scoping (2026-07-17): label quality is the whole
point of this task and call volume is low, so cost is secondary to
correctness here, unlike the OCR-corruption scan in llm_ocr_scan.py which
deliberately used free local Ollama models for a much higher call volume.

Read-only: never edits or deletes a corpus file. Output goes to
academic_resolutions/llm_thematic_scan/ (gitignored, same convention as
llm_ocr_scan/ and entity_tags/) as a resumable per-session JSON, so a crash
mid-run doesn't lose progress and a re-run skips sessions already done.

Run from the repo root, e.g.:

    .venv/Scripts/python.exe tools/corpus_prep/llm_thematic_bootstrap.py --session 2564 7
    .venv/Scripts/python.exe tools/corpus_prep/llm_thematic_bootstrap.py --sample 5 --seed 1
    .venv/Scripts/python.exe tools/corpus_prep/llm_thematic_bootstrap.py --all
    .venv/Scripts/python.exe tools/corpus_prep/llm_thematic_bootstrap.py --report
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
OUT_DIR = CORPUS_ROOT / "llm_thematic_scan"
REPORT_FILE = OUT_DIR / "thematic_candidates_report.md"

sys.path.insert(0, str(REPO / "src"))
from rag_lab.loaders.common import make_resolution_id, parse_path  # noqa: E402

load_dotenv(REPO / ".env")

MODEL = "claude-opus-4-8"

_EVENT_SEP = " — "
_SPLIT_PIECE = re.compile(r"__\d+$")


def _event_key(resolution_id: str) -> str:
    """Collapse an ADR-0004 split-bundle piece's resolution_id back to its
    shared original meeting-item identity -- see build_gold_candidates.py's
    _event_key for the corpus-wide verification this relies on (707/707
    split files use the same " — " separator, dense 1..N numbering)."""
    return resolution_id.split(_EVENT_SEP, 1)[0]


def session_dirs() -> list[Path]:
    return sorted(
        d
        for year_dir in CORPUS_ROOT.iterdir()
        if year_dir.is_dir() and re.fullmatch(r"\d{4}", year_dir.name)
        for d in year_dir.iterdir()
        if d.is_dir()
    )


def session_key(session_dir: Path) -> str:
    year = session_dir.parent.name
    session_match = re.search(r"(\d+s?)", session_dir.name)
    session = session_match.group(1) if session_match else session_dir.name
    return f"{year}__{session}"


class Event:
    __slots__ = ("event_key", "member_ids", "title", "text")

    def __init__(self, event_key: str, member_ids: list[str], title: str, text: str):
        self.event_key = event_key
        self.member_ids = member_ids
        self.title = title
        self.text = text


def session_events(session_dir: Path) -> list[Event]:
    """One Event per distinct original meeting item in the session, using the
    fullest/whole-document text as the representative body (same
    prefer-whole-over-split-piece convention as llm_ocr_scan.bak_files())."""
    files = sorted(
        f for f in session_dir.glob("*.md") if not f.name.endswith(".dup")
    )
    by_event: dict[str, list[Path]] = {}
    for f in files:
        year, session, title = parse_path(str(f))
        rid = make_resolution_id(str(f), year, session, title)
        by_event.setdefault(_event_key(rid), []).append(f)

    events = []
    for ekey, member_files in by_event.items():
        member_files.sort(key=lambda f: (bool(_SPLIT_PIECE.search(f.stem)), f.name))
        rep = member_files[0]
        try:
            text = rep.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = rep.read_text(encoding="utf-8-sig")
        member_ids = []
        for f in member_files:
            year, session, title = parse_path(str(f))
            member_ids.append(make_resolution_id(str(f), year, session, title))
        _, _, rep_title = parse_path(str(rep))
        events.append(Event(ekey, sorted(member_ids), rep_title, text))
    events.sort(key=lambda e: e.event_key)
    return events


STAGE_A_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "query": {"type": "string"},
                    "candidate_indices": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["theme", "query", "candidate_indices"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["themes"],
    "additionalProperties": False,
}

STAGE_A_PROMPT = """คุณกำลังช่วยสร้างชุดคำถามทดสอบระบบค้นคืนเอกสาร (retrieval eval) จากมติที่ประชุม\
สภาวิชาการ มหาวิทยาลัยแห่งหนึ่ง เอกสารด้านล่างคือรายชื่อเรื่อง (title) ของทุกวาระใน\
"การประชุมครั้งนี้ครั้งเดียว" (ปิด ไม่รวมการประชุมครั้งอื่น)

หน้าที่ของคุณ: เสนอ "ธีมข้ามวาระ" (cross-cutting theme) 0-3 ธีม ที่เชื่อมโยงวาระ\
หลายๆ เรื่องในการประชุมครั้งนี้เข้าด้วยกัน (เช่น เรื่องที่เกี่ยวกับการเปลี่ยนแปลง\
ค่าธรรมเนียม, การรับนักศึกษาต่างชาติ, การเทียบโอนหน่วยกิต, มาตรฐานหลักสูตรแบบใดแบบหนึ่ง)

**ห้าม** เสนอธีมที่ผูกกับชื่อบุคคลหรือชื่อหลักสูตรเดียว (เช่น "ประวัติ อ. X" หรือ\
"การเปลี่ยนแปลงหลักสูตร Y") -- กรณีเหล่านั้นมี ground truth ที่แน่นอนอยู่แล้วจาก\
dictionary ไม่ต้องการให้ LLM ช่วยตัดสิน

**ห้าม** เสนอธีมที่เป็นแค่ "หมวดวาระประจำ" ที่เกิดซ้ำแทบทุกการประชุม แม้ชื่อเรื่องจะดู\
เชื่อมโยงกันหลายคณะก็ตาม เพราะ query แบบนี้มักมีคำในตัว query ตรงกับ title ของทุก\
เอกสารที่ relevant แบบคำต่อคำ (เช่น query มีคำว่า "ปรับปรุงหลักสูตร" และทุก title ก็มี\
คำว่า "ปรับปรุงหลักสูตร") ทำให้ retriever ไหนก็เจอเท่ากันหมด ไม่มีอำนาจแยกแยะระบบค้นคืน\
เลย ตัวอย่างหมวดที่ห้ามเสนอซ้ำ (มี dictionary/ground truth ครอบคลุมแล้วหรือเจอซ้ำจน\
ไม่มีประโยชน์ทาง eval):
- การปรับปรุงหลักสูตร / ปรับปรุงแก้ไขหลักสูตร (ทั้งกรณีกระทบและไม่กระทบโครงสร้าง)
- การแต่งตั้งอาจารย์บัณฑิตประจำ และ/หรือ อาจารย์บัณฑิตพิเศษ
- อาจารย์พิเศษสอนเกินร้อยละ 50 (ครอบคลุมด้วย dictionary แยกต่างหากอยู่แล้ว)
- การจัดกลุ่มตาม "คณะเดียวกันเสนอ" ที่ไม่มีความเชื่อมโยงเชิงเนื้อหาอื่นใด (เช่น\
"เรื่องคณะวิศวกรรมศาสตร์" ที่รวมวาระคนละประเด็นเข้าด้วยกันเพราะบังเอิญเป็นคณะเดียวกัน)

เป้าหมายคือธีมที่ **เฉพาะเจาะจงกับการประชุมครั้งนี้จริงๆ** ไม่ใช่สิ่งที่เกิดซ้ำทุกครั้ง\
เช่น การเปลี่ยนแปลงค่าธรรมเนียม, การงด/ชะลอ/เปลี่ยนแปลงจำนวนรับนักศึกษา, การเทียบโอน\
หน่วยกิตหรือผลสอบ, โครงการหลักสูตรใหม่, หรือเรื่องเฉพาะกิจอื่นที่ไม่ใช่หมวดประจำข้างต้น\
-- ถ้าการประชุมนี้ไม่มีอะไรนอกเหนือจากหมวดประจำ ให้ตอบ themes เป็น array ว่างดีกว่าฝืนเสนอ

ธีมที่ดีต้องเชื่อมโยงวาระอย่างน้อย 2 เรื่องขึ้นไปในการประชุมนี้ (ดูจากชื่อเรื่องเท่านั้น\
ตอนนี้ -- ยังไม่ต้องอ่านเนื้อหาเต็ม การเดารวมเกินจำเป็น (over-inclusion) ในขั้นตอนนี้\
ไม่เป็นปัญหา เดี๋ยวจะมีขั้นตอนตรวจสอบเนื้อหาเต็มอีกครั้ง)

สำหรับแต่ละธีม ให้เขียน:
- theme: ชื่อธีมสั้นๆ ภาษาไทย
- query: คำถามภาษาไทยที่ผู้ใช้จริงอาจถามเกี่ยวกับธีมนี้ในการประชุมครั้งนี้ \
(เช่น "ในการประชุมครั้งนี้ มีการพิจารณาเรื่องค่าธรรมเนียมการศึกษาของหลักสูตรใดบ้าง")
- candidate_indices: เลข index ของวาระ (จากรายการด้านล่าง) ที่น่าจะเกี่ยวข้องกับธีมนี้

ถ้าไม่มีธีมข้ามวาระที่สมเหตุสมผลในการประชุมนี้ ให้ตอบ themes เป็น array ว่าง

--- รายชื่อวาระ (index: title) ---
%%TITLES%%
--- จบรายชื่อวาระ ---

ตอบเป็น JSON ตาม schema เท่านั้น"""


STAGE_B_SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "relevant": {"type": "boolean"},
                    "span": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "relevant", "span", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["judgments"],
    "additionalProperties": False,
}

STAGE_B_PROMPT = """ธีม: %%THEME%%
คำถามตัวอย่าง: %%QUERY%%

ด้านล่างคือเนื้อหาเต็มของวาระที่อาจเกี่ยวข้องกับธีมนี้ (จากการประชุมครั้งเดียวกัน) \
สำหรับแต่ละวาระ ให้ตัดสินว่าเนื้อหาจริงเกี่ยวข้องกับธีมนี้หรือไม่ (ไม่ใช่แค่ชื่อเรื่อง \
คล้ายกัน -- ต้องมีเนื้อหาจริงที่สนับสนุนธีม)

ถ้า relevant=true ต้องระบุ span เป็นข้อความคำต่อคำ (verbatim, <=200 ตัวอักษร) \
ที่คัดลอกมาจากเนื้อหาวาระนั้นโดยตรง เพื่อยืนยันความเกี่ยวข้อง -- ถ้าคัดลอกไม่ตรงตัวอักษร \
คำตอบนี้จะถูกทิ้งไปโดยอัตโนมัติ ถ้า relevant=false ให้ span เป็นสตริงว่าง

ตอบให้ครบทุก index ที่ปรากฏด้านล่าง

--- วาระ ---
%%DOCS%%
--- จบวาระ ---

ตอบเป็น JSON ตาม schema เท่านั้น"""


def _client() -> Anthropic:
    return Anthropic()


def _call(client: Anthropic, prompt: str, schema: dict, retries: int = 3) -> dict:
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=8000,
                thinking={"type": "adaptive"},
                output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": prompt}],
            )
            text = next(b.text for b in response.content if b.type == "text")
            return {"data": json.loads(text), "usage": response.usage.model_dump(), "error": None}
        except Exception as e:  # rate limit, malformed JSON, transient API error
            last_err = str(e)
            time.sleep(2 * attempt)
    return {"data": None, "usage": None, "error": last_err}


_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _verify_span(span: str, body: str) -> bool:
    span = span.strip()
    if not span:
        return False
    return _normalize(span) in _normalize(body)


def run_session(client: Anthropic, session_dir: Path) -> dict:
    events = session_events(session_dir)
    if len(events) < 2:
        return {"session": session_key(session_dir), "events": len(events), "themes": [], "usage": []}

    usage_log = []
    titles_block = "\n".join(f"{i}: {e.title}" for i, e in enumerate(events))
    stage_a = _call(client, STAGE_A_PROMPT.replace("%%TITLES%%", titles_block), STAGE_A_SCHEMA)
    if stage_a["usage"]:
        usage_log.append({"stage": "A", **stage_a["usage"]})
    if stage_a["error"] or not stage_a["data"]:
        return {
            "session": session_key(session_dir),
            "events": len(events),
            "themes": [],
            "usage": usage_log,
            "error": stage_a["error"],
        }

    themes_out = []
    for theme in stage_a["data"]["themes"]:
        indices = sorted({i for i in theme["candidate_indices"] if 0 <= i < len(events)})
        if len(indices) < 2:
            continue
        docs_block = "\n\n".join(
            f"[{i}] {events[i].title}\n{events[i].text}" for i in indices
        )
        prompt = (
            STAGE_B_PROMPT.replace("%%THEME%%", theme["theme"])
            .replace("%%QUERY%%", theme["query"])
            .replace("%%DOCS%%", docs_block)
        )
        stage_b = _call(client, prompt, STAGE_B_SCHEMA)
        if stage_b["usage"]:
            usage_log.append({"stage": "B", "theme": theme["theme"], **stage_b["usage"]})
        if stage_b["error"] or not stage_b["data"]:
            continue

        verified_events = []
        for j in stage_b["data"]["judgments"]:
            idx = j["index"]
            if not (0 <= idx < len(events)) or not j["relevant"]:
                continue
            ev = events[idx]
            if not _verify_span(j["span"], ev.text):
                continue
            verified_events.append({"event_key": ev.event_key, "member_ids": ev.member_ids, "span": j["span"], "reason": j["reason"]})

        if len(verified_events) < 2:
            continue
        relevant_resolution_ids = sorted(
            rid for ev in verified_events for rid in ev["member_ids"]
        )
        themes_out.append(
            {
                "theme": theme["theme"],
                "query": theme["query"],
                "relevant_resolution_ids": relevant_resolution_ids,
                "event_count": len(verified_events),
                "grounded_events": verified_events,
                "candidate_count": len(indices),
            }
        )

    return {
        "session": session_key(session_dir),
        "events": len(events),
        "themes": themes_out,
        "usage": usage_log,
    }


def out_path(session_dir: Path) -> Path:
    return OUT_DIR / f"{session_key(session_dir)}.json"


def run_sessions(session_list: list[Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = _client()
    for i, sdir in enumerate(session_list):
        path = out_path(sdir)
        if path.exists():
            print(f"  ({i+1}/{len(session_list)}) {session_key(sdir)} -- already done, skipping")
            continue
        print(f"  ({i+1}/{len(session_list)}) {session_key(sdir)} -- running", end="", flush=True)
        result = run_session(client, sdir)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        n_themes = len(result["themes"])
        total_in = sum(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) for u in result["usage"])
        total_out = sum(u.get("output_tokens", 0) for u in result["usage"])
        print(f" -- {n_themes} verified theme(s), {total_in} in / {total_out} out tokens")


def write_report() -> None:
    results = []
    for f in sorted(OUT_DIR.glob("*.json")):
        results.append(json.loads(f.read_text(encoding="utf-8")))

    total_themes = sum(len(r["themes"]) for r in results)
    total_in = sum(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) for r in results for u in r["usage"])
    total_out = sum(u.get("output_tokens", 0) for r in results for u in r["usage"])

    lines = [
        "# LLM thematic-query bootstrap candidates",
        "",
        "Auto-generated by `tools/corpus_prep/llm_thematic_bootstrap.py`. Read-only. "
        "Every relevant_resolution_id here is backed by a verbatim, deterministically-"
        "verified grounding span -- see the module docstring for the four checkability "
        "guarantees. This is still a CANDIDATE pool, not a finished gold set: read the "
        "grounded spans before trusting a theme, especially ones with a large "
        "event_count (large flat relevant sets cap recall@10 well below 1.0 by "
        "construction, same finding as faculty_adjunct_aggregate).",
        "",
        f"- Sessions scanned: {len(results)}",
        f"- Verified themes: {total_themes}",
        f"- Total tokens: {total_in} in (incl. cache reads) / {total_out} out",
        "",
        "## Themes, largest event_count first",
        "",
    ]
    all_themes = [
        (r["session"], t) for r in results for t in r["themes"]
    ]
    all_themes.sort(key=lambda st: -st[1]["event_count"])
    for session, t in all_themes:
        lines.append(f"### [{session}] {t['theme']} (event_count={t['event_count']})")
        lines.append(f"query: {t['query']}")
        lines.append("")
        for ev in t["grounded_events"]:
            lines.append(f"- `{ev['event_key']}` -- {ev['reason']}")
            lines.append(f"  > {ev['span']}")
        lines.append("")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {REPORT_FILE} ({total_themes} verified themes across {len(results)} sessions)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", nargs=2, metavar=("YEAR", "SESSION"), help="Run one session, e.g. --session 2564 7")
    parser.add_argument("--sample", type=int, default=0, help="Run N random sessions")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--all", action="store_true", help="Run every session in the corpus (resumable)")
    parser.add_argument("--report", action="store_true", help="(Re)generate the markdown report only")
    args = parser.parse_args()

    if args.report and not args.session and not args.sample and not args.all:
        write_report()
        return

    all_sessions = session_dirs()

    if args.session:
        year, session = args.session
        matches = [
            d for d in all_sessions
            if d.parent.name == year and re.fullmatch(rf"ครั้งที่ {re.escape(session)}", d.name)
        ]
        if not matches:
            raise SystemExit(f"no session folder matches {year} / ครั้งที่ {session}")
        run_sessions(matches)
    elif args.sample:
        rng = random.Random(args.seed)
        chosen = all_sessions[:]
        rng.shuffle(chosen)
        run_sessions(chosen[: args.sample])
    elif args.all:
        run_sessions(all_sessions)
    else:
        parser.error("pass --session YEAR SESSION, --sample N, --all, or --report")

    write_report()


if __name__ == "__main__":
    main()
