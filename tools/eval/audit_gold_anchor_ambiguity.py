"""Audit: does a gold query's anchor point at the same thing its qrels do?

Motivated by the one query in the whole set that scores 0.000 even when the
top-10 of all 36 combos are unioned (`data/results/oracle_union_ceiling.md`
section 4): `รายวิชา CONTROL SYSTEMS`. The Gold set also holds `รายวิชา CONTROL
SYSTEM`, one character apart, and their qrels are *disjoint*. CLAUDE.md's
instruction was to detect the shape rather than hard-code the pair, which is
what this script does -- and doing so turned the finding from "one unlucky
pair" into a structural statement about one entity type.

**The shape.** `build_gold_candidates.py` derives each query's relevant set
from an entity key, and the query text names an entity. For three of the four
entity types those are the *same key*:

    program  qrels = resolutions whose TITLE contains the canonical string
             query = that same canonical string          -> same key
    person   qrels = files where the canonical identity appears
             query = that same canonical full name       -> same key
    faculty  qrels = adjunct filings tagged with the faculty
             query = that same faculty name              -> same key
    course   qrels = files tagged with the 8-digit CODE
             query = the course's NAME                   -> DIFFERENT KEY

So `course` is the only type where the evidence the query supplies (a name)
and the evidence the qrels are built from (a code) can disagree, and the
exposure is exactly the 33 course queries of 106 -- a denominator, not an
estimate. `course_loader.match_courses` tags by code; the name is attached to
the code only in the document that declares the course. A different course
whose name *contains* this one's (`DIGITAL CONTROL SYSTEMS` contains `CONTROL
SYSTEMS`) puts the query's own anchor text into a document the qrels call
irrelevant.

**What is measured, per course query.**
  gold          |qrels| -- files carrying the course's code
  names it      corpus files whose raw text contains the course name as a
                standalone phrase (the same immediate-neighbour boundary rule
                `match_courses_by_name` uses, because regex \\b never fires at
                a Thai/Latin seam)
  anchor prec.  gold / names-it -- the share of the evidence the query
                supplies that the qrels actually credit
  name-only E   min(1, K/names-it) -- expected recall@10 of a system that can
                see the name and nothing else, i.e. one that ranks the naming
                files in an order uncorrelated with the code. This is the
                honest ceiling for a bag-of-words retriever on this query.
  qrels ceiling min(1, K/gold) -- the ceiling the project already publishes,
                which knows nothing about the name

The gap between those last two columns is the defect, quantified. Where they
agree the query is fine; where the name-only ceiling collapses, the query is
asking for something its own text cannot identify.

**Flagging rule**, deliberately a statement rather than a tuned threshold: a
query is flagged when anchor precision < 0.5, i.e. *most* of the documents
showing the query's own anchor text are judged irrelevant. The full
distribution is printed so the cut is not load-bearing -- read the table, not
the flag count.

Section 4 prices the obvious repair (drop the flagged queries) against the
persisted results, because "size the blast radius before deciding" is this
project's rule and a gold-set edit moves every published number in it.

Read-only: consumes the corpus, the gold set, `courses.json`,
`courses_by_file.json` and persisted hybrid results. Runs no retrieval and
writes nothing but its own report.

Run with:
    .venv/Scripts/python.exe tools/eval/audit_gold_anchor_ambiguity.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from embedder_matrix_9way import _EXCLUDED_COMBO_DIRS  # noqa: E402
from rag_lab.loaders.common import make_resolution_id, parse_path  # noqa: E402
from rag_lab.loaders.course_loader import match_courses_by_name  # noqa: E402

CORPUS_ROOT = REPO / "academic_resolutions"
TAGS_DIR = CORPUS_ROOT / "entity_tags"
DICT_PATH = REPO / "data" / "entity_dictionaries" / "courses.json"
INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
HYB = REPO / "data" / "results" / "gold_hybrid_73det"
HYB_REPORT = REPO / "data" / "results" / "gold_hybrid_73det_report.md"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
OUT = REPO / "data" / "results" / "gold_anchor_ambiguity.md"

K = 10
FLAG_ANCHOR_PRECISION = 0.5
COURSE_TEMPLATE = re.compile(r"รายวิชา\s+(.+?)\s+ถูกกล่าวถึง")

# Same rule as course_loader.match_courses_by_name: only the immediate
# neighbours are inspected, never regex \b.
_ALNUM = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
_WS = re.compile(r"\s+")


def contains_phrase(haystack: str, needle: str) -> bool:
    """Case-insensitive containment of `needle` as a standalone phrase.

    `haystack` must already be whitespace-collapsed (the caller collapses each
    document once rather than once per name -- 33 names x 2,853 documents makes
    that the difference between seconds and minutes). The needle is collapsed
    here.

    Why collapse at all: OCR'd minutes wrap a long course name across a line,
    and matching raw text calls 3 genuine mentions absent -- measured, e.g.
    `ENGLISH FOR ARCHITECTURAL PRESENTATION` gains a naming document and loses
    its only apparently-silent one. Collapsing is the conservative direction,
    since an inflated `gold_not_naming` would invent a second failure mechanism
    that isn't there.
    """
    needle = _WS.sub(" ", needle)
    for m in re.finditer(re.escape(needle), haystack, re.IGNORECASE):
        before = haystack[m.start() - 1] if m.start() > 0 else ""
        after = haystack[m.end()] if m.end() < len(haystack) else ""
        if before in _ALNUM or after in _ALNUM:
            continue
        return True
    return False


def load_ranked(results_dir: Path, arm: str, qrels: dict) -> dict[str, dict[str, list[str]]]:
    """combo prefix -> query -> ranked resolution_ids (best first). Same reader
    as oracle_union_ceiling.py, so S3 below can anchor on published numbers."""
    ranked: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
    for f in results_dir.glob(f"*__{arm}__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        q = d["query"]
        if q not in qrels:
            continue
        prefix = "__".join(f.stem.split("__")[:4])
        ranked[prefix][q] = [
            r["resolution_id"] for r in sorted(d["results"], key=lambda r: r["rank"])
        ]
    return ranked


def main() -> int:
    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: set(d["relevant_resolution_ids"]) for d in raw}
    etype = {d["query"]: d.get("entity_type", "?") for d in raw}
    entity = {d["query"]: d.get("entity", "") for d in raw}

    courses = json.loads(DICT_PATH.read_text(encoding="utf-8"))
    name_to_code: dict[str, str] = {c["canonical"]: c["code"] for c in courses}
    code_to_names: dict[str, list[str]] = collections.defaultdict(list)
    for c in courses:
        code_to_names[c["code"]].append(c["canonical"])
    by_file = json.loads((TAGS_DIR / "courses_by_file.json").read_text(encoding="utf-8"))

    course_qs = [q for q in queries if etype[q] == "course"]

    checks: list[tuple[str, bool, str]] = []

    # S0 -- duplicate query strings collapse silently in `qrels`; the thematic
    # set really did hold 5 of them once.
    checks.append((
        "S0 gold query strings unique",
        len(set(queries)) == len(raw),
        f"{len(set(queries))} unique of {len(raw)} entries",
    ))

    # S1 -- every course query must expose its name to one template, or a
    # query that stopped matching would leave the denominator quietly.
    parsed = {q: m.group(1) for q in course_qs if (m := COURSE_TEMPLATE.search(q))}
    checks.append((
        "S1 every course query exposes its name to the same template",
        len(parsed) == len(course_qs),
        f"{len(parsed)} of {len(course_qs)} parse",
    ))

    # S2 -- the parsed name must be the entry's own `entity`, and must be a
    # known dictionary name. Otherwise the whole report measures the wrong
    # string.
    bad_name = [q for q in parsed if parsed[q] != entity[q] or entity[q] not in name_to_code]
    checks.append((
        "S2 parsed name == entry's `entity` and is a courses.json name",
        not bad_name,
        f"{len(parsed) - len(bad_name)} of {len(parsed)} agree",
    ))

    # S3 -- THE LOAD-BEARING PREMISE. This whole report rests on "course qrels
    # are code-derived". Verify it rather than assume it: the qrels must equal
    # the resolution_ids of exactly the files courses_by_file.json tags with
    # that code.
    rid_cache: dict[str, str] = {}

    def rid_for(relpath: str) -> str:
        if relpath not in rid_cache:
            full = str(CORPUS_ROOT / relpath)
            y, s, t = parse_path(full)
            rid_cache[relpath] = make_resolution_id(full, y, s, t)
        return rid_cache[relpath]

    code_rids: dict[str, set[str]] = collections.defaultdict(set)
    for relpath, codes in by_file.items():
        r = rid_for(relpath)
        for code in codes:
            code_rids[code].add(r)

    mismatched = [
        q for q in parsed
        if code_rids.get(name_to_code[entity[q]], set()) != qrels[q]
    ]
    checks.append((
        "S3 course qrels reproduce exactly from the code tags (the premise)",
        not mismatched,
        f"{len(parsed) - len(mismatched)} of {len(parsed)} reproduce",
    ))

    # S4 -- the mirror-image premise. The query must at least identify its own
    # course, or a low score would mean "the query never named the thing" and
    # this report would be measuring nothing.
    #
    # NOTE: the *stronger* form -- resolves to its own code and nothing else --
    # was written first and FAILED on 3 of 33, which is a finding rather than a
    # broken premise, so it moved into section 3b. Keeping it as a gate would
    # have suppressed exactly the result the script exists to surface.
    query_matches = {q: match_courses_by_name(q) for q in parsed}
    query_side_bad = [q for q in parsed if name_to_code[entity[q]] not in query_matches[q]]
    checks.append((
        "S4 match_courses_by_name finds each course query's own code",
        not query_side_bad,
        f"{len(parsed) - len(query_side_bad)} of {len(parsed)} identify their own course",
    ))

    # S5 -- combo membership derived from the index dirs that exist, then
    # cross-checked against the retired set (either half alone goes stale).
    live = {d.name for d in INDEX_DIR.iterdir() if d.is_dir()}
    ranked_hyb = load_ranked(HYB, "hybrid", qrels)
    with_results = set(ranked_hyb)
    leftovers = with_results - live
    combos = sorted(with_results & live)
    checks.append((
        "S5 result combos with no index dir are exactly the known retired set",
        leftovers == _EXCLUDED_COMBO_DIRS,
        f"{len(leftovers)} leftover, {len(combos)} kept of {len(with_results)} with results",
    ))

    # S6 -- the anchor. Section 4 subtracts queries from a published average,
    # so this harness must first reproduce that average exactly.
    published = {}
    for line in HYB_REPORT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (plain__\S+)__hybrid \| ([0-9.]+) \|", line)
        if m:
            published[m.group(1)] = float(m.group(2))
    mism = []
    for c in combos:
        got = float(np.mean([
            len(set(ranked_hyb[c][q][:K]) & qrels[q]) / len(qrels[q]) for q in queries
        ]))
        if c not in published or abs(got - published[c]) > 5e-5:
            mism.append(c)
    checks.append((
        "S6 every kept combo reproduces its published recall@10 to 4dp",
        not mism,
        f"{len(combos) - len(mism)} of {len(combos)} reproduce",
    ))

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
    if not all(ok for _, ok, _ in checks):
        print("\nself-check failed; refusing to publish numbers", file=sys.stderr)
        return 1

    # ---- corpus scan -------------------------------------------------------
    print(f"scanning {len(by_file)} corpus files for {len(parsed)} course names ...")
    texts = {
        relpath: _WS.sub(
            " ", (CORPUS_ROOT / relpath).read_text(encoding="utf-8", errors="replace")
        )
        for relpath in by_file
    }

    # every gold resolution_id -> which gold entities claim it
    claimed_by: dict[str, set[str]] = collections.defaultdict(set)
    for d in raw:
        for rid in d["relevant_resolution_ids"]:
            claimed_by[rid].add(d.get("entity", ""))

    rows = []
    for q in course_qs:
        name = entity[q]
        gold_rids = qrels[q]
        naming = {rid_for(rp) for rp, t in texts.items() if contains_phrase(t, name)}
        stranger = naming - gold_rids
        contradicting = {r for r in stranger if r in claimed_by}
        n_names = len(naming)
        anchor_prec = len(gold_rids & naming) / n_names if n_names else 0.0
        name_only = min(1.0, K / n_names) if n_names else 0.0
        qrels_ceiling = min(1.0, K / len(gold_rids))
        union = set()
        for c in combos:
            union |= set(ranked_hyb[c][q][:K])
        union_recall = len(union & gold_rids) / len(gold_rids)
        rows.append({
            "name": name,
            "code": name_to_code[name],
            "gold": len(gold_rids),
            "names": n_names,
            "stranger": len(stranger),
            "silent": len(gold_rids - naming),
            "contradicting": len(contradicting),
            "anchor_prec": anchor_prec,
            "name_only": name_only,
            "qrels_ceiling": qrels_ceiling,
            "gap": name_only - qrels_ceiling,
            "union": union_recall,
        })

    rows.sort(key=lambda r: r["anchor_prec"])
    flagged = [r for r in rows if r["anchor_prec"] < FLAG_ANCHOR_PRECISION]

    # sub-phrase relation among dictionary names
    subphrase = []
    for r in rows:
        longer = [
            c["canonical"] for c in courses
            if c["canonical"] != r["name"]
            and len(c["canonical"]) > len(r["name"])
            and c["code"] != r["code"]
            and contains_phrase(_WS.sub(" ", c["canonical"]), r["name"])
        ]
        if longer:
            subphrase.append((r["name"], longer))

    # ---- section 4: price the repair --------------------------------------
    def macro(subset: list[str], combo: str) -> float:
        return float(np.mean([
            len(set(ranked_hyb[combo][q][:K]) & qrels[q]) / len(qrels[q]) for q in subset
        ]))

    flagged_names = {r["name"] for r in flagged}
    drop_all = [q for q in queries if entity[q] not in flagged_names]
    worst = min(rows, key=lambda r: (r["union"], r["anchor_prec"]))
    drop_worst = [q for q in queries if entity[q] != worst["name"]]

    scenarios = [
        ("ปัจจุบัน (106 คำถาม)", queries),
        (f"ตัดคำถามที่ union = {worst['union']:.3f} ออก 1 ข้อ ({len(drop_worst)} คำถาม)", drop_worst),
        (f"ตัดคำถามที่ถูก flag ทั้งหมด ({len(drop_all)} คำถาม)", drop_all),
    ]
    per_scenario = {
        label: {c: macro(subset, c) for c in combos} for label, subset in scenarios
    }
    base_label = scenarios[0][0]

    # ---- render ------------------------------------------------------------
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# qrels ยึดกับกุญแจเดียวกับที่คำถามให้มาหรือเปล่า")
    w()
    w("Generated by `tools/eval/audit_gold_anchor_ambiguity.py` · "
      f"gold {len(raw)} คำถาม · corpus {len(by_file)} ไฟล์ · k = {K}")
    w()
    w("**คำถามที่ตอบ** — `รายวิชา CONTROL SYSTEMS` เป็นคำถามเดียวในชุดที่ได้ **0.000** "
      "แม้จะรวม top-10 ของทุก combo เข้าด้วยกัน (`oracle_union_ceiling.md` หัวข้อ 4) "
      "รายงานนี้ไม่ได้ไล่ตามคู่นั้นโดยตรง แต่ไล่ตาม *รูปทรง* ของมัน")
    w()

    w("## 1. ชนิดคำถามไหนเสี่ยงบ้าง — และเสี่ยงกี่ข้อ")
    w()
    w("`build_gold_candidates.py` สร้าง qrels จากกุญแจของ entity ส่วนตัวคำถามก็เอ่ยชื่อ entity "
      "สามในสี่ชนิดใช้**กุญแจเดียวกัน**ทั้งสองฝั่ง เหลือชนิดเดียวที่ไม่ใช่")
    w()
    w("| ชนิด | qrels ยึดกับ | คำถามให้หลักฐานเป็น | กุญแจเดียวกัน | จำนวนคำถาม |")
    w("|---|---|---|---|---|")
    counts = collections.Counter(etype[q] for q in queries)
    w(f"| program | ชื่อเต็มในหัวเรื่องมติ | ชื่อเต็มนั้น | ใช่ | {counts['program']} |")
    w(f"| person | ตัวตน canonical ในตัวบท | ชื่อ-สกุลนั้น | ใช่ | {counts['person']} |")
    w(f"| faculty_adjunct_aggregate | คณะที่ถูก tag | ชื่อคณะนั้น | ใช่ | "
      f"{counts['faculty_adjunct_aggregate']} |")
    w(f"| **course** | **รหัสวิชา 8 หลัก** | **ชื่อวิชา** | **ไม่ใช่** | **{counts['course']}** |")
    w()
    w(f"จึงมี **{counts['course']} จาก {len(queries)}** คำถามที่เปิดรับความผิดพลาดชนิดนี้ได้เลย "
      "และอีก "
      f"{len(queries) - counts['course']} ข้อ**ปลอดโดยโครงสร้าง** ไม่ใช่เพราะบังเอิญสะอาด — "
      "S3/S4 ด้านล่างตรวจข้อสมมติทั้งสองข้างนี้จริง ไม่ได้เชื่อเอา")
    w()

    w("## 2. หลักฐานที่คำถามให้มา เทียบกับหลักฐานที่ qrels นับ")
    w()
    w("`ชื่อปรากฏ` = ไฟล์ในคลังที่มีชื่อวิชานั้นเป็นวลีเดี่ยว (กฎขอบคำเดียวกับ "
      "`match_courses_by_name`) · `anchor prec.` = `gold / ชื่อปรากฏ` คือสัดส่วนของหลักฐาน "
      "ที่คำถามให้มาแล้ว qrels ยอมนับ · `เพดานเห็นแต่ชื่อ` = `min(1, k/ชื่อปรากฏ)` "
      "คือค่าคาดหวัง recall@10 ของระบบที่เห็นแค่ชื่อ (เรียงไฟล์ที่มีชื่อนั้นแบบไม่รู้รหัส) · "
      "`เพดาน qrels` = `min(1, k/gold)` คือเพดานที่โปรเจกต์นี้ตีพิมพ์อยู่แล้ว ซึ่งไม่รู้จักชื่อเลย")
    w()
    w("**อ่านสองคอลัมน์เพดานเป็นคู่เสมอ ห้ามอ่านเดี่ยว** — `เพดานเห็นแต่ชื่อ` ต่ำได้ด้วยสองสาเหตุ "
      "คือชื่อกำกวม (ตัวปัญหา) หรือแค่มีเอกสารที่เกี่ยวข้องเยอะเกิน 10 ฉบับ (ไม่ใช่ปัญหา — "
      "`เพดาน qrels` ก็ต่ำตามไปด้วย) คอลัมน์ `Δ เพดาน` คือส่วนต่างที่ตัดสาเหตุที่สองทิ้งแล้ว")
    w()
    w("| วิชา | รหัส | gold | ชื่อปรากฏ | gold ที่ไม่เอ่ยชื่อ | ไม่ถูกตัดสิน | อยู่ใน qrels ของข้ออื่น | "
      "anchor prec. | เพดานเห็นแต่ชื่อ | เพดาน qrels | Δ เพดาน | union จริง |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        flag = " ⚠" if r["anchor_prec"] < FLAG_ANCHOR_PRECISION else ""
        w(f"| {r['name']}{flag} | {r['code']} | {r['gold']} | {r['names']} | {r['silent']} | "
          f"{r['stranger']} | {r['contradicting']} | {r['anchor_prec']:.3f} | "
          f"{r['name_only']:.3f} | {r['qrels_ceiling']:.3f} | {r['gap']:+.3f} | "
          f"{r['union']:.3f} |")
    w()
    w(f"**{len(flagged)} จาก {len(rows)}** ข้อมี anchor precision ต่ำกว่า "
      f"{FLAG_ANCHOR_PRECISION:.1f} — คือ*เอกสารส่วนใหญ่*ที่แสดงชื่อวิชาในคำถาม ถูกตัดสินว่าไม่เกี่ยวข้อง")
    w()
    w("ช่องว่างระหว่างสองคอลัมน์เพดานคือตัวปัญหาที่วัดออกมาเป็นตัวเลขได้ ตัวอย่างที่หนักที่สุด:")
    for r in flagged:
        w(f"- **{r['name']}** — เพดาน qrels บอก **{r['qrels_ceiling']:.3f}** "
          f"แต่เพดานของระบบที่เห็นแค่ชื่อคือ **{r['name_only']:.3f}** "
          f"(Δ {r['gap']:+.3f}, union จริง {r['union']:.3f})")
    w()
    w("### กลไกที่แยกกันสองอย่าง อย่ารวมเป็นก้อนเดียว")
    w()
    silent_rows = [r for r in rows if r["silent"] > 0]
    w("- **ชื่อกำกวม** (`ไม่ถูกตัดสิน` สูง) — เอกสารของวิชาอื่นแสดงข้อความชื่อในคำถาม "
      "หลักฐาน*มีอยู่ครบ*ในเอกสารที่ถูกต้อง แต่ต้องแข่งกับเอกสารที่ไม่เกี่ยวอีกจำนวนมาก")
    w(f"- **ชื่อหายไปเงียบ ๆ** (`gold ที่ไม่เอ่ยชื่อ` > 0, พบ {len(silent_rows)} จาก {len(rows)} ข้อ) "
      "— เอกสารที่ qrels บอกว่าเกี่ยวข้อง กลับ*ไม่มี*ข้อความชื่อวิชาอยู่เลย เพราะถูก tag ด้วยรหัส "
      "ระบบที่จับคู่ด้วยคำจึงหาไม่เจอไม่ว่าจะเก่งแค่ไหน คนละปัญหากับข้อบน และตัดคำถามทิ้งก็ไม่ช่วย")
    w()
    zero = min(rows, key=lambda r: r["union"])
    if zero["silent"] == 0:
        w(f"**แก้คำกล่าวอ้างเดิม** — `oracle_union_ceiling.md` และ `CLAUDE.md` เคยเขียนว่า "
          f"`รายวิชา {zero['name']}` \"ตอบไม่ได้*โดยโครงสร้าง*\" (unanswerable by construction) "
          f"**ซึ่งผิด** เอกสาร gold ของมัน **{zero['gold']} จาก {zero['gold']}** ฉบับ "
          f"มีข้อความ `{zero['name']}` อยู่จริง (คอลัมน์ `gold ที่ไม่เอ่ยชื่อ` = 0) "
          f"คำถามนี้จึง**ตอบได้ในหลักการ** — ระบบที่แยกแยะได้ว่า `{zero['name']}` "
          f"ตัวไหนเป็นวิชา {zero['code']} จะหาเจอครบทั้ง {zero['gold']} ฉบับ สิ่งที่มันไม่ใช่คือ "
          "*ตอบได้ด้วยการจับคู่ชื่อเพียงอย่างเดียว* ซึ่งเป็นคำกล่าวที่อ่อนกว่าและเป็นความจริง")
        w()

    w("## 3. รูปทรง: ชื่อวิชาที่เป็นวลีย่อยของชื่อวิชาอื่น")
    w()
    w("`courses.json` กรองชื่อไว้แล้วว่า *ชื่อหนึ่งต้องผูกกับรหัสเดียว* แต่ไม่ได้กรองว่า "
      "*ชื่อหนึ่งต้องไม่เป็นวลีย่อยของชื่ออื่น* — ซึ่งเป็นปัญหาเดียวกันในรูปที่ตัวกรองมองไม่เห็น "
      "เอกสารที่พูดถึงวิชาที่ชื่อยาวกว่า ย่อมมีข้อความของชื่อสั้นอยู่ในตัวด้วยเสมอ")
    w()
    if subphrase:
        w("| ชื่อในชุด gold | เป็นวลีย่อยของชื่อวิชาอื่นกี่ชื่อ | ตัวอย่าง |")
        w("|---|---|---|")
        for name, longer in subphrase:
            w(f"| {name} | {len(longer)} | {', '.join(longer[:2])} |")
    else:
        w("(ไม่พบ)")
    w()
    w(f"**{len(subphrase)} จาก {len(rows)}** ชื่อวิชาในชุด gold เป็นวลีย่อยของชื่อวิชาอื่น "
      "— แต่เป็นวลีย่อยไม่ได้แปลว่าพัง ให้ดูคู่กับ anchor precision ในตารางที่ 2 "
      "(เช่น `INDUSTRIAL AUTOMATION` เป็นวลีย่อยเหมือนกันแต่ anchor precision ยังสูง)")
    w()

    w("### 3b. ทิศทางตรงข้าม: ชื่อวิชา*อื่น*ที่เป็นวลีย่อยของคำถาม")
    w()
    w("ข้อ 3 ดูฝั่งคลังเอกสาร (ชื่อในคำถามไปโผล่ในเอกสารของวิชาอื่น) ส่วนนี้ดูฝั่งตัวคำถามเอง "
      "`match_courses_by_name` ยิงกับตัวคำถาม แล้วได้รหัส**เกิน**มา เพราะชื่อวิชาที่สั้นกว่า "
      "เป็นวลีย่อยของชื่อวิชาในคำถาม — คนละกลไกกับข้อ 3 แต่เป็นความกำกวมชุดเดียวกัน")
    w()
    extras = [
        (entity[q], name_to_code[entity[q]],
         [c for c in query_matches[q] if c != name_to_code[entity[q]]])
        for q in parsed if len(query_matches[q]) > 1
    ]
    if extras:
        w("| คำถามถามถึงวิชา | รหัสของตัวเอง | รหัสที่ติดมาเกิน | ชื่อที่ทำให้ติด |")
        w("|---|---|---|---|")
        for name, own, others in extras:
            names = "; ".join(
                n for c in others for n in code_to_names[c]
                if contains_phrase(_WS.sub(" ", name), n)
            )
            w(f"| {name} | {own} | {', '.join(others)} | `{names}` |")
    else:
        w("(ไม่พบ)")
    w()
    w(f"**{len(extras)} จาก {len(rows)}** ข้อ ตัวคำถามเองชี้ไปยังวิชามากกว่าหนึ่งวิชา "
      "ทั้งหมดมีรหัสของตัวเองติดมาด้วยเสมอ (S4) จึงไม่กระทบ `classify_query` "
      "ซึ่งสนใจแค่ว่า*มี*วิชาไหม ไม่สนว่ากี่วิชา — เส้นทาง `course` ยังคง 33/33 ตามที่ "
      "`tests/test_router.py` ตรึงไว้ ที่กระทบคือ `detect_entities`/`entity_lookup` "
      "ซึ่งอ่านรายการรหัสจริง ๆ")
    w()
    w("ต้นตอที่แก้ได้ตรงที่สุดคือชื่อกว้างเกินไปใน `courses.json` เอง "
      "ตัวกรองตอนสร้างพจนานุกรมบังคับแค่ *ชื่อผูกกับรหัสเดียว* และ *ยาวอย่างน้อย 2 คำ* "
      "ซึ่ง `ENGLISH FOR` ผ่านทั้งสองข้อ **ไม่แก้ในงานนี้** เพราะเป็นการแก้พจนานุกรมที่ "
      "shipped router อ่านอยู่ — บันทึกไว้เป็นข้อค้นพบ")
    w()

    w("## 4. ราคาของการซ่อมด้วยการตัดคำถามทิ้ง")
    w()
    w("การแก้ชุด gold ขยับ**ทุกตัวเลขที่ตีพิมพ์ไปแล้ว** จึงต้องรู้ราคาก่อนตัดสินใจ "
      f"ตารางนี้คิด macro recall@10 ใหม่บน {len(combos)} combo เดิม เปลี่ยนแค่ชุดคำถาม")
    w()
    w("| ทางเลือก | ค่าเฉลี่ยข้าม combo | Δ จากปัจจุบัน | combo ที่ขยับมากที่สุด |")
    w("|---|---|---|---|")
    base_mean = float(np.mean(list(per_scenario[base_label].values())))
    for label, _ in scenarios:
        vals = per_scenario[label]
        mean = float(np.mean(list(vals.values())))
        deltas = {c: vals[c] - per_scenario[base_label][c] for c in combos}
        worst_c = max(deltas, key=lambda c: abs(deltas[c]))
        if label == base_label:
            w(f"| {label} | {mean:.4f} | — | — |")
        else:
            w(f"| {label} | {mean:.4f} | {mean - base_mean:+.4f} | "
              f"`{worst_c}` {deltas[worst_c]:+.4f} |")
    w()
    w("อ่านตาราง: การตัดคำถามที่ตอบไม่ได้ออก **ดัน**ตัวเลขที่ตีพิมพ์ขึ้นทุกตัว "
      "ไม่ใช่แค่ขยับไปมา เพราะคำถามที่ตัดออกคือคำถามที่ทำคะแนนได้ต่ำ "
      "ราคาที่แท้จริงจึงไม่ใช่ขนาดของ Δ แต่คือ**ทุกตารางในโปรเจกต์ต้องรันใหม่และคัดลอกตัวเลขใหม่** "
      "(ไม่ต้องใช้ GPU — ผลลัพธ์ที่ persist ไว้ใช้ได้ต่อ) และคำกล่าวอ้างที่อ้างอิง "
      f"'{len(raw)} คำถาม' ทุกที่ต้องแก้ตาม")
    w()

    w("## self-check")
    w()
    for name, ok, detail in checks:
        w(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    w()

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {OUT}")
    print(f"flagged {len(flagged)} of {len(rows)} course queries; "
          f"{len(subphrase)} names are sub-phrases of another course's name")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
