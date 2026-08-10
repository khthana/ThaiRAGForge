"""Build the *simple* relation graph -- the two edges that need no new model,
no GPU and no new dictionary:

    A   หลักสูตร --belongs_to-->      คณะ      (program -> faculty)
    A'  บุคคล    --affiliated_with--> คณะ      (person  -> faculty)

Scope is deliberately these two only. The other two edges the graph notes
propose -- person->responsible_for->program and person->replaces->person --
need a parser for a document type this corpus renders badly, and are priced at
2-4 days each; they are out of scope here.

WHAT THIS IS AND IS NOT
-----------------------
**No gold query in either query set is multi-hop.** `gold_query_set_73det.yaml`
asks "รายวิชา X" / "หลักสูตร Y" / "อาจารย์ Z" -- one anchor, one hop. So this
graph is a *capability* the current evaluation is structurally unable to score,
and no retrieval gain may be claimed from it. What it can be scored on is its
own accuracy, which is what this script reports.

**Coverage is "of what was found", never "of what exists".** The affiliation
evidence for A' lives in exactly the document type this corpus OCRs worst
(wide rank/affiliation tables), and the A evidence is co-occurrence inside
documents whose text has been re-OCR'd three times. Every denominator printed
here is the entity *dictionary*, which is itself a curated subset. Read the
report as a lower bound.

WHY THE TAGS ARE RECOMPUTED
---------------------------
`academic_resolutions/entity_tags/*_by_file.json` exist and would have saved
the ~20 minute walk, but they are dated 2026-07-17..25 and therefore predate
the person-loader fixes (`a4e250e` bare "อ." rank, `e1523b3` cross-cell split
names), the 2026-07-28 OCR remediation, the 2026-08-08 title repair and the
2026-08-09 `2566/ครั้งที่ 3` re-OCR. Building a graph on them would be the
project's signature failure: two artifacts produced at different times by
different scripts, which never crashes, it just makes a number wrong. Tags are
recomputed from the tested matchers instead, and the raw evidence is cached to
`data/results/relation_graph_raw.json` so `--render` re-derives the graph and
the report without a second walk.

CLASSIFICATION IS THREE-WAY, NOT A SCORE
----------------------------------------
A program with zero faculty votes is `no_evidence` -- its share is *undefined*,
not low. Collapsing that into `ambiguous` is the mistake that buried 66 real
flags under 198 artifacts in the gold-anchor-ambiguity work; the buckets are
kept apart here for the same reason, and they are checked to partition the
dictionary exactly (S2).

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/build_relation_graph.py
    .venv/Scripts/python.exe tools/corpus_prep/build_relation_graph.py --render
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
DICT_DIR = REPO / "data" / "entity_dictionaries"
GRAPH_PATH = REPO / "data" / "graph" / "relations.json"
RAW_PATH = REPO / "data" / "results" / "relation_graph_raw.json"
REPORT_PATH = REPO / "docs" / "relation-graph.md"

# tools/corpus_prep is run directly (not via pytest's configured pythonpath),
# so src/ needs to be on sys.path explicitly to reuse the already-tested
# matchers instead of duplicating them here.
sys.path.insert(0, str(REPO / "src"))
from rag_lab.loaders.common import (  # noqa: E402
    iter_corpus_files,
    strip_document_header,
    strip_mapping_tables,
)
from rag_lab.loaders.faculty_loader import match_faculties  # noqa: E402
from rag_lab.loaders.person_loader import find_people  # noqa: E402
from rag_lab.loaders.program_loader import match_programs  # noqa: E402

# --- edge A' extraction parameters (all three are reported, not hidden) ------
FACULTY_WINDOW = 60  # chars after "สังกัด" the faculty name must fit inside
PERSON_WINDOW = 120  # max chars between the end of a name and "สังกัด"
# --- classification thresholds ----------------------------------------------
MIN_SHARE = 0.60
MIN_VOTES_PROGRAM = 2  # a program is named in many documents; demand a repeat
MIN_VOTES_PERSON = 1  # a person is usually named once, in one table row

_SANGKAT = re.compile(r"สังกัด")
# The faculty must start where "สังกัด" ends, modulo punctuation/space. Without
# this anchor a 60-char window would happily match a faculty mentioned later in
# the sentence and attribute it to the wrong person.
_FACULTY_PREFIX = re.compile(r"^[\s:：.\-—_|]*(คณะ|วิทยาลัย|สถาบัน|วิทยาเขต)")


# ---------------------------------------------------------------- extraction
def affiliations_in(text: str) -> list[dict]:
    """Every "<person> ... สังกัด<faculty>" occurrence in one document.

    `d_before`/`d_after` (distance in characters to the nearest name on either
    side) are recorded even when unused, so the report can show the
    distribution the PERSON_WINDOW cut was chosen against instead of asserting
    it, and so the "is the name ever *after* the marker" question is answered
    by measurement.
    """
    people = find_people(text)
    out: list[dict] = []
    for m in _SANGKAT.finditer(text):
        tail = text[m.end() : m.end() + FACULTY_WINDOW]
        if not _FACULTY_PREFIX.match(tail):
            continue
        faculties = match_faculties(tail)
        if len(faculties) != 1:
            continue
        before = [p for p in people if p[1] <= m.start()]
        after = [p for p in people if p[0] >= m.end()]
        row: dict = {
            "faculty": faculties[0],
            "d_before": (m.start() - before[-1][1]) if before else None,
            "d_after": (after[0][0] - m.end()) if after else None,
        }
        if row["d_before"] is not None and row["d_before"] <= PERSON_WINDOW:
            row["person"] = before[-1][2]["full_name"]
        out.append(row)
    return out


def file_evidence(text: str) -> dict:
    return {
        "programs": match_programs(text),
        "faculties": match_faculties(text),
        "affiliations": affiliations_in(text),
        "people": sorted({p[2]["full_name"] for p in find_people(text)}),
    }


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def collect(corpus_root: Path, progress_every: int = 300) -> dict:
    """Walk the corpus once and return everything the graph is derived from."""
    t0 = time.time()
    files: dict[str, dict] = {}
    for i, path in enumerate(iter_corpus_files(corpus_root)):
        text = strip_mapping_tables(strip_document_header(read_text(path)))
        rel = str(path.relative_to(corpus_root).as_posix())
        files[rel] = file_evidence(text)
        if progress_every and i % progress_every == 0:
            print(f"  ...{i} files, {time.time() - t0:.0f}s", flush=True)

    # Manifest titles are a *second, independent* text source for edge A: the
    # title is written by the secretariat, the body is OCR'd from a scan, so
    # agreement between them is evidence rather than a tuned parameter.
    titles: dict[str, dict] = {}
    for manifest in sorted(corpus_root.rglob("meeting_manifest.json")):
        folder = manifest.parent.relative_to(corpus_root).as_posix()
        for item in json.loads(read_text(manifest)):
            title, name = item.get("title"), item.get("file")
            if not title or not name:
                continue
            titles[f"{folder}/{name}"] = {
                "title": title,
                "programs": match_programs(title),
                "faculties": match_faculties(title),
            }
    return {
        "generated_by": "tools/corpus_prep/build_relation_graph.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "walk_seconds": round(time.time() - t0, 1),
        "files": files,
        "titles": titles,
    }


# ------------------------------------------------------------------- voting
def one_to_one_votes(tagsets: dict[str, dict]) -> dict[str, Counter]:
    """program -> Counter(faculty), counting only sources that name exactly one
    of each. A document naming two faculties cannot say which one owns the
    programme, so it abstains rather than voting for both."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for ev in tagsets.values():
        if len(ev["programs"]) == 1 and len(ev["faculties"]) == 1:
            votes[ev["programs"][0]][ev["faculties"][0]] += 1
    return votes


def person_votes(files: dict[str, dict]) -> dict[str, Counter]:
    votes: dict[str, Counter] = defaultdict(Counter)
    for ev in files.values():
        for row in ev["affiliations"]:
            if "person" in row:
                votes[row["person"]][row["faculty"]] += 1
    return votes


def classify(counter: Counter, min_votes: int) -> dict:
    """Three-way. `no_evidence` is *undefined*, not a low score -- see module
    docstring."""
    total = sum(counter.values())
    if total == 0:
        return {"status": "no_evidence", "faculty": None, "votes": 0, "total": 0,
                "share": None}
    faculty, n = counter.most_common(1)[0]
    share = n / total
    status = "resolved" if (n >= min_votes and share >= MIN_SHARE) else "ambiguous"
    return {"status": status, "faculty": faculty, "votes": n, "total": total,
            "share": round(share, 4)}


def merge(*counters: Counter) -> Counter:
    out: Counter = Counter()
    for c in counters:
        out.update(c)
    return out


# ------------------------------------------------------------------- graph
def build_graph(raw: dict) -> dict:
    programs = [p["canonical"] for p in json.loads(read_text(DICT_DIR / "programs.json"))]
    body = one_to_one_votes(raw["files"])
    title = one_to_one_votes(raw["titles"])

    program_edges: dict[str, dict] = {}
    for name in programs:
        record = classify(merge(body.get(name, Counter()), title.get(name, Counter())),
                          MIN_VOTES_PROGRAM)
        record["by_source"] = {
            "body": dict(body.get(name, Counter())),
            "title": dict(title.get(name, Counter())),
        }
        program_edges[name] = record

    people = person_votes(raw["files"])
    person_edges = {
        name: classify(counter, MIN_VOTES_PERSON) for name, counter in people.items()
    }
    return {
        "generated_by": "tools/corpus_prep/build_relation_graph.py",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_generated_at": raw["generated_at"],
        "corpus_files": len(raw["files"]),
        "params": {
            "faculty_window": FACULTY_WINDOW,
            "person_window": PERSON_WINDOW,
            "min_share": MIN_SHARE,
            "min_votes_program": MIN_VOTES_PROGRAM,
            "min_votes_person": MIN_VOTES_PERSON,
        },
        "program_belongs_to_faculty": program_edges,
        "person_affiliated_with_faculty": person_edges,
    }


# ------------------------------------------------------------------- checks
def run_checks(raw: dict, graph: dict) -> list[dict]:
    """Structural self-checks. Gated ones make the script exit 1; the rest are
    classified rather than failed."""
    faculties = {f["canonical"] for f in json.loads(read_text(DICT_DIR / "faculties.json"))}
    programs = {p["canonical"] for p in json.loads(read_text(DICT_DIR / "programs.json"))}
    rows: list[dict] = []

    used = (
        {r["faculty"] for r in graph["program_belongs_to_faculty"].values() if r["faculty"]}
        | {r["faculty"] for r in graph["person_affiliated_with_faculty"].values() if r["faculty"]}
    )
    bad_nodes = sorted(used - faculties)  # parenthesised: `-` binds tighter than `|`
    rows.append({
        "id": "S1", "gate": True, "ok": not bad_nodes,
        "what": "ทุก node คณะเป็นชื่อ canonical ใน faculties.json",
        "detail": f"{len(bad_nodes)} node นอกพจนานุกรม จาก {len(used)} ที่ใช้จริง"
                  + (f": {bad_nodes[:5]}" if bad_nodes else ""),
    })

    # Not "do the buckets add up" -- iterating the dictionary makes that true by
    # construction, i.e. a vacuous PASS. What is *not* guaranteed is that the
    # three buckets stay distinguishable: a refactor of `classify` that returns
    # share=0.0 instead of None for an empty counter would silently turn
    # "undefined" into "worst", which is the exact collapse this graph is built
    # to avoid.
    records = list(graph["program_belongs_to_faculty"].values())
    buckets = Counter(r["status"] for r in records)
    leaked = [
        r for r in records
        if (r["status"] == "no_evidence")
        != (r["total"] == 0 and r["share"] is None and r["faculty"] is None)
    ]
    off_dict = sorted(set(graph["program_belongs_to_faculty"]) - programs)
    rows.append({
        "id": "S2", "gate": True,
        "ok": not leaked and not off_dict and len(records) == len(programs),
        "what": "no_evidence ต้อง 'ไม่นิยาม' จริง (total=0, share=None) ไม่ใช่คะแนนต่ำ",
        "detail": f"{buckets['resolved']}+{buckets['ambiguous']}+{buckets['no_evidence']}"
                  f" = {len(records)} จาก {len(programs)}; {len(leaked)} record ที่ปนกัน",
    })

    # A window match that the whole document does not also produce would mean
    # the short slice changed the fuzzy denominator enough to invent a faculty.
    leaks = 0
    checked = 0
    for ev in raw["files"].values():
        doc = set(ev["faculties"])
        for row in ev["affiliations"]:
            checked += 1
            if row["faculty"] not in doc:
                leaks += 1
    rows.append({
        "id": "S3", "gate": True, "ok": leaks == 0,
        "what": "คณะที่ดึงจาก window ต้องปรากฏใน tag ระดับเอกสารด้วย",
        "detail": f"{leaks} จาก {checked} occurrence",
    })

    # A person edge whose name the file-level tagger never saw would mean
    # find_people and the affiliation walk disagree about the same text.
    unknown = 0
    for ev in raw["files"].values():
        known = set(ev["people"])
        for row in ev["affiliations"]:
            if "person" in row and row["person"] not in known:
                unknown += 1
    rows.append({
        "id": "S4", "gate": True, "ok": unknown == 0,
        "what": "ทุกชื่อบนขอบ A′ ต้องอยู่ในชุดชื่อที่ find_people เจอในไฟล์เดียวกัน",
        "detail": f"{unknown} ชื่อที่หาไม่เจอ",
    })
    return rows


def affiliation_context(raw: dict) -> dict:
    """Is the marked faculty the document's *own* faculty, or a different one?

    Reading the samples suggests the parenthetical is written precisely when
    the person does NOT belong to the faculty that owns the document -- it is a
    cross-appointment disambiguator. If so, two things follow and both matter
    more than the edge count: A' is a *biased* sample of people (only the
    cross-faculty ones get marked), and deriving person->faculty from plain
    document co-occurrence the way edge A does would be wrong in a specific
    direction rather than merely noisy. Measured, not assumed.
    """
    same = different = unknown = 0
    for rel, ev in raw["files"].items():
        rows = [r for r in ev["affiliations"] if "person" in r]
        if not rows:
            continue
        title = raw["titles"].get(rel, {})
        own = None
        if len(title.get("faculties", [])) == 1:
            own = title["faculties"][0]
        elif len(ev["faculties"]) == 1:
            own = ev["faculties"][0]
        for row in rows:
            if own is None:
                unknown += 1
            elif row["faculty"] == own:
                same += 1
            else:
                different += 1
    return {"same": same, "different": different, "unknown": unknown}


def agreement(raw: dict, graph: dict) -> dict:
    """Independent cross-checks on edge A. Reported, never gated -- these are
    measurements of the corpus, not invariants of the code."""
    body = one_to_one_votes(raw["files"])
    title = one_to_one_votes(raw["titles"])
    both = sorted(set(body) & set(title))
    agree = sum(1 for p in both
                if body[p].most_common(1)[0][0] == title[p].most_common(1)[0][0])

    # split-half: alternate documents by sorted relpath, so the two halves are
    # disjoint document sets rather than two views of the same one.
    keys = sorted(raw["files"])
    halves = [
        one_to_one_votes({k: raw["files"][k] for k in keys[i::2]}) for i in (0, 1)
    ]
    shared = sorted(set(halves[0]) & set(halves[1]))
    stable = sum(1 for p in shared
                 if halves[0][p].most_common(1)[0][0] == halves[1][p].most_common(1)[0][0])
    return {
        "title_vs_body_n": len(both), "title_vs_body_agree": agree,
        "split_half_n": len(shared), "split_half_agree": stable,
    }


# ------------------------------------------------------------------- report
def _pct(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "n/a"


def _votes_of(record: dict) -> Counter:
    return merge(Counter(record["by_source"]["body"]), Counter(record["by_source"]["title"]))


def render_report(raw: dict, graph: dict, checks: list[dict], agree: dict,
                  ctx: dict) -> str:
    prog = graph["program_belongs_to_faculty"]
    pers = graph["person_affiliated_with_faculty"]
    pb = Counter(r["status"] for r in prog.values())
    sb = Counter(r["status"] for r in pers.values())
    people_seen = {n for ev in raw["files"].values() for n in ev["people"]}
    aff_rows = [r for ev in raw["files"].values() for r in ev["affiliations"]]
    with_person = [r for r in aff_rows if "person" in r]
    files_with_aff = sum(1 for ev in raw["files"].values() if ev["affiliations"])

    L: list[str] = []
    A = L.append
    A("# กราฟความสัมพันธ์แบบง่าย (หลักสูตร→คณะ, บุคคล→คณะ)")
    A("")
    A(f"Generated by `tools/corpus_prep/build_relation_graph.py` "
      f"({graph['generated_at']}; evidence walk {raw['generated_at']}, "
      f"{raw['walk_seconds']:.0f}s)")
    A("")
    A("## 0. อ่านตัวเลขในเอกสารนี้อย่างไร")
    A("")
    A("- **ไม่มี gold query ชุดใดเป็น multi-hop** — ทั้ง `gold_query_set_73det.yaml` และชุด "
      "thematic ถามด้วย anchor เดียว hop เดียว กราฟนี้จึงเป็น *ความสามารถที่เพิ่มเข้ามา* "
      "ซึ่งการวัดผลชุดปัจจุบัน **วัดไม่ได้โดยโครงสร้าง** — ห้ามอ้างว่าได้ retrieval gain")
    A("- ตัวส่วนทุกตัวคือ **พจนานุกรม entity** ซึ่งเองก็เป็น subset ที่คัดมา และหลักฐานของขอบ A′ "
      "อยู่ในเอกสารประเภทที่ OCR เสียหายที่สุด — อ่านทุกเปอร์เซ็นต์ว่าเป็น "
      "\"สกัดได้กี่ % ของที่เจอ\" ไม่ใช่ \"ของที่มีอยู่จริง\"")
    A(f"- tag ทั้งหมดคำนวณใหม่จาก loader ที่มี test ไม่ได้อ่านจาก "
      f"`academic_resolutions/entity_tags/*_by_file.json` (ลงวันที่ 2026-07-17..25 "
      f"เก่ากว่าการซ่อม person loader, OCR remediation, การซ่อม title และการ re-OCR)")
    A("")
    A(f"เดินคลัง {graph['corpus_files']:,} ไฟล์ · พารามิเตอร์: "
      f"faculty_window={FACULTY_WINDOW}, person_window={PERSON_WINDOW}, "
      f"min_share={MIN_SHARE}, min_votes={MIN_VOTES_PROGRAM} (หลักสูตร) / "
      f"{MIN_VOTES_PERSON} (บุคคล)")
    A("")
    A("## 1. ขอบ A — หลักสูตร → คณะ")
    A("")
    A(f"| สถานะ | จำนวนหลักสูตร | สัดส่วนของ {len(prog)} |")
    A("|---|---:|---:|")
    for status in ("resolved", "ambiguous", "no_evidence"):
        A(f"| `{status}` | {pb[status]} | {_pct(pb[status], len(prog))} |")
    A("")
    A("`no_evidence` = ไม่มีเอกสารใดที่เอ่ยหลักสูตรนี้พร้อมคณะเดียวเลย — สัดส่วนของมัน "
      "**ไม่นิยาม** ไม่ใช่ต่ำ จึงแยกถังจาก `ambiguous` (ที่มีหลักฐานแต่ขัดกัน)")
    A("")
    amb_all = [r for r in prog.values() if r["status"] == "ambiguous"]
    thin = [r for r in amb_all if len(_votes_of(r)) == 1]
    A(f"`ambiguous` ยังไม่ใช่ถังเดียวกันทั้งหมด — **{len(amb_all) - len(thin)}** "
      f"รายการมีคณะมากกว่าหนึ่งชี้กันจริง (*ขัดแย้ง*) ส่วนอีก **{len(thin)}** รายการมีคณะ"
      f"เดียวแต่พยานน้อยกว่า `min_votes={MIN_VOTES_PROGRAM}` (*หลักฐานบาง*) — คนละอาการกัน "
      "และซ่อมคนละทาง (อย่างหลังแค่ลด threshold ก็ resolve แต่จะแลกมาด้วยความเชื่อถือ)")
    A("")
    per_faculty = Counter(r["faculty"] for r in prog.values() if r["status"] == "resolved")
    A("| คณะ | หลักสูตรที่ผูกได้ |")
    A("|---|---:|")
    for fac, n in per_faculty.most_common():
        A(f"| {fac} | {n} |")
    A("")
    amb = [(n, r) for n, r in prog.items() if r["status"] == "ambiguous"]
    amb.sort(key=lambda x: -x[1]["total"])
    if amb:
        A(f"### {len(amb)} หลักสูตรที่ `ambiguous` (หลักฐานขัดกัน) — 15 อันดับแรกตามจำนวนโหวต")
        A("")
        A("| หลักสูตร | โหวต |")
        A("|---|---|")
        for name, r in amb[:15]:
            detail = ", ".join(f"{f} {c}" for f, c in _votes_of(r).most_common())
            A(f"| {name} | {detail} |")
        A("")
    A("### สองสาเหตุคนละเรื่องที่ทำให้แถวบนสุดเป็น `ambiguous`")
    A("")
    A("- **หลักสูตรชื่อเดียวเปิดจริงในหลายคณะ** — `วิศวกรรมเครื่องกล` ได้ "
      "`คณะวิศวกรรมศาสตร์` และ `วิทยาเขตชุมพรเขตรอุดมศักดิ์` คนละ 20 กว่าโหวต · "
      "นี่ไม่ใช่ข้อผิดพลาด แต่แปลว่า **หลักสูตร→คณะ ไม่ใช่ฟังก์ชัน** — กราฟที่บังคับให้"
      "หนึ่งหลักสูตรมีคณะเดียวจะผิดสำหรับหลักสูตรกลุ่มนี้เสมอ")
    A("- **ชื่อที่ไม่มีในพจนานุกรมถูกดูดเข้าเพื่อนบ้านที่ใกล้ที่สุด** — `match_programs` "
      "ใช้ `SequenceMatcher` และไม่มีทางออก \"ไม่ตรงกับอะไรเลย\" สำหรับชื่อที่เฉียด: "
      "`หลักสูตรทันตแพทยศาสตรบัณฑิต` และ `หลักสูตรพยาบาลศาสตรบัณฑิต` **ทั้งคู่** "
      "ถูก match เป็น `หลักสูตรแพทยศาสตรบัณฑิต` (ทั้งสองชื่อไม่มีใน `programs.json`) "
      "ซึ่งอธิบายแถว `หลักสูตรแพทยศาสตรบัณฑิต` ทั้งแถว (ทันตแพทยศาสตร์ 20, พยาบาลศาสตร์ 5)")
    A("")
    A("วัดขอบเขตแล้ว ไม่ได้เดา: ชื่อใน `programs.json` **0 จาก 253** ชื่อที่ match ไป"
      "โดนชื่ออื่นในพจนานุกรมเดียวกัน — การชนกันจึงเกิดกับชื่อที่ *อยู่นอก* พจนานุกรม"
      "เท่านั้น และพจนานุกรม 253 รายการเป็น subset ที่คัดมา · **ไม่แก้ที่นี่**: "
      "`match_programs` ถูกใช้โดย `build_gold_candidates.py` และ `router` ด้วย "
      "การขยับ threshold จะเลื่อนตัวเลขที่ตีพิมพ์ไปแล้ว — บันทึกไว้เป็นงานแยก")
    A("")
    A("## 2. ขอบ A′ — บุคคล → คณะ")
    A("")
    A(f"- ไฟล์ที่มี pattern `สังกัด<คณะ>` ที่ anchor ผ่าน: **{files_with_aff:,}** "
      f"จาก {graph['corpus_files']:,} ({_pct(files_with_aff, graph['corpus_files'])})")
    A(f"- occurrence ทั้งหมด **{len(aff_rows):,}** · จับคู่กับชื่อคนได้ **{len(with_person):,}** "
      f"({_pct(len(with_person), len(aff_rows))})")
    A(f"- คนที่ได้ขอบ **{len(pers):,}** จาก **{len(people_seen):,}** คนที่ `find_people` "
      f"เจอทั้งคลัง ({_pct(len(pers), len(people_seen))})")
    A("")
    A(f"| สถานะ | จำนวนคน | สัดส่วนของ {len(pers)} |")
    A("|---|---:|---:|")
    for status in ("resolved", "ambiguous"):
        A(f"| `{status}` | {sb[status]} | {_pct(sb[status], len(pers))} |")
    A("")
    single = sum(1 for r in pers.values() if r["total"] == 1)
    A(f"**อย่าอ่าน `resolved` 100% ว่าเป็นคุณภาพ** — `min_votes` ของฝั่งบุคคลคือ "
      f"{MIN_VOTES_PERSON} (คนหนึ่งมักถูกเอ่ยครั้งเดียว) ดังนั้น {single} จาก "
      f"{len(pers)} คน ({_pct(single, len(pers))}) มีพยาน **แค่ครั้งเดียว** และไม่มีใคร"
      f"ค้าน · ตัวเลขนี้บอกว่า \"ไม่มีหลักฐานขัดกัน\" ไม่ได้บอกว่า \"ยืนยันซ้ำแล้ว\"")
    A("")
    A("### ระยะห่างระหว่างชื่อกับคำว่า `สังกัด` (เหตุผลของ person_window)")
    A("")
    A("| ระยะ (อักขระ) | occurrence ที่ชื่ออยู่ *ก่อน* | ที่ชื่ออยู่ *หลัง* |")
    A("|---|---:|---:|")
    edges = [(0, 20), (20, 60), (60, 120), (120, 300), (300, 10**9)]
    for lo, hi in edges:
        nb = sum(1 for r in aff_rows if r["d_before"] is not None and lo <= r["d_before"] < hi)
        na = sum(1 for r in aff_rows if r["d_after"] is not None and lo <= r["d_after"] < hi)
        A(f"| {lo}–{hi if hi < 10**9 else '∞'} | {nb:,} | {na:,} |")
    # The price of extracting one direction only is NOT "how often a name sits
    # after the marker" -- it is how often one sits after it *within the window
    # we would have accepted*, with nothing acceptable before. Everything else
    # would have been rejected by distance anyway, in either direction.
    lost = sum(
        1 for r in aff_rows
        if "person" not in r
        and r["d_after"] is not None and r["d_after"] <= PERSON_WINDOW
    )
    closer_after = sum(
        1 for r in aff_rows
        if r["d_after"] is not None and (r["d_before"] is None or r["d_after"] < r["d_before"])
    )
    A("")
    A(f"occurrence ที่ชื่ออยู่ *หลัง* marker และใกล้กว่าชื่อที่อยู่ก่อน: "
      f"**{closer_after:,}** จาก {len(aff_rows):,} — แต่**นั่นไม่ใช่ราคาของการเลือก"
      f"ทิศเดียว** เพราะเกือบทั้งหมดไกลเกิน `person_window` อยู่แล้ว จึงถูกตัดทิ้ง"
      f"ทั้งสองทิศ · ราคาจริงคือ occurrence ที่ *ไม่ได้* ชื่อ ทั้งที่มีชื่ออยู่หลัง "
      f"marker ภายในระยะที่จะรับ = **{lost:,}** จาก {len(aff_rows):,}")
    A("")
    A("### marker นี้เขียนไว้ตอนไหน — คนในคณะเดียวกับเอกสาร หรือคนละคณะ")
    A("")
    known = ctx["same"] + ctx["different"]
    A(f"| ความสัมพันธ์กับคณะเจ้าของเอกสาร | occurrence |")
    A("|---|---:|")
    A(f"| คนละคณะ | {ctx['different']:,} ({_pct(ctx['different'], known)}) |")
    A(f"| คณะเดียวกัน | {ctx['same']:,} ({_pct(ctx['same'], known)}) |")
    A(f"| ระบุคณะเจ้าของเอกสารไม่ได้ | {ctx['unknown']:,} |")
    A("")
    A("อ่านตัวเลขนี้ก่อนใช้ขอบ A′: ถ้าส่วนใหญ่เป็น \"คนละคณะ\" แปลว่าวงเล็บนี้ถูกเขียนไว้ "
      "**เพราะ**คนคนนั้นมาจากคณะอื่น — A′ จึงเป็น sample ที่ **เอียง** (คนที่สังกัดคณะ"
      "เดียวกับเอกสารจะไม่ถูก mark) และการอนุมาน person→faculty จากการอยู่ร่วมเอกสาร "
      "(วิธีเดียวกับขอบ A) จะผิด **อย่างมีทิศทาง** ไม่ใช่แค่ noisy")
    A("")
    A("## 3. ตรวจสอบข้าม (รายงาน ไม่ใช่ตัวตัดสิน)")
    A("")
    A(f"- **ชื่อเรื่องใน manifest vs เนื้อเอกสาร** — สองแหล่งข้อความที่เป็นอิสระต่อกัน "
      f"(เลขานุการพิมพ์ vs OCR): หลักสูตรที่ทั้งสองแหล่งชี้ได้ {agree['title_vs_body_n']} "
      f"หลักสูตร ตรงกัน **{agree['title_vs_body_agree']}** "
      f"({_pct(agree['title_vs_body_agree'], agree['title_vs_body_n'])})")
    A(f"- **split-half** — แบ่งเอกสารสลับกันเป็นสองกอง: หลักสูตรที่ทั้งสองกองชี้ได้ "
      f"{agree['split_half_n']} หลักสูตร ตรงกัน **{agree['split_half_agree']}** "
      f"({_pct(agree['split_half_agree'], agree['split_half_n'])})")
    A("")
    A("## 4. self-checks")
    A("")
    A("| id | ตรวจอะไร | ผล | รายละเอียด |")
    A("|---|---|---|---|")
    for row in checks:
        A(f"| {row['id']} | {row['what']} | {'PASS' if row['ok'] else 'FAIL'} | {row['detail']} |")
    A("")
    A(f"ผลลัพธ์กราฟ: `data/graph/relations.json` · หลักฐานดิบ: "
      f"`data/results/relation_graph_raw.json` (`--render` สร้างรายงานใหม่จากไฟล์นี้ "
      f"โดยไม่ต้องเดินคลังซ้ำ)")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", action="store_true",
                    help="rebuild graph+report from the cached evidence (no corpus walk)")
    args = ap.parse_args()

    if args.render:
        raw = json.loads(read_text(RAW_PATH))
        print(f"[render] cached evidence from {raw['generated_at']}")
    else:
        raw = collect(CORPUS_ROOT)
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        print(f"[walk] {len(raw['files'])} files in {raw['walk_seconds']:.0f}s")

    graph = build_graph(raw)
    checks = run_checks(raw, graph)
    agree = agreement(raw, graph)
    ctx = affiliation_context(raw)

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_PATH.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT_PATH.write_text(render_report(raw, graph, checks, agree, ctx), encoding="utf-8")

    pb = Counter(r["status"] for r in graph["program_belongs_to_faculty"].values())
    print(f"[graph] programs: {pb['resolved']} resolved / {pb['ambiguous']} ambiguous"
          f" / {pb['no_evidence']} no_evidence")
    print(f"[graph] people : {len(graph['person_affiliated_with_faculty'])} with an edge")
    failed = [c for c in checks if c["gate"] and not c["ok"]]
    for c in checks:
        print(f"  {c['id']} {'PASS' if c['ok'] else 'FAIL'}  {c['detail']}")
    print(f"[out] {GRAPH_PATH}\n[out] {REPORT_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
