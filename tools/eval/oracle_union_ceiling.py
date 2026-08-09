"""Oracle-union ceiling: how much of the Gold set is reachable at all, and at what budget.

Answers ticket 01 of `road-to-wow-demo/` ("what is the real ceiling, and is the
0.9 target closed off by document quality?") by unioning the persisted top-10 of
every chunker x embedder combo. If a (query, resolution) pair is not found by
*any* system, no reranker, ensemble or fine-tune can reach it while the index
family and k stay fixed.

Rewritten from `road-to-wow-demo/assets/01-oracle-union.py` (2026-07-31). Four
things changed, and each is a correction rather than a refresh:

1. **Membership is derived, not incidental.** The original selects combos by
   `glob("*.json")` with no filter, which is not portable between checkouts: run
   against *this* repo it unions 44 combos, because `gold_hybrid_73det/` still
   holds results for the 8 in `_EXCLUDED_COMBO_DIRS` (the old 128-cap `sct`, the
   rejected 510-cap `congen`) whose index directories were deleted and whose
   results predate rebuild #3. Measured cost of not filtering: the ceiling reads
   0.9046 instead of 0.8948. **This is a portability defect, not an indictment
   of the original's published numbers** -- its own header reports 36 combos, so
   those 8 result files evidently were not present on the machine it ran on
   (`REPO` is hardcoded to a different user's OneDrive path). Here the kept set
   is derived from which index directories exist, then cross-checked against the
   exclusion list, so neither half can go stale unnoticed.

2. **106 queries, not 73.** This is the real source of the gap, and it is not a
   flaw in the original either -- it ran against a checkout whose gold set still
   had 73 entries. The 33 `course` queries landed 2026-07-25 and the filename
   still says `73det`. Gold pairs go 644 -> 1,046. Holding this script's own
   combo set fixed and scoring only the 73 non-course queries reproduces the
   original's shape: best single 0.6728 (published 0.6935), union 0.9125
   (published 0.9201). The residual is not attributable from here -- that
   checkout's results predate rebuild #3 and both `resolution_id` repairs.

3. **The router section is gone, not recomputed.** It has a tested descendant:
   `tools/eval/routing_eval.py` does per-route selection with 5 routes, matched
   fitting budgets and bootstrap significance. Recomputing an untested +0.0465
   here would put it in open conflict with a tested +0.0499 that does *not*
   clear significance. Cite `data/results/routing_eval.md` for that question.

4. **Macro and micro are reported side by side.** The original mixed them: a
   per-query mean in section 1, a pair count in section 2, with no bridge. Gold
   queries carry 1-43 relevant resolutions each, so the two disagree by a lot
   and the gap is itself a finding (see [[feedback_report_ties_as_bounds]]'s
   sibling note on macro-averaging making headroom look unmovable).

Sections 6 and 7 are new. The original could only say "the information is in
the index" about the hybrid arm; dense and BM25 have persisted results too, so
section 6 takes the union across retrieval paths as well as across combos. And
the original could say *where* the ceiling is low but not *why*: section 7
re-reads the actual evidence behind the `course` pairs nothing retrieves, which
turns out to be a distinct third category -- neither a ranking failure nor a
document-quality one.

Read-only. Consumes persisted results and one index's `chunks.parquet` (for the
text the retrievers actually saw); runs no retrieval and writes to no index.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from embedder_matrix_9way import _EXCLUDED_COMBO_DIRS, _embedder_label  # noqa: E402

INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
HYB = REPO / "data" / "results" / "gold_hybrid_73det"
DENSE = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
BM25 = REPO / "data" / "results" / "gold_bm25_73det"
HYB_REPORT = REPO / "data" / "results" / "gold_hybrid_73det_report.md"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
OUT = REPO / "data" / "results" / "oracle_union_ceiling.md"
K = 10


def load_ranked(results_dir: Path, arm: str, qrels: dict) -> dict[str, dict[str, list[str]]]:
    """combo prefix -> query -> ranked resolution_ids (best first)."""
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
    qrels = {d["query"]: set(d["relevant_resolution_ids"]) for d in raw}
    queries = [d["query"] for d in raw]
    etype = {d["query"]: d.get("entity_type", "?") for d in raw}

    checks: list[tuple[str, bool, str]] = []

    # S0 -- duplicate query strings would silently collapse in `qrels`, and the
    # thematic set really did contain 5 of them once.
    checks.append((
        "S0 gold query strings unique",
        len(set(queries)) == len(raw),
        f"{len(set(queries))} unique of {len(raw)} entries",
    ))

    # S1 -- membership. Derive the live combo set from the index directories that
    # exist, not from a constant, then assert the leftovers are exactly the known
    # retired ones. Either half alone can go stale without failing.
    live = {d.name for d in INDEX_DIR.iterdir() if d.is_dir()}
    ranked_hyb_all = load_ranked(HYB, "hybrid", qrels)
    with_results = set(ranked_hyb_all)
    leftovers = with_results - live
    combos = sorted(with_results & live)
    checks.append((
        "S1 result combos with no index dir are exactly the known retired set",
        leftovers == _EXCLUDED_COMBO_DIRS,
        f"{len(leftovers)} leftover, {len(combos)} kept of {len(with_results)} with results",
    ))

    # S2 -- coverage and depth. A missing query would KeyError deep inside the
    # greedy loops; a file holding fewer than K results would quietly shrink a union.
    bad_cov = [c for c in combos if set(ranked_hyb_all[c]) != set(queries)]
    bad_depth = [
        c for c in combos
        if any(len(ranked_hyb_all[c][q]) != K for q in ranked_hyb_all[c])
    ]
    checks.append((
        f"S2 every kept combo covers all {len(queries)} queries at exactly {K} results",
        not bad_cov and not bad_depth,
        f"{len(bad_cov)} short on coverage, {len(bad_depth)} short on depth",
    ))

    # S3 -- the anchor. If this harness reads the persisted files the way the
    # published eval did, every kept combo must reproduce its own published
    # recall@10 exactly. Any later disagreement is then a design choice of this
    # script, not a parsing bug.
    published = {}
    for line in HYB_REPORT.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (plain__\S+)__hybrid \| ([0-9.]+) \|", line)
        if m:
            published[m.group(1)] = float(m.group(2))
    mism = []
    for c in combos:
        got = float(np.mean([
            len(set(ranked_hyb_all[c][q][:K]) & qrels[q]) / len(qrels[q]) for q in queries
        ]))
        if c not in published or abs(got - published[c]) > 5e-5:
            mism.append(c)
    checks.append((
        "S3 every kept combo reproduces its published recall@10 to 4dp",
        not mism,
        f"{len(combos) - len(mism)} of {len(combos)} reproduce",
    ))

    # S4 -- section 7 pulls the course name back out of the query template. A
    # query that stopped matching the template would be silently dropped from
    # that section's denominator rather than counted as unparsed.
    course_qs = [d["query"] for d in raw if d.get("entity_type") == "course"]
    course_name = {}
    for q in course_qs:
        m = re.search(r"รายวิชา\s+(.+?)\s+ถูกกล่าวถึง", q)
        if m:
            course_name[q] = m.group(1)
    checks.append((
        "S4 every course query exposes its course name to the same template",
        len(course_name) == len(course_qs),
        f"{len(course_name)} of {len(course_qs)} parse",
    ))

    early = list(checks)
    for name, ok, detail in early:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
    if not all(ok for _, ok, _ in early):
        print("\nself-check failed; refusing to publish numbers", file=sys.stderr)
        return 1

    top = {c: {q: set(ranked_hyb_all[c][q][:K]) for q in queries} for c in combos}

    # keyed on every live index, not just the hybrid ones -- section 6 labels
    # dense/BM25 combos out of the same dict
    manifests = {
        c: json.loads((INDEX_DIR / c / "manifest.json").read_text(encoding="utf-8"))["combo"]
        for c in sorted(live)
    }
    labels = {
        c: f"{m['chunker']['type']} x {_embedder_label(m)}" for c, m in manifests.items()
    }

    n_pairs = sum(len(qrels[q]) for q in queries)

    def macro(sets_by_q: dict[str, set[str]]) -> float:
        return float(np.mean([
            len(sets_by_q[q] & qrels[q]) / len(qrels[q]) for q in queries
        ]))

    def micro(sets_by_q: dict[str, set[str]]) -> float:
        return sum(len(sets_by_q[q] & qrels[q]) for q in queries) / n_pairs

    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# เพดานของ index family ปัจจุบัน — oracle-union recall@10")
    w()
    w("Generated by `tools/eval/oracle_union_ceiling.py` from persisted results ")
    w("(no retrieval, no GPU). แทนที่ `road-to-wow-demo/assets/01-oracle-union.py` ")
    w("ซึ่งรันบน **73 คำถาม** (checkout คนละเครื่อง ตอนนั้น gold set ยังมี 73 ข้อ) และเลือก combo ")
    w("ด้วย `glob` ล้วน ๆ — ถ้ารันสคริปต์นั้นบน repo นี้จะได้ 44 combo (มี 8 ตัวที่เลิกใช้แล้วปนมา)")
    w()
    w(f"- combo ที่นับ: **{len(combos)}** (จาก {len(with_results)} ที่มีไฟล์ผล — "
      f"ตัด {len(leftovers)} combo ที่ index ถูกลบไปแล้วออก)")
    w(f"- คำถาม: **{len(queries)}** · คู่ (query, resolution) ทั้งหมด: **{n_pairs}**")
    w(f"- k = {K} · retriever = hybrid (RRF, BM25 + dense)")
    w()
    w("**นิยาม** oracle-union = รวม top-10 ของทุกระบบเข้าด้วยกัน เท่ากับสมมติว่ามี oracle ")
    w("ที่เลือกถูกเสมอว่าคำถามไหนควรใช้ระบบไหน ไม่มี rerank / ensemble / fine-tune ใด ")
    w("ทำได้เกินเส้นนี้ ตราบใดที่ยังใช้ index ชุดนี้และ k=10")
    w()
    w("**macro vs micro** — ทุกตารางรายงานคู่กัน macro = ค่าเฉลี่ยของ recall ต่อคำถาม ")
    w("(ทุกคำถามน้ำหนักเท่ากัน) micro = คู่ที่เจอทั้งหมด ÷ คู่ทั้งหมด (ทุก *คู่* น้ำหนักเท่ากัน) ")
    w(f"คำถามในชุดนี้มี gold ตั้งแต่ {min(len(v) for v in qrels.values())} ถึง ")
    w(f"{max(len(v) for v in qrels.values())} มติ ทั้งสองตัวจึงต่างกันมาก และตัวเลข ")
    w("ในเอกสารเก่าที่เขียนว่า \"55.1%\" กับ \"0.6935\" คือสองมาตรวัดนี้ ไม่ใช่ความขัดแย้ง")
    w()

    # ---- 1. ceiling -------------------------------------------------------
    w("## 1. ระบบเดี่ยวที่ดีที่สุด vs เพดาน")
    w()
    w("⚠️ **แถวเหล่านี้ดึงเอกสารไม่เท่ากัน** — ห้ามอ่านคอลัมน์ recall เทียบกันตรง ๆ ")
    w("ดูหัวข้อ 1b/1c สำหรับการเทียบที่งบเท่ากัน")
    w()
    per_combo = {c: macro(top[c]) for c in combos}
    best = max(per_combo, key=lambda c: per_combo[c])
    sem = [c for c in combos if manifests[c]["chunker"]["type"] == "semantic"]
    union_all = {q: set().union(*(top[c][q] for c in combos)) for q in queries}
    union_sem = {q: set().union(*(top[c][q] for c in sem)) for q in queries}

    w("| ระบบ | เอกสารที่ดึง (ก่อนตัดซ้ำ) | recall macro | recall micro | ห่าง (macro) |")
    w("|---|---|---|---|---|")
    w(f"| ระบบเดี่ยวที่ดีที่สุด: **{labels[best]}** | {K} | **{macro(top[best]):.4f}** "
      f"| {micro(top[best]):.4f} | — |")
    w(f"| union ของ {len(sem)} combo ที่ใช้ semantic chunker | {K*len(sem)} "
      f"| {macro(union_sem):.4f} | {micro(union_sem):.4f} | {macro(union_sem)-macro(top[best]):+.4f} |")
    w(f"| union ของทั้ง {len(combos)} combo | {K*len(combos)} | {macro(union_all):.4f} "
      f"| {micro(union_all):.4f} | {macro(union_all)-macro(top[best]):+.4f} |")
    w(f"| เพดานทฤษฎี | — | 1.0000 | 1.0000 | {1-macro(top[best]):+.4f} |")
    w()
    # How much the membership filter is actually worth, measured rather than
    # asserted. It buys nothing on the best-single row (the retired combos are
    # the weak `sct`/`congen` ones, never the argmax) and inflates only the
    # ceiling -- so the filter matters for *this* document's headline number
    # specifically, which is the one thing it publishes.
    stale_union = {
        q: set().union(*(set(ranked_hyb_all[c][q][:K]) for c in with_results)) for q in queries
    }
    w(f"**ผลของการกรอง combo ที่เลิกใช้ออก** ถ้าไม่กรอง (รวม {len(with_results)} combo ตามที่ ")
    w(f"`glob` เจอไฟล์ผล) เพดานจะอ่านได้ **{macro(stale_union):.4f}** แทน "
      f"**{macro(union_all):.4f}** — สูงเกินจริง **{macro(stale_union)-macro(union_all):+.4f}** ")
    w("แถวระบบเดี่ยวไม่ขยับเลย เพราะ combo ที่เลิกใช้เป็นตัวอ่อน (`sct`/`congen`) ไม่เคยเป็น ")
    w("argmax อยู่แล้ว การกรองจึงมีผลกับ *เพดาน* ซึ่งเป็นตัวเลขเดียวที่เอกสารนี้ตีพิมพ์")
    w()

    # Reproduce the retired 73-query shape. This exists so the attribution in
    # `paper-results-summary.md` ("the gap is query-set coverage, not method")
    # is a measured row in a report rather than a number computed in chat --
    # holding *this* script's combo set and code path fixed and changing only
    # the query set has to land near the published 0.6935 / 0.9201 for that
    # attribution to hold.
    q73 = [q for q in queries if etype[q] != "course"]

    def macro73(sets_by_q: dict[str, set[str]]) -> float:
        return float(np.mean([
            len(sets_by_q[q] & qrels[q]) / len(qrels[q]) for q in q73
        ]))

    best73 = max(combos, key=lambda c: macro73(top[c]))
    w(f"**ทำซ้ำรูปเดิมที่ {len(q73)} คำถาม** (ตัด `course` ออก ซึ่งเป็นชุดที่ยังไม่มีตอนนั้น) ")
    w("โดยคง combo set และ code path ของสคริปต์นี้ไว้เหมือนเดิม เปลี่ยนแค่ชุดคำถาม: ")
    w(f"ระบบเดี่ยวที่ดีที่สุด **{macro73(top[best73]):.4f}** (ตีพิมพ์ไว้ 0.6935) เพดาน union ")
    w(f"**{macro73(union_all):.4f}** (ตีพิมพ์ไว้ 0.9201) — ใกล้พอที่จะสรุปได้ว่าช่องว่างมาจาก ")
    w("*ชุดคำถามคนละชุด* ไม่ใช่วิธีคำนวณคนละวิธี")
    w()

    # ---- 1b. equal budget -------------------------------------------------
    def topn(c: str, q: str, n: int) -> set[str]:
        return set(ranked_hyb_all[c][q][:n])

    def greedy_at(depth: int, m: int):
        chosen: list[str] = []
        cur = {q: set() for q in queries}
        val = 0.0
        for _ in range(m):
            bc, bv, bu = None, -1.0, None
            for c in combos:
                if c in chosen:
                    continue
                cand = {q: cur[q] | topn(c, q, depth) for q in queries}
                v = macro(cand)
                if v > bv:
                    bc, bv, bu = c, v, cand
            chosen.append(bc)
            cur, val = bu, bv
        return chosen, val, cur

    w("## 1b. เทียบที่ **งบเท่ากัน** — ส่งให้ LLM 10 ใบเท่ากันทุกแถว")
    w()
    w("บังคับให้ทุกแถวส่ง 10 ใบ: m ระบบ ระบบละ 10/m อันดับแรก จึงแยก 'ความหลากหลาย' ")
    w("ออกจาก 'ดึงลึกขึ้น' ได้")
    w()
    w("หมายเหตุ: การเลือกระบบเป็น greedy บนชุดทดสอบเอง **ทั้งแถว 1 ระบบและแถวหลายระบบ** ")
    w("ผลลัพธ์ข้างล่างจึงเข้าข้างความหลากหลายอยู่แล้ว — ถ้ามันยังติดลบ ก็ติดลบจริง")
    w()
    w("| จำนวนระบบ | ดึงระบบละ | รวม (ก่อนตัดซ้ำ) | recall macro | ระบบที่ถูกเลือก |")
    w("|---|---|---|---|---|")
    eq = {}
    for m, depth in ((1, K), (2, K // 2), (5, K // 5)):
        ch, v, sets_m = greedy_at(depth, m)
        eq[m] = (v, ch, sets_m)
        w(f"| {m} | {depth} | {m*depth} | **{v:.4f}** | {', '.join(labels[c] for c in ch)} |")
    ch20, v20, sets20 = greedy_at(K, 2)
    w(f"| 2 (งบสองเท่า — วิธีเทียบเดิมที่ถอนคืนแล้ว) | {K} | {2*K} | {v20:.4f} "
      f"| {', '.join(labels[c] for c in ch20)} |")
    w()
    w(f"**ผลของความหลากหลายล้วน ๆ ที่งบ 10 ใบ = {eq[2][0]-eq[1][0]:+.4f}** · "
      f"ผลของการดึงลึกขึ้นเป็นสองเท่า = {v20-eq[1][0]:+.4f}")
    w()

    # ---- 1c. oracles at a fixed 10-doc budget -----------------------------
    w("## 1c. เพดานที่ยัง 'ส่ง 10 ใบ' เท่าเดิม")
    w()
    sel = {q: max((top[c][q] for c in combos), key=lambda s: len(s & qrels[q])) for q in queries}

    def capped(pool: dict[str, set[str]]) -> tuple[float, float]:
        # a perfect reranker keeps at most K documents out of the pool
        hits = {q: min(len(pool[q] & qrels[q]), K) for q in queries}
        mac = float(np.mean([hits[q] / len(qrels[q]) for q in queries]))
        mic = sum(hits.values()) / n_pairs
        return mac, mic

    rr2 = capped(sets20)
    rrall = capped(union_all)
    w("| วิธี | ส่งให้ LLM | ต้องดึง+จัดอันดับ | macro | micro | ห่างจากระบบเดียว (macro) |")
    w("|---|---|---|---|---|---|")
    w(f"| ระบบเดียวที่ดีที่สุด (ของจริงวันนี้) | {K} | {K} | {macro(top[best]):.4f} "
      f"| {micro(top[best]):.4f} | — |")
    w(f"| **oracle เลือกระบบให้แต่ละคำถาม** | {K} | {K} | **{macro(sel):.4f}** "
      f"| {micro(sel):.4f} | {macro(sel)-macro(top[best]):+.4f} |")
    w(f"| union 2 ระบบ + reranker สมบูรณ์แบบ | {K} | {2*K} | {rr2[0]:.4f} | {rr2[1]:.4f} "
      f"| {rr2[0]-macro(top[best]):+.4f} |")
    w(f"| union {len(combos)} ระบบ + reranker สมบูรณ์แบบ | {K} | {K*len(combos)} "
      f"| {rrall[0]:.4f} | {rrall[1]:.4f} | {rrall[0]-macro(top[best]):+.4f} |")
    w()
    w("ทุกแถวล่างเป็น **oracle** ไม่ใช่ระบบ — อ่านเป็น 'เพดานของแนวทางนั้น' เท่านั้น")
    w()
    # S5 -- coherence with the other ceiling this project publishes. Any row that
    # *sends* K documents is bound by the qrels' own min(1, K/n_relevant) ceiling
    # (`docs/paper-results-summary.md`, 0.8856). Rows that send more are not, which
    # is why section 1's unions legitimately exceed it -- the trap is reading the
    # two tables as if they shared a budget. Pin it so the tables cannot drift apart.
    struct_ceiling = float(np.mean([min(1.0, K / len(qrels[q])) for q in queries]))
    capped_rows = {
        "best single": macro(top[best]),
        "oracle picks a system": macro(sel),
        "union 2 + perfect rerank": rr2[0],
        f"union {len(combos)} + perfect rerank": rrall[0],
    }
    over = {k: v for k, v in capped_rows.items() if v > struct_ceiling + 5e-5}
    checks.append((
        f"S5 every row that sends only {K} docs stays under the qrels ceiling",
        not over,
        f"ceiling {struct_ceiling:.4f}; highest capped row "
        f"{max(capped_rows.values()):.4f}; {len(over)} over",
    ))
    w(f"> **เพดานอีกตัวที่เอกสารนี้ไม่ขัดกัน** `docs/paper-results-summary.md` รายงาน ")
    w(f"> structural ceiling = **{struct_ceiling:.4f}** ซึ่งคือ `mean(min(1, {K}/n_relevant))` — ")
    w("> เพดานของระบบที่**ส่ง 10 ใบ** ตารางนี้ทุกแถวอยู่ใต้เส้นนั้น (ตรวจใน S5) ส่วนแถว union ")
    w("> ในหัวข้อ 1 ที่สูงกว่าเส้นนั้นได้ ก็เพราะมันไม่ได้ส่ง 10 ใบ ไม่ใช่เพราะขัดกัน")
    w()
    w("> **นโยบายที่สร้างได้จริงไม่ได้อยู่ในเอกสารนี้แล้ว** ฉบับเดิมคำนวณ router แบบ ")
    w("> leave-one-out ไว้ที่นี่ (+0.0465) แต่มันมีทายาทที่ทดสอบนัยสำคัญแล้วคือ ")
    w("> `tools/eval/routing_eval.py` → `data/results/routing_eval.md` ซึ่งทำด้วย 5 route ")
    w("> และจับคู่ baseline ตามงบ fitting: `routed (loo)` hybrid = **0.6780 (+0.0499) "
      "แต่ไม่ significant** ")
    w("> ให้อ้างตัวเลขจากที่นั่น ไม่ใช่จากที่นี่")
    w()

    # ---- 2. decomposition -------------------------------------------------
    w("## 2. แยก 'ส่วนที่พลาด' ออกเป็นสองก้อน (นับรายคู่ = micro)")
    w()
    found_best = sum(len(top[best][q] & qrels[q]) for q in queries)
    found_all = sum(len(union_all[q] & qrels[q]) for q in queries)
    gap_rank = found_all - found_best
    gap_struct = n_pairs - found_all
    missed = n_pairs - found_best
    w(f"- คู่ทั้งหมด: **{n_pairs}** · ระบบที่ดีที่สุดเจอ **{found_best}** "
      f"({found_best/n_pairs:.1%}) · union ของทุกระบบเจอ **{found_all}** ({found_all/n_pairs:.1%})")
    w()
    w("| ก้อน | คู่ | % ของทั้งหมด | % ของที่พลาด | แก้ด้วยอะไร |")
    w("|---|---|---|---|---|")
    w(f"| **A. การจัดอันดับ/การเลือกระบบ** — มีระบบใดระบบหนึ่งเจอ แต่ระบบที่ใช้อยู่ไม่เจอ "
      f"| {gap_rank} | {gap_rank/n_pairs:.1%} | {gap_rank/missed:.1%} "
      f"| เลือก combo ต่อคำถาม / rerank บน pool ที่รวมมาแล้ว |")
    w(f"| **B. โครงสร้าง** — ไม่มีระบบไหนเจอเลยใน top-{K} | {gap_struct} "
      f"| {gap_struct/n_pairs:.1%} | {gap_struct/missed:.1%} | OCR / chunk / label / เพิ่ม k |")
    w()
    w(f"⚠️ **ก้อน A ไม่ได้แปลว่า 'แก้ได้ {gap_rank/missed:.0%} ที่งบเดิม'** — นิยามของมันคือ ")
    w(f"'มีระบบใดระบบหนึ่งใน {len(combos)} ระบบเจอ' ซึ่งใช้งบ {K*len(combos)} ใบ ที่งบจริง 10 ใบ ")
    w(f"เพดานคือ {macro(sel):.4f} (oracle เลือกระบบ) ถึง {rrall[0]:.4f} (rerank สมบูรณ์แบบบน pool เต็ม) ")
    w(f"ไม่ใช่ {macro(union_all):.4f}")
    w()
    w(f"⚠️ ก้อน B ยังแยกไม่ออกว่า 'อยู่อันดับ 11-50' หรือ 'ไม่ติดเลย' เพราะไฟล์ผลเก็บแค่ rank ≤ {K} ")
    w("(ตรวจแล้วในหัวข้อ self-check) ต้องรัน retrieval ใหม่ที่ k=50 ถึงจะแยกได้")
    w()

    # ---- 3. per entity type ----------------------------------------------
    w("## 3. แยกตามชนิดคำถาม")
    w()
    w("| ชนิด | n | คู่ | ระบบดีที่สุด (macro) | union (macro) | headroom | union (micro) |")
    w("|---|---|---|---|---|---|---|")
    for t in sorted({etype[q] for q in queries}):
        qs = [q for q in queries if etype[q] == t]
        np_t = sum(len(qrels[q]) for q in qs)
        b = float(np.mean([len(top[best][q] & qrels[q]) / len(qrels[q]) for q in qs]))
        u = float(np.mean([len(union_all[q] & qrels[q]) / len(qrels[q]) for q in qs]))
        um = sum(len(union_all[q] & qrels[q]) for q in qs) / np_t
        w(f"| {t} | {len(qs)} | {np_t} | {b:.4f} | {u:.4f} | {u-b:+.4f} | {um:.4f} |")
    w()

    # ---- 4. hard queries --------------------------------------------------
    w("## 4. คำถามที่แม้แต่ union ก็ยังพลาดหนัก (recall union < 0.5)")
    w()
    w("คำถามเหล่านี้คือที่ที่ **embedding ไม่ใช่ปัญหา** — ควรไล่ดู OCR / label เป็นรายใบ")
    w()
    w("| คำถาม | ชนิด | gold | union เจอ | union recall | ระบบดีที่สุด |")
    w("|---|---|---|---|---|---|")
    hard = sorted(
        ((len(union_all[q] & qrels[q]) / len(qrels[q]), q) for q in queries),
        key=lambda t: t[0],
    )
    n_hard = 0
    for u, q in hard:
        if u >= 0.5:
            continue
        n_hard += 1
        b = len(top[best][q] & qrels[q]) / len(qrels[q])
        w(f"| {q} | {etype[q]} | {len(qrels[q])} | {len(union_all[q] & qrels[q])} "
          f"| {u:.3f} | {b:.3f} |")
    w()
    w(f"รวม **{n_hard} / {len(queries)} คำถาม**")
    w()

    # ---- 5. greedy --------------------------------------------------------
    w("## 5. ต้องรวมกี่ระบบถึงเข้าใกล้เพดาน (greedy, งบไม่เท่ากัน)")
    w()
    w("⚠️ ทุกแถวดึงเอกสารมากขึ้นตามจำนวนระบบ — ตอบได้แค่ 'ถ้ายอมดึงลึกขึ้น เนื้อหาโผล่เร็วแค่ไหน'")
    w()
    w("| ระบบใน union | เอกสารที่ดึง | ที่เพิ่มเข้ามา | recall macro | recall micro |")
    w("|---|---|---|---|---|")
    chosen: list[str] = []
    cur = {q: set() for q in queries}
    for _ in range(6):
        bc, bv, bu = None, -1.0, None
        for c in combos:
            if c in chosen:
                continue
            cand = {q: cur[q] | top[c][q] for q in queries}
            v = macro(cand)
            if v > bv:
                bc, bv, bu = c, v, cand
        chosen.append(bc)
        cur = bu
        w(f"| {len(chosen)} | {K*len(chosen)} | {labels[bc]} | {bv:.4f} | {micro(cur):.4f} |")
    w()

    # ---- 6. cross-retriever ----------------------------------------------
    w("## 6. เพิ่ม dense และ BM25 เข้าไปใน union แล้วเพดานขยับไหม")
    w()
    w("หัวข้อนี้ไม่มีในฉบับเดิม ซึ่งตอบได้แค่ว่า 'ข้อมูลอยู่ใน index' สำหรับแขน hybrid ")
    w("อย่างเดียว แต่ dense และ BM25 มีไฟล์ผลเก็บไว้เหมือนกัน — ถ้า union ที่กว้างกว่านี้ ")
    w("ยังหาก้อน B ไม่เจอ ก้อนนั้นก็ไม่ได้เกิดจากการเลือก retriever")
    w()
    ranked_dense = load_ranked(DENSE, "dense", qrels)
    ranked_bm25 = load_ranked(BM25, "bm25", qrels)

    def full_cover(ranked_map: dict) -> list[str]:
        """Live combos whose results cover every query -- a partially-covered
        arm would shrink the union silently instead of raising."""
        return sorted(
            c for c in set(ranked_map) & live
            if set(ranked_map[c]) == set(queries)
            and all(len(ranked_map[c][q]) == K for q in queries)
        )

    dense_combos = full_cover(ranked_dense)
    dense_dropped = len(set(ranked_dense) & live) - len(dense_combos)
    bm25_live = full_cover(ranked_bm25)
    bm25_dropped = len(set(ranked_bm25) & live) - len(bm25_live)
    # BM25 reads the lexical index, which depends on the chunker only -- the two
    # e5 variants per chunker share chunks and therefore return identical hits.
    # Deduplicate by result, not by assumption, so a surprise shows up as a FAIL.
    bm25_by_chunker: dict[str, str] = {}
    bm25_dupe_ok = True
    for c in bm25_live:
        ch = manifests[c]["chunker"]["type"]
        if ch in bm25_by_chunker:
            prev = bm25_by_chunker[ch]
            if any(ranked_bm25[c][q] != ranked_bm25[prev][q] for q in queries):
                bm25_dupe_ok = False
        else:
            bm25_by_chunker[ch] = c
    bm25_combos = sorted(bm25_by_chunker.values())

    w(f"- dense: **{len(dense_combos)}** combo (ตกเกณฑ์ความครบถ้วน {dense_dropped}) · "
      f"BM25: {len(bm25_live)} combo ที่มีผล (ตกเกณฑ์ {bm25_dropped}) แต่ยุบเหลือ "
      f"**{len(bm25_combos)}** ตัวจริง — BM25 อ่าน lexical index ซึ่งขึ้นกับ chunker "
      f"อย่างเดียว ตรวจแล้วว่าผลซ้ำกันจริง: "
      f"{'ใช่' if bm25_dupe_ok else '**ไม่ใช่ — ต้องสอบสวน**'}")
    w()

    def union_of(*groups) -> dict[str, set[str]]:
        out = {}
        for q in queries:
            s: set[str] = set()
            for ranked_map, cs in groups:
                for c in cs:
                    s |= set(ranked_map[c][q][:K])
            out[q] = s
        return out

    u_h = union_all
    u_hd = union_of((ranked_hyb_all, combos), (ranked_dense, dense_combos))
    u_hdb = union_of((ranked_hyb_all, combos), (ranked_dense, dense_combos),
                     (ranked_bm25, bm25_combos))
    w("| union | ระบบ | เอกสารที่ดึง | macro | micro | คู่ที่ยังไม่เจอ |")
    w("|---|---|---|---|---|---|")
    for name, u, n_sys in (
        (f"hybrid ({len(combos)})", u_h, len(combos)),
        (f"hybrid + dense ({len(combos)}+{len(dense_combos)})", u_hd,
         len(combos) + len(dense_combos)),
        (f"hybrid + dense + BM25 (+{len(bm25_combos)})", u_hdb,
         len(combos) + len(dense_combos) + len(bm25_combos)),
    ):
        found = sum(len(u[q] & qrels[q]) for q in queries)
        w(f"| {name} | {n_sys} | {K*n_sys} | {macro(u):.4f} | {micro(u):.4f} | {n_pairs-found} |")
    w()
    struct_floor = n_pairs - sum(len(u_hdb[q] & qrels[q]) for q in queries)
    w(f"**คู่ที่ไม่มี retriever ใดเลยดึงขึ้นมาติด top-{K} ได้: {struct_floor} "
      f"({struct_floor/n_pairs:.1%} ของทั้งหมด)** — นี่คือก้อน B ฉบับที่รัดกุมกว่าหัวข้อ 2 ")
    w("เพราะมันตัดคำแก้ตัวเรื่อง 'เลือก retriever ผิด' ออกไปด้วย")
    w()

    # ---- 7. what the hard course pairs actually contain --------------------
    # Section 4 says where the ceiling is low; it cannot say why. `course` owns
    # most of those rows, and the qrels for that type come from a literal
    # name match, so the evidence behind each pair can be re-read directly.
    w("## 7. หลักฐานจริงหลังคู่ course ที่ union หาไม่เจอ")
    w()
    w("หัวข้อนี้ไม่มีในฉบับเดิมเช่นกัน หัวข้อ 4 บอกได้แค่ว่าเพดานต่ำตรงไหน แต่บอกไม่ได้ว่าทำไม ")
    w("qrels ของ `course` มาจากการจับชื่อวิชาแบบตรงตัว จึงย้อนไปอ่านหลักฐานของแต่ละคู่ได้เลย ")
    w(f"อ่านจาก `chunks.parquet` ของ combo ที่ดีที่สุด (`{labels[best]}`) — เป็นตัวบทที่ retriever ")
    w("เห็นจริง ไม่ใช่ไฟล์ต้นฉบับ")
    w()
    df = pd.read_parquet(INDEX_DIR / best / "chunks.parquet", columns=["resolution_id", "text"])
    text_of: dict[str, str] = {
        rid: " ".join(g["text"].tolist()) for rid, g in df.groupby("resolution_id")
    }

    def evidence(name: str, rid: str) -> str:
        """PREREQ = every occurrence sits inside another course's prerequisite
        line; ROW = at least one occurrence stands as an entry in its own right."""
        txt = text_of.get(rid, "")
        pat = re.escape(name).replace(r"\ ", r"\s+")
        hits = list(re.finditer(pat, txt, re.I))
        if not hits:
            return "NONE"
        kinds = {
            "PREREQ" if "PREREQUISITE" in txt[max(0, m.start() - 80):m.start()].upper()
            else "ROW"
            for m in hits
        }
        return "PREREQ" if kinds == {"PREREQ"} else "ROW"

    tally = collections.Counter()
    per_q: list[tuple[str, int, int, int, int, float]] = []
    for q in course_qs:
        counts = collections.Counter(evidence(course_name[q], r) for r in qrels[q])
        tally.update(counts)
        per_q.append((
            course_name[q], len(qrels[q]), counts["ROW"], counts["PREREQ"], counts["NONE"],
            len(union_all[q] & qrels[q]) / len(qrels[q]),
        ))
    c_pairs = sum(tally.values())
    w(f"- คู่ของคำถาม course ทั้งหมด **{c_pairs}** · ปรากฏเป็นรายการวิชาของตัวเอง "
      f"**{tally['ROW']}** ({tally['ROW']/c_pairs:.1%}) · ปรากฏเฉพาะในบรรทัด "
      f"`PREREQUISITE:` ของ**วิชาอื่น** **{tally['PREREQ']}** ({tally['PREREQ']/c_pairs:.1%}) · "
      f"ไม่พบชื่อวิชาแบบตรงตัวเลย **{tally['NONE']}** ({tally['NONE']/c_pairs:.1%})")
    w()
    w("| วิชา | gold | เป็นรายการเอง | PREREQ ของวิชาอื่น | ไม่พบตรงตัว | union recall |")
    w("|---|---|---|---|---|---|")
    for nm, g, r_, p_, n_, u in sorted(per_q, key=lambda t: (-t[3] / t[1], t[5]))[:8]:
        w(f"| {nm} | {g} | {r_} | {p_} | {n_} | {u:.3f} |")
    w()
    prereq_heavy = [t for t in per_q if t[3] / t[1] >= 0.5]
    rest = [t for t in per_q if t[3] / t[1] < 0.5]
    w(f"**{len(prereq_heavy)} คำถาม** มีหลักฐานเป็น `PREREQUISITE` ของวิชาอื่นตั้งแต่ครึ่งหนึ่งขึ้นไป "
      f"union recall เฉลี่ยของกลุ่มนี้ **{np.mean([t[5] for t in prereq_heavy]):.4f}** "
      f"เทียบกับ **{np.mean([t[5] for t in rest]):.4f}** ของ course ที่เหลือ")
    w()
    w("**แต่ค่าเฉลี่ยนี้อธิบายอะไรไม่ได้เลย และนี่คือประเด็นของหัวข้อนี้** ในกลุ่มเดียวกันนั้น ")
    w("`SIGNALS AND SYSTEMS` มีหลักฐานเป็น PREREQUISITE 9 จาก 10 ใบ แต่ union recall = **1.000** ")
    w("ส่วน `ELECTRONICS ENGINEERING 1` เป็น PREREQUISITE ทั้ง 10 ใบ ได้ **0.900** — หลักฐานแบบ ")
    w("PREREQUISITE จึงถูกดึงขึ้นมาได้ตามปกติ **สมมติฐานว่า 'มันคือ needle ที่หาไม่เจอ' ถูกข้อมูล ")
    w("ปฏิเสธ** เก็บสถิติ 9.5% ไว้ในฐานะหมวดหมู่ที่มีอยู่จริง ไม่ใช่ในฐานะคำอธิบาย")
    w()

    # The single 0.000 row needs its own explanation, and it generalises: two
    # course names where one is a token-prefix of the other are one edit apart
    # in query space but were labelled by exact-token match, so their qrels can
    # be disjoint. Detect the shape rather than hard-coding the pair.
    #
    # CORRECTED 2026-08-09. This block used to call the disjoint qrels a
    # "labelling artifact" and subtract those pairs from the structural floor,
    # citing 76 rather than 84. tools/eval/audit_gold_anchor_ambiguity.py
    # measured the premise and refuted it: 01046707 and 01306023 are genuinely
    # different courses, every one of the plural query's 8 relevant documents
    # really does contain the phrase `CONTROL SYSTEMS`, and the qrels reproduce
    # exactly from the code tags. Nothing is mislabelled. The query is simply
    # under-specified -- its name competes with 57 documents naming other
    # courses -- so the pairs are unreachable by name matching, not unreachable
    # by mistake, and they belong IN the floor. The subtraction is withdrawn.
    w("### คำถามที่ union ได้ 0.000: ชื่อไม่พอจะระบุตัววิชา")
    w()
    prefix_pairs = [
        (a, b) for a in course_name.values() for b in course_name.values()
        if a != b and b.startswith(a)
    ]
    q_of = {v: k for k, v in course_name.items()}
    w("| ชื่อสั้น | ชื่อยาว | gold ซ้อนกัน | union ของคำถาม 'ยาว' ดึงมา | ตรง gold ตัวเอง | ตรง gold ของ 'สั้น' |")
    w("|---|---|---|---|---|---|")
    for a, b in prefix_pairs:
        qa, qb = q_of[a], q_of[b]
        ub = union_all[qb]
        w(f"| {a} | {b} | {len(qrels[qa] & qrels[qb])} | {len(ub)} "
          f"| **{len(ub & qrels[qb])}** | {len(ub & qrels[qa])} |")
    w()
    w("อ่านตามตาราง: ชื่อวิชาสองอันนี้ต่างกันแค่ตัว `S` ตัวเดียว ระบบทุกตัวจึงดึงเอกสาร ")
    w("ย่านเดียวกันมาให้ทั้งคู่ — แต่ qrels แบ่งย่านนั้นให้คำถาม 'สั้น' ไปทั้งหมด ")
    w("**โดยไม่ซ้อนกันเลยแม้แต่ใบเดียว** คำถาม 'ยาว' จึงตอบถูกไม่ได้ด้วยระบบใด ๆ ")
    w("ที่ยังแยกสองคำถามนี้ไม่ออก")
    w()
    lost = sum(len(qrels[q_of[b]]) for _, b in prefix_pairs)
    w(f"**แก้ไข 2026-08-09 — เดิมที่นี่เคยเรียกสิ่งนี้ว่า 'ผลของการติดป้าย' แล้วหัก {lost} คู่ ")
    w(f"ออกจาก {struct_floor} คู่ เหลือ {struct_floor - lost} ")
    w("คู่ ข้อสมมตินั้นถูกวัดแล้วและผิด** — `tools/eval/audit_gold_anchor_ambiguity.py` ")
    w("พบว่า `01046707` กับ `01306023` เป็นคนละวิชากันจริง qrels สร้างขึ้นใหม่จาก tag รหัส ")
    w("ได้ตรงเป๊ะทั้ง 33 ข้อ และเอกสาร gold ของคำถาม 'ยาว' **มีข้อความชื่อวิชาอยู่ครบทุกใบ** ")
    w("จึงไม่มีอะไรติดป้ายผิดเลย สิ่งที่เกิดขึ้นคือ*ตัวคำถามระบุตัววิชาไม่พอ* — ชื่อของมัน ")
    w("ปรากฏในเอกสารของวิชาอื่นอีก 57 ใบ คู่เหล่านี้จึงเข้าไม่ถึงด้วย *การจับคู่ชื่อ* ")
    w("ไม่ใช่เข้าไม่ถึงเพราะความผิดพลาด และต้อง**นับรวมอยู่ในพื้น**")
    w()
    w(f"**เพดานเชิงโครงสร้างที่ควรอ้างจึงเป็น {struct_floor} คู่ "
      f"({struct_floor/n_pairs:.1%}) ไม่ใช่ {struct_floor - lost} คู่** ")
    w(f"โดยในนั้น {lost} คู่เป็นชนิด 'ชื่อกำกวม' ซึ่งวัดแยกไว้ใน `gold_anchor_ambiguity.md` ")
    w("ที่เหลือยังไม่ได้แยกว่าเป็น OCR, chunk หรือแค่อันดับเกิน 10 (ต้องรัน k=50)")
    w()

    # ---- provenance -------------------------------------------------------
    w("## Self-checks")
    w()
    w("| check | ผล | รายละเอียด |")
    w("|---|---|---|")
    for name, ok, detail in checks:
        w(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    w()
    w("S3 คือจุดยึด: ถ้า harness นี้อ่านไฟล์ผลแบบเดียวกับ eval ที่ตีพิมพ์ไปแล้ว ")
    w("recall@10 ต่อ combo ต้องตรงกันทุกตัว ตัวเลขที่ต่างจากเอกสารเก่าหลังจากนั้น ")
    w("จึงเป็นผลของการออกแบบ ไม่ใช่ของการอ่านข้อมูลผิด")
    w()

    # S5 is appended mid-run (it needs section 1c's numbers), so the gate runs
    # a second time here -- a late FAIL must not reach the published file either.
    for name, ok, detail in checks[len(early):]:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
    if not all(ok for _, ok, _ in checks):
        print("\nself-check failed; refusing to publish numbers", file=sys.stderr)
        return 1

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
