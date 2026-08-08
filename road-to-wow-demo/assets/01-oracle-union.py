"""ตั๋ว 01 (บางส่วน) — เพดานของ index family ปัจจุบัน: oracle-union recall@10

ถ้ารวม top-10 ของทุก combo เข้าด้วยกันแล้วยังหามติที่ควรเจอไม่ครบ
แปลว่าที่พลาดไปนั้น **ไม่มีวิธีจัดอันดับใดช่วยได้** ต้องแก้ที่ chunk/OCR/label
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
import numpy as np

REPO = Path(r"C:/Users/khtha/OneDrive/Desktop/Code/RAG")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))
from rag_lab.query_sets import load_gold_query_set          # noqa: E402
from embedder_matrix_9way import _INDEX_DIR, build_combo_to_chunker_embedder  # noqa: E402

import yaml as _yaml

HYB = REPO / "data" / "results" / "gold_hybrid_73det"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
K = 10
OUT = Path(__file__).with_name("oracle_union_out.md")

entries = load_gold_query_set(GOLD)
qrels = {e.query: set(e.relevant_resolution_ids) for e in entries}
queries = [e.query for e in entries]
_raw = _yaml.safe_load(Path(GOLD).read_text(encoding="utf-8"))
etype = {d["query"]: d.get("entity_type", "?") for d in _raw}

_c2ce = build_combo_to_chunker_embedder(_INDEX_DIR)
# คีย์ของ map มี suffix "__dense" ต่อท้าย — ตัดทิ้งให้เหลือ prefix 4 ส่วน
combo2ce = {"__".join(k.split("__")[:4]): v for k, v in _c2ce.items()}


def label_of(prefix: str) -> str:
    ce = combo2ce.get(prefix)
    return f"{ce[0]} x {ce[1]}" if ce else prefix

# per (combo, query) -> ranked list of resolution_ids (ไฟล์เก็บไว้แค่ rank <= 10)
ranked: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
for f in HYB.glob("*.json"):
    d = json.loads(f.read_text(encoding="utf-8"))
    q = d["query"]
    if q not in qrels:
        continue
    prefix = "__".join(f.stem.split("__")[:4])
    ranked[prefix][q] = [r["resolution_id"] for r in sorted(d["results"], key=lambda r: r["rank"])]


def topn(combo: str, q: str, n: int) -> set[str]:
    return set(ranked[combo][q][:n])


top: dict[str, dict[str, set[str]]] = {
    c: {q: topn(c, q, K) for q in ranked[c]} for c in ranked
}

combos = sorted(top)
labels = {c: label_of(c) for c in combos}

lines: list[str] = []


def w(s: str = "") -> None:
    lines.append(s)

w("# ตั๋ว 01 (บางส่วน) — เพดานของระบบปัจจุบัน: oracle-union recall@10")
w()
w(f"combo ทั้งหมด {len(combos)} ชุด · คำถาม {len(queries)} ข้อ · k={K}")
w()
w("**นิยาม**: oracle-union = สมมติว่ามี oracle เลือกให้ได้ว่าคำถามไหนควรใช้ระบบไหน")
w("(ที่จริงคือรวม top-10 ของทุกระบบเข้าด้วยกัน) — นี่คือ **เพดานบนที่หลวมมาก**")
w("ไม่มีวิธี re-rank / fine-tune / ensemble ใด ๆ ที่ทำได้เกินเส้นนี้ ตราบใดที่ยังใช้ index ชุดนี้และ k=10")
w()


def recall_of(sets_by_q: dict[str, set[str]]) -> np.ndarray:
    return np.array([len(sets_by_q[q] & qrels[q]) / len(qrels[q]) for q in queries])


# 1. best single combo
w("## 1. ระบบเดี่ยวที่ดีที่สุด vs เพดาน")
w()
per_combo = {c: recall_of(top[c]) for c in combos}
best = max(per_combo, key=lambda c: per_combo[c].mean())

sem_combos = [c for c in combos if combo2ce.get(c, ("", ""))[0] == "semantic"]
union_all = {q: set().union(*(top[c][q] for c in combos)) for q in queries}
union_sem = {q: set().union(*(top[c][q] for c in sem_combos)) for q in queries}
r_all, r_sem = recall_of(union_all), recall_of(union_sem)
r_best = per_combo[best]

w("⚠️ **แถวเหล่านี้ดึงเอกสารไม่เท่ากัน** — union ของ 36 combo ดึงถึง 360 ใบก่อนตัดซ้ำ")
w("จึงเทียบกันตรง ๆ ไม่ได้ ดูหัวข้อ 1b สำหรับการเทียบที่งบเท่ากัน")
w()
w("| ระบบ | เอกสารที่ดึง (ก่อนตัดซ้ำ) | recall | ห่างจากระบบที่ดีที่สุด |")
w("|---|---|---|---|")
w(f"| ระบบเดี่ยวที่ดีที่สุด: **{labels[best]}** | 10 | **{r_best.mean():.4f}** | — |")
w(f"| union ของ {len(sem_combos)} combo ที่ใช้ semantic chunker | {10*len(sem_combos)} | {r_sem.mean():.4f} | {r_sem.mean()-r_best.mean():+.4f} |")
w(f"| union ของทั้ง {len(combos)} combo | {10*len(combos)} | {r_all.mean():.4f} | {r_all.mean()-r_best.mean():+.4f} |")
w(f"| เพดานทฤษฎี | — | 1.0000 | {1-r_best.mean():+.4f} |")
w()

# 1b. equal-budget comparison — งบเท่ากันจริง ๆ
w("## 1b. เทียบที่ **งบเท่ากัน** — 10 เอกสารเท่ากันทุกแถว")
w()
w("ตัวเลข 0.8130 ข้างบนได้มาจากการดึง **20 เอกสาร** (10 จากแต่ละระบบ) ไม่ใช่ 10")
w("ส่วนหนึ่งของกำไรจึงมาจาก 'ดึงลึกขึ้น' ไม่ใช่ 'ความหลากหลาย' — ตารางนี้แยกสองอย่างนั้นออก")
w("โดยบังคับให้ทุกแถวส่ง **10 เอกสาร** ให้ LLM เท่ากัน: m ระบบ ระบบละ 10/m อันดับแรก")
w()
w("| จำนวนระบบ | ดึงระบบละ | เอกสารรวม (ก่อนตัดซ้ำ) | recall | ระบบที่เลือก (greedy) |")
w("|---|---|---|---|---|")


def greedy_at(depth: int, m: int):
    chosen_l: list[str] = []
    cur_l = {q: set() for q in queries}
    val = 0.0
    for _ in range(m):
        bc, bv, bu = None, -1.0, None
        for c in combos:
            if c in chosen_l:
                continue
            cand = {q: cur_l[q] | topn(c, q, depth) for q in queries}
            v = recall_of(cand).mean()
            if v > bv:
                bc, bv, bu = c, v, cand
        chosen_l.append(bc)
        cur_l = bu
        val = bv
    return chosen_l, val, cur_l


_eq = {}
for m, depth in ((1, 10), (2, 5), (5, 2)):
    ch, v, sets_m = greedy_at(depth, m)
    _eq[m] = (v, ch, sets_m)
    w(f"| {m} | {depth} | {m*depth} | **{v:.4f}** | {', '.join(labels[c] for c in ch)} |")
w()
_d = _eq[2][0] - _eq[1][0]
w(f"**ผลของความหลากหลายล้วน ๆ ที่งบ 10 เอกสาร = {_d:+.4f}**")
w("(เทียบกับ +0.1195 ที่รายงานไว้ตอนแรก ซึ่งรวมผลของการดึงลึกขึ้นเป็นสองเท่าไว้ด้วย)")
w()
w("| ทางเลือก | เอกสารที่ส่งให้ LLM | recall |")
w("|---|---|---|")
w(f"| ระบบเดียว top-10 | 10 | {_eq[1][0]:.4f} |")
w(f"| สองระบบ top-5 | 10 | {_eq[2][0]:.4f} |")
w(f"| ห้าระบบ top-2 | 10 | {_eq[5][0]:.4f} |")
_ch20, _v20, _sets20 = greedy_at(10, 2)
w(f"| สองระบบ top-10 (ที่รายงานไว้เดิม) | 20 | {_v20:.4f} |")
w()
w("→ **ที่งบเท่ากัน การรวมหลายระบบแพ้ระบบเดียว** กำไร +0.1195 ที่รายงานไว้เดิม")
w("มาจาก 'ดึงเอกสารเป็นสองเท่า' ล้วน ๆ ไม่ได้มาจากความหลากหลาย")
w()

# 1c. สองเพดานที่งบ 10 ใบเท่าเดิม — อันนี้คือของจริงที่ต้องดู
w("## 1c. สองเพดานที่ยัง 'ส่ง 10 ใบ' เท่าเดิม")
w()
w("ถ้าความหลากหลายแบบแบ่งงบไม่ได้ผล ยังเหลืออีกสองวิธีที่ส่ง 10 ใบเท่าเดิม")
w()
# (i) oracle เลือก combo ทั้งชุดให้แต่ละคำถาม — ดึง 10 ใบจากระบบเดียว
_sel = np.array([
    max(len(top[c][q] & qrels[q]) for c in combos) / len(qrels[q]) for q in queries
])
# (ii) union 2 ระบบ (20 ใบ) แล้ว rerank สมบูรณ์แบบเหลือ 10 ใบ
_rr2 = np.array([
    min(len(_sets20[q] & qrels[q]), K) / len(qrels[q]) for q in queries
])
# (iii) union ทุกระบบ (360 ใบ) แล้ว rerank สมบูรณ์แบบเหลือ 10 ใบ
_rrall = np.array([
    min(len(union_all[q] & qrels[q]), K) / len(qrels[q]) for q in queries
])
w("| วิธี | ใบที่ส่งให้ LLM | ใบที่ต้องดึง+จัดอันดับ | recall@10 | ห่างจากระบบเดียว |")
w("|---|---|---|---|---|")
w(f"| ระบบเดียวที่ดีที่สุด (ของจริงวันนี้) | 10 | 10 | {r_best.mean():.4f} | — |")
w(f"| **oracle เลือกระบบให้แต่ละคำถาม** | 10 | 10 | **{_sel.mean():.4f}** | {_sel.mean()-r_best.mean():+.4f} |")
w(f"| union 2 ระบบ + reranker สมบูรณ์แบบ | 10 | 20 | **{_rr2.mean():.4f}** | {_rr2.mean()-r_best.mean():+.4f} |")
w(f"| union 36 ระบบ + reranker สมบูรณ์แบบ | 10 | 360 | {_rrall.mean():.4f} | {_rrall.mean()-r_best.mean():+.4f} |")
w()
w("ทั้งสามแถวล่างเป็น **oracle** — ไม่มีระบบจริงทำได้เท่านี้ แต่มันบอกว่า *เพดานของแนวทางนั้นอยู่ตรงไหน*")
w("แถวที่สองสำคัญที่สุด เพราะมันไม่ต้องดึงเพิ่มเลย แค่ **เลือกให้ถูก**")
w()

# 1d. router ที่สร้างได้จริง — leave-one-out
w("## 1d. เพดาน oracle vs **นโยบายที่สร้างได้จริง**")
w()
w("0.8097 ข้างบนเป็น oracle ที่ 'รู้คำตอบอยู่แล้ว' — เท่ากับนโยบาย 73 พารามิเตอร์ที่ fit บนจุด 73 จุด")
w("ตารางนี้วัด router ที่สร้างได้จริง: **เลือก combo ตามชนิดคำถาม** (person/program/faculty)")
w("ด้วย leave-one-out — ตัดสินคำถามข้อไหนก็ห้ามใช้ข้อนั้นเลือก combo")
w()
per_q = {c: {q: len(top[c][q] & qrels[q]) / len(qrels[q]) for q in queries} for c in combos}

_r_loo, _s_loo, _flips = [], [], []
for q in queries:
    peers_t = [p for p in queries if etype[p] == etype[q] and p != q]
    peers_a = [p for p in queries if p != q]
    bc_t = max(combos, key=lambda c: float(np.mean([per_q[c][p] for p in peers_t])))
    bc_a = max(combos, key=lambda c: float(np.mean([per_q[c][p] for p in peers_a])))
    _r_loo.append(per_q[bc_t][q])
    _s_loo.append(per_q[bc_a][q])
    _flips.append(bc_t != bc_a)
router_loo, single_loo = float(np.mean(_r_loo)), float(np.mean(_s_loo))

w("| นโยบาย | ส่งให้ LLM | recall | ห่างจาก baseline |")
w("|---|---|---|---|")
w(f"| ระบบเดียว (เลือกแบบ LOO) | 10 | {single_loo:.4f} | — |")
w(f"| **router ตามชนิดคำถาม (LOO)** | 10 | **{router_loo:.4f}** | **{router_loo-single_loo:+.4f}** |")
w(f"| oracle เลือกต่อคำถาม (เพดาน) | 10 | {_sel.mean():.4f} | {_sel.mean()-single_loo:+.4f} |")
w()
w(f"router เปลี่ยนใจจาก baseline ใน {sum(_flips)}/{len(queries)} คำถาม")
w(f"— เก็บได้ **{(router_loo-single_loo)/max(_sel.mean()-single_loo,1e-9):.1%}** ของ headroom ที่ oracle เห็น")
w()
_diffs = np.array(_r_loo) - np.array(_s_loo)
w(f"คำถามที่ดีขึ้น {int((_diffs>0).sum())} · แย่ลง {int((_diffs<0).sum())} · เท่าเดิม {int((_diffs==0).sum())}")
w()

# 2. decomposition of the missing 31%
w("## 2. แยก 'ส่วนที่พลาด' ออกเป็นสองก้อน")
w()
w("นับเป็น **รายคู่ (query, resolution)** ไม่ใช่ค่าเฉลี่ยต่อคำถาม")
w("(ตัวเลขจึงไม่เท่ากับ recall@10 ข้างบน ซึ่งเป็น macro-average ต่อคำถาม)")
w()
tot_pairs = sum(len(qrels[q]) for q in queries)
found_best = sum(len(top[best][q] & qrels[q]) for q in queries)
found_all = sum(len(union_all[q] & qrels[q]) for q in queries)
w(f"- คู่ (query, resolution) ทั้งหมด: **{tot_pairs}**")
w(f"- ระบบที่ดีที่สุดเจอ: {found_best} ({found_best/tot_pairs:.1%})")
w(f"- union ของทุกระบบเจอ: {found_all} ({found_all/tot_pairs:.1%})")
w()
gap_rank = found_all - found_best
gap_struct = tot_pairs - found_all
w(f"| ก้อน | จำนวนคู่ | สัดส่วนของทั้งหมด | สัดส่วนของส่วนที่พลาด | แก้ด้วยอะไร |")
w("|---|---|---|---|---|")
w(f"| **A. การจัดอันดับ** — มีระบบใดระบบหนึ่งเจอแล้ว แต่ระบบที่เลือกใช้ไม่เจอ | {gap_rank} | {gap_rank/tot_pairs:.1%} | {gap_rank/(tot_pairs-found_best):.1%} | reranker / ensemble / เลือก combo ต่อคำถาม |")
w(f"| **B. โครงสร้าง** — ไม่มีระบบไหนเจอเลยใน top-10 | {gap_struct} | {gap_struct/tot_pairs:.1%} | {gap_struct/(tot_pairs-found_best):.1%} | OCR / chunk / label / เพิ่ม k — ไม่ใช่ embedding |")
w()
w("⚠️ ก้อน B ยังแยกไม่ออกว่าเป็น 'อยู่อันดับ 11–50' หรือ 'ไม่ติดเลย' เพราะไฟล์ผลเก็บไว้แค่ rank ≤ 10")
w("ต้องรัน retrieval ใหม่ที่ k=50 ถึงจะแยกได้ — ดูหมายเหตุในแผนที่")
w()

# 3. by entity type
w("## 3. แยกตามชนิดคำถาม")
w()
w("| ชนิด | n | recall ระบบที่ดีที่สุด | recall union | headroom จากการจัดอันดับ |")
w("|---|---|---|---|---|")
for t in sorted({etype[q] for q in queries}):
    qs = [q for q in queries if etype[q] == t]
    b = np.mean([len(top[best][q] & qrels[q]) / len(qrels[q]) for q in qs])
    u = np.mean([len(union_all[q] & qrels[q]) / len(qrels[q]) for q in qs])
    w(f"| {t} | {len(qs)} | {b:.4f} | {u:.4f} | {u-b:+.4f} |")
w()

# 4. queries where even the union fails badly
w("## 4. คำถามที่แม้แต่ union ก็ยังพลาดหนัก (recall union < 0.5)")
w()
w("คำถามเหล่านี้คือที่ที่ **embedding ไม่ใช่ปัญหา** — ส่งต่อให้ตั๋ว 01 ตรวจ OCR/label เป็นรายใบ")
w()
w("| คำถาม | ชนิด | gold | union เจอ | recall union | recall ระบบดีที่สุด |")
w("|---|---|---|---|---|---|")
rows = []
for q in queries:
    u = len(union_all[q] & qrels[q]) / len(qrels[q])
    if u < 0.5:
        rows.append((u, q))
for u, q in sorted(rows):
    b = len(top[best][q] & qrels[q]) / len(qrels[q])
    w(f"| {q} | {etype[q]} | {len(qrels[q])} | {len(union_all[q] & qrels[q])} | {u:.3f} | {b:.3f} |")
w()
w(f"รวม **{len(rows)} / {len(queries)} คำถาม**")
w()

# 5. how many combos does it take
w("## 5. ต้องใช้กี่ระบบถึงจะเข้าใกล้เพดาน (greedy)")
w()
w("เลือกทีละ combo ที่เพิ่ม recall ของ union ได้มากที่สุด")
w()
w("⚠️ แถวล่าง ๆ **ไม่ได้เทียบกับ recall@10** — มันดึงเอกสารมากกว่าหลายเท่า ดูคอลัมน์ 'เอกสารที่ดึง'")
w()
w("| จำนวนระบบใน union | เอกสารที่ดึง (ก่อนตัดซ้ำ) | combo ที่เพิ่มเข้ามา | recall union |")
w("|---|---|---|---|")
chosen: list[str] = []
cur = {q: set() for q in queries}
for _ in range(6):
    bestc, bestv, bestu = None, -1.0, None
    for c in combos:
        if c in chosen:
            continue
        cand = {q: cur[q] | top[c][q] for q in queries}
        v = recall_of(cand).mean()
        if v > bestv:
            bestc, bestv, bestu = c, v, cand
    chosen.append(bestc)
    cur = bestu
    w(f"| {len(chosen)} | {10*len(chosen)} | {labels[bestc]} | {bestv:.4f} |")
w()

OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT)
