"""Ticket 05: minimum detectable effect at n=73, using the repo's own bootstrap."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

REPO = Path(r"C:/Users/khtha/OneDrive/Desktop/Code/RAG")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from rag_lab.query_sets import load_gold_query_set
from embedder_matrix_9way import (
    _INDEX_DIR, build_combo_to_chunker_embedder, bootstrap_pvalue, holm_correct,
)

HYB = REPO / "data" / "results" / "gold_hybrid_73det"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
K = 10
N_BOOT = 10000
SEED = 42
OUT = Path(__file__).with_name("mde_73_out.md")

qs = load_gold_query_set(GOLD)
qrels = {e.query: list(e.relevant_resolution_ids) for e in qs}
import yaml as _yaml
_raw = _yaml.safe_load(Path(GOLD).read_text(encoding="utf-8"))
etype = {d["query"]: d.get("entity_type", "?") for d in _raw}
queries = [e.query for e in qs]

mapping = build_combo_to_chunker_embedder(_INDEX_DIR)  # "<dir>__dense" -> (chunker, embedder)
combo_of = {}
for key, (ch, emb) in mapping.items():
    base = key[: -len("__dense")]
    combo_of[base] = (ch, emb)

# group hybrid result files by combo prefix
files = sorted(HYB.glob("*.json"))
by_combo: dict[str, list[Path]] = {}
for f in files:
    parts = f.stem.split("__")
    prefix = "__".join(parts[:4])  # mode__chunker__embedder__hash
    by_combo.setdefault(prefix, []).append(f)

lines: list[str] = []
def w(s=""):
    lines.append(s)

def recall_vec(paths: list[Path]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    per = {}
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        q = d["query"]
        rel = set(qrels[q])
        top = [r for r in sorted(d["results"], key=lambda r: r["rank"]) if r["rank"] <= K]
        hit = {r["resolution_id"] for r in top} & rel
        per[q] = len(hit) / len(rel)
    return np.array([per[q] for q in queries]), per

w("# ตั๋ว 05 — 73 คำถามพิสูจน์อะไรได้ (ผลคำนวณ)")
w()
w("## 0. ตรวจสอบว่าอ่านข้อมูลตรงกับตัวเลขที่ตีพิมพ์")
w()
w("| combo | recall@10 (mean) |")
w("|---|---|")
summary = {}
for prefix, paths in sorted(by_combo.items()):
    base = prefix
    ch_emb = combo_of.get(base)
    v, per = recall_vec(paths)
    label = f"{ch_emb[0]} x {ch_emb[1]}" if ch_emb else base
    summary[prefix] = (label, v, per)
    w(f"| {label} (`{prefix}`) | {v.mean():.4f} |")
w()

# pick best combo by mean recall
best_prefix = max(summary, key=lambda p: summary[p][1].mean())
best_label, base_vec, base_per = summary[best_prefix]
w(f"ระบบฐานที่ใช้จำลอง: **{best_label}** recall@10 = **{base_vec.mean():.4f}** (`{best_prefix}`)")
w()

rng = np.random.default_rng(SEED)

_IDX_CACHE: dict[int, np.ndarray] = {}

def _idx(n: int) -> np.ndarray:
    if n not in _IDX_CACHE:
        _IDX_CACHE[n] = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    return _IDX_CACHE[n]

def p_of(diffs: np.ndarray) -> float:
    """เหมือน bootstrap_pvalue ของ repo ทุกประการ (n_boot, seed, two-sided percentile)
    ต่างแค่ reuse index matrix เพื่อให้จำลองหลายพันรอบได้"""
    boot = diffs[_idx(len(diffs))].mean(axis=1)
    p_le = float((boot <= 0).mean())
    p_ge = float((boot >= 0).mean())
    return min(2 * min(p_le, p_ge), 1.0)

def simulate(vec: np.ndarray, idxs: np.ndarray, n_gain: int, churn: float, n_sim: int, rng) -> tuple[float, float]:
    """flip n_gain missed golds -> found; churn*n_gain found golds -> missed.
    returns (mean net delta, power)"""
    gold_n = np.array([len(qrels[queries[i]]) for i in idxs])
    v = vec[idxs]
    can_gain = np.where(v < 1.0)[0]
    can_lose = np.where(v > 0.0)[0]
    deltas, sig = [], 0
    n_lose = int(round(churn * n_gain))
    for _ in range(n_sim):
        d = np.zeros(len(v))
        g = rng.choice(can_gain, size=min(n_gain, len(can_gain)), replace=False)
        for i in g:
            d[i] += 1.0 / gold_n[i]
        if n_lose:
            l = rng.choice(can_lose, size=min(n_lose, len(can_lose)), replace=False)
            for i in l:
                d[i] -= 1.0 / gold_n[i]
        new = np.clip(v + d, 0, 1)
        diffs = new - v
        deltas.append(diffs.mean())
        if p_of(diffs) < 0.05:
            sig += 1
    return float(np.mean(deltas)), sig / n_sim

N_SIM = 200
all_idx = np.arange(len(queries))

w("## 1. MDE: ต้องพลิกกี่คำถาม ถึงจะได้ p < 0.05")
w()
w("จำลอง: สุ่มพลิก 'มติที่ควรเจอแต่ไม่เจอ' ให้เจอเพิ่ม n_gain ก้อน และ (ถ้ามี churn) ทำให้ที่เคยเจอหลุดไป churn x n_gain ก้อน")
w(f"แล้วรัน paired bootstrap แบบเดียวกับ repo (n_boot={N_BOOT}, seed={SEED}, two-sided percentile) จำลอง {N_SIM} รอบต่อจุด")
w()
for churn in (0.0, 0.25, 0.5):
    w(f"### churn = {churn:.2f} (ทุก {1/churn if churn else float('inf'):.0f} ก้อนที่ได้ มี 1 ก้อนที่หลุด)" if churn else "### churn = 0 (ดีขึ้นอย่างเดียว ไม่มีอะไรแย่ลง — กรณีในอุดมคติ)")
    w()
    w("| n_gain | Δ recall@10 เฉลี่ย | power (สัดส่วนที่ p<0.05) | power หลัง Holm m=2 | m=9 |")
    w("|---|---|---|---|---|")
    for n_gain in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30):
        r = np.random.default_rng(1234 + n_gain)
        gold_n = np.array([len(qrels[q]) for q in queries])
        can_gain = np.where(base_vec < 1.0)[0]
        can_lose = np.where(base_vec > 0.0)[0]
        n_lose = int(round(churn * n_gain))
        ds, s1, s2, s9 = [], 0, 0, 0
        for _ in range(N_SIM):
            d = np.zeros(len(base_vec))
            g = r.choice(can_gain, size=min(n_gain, len(can_gain)), replace=False)
            for i in g:
                d[i] += 1.0 / gold_n[i]
            if n_lose:
                l = r.choice(can_lose, size=min(n_lose, len(can_lose)), replace=False)
                for i in l:
                    d[i] -= 1.0 / gold_n[i]
            new = np.clip(base_vec + d, 0, 1)
            diffs = new - base_vec
            ds.append(diffs.mean())
            p = p_of(diffs)
            s1 += p < 0.05
            s2 += 2 * p < 0.05
            s9 += 9 * p < 0.05
        w(f"| {n_gain} | {np.mean(ds):+.4f} | {s1/N_SIM:.2f} | {s2/N_SIM:.2f} | {s9/N_SIM:.2f} |")
    w()

w("## 2. subgroup: person (n=30) / program (n=30) / faculty (n=13)")
w()
for et in ("person", "program", "faculty_adjunct_aggregate"):
    idxs = np.array([i for i, q in enumerate(queries) if etype[q] == et])
    sub = base_vec[idxs]
    w(f"### {et} — n={len(idxs)}, recall@10 = {sub.mean():.4f}")
    w()
    w("| n_gain | Δ เฉลี่ย (churn 0) | power (churn 0) | Δ เฉลี่ย (churn .25) | power (churn .25) |")
    w("|---|---|---|---|---|")
    gold_n = np.array([len(qrels[queries[i]]) for i in idxs])
    can_gain = np.where(sub < 1.0)[0]
    can_lose = np.where(sub > 0.0)[0]
    for n_gain in (1, 2, 3, 4, 5, 8, 10):
        if n_gain > len(can_gain):
            break
        row = [str(n_gain)]
        for churn in (0.0, 0.25):
            r = np.random.default_rng(999 + n_gain)
            n_lose = int(round(churn * n_gain))
            ds, s = [], 0
            for _ in range(N_SIM):
                d = np.zeros(len(sub))
                for i in r.choice(can_gain, size=n_gain, replace=False):
                    d[i] += 1.0 / gold_n[i]
                if n_lose:
                    for i in r.choice(can_lose, size=min(n_lose, len(can_lose)), replace=False):
                        d[i] -= 1.0 / gold_n[i]
                diffs = np.clip(sub + d, 0, 1) - sub
                ds.append(diffs.mean())
                s += p_of(diffs) < 0.05
            row += [f"{np.mean(ds):+.4f}", f"{s/N_SIM:.2f}"]
        w("| " + " | ".join(row) + " |")
    w()

w("## 3. โครงสร้างของชุดคำถาม (ทำไม Δ ถึงไม่เป็นก้อนละ 1/73)")
w()
gold_n = np.array([len(qrels[q]) for q in queries])
w(f"- จำนวนมติที่ถูกต้องต่อคำถาม: min={gold_n.min()}, median={np.median(gold_n):.0f}, max={gold_n.max()}, mean={gold_n.mean():.2f}")
w(f"- รวมคู่ (query, resolution) ทั้งหมด = {gold_n.sum()}")
w(f"- คำถามที่ recall=1.0 (เจอครบ): {(base_vec==1.0).sum()} / {len(base_vec)}")
w(f"- คำถามที่ recall=0.0 (ไม่เจอเลย): {(base_vec==0.0).sum()} / {len(base_vec)}")
w(f"- คำถามที่เจอบางส่วน: {((base_vec>0)&(base_vec<1)).sum()}")
w(f"- การพลิก 1 มติในคำถามที่มี gold {gold_n.min()} ก้อน ทำให้ mean recall ขยับ {1/gold_n.min()/len(queries):+.4f}; ถ้ามี {gold_n.max()} ก้อน ขยับแค่ {1/gold_n.max()/len(queries):+.4f}")
w()

lab2vec = {summary[p][0]: summary[p][1] for p in summary}

w("## 3b. ความไม่สมมาตรของชุดคำถามนี้ — เสีย 1 ก้อน เจ็บกว่าได้ 1 ก้อน ราวสองเท่า")
w()
w("นับเป็นรายคู่ (query, resolution): ก้อนที่ **ยังไม่เจอ** (มีให้ได้เพิ่ม) กับก้อนที่ **เจอแล้ว** (มีให้เสีย)")
w("มูลค่าของแต่ละก้อนคือ 1/gold_n ของคำถามนั้น")
_miss = np.rint(gold_n * (1.0 - base_vec)).astype(int)
_hit = np.rint(gold_n * base_vec).astype(int)
_gval = (_miss / gold_n).sum() / _miss.sum()
_lval = (_hit / gold_n).sum() / _hit.sum()
w()
w(f"- ก้อนที่ยังไม่เจอ: {_miss.sum()} ก้อน มูลค่าเฉลี่ยก้อนละ **{_gval:.4f}**")
w(f"- ก้อนที่เจอแล้ว: {_hit.sum()} ก้อน มูลค่าเฉลี่ยก้อนละ **{_lval:.4f}**")
w(f"- อัตราส่วน = **{_lval/_gval:.2f} เท่า**")
w()
w("เหตุผล: ก้อนที่ยังไม่เจอกระจุกอยู่ในคำถามที่มี gold เยอะ (แต่ละก้อนจึงมีค่าน้อย)")
w("ส่วนก้อนที่เจอแล้วกระจุกอยู่ในคำถามที่มี gold น้อย (แต่ละก้อนจึงมีค่ามาก)")
w()
w(f"**ผลตรง**: ถ้าระบบใหม่พลิกก้อนแบบสุ่มด้วย churn 0.75 (ได้ 4 เสีย 3) ผลสุทธิจะเป็น")
w(f"{4*_gval - 3*_lval:+.4f} ต่อรอบ — **ติดลบ** ระบบที่ churn สูงแบบสุ่มจึงไม่ได้แค่พิสูจน์ยาก แต่ **แย่ลงจริง**")
w("ระบบที่ดีขึ้นจริงจึงต้องเลือกก้อนที่ได้ ไม่ใช่พลิกแบบสุ่ม")
w()

w("## 4. ต้องมีกี่คำถามถึงจะพิสูจน์ผลขนาดที่ 'เกิดขึ้นจริงแล้ว' ได้")
w()
w("ไม่จำลองผลกระทบสมมติอีกต่อไป — ใช้ **เวกเตอร์ diff ต่อคำถามที่วัดได้จริง** จากคู่ระบบที่มีอยู่")
w("แล้วสุ่มซ้ำ (resample with replacement) ให้ได้ n คำถาม เพื่อตอบว่า 'ถ้ามีคำถามแบบนี้ n ข้อ")
w("จะพิสูจน์ความต่างขนาดที่เห็นอยู่แล้วได้มั้ย' — ไม่มีสมมติฐานเรื่อง churn หรือขนาดผลเลย")
w("แถว n=73 คือชุดจริงทั้งชุด ไม่ resample")
w()
_p4 = [
    ("semantic x qwen3_0.6b", "semantic x congen"),
    ("semantic x qwen3_0.6b", "sentence x qwen3_0.6b"),
    ("semantic x bge_m3", "recursive x bge_m3"),
]
_hdr = [f"{a.split(' x ')[1]} vs {b.split(' x ')[1] if a.split(' x ')[0]==b.split(' x ')[0] else b}" for a, b in _p4]
w("| n คำถาม | " + " | ".join(f"{h} (Δ={(lab2vec[a]-lab2vec[b]).mean():+.4f})" for h, (a, b) in zip(_hdr, _p4)) + " |")
w("|---|" + "---|" * len(_p4))
for n in (73, 100, 150, 200, 300, 500, 1000):
    row = [str(n)]
    for a, b in _p4:
        dv = lab2vec[a] - lab2vec[b]
        r = np.random.default_rng(4242 + n)
        s = 0
        for _ in range(N_SIM):
            d = dv if n == len(dv) else dv[r.integers(0, len(dv), size=n)]
            s += p_of(d) < 0.05
        row.append(f"{s/N_SIM:.2f}")
    w("| " + " | ".join(row) + " |")
w()
w("ตัวเลขคือ power (สัดส่วนที่ได้ p<0.05) **ก่อน** Holm — ถ้าทดสอบหลายสมมติฐานต้องหารความไวลงอีก")
w("(หัวข้อ 5 ของเอกสารเดิม: m=2 เหลือราว 3/4, m=9 เหลือราวครึ่ง)")
w()

w("## 5. ผลข้างเคียงของ bootstrap แบบนี้เมื่อระบบใหม่ 'ดีขึ้นอย่างเดียว'")
w()
w("ถ้าไม่มีคำถามไหนแย่ลงเลย diffs จะไม่ติดลบสักตัว ทำให้ p ของ bootstrap แบบ percentile เท่ากับ")
w("2 x (สัดส่วนที่สุ่มได้แต่คำถามที่ diff=0) = 2 x (n_zero/n)^n ซึ่งเล็กมากแม้พลิกแค่ 1-2 คำถาม")
for nz in (72, 71, 70, 68, 63):
    w(f"- ถ้าพลิก {73-nz} คำถาม: p ทฤษฎี = 2 x ({nz}/73)^73 = {2*(nz/73)**73:.2e}")
w()
w("**นี่คือจุดที่ตัวเลข power ข้างบนต้องอ่านอย่างระวัง**: power สูงในตาราง churn=0 ไม่ได้แปลว่าเครื่องมือวัดไว")
w("แต่แปลว่าสมมติฐาน 'ดีขึ้นอย่างเดียว' นั้นแรงเกินจริง ตาราง churn>0 คือกรณีที่ควรใช้ตัดสิน")

w()
w("## 6. churn จริงเป็นเท่าไหร่ — วัดจากคู่ระบบที่มีอยู่แล้ว (ไม่ใช่สมมติ)")
w()
w("เทียบ per-query recall@10 ของระบบจริงคู่ต่าง ๆ: เวลาระบบ A ดีกว่า B โดยรวม มีกี่คำถามที่ A แย่กว่า B")
w()
w("| A | B | Δ mean | #คำถามที่ดีขึ้น | #ที่แย่ลง | churn จริง (แย่/ดี) | p (bootstrap) |")
w("|---|---|---|---|---|---|---|")
sem = {lab: v for lab, v, _ in (summary[p] for p in summary) if lab.startswith("semantic")}
pairs_to_test = [
    ("semantic x qwen3_0.6b", "semantic x qwen3"),
    ("semantic x qwen3_0.6b", "semantic x bge_m3"),
    ("semantic x qwen3", "semantic x bge_m3"),
    ("semantic x qwen3_0.6b", "semantic x congen"),
    ("semantic x qwen3_0.6b", "sentence x qwen3_0.6b"),
    ("semantic x bge_m3", "recursive x bge_m3"),
]
lab2vec = {summary[p][0]: summary[p][1] for p in summary}
churns = []
for a, b in pairs_to_test:
    if a not in lab2vec or b not in lab2vec:
        continue
    d = lab2vec[a] - lab2vec[b]
    up = int((d > 0).sum()); dn = int((d < 0).sum())
    ratio = dn / up if up else float("nan")
    churns.append(ratio)
    w(f"| {a} | {b} | {d.mean():+.4f} | {up} | {dn} | {ratio:.2f} | {p_of(d):.4f} |")
w()
w(f"**churn จริงเฉลี่ย ≈ {np.mean(churns):.2f}** — สูงกว่า 0.25 ที่ใช้จำลองไว้มาก และใกล้เคียง 0.50–1.00")
w("แปลว่าตาราง churn = 0.50 ในหัวข้อ 1 คือกรณีที่ตรงกับความจริงที่สุด ไม่ใช่กรณีมองโลกในแง่ร้าย")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT)
