"""How deep are the misses? Splitting group B of the oracle-union ceiling.

`oracle_union_ceiling.py` (task #16) established that unioning the top-10 of
every live combo across all three retrieval paths still leaves 84 (query,
resolution) pairs unfound. (This line used to read "76 after removing the
CONTROL SYSTEM(S) labelling artifact"; `audit_gold_anchor_ambiguity.py`
refuted that premise on 2026-08-09 -- the qrels reproduce exactly and every one
of those 8 pairs' gold documents does contain the queried phrase -- so the
subtraction is withdrawn and the floor is 84. This script always counted all
84; only the docstring was wrong.) That report could not say *why*, because persisted results store rank
<= 10 only: a pair sitting at rank 11 and a pair no system ranks at all look
identical from there. The distinction decides what is worth building. Rank 11-50
is a **reranking** problem (the evidence is in the candidate pool, just ordered
wrong); rank 10,000 is a **representation** problem (no reranker over any
affordable pool reaches it); absent from the index is neither, it is a corpus or
labelling problem.

Task #18 was framed as "run k=50 retrieval". That framing is discarded, and this
is the one design decision in this script worth arguing:

  **k=50 answers a worse question than exhaustive ranks, for the same cost.**
  Both `DenseRetriever` and `BM25Retriever` already score the entire corpus and
  slice at k (see their `retrieve`), so k=50 costs exactly what k=n costs -- the
  truncation buys nothing and throws away the answer for anything past 50. Worse,
  it cannot distinguish "rank 51" from "rank 40,000", which is the very split the
  ticket exists to make. So this computes the **exact rank of every gold
  resolution under every system**, uncapped, and reports the distribution.

Method. For each arm the arithmetic is replicated from the shipped retriever
rather than approximated, and every replication is gated by a self-check against
the persisted top-10 (S2/S3/S4): 36 x 106 dense, 4 x 106 BM25, 36 x 106 hybrid
must reproduce byte-for-byte before any number is published. Two traps that cost
a debugging round and are now pinned in the code:

  1. **Batching changes the answer.** Computing all 106 queries as one
     (N, 1024) @ (1024, 106) matmul and arg-sorting along axis 0 reproduced only
     98 of 106 -- BLAS accumulates a matrix-matrix product in a different order
     than a matrix-vector one, and `np.argsort`'s default quicksort is unstable,
     so exact score ties land in a different order. `_dense_scores` therefore
     does one gemv per query exactly as `DenseRetriever` does.
  2. **Hybrid's tie-break is dense-first.** `HybridRetriever` builds `fused` by
     iterating the dense ranking first and settles equal RRF scores with a stable
     `sorted`, so an equal-score pair keeps dense order. Replicated by stable-
     sorting in dense order (`kind="stable"` over the dense permutation).

The rank reported for a resolution is the rank of its **best-placed chunk**,
because the retrieval budget is counted in chunks (k=10 means 10 chunks, which
is 7 distinct resolutions in the median result file).

Read-only: consumes indices, persisted results and the gold set; writes one
report and no index.
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from rank_bm25 import BM25Okapi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from embedder_matrix_9way import _EXCLUDED_COMBO_DIRS, _embedder_label  # noqa: E402
from pythainlp.tokenize import word_tokenize  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder  # noqa: E402

INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
DENSE_RES = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
BM25_RES = REPO / "data" / "results" / "gold_bm25_73det"
HYB_RES = REPO / "data" / "results" / "gold_hybrid_73det"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
CEILING = REPO / "data" / "results" / "oracle_union_ceiling.md"
OUT = REPO / "data" / "results" / "miss_depth_profile.md"

K = 10
RRF_K = 60
BANDS = [(1, 10), (11, 50), (51, 100), (101, 1000), (1001, 10**9)]


def persisted_top10(results_dir: Path, combo: str, arm: str) -> dict[str, list[str]]:
    """query -> the top-10 chunk_ids actually written to disk."""
    out: dict[str, list[str]] = {}
    for f in results_dir.glob(f"{combo}__{arm}__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["query"]] = [
            r["chunk_id"] for r in sorted(d["results"], key=lambda r: r["rank"])
        ]
    return out


def band_of(rank: int | None) -> str:
    if rank is None:
        return "ไม่มีในดัชนี"
    for lo, hi in BANDS:
        if lo <= rank <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    raise AssertionError


def main() -> int:
    # --smoke exercises every code path on 2 combos x 8 queries (~30s) so the
    # replication logic is verified before committing to the full ~20min run.
    # It cannot publish: the aggregate checks are scoped to the full run and it
    # returns without writing.
    smoke = "--smoke" in sys.argv
    t_start = time.time()
    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: set(d["relevant_resolution_ids"]) for d in raw}
    etype = {d["query"]: d.get("entity_type", "?") for d in raw}
    if smoke:
        queries = queries[:4] + queries[-4:]
    pairs = [(q, r) for q in queries for r in sorted(qrels[q])]
    q_tokens = {q: word_tokenize(q) for q in queries}

    checks: list[tuple[str, bool, str]] = []

    # ---- combo membership: derived from which index dirs exist ------------
    with_results = sorted({"__".join(f.stem.split("__")[:4]) for f in HYB_RES.glob("*.json")})
    combos = [
        c for c in with_results
        if (INDEX_DIR / c).is_dir() and c not in _EXCLUDED_COMBO_DIRS
    ]
    checks.append((
        "S1 combo set matches the oracle-union ceiling's derivation",
        len(combos) == 36 and all((INDEX_DIR / c).is_dir() for c in combos),
        f"{len(combos)} kept of {len(with_results)} with results",
    ))

    if smoke:
        combos = sorted(combos)[:2]
    manifests = {
        c: json.loads((INDEX_DIR / c / "manifest.json").read_text(encoding="utf-8"))
        for c in combos
    }
    labels = {c: _embedder_label(manifests[c]["combo"]) for c in combos}
    chunker_of = {c: manifests[c]["combo"]["chunker"]["type"] for c in combos}

    # ---- query vectors: load each embedder once, then drop it -------------
    by_embedder: dict[str, list[str]] = collections.defaultdict(list)
    for c in combos:
        by_embedder[json.dumps(manifests[c]["combo"]["embedder"], sort_keys=True)].append(c)
    qvecs: dict[str, list] = {}
    for spec_json in sorted(by_embedder):
        emb_obj = build_embedder(StrategySpec.model_validate(json.loads(spec_json)))
        qvecs[spec_json] = [emb_obj.embed_query(q) for q in queries]
        del emb_obj
        print(f"  encoded 106 queries for {json.loads(spec_json).get('model_name', '?')}",
              file=sys.stderr)
    checks.append((
        "S1b one query-vector set per distinct embedder",
        len(qvecs) == 9,
        f"{len(qvecs)} distinct embedders over {len(combos)} combos",
    ))

    # rank[arm][combo][(query, resolution)] = 1-based chunk rank, or None if the
    # resolution has no chunk in that index at all.
    rank: dict[str, dict[str, dict[tuple[str, str], int | None]]] = {
        "dense": {}, "bm25": {}, "hybrid": {}
    }
    absent: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    dense_ok = dense_bad = bm_ok = bm_bad = hyb_ok = hyb_bad = 0
    bm25_cache: dict[str, tuple[list[str], np.ndarray]] = {}
    cache_misaligned: list[str] = []

    for ci, combo in enumerate(sorted(combos), 1):
        d = INDEX_DIR / combo
        cols = pq.read_table(d / "chunks.parquet", columns=["chunk_id", "resolution_id"]).to_pydict()
        cid, rid = cols["chunk_id"], cols["resolution_id"]
        rows_of: dict[str, list[int]] = collections.defaultdict(list)
        for i, r in enumerate(rid):
            rows_of[r].append(i)

        emb = np.load(d / "embeddings.npy")
        row_norms = np.linalg.norm(emb, axis=1)
        qv = qvecs[json.dumps(manifests[combo]["combo"]["embedder"], sort_keys=True)]

        # BM25 reads only the lexical index, which is a function of loader +
        # chunker -- the embedder never touches it. So 32 of the 36 rebuilds
        # would be redundant work (~27s each). Cached per chunker, but only
        # after checking the condition that licenses the cache: combos sharing
        # a chunker must have byte-identical chunk rows, or a cached rank
        # vector would be silently misaligned to a different index's rows.
        ck = chunker_of[combo]
        if ck in bm25_cache:
            cached_cid, bpos_all = bm25_cache[ck]
            if cached_cid != cid:
                cache_misaligned.append(combo)
                bpos_all = None
        else:
            bpos_all = None
        if bpos_all is None:
            lex = json.loads((d / "lexical.json").read_text(encoding="utf-8"))
            bm = BM25Okapi(lex)
            bpos_all = np.empty((len(queries), len(cid)), dtype=np.int64)
            for j, q in enumerate(queries):
                border = np.argsort(-bm.get_scores(q_tokens[q]))
                bpos_all[j][border] = np.arange(len(cid))
            bm25_cache[ck] = (cid, bpos_all)
            del lex, bm

        ptop_d = persisted_top10(DENSE_RES, combo, "dense")
        ptop_b = persisted_top10(BM25_RES, combo, "bm25")
        ptop_h = persisted_top10(HYB_RES, combo, "hybrid")

        rank["dense"][combo] = {}
        rank["hybrid"][combo] = {}
        if ptop_b:
            rank["bm25"][combo] = {}

        for j, q in enumerate(queries):
            # --- dense, replicating DenseRetriever.retrieve exactly ---------
            qq = np.asarray(qv[j], dtype=np.float64)
            denom = row_norms * np.linalg.norm(qq)
            dots = emb @ qq
            dscore = np.divide(
                dots, denom, out=np.zeros_like(dots, dtype=np.float64), where=denom > 0
            )
            dorder = np.argsort(-dscore)
            dpos = np.empty(len(cid), dtype=np.int64)
            dpos[dorder] = np.arange(len(cid))
            if q in ptop_d:
                if [cid[i] for i in dorder[:K]] == ptop_d[q]:
                    dense_ok += 1
                else:
                    dense_bad += 1

            # --- bm25, replicating BM25Retriever.retrieve exactly -----------
            bpos = bpos_all[j]
            border = np.argsort(bpos)
            if q in ptop_b:
                if [cid[i] for i in border[:K]] == ptop_b[q]:
                    bm_ok += 1
                else:
                    bm_bad += 1

            # --- hybrid: RRF over the two full rankings, dense-first ties ---
            fused = 0.5 / (RRF_K + dpos + 1) + 0.5 / (RRF_K + bpos + 1)
            horder = dorder[np.argsort(-fused[dorder], kind="stable")]
            hpos = np.empty(len(cid), dtype=np.int64)
            hpos[horder] = np.arange(len(cid))
            if q in ptop_h:
                if [cid[i] for i in horder[:K]] == ptop_h[q]:
                    hyb_ok += 1
                else:
                    hyb_bad += 1

            for r in qrels[q]:
                rows = rows_of.get(r)
                if not rows:
                    absent[combo].add((q, r))
                    rank["dense"][combo][(q, r)] = None
                    rank["hybrid"][combo][(q, r)] = None
                    if ptop_b:
                        rank["bm25"][combo][(q, r)] = None
                    continue
                rank["dense"][combo][(q, r)] = int(dpos[rows].min()) + 1
                rank["hybrid"][combo][(q, r)] = int(hpos[rows].min()) + 1
                if ptop_b:
                    rank["bm25"][combo][(q, r)] = int(bpos[rows].min()) + 1

        del emb
        print(f"  [{ci}/{len(combos)}] {combo}  {time.time()-t_start:.0f}s", file=sys.stderr)

    checks.append((
        "S2 dense top-10 reproduces the persisted results",
        dense_bad == 0, f"{dense_ok} reproduce, {dense_bad} differ",
    ))
    checks.append((
        "S3 BM25 top-10 reproduces the persisted results",
        bm_bad == 0, f"{bm_ok} reproduce, {bm_bad} differ",
    ))
    checks.append((
        "S3b combos sharing a chunker share chunk rows (licenses the BM25 cache)",
        not cache_misaligned,
        f"{len(bm25_cache)} lexical indices built for {len(combos)} combos; "
        f"{len(cache_misaligned)} misaligned",
    ))
    checks.append((
        "S4 hybrid top-10 reproduces the persisted results",
        hyb_bad == 0, f"{hyb_ok} reproduce, {hyb_bad} differ",
    ))

    if smoke:
        for name, ok, detail in checks[-4:]:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        print(f"\nsmoke run ({len(combos)} combos x {len(queries)} queries, "
              f"{time.time()-t_start:.0f}s) -- nothing written")
        return 0 if all(ok for _, ok, _ in checks[-4:]) else 1

    # ---- best rank per pair ----------------------------------------------
    def best_over(arms: list[str]) -> dict[tuple[str, str], int | None]:
        out: dict[tuple[str, str], int | None] = {}
        for p in pairs:
            vals = [
                rank[a][c][p]
                for a in arms for c in rank[a]
                if rank[a][c].get(p) is not None
            ]
            out[p] = min(vals) if vals else None
        return out

    best_hyb = best_over(["hybrid"])
    best_all = best_over(["dense", "bm25", "hybrid"])

    found10_hyb = {p for p, v in best_hyb.items() if v is not None and v <= K}
    found10_all = {p for p, v in best_all.items() if v is not None and v <= K}
    unfound_all = [p for p in pairs if p not in found10_all]

    # S5: this has to agree with the ceiling report, which counted the same
    # thing from persisted top-10s by a completely different code path.
    ceiling_txt = CEILING.read_text(encoding="utf-8") if CEILING.exists() else ""
    checks.append((
        "S5 pairs unfound at k=10 by any arm agrees with oracle_union_ceiling.md",
        len(unfound_all) == 84,
        f"{len(unfound_all)} unfound here; the ceiling report says 84"
        f"{'' if '84' in ceiling_txt else ' (report not found or changed)'}",
    ))
    checks.append((
        "S6 hybrid-only unfound count agrees with the ceiling report",
        len(pairs) - len(found10_hyb) == 164,
        f"{len(pairs) - len(found10_hyb)} unfound under hybrid alone; the report says 164",
    ))

    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# ความลึกของคู่ที่หาไม่เจอ — แยกกลุ่ม B ของเพดาน oracle-union")
    w()
    w(f"Generated by `tools/eval/miss_depth_profile.py` · {len(combos)} combo · ")
    w(f"{len(queries)} คำถาม · {len(pairs)} คู่ · arm = dense / bm25 / hybrid")
    w()
    w("**คำถามที่ตอบ**: คู่ที่ระบบหาไม่เจอที่ k=10 นั้น อยู่ที่อันดับเท่าไรกันแน่ ")
    w("ถ้าอยู่อันดับ 11-50 แปลว่าเป็นปัญหา *การจัดอันดับ* (reranker ช่วยได้) ")
    w("ถ้าอยู่อันดับหมื่น แปลว่าเป็นปัญหา *การแทนความหมาย* (reranker ไม่ช่วย)")
    w()
    w("**ไม่ได้รันที่ k=50** — retriever ทั้งสองตัวให้คะแนนทั้งคลังอยู่แล้วแล้วค่อยตัดที่ k ")
    w("การตัดที่ 50 จึงไม่ได้ถูกลงเลย แถมยังแยก 'อันดับ 51' ออกจาก 'อันดับ 40,000' ไม่ได้ ")
    w("ซึ่งเป็นการแยกที่ทั้งหมดนี้มีไว้เพื่อทำ สคริปต์นี้จึงคำนวณ **อันดับจริงแบบไม่ตัด**")
    w()
    w("อันดับของมติหนึ่ง = อันดับของ chunk ที่ดีที่สุดของมตินั้น เพราะงบวัดเป็น chunk ")
    w("(k=10 คือ 10 chunk ซึ่งเป็นมติราว 7 รายการ)")
    w()

    # ---- 1. where the misses live ----------------------------------------
    w("## 1. คู่ที่หาไม่เจอที่ k=10 อยู่ลึกแค่ไหน")
    w()
    for scope, bestmap, label in (
        ("hybrid อย่างเดียว", best_hyb, "hybrid"),
        ("รวมทั้ง 3 arm", best_all, "all"),
    ):
        miss = [p for p in pairs if (bestmap[p] is None or bestmap[p] > K)]
        w(f"**{scope}** — หาไม่เจอ {len(miss)} คู่ จาก {len(pairs)}")
        w()
        w("| อันดับที่ดีที่สุดที่ระบบใดระบบหนึ่งทำได้ | จำนวนคู่ | % ของที่หาไม่เจอ |")
        w("|---|---|---|")
        cnt = collections.Counter(band_of(bestmap[p]) for p in miss)
        for lo, hi in BANDS[1:]:
            key = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            w(f"| {key} | {cnt.get(key, 0)} | {100*cnt.get(key, 0)/max(1, len(miss)):.1f}% |")
        w(f"| ไม่มีในดัชนี | {cnt.get('ไม่มีในดัชนี', 0)} | "
          f"{100*cnt.get('ไม่มีในดัชนี', 0)/max(1, len(miss)):.1f}% |")
        w()

    # ---- 2. the reranker question ----------------------------------------
    w("## 2. reranker ต้องดึง candidate ลึกแค่ไหนถึงจะคุ้ม")
    w()
    w("**สองคอลัมน์นี้ไม่ใช่อย่างเดียวกัน อย่าอ่านสลับกัน** — `ใน pool` คือสัดส่วน gold ที่ ")
    w("*อยู่ใน* candidate pool ขนาด P (คือวัตถุดิบที่ reranker มีให้ทำงาน) ส่วน `ส่งได้จริง` ")
    w("คือ recall หลัง rerank เมื่อ reranker ยัง **ส่งออกได้แค่ 10 ใบ** ตามงบเดิม ")
    w("คอลัมน์แรกทะลุเพดาน qrels ได้เพราะมันไม่ได้ถูกจำกัดที่ 10 ใบ คอลัมน์หลังทะลุไม่ได้เลย")
    w()
    w("| P | ระบบเดี่ยว ใน pool | ระบบเดี่ยว **ส่งได้จริง** | ทุก arm ใน pool | ทุก arm **ส่งได้จริง** |")
    w("|---|---|---|---|---|")
    per_combo_at10 = {
        c: np.mean([
            len([r for r in qrels[q] if (rank["hybrid"][c][(q, r)] or 10**9) <= K]) / len(qrels[q])
            for q in queries
        ]) for c in combos
    }
    bestc = max(per_combo_at10, key=lambda c: per_combo_at10[c])
    delivered: dict[tuple[str, int], float] = {}

    def rec(getter, P: int) -> tuple[float, float]:
        """(recall of the pool, recall a reranker could deliver in a top-10)."""
        pool, deliv = [], []
        for q in queries:
            h = len([r for r in qrels[q] if (getter((q, r)) or 10**9) <= P])
            pool.append(h / len(qrels[q]))
            deliv.append(min(h, K) / len(qrels[q]))
        return float(np.mean(pool)), float(np.mean(deliv))

    for P in (10, 20, 50, 100, 200, 500, 1000):
        sp, sd = rec(lambda p: rank["hybrid"][bestc][p], P)
        ap, ad = rec(lambda p: best_all[p], P)
        delivered[("single", P)] = sd
        delivered[("all", P)] = ad
        w(f"| {P} | {sp:.4f} | **{sd:.4f}** | {ap:.4f} | **{ad:.4f}** |")
    w()
    w(f"ระบบเดี่ยวที่ดีที่สุด = `{chunker_of[bestc]} x {labels[bestc]}` (hybrid) · ")
    w("macro ทั้งตาราง")
    w()
    struct_ceiling = float(np.mean([min(1.0, K / len(qrels[q])) for q in queries]))
    w(f"เพดาน qrels ที่งบ 10 ใบคือ **{struct_ceiling:.4f}** ทุกค่าในคอลัมน์ `ส่งได้จริง` ")
    w("ถูกบีบด้วยเส้นนี้ ไม่ว่า pool จะลึกแค่ไหน — นี่คือเหตุผลที่ตัวเลข 'rerank สมบูรณ์แบบ' ")
    w("ไม่ได้แปลว่า recall จะขึ้นไปถึง 0.98 ตามคอลัมน์ซ้าย")
    w()
    checks.append((
        "S7 the delivered column never exceeds the qrels ceiling at k=10",
        all(v <= struct_ceiling + 5e-5 for v in delivered.values()),
        f"ceiling {struct_ceiling:.4f}; highest delivered {max(delivered.values()):.4f}",
    ))

    # ---- 3. per entity_type ----------------------------------------------
    w("## 3. แยกตามชนิดคำถาม (รวมทุก arm)")
    w()
    w("| entity_type | คู่ | หาไม่เจอที่ k=10 | 11-50 | 51-1000 | 1001+ | ไม่มีในดัชนี |")
    w("|---|---|---|---|---|---|---|")
    for et in sorted({etype[q] for q in queries}):
        ps = [p for p in pairs if etype[p[0]] == et]
        miss = [p for p in ps if (best_all[p] is None or best_all[p] > K)]
        c = collections.Counter(band_of(best_all[p]) for p in miss)
        mid = c.get("51-100", 0) + c.get("101-1000", 0)
        w(f"| {et} | {len(ps)} | {len(miss)} | {c.get('11-50', 0)} | {mid} | "
          f"{c.get('1001+', 0)} | {c.get('ไม่มีในดัชนี', 0)} |")
    w()

    # ---- 4. the pairs no system ranks anywhere near ----------------------
    deep = sorted(
        [p for p in unfound_all if best_all[p] is not None and best_all[p] > 1000],
        key=lambda p: -best_all[p],
    )
    gone = [p for p in unfound_all if best_all[p] is None]
    w("## 4. คู่ที่ลึกที่สุด — reranker ไม่มีทางช่วย")
    w()
    w(f"อยู่ลึกกว่าอันดับ 1000 ในทุกระบบ: **{len(deep)}** คู่ · ")
    w(f"ไม่มี chunk ในดัชนีเลย: **{len(gone)}** คู่")
    w()
    if deep:
        w("| คำถาม | มติ | อันดับที่ดีที่สุด |")
        w("|---|---|---|")
        for q, r in deep[:15]:
            w(f"| {q[:46]} | {r[:52]} | {best_all[(q, r)]:,} |")
        w()

    # ---- 5. which arm gets closest --------------------------------------
    w("## 5. arm ไหนเข้าใกล้ที่สุดสำหรับคู่ที่หาไม่เจอ")
    w()
    w("| arm | median อันดับที่ดีที่สุด | คู่ที่ arm นี้ทำได้ดีที่สุด (นับ tie) |")
    w("|---|---|---|")
    for arm in ("dense", "bm25", "hybrid"):
        vals = []
        wins = 0
        for p in unfound_all:
            v = [rank[arm][c][p] for c in rank[arm] if rank[arm][c].get(p) is not None]
            if not v:
                continue
            vals.append(min(v))
            if best_all[p] is not None and min(v) == best_all[p]:
                wins += 1
        w(f"| {arm} | {int(np.median(vals)):,} | {wins} |" if vals else f"| {arm} | — | 0 |")
    w()

    w("## self-check")
    w()
    for name, ok, detail in checks:
        w(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    w()

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
    if not all(ok for _, ok, _ in checks):
        print("\nself-check failed; refusing to publish numbers", file=sys.stderr)
        return 1

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}  ({time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
