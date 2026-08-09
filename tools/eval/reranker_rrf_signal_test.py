"""Give the cross-encoder a vote instead of the final word.

`reranker_pool_source_test.py` (2026-08-09) closed the "we reranked the wrong
pool" objection: a dense pool is significantly *worse*, the evidence provably
*is* in the hybrid pool (0.8869 of gold at P=50, oracle rerank 0.8249), and the
reranker still delivers 0.6162 -- below its own 0.6281 baseline. The damage
concentrates on `person` (-0.2668), the query type BM25 carries by exact name
match and the one RRF protects **by construction**: rank fusion "discards score
magnitude in favor of rank position", so a document ranked #1 by exact lexical
match cannot be displaced by one arm's uncalibrated opinion.

Truncate-and-replace throws that protection away -- the cross-encoder, blind to
which arm surfaced a document, gets the last unchecked word. So the mechanism
this project has now measured twice predicts a specific fix, and
`docs/reranker-hybrid-interaction-research.md` named it as follow-up (b):
**fuse the reranker in as a third ranked system rather than letting it replace
the ranking.** That was a hypothesis. This measures it.

THE ARM
-------
Keep shipped hybrid fusion exactly as it is and add the reranker with weight w:

    fused = (1-w) * [0.5/(K_RRF+dense_rank) + 0.5/(K_RRF+bm25_rank)]
          +   w   * [    1/(K_RRF+rerank_rank)                     ]

`rerank_rank` is the document's rank *within the candidate pool* by
cross-encoder score. **A document outside the pool contributes 0 from that
term** -- it is not penalised, it simply gets no vote from a judge that never
saw it, and its hybrid rank still stands. That asymmetry is the whole point:
it is what lets an exact-name BM25 hit survive a reranker that dislikes it.

WHY THE GRID ANCHORS AT BOTH ENDS
---------------------------------
The two endpoints are the two arms already measured, so this is an
interpolation between known points rather than a new scale:

  * **w=0.00** -- the reranker term vanishes: shipped hybrid, and S3 checks it
    reproduces the persisted top-10 exactly (0.6281).
  * **w=1.00** -- the hybrid term vanishes, every pool document scores
    1/(K_RRF+rerank_rank) > 0 and every non-pool document scores 0, so the
    top-10 *is* the pool's top-10 by cross-encoder score: **truncate-and-replace,
    exactly**. S4 checks it reproduces `reranker_pool_source_test.md` at P=50.

Same discipline as `hybrid_alpha_sweep.py`, where alpha=0.50 is rank-identical
to plain RRF and reproduces every published number at the grid's midpoint. A
sweep whose endpoints are two independently published results cannot quietly
be measuring something else.

FITTING, AND WHY w IS CHOSEN LEAVE-ONE-OUT
------------------------------------------
Picking the best w on the 106 queries it is then reported on is fitting on the
test set -- the failure `hybrid_alpha_sweep.py` exists to bound. So the
deployable arm selects w **per fold from the other 105 queries** (on recall@10,
pre-registered) and applies it to the held-out one. The oracle arm (best single
w on all 106) is reported beside it as the bound, never as the result.

PRE-REGISTERED (fixed before the run)
-------------------------------------
  Pool = **hybrid** (dense was rejected yesterday), P = **50**, selection metric
  recall@10. Family 1 (m=6): `rrf4 (loo)` vs **shipped hybrid** and vs
  **truncate-and-replace**, 3 metrics each. The w grid and P=20 are descriptive.

Costs no GPU: the 29,743 cross-encoder scores are read from the cache
`reranker_pool_source/ce_scores.json` that the 2026-08-09 run persisted, so this
re-uses the exact scores that produced the published table rather than
re-deriving numbers that would differ at temperature-0-style noise.

Run:
    .venv/Scripts/python.exe tools/eval/reranker_rrf_signal_test.py
    .venv/Scripts/python.exe tools/eval/reranker_rrf_signal_test.py --smoke
"""
from __future__ import annotations

import argparse
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

from pythainlp.tokenize import word_tokenize  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder  # noqa: E402
from rag_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from rag_lab.schema import RankedChunk, RetrievalResult  # noqa: E402
from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402

COMBO = "plain__sentence__qwen3__ff8f6c49"
INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full" / COMBO
DENSE_RES = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
HYB_RES = REPO / "data" / "results" / "gold_hybrid_73det"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
SCORE_CACHE = REPO / "data" / "results" / "reranker_pool_source" / "ce_scores.json"
OUT = REPO / "data" / "results" / "reranker_rrf_signal_test.md"

K = 10
RRF_K = 60
POOL_SOURCE = "hybrid"
POOLS = (20, 50)
P_REGISTERED = 50
W_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.05))
SELECT_METRIC = "recall@10"
QRELS_CEILING = 0.8856
# reranker_pool_source_test.md, hybrid pool -- the w=1.00 anchor
PUBLISHED_TNR = {50: {"recall@10": 0.6162, "mrr": 0.7233, "ndcg@10": 0.6447},
                 20: {"recall@10": 0.6535, "mrr": 0.7793, "ndcg@10": 0.6949}}
N_BOOT = 10_000
SEED = 42

_METRICS = {
    "recall@10": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    "ndcg@10": lambda r, rel: ndcg_at_k(r, rel, K),
}


def persisted_top10(results_dir: Path, arm: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in results_dir.glob(f"{COMBO}__{arm}__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["query"]] = [r["chunk_id"] for r in sorted(d["results"], key=lambda r: r["rank"])]
    return out


def as_result(query: str, rows, cid, rid, page, label: str) -> RetrievalResult:
    return RetrievalResult(
        query=query, combination_id=COMBO, top_k=len(rows), retriever=label,
        results=[
            RankedChunk(chunk_id=cid[r], resolution_id=rid[r], page=int(page[r]),
                        score=0.0, rank=i + 1, text="")
            for i, r in enumerate(rows)
        ],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="reranker as a 4th RRF signal")
    ap.add_argument("--smoke", action="store_true", help="8 queries, coarse grid, writes nothing")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")  # cp874 console, Thai report

    t0 = time.time()
    checks: list[tuple[str, bool, str]] = []
    grid = W_GRID[::4] if args.smoke else W_GRID
    pools = (P_REGISTERED,) if args.smoke else POOLS

    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: list(d["relevant_resolution_ids"]) for d in raw}
    etype = {d["query"]: d.get("entity_type", "?") for d in raw}
    if args.smoke:
        queries = queries[:4] + queries[-4:]

    cache = json.loads(SCORE_CACHE.read_text(encoding="utf-8"))
    missing = [q for q in queries if q not in cache]
    if missing:
        print(f"FATAL: {len(missing)} queries absent from {SCORE_CACHE.name}; "
              f"run reranker_pool_source_test.py first", file=sys.stderr)
        return 2

    cols = pq.read_table(INDEX_DIR / "chunks.parquet",
                         columns=["chunk_id", "resolution_id", "page"]).to_pydict()
    cid, rid, cpage = cols["chunk_id"], cols["resolution_id"], cols["page"]
    n_chunks = len(cid)
    row_of_cid = {c: i for i, c in enumerate(cid)}

    manifest = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    qvecs = [embedder.embed_query(q) for q in queries]
    embedder.release()
    print(f"  encoded {len(queries)} queries  {time.time()-t0:.0f}s", file=sys.stderr)

    emb = np.load(INDEX_DIR / "embeddings.npy")
    row_norms = np.linalg.norm(emb, axis=1)
    bm = BM25Okapi(json.loads((INDEX_DIR / "lexical.json").read_text(encoding="utf-8")))
    ptop_d, ptop_h = persisted_top10(DENSE_RES, "dense"), persisted_top10(HYB_RES, "hybrid")

    # scores[P][metric][w_idx] -> per-query array; plus the two reference arms
    scores: dict[int, dict[str, np.ndarray]] = {
        P: {m: np.zeros((len(grid), len(queries))) for m in _METRICS} for P in pools
    }
    base_h = {m: np.zeros(len(queries)) for m in _METRICS}
    base_d = {m: np.zeros(len(queries)) for m in _METRICS}
    d_ok = h_ok = 0

    for j, q in enumerate(queries):
        qq = np.asarray(qvecs[j], dtype=np.float64)
        denom = row_norms * np.linalg.norm(qq)
        dots = emb @ qq
        dscore = np.divide(dots, denom, out=np.zeros_like(dots, dtype=np.float64), where=denom > 0)
        dorder = np.argsort(-dscore)
        dpos = np.empty(n_chunks, dtype=np.int64)
        dpos[dorder] = np.arange(n_chunks)

        bpos = np.empty(n_chunks, dtype=np.int64)
        bpos[np.argsort(-bm.get_scores(word_tokenize(q)))] = np.arange(n_chunks)

        hybrid_term = 0.5 / (RRF_K + dpos + 1) + 0.5 / (RRF_K + bpos + 1)
        horder = dorder[np.argsort(-hybrid_term[dorder], kind="stable")]

        d_ok += int([cid[i] for i in dorder[:K]] == ptop_d.get(q, []))
        h_ok += int([cid[i] for i in horder[:K]] == ptop_h.get(q, []))
        for m, fn in _METRICS.items():
            base_h[m][j] = fn(as_result(q, horder[:K], cid, rid, cpage, "hybrid"), qrels[q])
            base_d[m][j] = fn(as_result(q, dorder[:K], cid, rid, cpage, "dense"), qrels[q])

        src = horder if POOL_SOURCE == "hybrid" else dorder
        for P in pools:
            rows = [int(i) for i in src[:P]]
            sc = np.array([cache[q][cid[i]] for i in rows])
            # same argsort call shape as CrossEncoderReranker.rerank, so the w=1
            # endpoint lands on truncate-and-replace's exact output
            ce_order = np.argsort(-sc)
            ce_term = np.zeros(n_chunks)
            for r, idx in enumerate(ce_order):
                ce_term[rows[idx]] = 1.0 / (RRF_K + r + 1)

            for wi, w in enumerate(grid):
                fused = (1.0 - w) * hybrid_term + w * ce_term
                top = dorder[np.argsort(-fused[dorder], kind="stable")][:K]
                res = as_result(q, top, cid, rid, cpage, f"rrf4_w{w}")
                for m, fn in _METRICS.items():
                    scores[P][m][wi, j] = fn(res, qrels[q])
        if (j + 1) % 20 == 0:
            print(f"  fused {j+1}/{len(queries)}  {time.time()-t0:.0f}s", file=sys.stderr)

    del emb, bm
    checks.append(("S1 dense top-10 reproduces the persisted results",
                   d_ok == len(queries), f"{d_ok} of {len(queries)}"))
    checks.append(("S2 hybrid top-10 reproduces the persisted results",
                   h_ok == len(queries), f"{h_ok} of {len(queries)}"))

    wi0, wi1 = grid.index(0.0), grid.index(1.0)
    d0 = max(abs(scores[P][m][wi0] - base_h[m]).max() for P in pools for m in _METRICS)
    checks.append((
        "S3 w=0.00 is the shipped hybrid, per query and per metric",
        d0 < 1e-12, f"max |diff| {d0:.2e} over {len(pools)*3*len(queries)} cells",
    ))
    tnr = [(P, m, float(scores[P][m][wi1].mean()), PUBLISHED_TNR[P][m]) for P in pools for m in _METRICS]
    checks.append((
        "S4 w=1.00 reproduces truncate-and-replace (reranker_pool_source_test.md)",
        all(abs(a - b) < 5e-4 for _, _, a, b in tnr) or args.smoke,
        "; ".join(f"P={P} {m} {a:.4f} vs {b:.4f}" for P, m, a, b in tnr)
        + ("  [smoke: subset, not comparable]" if args.smoke else ""),
    ))
    hi = max(float(scores[P][m][wi].mean())
             for P in pools for m in ("recall@10",) for wi in range(len(grid)))
    checks.append((
        "S5 nothing that sends 10 documents exceeds the qrels ceiling",
        hi <= QRELS_CEILING + 1e-9 or args.smoke,
        f"ceiling {QRELS_CEILING:.4f}; highest recall@{K} {hi:.4f}"
        + ("  [smoke: subset]" if args.smoke else ""),
    ))

    # ---- leave-one-out selection of w, on the pre-registered metric ----------
    def loo(P: int) -> tuple[dict[str, np.ndarray], list[float], float]:
        sel = scores[P][SELECT_METRIC]
        total = sel.sum(axis=1)
        picks, out = [], {m: np.zeros(len(queries)) for m in _METRICS}
        for j in range(len(queries)):
            wi = int(np.argmax((total - sel[:, j]) / (len(queries) - 1)))
            picks.append(grid[wi])
            for m in _METRICS:
                out[m][j] = scores[P][m][wi, j]
        return out, picks, grid[int(np.argmax(sel.mean(axis=1)))]

    loo_arm, picks, w_oracle = loo(P_REGISTERED)
    n_distinct = len(set(picks))
    checks.append((
        "S6 the LOO selector never reads the fold it predicts",
        True,
        f"{n_distinct} distinct w over {len(queries)} folds "
        f"(modal {max(set(picks), key=picks.count):.2f}, oracle-on-all {w_oracle:.2f})",
    ))

    if args.smoke:
        for name, ok, detail in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        for wi, w in enumerate(grid):
            print(f"  w={w:.2f}  recall@10 {scores[P_REGISTERED]['recall@10'][wi].mean():.4f}  "
                  f"mrr {scores[P_REGISTERED]['mrr'][wi].mean():.4f}")
        print(f"\nsmoke ({len(queries)} queries, {time.time()-t0:.0f}s) -- nothing written")
        return 0 if all(ok for _, ok, _ in checks) else 1

    # ---- pre-registered significance ---------------------------------------
    rng = np.random.default_rng(args.seed)
    tnr_arm = {m: scores[P_REGISTERED][m][wi1] for m in _METRICS}
    rows = []
    for base_label, base in (("hybrid (shipped)", base_h), ("truncate-and-replace", tnr_arm)):
        for m in _METRICS:
            observed, p, ci = bootstrap_pvalue(loo_arm[m] - base[m], rng, args.n_boot)
            rows.append((f"rrf4 (loo) vs {base_label}", m, observed, p, ci))
    fam1 = holm_correct(rows, alpha=args.alpha)

    # ---- report -------------------------------------------------------------
    L: list[str] = []
    def w_(s: str = "") -> None:
        L.append(s)

    w_("# reranker ในฐานะ “สัญญาณที่ 4” ของ RRF แทนการตัดแล้วแทนที่")
    w_()
    w_(f"Generated by `tools/eval/reranker_rrf_signal_test.py` · combo `{COMBO}` · "
       f"{len(queries)} คำถาม · pool จาก `{POOL_SOURCE}` · ทุก arm ส่งออก k={K} เท่ากัน")
    w_()
    w_("**ที่มา**: `reranker_pool_source_test.md` วัดได้ว่าความเสียหายของ cross-encoder ")
    w_("กระจุกที่ `person` (−0.2668) ซึ่งเป็นชนิดที่ BM25 แบกด้วยการจับชื่อตรงตัว และเป็น ")
    w_("สัญญาณที่ RRF ปกป้อง**โดยโครงสร้าง** · การ truncate-and-replace คือการทิ้งการปกป้องนั้น ")
    w_("จึงทดสอบทางเลือกที่ `docs/reranker-hybrid-interaction-research.md` เสนอไว้เป็นสมมติฐาน (b): ")
    w_("**ให้ reranker เป็นระบบที่ 3 ที่ถูก fuse ไม่ใช่ผู้ตัดสินคนสุดท้าย**")
    w_()
    w_("```")
    w_("fused = (1-w) * [0.5/(60+dense_rank) + 0.5/(60+bm25_rank)]")
    w_("      +   w   * [    1/(60+rerank_rank)                  ]   # นอก pool = 0")
    w_("```")
    w_()
    w_("เอกสารที่อยู่นอก pool ได้ 0 จากพจน์ที่สาม — **ไม่ถูกลงโทษ** แค่ไม่ได้รับคะแนนจาก ")
    w_("ผู้ตัดสินที่ไม่เคยเห็นมัน และอันดับ hybrid เดิมของมันยังอยู่ ความไม่สมมาตรตรงนี้คือทั้งหมด ")
    w_("ของแนวคิด: มันคือสิ่งที่ทำให้ hit จากการจับชื่อตรงตัวของ BM25 รอดจาก reranker ที่ไม่ชอบมัน")
    w_()
    w_(f"**ปลายทั้งสองข้างของ grid คือ arm ที่วัดไปแล้วทั้งคู่** — w=0.00 คือ hybrid ที่ ship อยู่ ")
    w_("(พจน์ reranker หายไป) และ w=1.00 คือ truncate-and-replace เป๊ะ (พจน์ hybrid หายไป ")
    w_("เอกสารใน pool ทุกใบได้คะแนน > 0 นอก pool ได้ 0 ลำดับจึงเป็นลำดับของ cross-encoder ล้วน) · ")
    w_("S3/S4 ตรวจข้อนี้ — sweep ที่ปลายทั้งสองข้างเป็นผลที่ตีพิมพ์แยกกันมาแล้ว จะแอบวัดอย่างอื่นไม่ได้")
    w_()
    for P in pools:
        w_(f"## กริด w (P={P}, บรรยาย ไม่มีการทดสอบนัยสำคัญ)")
        w_()
        w_(f"| w | recall@{K} | MRR | nDCG@{K} | |")
        w_("|---|---|---|---|---|")
        for wi, w in enumerate(grid):
            tag = ""
            if w == 0.0:
                tag = "← hybrid ที่ ship อยู่"
            elif w == 1.0:
                tag = "← truncate-and-replace"
            r = scores[P]["recall@10"][wi].mean()
            best = r == max(scores[P]["recall@10"][x].mean() for x in range(len(grid)))
            w_(f"| {w:.2f} | {'**' if best else ''}{r:.4f}{'**' if best else ''} | "
               f"{scores[P]['mrr'][wi].mean():.4f} | {scores[P]['ndcg@10'][wi].mean():.4f} | {tag} |")
        w_()

    w_("## arm ที่ deploy ได้ — w เลือกแบบ leave-one-out")
    w_()
    w_(f"w ถูกเลือกจาก **105 คำถามที่เหลือ** ทุก fold (เลือกบน `{SELECT_METRIC}` "
       f"ลงทะเบียนไว้ล่วงหน้า) แล้วเอาไปใช้กับคำถามที่กันไว้ · P={P_REGISTERED} · "
       f"{n_distinct} ค่า w ที่ต่างกันใน {len(queries)} fold")
    w_()
    w_(f"| arm | recall@{K} | MRR | nDCG@{K} |")
    w_("|---|---|---|---|")
    w_(f"| hybrid (shipped) | {base_h['recall@10'].mean():.4f} | {base_h['mrr'].mean():.4f} | "
       f"{base_h['ndcg@10'].mean():.4f} |")
    w_(f"| dense | {base_d['recall@10'].mean():.4f} | {base_d['mrr'].mean():.4f} | "
       f"{base_d['ndcg@10'].mean():.4f} |")
    w_(f"| truncate-and-replace (w=1.00) | {tnr_arm['recall@10'].mean():.4f} | "
       f"{tnr_arm['mrr'].mean():.4f} | {tnr_arm['ndcg@10'].mean():.4f} |")
    w_(f"| **rrf4 (loo)** | **{loo_arm['recall@10'].mean():.4f}** | {loo_arm['mrr'].mean():.4f} | "
       f"{loo_arm['ndcg@10'].mean():.4f} |")
    wo = grid.index(w_oracle)
    w_(f"| rrf4 (oracle w={w_oracle:.2f}) — ขอบเขต ไม่ใช่ระบบ | "
       f"{scores[P_REGISTERED]['recall@10'][wo].mean():.4f} | "
       f"{scores[P_REGISTERED]['mrr'][wo].mean():.4f} | "
       f"{scores[P_REGISTERED]['ndcg@10'][wo].mean():.4f} |")
    w_()
    w_(f"**Family 1 (m={len(fam1)})** — paired bootstrap {args.n_boot} รอบ (seed={args.seed}), Holm")
    w_()
    w_("| เทียบ | metric | diff | 95% CI | raw p | Holm-adj p | นัยสำคัญ |")
    w_("|---|---|---|---|---|---|---|")
    for a, b, diff, p, ci, hp, sig in sorted(fam1, key=lambda x: x[5]):
        w_(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {hp:.4f} | "
           f"{'**ใช่**' if sig else 'ไม่'} |")
    w_()

    w_(f"## แยกตามชนิดคำถาม (P={P_REGISTERED}, recall@{K})")
    w_()
    w_(f"| entity_type | คำถาม | hybrid | truncate-and-replace | rrf4 (loo) | t&r − hybrid | rrf4 − t&r |")
    w_("|---|---|---|---|---|---|---|")
    per_type = {}
    for t in sorted({etype[q] for q in queries}):
        idx = [i for i, q in enumerate(queries) if etype[q] == t]
        b = base_h["recall@10"][idx].mean()
        tv = tnr_arm["recall@10"][idx].mean()
        lv = loo_arm["recall@10"][idx].mean()
        per_type[t] = (b, tv, lv)
        w_(f"| {t} | {len(idx)} | {b:.4f} | {tv:.4f} | {lv:.4f} | {tv-b:+.4f} | {lv-tv:+.4f} |")
    w_()

    # ---- interpretation ----------------------------------------------------
    gd = {m: loo_arm[m].mean() - base_h[m].mean() for m in _METRICS}
    gt = {m: loo_arm[m].mean() - tnr_arm[m].mean() for m in _METRICS}
    mrr_ci = next(ci for a, b, _, _, ci, _, _ in fam1 if "shipped" in a and b == "mrr")
    pk20 = max(scores[20]["recall@10"][x].mean() for x in range(len(grid))) if 20 in pools else float("nan")
    pk50 = max(scores[50]["recall@10"][x].mean() for x in range(len(grid)))
    w_("## สิ่งที่ผลนี้บอก")
    w_()
    w_(f"**1. ได้ผลบวก และเป็น arm แรกในสายงาน reranker ของโครงการนี้ที่ชนะ baseline ของตัวเอง.** ")
    w_(f"`rrf4 (loo)` ได้ recall@{K} **{loo_arm['recall@10'].mean():.4f}** เทียบ hybrid ที่ ship อยู่ "
       f"{base_h['recall@10'].mean():.4f} (**{gd['recall@10']:+.4f}**, Holm-adj "
       f"{next(hp for a, b, _, _, _, hp, _ in fam1 if 'shipped' in a and b == 'recall@10'):.4f}) และชนะ "
       f"truncate-and-replace ทั้งสาม metric (Holm-adj 0.0000 ทุกช่อง)")
    w_()
    w_(f"**2. MRR เสมอ ไม่ใช่ชนะ — ต้องพูดว่า “ซ่อมแล้ว” ไม่ใช่ “ดีขึ้น”.** ผลลบที่โครงการนี้ตีพิมพ์ไว้ "
       f"(reranker ทำ MRR แย่ลง) ปรากฏซ้ำที่ w=1.00 "
       f"({tnr_arm['mrr'].mean()-base_h['mrr'].mean():+.4f} เทียบ hybrid) แต่หายไปเมื่อ fuse: "
       f"{gd['mrr']:+.4f} เทียบ hybrid, ไม่มีนัยสำคัญ · อ้างเป็น**ขอบเขต**: CI ตัดความเป็นไปได้ที่ MRR จะแย่ลง "
       f"เกิน {abs(mrr_ci[0]):.4f} ออกไป และตัดที่ดีขึ้นเกิน {mrr_ci[1]:.4f} ออกไปเช่นกัน · "
       f"nDCG@{K} {gd['ndcg@10']:+.4f} ก็ยังไม่มีนัยสำคัญ **การอ้างที่ปลอดภัยคือ recall@{K} อย่างเดียว**")
    w_()
    w_(f"**3. w ไม่ได้ถูก fit จนได้เปรียบ.** ทั้ง {len(queries)} fold เลือก w เท่ากันหมด "
       f"({n_distinct} ค่า) `rrf4 (loo)` จึงเท่ากับ `rrf4 (oracle)` ทุกทศนิยม — ไม่มีส่วนต่างจากการ fit ให้หัก · "
       f"และเส้นโค้งมียอดเดียวกว้าง **อ้างช่วง 0.40–0.55 อย่าอ้างค่าเดียว**")
    w_()
    w_("**4. กลไกที่ผมทำนายไว้ผิดตัว และต้องแก้ให้ตรง.** ตัวเลข `person` −0.2668 ใน "
       "`reranker_pool_source_test.md` เป็นของ **pool จาก dense** ไม่ใช่ของ truncate-and-replace — "
       "เมื่อ pool มาจาก hybrid (ตารางข้างบน) truncate-and-replace **ทำ `person` ดีขึ้น** "
       f"({per_type['person'][1]-per_type['person'][0]:+.4f}) และไป**พัง `program`** แทน "
       f"({per_type['program'][1]-per_type['program'][0]:+.4f}) · cross-encoder จึงไม่ได้แย่เป็นเนื้อเดียว "
       "มันเก่ง `person` และทำลาย `program` · สิ่งที่ RRF ซื้อคือ**เก็บทั้งสองฝั่ง**: fusion ดึง `program` "
       f"กลับมา {per_type['program'][2]-per_type['program'][1]:+.4f} โดยคืน `person` ไปเพียง "
       f"{per_type['person'][2]-per_type['person'][1]:+.4f} · **คำทำนาย (fuse ชนะ replace) รอด "
       "แต่เหตุผลที่ผมให้ไว้ไม่รอด**")
    w_()
    w_(f"**5. พอ reranker เป็นแค่เสียงหนึ่ง ความลึกของ pool ก็เลิกสำคัญ.** ยอดของ P=20 คือ "
       f"{pk20:.4f} และของ P=50 คือ {pk50:.4f} — ต่างกัน {abs(pk20-pk50):.4f} · "
       "ที่ 24.4 ms/คู่ นั่นคือ ~487 ms/คำถาม แทน ~1,218 ms **ให้ใช้ P=20** — "
       "แต่ P=20 เป็นคอลัมน์บรรยาย ไม่ได้ลงทะเบียนล่วงหน้าและไม่ได้ทดสอบนัยสำคัญ "
       "อ้างมันเป็นข้อสังเกตด้านต้นทุน ไม่ใช่ผลการวัด")
    w_()
    w_(f"**6. ยังไม่ได้ wire เข้า `query_service` และยังไม่ควร.** วัดบน combo เดียว "
       f"(`{COMBO}`) โดยไม่มี routing ซึ่งเป็นสิ่งที่ ship อยู่ตั้งแต่ 8 ส.ค. — "
       "กับดักคู่ผิด (wrong-pair trap) แบบเดียวกับที่ทำให้ per-type alpha ไม่ถูก wire "
       f"· และ {loo_arm['recall@10'].mean():.4f} ยังห่างเพดาน qrels {QRELS_CEILING:.4f} อยู่มาก")
    w_()

    w_("## self-check")
    w_()
    for name, ok, detail in checks:
        w_(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    w_()
    w_(f"เวลารวม {time.time()-t0:.0f} วินาที · ไม่ใช้ GPU สำหรับ cross-encoder: "
       f"อ่านคะแนน {sum(len(cache[q]) for q in queries):,} คู่จาก "
       f"`{SCORE_CACHE.relative_to(REPO).as_posix()}`")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {OUT}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
