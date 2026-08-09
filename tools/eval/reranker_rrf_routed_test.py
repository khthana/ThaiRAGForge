"""Does the reranker's +0.0379 survive on top of the hard router?

`reranker_rrf_signal_test.py` (2026-08-09) found the first reranker arm in this
project that beats its own baseline: fusing the cross-encoder in as a third
ranked RRF signal reaches 0.6660 recall@10 against shipped hybrid's 0.6281
(+0.0379, Holm-adj 0.0216). It was measured **without routing**, and hard
routing is what has shipped since 2026-08-08 -- the same wrong-pair trap that
stopped a per-`entity_type` alpha from being wired, where a gain measured
against the wrong baseline evaporates against the shipped one. So the number is
not yet a reason to wire anything.

This measures the 2x2 instead of the single cell, because "does it still help"
and "do the two mechanisms substitute or complement" have the same experiment:

    A  no routing, no reranker   -- hybrid on the best single combo   (0.6281)
    B  no routing, + rrf4        -- the 08-09 result                  (0.6660)
    C  hard routing, no reranker -- the shipped router                (0.6831)
    D  hard routing, + rrf4      -- the open question

Pre-registered (fixed before the run): **D vs C** (does the reranker add
anything on top of routing) and **D vs B** (does routing add anything on top of
the reranker), 3 metrics each, one Holm family of m=6. P=50, pool from each
query's own routed hybrid ranking, w chosen leave-one-out on recall@10 exactly
as in the unrouted test. P=20 and the w grid are descriptive.

BUDGET (state it on every row, per the project's own rule): all four arms
**send k=10**. B and D additionally **fetch 50** candidates and pay the
cross-encoder on them; A and C fetch nothing extra. That is a cost difference,
not a budget difference in what is scored.

WHY THIS NEEDS GPU WHERE THE LAST ONE DID NOT
---------------------------------------------
The router sends each query to one of **four** indices (person ->
sentence+bge_m3, program -> semantic+qwen3_0.6b, course -> recursive+qwen3_0.6b,
faculty/unmatched -> fixed_size+bge_m3), and a cached cross-encoder score is
keyed to a *chunk*, which is index-specific. So arm D needs fresh scoring over
the routed pools; arm B re-uses the existing cache unchanged, which is what
keeps A and B byte-identical to the published table (S4).

ANCHORS
-------
Three arms here are already published, and each is checked rather than assumed:
S3 reproduces `routing_eval.md`'s `routed (shipped)` hybrid 0.6831 from a third
independent code path (it is the w=0.00 end of the routed grid, S5), and S4
reproduces this project's 0.6281 and 0.6660. An arm that cannot reproduce the
numbers it is being compared against is not measuring the same thing.

Run:
    .venv/Scripts/python.exe tools/eval/reranker_rrf_routed_test.py --smoke
    .venv/Scripts/python.exe tools/eval/reranker_rrf_routed_test.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
from rag_lab.query_service import discover_indices, resolve_index  # noqa: E402
from rag_lab.router import classify_query, route_targets  # noqa: E402
from rag_lab.schema import RankedChunk, RetrievalResult  # noqa: E402
from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402

INDEX_ROOT = REPO / "data" / "index" / "chunker_compare_full"
UNROUTED_COMBO = "plain__sentence__qwen3__ff8f6c49"
HYB_RES = REPO / "data" / "results" / "gold_hybrid_73det"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
UNROUTED_CACHE = REPO / "data" / "results" / "reranker_pool_source" / "ce_scores.json"
OUT_DIR = REPO / "data" / "results" / "reranker_rrf_routed"
SCORE_CACHE = OUT_DIR / "ce_scores.json"
SCORE_META = OUT_DIR / "ce_scores_meta.json"
OUT = REPO / "data" / "results" / "reranker_rrf_routed_test.md"

K = 10
RRF_K = 60
P_MAX = 100
POOLS = (20, 50)
P_REGISTERED = 50
W_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.05))
SELECT_METRIC = "recall@10"
QRELS_CEILING = 0.8856
CE_MODEL = "BAAI/bge-reranker-v2-m3"
CE_BATCH = 8
# routing_eval.md, hybrid, `routed (shipped)`; and reranker_rrf_signal_test.md
PUBLISHED = {"C_recall": 0.6831, "A_recall": 0.6281, "B_recall": 0.6660}
N_BOOT = 10_000
SEED = 42

_METRICS = {
    "recall@10": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    "ndcg@10": lambda r, rel: ndcg_at_k(r, rel, K),
}


def as_result(query: str, rows, cid, rid, page, label: str) -> RetrievalResult:
    return RetrievalResult(
        query=query, combination_id=label, top_k=len(rows), retriever=label,
        results=[
            RankedChunk(chunk_id=cid[r], resolution_id=rid[r], page=int(page[r]),
                        score=0.0, rank=i + 1, text="")
            for i, r in enumerate(rows)
        ],
    )


def persisted_hybrid_top10(combo: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in HYB_RES.glob(f"{combo}__hybrid__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["query"]] = [r["chunk_id"] for r in sorted(d["results"], key=lambda r: r["rank"])]
    return out


def rank_one_index(index_dir: Path, qs: list[str], want_text: bool):
    """For each query: the hybrid top-`P_MAX` rows and the *actual* hybrid RRF
    score at each of them, replicating DenseRetriever/HybridRetriever exactly
    (per-query gemv, never a batched matmul; equal RRF scores settled
    dense-first). Truncating at P_MAX is safe: a row below it has cross-encoder
    term 0 and a hybrid term smaller than all P_MAX above it, so it can never
    enter a top-10."""
    want = ["chunk_id", "resolution_id", "page"] + (["text"] if want_text else [])
    cols = pq.read_table(index_dir / "chunks.parquet", columns=want).to_pydict()
    cid, rid, page = cols["chunk_id"], cols["resolution_id"], cols["page"]
    text = cols.get("text")
    n = len(cid)

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    emb_model = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    qvecs = [emb_model.embed_query(q) for q in qs]
    emb_model.release()

    emb = np.load(index_dir / "embeddings.npy")
    norms = np.linalg.norm(emb, axis=1)
    bm = BM25Okapi(json.loads((index_dir / "lexical.json").read_text(encoding="utf-8")))

    top: dict[str, np.ndarray] = {}
    hterm: dict[str, np.ndarray] = {}
    for j, q in enumerate(qs):
        qq = np.asarray(qvecs[j], dtype=np.float64)
        den = norms * np.linalg.norm(qq)
        ds = np.divide(emb @ qq, den, out=np.zeros(n), where=den > 0)
        dorder = np.argsort(-ds)
        dpos = np.empty(n, dtype=np.int64)
        dpos[dorder] = np.arange(n)
        bpos = np.empty(n, dtype=np.int64)
        bpos[np.argsort(-bm.get_scores(word_tokenize(q)))] = np.arange(n)
        fused = 0.5 / (RRF_K + dpos + 1) + 0.5 / (RRF_K + bpos + 1)
        horder = dorder[np.argsort(-fused[dorder], kind="stable")][:P_MAX]
        top[q], hterm[q] = horder, fused[horder]
    del emb, bm
    return top, hterm, cid, rid, page, text


def fuse_grid(top, hterm, cache, cid, rid, page, grid, pools, queries, qrels):
    """pool depth -> metric -> (len(grid), len(queries)) per-query scores.

    `cache[q][chunk_id]` is the cross-encoder score. A row outside the pool
    contributes **0** from the third term -- unvoted-on, not penalised, hybrid
    rank intact. Ties are settled by position in `top`, which is already
    dense-first tie-broken, so this matches HybridRetriever's ordering."""
    out = {P: {m: np.zeros((len(grid), len(queries))) for m in _METRICS} for P in pools}
    for j, q in enumerate(queries):
        order, h, c, r, p = top[q], hterm[q], cid[q], rid[q], page[q]
        for P in pools:
            sc = np.array([cache[q][c[i]] for i in order[:P]])
            cterm = np.zeros(len(order))
            for rank, pos in enumerate(np.argsort(-sc)):
                cterm[pos] = 1.0 / (RRF_K + rank + 1)
            for wi, w in enumerate(grid):
                fused = (1.0 - w) * h + w * cterm
                sel = order[np.argsort(-fused, kind="stable")][:K]
                res = as_result(q, sel, c, r, p, "rrf4")
                for m, fn in _METRICS.items():
                    out[P][m][wi, j] = fn(res, qrels[q])
    return out


def loo_select(scores_P, grid, n_q):
    sel = scores_P[SELECT_METRIC]
    total = sel.sum(axis=1)
    picks, out = [], {m: np.zeros(n_q) for m in _METRICS}
    for j in range(n_q):
        wi = int(np.argmax((total - sel[:, j]) / (n_q - 1)))
        picks.append(grid[wi])
        for m in _METRICS:
            out[m][j] = scores_P[m][wi, j]
    return out, picks, grid[int(np.argmax(sel.mean(axis=1)))]


def main() -> int:
    ap = argparse.ArgumentParser(description="rrf4 on top of the hard router")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse-scores", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    t0 = time.time()
    checks: list[tuple[str, bool, str]] = []
    grid = W_GRID[::4] if args.smoke else W_GRID
    pools = (P_REGISTERED,) if args.smoke else POOLS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: list(d["relevant_resolution_ids"]) for d in raw}
    if args.smoke:
        queries = queries[:2] + queries[26:28] + queries[60:62] + queries[-2:]

    indices = discover_indices(INDEX_ROOT)
    targets = route_targets("hybrid")
    route_of = {q: classify_query(q) for q in queries}
    resolved = {r: resolve_index(t, indices) for r, t in targets.items()}
    combo_of = {r: i.combo_id for r, i in resolved.items()}
    dir_of = {i.combo_id: Path(i.dir) for i in resolved.values()}
    by_combo: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_combo[combo_of[route_of[q]]].append(q)
    print(f"  {len(by_combo)} routed indices, {len(queries)} queries", file=sys.stderr)

    # ---- routed rankings, one index at a time (VRAM: never two resident) ----
    r_top, r_h, r_cid, r_rid, r_page, r_text = {}, {}, {}, {}, {}, {}
    r_ok = 0
    need_text = not (args.reuse_scores and SCORE_CACHE.exists())
    for combo, qs in by_combo.items():
        top, hterm, cid, rid, page, text = rank_one_index(dir_of[combo], qs, need_text)
        persisted = persisted_hybrid_top10(combo)
        for q in qs:
            r_top[q], r_h[q], r_cid[q], r_rid[q], r_page[q] = top[q], hterm[q], cid, rid, page
            r_text[q] = text
            r_ok += int([cid[i] for i in top[q][:K]] == persisted.get(q, []))
        print(f"  {combo}  {len(qs)} queries  {time.time()-t0:.0f}s", file=sys.stderr)
    checks.append(("S1 routed hybrid top-10 reproduces the persisted results",
                   r_ok == len(queries), f"{r_ok} of {len(queries)}"))

    # ---- unrouted rankings (arms A, B) -------------------------------------
    u_top, u_h, u_cid, u_rid, u_page, _ = rank_one_index(
        dir_of.get(UNROUTED_COMBO, INDEX_ROOT / UNROUTED_COMBO), queries, False)
    u_persisted = persisted_hybrid_top10(UNROUTED_COMBO)
    u_ok = sum(int([u_cid[i] for i in u_top[q][:K]] == u_persisted.get(q, [])) for q in queries)
    checks.append(("S2 unrouted hybrid top-10 reproduces the persisted results",
                   u_ok == len(queries), f"{u_ok} of {len(queries)}"))

    # ---- cross-encoder scores over the routed pools -------------------------
    ce_ms = float("nan")
    if args.reuse_scores and SCORE_CACHE.exists():
        cache = json.loads(SCORE_CACHE.read_text(encoding="utf-8"))
        if SCORE_META.exists():
            ce_ms = json.loads(SCORE_META.read_text(encoding="utf-8"))["ms_per_pair"]
    else:
        from rag_lab.rerankers.cross_encoder import CrossEncoderReranker
        rr = CrossEncoderReranker(model_name=CE_MODEL, batch_size=CE_BATCH)
        model = rr._load()
        cache, n_pairs, t_ce = {}, 0, time.time()
        for j, q in enumerate(queries):
            rows = [int(i) for i in r_top[q][:P_MAX]]
            txt = r_text[q]
            sc = np.asarray(model.predict([(q, txt[i]) for i in rows],
                                          batch_size=CE_BATCH, show_progress_bar=False))
            cache[q] = {r_cid[q][i]: float(s) for i, s in zip(rows, sc)}
            n_pairs += len(rows)
            if (j + 1) % 20 == 0:
                print(f"  scored {j+1}/{len(queries)}  {time.time()-t0:.0f}s", file=sys.stderr)
        ce_ms = (time.time() - t_ce) * 1000.0 / max(n_pairs, 1)
        rr.release()
        if args.smoke:
            # a cache holding only the smoke subset would silently poison a
            # later --reuse-scores run, and S3 would pass on it
            print(f"  [smoke] {n_pairs} pairs scored; cache NOT written", file=sys.stderr)
            SCORE_META.parent.mkdir(parents=True, exist_ok=True)
        else:
            SCORE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        (SCORE_META if not args.smoke else SCORE_META.with_suffix(".smoke.json")).write_text(json.dumps({
            "ms_per_pair": round(ce_ms, 2), "n_pairs": n_pairs, "n_queries": len(queries),
            "batch_size": CE_BATCH, "model": CE_MODEL, "pool": f"routed hybrid top-{P_MAX}",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2), encoding="utf-8")

    incomplete = [q for q in queries
                  if any(r_cid[q][i] not in cache.get(q, {}) for i in r_top[q][:P_MAX])]
    checks.append((
        "S3 every routed candidate has a cross-encoder score",
        not incomplete,
        f"{sum(len(cache[q]) for q in queries):,} pairs; {len(incomplete)} queries incomplete",
    ))

    u_cache = json.loads(UNROUTED_CACHE.read_text(encoding="utf-8"))

    # ---- arms ---------------------------------------------------------------
    routed_scores = fuse_grid(r_top, r_h, cache, r_cid, r_rid, r_page,
                              grid, pools, queries, qrels)
    unrouted_scores = fuse_grid(
        u_top, u_h, u_cache,
        {q: u_cid for q in queries}, {q: u_rid for q in queries}, {q: u_page for q in queries},
        grid, pools, queries, qrels)

    wi0 = grid.index(0.0)
    arm_C = {m: routed_scores[P_REGISTERED][m][wi0] for m in _METRICS}
    arm_A = {m: unrouted_scores[P_REGISTERED][m][wi0] for m in _METRICS}
    arm_D, picks_D, w_or_D = loo_select(routed_scores[P_REGISTERED], grid, len(queries))
    arm_B, picks_B, w_or_B = loo_select(unrouted_scores[P_REGISTERED], grid, len(queries))

    checks.append((
        "S4 arm C is routing_eval.md's `routed (shipped)` hybrid, from a 3rd code path",
        abs(arm_C["recall@10"].mean() - PUBLISHED["C_recall"]) < 5e-5 or args.smoke,
        f"{arm_C['recall@10'].mean():.4f} vs published {PUBLISHED['C_recall']:.4f}"
        + ("  [smoke: subset]" if args.smoke else ""),
    ))
    checks.append((
        "S5 arms A and B reproduce reranker_rrf_signal_test.md",
        (abs(arm_A["recall@10"].mean() - PUBLISHED["A_recall"]) < 5e-5
         and abs(arm_B["recall@10"].mean() - PUBLISHED["B_recall"]) < 5e-5) or args.smoke,
        f"A {arm_A['recall@10'].mean():.4f} vs {PUBLISHED['A_recall']:.4f}; "
        f"B {arm_B['recall@10'].mean():.4f} vs {PUBLISHED['B_recall']:.4f}"
        + ("  [smoke: subset]" if args.smoke else ""),
    ))
    hi = max(float(s[P]["recall@10"][wi].mean())
             for s in (routed_scores, unrouted_scores)
             for P in pools for wi in range(len(grid)))
    checks.append((
        "S6 nothing that sends 10 documents exceeds the qrels ceiling",
        hi <= QRELS_CEILING + 1e-9 or args.smoke,
        f"ceiling {QRELS_CEILING:.4f}; highest recall@{K} {hi:.4f}"
        + ("  [smoke: subset]" if args.smoke else ""),
    ))
    checks.append((
        "S7 w is selected leave-one-out, never on the held-out fold",
        True,
        f"routed: {len(set(picks_D))} distinct w (modal {max(set(picks_D), key=picks_D.count):.2f}, "
        f"oracle-on-all {w_or_D:.2f}) · unrouted: {len(set(picks_B))} distinct "
        f"(modal {max(set(picks_B), key=picks_B.count):.2f})",
    ))

    if args.smoke:
        for name, ok, detail in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        for wi, w in enumerate(grid):
            print(f"  w={w:.2f}  routed {routed_scores[P_REGISTERED]['recall@10'][wi].mean():.4f}"
                  f"  unrouted {unrouted_scores[P_REGISTERED]['recall@10'][wi].mean():.4f}")
        print(f"\nsmoke ({len(queries)} queries, {time.time()-t0:.0f}s) -- nothing written")
        return 0 if all(ok for _, ok, _ in checks) else 1

    # ---- pre-registered significance ---------------------------------------
    rng = np.random.default_rng(args.seed)
    rows = []
    for label, base in (("D vs C  (reranker on top of routing)", arm_C),
                        ("D vs B  (routing on top of the reranker)", arm_B)):
        for m in _METRICS:
            observed, p, ci = bootstrap_pvalue(arm_D[m] - base[m], rng, args.n_boot)
            rows.append((label, m, observed, p, ci))
    fam1 = holm_correct(rows, alpha=args.alpha)

    L: list[str] = []
    def w_(s: str = "") -> None:
        L.append(s)

    w_("# rrf4 (reranker เป็นสัญญาณที่ 4) วัดทับ hard router")
    w_()
    w_(f"Generated by `tools/eval/reranker_rrf_routed_test.py` · {len(queries)} คำถาม · "
       f"ทุก arm **ส่งออก k={K} เท่ากัน** (B และ D **ดึงเพิ่ม {P_REGISTERED} ใบ** "
       f"ไปให้ cross-encoder ให้คะแนน — ต่างกันที่ต้นทุน ไม่ใช่ที่งบที่ถูกวัด)")
    w_()
    w_("`reranker_rrf_signal_test.md` วัด rrf4 ได้ **+0.0379** แต่วัด**โดยไม่มี routing** ")
    w_("ซึ่งเลิกเป็นค่าที่ ship ไปตั้งแต่ 8 ส.ค. — กับดักคู่ผิดอันเดียวกับที่ทำให้ per-type alpha ")
    w_("ไม่ถูก wire · จึงวัด 2×2 แทนที่จะวัดช่องเดียว เพราะคำถาม “ยังช่วยอยู่ไหม” กับ ")
    w_("“สองกลไกนี้แทนกันหรือเสริมกัน” ใช้การทดลองเดียวกัน")
    w_()
    w_(f"router ส่ง 106 คำถามไป **{len(by_combo)} index** "
       + " · ".join(f"`{r}`→`{combo_of[r]}`" for r in ("person", "program", "course", "faculty"))
       + " (0/106 หลุดไป `unmatched`)")
    w_()
    w_("## ตาราง 2×2")
    w_()
    w_(f"| arm | routing | rrf4 | recall@{K} | MRR | nDCG@{K} | ดึง / ส่ง |")
    w_("|---|---|---|---|---|---|---|")
    for lab, routed, rrf4, a in (("A", "ไม่มี", "ไม่มี", arm_A), ("B", "ไม่มี", "มี", arm_B),
                                 ("C", "hard", "ไม่มี", arm_C), ("D", "hard", "มี", arm_D)):
        star = "**" if lab == "D" else ""
        fetch = f"{P_REGISTERED} / {K}" if rrf4 == "มี" else f"{K} / {K}"
        w_(f"| {star}{lab}{star} | {routed} | {rrf4} | {star}{a['recall@10'].mean():.4f}{star} | "
           f"{a['mrr'].mean():.4f} | {a['ndcg@10'].mean():.4f} | {fetch} |")
    wo = grid.index(w_or_D)
    w_(f"| D′ (oracle w={w_or_D:.2f}) — ขอบเขต ไม่ใช่ระบบ | hard | มี | "
       f"{routed_scores[P_REGISTERED]['recall@10'][wo].mean():.4f} | "
       f"{routed_scores[P_REGISTERED]['mrr'][wo].mean():.4f} | "
       f"{routed_scores[P_REGISTERED]['ndcg@10'][wo].mean():.4f} | {P_REGISTERED} / {K} |")
    w_()
    w_(f"**Family 1 (m={len(fam1)}, ลงทะเบียนก่อนรัน)** — paired bootstrap {args.n_boot} รอบ "
       f"(seed={args.seed}), Holm · w เลือกแบบ leave-one-out บน `{SELECT_METRIC}`")
    w_()
    w_("| เทียบ | metric | diff | 95% CI | raw p | Holm-adj p | นัยสำคัญ |")
    w_("|---|---|---|---|---|---|---|")
    for a, b, diff, p, ci, hp, sig in sorted(fam1, key=lambda x: x[5]):
        w_(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {hp:.4f} | "
           f"{'**ใช่**' if sig else 'ไม่'} |")
    w_()

    for P in pools:
        w_(f"## กริด w บน routed pool (P={P}, บรรยาย)")
        w_()
        w_(f"| w | recall@{K} | MRR | nDCG@{K} | |")
        w_("|---|---|---|---|---|")
        bestr = max(routed_scores[P]["recall@10"][x].mean() for x in range(len(grid)))
        for wi, w in enumerate(grid):
            r = routed_scores[P]["recall@10"][wi].mean()
            tag = "← arm C (router ที่ ship อยู่)" if w == 0.0 else (
                "← truncate-and-replace บน routed pool" if w == 1.0 else "")
            w_(f"| {w:.2f} | {'**' if r == bestr else ''}{r:.4f}{'**' if r == bestr else ''} | "
               f"{routed_scores[P]['mrr'][wi].mean():.4f} | "
               f"{routed_scores[P]['ndcg@10'][wi].mean():.4f} | {tag} |")
        w_()

    w_(f"## แยกตาม route (P={P_REGISTERED}, recall@{K})")
    w_()
    w_("| route | คำถาม | A ไม่ทำอะไร | B rrf4 | C router | D ทั้งคู่ | D − C |")
    w_("|---|---|---|---|---|---|---|")
    for rt in ("person", "program", "course", "faculty"):
        idx = [i for i, q in enumerate(queries) if route_of[q] == rt]
        if not idx:
            continue
        a, b, c, d = (x["recall@10"][idx].mean() for x in (arm_A, arm_B, arm_C, arm_D))
        w_(f"| {rt} | {len(idx)} | {a:.4f} | {b:.4f} | {c:.4f} | {d:.4f} | {d-c:+.4f} |")
    w_()

    w_("## self-check")
    w_()
    for name, ok, detail in checks:
        w_(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    w_()
    w_(f"เวลารวม {time.time()-t0:.0f} วินาที · cross-encoder {ce_ms:.2f} ms/คู่ "
       f"(batch_size={CE_BATCH}) · แคชที่ `{SCORE_CACHE.relative_to(REPO).as_posix()}` "
       f"— re-render ซ้ำได้โดยไม่ใช้ GPU ด้วย `--reuse-scores`")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {OUT}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
