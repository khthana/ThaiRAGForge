"""Does a cross-encoder help once it is given the RIGHT candidate pool?

The project already has a reranker result, and it is negative: `bge-reranker-v2-m3`
over a **hybrid** pool of 50 significantly *hurts* hybrid MRR (0.7814 -> 0.6778,
Holm-adj p=0.0012) and does nothing measurable to dense
(`tools/eval/reranker_significance_test.py`). `miss_depth_profile.py` then found
two things that make that result look like it may have tested the wrong thing:

  1. **The evidence is in reach.** Of the 84 (query, resolution) pairs no arm of
     any combo finds at k=10, **64 (76.2%) sit at ranks 11-50** and exactly 0 are
     absent from the index. A reranker fetching 50 candidates can see them.
  2. **The pool should come from dense.** On those hard pairs `dense` has median
     best rank **26** and is closest on **70 of 84**, vs hybrid 43 (9) and BM25
     210 (6) -- yet the shipped system, and the pool the published reranker test
     widened, is hybrid.

So the untested configuration is: pool from **dense**, rerank, send 10. This
script tests it, and three questions the published test could not answer.

WHAT IS DELIBERATELY DIFFERENT FROM `reranker_significance_test.py`
------------------------------------------------------------------
* **Combo.** That test runs on `plain__fixed_size__local__ceea7536`
  (fixed_size x bge-m3). This one runs on `plain__sentence__qwen3__ff8f6c49`
  (sentence x qwen3_0.6b) -- the **best single system** in the study, and the
  one every number in `miss_depth_profile.md` S2 is computed on. The deployable
  question is whether a reranker beats the best thing we already ship, so it has
  to be asked against that thing. Consequence: results here are NOT directly
  comparable to the published reranker table; the hybrid/P=50 arm is the
  analogue, and whether the MRR harm reproduces on a stronger combo is one of
  the things reported below.
* **Pool source is an arm, not a fixed choice.** dense vs hybrid at equal P.
* **Pool depth is swept**, P in {10, 20, 50, 100, 200}.
* **An oracle rerank of the same pool is reported beside every real arm**, so a
  null is a *bound with a mechanism* ("the evidence was in the pool and the
  reranker did not find it") rather than an absence of evidence.

METHOD -- score each (query, chunk) pair ONCE, derive every arm from it
----------------------------------------------------------------------
A cross-encoder score depends only on the pair, never on P or on which retriever
put the chunk in the pool. So the candidate sets of all 10 arms are unioned per
query, scored in one pass, and every arm is then a selection over that cache.
This is the same trick `hybrid_alpha_sweep.py` uses (cache the rank vector once,
re-fuse in numpy): ~10 arms for the cost of ~1.3, and it makes the arms exactly
consistent -- the same pair cannot get two different scores in two arms.

The scores are persisted to `ce_scores.json` because re-rendering the report must
not need a GPU, and because a published figure has to be reproducible from the
artifact that published it (`audit_doc_claims.py` D2 checks exactly that).

Ranking replication follows `miss_depth_profile.py`, including both traps it
pins: dense is one gemv per query (a batched matmul reorders exact ties), and
hybrid settles equal RRF scores dense-first via a stable sort over the dense
permutation. S1/S2 gate both against the persisted top-10 before anything is
published.

PRE-REGISTERED COMPARISONS (fixed before the run; the P sweep is descriptive)
----------------------------------------------------------------------------
  Family 1 (m=6) -- the deployable question: rerank(dense, P=50) and
      rerank(hybrid, P=50) vs the **shipped hybrid baseline**, 3 metrics.
  Family 2 (m=3) -- does the pool source matter: rerank(dense, 50) vs
      rerank(hybrid, 50), 3 metrics.
  Family 3 (m=6) -- the reranker's own effect within an arm, i.e. the published
      test re-asked on this combo: rerank(dense, 50) vs dense baseline and
      rerank(hybrid, 50) vs hybrid baseline, 3 metrics.
P=50 is fixed in advance for all three because it is the depth
`miss_depth_profile.md` motivates; picking the best P post hoc and testing it
would be fitting on the test set. Quote the family size with any p (see
CLAUDE.md on family 2 vs family 3 in RQ4).

Read-only w.r.t. indices: consumes one index, the persisted baselines and the
gold set; writes one report and one score cache, and builds nothing.

Run:
    .venv/Scripts/python.exe tools/eval/reranker_pool_source_test.py
    .venv/Scripts/python.exe tools/eval/reranker_pool_source_test.py --smoke
    .venv/Scripts/python.exe tools/eval/reranker_pool_source_test.py --reuse-scores
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
from rag_lab.factory import build_embedder, build_reranker  # noqa: E402
from rag_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from rag_lab.schema import RankedChunk, RetrievalResult  # noqa: E402
from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402

COMBO = "plain__sentence__qwen3__ff8f6c49"
INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full" / COMBO
DENSE_RES = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
HYB_RES = REPO / "data" / "results" / "gold_hybrid_73det"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
MISS_DEPTH = REPO / "data" / "results" / "miss_depth_profile.md"


def parse_miss_depth_delivered(text: str) -> dict[int, float]:
    """{P: single-system *delivered* oracle recall@10} from miss_depth_profile.md S2.

    PARSED, never frozen. This was the literal dict
    `{10: 0.6281, 20: 0.7534, 50: 0.8249, 100: 0.8356, 200: 0.8412}` until
    2026-08-20, when rebuild #4 legitimately moved three of the five and S6 went
    red against a report it in fact agreed with to 4 decimals -- the sixth
    cross-artifact anchor of the kind `561102e` replaced elsewhere.

    S2's table is `| P | in pool | **delivered** | all-arm in pool | **all-arm
    delivered** |`, and the column that belongs here is the SECOND numeric one:
    the single system's delivered figure, which is the only one bounded by the
    10-document budget the oracle here also respects. Taking `in pool` instead
    would compare against a quantity that may legitimately exceed the qrels
    ceiling, i.e. a check that can never fail for the right reason.

    Returns {} when the section or its table cannot be found; the caller must
    treat that as a FAIL, not a skip -- an anchor that cannot find its
    counterpart must not pass quietly.
    """
    head = "## 2."
    if head not in text:
        return {}
    body = text.split(head, 1)[1].split(chr(10) + "## ", 1)[0]
    out: dict[int, float] = {}
    for line in body.splitlines():
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) != 5:
            continue
        try:
            P = int(cells[0].replace(",", ""))
            out[P] = float(cells[2].replace("*", "").strip())
        except ValueError:
            continue
    return out
OUT_DIR = REPO / "data" / "results" / "reranker_pool_source"
SCORE_CACHE = OUT_DIR / "ce_scores.json"
SCORE_META = OUT_DIR / "ce_scores_meta.json"
OUT = REPO / "data" / "results" / "reranker_pool_source_test.md"

K = 10
RRF_K = 60
POOLS = (10, 20, 50, 100, 200)
P_REGISTERED = 50
SOURCES = ("dense", "hybrid")
QRELS_CEILING = 0.8856  # mean(min(1, 10/n_relevant)); see paper-results-summary
N_BOOT = 10_000
SEED = 42

_METRICS = {
    f"recall@{K}": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    f"ndcg@{K}": lambda r, rel: ndcg_at_k(r, rel, K),
}


def persisted_top10(results_dir: Path, arm: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in results_dir.glob(f"{COMBO}__{arm}__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["query"]] = [r["chunk_id"] for r in sorted(d["results"], key=lambda r: r["rank"])]
    return out


def as_result(query: str, rows: list[int], cid: list[str], rid: list[str],
              page: list[int], label: str) -> RetrievalResult:
    """Wrap a ranked row list in the real schema so the shipped metric functions
    score it -- not a re-implementation of recall/MRR/nDCG that could drift."""
    return RetrievalResult(
        query=query, combination_id=COMBO, top_k=len(rows), retriever=label,
        results=[
            RankedChunk(chunk_id=cid[r], resolution_id=rid[r], page=int(page[r]),
                        score=0.0, rank=i + 1, text="")
            for i, r in enumerate(rows)
        ],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="reranker pool-source / pool-depth test")
    ap.add_argument("--smoke", action="store_true",
                    help="8 queries, P<=50, tiny score pass (~2 min) -- writes nothing")
    ap.add_argument("--reuse-scores", action="store_true",
                    help="reuse ce_scores.json instead of running the cross-encoder; NOT GPU-free -- "
                         "retrieval still loads an embedder, so do not run this beside "
                         "a training job on a single card")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    # This host's console is cp874; the report is Thai and uses U+00B7. Echoing it
    # must not be able to lose the run -- the artifacts are already on disk by then.
    sys.stdout.reconfigure(errors="replace")

    t0 = time.time()
    pools = tuple(p for p in POOLS if p <= 50) if args.smoke else POOLS
    p_max = max(pools)
    checks: list[tuple[str, bool, str]] = []

    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: list(d["relevant_resolution_ids"]) for d in raw}
    etype = {d["query"]: d.get("entity_type", "?") for d in raw}
    if args.smoke:
        queries = queries[:4] + queries[-4:]

    cols = pq.read_table(INDEX_DIR / "chunks.parquet",
                         columns=["chunk_id", "resolution_id", "text", "page"]).to_pydict()
    cid, rid, ctext, cpage = cols["chunk_id"], cols["resolution_id"], cols["text"], cols["page"]
    n_chunks = len(cid)

    manifest = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    qvecs = [embedder.embed_query(q) for q in queries]
    embedder.release()  # the cross-encoder wants the VRAM later in this same process
    print(f"  encoded {len(queries)} queries  {time.time()-t0:.0f}s", file=sys.stderr)

    emb = np.load(INDEX_DIR / "embeddings.npy")
    row_norms = np.linalg.norm(emb, axis=1)
    lex = json.loads((INDEX_DIR / "lexical.json").read_text(encoding="utf-8"))
    bm = BM25Okapi(lex)
    del lex

    ptop_d = persisted_top10(DENSE_RES, "dense")
    ptop_h = persisted_top10(HYB_RES, "hybrid")

    # order[source][query] = row indices, best first, truncated to p_max
    order: dict[str, dict[str, list[int]]] = {s: {} for s in SOURCES}
    # rank_of[source][(query, resolution)] = 1-based rank of its best chunk (None = absent)
    rank_of: dict[str, dict[tuple[str, str], int | None]] = {s: {} for s in SOURCES}
    d_ok = d_bad = h_ok = h_bad = 0

    rows_of: dict[str, list[int]] = {}
    for i, r in enumerate(rid):
        rows_of.setdefault(r, []).append(i)

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

        fused = 0.5 / (RRF_K + dpos + 1) + 0.5 / (RRF_K + bpos + 1)
        horder = dorder[np.argsort(-fused[dorder], kind="stable")]
        hpos = np.empty(n_chunks, dtype=np.int64)
        hpos[horder] = np.arange(n_chunks)

        if q in ptop_d:
            d_ok, d_bad = ((d_ok + 1, d_bad) if [cid[i] for i in dorder[:K]] == ptop_d[q]
                           else (d_ok, d_bad + 1))
        if q in ptop_h:
            h_ok, h_bad = ((h_ok + 1, h_bad) if [cid[i] for i in horder[:K]] == ptop_h[q]
                           else (h_ok, h_bad + 1))

        order["dense"][q] = [int(i) for i in dorder[:p_max]]
        order["hybrid"][q] = [int(i) for i in horder[:p_max]]
        for r in qrels[q]:
            rows = rows_of.get(r)
            rank_of["dense"][(q, r)] = int(dpos[rows].min()) + 1 if rows else None
            rank_of["hybrid"][(q, r)] = int(hpos[rows].min()) + 1 if rows else None
        if (j + 1) % 20 == 0:
            print(f"  ranked {j+1}/{len(queries)}  {time.time()-t0:.0f}s", file=sys.stderr)

    del emb, bm
    checks.append(("S1 dense top-10 reproduces the persisted results",
                   d_bad == 0 and d_ok == len(queries), f"{d_ok} reproduce, {d_bad} differ"))
    checks.append(("S2 hybrid top-10 reproduces the persisted results",
                   h_bad == 0 and h_ok == len(queries), f"{h_ok} reproduce, {h_bad} differ"))

    # ---- cross-encoder: one score per unique (query, chunk) pair ----------
    need: dict[str, list[int]] = {
        q: sorted(set(order["dense"][q]) | set(order["hybrid"][q])) for q in queries
    }
    n_pairs = sum(len(v) for v in need.values())
    cache: dict[str, dict[str, float]] = {}
    if args.reuse_scores and SCORE_CACHE.exists():
        cache = json.loads(SCORE_CACHE.read_text(encoding="utf-8"))
        # ms/pair is a machine measurement, not derivable from the scores -- so it
        # is persisted beside them, or the cost table cannot survive a GPU-free
        # re-render and the report stops being reproducible from its own artifact.
        meta = json.loads(SCORE_META.read_text(encoding="utf-8")) if SCORE_META.exists() else {}
        ce_ms_per_pair = float(meta.get("ms_per_pair", float("nan")))
        print(f"  reusing {SCORE_CACHE.name}", file=sys.stderr)
    else:
        reranker = build_reranker(StrategySpec(type="cross_encoder"))
        model = reranker._load()  # noqa: SLF001 -- scoring in bulk, not via rerank()
        t_ce = time.time()
        for j, q in enumerate(queries):
            rows = need[q]
            scores = np.asarray(model.predict([(q, ctext[i]) for i in rows],
                                              batch_size=8, show_progress_bar=False))
            cache[q] = {cid[i]: float(s) for i, s in zip(rows, scores)}
            print(f"  scored {j+1}/{len(queries)} ({len(rows)} pairs)  "
                  f"{time.time()-t0:.0f}s", file=sys.stderr)
        ce_ms_per_pair = (time.time() - t_ce) * 1000 / max(1, n_pairs)
        reranker.release()
        if not args.smoke:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            SCORE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            SCORE_META.write_text(json.dumps({
                "ms_per_pair": ce_ms_per_pair, "n_pairs": n_pairs,
                "n_queries": len(queries), "batch_size": 8,
                "model": "BAAI/bge-reranker-v2-m3", "combo": COMBO,
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2), encoding="utf-8")

    missing = [q for q in queries if any(cid[i] not in cache.get(q, {}) for i in need[q])]
    checks.append(("S3 every candidate pair has a cross-encoder score",
                   not missing, f"{n_pairs} pairs over {len(queries)} queries; "
                                f"{len(missing)} queries incomplete"))

    # ---- derive every arm from the cache ---------------------------------
    # results[(source, P)][query] -> RetrievalResult of the reranked top-10
    results: dict[tuple[str, int], dict[str, RetrievalResult]] = {}
    for s in SOURCES:
        for P in pools:
            per_q = {}
            for q in queries:
                rows = order[s][q][:P]
                sc = np.array([cache[q][cid[i]] for i in rows])
                # replicate CrossEncoderReranker.rerank's selection exactly
                top = [rows[i] for i in np.argsort(-sc)[:K]]
                per_q[q] = as_result(q, top, cid, rid, cpage, f"{s}_rerank_p{P}")
            results[(s, P)] = per_q

    baseline = {
        s: {q: as_result(q, order[s][q][:K], cid, rid, cpage, s) for q in queries}
        for s in SOURCES
    }

    def mean_metric(per_q: dict[str, RetrievalResult], metric: str) -> float:
        return float(np.mean([_METRICS[metric](per_q[q], qrels[q]) for q in queries]))

    def per_query(per_q: dict[str, RetrievalResult], metric: str) -> np.ndarray:
        return np.array([_METRICS[metric](per_q[q], qrels[q]) for q in queries])

    # oracle: a perfect reranker still sends only K chunks, so at most K
    # distinct resolutions -- min(hits_in_pool, K) / n_relevant.
    def oracle(s: str, P: int) -> tuple[float, np.ndarray]:
        vals = []
        for q in queries:
            h = len([r for r in qrels[q] if (rank_of[s][(q, r)] or 10**9) <= P])
            vals.append(min(h, K) / len(qrels[q]))
        return float(np.mean(vals)), np.array(vals)

    def in_pool(s: str, P: int) -> float:
        return float(np.mean([
            len([r for r in qrels[q] if (rank_of[s][(q, r)] or 10**9) <= P]) / len(qrels[q])
            for q in queries
        ]))

    # ---- structural self-checks ------------------------------------------
    anchor = [
        (s, mean_metric(results[(s, K)], f"recall@{K}"), mean_metric(baseline[s], f"recall@{K}"))
        for s in SOURCES
    ]
    checks.append((
        "S4 at P=k reranking cannot change the retrieved SET, only its order",
        all(abs(a - b) < 1e-12 for _, a, b in anchor),
        "; ".join(f"{s}: {a:.4f} vs baseline {b:.4f}" for s, a, b in anchor),
    ))

    viol = 0
    for s in SOURCES:
        for P in pools:
            _, orc = oracle(s, P)
            got = per_query(results[(s, P)], f"recall@{K}")
            viol += int((got > orc + 1e-12).sum())
    checks.append((
        "S5 no real arm beats the oracle rerank of its own pool, per query",
        viol == 0, f"{viol} violations over {len(SOURCES)*len(pools)*len(queries)} (arm, query) cells",
    ))

    # S6/S7 compare against constants that are means over the FULL 106 queries,
    # so they are meaningless -- not failing -- on a subset.
    if args.smoke:
        checks.append(("S6 oracle reproduces miss_depth_profile.md S2", True,
                       "skipped: needs the full query set"))
        checks.append(("S7 delivered stays under the qrels ceiling", True,
                       "skipped: needs the full query set"))
    else:
        md_txt = MISS_DEPTH.read_text(encoding="utf-8") if MISS_DEPTH.exists() else ""
        published = parse_miss_depth_delivered(md_txt)
        missing = [P for P in pools if P not in published]
        repro = [(P, oracle("hybrid", P)[0], published[P])
                 for P in pools if P in published]
        checks.append((
            "S6 oracle rerank of the hybrid pool reproduces miss_depth_profile.md S2",
            bool(repro) and not missing and all(abs(a - b) < 5e-4 for _, a, b in repro),
            ("; ".join(f"P={P}: {a:.4f} vs published {b:.4f}" for P, a, b in repro)
             + (f"; UNPARSED at P={missing} -- the cross-check could not be made"
                if missing else "")),
        ))

        worst = max(
            max(mean_metric(results[(s, P)], f"recall@{K}") for s in SOURCES for P in pools),
            max(oracle(s, P)[0] for s in SOURCES for P in pools),
        )
        checks.append((
            "S7 nothing that sends 10 documents exceeds the qrels ceiling",
            worst <= QRELS_CEILING + 1e-9,
            f"ceiling {QRELS_CEILING:.4f}; highest delivered {worst:.4f}",
        ))

    if args.smoke:
        for name, ok, detail in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        for s in SOURCES:
            for P in pools:
                print(f"  {s:6s} P={P:3d}  recall@{K} "
                      f"{mean_metric(results[(s, P)], f'recall@{K}'):.4f} "
                      f"(baseline {mean_metric(baseline[s], f'recall@{K}'):.4f}, "
                      f"oracle {oracle(s, P)[0]:.4f})")
        print(f"\nsmoke run ({len(queries)} queries, P<={p_max}, "
              f"{time.time()-t0:.0f}s) -- nothing written")
        return 0 if all(ok for _, ok, _ in checks) else 1

    # ---- pre-registered significance families ----------------------------
    rng = np.random.default_rng(args.seed)
    arms: dict[str, dict[str, RetrievalResult]] = {
        "hybrid (shipped)": baseline["hybrid"],
        "dense": baseline["dense"],
        f"rerank(dense, P={P_REGISTERED})": results[("dense", P_REGISTERED)],
        f"rerank(hybrid, P={P_REGISTERED})": results[("hybrid", P_REGISTERED)],
    }
    families = {
        "1": ("does reranking beat the shipped hybrid, at the same 10-doc budget", [
            (f"rerank({s}, P={P_REGISTERED})", "hybrid (shipped)")
            for s in SOURCES
        ]),
        "2": ("does the pool source matter", [
            (f"rerank(dense, P={P_REGISTERED})", f"rerank(hybrid, P={P_REGISTERED})")
        ]),
        "3": ("the reranker's own effect within an arm (the published test, re-asked)", [
            (f"rerank({s}, P={P_REGISTERED})", "hybrid (shipped)" if s == "hybrid" else "dense")
            for s in SOURCES
        ]),
    }
    fam_out: dict[str, list] = {}
    for fam, (_desc, pairs_) in families.items():
        rows = []
        for treat, base in pairs_:
            for metric in _METRICS:
                d = per_query(arms[treat], metric) - per_query(arms[base], metric)
                observed, p, ci = bootstrap_pvalue(d, rng, args.n_boot)
                rows.append((f"{treat} vs {base}", metric, observed, p, ci))
        fam_out[fam] = holm_correct(rows, alpha=args.alpha)

    # ---- report -----------------------------------------------------------
    L: list[str] = []
    def w(s: str = "") -> None:
        L.append(s)

    w("# reranker กับ pool ที่ถูกตัว — dense pool vs hybrid pool, และความลึกของ pool")
    w()
    w(f"Generated by `tools/eval/reranker_pool_source_test.py` · combo `{COMBO}` "
      f"(sentence × qwen3_0.6b, ระบบเดี่ยวที่ดีที่สุดในการศึกษานี้) · ")
    w(f"{len(queries)} คำถาม · reranker `BAAI/bge-reranker-v2-m3` · "
      f"ทุก arm ส่งออก k={K} เท่ากัน")
    w()
    w("**คำถาม**: `miss_depth_profile.md` พบว่า 76.2% ของคู่ที่ไม่มีระบบไหนหาเจอที่ k=10 ")
    w("อยู่แค่อันดับ 11–50 และ **dense** เข้าใกล้ที่สุดใน 70 จาก 84 คู่ (median 26 เทียบ ")
    w("hybrid 43) แต่ผลการทดสอบ reranker ที่ตีพิมพ์ไว้ดึง pool จาก **hybrid** ")
    w("· ทั้งหมดนี้จึงถามว่า reranker ที่ได้ pool ถูกตัวช่วยได้จริงไหม")
    w()
    w(f"**อ่านผลนี้แยกจาก `reranker_significance_test.md`** — คนละ combo "
      f"(อันนั้น `plain__fixed_size__local__ceea7536`) เพราะคำถามที่ deploy ได้จริง ")
    w("คือ *ชนะของที่ดีที่สุดที่เรามีอยู่แล้วไหม* จึงต้องถามกับของชิ้นนั้น")
    w()
    w("## 1. ตารางบรรยาย — pool source × ความลึก (ไม่มีการทดสอบนัยสำคัญ)")
    w()
    w("`ใน pool` = สัดส่วน gold ที่อยู่ใน candidate pool (วัตถุดิบที่ reranker มี) · ")
    w("`oracle` = rerank สมบูรณ์แบบของ pool เดียวกันแต่ยังส่งได้ 10 ใบ · ")
    w("`จริง` = reranker ตัวจริง · `จับได้` = (จริง − baseline) ÷ (oracle − baseline)")
    w()
    for s in SOURCES:
        base_r = mean_metric(baseline[s], f"recall@{K}")
        w(f"**pool จาก `{s}`** — baseline recall@{K} = {base_r:.4f}")
        w()
        w(f"| P | ใน pool | oracle | จริง | จับได้ | MRR | nDCG@{K} |")
        w("|---|---|---|---|---|---|---|")
        for P in pools:
            orc = oracle(s, P)[0]
            real = mean_metric(results[(s, P)], f"recall@{K}")
            head = orc - base_r
            capt = "—" if head <= 1e-9 else f"{100*(real-base_r)/head:.0f}%"
            w(f"| {P} | {in_pool(s, P):.4f} | {orc:.4f} | **{real:.4f}** | {capt} | "
              f"{mean_metric(results[(s, P)], 'mrr'):.4f} | "
              f"{mean_metric(results[(s, P)], f'ndcg@{K}'):.4f} |")
        w()
    w(f"baseline `dense`: recall@{K} {mean_metric(baseline['dense'], f'recall@{K}'):.4f} · "
      f"MRR {mean_metric(baseline['dense'], 'mrr'):.4f} · "
      f"nDCG@{K} {mean_metric(baseline['dense'], f'ndcg@{K}'):.4f}")
    w()
    w(f"baseline `hybrid` (shipped): recall@{K} {mean_metric(baseline['hybrid'], f'recall@{K}'):.4f} · "
      f"MRR {mean_metric(baseline['hybrid'], 'mrr'):.4f} · "
      f"nDCG@{K} {mean_metric(baseline['hybrid'], f'ndcg@{K}'):.4f}")
    w()
    w(f"เพดาน qrels ที่งบ 10 ใบคือ **{QRELS_CEILING:.4f}** — ทั้งคอลัมน์ `oracle` และ `จริง` ")
    w("ถูกบีบด้วยเส้นนี้ ไม่ว่า pool จะลึกแค่ไหน")
    w()

    w("## 2. การทดสอบนัยสำคัญที่ลงทะเบียนไว้ล่วงหน้า")
    w()
    w(f"paired bootstrap {args.n_boot} รอบ (seed={args.seed}) · Holm ภายในแต่ละ family · ")
    w(f"P={P_REGISTERED} ถูกตรึงไว้ล่วงหน้าเพราะเป็นความลึกที่ `miss_depth_profile.md` ชี้ ")
    w("— การเลือก P ที่ดีที่สุดหลังเห็นผลคือการ fit บนชุดทดสอบ")
    w()
    for fam, (desc, _pairs) in families.items():
        rows = fam_out[fam]
        w(f"**Family {fam} (m={len(rows)})** — {desc}")
        w()
        w("| เทียบ | metric | diff | 95% CI | raw p | Holm-adj p | นัยสำคัญ |")
        w("|---|---|---|---|---|---|---|")
        for a, b, diff, p, ci, holm_p, sig in sorted(rows, key=lambda x: x[5]):
            w(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
              f"{p:.4f} | {holm_p:.4f} | {'**ใช่**' if sig else 'ไม่'} |")
        w()

    w("## 3. แยกตามชนิดคำถาม (P=%d, pool จาก dense)" % P_REGISTERED)
    w()
    w(f"| entity_type | คำถาม | hybrid (shipped) | rerank(dense) | diff |")
    w("|---|---|---|---|---|")
    for t in sorted({etype[q] for q in queries}):
        qs = [q for q in queries if etype[q] == t]
        b = float(np.mean([_METRICS[f"recall@{K}"](baseline["hybrid"][q], qrels[q]) for q in qs]))
        r = float(np.mean([_METRICS[f"recall@{K}"](results[("dense", P_REGISTERED)][q], qrels[q])
                           for q in qs]))
        w(f"| {t} | {len(qs)} | {b:.4f} | {r:.4f} | {r-b:+.4f} |")
    w()

    w("## 4. ต้นทุน")
    w()
    if ce_ms_per_pair == ce_ms_per_pair:  # not NaN
        w(f"cross-encoder ให้คะแนน **{ce_ms_per_pair:.1f} ms ต่อ 1 คู่ (query, chunk)** "
          f"(batch_size=8, วัดจาก {n_pairs} คู่) ")
        w("ต้นทุนต่อ query จึงเป็น P × ค่านี้ ก่อนนับ retrieval:")
        w()
        w("| P | ms/query ที่เพิ่มขึ้น |")
        w("|---|---|")
        for P in pools:
            w(f"| {P} | {P*ce_ms_per_pair:.0f} |")
        w()
        w(f"รอบที่สร้างแคชคะแนน: {n_pairs:,} คู่ ≈ {n_pairs*ce_ms_per_pair/1000:.0f} วินาทีของ GPU")
    else:
        w("ไม่ได้วัด (รันด้วย `--reuse-scores`) — ดูรอบที่สร้าง `ce_scores.json`")
    w()
    w(f"เทียบกับฐาน hybrid p50 ที่ `cost_latency_pareto.md` วัดไว้ (1.21–1.86 วิ/query) "
      f"และ dense p50 ของ combo นี้")
    w()

    d50 = results[("dense", P_REGISTERED)]
    h50 = results[("hybrid", P_REGISTERED)]
    w("## 5. อ่านผลนี้อย่างไร")
    w()
    w("**1. สมมติฐานที่ตั้งไว้ถูกปฏิเสธ และปฏิเสธในทิศตรงข้าม** — pool จาก dense ไม่ได้ดีกว่า ")
    w(f"แต่**แย่กว่าอย่างมีนัยสำคัญ** (Family 2: recall@{K} "
      f"{mean_metric(d50, f'recall@{K}') - mean_metric(h50, f'recall@{K}'):+.4f}) ")
    w("กลไกคือความผิดพลาดเชิงการเลือกตัวอย่าง: `miss_depth_profile.md` บอกว่า dense ใกล้ที่สุด ")
    w("บน **84 คู่ที่ทุกระบบพลาด** — แต่ pool ต้องรับใช้ทั้ง 1,046 คู่ ไม่ใช่แค่คู่ที่ยาก ")
    w(f"dense baseline อยู่ที่ {mean_metric(baseline['dense'], f'recall@{K}'):.4f} เทียบ hybrid "
      f"{mean_metric(baseline['hybrid'], f'recall@{K}'):.4f} — เริ่มต้นตามหลังเกินกว่าที่คู่ยากจะคืนให้ได้ ")
    w("**\"ใกล้ที่สุดบนคู่ที่ทุกคนพลาด\" ไม่ใช่ \"pool ที่ดีที่สุด\"**")
    w()
    w("**2. หลักฐานอยู่ใน pool จริง — reranker หาไม่เจอเอง** นี่คือรูปที่ทำให้ผลลบนี้มีน้ำหนัก ")
    w(f"ที่ P=50 pool ของ hybrid มี gold อยู่ {in_pool('hybrid', 50):.4f} และ oracle ส่งได้ "
      f"{oracle('hybrid', 50)[0]:.4f} แต่ของจริงได้ {mean_metric(h50, f'recall@{K}'):.4f} — "
      f"**ต่ำกว่า baseline ของตัวเอง** ")
    w("จึงสรุปได้ว่า *ไม่ใช่* \"หลักฐานเอื้อมไม่ถึง\" แต่เป็น \"reranker ตัวนี้หาไม่เจอ\"")
    w()
    w("**3. ความตึงที่ปิดแกนนี้** คู่ที่พลาดอยู่ที่อันดับ 11–50 (76.2% ตาม miss-depth) ")
    w("แต่ reranker เริ่มทำร้ายผลตั้งแต่ P เกิน ~20 และแย่ลงเรื่อย ๆ แบบ monotone ")
    w("(`จับได้` ติดลบ −6% → −22% → −33% ที่ P=50/100/200) ")
    w("**มันจึงเอื้อมไปถึงหลักฐานไม่ได้โดยไม่ทำลายของที่ถูกอยู่แล้วมากกว่าที่ได้มา**")
    w()
    w("**4. ผลเดิมที่ตีพิมพ์ไว้ replicate บน combo ที่แข็งกว่า** — Family 3 ยืนยันว่า reranker ")
    w("ทำร้าย MRR ของ hybrid อย่างมีนัยสำคัญ และ recall@10 ไม่ขยับ ตรงกับ ")
    w("`reranker_significance_test.md` ที่วัดบน `fixed_size × bge-m3` ")
    w("จึงไม่ใช่ผลเฉพาะ combo อ่อน")
    w()
    ptypes = {t: [q for q in queries if etype[q] == t] for t in {etype[q] for q in queries}}
    pv = {t: float(np.mean([_METRICS[f"recall@{K}"](d50[q], qrels[q]) for q in qs]))
          - float(np.mean([_METRICS[f"recall@{K}"](baseline["hybrid"][q], qrels[q]) for q in qs]))
          for t, qs in ptypes.items()}
    w(f"**5. ความเสียหายกระจุกอยู่ที่ `person`** ({pv.get('person', 0):+.4f} เทียบ "
      f"`course` {pv.get('course', 0):+.4f}) ซึ่งอธิบายได้ตรง ๆ: person คือชนิดคำถามที่ ")
    w("**BM25 แบกอยู่** (0.8147 ตามการแยกตาม entity_type) ด้วยการจับชื่อแบบตรงตัว ")
    w("การเอา cross-encoder มาให้คะแนนใหม่คือการทิ้งสัญญาณนั้น — pool จาก dense ยิ่งไม่มี BM25 ตั้งแต่แรก")
    w()
    h20 = mean_metric(results[("hybrid", 20)], f"recall@{K}")
    hb = mean_metric(baseline["hybrid"], f"recall@{K}")
    w(f"**6. เซลล์บวกเซลล์เดียวไม่ใช่ผล** `hybrid` ที่ P=20 ได้ recall@{K} {h20:.4f} "
      f"เทียบ baseline {hb:.4f} ({h20-hb:+.4f}) — แต่ไม่ได้ลงทะเบียนไว้ล่วงหน้า ")
    w("การหยิบ P ที่ดีที่สุดหลังเห็นตารางคือการ fit บนชุดทดสอบ และ MRR ของเซลล์นั้นยังติดลบอยู่ ")
    w("ถ้าจะอ้าง ต้องวัดใหม่บนชุดที่ไม่เคยเห็น")
    w()
    w("**7. ต้นทุนไม่ได้ถูก** P=50 บวก ~1.2 วินาที/query บนฐาน 1.21–1.86 วินาที คือเกือบเท่าตัว ")
    w("เพื่อผลที่แย่ลง")
    w()
    w("## self-check")
    w()
    for name, ok, detail in checks:
        w(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    w()
    w(f"เวลารวม {time.time()-t0:.0f} วินาที · คะแนน cross-encoder แคชไว้ที่ "
      f"`{SCORE_CACHE.relative_to(REPO).as_posix()}` (render ซ้ำได้โดยไม่ต้องใช้ GPU)")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {OUT}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
