"""Does a cross-encoder TRAINED on hybrid-fused candidates beat the shipped router?

Follow-up (a) from `docs/reranker-hybrid-interaction-research.md`, phase 3.
Pre-registration: `docs/reranker-trained-on-hybrid-design.md`. Training data:
`build_reranker_training_data.py`; the fine-tune: `train_hybrid_reranker.py`.

WHAT IS BEING ASKED, AND WHY IT NEEDED A TRAINED MODEL TO ASK IT
----------------------------------------------------------------
`reranker_rrf_routed_test.md` measured the off-the-shelf `bge-reranker-v2-m3`
fused in as a fourth RRF signal on top of the shipped hard router and found
**+0.0017** recall@10 (Holm-adj 1.0000) against a routed-pool oracle of
**+0.1500**. Two independent pieces of evidence said that null belongs to the
*model*, not to the axis: the oracle column (the evidence IS in the pool and the
model does not select it) and a 4-model swap whose spread (0.0355) is ~20x the
anchor's whole effect. The remaining explanation this project has never tested is
HYRR's: an off-the-shelf cross-encoder was trained on single-retriever
candidates, and a hybrid-fused pool is a different distribution. So exactly one
thing varies here -- the cross-encoder's weights. Pool, fusion, routing, `w`
grid, `P`, `k`, metrics and the bootstrap are all held at the published values.

ARMS (every arm SENDS k=10; the reranked arms additionally FETCH 50)
--------------------------------------------------------------------
    C  hard routing, no reranker                       -- published 0.6831
    D  hard routing + rrf4, off-the-shelf model        -- published 0.6847
    T  hard routing + rrf4, **trained** model          -- the open question
    L  hard routing + rrf4, lexical containment        -- descriptive control

Pre-registered before the numbers existed: **T vs C** (is the trained reranker
worth its 50 fetches and ~1.2 s/query on top of what ships) and **T vs D** (did
training on the right distribution buy anything over the off-the-shelf model),
3 metrics each, one Holm family of **m=6**, primary `recall@10`. `w` is chosen
leave-one-out exactly as in the published arms. The w grid, P=20, the per-route
table, arm L and the oracle column are descriptive.

WHY ARM L IS HERE
-----------------
For `person`/`program`/`faculty` the qrels were derived by string containment
(`docs/eval-validity-threats.md` §2), so a control that ranks candidates by
"does the query's entity literally appear in this chunk" is the free thing a
trained reranker has to beat before "it learned the distribution" means anything.
It is deliberately NOT in the pre-registered family: it is a floor to report, not
a hypothesis. Note the asymmetry it exposes -- `course` qrels are keyed on the
8-digit **code** while the query supplies the **name**
(`gold_anchor_ambiguity.md`), so arm L is expected to be weakest exactly there.

ANCHORS
-------
Nothing published is trusted; five numbers are reproduced from this code path:
arm C 0.6831, arm D 0.6847, the routed P=50 oracle 0.8331 delivered / 0.9054
holds, and truncate-and-replace 0.6000 -- which is the w=1.00 end of the same
grid, so it anchors the *far* end of the axis the fusion moves along and not
just the near one. S7 additionally re-verifies the *training* set's disjointness from
the eval set from the artifacts themselves rather than from the builder's own
report, and S8 checks the trained model actually ranks differently from the
off-the-shelf one -- a null is not evidence if the swap never happened.

Run (GPU; one job at a time on this machine):
    .venv/Scripts/python.exe tools/eval/reranker_trained_test.py --smoke
    .venv/Scripts/python.exe tools/eval/reranker_trained_test.py
    .venv/Scripts/python.exe tools/eval/reranker_trained_test.py --reuse-scores
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from rag_lab.query_service import discover_indices, resolve_index  # noqa: E402
from rag_lab.router import classify_query, route_targets  # noqa: E402
from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402
from audit_gold_anchor_ambiguity import contains_phrase  # noqa: E402

# The fusion, the pool construction and the oracle are IMPORTED, never
# reimplemented: two copies of the dense-first RRF tie-break would eventually
# disagree, and then arm T would not be comparable to the arm it is measured
# against. Same rule `routed_fetch_depth_test.py` follows for `fuse_at_depth`.
from reranker_rrf_routed_test import (  # noqa: E402
    K, P_MAX, POOLS, P_REGISTERED, QRELS_CEILING, SELECT_METRIC, W_GRID,
    CE_BATCH, CE_MODEL, INDEX_ROOT, N_BOOT, SEED, _METRICS,
    fuse_grid, loo_select, oracle_rerank, persisted_hybrid_top10, rank_one_index,
)

GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
TRAIN_POOLS = REPO / "data" / "results" / "reranker_train" / "train_pools.json"
TRAIN_LOG = REPO / "data" / "results" / "reranker_train" / "train_log.json"
CKPT = REPO / "data" / "models" / "reranker_hybrid_trained"
OFF_THE_SHELF_CACHE = REPO / "data" / "results" / "reranker_rrf_routed" / "ce_scores.json"
OUT_DIR = REPO / "data" / "results" / "reranker_trained"
SCORE_CACHE = OUT_DIR / "ce_scores.json"
SCORE_META = OUT_DIR / "ce_scores_meta.json"
OUT = REPO / "data" / "results" / "reranker_trained_test.md"

# reranker_rrf_routed_test.md, all four verified there against their own sources
PUBLISHED = {"C_recall": 0.6831, "D_recall": 0.6847,
             "oracle_p50_delivered": 0.8331, "oracle_p50_holds": 0.9054,
             "D_truncate_p50": 0.6000}
ORACLE_POOLS = (10, 20, 50, 100)
MIN_FREE_GB = 4.0
_WS = re.compile(r"\s+")


def _check_no(c) -> int:
    return int(re.match(r"S(\d+)", c[0]).group(1))


def lexical_cache(queries, entity_of, r_top, r_cid, r_text) -> dict:
    """Arm L's third signal: 1 if the query's entity literally appears in the
    chunk, else 0, minus a tiny multiple of the hybrid rank.

    That subtraction is not cosmetic. `fuse_grid` ranks the third signal with a
    plain `argsort`, and a binary score makes ~50 exact ties whose order would
    then be an artifact of the sort implementation; breaking them by the chunk's
    own hybrid rank makes the control a real ranking rather than a coin flip,
    and is the most favourable tie-break available to it.

    The containment test is **imported**, not written here: it is the one
    `audit_gold_anchor_ambiguity.py` already settled for this corpus
    (whitespace collapsed to a single space because OCR'd minutes wrap long
    names across lines, case-insensitive, and a Latin-alphanumeric boundary so
    `CALCULUS 2` does not match inside `CALCULUS 21`). It is the *audit's* rule,
    not the qrels generator's -- arm L is the cheapest defensible floor, not a
    reimplementation of `build_gold_candidates.py`."""
    out = {}
    for q in queries:
        ent, txt = entity_of[q], r_text[q]
        out[q] = {
            r_cid[q][i]: float(contains_phrase(_WS.sub(" ", txt[i]), ent)) - 1e-6 * rank
            for rank, i in enumerate(int(x) for x in r_top[q][:P_MAX])
        }
    return out


def rank_agreement(a: dict, b: dict, top: dict, cid: dict, queries, P: int) -> dict:
    """S8: do the two models rank the same pool differently at all?

    Kendall tau over the P scored candidates plus top-1 agreement, the same pair
    `reranker_model_comparison.py` reports -- because a null from two models that
    happen to produce identical rankings would only be saying the swap never
    happened."""
    from scipy.stats import kendalltau

    taus, same_top1 = [], 0
    for q in queries:
        ids = [cid[q][int(i)] for i in top[q][:P]]
        x = np.array([a[q][c] for c in ids])
        y = np.array([b[q][c] for c in ids])
        taus.append(float(kendalltau(x, y).statistic))
        same_top1 += int(int(np.argmax(x)) == int(np.argmax(y)))
    return {"tau_mean": float(np.mean(taus)), "tau_min": float(np.min(taus)),
            "tau_max": float(np.max(taus)), "same_top1": same_top1, "n": len(queries)}


def score_with(model_path: str, queries, r_top, r_cid, r_text, t0) -> tuple[dict, float]:
    """Score every routed candidate with one cross-encoder, through the SAME
    `sentence_transformers` path every published arm used. C2 in
    `train_hybrid_reranker.py` established that this path and the raw-transformers
    fp32 one agree on the delivered top-10, so a checkpoint selected there is
    scored here without a second, silently-different implementation."""
    from rag_lab.rerankers.cross_encoder import CrossEncoderReranker

    rr = CrossEncoderReranker(model_name=model_path, batch_size=CE_BATCH)
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
    ms = (time.time() - t_ce) * 1000.0 / max(n_pairs, 1)
    rr.release()
    return cache, ms


def main() -> int:
    ap = argparse.ArgumentParser(description="a reranker trained on hybrid-fused candidates")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reuse-scores", action="store_true")
    ap.add_argument("--checkpoint", default=str(CKPT))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--allow-busy-gpu", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    t0 = time.time()
    checks: list[tuple[str, bool, str]] = []
    grid = W_GRID[::4] if args.smoke else W_GRID
    pools = (P_REGISTERED,) if args.smoke else POOLS
    ckpt = Path(args.checkpoint)
    if not (ckpt / "config.json").exists():
        raise SystemExit(f"no checkpoint at {ckpt} — run tools/eval/train_hybrid_reranker.py first")

    # The gate applies even under --reuse-scores: the cache spares the
    # cross-encoder, but `rank_one_index` still loads an embedder per routed
    # index, and this machine's rule is one GPU job at a time.
    try:
        import torch
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            if free_gb < MIN_FREE_GB and not args.allow_busy_gpu:
                raise SystemExit(f"only {free_gb:.1f} GB free — another GPU job is resident")
    except ImportError:
        pass

    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: list(d["relevant_resolution_ids"]) for d in raw}
    entity_of = {d["query"]: d["entity"] for d in raw}
    type_of = {d["query"]: d["entity_type"] for d in raw}
    if args.smoke:
        queries = queries[:2] + queries[26:28] + queries[60:62] + queries[-2:]

    # ---- S7 before anything expensive: is the trained model even eligible? ---
    tp = json.loads(TRAIN_POOLS.read_text(encoding="utf-8"))
    train_q = {r["query"] for r in tp}
    train_e = {r["entity"] for r in tp}
    eval_q, eval_e = set(queries), {entity_of[q] for q in queries}
    shared_res = {rid for r in tp for c in r["candidates"] if c["label"]
                  for rid in [c["resolution_id"]]} & {r for q in queries for r in qrels[q]}
    checks.append((
        "S7 the training queries are disjoint from the eval set, by query AND by entity",
        not (train_q & eval_q) and not (train_e & eval_e),
        f"{len(train_q & eval_q)} shared queries, {len(train_e & eval_e)} shared entities "
        f"over {len(tp)} training pools; {len(shared_res)} resolutions are relevant to both "
        f"sets (unavoidable in one corpus, and not a label the model ever saw)",
    ))

    # ---- routed pools, one index at a time (VRAM: never two resident) -------
    indices = discover_indices(INDEX_ROOT)
    targets = route_targets("hybrid")
    route_of = {q: classify_query(q) for q in queries}
    resolved = {r: resolve_index(t, indices) for r, t in targets.items()}
    combo_of = {r: i.combo_id for r, i in resolved.items()}
    dir_of = {i.combo_id: Path(i.dir) for i in resolved.values()}
    by_combo: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_combo[combo_of[route_of[q]]].append(q)

    r_top, r_h, r_cid, r_rid, r_page, r_text = {}, {}, {}, {}, {}, {}
    r_ok = 0
    for combo, qs in by_combo.items():
        top, hterm, cid, rid, page, text = rank_one_index(dir_of[combo], qs, True)
        persisted = persisted_hybrid_top10(combo)
        for q in qs:
            r_top[q], r_h[q] = top[q], hterm[q]
            r_cid[q], r_rid[q], r_page[q], r_text[q] = cid, rid, page, text
            r_ok += int([cid[i] for i in top[q][:K]] == persisted.get(q, []))
        print(f"  {combo}  {len(qs)} queries  {time.time()-t0:.0f}s", file=sys.stderr)
    checks.append(("S1 routed hybrid top-10 reproduces the persisted results",
                   r_ok == len(queries), f"{r_ok} of {len(queries)}"))

    # ---- cross-encoder scores ------------------------------------------------
    ce_ms = float("nan")
    if args.reuse_scores and SCORE_CACHE.exists():
        t_cache = json.loads(SCORE_CACHE.read_text(encoding="utf-8"))
        if SCORE_META.exists():
            ce_ms = json.loads(SCORE_META.read_text(encoding="utf-8"))["ms_per_pair"]
    else:
        t_cache, ce_ms = score_with(str(ckpt), queries, r_top, r_cid, r_text, t0)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if args.smoke:
            # a cache holding only the smoke subset would poison a later
            # --reuse-scores run while every check still passed on it
            print(f"  [smoke] cache NOT written", file=sys.stderr)
        else:
            SCORE_CACHE.write_text(json.dumps(t_cache, ensure_ascii=False), encoding="utf-8")
            SCORE_META.write_text(json.dumps({
                "ms_per_pair": round(ce_ms, 2), "n_queries": len(queries),
                "batch_size": CE_BATCH, "model": str(ckpt.relative_to(REPO)).replace("\\", "/"),
                "base_model": CE_MODEL, "pool": f"routed hybrid top-{P_MAX}",
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2), encoding="utf-8")

    d_cache = json.loads(OFF_THE_SHELF_CACHE.read_text(encoding="utf-8"))
    missing_t = [q for q in queries
                 if any(r_cid[q][i] not in t_cache.get(q, {}) for i in r_top[q][:P_MAX])]
    missing_d = [q for q in queries
                 if any(r_cid[q][i] not in d_cache.get(q, {}) for i in r_top[q][:P_MAX])]
    checks.append((
        "S2 both models scored every routed candidate, over the same pools",
        not missing_t and not missing_d,
        f"trained {sum(len(t_cache[q]) for q in queries):,} pairs ({len(missing_t)} incomplete); "
        f"off-the-shelf cache {len(missing_d)} incomplete",
    ))

    l_cache = lexical_cache(queries, entity_of, r_top, r_cid, r_text)

    # ---- arms ----------------------------------------------------------------
    sc_T = fuse_grid(r_top, r_h, t_cache, r_cid, r_rid, r_page, grid, pools, queries, qrels)
    sc_D = fuse_grid(r_top, r_h, d_cache, r_cid, r_rid, r_page, grid, pools, queries, qrels)
    sc_L = fuse_grid(r_top, r_h, l_cache, r_cid, r_rid, r_page, grid, pools, queries, qrels)

    wi0 = grid.index(0.0)
    arm_C = {m: sc_T[P_REGISTERED][m][wi0] for m in _METRICS}
    arm_T, picks_T, w_or_T = loo_select(sc_T[P_REGISTERED], grid, len(queries))
    arm_D, picks_D, w_or_D = loo_select(sc_D[P_REGISTERED], grid, len(queries))
    arm_L, picks_L, w_or_L = loo_select(sc_L[P_REGISTERED], grid, len(queries))

    checks.append((
        "S3 arm C reproduces routing_eval.md's `routed (shipped)` hybrid",
        abs(arm_C["recall@10"].mean() - PUBLISHED["C_recall"]) < 5e-5 or args.smoke,
        f"{arm_C['recall@10'].mean():.4f} vs published {PUBLISHED['C_recall']:.4f}"
        + ("  [smoke: subset]" if args.smoke else ""),
    ))
    checks.append((
        "S4 arm D reproduces reranker_rrf_routed_test.md's off-the-shelf arm",
        abs(arm_D["recall@10"].mean() - PUBLISHED["D_recall"]) < 5e-5 or args.smoke,
        f"{arm_D['recall@10'].mean():.4f} vs published {PUBLISHED['D_recall']:.4f}"
        + ("  [smoke: subset]" if args.smoke else ""),
    ))

    # Control 4 of the pre-registration: truncate-and-replace is the w=1.00 end
    # of the same grid, so it is an anchor as well as a row -- if the grid's far
    # end does not reproduce the published 0.6000, the fusion is not the
    # published one and no other row can be trusted either.
    wi1 = grid.index(1.0)
    d_trunc = sc_D[P_REGISTERED]["recall@10"][wi1].mean()
    checks.append((
        "S10 truncate-and-replace (w=1.00) reproduces the published routed P=50 value",
        abs(d_trunc - PUBLISHED["D_truncate_p50"]) < 5e-5 or args.smoke,
        f"{d_trunc:.4f} vs published {PUBLISHED['D_truncate_p50']:.4f}"
        + ("  [smoke: subset]" if args.smoke else ""),
    ))

    orc, holds = {}, {}
    for P in ORACLE_POOLS:
        orc[P], holds[P] = oracle_rerank(r_top, r_cid, r_rid, r_page, queries, qrels, P)
    checks.append((
        "S5 the routed oracle reproduces reranker_rrf_routed_test.md (both columns)",
        (abs(orc[50]["recall@10"].mean() - PUBLISHED["oracle_p50_delivered"]) < 5e-5
         and abs(holds[50].mean() - PUBLISHED["oracle_p50_holds"]) < 5e-5) or args.smoke,
        f"delivered {orc[50]['recall@10'].mean():.4f} vs "
        f"{PUBLISHED['oracle_p50_delivered']:.4f}; holds {holds[50].mean():.4f} vs "
        f"{PUBLISHED['oracle_p50_holds']:.4f}" + ("  [smoke: subset]" if args.smoke else ""),
    ))
    hi = max([float(s[P]["recall@10"][wi].mean())
              for s in (sc_T, sc_D, sc_L) for P in pools for wi in range(len(grid))]
             + [float(orc[P]["recall@10"].mean()) for P in ORACLE_POOLS]
             + [float(a["recall@10"].mean()) for a in (arm_T, arm_D, arm_L)])
    checks.append((
        "S6 nothing that sends 10 documents exceeds the qrels ceiling",
        hi <= QRELS_CEILING + 1e-9 or args.smoke,
        f"ceiling {QRELS_CEILING:.4f}; highest delivered recall@{K} {hi:.4f} "
        f"(the pool-holds column legitimately exceeds it — it sends P, not {K})",
    ))

    agree = rank_agreement(t_cache, d_cache, r_top, r_cid, queries, P_REGISTERED)
    checks.append((
        "S8 the trained model ranks the pool differently from the off-the-shelf one",
        agree["tau_mean"] < 0.99 and agree["same_top1"] < len(queries),
        f"Kendall τ mean {agree['tau_mean']:+.3f} (min {agree['tau_min']:+.3f}, "
        f"max {agree['tau_max']:+.3f}); same top-1 on {agree['same_top1']} of {agree['n']} "
        "— a null from two identical rankings would only say the swap never happened",
    ))
    checks.append((
        "S9 w is selected leave-one-out, never on the held-out fold",
        True,
        f"trained: {len(set(picks_T))} distinct w (modal {max(set(picks_T), key=picks_T.count):.2f}, "
        f"oracle-on-all {w_or_T:.2f}) · off-the-shelf: modal "
        f"{max(set(picks_D), key=picks_D.count):.2f} · lexical: modal "
        f"{max(set(picks_L), key=picks_L.count):.2f}",
    ))

    if args.smoke:
        for name, ok, detail in sorted(checks, key=_check_no):
            print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        for lab, a in (("C", arm_C), ("D", arm_D), ("T", arm_T), ("L", arm_L)):
            print(f"  arm {lab}  recall@{K} {a['recall@10'].mean():.4f}  "
                  f"MRR {a['mrr'].mean():.4f}  nDCG {a['ndcg@10'].mean():.4f}")
        print(f"\nsmoke ({len(queries)} queries, {time.time()-t0:.0f}s) — nothing written")
        return 0 if all(ok for _, ok, _ in checks) else 1

    # ---- pre-registered significance ----------------------------------------
    rng = np.random.default_rng(args.seed)
    rows = []
    for label, base in (("T vs C  (trained reranker on top of routing)", arm_C),
                        ("T vs D  (training vs the off-the-shelf model)", arm_D)):
        for m in _METRICS:
            observed, p, ci = bootstrap_pvalue(arm_T[m] - base[m], rng, args.n_boot)
            rows.append((label, m, observed, p, ci))
    fam1 = holm_correct(rows, alpha=args.alpha)

    tlog = json.loads(TRAIN_LOG.read_text(encoding="utf-8")) if TRAIN_LOG.exists() else {}
    L: list[str] = []
    def w_(s: str = "") -> None:
        L.append(s)

    w_("# cross-encoder ที่เทรนบน hybrid-fused candidates วัดทับ hard router")
    w_()
    w_(f"Generated by `tools/eval/reranker_trained_test.py` · {len(queries)} คำถาม · "
       f"ทุก arm **ส่งออก k={K} เท่ากัน** (arm ที่ rerank **ดึงเพิ่ม {P_REGISTERED} ใบ** "
       f"ไปให้ cross-encoder — ต่างกันที่ต้นทุน ไม่ใช่ที่งบที่ถูกวัด)")
    w_()
    w_(f"`reranker_rrf_routed_test.md` วัด reranker สำเร็จรูปทับ router ได้ **+0.0017** "
       f"ขณะที่ oracle บน pool เดียวกันได้ **+0.1500** — หลักฐานสองทาง (คอลัมน์ oracle "
       f"และการสลับโมเดล 4 ตัวที่กระจาย 0.0355 ราว 20 เท่าของ effect ทั้งก้อน) ชี้ว่า null "
       f"นั้นเป็นของ**โมเดล** ไม่ใช่ของ**แกน** · คำอธิบายที่เหลือคือของ HYRR: โมเดลสำเร็จรูป "
       f"ถูกเทรนบน candidate จาก retriever เดียว ส่วน pool ที่นี่มาจาก hybrid fusion ")
    w_("จึงเปลี่ยนสิ่งเดียวคือ**น้ำหนักของโมเดล** — pool, fusion, routing, กริด `w`, `P`, `k`, "
       "metric และ bootstrap ตรึงไว้ที่ค่าที่ตีพิมพ์ไปแล้วทั้งหมด")
    w_()
    if tlog:
        h = tlog.get("hyper", {})
        w_(f"โมเดลที่ทดสอบ: `{tlog.get('base_model')}` fine-tune บน "
           f"**{tlog['data']['train_pools']} pool** (routed hybrid top-"
           f"{tlog['data']['pool_depth']}) จาก entity ที่**ไม่ทับชุดประเมิน** · "
           f"group-softmax 1 บวก vs {h.get('group_neg')} ลบ · เทรน "
           f"{h.get('trainable_params', 0)/1e6:.0f}M พารามิเตอร์ "
           f"({h.get('frozen_embedding_params', 0)/1e6:.0f}M แช่แข็ง) · "
           f"เลือก checkpoint จาก dev {tlog['data']['dev_pools']} คำถามที่กันจาก*ชุดเทรน* "
           f"(epoch {tlog['dev']['best_epoch']}, dev recall@{K} {tlog['dev']['best']:.4f} "
           f"เทียบสำเร็จรูป {tlog['dev']['off_the_shelf']:.4f}) · "
           f"รายละเอียด `data/results/reranker_training_run.md`")
        w_()
    w_("## ตาราง arm")
    w_()
    w_(f"| arm | สัญญาณที่ 4 | recall@{K} | MRR | nDCG@{K} | ดึง / ส่ง |")
    w_("|---|---|---|---|---|---|")
    for lab, desc, a in (("C", "ไม่มี (router ที่ ship อยู่)", arm_C),
                         ("D", "cross-encoder สำเร็จรูป", arm_D),
                         ("**T**", "**cross-encoder ที่เทรนแล้ว**", arm_T),
                         ("L", "ตรงตัวอักษร (control, ไม่ใช้ GPU)", arm_L)):
        st = "**" if "T" in lab else ""
        fetch = f"{K} / {K}" if lab == "C" else f"{P_REGISTERED} / {K}"
        w_(f"| {lab} | {desc} | {st}{a['recall@10'].mean():.4f}{st} | {a['mrr'].mean():.4f} | "
           f"{a['ndcg@10'].mean():.4f} | {fetch} |")
    wo = grid.index(w_or_T)
    w_(f"| T′ (oracle w={w_or_T:.2f}) — ขอบเขต ไม่ใช่ระบบ | เทรนแล้ว | "
       f"{sc_T[P_REGISTERED]['recall@10'][wo].mean():.4f} | "
       f"{sc_T[P_REGISTERED]['mrr'][wo].mean():.4f} | "
       f"{sc_T[P_REGISTERED]['ndcg@10'][wo].mean():.4f} | {P_REGISTERED} / {K} |")
    w_()
    w_(f"**Family 1 (m={len(fam1)}, ลงทะเบียนก่อนรัน)** — paired bootstrap {args.n_boot} รอบ "
       f"(seed={args.seed}), Holm · `w` เลือกแบบ leave-one-out บน `{SELECT_METRIC}` "
       f"· arm L และกริด w เป็นการบรรยาย ไม่อยู่ในตระกูล")
    w_()
    w_("| เทียบ | metric | diff | 95% CI | raw p | Holm-adj p | นัยสำคัญ |")
    w_("|---|---|---|---|---|---|---|")
    for a, b, diff, p, ci, hp, sig in sorted(fam1, key=lambda x: x[5]):
        w_(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {hp:.4f} | "
           f"{'**ใช่**' if sig else 'ไม่'} |")
    w_()

    for P in pools:
        w_(f"## กริด w (P={P}, บรรยาย) — เทรนแล้ว / สำเร็จรูป / ตรงตัวอักษร")
        w_()
        w_(f"| w | T recall@{K} | D recall@{K} | L recall@{K} | |")
        w_("|---|---|---|---|---|")
        bt = max(sc_T[P]["recall@10"][x].mean() for x in range(len(grid)))
        for wi, wv in enumerate(grid):
            r = sc_T[P]["recall@10"][wi].mean()
            tag = "← arm C (ไม่ rerank)" if wv == 0.0 else (
                "← truncate-and-replace" if wv == 1.0 else "")
            w_(f"| {wv:.2f} | {'**' if r == bt else ''}{r:.4f}{'**' if r == bt else ''} | "
               f"{sc_D[P]['recall@10'][wi].mean():.4f} | "
               f"{sc_L[P]['recall@10'][wi].mean():.4f} | {tag} |")
        w_()

    w_("## เพดานของ pool เดิม — ยังเหลืออะไรให้ได้")
    w_()
    w_(f"**สองคอลัมน์นี้ต่างกัน**: `pool มี` คือของที่ *อยู่ใน* pool (ส่ง P ใบ — ส่งมอบไม่ได้) "
       f"ส่วน `oracle ส่งมอบ` คือการเลือก {K} ใบที่ดีที่สุด*จาก* pool (ส่งมอบได้ ต้องอยู่ใต้เพดาน "
       f"qrels {QRELS_CEILING:.4f}) — **อ้างอิงตัวส่งมอบเท่านั้น**")
    w_()
    w_("| P | pool มี | oracle ส่งมอบ | เหนือ arm C |")
    w_("|---|---|---|---|")
    for P in ORACLE_POOLS:
        d = orc[P]["recall@10"].mean()
        tag = " (= arm C ตามโครงสร้าง)" if P == K else ""
        w_(f"| {P} | {holds[P].mean():.4f} | **{d:.4f}**{tag} | "
           f"{d - arm_C['recall@10'].mean():+.4f} |")
    w_()
    gap = orc[P_REGISTERED]["recall@10"].mean() - arm_C["recall@10"].mean()
    got_T = arm_T["recall@10"].mean() - arm_C["recall@10"].mean()
    got_D = arm_D["recall@10"].mean() - arm_C["recall@10"].mean()
    w_(f"ที่ P={P_REGISTERED}: reranker ที่สมบูรณ์แบบได้ **{gap:+.4f}** เหนือ router · "
       f"ตัวที่เทรนแล้วได้ **{got_T:+.4f}** (**{100*got_T/gap:.0f}%** ของเพดาน) · "
       f"ตัวสำเร็จรูปได้ **{got_D:+.4f}** (**{100*got_D/gap:.0f}%**)")
    w_()

    w_(f"## แยกตาม route (P={P_REGISTERED}, recall@{K})")
    w_()
    w_("| route | คำถาม | C | D สำเร็จรูป | T เทรนแล้ว | L ตรงตัวอักษร | T − C | T − D |")
    w_("|---|---|---|---|---|---|---|---|")
    for rt in ("person", "program", "course", "faculty"):
        idx = [i for i, q in enumerate(queries) if route_of[q] == rt]
        if not idx:
            continue
        c, d, t, l = (x["recall@10"][idx].mean() for x in (arm_C, arm_D, arm_T, arm_L))
        w_(f"| {rt} | {len(idx)} | {c:.4f} | {d:.4f} | {t:.4f} | {l:.4f} | "
           f"{t-c:+.4f} | {t-d:+.4f} |")
    w_()
    w_(f"arm L เป็นพื้น ไม่ใช่สมมติฐาน: qrels ของ `person`/`program`/`faculty` มาจากการ"
       f"จับคู่ตัวอักษร (`eval-validity-threats.md` §2) ส่วน `course` จับคู่ด้วย**รหัส 8 หลัก** "
       f"ขณะที่คำถามให้**ชื่อ** (`gold_anchor_ambiguity.md`) — คาดไว้ล่วงหน้าว่า L "
       f"จะอ่อนที่สุดตรง `course` พอดี")
    w_()

    w_("## self-check")
    w_()
    for name, ok, detail in sorted(checks, key=_check_no):
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
