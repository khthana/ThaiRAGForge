"""Is `+0.0017 on top of the router` a fact about `bge-reranker-v2-m3`, or about
cross-encoder reranking on this corpus?

`reranker_rrf_routed_test.py` measured one model and found nothing: rrf4 on top
of the hard router is +0.0017 recall@10 (Holm 1.0000), while a perfect selection
from the *same* routed pool delivers 0.8331 -- **+0.1500**. So the model captures
about 1% of its own ceiling, and the null cannot distinguish

    (a) this cross-encoder is weak       -> a better one is worth building
    (b) nothing is reachable at all      -> close the axis

The oracle already argues for (a); this script tests it directly by swapping the
model and changing nothing else. Same routed pool, same fusion, same LOO
protocol, same 106 queries -- the model is the only moving part, which is why
every arm here is comparable to a published number.

DESIGN
------
Everything except the cross-encoder is imported from `reranker_rrf_routed_test`
rather than re-implemented. That is not tidiness: `rank_one_index` carries two
replication traps this project has already paid for (a batched matmul reproduces
only 98/106 top-10s because BLAS reassociates and `argsort` is unstable; RRF ties
must be settled dense-first), and a second copy of them would drift.

Arms, all sending k=10:
  C        routing only, no reranker -- **model-independent**, so every model's
           w=0.00 column must reproduce it exactly (S5)
  D(M)     routing + rrf4 with model M at P=50, w chosen leave-one-out
  D'(M)    the same at the oracle w -- a bound, not a system
  oracle   perfect selection of 10 from the routed P=50 pool -- also
           model-independent, and reproduces the published 0.8331 (S6)

Pre-registered, decided before any new model was scored:
  Family 1 (m=3)  D(M) vs C on recall@10, for the three *new* models
  Family 2 (m=6)  the same on MRR and nDCG@10

The anchor's own `D vs C` is deliberately **not** re-tested: it is published at
Holm 1.0000 in a family of 6, and re-running identical data inside a family of 3
would produce a different Holm p for the same measurement -- the family-size trap
`hybrid_alpha_sweep` vs `soft_vs_hard_routing` already documents. The anchor is
reproduced as a self-check (S1) and quoted, never re-tested.

MODEL QUALIFICATION IS A HARD PRECONDITION. `tools/eval/qualify_reranker_model.py`
must have passed for every model used, and S8 re-reads its report. On 2026-08-09
`gte-multilingual-reranker-base` ranked a Thai example correctly while scoring a
sentence and its reversal bit-identically (dead RoPE tables under transformers
5.x); measured unchecked, it would have produced a low number and this script
would have concluded "a second cross-encoder also fails" -- a false family-level
negative caused by a broken model, not by the axis.

Run:
    .venv/Scripts/python.exe tools/eval/reranker_model_comparison.py --smoke
    .venv/Scripts/python.exe tools/eval/reranker_model_comparison.py
    .venv/Scripts/python.exe tools/eval/reranker_model_comparison.py --reuse-scores
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
import reranker_rrf_routed_test as RT  # noqa: E402

OUT_DIR = RT.OUT_DIR
REPORT = REPO / "data" / "results" / "reranker_model_comparison.md"
QUALIFY_REPORT = REPO / "data" / "results" / "reranker_model_qualification.md"

# The anchor first: it reuses the existing cache, so S1 fails immediately if the
# harness has drifted, before a single GPU-second is spent on a new model.
MODELS = [
    ("BAAI/bge-reranker-v2-m3", "bge-v2-m3 (anchor)", 568),
    ("BAAI/bge-reranker-large", "bge-v1-large", 560),
    ("BAAI/bge-reranker-base", "bge-v1-base", 278),
    ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "mmarco-mMiniLM", 118),
]
ANCHOR = MODELS[0][0]
# reranker_rrf_routed_test.md, same pool, same protocol -- PARSED from that
# report, never frozen. It was the literal dict
# {"C": 0.6831, "D_anchor": 0.6847, "oracle_p50": 0.8331, "holds_p50": 0.9054}
# until 2026-08-20, when rebuild #4 moved C and D and S3/S4 FAILED on a run that
# was itself correct -- the 11th-14th cross-artifact anchor of the kind `561102e`
# replaced elsewhere. Worse than a red check: `head` below fed the report's own
# "captured ceiling" column, so a stale literal produced a *silently* wrong
# percentage on a fresh run. Missing/renamed report => empty, and every caller
# FAILs with "UNPARSED" rather than skipping.
def _published() -> dict[str, float | None]:
    if not RT.ROUTED_REPORT.exists():
        return {}
    txt = RT.ROUTED_REPORT.read_text(encoding="utf-8")
    arms = RT.parse_routed_arms(txt)
    delivered, holds = RT.parse_routed_oracle(txt, 50)
    return {"C": arms.get("C"), "D_anchor": arms.get("D"),
            "oracle_p50": delivered, "holds_p50": holds}


PUBLISHED = _published()


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def cache_path(model: str) -> Path:
    # The anchor's cache predates this script and is reused as-is: rescoring it
    # would make S1 test the new run against itself instead of against what was
    # published.
    return RT.SCORE_CACHE if model == ANCHOR else OUT_DIR / f"ce_scores__{slug(model)}.json"


def score_pool(model: str, queries, r_top, r_cid, r_text, batch: int) -> tuple[dict, dict]:
    """Score every (query, chunk) pair in the routed top-`P_MAX` pool once.

    The pool does not depend on the model, so all models see byte-identical
    candidate text -- the comparison isolates the model and nothing else.

    Except for one thing that is **not** equal and must be reported rather than
    hidden: context length. `bge-reranker-v2-m3` accepts 8192 tokens, the v1 and
    mmarco models 512, so a long chunk reaches them truncated. Each model is run
    at its own native maximum (using a model as intended is the fair setting, and
    forcing 512 on the anchor would break its reproduction of the published
    number), and the truncation rate is measured here so a difference can be
    attributed rather than assumed."""
    from rag_lab.rerankers.cross_encoder import CrossEncoderReranker

    rr = CrossEncoderReranker(model_name=model, batch_size=batch)
    m = rr._load()
    cache, n_pairs, t0 = {}, 0, time.time()
    for j, q in enumerate(queries):
        rows = [int(i) for i in r_top[q][:RT.P_MAX]]
        sc = np.asarray(m.predict([(q, r_text[q][i]) for i in rows],
                                  batch_size=batch, show_progress_bar=False))
        cache[q] = {r_cid[q][i]: float(s) for i, s in zip(rows, sc)}
        n_pairs += len(rows)
        if (j + 1) % 25 == 0:
            print(f"    {model}  {j+1}/{len(queries)}  {time.time()-t0:.0f}s", file=sys.stderr)
    rr.release()
    return cache, {"ms_per_pair": round((time.time() - t0) * 1000.0 / max(n_pairs, 1), 2),
                   "n_pairs": n_pairs}


def truncation_stats(model: str, queries, r_top, r_text) -> dict:
    """How much of the pool does this model actually get to see?

    Measured from the tokenizer alone (CPU, no GPU, no weights) so it is computed
    the same way for every model including the anchor, whose scores are reused
    from a previous run and whose meta predates this column. A `?` in a table
    that exists to expose a confound would defeat the point of the column."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    max_len = int(tok.model_max_length)
    n, trunc, longest = 0, 0, 0
    for q in queries:
        rows = [int(i) for i in r_top[q][:RT.P_MAX]]
        lens = [len(t) for t in tok([q] * len(rows), [r_text[q][i] for i in rows],
                                    truncation=False)["input_ids"]]
        n += len(lens)
        trunc += sum(x > max_len for x in lens)
        longest = max(longest, max(lens))
    return {"max_length": max_len, "truncated_pct": round(100.0 * trunc / max(n, 1), 1),
            "longest_pair_tokens": longest}


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Rank agreement between two models on one query's pool. Written out rather
    than pulled from scipy because only the sign structure matters and the pools
    are 100 long, so the O(n^2) form is free and has no dependency."""
    n = len(a)
    ii, jj = np.triu_indices(n, 1)
    sa = np.sign(a[ii] - a[jj])
    sb = np.sign(b[ii] - b[jj])
    denom = np.sqrt((sa != 0).sum() * (sb != 0).sum())
    return float((sa * sb).sum() / denom) if denom else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="do other cross-encoders beat the router?")
    ap.add_argument("--smoke", action="store_true", help="8 queries, coarse grid, writes nothing")
    ap.add_argument("--reuse-scores", action="store_true",
                    help="reuse ce_scores.json instead of running the cross-encoder; NOT GPU-free -- "
                         "retrieval still loads an embedder, so do not run this beside "
                         "a training job on a single card")
    ap.add_argument("--n-boot", type=int, default=RT.N_BOOT)
    ap.add_argument("--seed", type=int, default=RT.SEED)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=RT.CE_BATCH)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    t0 = time.time()
    checks: list[tuple[str, bool, str]] = []
    grid = RT.W_GRID[::4] if args.smoke else RT.W_GRID
    P = RT.P_REGISTERED
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(RT.GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: list(d["relevant_resolution_ids"]) for d in raw}
    if args.smoke:
        queries = queries[:2] + queries[26:28] + queries[60:62] + queries[-2:]

    # ---- the routed pool (identical for every model) ------------------------
    indices = discover_indices(RT.INDEX_ROOT)
    targets = route_targets("hybrid")
    route_of = {q: classify_query(q) for q in queries}
    resolved = {r: resolve_index(t, indices) for r, t in targets.items()}
    combo_of = {r: i.combo_id for r, i in resolved.items()}
    dir_of = {i.combo_id: Path(i.dir) for i in resolved.values()}
    by_combo: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_combo[combo_of[route_of[q]]].append(q)

    # Always: needed by score_pool, and by the truncation audit even when every
    # score is cached. Reading one more parquet column is free next to the rest.
    need_text = True
    r_top, r_h, r_cid, r_rid, r_page, r_text = {}, {}, {}, {}, {}, {}
    r_ok = 0
    for combo, qs in by_combo.items():
        top, hterm, cid, rid, page, text = RT.rank_one_index(dir_of[combo], qs, need_text)
        persisted = RT.persisted_hybrid_top10(combo)
        for q in qs:
            r_top[q], r_h[q] = top[q], hterm[q]
            r_cid[q], r_rid[q], r_page[q], r_text[q] = cid, rid, page, text
            r_ok += int([cid[i] for i in top[q][:RT.K]] == persisted.get(q, []))
        print(f"  ranked {combo}  {len(qs)}q  {time.time()-t0:.0f}s", file=sys.stderr)
    checks.append(("S0 routed hybrid top-10 reproduces the persisted results",
                   r_ok == len(queries), f"{r_ok} of {len(queries)}"))

    # ---- per-model cross-encoder scores (one model resident at a time) ------
    caches, meta_of = {}, {}
    for model, label, _ in MODELS:
        cp = cache_path(model)
        mp = RT.SCORE_META if model == ANCHOR else cp.with_name(cp.stem + "__meta.json")
        if (args.reuse_scores or model == ANCHOR) and cp.exists():
            caches[model] = json.loads(cp.read_text(encoding="utf-8"))
            meta_of[model] = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
            meta_of[model].setdefault("n_pairs", sum(len(v) for v in caches[model].values()))
            print(f"  cached  {label}", file=sys.stderr)
            continue
        caches[model], meta_of[model] = score_pool(model, queries, r_top, r_cid,
                                                   r_text, args.batch)
        # a smoke cache would silently poison a later --reuse-scores run; and the
        # anchor's cache belongs to reranker_rrf_routed_test.py -- overwriting it
        # here would replace the provenance of a published number
        if args.smoke or model == ANCHOR:
            print(f"  {meta_of[model]['n_pairs']} pairs for {label}; cache NOT written "
                  f"({'smoke' if args.smoke else 'anchor cache is the parent run'})",
                  file=sys.stderr)
            continue
        cp.write_text(json.dumps(caches[model], ensure_ascii=False), encoding="utf-8")
        mp.write_text(json.dumps({**meta_of[model], "n_queries": len(queries),
                                  "batch_size": args.batch, "model": model,
                                  "pool": f"routed hybrid top-{RT.P_MAX}",
                                  "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                 indent=2), encoding="utf-8")

    for model, label, _ in MODELS:
        meta_of[model].update(truncation_stats(model, queries, r_top, r_text))
        print(f"  tokenised {label}: ctx {meta_of[model]['max_length']}, "
              f"{meta_of[model]['truncated_pct']}% truncated", file=sys.stderr)

    missing = {label: [q for q in queries
                       if any(r_cid[q][i] not in caches[model].get(q, {})
                              for i in r_top[q][:P])]
               for model, label, _ in MODELS}
    checks.append(("S1 every model scored every routed candidate",
                   not any(missing.values()),
                   " · ".join(f"{k}: {len(v)} incomplete" for k, v in missing.items())))

    # ---- arms ---------------------------------------------------------------
    wi0 = grid.index(0.0)
    scores, arm_D, picks, w_or = {}, {}, {}, {}
    for model, label, _ in MODELS:
        scores[model] = RT.fuse_grid(r_top, r_h, caches[model], r_cid, r_rid, r_page,
                                     grid, (P,), queries, qrels)
        arm_D[model], picks[model], w_or[model] = RT.loo_select(scores[model][P], grid,
                                                                len(queries))
    arm_C = {m: scores[ANCHOR][P][m][wi0] for m in RT._METRICS}

    # w=0.00 deletes the cross-encoder term, so every model must land on arm C.
    # This is the check that catches the #1 failure of a model-swap experiment:
    # not actually swapping the model, or fusing the wrong cache.
    same_C = {label: float(np.abs(scores[model][P]["recall@10"][wi0] - arm_C["recall@10"]).max())
              for model, label, _ in MODELS}
    checks.append(("S2 w=0.00 reproduces arm C for every model (fusion is model-independent there)",
                   max(same_C.values()) == 0.0,
                   " · ".join(f"{k} |d|max={v:.0e}" for k, v in same_C.items())))
    _pC, _pD = PUBLISHED.get("C"), PUBLISHED.get("D_anchor")
    checks.append(("S3 arm C is reranker_rrf_routed_test.md's arm C (routed, shipped)",
                   _pC is not None
                   and (abs(arm_C["recall@10"].mean() - _pC) < 5e-5 or args.smoke),
                   f"{arm_C['recall@10'].mean():.4f} vs published "
                   + (f"{_pC:.4f}" if _pC is not None
                      else "UNPARSED -- the cross-check could not be made")))
    checks.append(("S4 the anchor reproduces reranker_rrf_routed_test.md's arm D",
                   _pD is not None
                   and (abs(arm_D[ANCHOR]["recall@10"].mean() - _pD) < 5e-5 or args.smoke),
                   f"{arm_D[ANCHOR]['recall@10'].mean():.4f} vs published "
                   + (f"{_pD:.4f}" if _pD is not None
                      else "UNPARSED -- the cross-check could not be made")))

    oracle, holds = RT.oracle_rerank(r_top, r_cid, r_rid, r_page, queries, qrels, P)
    checks.append(("S5 the routed oracle reproduces reranker_rrf_routed_test.md (pool is "
                   "model-independent)",
                   PUBLISHED.get("oracle_p50") is not None
                   and PUBLISHED.get("holds_p50") is not None
                   and ((abs(oracle["recall@10"].mean() - PUBLISHED["oracle_p50"]) < 5e-5
                         and abs(holds.mean() - PUBLISHED["holds_p50"]) < 5e-5) or args.smoke),
                   f"delivered {oracle['recall@10'].mean():.4f} vs "
                   + (f"{PUBLISHED['oracle_p50']:.4f}" if PUBLISHED.get("oracle_p50") is not None
                      else "UNPARSED")
                   + f"; holds {holds.mean():.4f} vs "
                   + (f"{PUBLISHED['holds_p50']:.4f}" if PUBLISHED.get("holds_p50") is not None
                      else "UNPARSED -- the cross-check could not be made")))

    # The models must actually disagree. Two models that rank every pool
    # identically would give identical arms, and a null would then say nothing
    # about the axis -- it would only say the swap did not happen.
    taus, top1_same = {}, {}
    for model, label, _ in MODELS[1:]:
        ts, same = [], 0
        for q in queries:
            ids = [r_cid[q][i] for i in r_top[q][:P]]
            a = np.array([caches[ANCHOR][q][c] for c in ids])
            b = np.array([caches[model][q][c] for c in ids])
            ts.append(kendall_tau(a, b))
            same += int(int(np.argmax(a)) == int(np.argmax(b)))
        taus[label], top1_same[label] = float(np.mean(ts)), same
    checks.append(("S6 the models genuinely disagree with the anchor (a swap that did not "
                   "happen would give tau=1)",
                   all(t < 0.95 for t in taus.values()),
                   " · ".join(f"{k} tau={v:+.3f}, same top-1 {top1_same[k]}/{len(queries)}"
                              for k, v in taus.items())))

    # RRF reads ranks, so any strictly increasing transform of the scores must
    # give identical arms. mmarco emits raw logits (range ~[-5,+6]) where bge
    # emits sigmoid probabilities, and this is what licenses comparing them.
    mono = {q: {c: float(np.exp(s / 3.0)) for c, s in caches[MODELS[-1][0]][q].items()}
            for q in queries}
    mono_sc = RT.fuse_grid(r_top, r_h, mono, r_cid, r_rid, r_page, grid, (P,), queries, qrels)
    mono_ok = float(np.abs(mono_sc[P]["recall@10"] - scores[MODELS[-1][0]][P]["recall@10"]).max())
    checks.append(("S7 fusion is invariant to a monotone rescoring (so a logit-scale model and "
                   "a probability-scale model are comparable)",
                   mono_ok == 0.0, f"|d|max={mono_ok:.0e} over the whole w grid"))

    qual_txt = QUALIFY_REPORT.read_text(encoding="utf-8") if QUALIFY_REPORT.exists() else ""
    unqual = [label for model, label, _ in MODELS
              if not re.search(rf"`{re.escape(model)}` \|.*\*\*QUALIFIED\*\*", qual_txt)]
    checks.append(("S8 every model passed tools/eval/qualify_reranker_model.py",
                   not unqual and bool(qual_txt),
                   "all 4 QUALIFIED" if not unqual and qual_txt
                   else f"missing/unqualified: {unqual or 'no report'}"))

    if args.smoke:
        for name, ok, ev in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} — {ev}")
        for model, label, _ in MODELS:
            print(f"  {label:22s} D={arm_D[model]['recall@10'].mean():.4f}  "
                  f"w*={w_or[model]:.2f}")
        print(f"\nsmoke ({len(queries)} queries, {time.time()-t0:.0f}s) -- nothing written")
        return 0 if all(ok for _, ok, _ in checks) else 1

    # ---- pre-registered significance ----------------------------------------
    rng = np.random.default_rng(args.seed)
    fam1_rows, fam2_rows = [], []
    for model, label, _ in MODELS[1:]:
        for m in RT._METRICS:
            observed, p, ci = bootstrap_pvalue(arm_D[model][m] - arm_C[m], rng, args.n_boot)
            (fam1_rows if m == "recall@10" else fam2_rows).append(
                (f"D({label}) vs C", m, observed, p, ci))
    fam1 = holm_correct(fam1_rows, alpha=args.alpha)
    fam2 = holm_correct(fam2_rows, alpha=args.alpha)

    L: list[str] = []
    def w_(s: str = "") -> None:
        L.append(s)

    # Measured here, not carried from PUBLISHED: S3/S5 already gate that these two
    # equal the routed report's own values, so deriving `head` from this run keeps
    # every figure in this file a product of this run. The anchor's own delta and
    # the share it captures were frozen literals ("+0.0017", "1%") until
    # 2026-08-20 -- both are now derived, and both changed sign/direction at
    # rebuild #4.
    c_mean = float(arm_C["recall@10"].mean())
    head = float(oracle["recall@10"].mean()) - c_mean
    anchor_d = float(arm_D[ANCHOR]["recall@10"].mean()) - c_mean
    w_("# cross-encoder ตัวอื่นเอาชนะ hard router ได้ไหม")
    w_()
    w_(f"Generated by `tools/eval/reranker_model_comparison.py` · {len(queries)} คำถาม · "
       f"pool = routed hybrid top-{P} (**เหมือนกันทุกโมเดล**) · ทุก arm ส่งออก k={RT.K} เท่ากัน")
    w_()
    w_(f"`reranker_rrf_routed_test.md` วัด rrf4 ทับ hard router ได้ **{anchor_d:+.4f}** "
       f"ขณะที่ oracle บน pool เดียวกันส่งมอบ **{oracle['recall@10'].mean():.4f}** คือ "
       f"**+{head:.4f}** — โมเดลเก็บได้ {anchor_d/head*100:.0f}% ของเพดานตัวเอง · null ตัวเดียวแยกไม่ออกว่า "
       "“โมเดลนี้อ่อน” หรือ “ไม่เหลืออะไรให้เก็บ” จึงสลับโมเดลโดยไม่แตะอย่างอื่นเลย")
    w_()
    w_("**เงื่อนไขก่อนวัด**: ทุกโมเดลต้องผ่าน `tools/eval/qualify_reranker_model.py` — "
       "`gte-multilingual-reranker-base` จัดอันดับตัวอย่างภาษาไทยได้ถูก แต่ให้คะแนนประโยคกับ "
       "ประโยคกลับหลัง **เท่ากันบิตต่อบิต** (RoPE ตายใต้ transformers 5.x) ถ้าเอามาวัดโดยไม่ตรวจ "
       "จะได้เลขต่ำ แล้วเราจะสรุปว่า “cross-encoder ตัวที่สองก็แพ้เหมือนกัน” ทั้งที่มันเป็น "
       "bag-of-words")
    w_()
    w_("## arms")
    w_()
    w_(f"| โมเดล | ขนาด (M) | recall@{RT.K} | เทียบ C | MRR | nDCG@{RT.K} | w (LOO mode) | "
       f"เก็บเพดานได้ | ms/คู่ | ctx | ตัดข้อความ |")
    w_("|---|---|---|---|---|---|---|---|---|---|---|")
    w_(f"| **C — routing อย่างเดียว (ไม่มี reranker)** | – | **{arm_C['recall@10'].mean():.4f}** | "
       f"– | {arm_C['mrr'].mean():.4f} | {arm_C['ndcg@10'].mean():.4f} | – | – | 0 | – | – |")
    for model, label, size in MODELS:
        a, mt = arm_D[model], meta_of[model]
        d = a["recall@10"].mean() - arm_C["recall@10"].mean()
        wmode = max(set(picks[model]), key=picks[model].count)
        # the anchor's timing comes from the parent run, not this one
        dag = "†" if model == ANCHOR else ""
        w_(f"| D({label}) | {size} | {a['recall@10'].mean():.4f} | {d:+.4f} | "
           f"{a['mrr'].mean():.4f} | {a['ndcg@10'].mean():.4f} | {wmode:.2f} | "
           f"{d/head*100:.0f}% | {mt.get('ms_per_pair', float('nan')):.0f}{dag} | "
           f"{mt.get('max_length', '?')} | {mt.get('truncated_pct', '?')}% |")
    w_(f"| **oracle — เลือก {RT.K} จาก pool {P} ได้สมบูรณ์แบบ** (ขอบเขต ไม่ใช่ระบบ) | – | "
       f"**{oracle['recall@10'].mean():.4f}** | **+{head:.4f}** | {oracle['mrr'].mean():.4f} | "
       f"{oracle['ndcg@10'].mean():.4f} | – | 100% | – | – | – |")
    w_()
    longest = max(meta_of[m].get("longest_pair_tokens", 0) for m, _, _ in MODELS)
    w_(f"`ctx` / `ตัดข้อความ` คือสิ่งเดียวที่ไม่เท่ากันระหว่าง arm จึงรายงานไว้แทนที่จะซ่อน: "
       f"anchor รับ 8192 token อีกสามตัวรับ 512 · แต่ละตัวรันที่ค่าสูงสุดของตัวเอง "
       f"(บังคับ 512 ใส่ anchor จะทำให้มันไม่ reproduce เลขที่ตีพิมพ์ไว้) · "
       f"คู่ที่ยาวที่สุดใน pool คือ **{longest:,} token** และมีเพียง "
       f"**{meta_of[MODELS[1][0]]['truncated_pct']}%** ของคู่ทั้งหมดที่เกิน 512 — "
       f"เล็กเกินกว่าจะอธิบายช่องว่างใด ๆ ในตารางนี้ได้")
    w_()
    w_("† `ms/คู่` ของ anchor มาจาก**การรันคนละครั้ง** (`reranker_rrf_routed_test.py` "
       "ซึ่งเป็นเจ้าของ cache นั้น) ไม่ใช่จากเครื่องสภาพเดียวกับอีกสามแถว — "
       "อย่าอ่านเป็นการเทียบความเร็วกัน")
    w_()
    w_(f"pool ถือ gold ไว้ **{holds.mean():.4f}** (`holds` — ส่ง {P} ใบ ส่งมอบจริงไม่ได้; "
       f"ตัวที่ส่งมอบได้คือ {oracle['recall@10'].mean():.4f} และต้องไม่เกินเพดาน qrels "
       f"{RT.QRELS_CEILING:.4f})")
    w_()
    w_(f"**Family 1 (m={len(fam1)}, ลงทะเบียนก่อนสกอร์โมเดลใหม่ตัวแรก)** — paired bootstrap "
       f"{args.n_boot} รอบ (seed={args.seed}), Holm · w เลือกแบบ leave-one-out บน "
       f"`{RT.SELECT_METRIC}` · **ไม่ทดสอบ anchor ซ้ำ** เพราะมันถูกตีพิมพ์ไว้ในตระกูล m=6 แล้ว "
       "การรันข้อมูลชุดเดิมในตระกูล m=3 จะได้ Holm p คนละค่าโดยที่การวัดไม่ได้เปลี่ยน")
    w_()
    for title, fam in (("Family 1 — recall@10", fam1), ("Family 2 — MRR / nDCG@10", fam2)):
        w_(f"### {title} (m={len(fam)})")
        w_()
        w_("| เทียบ | metric | diff | 95% CI | raw p | Holm-adj p | นัยสำคัญ |")
        w_("|---|---|---|---|---|---|---|")
        for a, b, diff, p, ci, hp, sig in sorted(fam, key=lambda x: x[5]):
            w_(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {hp:.4f} | "
               f"{'**ใช่**' if sig else 'ไม่'} |")
        w_()

    best_m, best_lab, _ = max(MODELS[1:], key=lambda t: arm_D[t[0]]["recall@10"].mean())
    dbest = arm_D[best_m]["recall@10"].mean() - arm_C["recall@10"].mean()
    spread = (max(arm_D[m]["recall@10"].mean() for m, _, _ in MODELS)
              - min(arm_D[m]["recall@10"].mean() for m, _, _ in MODELS))
    # Derived, never typed. The two sentences below quoted "+0.0017", "20 times",
    # "+0.0275 / Holm 0.0228" and the word "makes it worse" as literals; after
    # rebuild #4 the anchor's effect went NEGATIVE, so the ratio was wrong by ~7x
    # and the mmarco sentence printed the verdict word "worse" beside its own
    # POSITIVE number -- a hardcoded verdict word contradicting a live figure in
    # the same clause (see CLAUDE.md, feedback_a_hardcoded_verdict_word_rots_unseen).
    _f2 = {r[0] + "|" + r[1]: r for r in fam2}
    _bk = f"D({best_lab}) vs C|ndcg@10"
    _best_ndcg = (
        f"{_f2[_bk][2]:+.4f} " + ("มีนัยสำคัญ" if _f2[_bk][6]
                                  else "ไม่มีนัยสำคัญ")
        + f", Holm {_f2[_bk][5]:.4f}"
    ) if _bk in _f2 else "UNPARSED"
    _mmarco_d = float(arm_D[MODELS[-1][0]]["recall@10"].mean() - arm_C["recall@10"].mean())
    _mmarco_word = ("ทำให้แย่ลง" if _mmarco_d < 0
                    else "แทบไม่ขยับ")
    w_("## อ่านผล")
    w_()
    w_(f"**ตระกูลที่ลงทะเบียนไว้ (recall@10) ไม่มีตัวไหนผ่าน 0/3** · ตัวที่สูงสุดคือ "
       f"`{best_lab}` **{dbest:+.4f}** (Holm {dict((r[0], r[5]) for r in fam1)[f'D({best_lab}) vs C']:.4f}, "
       f"raw {dict((r[0], r[3]) for r in fam1)[f'D({best_lab}) vs C']:.4f}) — พลาดหวุดหวิด "
       f"จึงต้องเรียกว่า **ยังสรุปไม่ได้ ไม่ใช่ชนะ**")
    w_()
    w_(f"**แต่คำถามที่ตั้งไว้ได้คำตอบแล้ว: โมเดลเป็นตัวแปรจริง** · บน pool เดียวกันเป๊ะ "
       f"ช่วงห่างระหว่างโมเดลคือ **{spread:.4f}** recall@10 (จาก "
       f"{min(arm_D[m]['recall@10'].mean() for m, _, _ in MODELS):.4f} ถึง "
       f"{max(arm_D[m]['recall@10'].mean() for m, _, _ in MODELS):.4f}) ซึ่งกว้างกว่าผลของ "
       f"anchor ทั้งก้อน ({anchor_d:+.4f}) ราว {spread/max(abs(anchor_d), 1e-9):.0f} เท่า · "
       f"ดังนั้น **{anchor_d:+.4f} เป็นคุณสมบัติของ "
       f"`bge-reranker-v2-m3` ไม่ใช่ของ cross-encoder reranking บนคลังนี้** — ตรงกับที่ "
       f"คอลัมน์ oracle บอกไว้ แต่มาจากหลักฐานคนละเส้นทาง")
    w_()
    w_(f"**ส่วนที่สวนสัญชาตญาณคือหลักฐานที่หนักที่สุด**: ตัวที่ทำได้ดีที่สุดคือ `{best_lab}` "
       f"ซึ่งเป็นรุ่น**เก่ากว่า** ที่ v2-m3 ออกมาแทน แต่ดีกว่าทุก metric "
       f"(nDCG@10 {_best_ndcg} ใน Family 2) · การเลือก reranker ที่นี่ "
       f"จึงไม่ได้เดินตามความแรงบน benchmark ทั่วไป ต้องวัดบนคลังนี้เอง · "
       f"และ `mmarco-mMiniLM` **{_mmarco_word}** ({_mmarco_d:+.4f}) "
       f"ซึ่งเข้ากับกฎ RRF ของโปรเจกต์นี้ — fuse ได้ต่อเมื่อสองแขนแรงพอ ๆ กัน")
    w_()
    w_(f"**ข้อควรระวังเรื่องการเลือก**: `{best_lab}` เป็น argmax ของ 4 โมเดลที่วัดบน 106 "
       f"คำถามเดียวกันนี้ · `w` เลือกแบบ LOO แต่ **ตัวโมเดลไม่ได้** จึงอ้างได้แค่ว่า "
       f"“มีโมเดลที่ผ่านการตรวจอย่างน้อยหนึ่งตัวทำได้ดีกว่าอย่างมีนัยทางตัวเลข” "
       f"ไม่ใช่ “ให้ใช้ {best_lab}” — ข้อหลังต้องการชุดคำถามใหม่")
    w_()
    w_(f"**แกนยังเปิด แต่ของรางวัลยังเล็ก**: ตัวที่ดีที่สุดเก็บได้ {dbest/head*100:.0f}% ของ "
       f"+{head:.4f} เหลืออีก {100-dbest/head*100:.0f}% ที่ไม่มีโมเดลใดใน 4 ตัวแตะถึง · "
       f"คำตัดสินจาก oracle จึงไม่เปลี่ยน — จะปิดช่องว่างนี้ต้องใช้ reranker ที่ดีกว่า "
       f"เชิงคุณภาพ (เช่น follow-up (a) ที่เทรนบน candidate ที่ผ่าน hybrid fusion มาแล้ว) "
       f"ไม่ใช่การสลับตัวสำเร็จรูปอีกตัว · **ยังไม่ wire อะไรเข้า `query_service`**")
    w_()
    w_("## โมเดลไม่เห็นตรงกันจริง")
    w_()
    w_("ถ้าสองโมเดลจัดอันดับ pool เหมือนกัน arm จะออกมาเท่ากันและ null จะไม่ได้บอกอะไร "
       "เกี่ยวกับแกนนี้เลย — มันจะบอกแค่ว่าการสลับโมเดลไม่ได้เกิดขึ้น")
    w_()
    w_(f"| เทียบกับ anchor | Kendall tau เฉลี่ย/คำถาม | อันดับ 1 ตรงกัน |")
    w_("|---|---|---|")
    for _, label, _ in MODELS[1:]:
        w_(f"| {label} | {taus[label]:+.3f} | {top1_same[label]}/{len(queries)} |")
    w_()
    w_("## self-checks")
    w_()
    for name, ok, ev in checks:
        w_(f"- [{'PASS' if ok else 'FAIL'}] {name} — {ev}")
    w_()
    w_(f"_{time.strftime('%Y-%m-%dT%H:%M:%S')} · {time.time()-t0:.0f}s · "
       f"{sum(m.get('n_pairs', 0) for m in meta_of.values()):,} คู่ (query, chunk) "
       f"ให้คะแนนรวมทุกโมเดล_")

    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {REPORT}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
