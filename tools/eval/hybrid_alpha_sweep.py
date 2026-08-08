# -*- coding: utf-8 -*-
"""Sweep the hybrid fusion weight (alpha = dense_weight, bm25_weight = 1-alpha)
globally and *per entity_type*, on the 106-query 73det Gold set.

Why this exists. Every hybrid figure this project has ever reported was produced
at an implicit, unswept 50:50 -- `HybridRetriever`'s `dense_weight`/`bm25_weight`
existed but a full-repo scan found them used nowhere outside one degenerate unit
test. That would be a minor omission if the two arms were uniformly matched, but
they are not: BM25 alone scores **0.8147** recall@10 on `person` queries and
**0.3497** on `program` (bm25_hybrid_entity_type_breakdown.md) -- a 0.465 swing
wider than any embedder-to-embedder gap in the whole study. The project's own
fusion rule ("RRF helps the weaker arm and taxes the stronger one") predicts that
a single global weight fits one compromise to two opposite regimes.

The design decision that makes the sweep interpretable: alpha is applied to the
**rrf** branch (each arm's reciprocal-rank contribution scaled by its weight),
not to the separate `weighted` score-fusion branch. A uniform 0.5x factor cannot
reorder anything, so **alpha=0.5 is rank-order-identical to the plain unweighted
RRF behind every published number** -- it is a true no-op control, and the sweep
isolates the weight instead of confounding it with a switch from rank fusion to
score fusion. `--self-check` verifies that identity against the real
`HybridRetriever` rather than assuming it.

Cost note: the fusion is a deterministic function of the two arms' rank vectors,
so each arm is retrieved **once** per (index, query) at k=n and every alpha is
then a vectorised re-fusion in memory. That turns a 21-alpha sweep from 21 full
retrieval passes into one, which is what makes a fine grid affordable at all
(the dominant cost is `BM25Okapi` being rebuilt per query, ~1.9s -- see
cost_latency_pareto.md).

Reporting rule, per the task that commissioned this: report a **range of alpha
that does not degrade**, not a single best value. Tuning alpha on the same 106
queries it is reported on is overfitting, which is why the headline arms include
a leave-one-out variant fitted per entity_type on the other queries only.

Run with:
    .venv/Scripts/python.exe tools/eval/hybrid_alpha_sweep.py
    .venv/Scripts/python.exe tools/eval/hybrid_alpha_sweep.py --self-check
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder, build_retriever  # noqa: E402
from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from rag_lab.query_service import discover_indices  # noqa: E402
from rag_lab.schema import Query, RankedChunk, RetrievalResult  # noqa: E402

from pythainlp.tokenize import word_tokenize  # noqa: E402

GOLD_PATH = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
OUTPUT = REPO / "data" / "results" / "hybrid_alpha_sweep.md"
K = 10
RRF_K = 60
N_BOOT = 10_000
SEED = 42

# Chosen to span dense-arm strength, because the fusion rule predicts the
# optimum moves with the *relative* strength of the two arms -- a sweep on one
# combo could not tell "alpha matters" from "alpha matters for this embedder".
# sentence+qwen3_0.6b is the best single hybrid combo on this set;
# semantic+bge_m3 is mid and the person specialist; fixed_size+m2v is the known
# RRF failure case, where hybrid is significantly WORSE than BM25 alone -- if
# alpha is doing what the rule says, that combo's optimum must sit far toward
# BM25, and that is a prediction the sweep can falsify.
DEFAULT_COMBOS = {
    "sentence+qwen3_0.6b": "plain__sentence__qwen3__ff8f6c49",
    "semantic+bge_m3": "plain__semantic__local__8aae9bcd",
    "fixed_size+m2v": "plain__fixed_size__local__f71f693a",
}

METRICS = {
    "recall@10": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    "ndcg@10": lambda r, rel: ndcg_at_k(r, rel, K),
}


def load_gold() -> tuple[dict[str, list[str]], dict[str, str]]:
    entries = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))
    return (
        {e["query"]: e["relevant_resolution_ids"] for e in entries},
        {e["query"]: e.get("entity_type", "unknown") for e in entries},
    )


def arm_ranks(index, embedder, queries: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(dense_rank, bm25_rank, resolution_id per chunk) for every query.

    Each rank array is (n_queries, n_chunks) int32 holding the 1-based rank of
    that chunk under that arm -- the only thing RRF fusion reads. Retrieving at
    k=n mirrors what HybridRetriever itself does internally.
    """
    from rag_lab.retrievers import BM25Retriever, DenseRetriever

    dense_r, bm25_r = DenseRetriever(), BM25Retriever()
    n = len(index.chunks)
    pos = {c.chunk_id: i for i, c in enumerate(index.chunks)}
    res_ids = [c.resolution_id for c in index.chunks]

    d_ranks = np.zeros((len(queries), n), dtype=np.int32)
    b_ranks = np.zeros((len(queries), n), dtype=np.int32)
    for qi, q in enumerate(queries):
        prepared = Query(text=q, vector=embedder.embed_query(q), tokens=word_tokenize(q))
        for ranked, out in ((dense_r.retrieve(prepared, index, n), d_ranks),
                            (bm25_r.retrieve(prepared, index, n), b_ranks)):
            for rc in ranked:
                out[qi, pos[rc.chunk_id]] = rc.rank
    # A chunk missing from an arm would keep rank 0, and 1/(rrf_k+0) outscores
    # rank 1 -- the failure would be silent and would flatter whichever arm was
    # short. Both arms are retrieved at k=n and do return all n today; assert it
    # so a future change to either retriever fails loudly here.
    assert d_ranks.min() > 0 and b_ranks.min() > 0, "an arm returned fewer than n chunks"
    return d_ranks, b_ranks, res_ids


def fuse(
    d_rank: np.ndarray, b_rank: np.ndarray, res_ids: list[str], query: str, alpha: float
) -> RetrievalResult:
    """One query's top-K under weighted RRF, as a RetrievalResult so the real
    metric functions (resolution-level, ADR-0002) score it unmodified."""
    score = alpha / (RRF_K + d_rank) + (1.0 - alpha) / (RRF_K + b_rank)
    # Tie-break by DENSE RANK, not by chunk index. HybridRetriever fills its
    # `fused` dict from the dense ranking first (which covers all n chunks), then
    # sorts it with Python's stable `sorted` -- so an exact score tie there is
    # settled by dense rank. Exact ties are not hypothetical: they happen
    # whenever two chunks swap ranks between the arms (e.g. dense 3/bm25 4 vs
    # dense 4/bm25 3, identical at alpha=0.5), and a naive argsort tie-break by
    # index makes the alpha=0.5 control disagree with the retriever it is
    # supposed to reproduce. Caught by --self-check, not by reading the code.
    top = np.lexsort((d_rank, -score))[:K]
    return RetrievalResult(
        query=query,
        combination_id=f"alpha_{alpha:.2f}",
        top_k=K,
        retriever="hybrid",
        results=[
            RankedChunk(
                chunk_id=str(i), resolution_id=res_ids[i], page=1,
                score=float(score[i]), rank=r + 1, text="",
            )
            for r, i in enumerate(top)
        ],
    )


def self_check(index, embedder, queries: list[str], d_ranks, b_ranks, res_ids) -> list[str]:
    """Verify the claims the whole sweep rests on, against the real retrievers
    rather than by assertion, at all THREE points of the grid where an
    independent ground truth exists:

      alpha=0.00 -> must reproduce BM25Retriever alone,
      alpha=0.50 -> must reproduce HybridRetriever (plain unweighted RRF),
      alpha=1.00 -> must reproduce DenseRetriever alone.

    The midpoint alone would not catch a fusion that is correct on average but
    wrong in how it weights an arm; pinning both endpoints as well means any
    alpha in between is an interpolation between two verified anchors.
    """
    from rag_lab.retrievers import BM25Retriever, DenseRetriever

    out = []
    anchors = (
        (0.0, "bm25-alone", BM25Retriever()),
        (0.5, "hybrid rrf", build_retriever(StrategySpec(type="hybrid"))),
        (1.0, "dense-alone", DenseRetriever()),
    )
    for qi, q in enumerate(queries):
        prepared = Query(text=q, vector=embedder.embed_query(q), tokens=word_tokenize(q))
        for alpha, name, retriever in anchors:
            got = [rc.resolution_id for rc in retriever.retrieve(prepared, index, K)]
            mine = [
                rc.resolution_id
                for rc in fuse(d_ranks[qi], b_ranks[qi], res_ids, q, alpha).results
            ]
            ok = "MATCH" if got == mine else "MISMATCH"
            out.append(f"  q{qi} alpha={alpha:.2f} vs {name:11s}: {ok}  {q[:40]}")
            if got != mine:
                out.append(f"    real: {got}\n    mine: {mine}")
    return out


def mean_by(vals: dict[str, float], queries: list[str]) -> float:
    return statistics.mean(vals[q] for q in queries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combos", nargs="*", default=None,
                    help="labels from DEFAULT_COMBOS; default = all three")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--self-check", type=int, default=0,
                    help="verify the vectorised fusion against HybridRetriever on N queries")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha-level", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=OUTPUT)
    args = ap.parse_args()

    qrels, etype = load_gold()
    queries = sorted(qrels)
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_type[etype[q]].append(q)

    alphas = [round(i * args.step, 4) for i in range(int(round(1.0 / args.step)) + 1)]
    combos = {k: v for k, v in DEFAULT_COMBOS.items() if not args.combos or k in args.combos}
    indices = {i.combo_id: i for i in discover_indices(INDEX_DIR)}
    store = ArtifactStore()
    rng = np.random.default_rng(args.seed)

    lines: list[str] = []
    out = lines.append
    out(f"# Hybrid fusion-weight (alpha) sweep -- global and per `entity_type` (n={len(queries)})")
    out("")
    out("Generated by `tools/eval/hybrid_alpha_sweep.py`. `alpha` is the **dense**")
    out("weight; BM25 gets `1-alpha`. Weights scale each arm's reciprocal-rank")
    out("contribution inside RRF, so **alpha=0.50 is rank-order-identical to the plain")
    out("unweighted RRF behind every hybrid number this project has published** -- the")
    out(f"sweep isolates the weight. k={K}, rrf_k={RRF_K}, grid step {args.step}.")
    out("")

    for label, combo_id in combos.items():
        info = indices[combo_id]
        embedder = build_embedder(info.embedder)
        index = store.load(info.dir)
        t0 = time.time()
        d_ranks, b_ranks, res_ids = arm_ranks(index, embedder, queries)
        elapsed = time.time() - t0
        print(f"[{label}] retrieved both arms for {len(queries)} queries in {elapsed:.0f}s "
              f"({len(index.chunks)} chunks)")

        if args.self_check:
            checks = self_check(index, embedder, queries[: args.self_check], d_ranks, b_ranks,
                                res_ids)
            out(f"## Self-check -- {label}")
            out("")
            out("Vectorised fusion vs the real retrievers, top-10 resolution ids, at the")
            out("three grid points with an independent ground truth: alpha=0.00 must equal")
            out("`BM25Retriever` alone, alpha=0.50 the shipped `HybridRetriever` (plain")
            out("unweighted RRF), alpha=1.00 `DenseRetriever` alone.")
            out("")
            out("```")
            for c in checks:
                out(c)
            out("```")
            out("")
            print("\n".join(checks))

        # per-query score for every (alpha, metric)
        scored: dict[tuple[float, str], dict[str, float]] = {}
        for a in alphas:
            results = {q: fuse(d_ranks[qi], b_ranks[qi], res_ids, q, a)
                       for qi, q in enumerate(queries)}
            for m, fn in METRICS.items():
                scored[(a, m)] = {q: fn(results[q], qrels[q]) for q in queries}

        out(f"## {label} -- recall@10 by alpha and entity_type")
        out("")
        out(f"Index `{combo_id}`, {len(index.chunks):,} chunks. Column `all` is the")
        out("aggregate over all 106 queries; alpha=0.50 is the shipped setting.")
        out("")
        types = sorted(by_type)
        out("| alpha | " + " | ".join(f"{t} (n={len(by_type[t])})" for t in types) + " | all |")
        out("|---" * (len(types) + 2) + "|")
        for a in alphas:
            vals = scored[(a, "recall@10")]
            cells = [f"{mean_by(vals, by_type[t]):.4f}" for t in types]
            star = " **(shipped)**" if abs(a - 0.5) < 1e-9 else ""
            out(f"| {a:.2f}{star} | " + " | ".join(cells)
                + f" | {statistics.mean(vals.values()):.4f} |")
        out("")

        # --- non-degrading range, per entity_type --------------------------
        out(f"### Non-degrading alpha range -- {label}")
        out("")
        out("For each scope, the argmax alpha, and the contiguous range of alpha whose")
        out("paired-bootstrap 95% CI against that argmax includes zero (i.e. not")
        out("distinguishable from the best). **Descriptive, not a corrected test** --")
        out("with 21 alphas per scope a Holm correction over the grid would be")
        out("uninformative; the corrected tests are in the next section.")
        out("")
        out("| scope | n | best alpha | recall@10 at best | alpha=0.50 | plateau (CI includes 0) |")
        out("|---|---|---|---|---|---|")
        plateaus: dict[str, tuple[float, list[float]]] = {}
        for scope in types + ["all"]:
            qs = queries if scope == "all" else by_type[scope]
            means = {a: mean_by(scored[(a, "recall@10")], qs) for a in alphas}
            best = max(alphas, key=lambda a: means[a])
            keep = []
            for a in alphas:
                diffs = np.array([scored[(a, "recall@10")][q] - scored[(best, "recall@10")][q]
                                  for q in qs])
                _obs, _p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
                if ci[0] <= 0.0 <= ci[1]:
                    keep.append(a)
            plateaus[scope] = (best, keep)
            span = f"{min(keep):.2f}-{max(keep):.2f}" if keep else "-"
            out(f"| {scope} | {len(qs)} | {best:.2f} | {means[best]:.4f} | {means[0.5]:.4f} "
                f"| {span} ({len(keep)}/{len(alphas)} grid points) |")
        out("")

        # --- does a tuned alpha beat the shipped 0.50? --------------------
        out(f"### Is a tuned alpha worth adopting? -- {label}")
        out("")
        out("Arms, all at identical retrieval budget (k=10 from one index):")
        out("")
        out("- `alpha=0.50` -- shipped, nothing fitted.")
        out("- `global best` -- one alpha, argmax over all 106. Oracle (fitted on the")
        out("  test set), an upper bound for a single global weight.")
        out("- `per-type best` -- a different alpha per entity_type, each argmax on its")
        out("  own queries. Oracle again, and the ceiling for this whole idea.")
        out("- `per-type (loo)` -- per entity_type, alpha chosen from that type's OTHER")
        out("  queries only, then scored on the held-out one. The constructible")
        out("  estimate, and the one to cite.")
        out("")
        arms: dict[str, dict[str, list[float]]] = {}
        for metric in METRICS:
            gbest = max(alphas, key=lambda a: mean_by(scored[(a, metric)], queries))
            pbest = {t: max(alphas, key=lambda a: mean_by(scored[(a, metric)], by_type[t]))
                     for t in types}
            loo = []
            for q in queries:
                peers = [x for x in by_type[etype[q]] if x != q]
                a = max(alphas, key=lambda a: mean_by(scored[(a, metric)], peers)) if peers else 0.5
                loo.append(scored[(a, metric)][q])
            arms[metric] = {
                "alpha=0.50": [scored[(0.5, metric)][q] for q in queries],
                "global best": [scored[(gbest, metric)][q] for q in queries],
                "per-type best": [scored[(pbest[etype[q]], metric)][q] for q in queries],
                "per-type (loo)": loo,
            }
            out(f"- {metric}: global best alpha = **{gbest:.2f}**, per-type = "
                + ", ".join(f"{t} **{pbest[t]:.2f}**" for t in types))
        out("")
        pairs, mlabels = [], []
        for metric in METRICS:
            for arm in ("global best", "per-type best", "per-type (loo)"):
                diffs = np.array(arms[metric][arm]) - np.array(arms[metric]["alpha=0.50"])
                obs, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
                pairs.append((arm, "alpha=0.50", obs, p, ci))
                mlabels.append(metric)
        corrected = holm_correct(pairs, alpha=args.alpha_level)
        out(f"Paired bootstrap, {args.n_boot:,} resamples, seed {args.seed}, Holm within a "
            f"family of m = **{len(corrected)}**.")
        out("")
        out("| metric | arm | mean | vs alpha=0.50 | 95% CI | Holm-adj p | significant |")
        out("|---|---|---|---|---|---|---|")
        for metric, (a, _b, diff, _p, ci, adj, sig) in zip(mlabels, corrected):
            out(f"| {metric} | {a} | {statistics.mean(arms[metric][a]):.4f} | {diff:+.4f} "
                f"| [{ci[0]:+.4f}, {ci[1]:+.4f}] | {adj:.4f} | {'**yes**' if sig else 'no'} |")
        out("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
