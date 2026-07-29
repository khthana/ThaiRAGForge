"""Chunker-vs-chunker significance test, hybrid retrieval, per embedder.

Resolves an open item from docs/paper-results-summary.md ("Top single-combo
across the entire study"): every existing significance test compares either
(a) embedders averaged *across* chunkers, or (b) embedders *within* one fixed
chunker -- none compares chunkers against each other at a fixed
embedder+retriever. The previously-cited "semantic x qwen3_0.6b is the best
combo in the study" (recall@10=0.7048, retracted 2026-07-29 after the
OCR-remediation rebuild refresh -- fresh value 0.6152, no longer even the top
chunker for that embedder numerically) was never actually tested this way.

For each embedder, one family of C(4,2)=6 pairwise chunker comparisons
(fixed_size/recursive/semantic/sentence), Holm-corrected per embedder per
metric. Pure recompute from already-persisted hybrid retrieval results
(data/results/gold_hybrid_73det) -- no new retrieval needed.

Run with:
    .venv/Scripts/python.exe tools/eval/hybrid_chunker_significance_test.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from embedder_matrix_9way import (  # noqa: E402
    _INDEX_DIR,
    EMBEDDER_ORDER,
    build_combo_to_chunker_embedder,
    bootstrap_pvalue,
    holm_correct,
)

_HYBRID_RESULTS_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "hybrid_chunker_significance_test.md"
CHUNKERS = ["fixed_size", "recursive", "semantic", "sentence"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    queries = list(qrels.keys())
    query_idx = {q: i for i, q in enumerate(queries)}
    n_q = len(queries)

    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)  # keys end in __dense
    combo_ce = {k[: -len("__dense")]: v for k, v in combo_ce.items()}

    persisted = [load_retrieval_result(p) for p in _HYBRID_RESULTS_DIR.glob("*.json")]
    print(f"loaded {len(persisted)} persisted hybrid retrieval results")

    embedders = [e for e in EMBEDDER_ORDER if e in {v[1] for v in combo_ce.values()}]

    # per_query[embedder][chunker][metric] -> np.ndarray over queries
    sums = {e: {c: {m: np.zeros(n_q) for m in ("recall", "mrr", "ndcg")} for c in CHUNKERS} for e in embedders}
    counts = {e: {c: {m: np.zeros(n_q) for m in ("recall", "mrr", "ndcg")} for c in CHUNKERS} for e in embedders}

    for r in persisted:
        base = r.combination_id[: -len("__hybrid")] if r.combination_id.endswith("__hybrid") else None
        if base is None or base not in combo_ce:
            continue
        chunker, embedder = combo_ce[base]
        if embedder not in sums or chunker not in CHUNKERS:
            continue
        qi = query_idx.get(r.query)
        if qi is None:
            continue
        relevant = qrels[r.query]
        sums[embedder][chunker]["recall"][qi] += recall_at_k(r, relevant, args.k)
        sums[embedder][chunker]["mrr"][qi] += reciprocal_rank(r, relevant)
        sums[embedder][chunker]["ndcg"][qi] += ndcg_at_k(r, relevant, args.k)
        for m in ("recall", "mrr", "ndcg"):
            counts[embedder][chunker][m][qi] += 1

    per_query = {
        e: {
            c: {
                m: np.divide(sums[e][c][m], counts[e][c][m], out=np.zeros(n_q), where=counts[e][c][m] > 0)
                for m in ("recall", "mrr", "ndcg")
            }
            for c in CHUNKERS
        }
        for e in embedders
    }
    for e in embedders:
        for c in CHUNKERS:
            for m in ("recall", "mrr", "ndcg"):
                missing = int((counts[e][c][m] == 0).sum())
                if missing:
                    print(f"WARNING: {e}/{c} missing {missing}/{n_q} queries for {m}")

    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}
    n_pairs = len(list(itertools.combinations(CHUNKERS, 2)))

    lines = [
        "# Chunker-vs-chunker significance test, hybrid retrieval, per embedder (Gold 73-det)",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}), hybrid "
        f"(RRF) retrieval only, one chunker-pairwise family ({n_pairs} tests) per embedder, "
        f"Holm-corrected within each embedder's family per metric (alpha={args.alpha}). Pure "
        "recompute from already-persisted `gold_hybrid_73det` results -- no new retrieval.",
        "",
        "Resolves the open item flagged in docs/paper-results-summary.md after the "
        "2026-07-29 stale-cache fix: whether `semantic` (or any chunker) is actually the "
        "best chunker for a fixed embedder+hybrid, vs. just numerically highest.",
        "",
    ]

    # ---- aggregate: chunker vs chunker, averaged across all embedders first ----
    # Directly tests the project's headline "semantic chunking wins" claim
    # (originally based on raw cross-chunker means only, never paired-tested).
    agg_per_query = {
        c: {m: np.mean(np.stack([per_query[e][c][m] for e in embedders]), axis=0) for m in ("recall", "mrr", "ndcg")}
        for c in CHUNKERS
    }
    rng = np.random.default_rng(args.seed)
    lines.append("## Aggregate: chunker vs chunker, averaged across all 9 embedders first")
    lines.append("")
    lines.append(
        "Mirrors the embedder-matrix convention (each system's per-query score averaged "
        "across the other axis first) -- here each chunker's per-query hybrid score is "
        "averaged across all 9 embedders before the paired bootstrap, one 6-pair family "
        "per metric."
    )
    lines.append("")
    for metric_key, metric_label in metric_labels.items():
        pairs = []
        for a, b in itertools.combinations(CHUNKERS, 2):
            diffs = agg_per_query[a][metric_key] - agg_per_query[b][metric_key]
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((a, b, observed, p, ci))
        corrected = holm_correct(pairs, alpha=args.alpha)
        lines.append(f"### {metric_label}")
        lines.append("")
        lines.append("| A | B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |")
        lines.append("|---|---|---|---|---|---|---|")
        for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
            mark = "**yes**" if sig else "no"
            lines.append(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |")
        lines.append("")

    lines.append("### Per-chunker mean, aggregate across 9 embedders")
    lines.append("")
    lines.append("| chunker | recall@{0} | mrr | ndcg@{0} |".format(args.k))
    lines.append("|---|---|---|---|")
    for c in sorted(CHUNKERS, key=lambda c: -agg_per_query[c]["recall"].mean()):
        lines.append(
            f"| {c} | {agg_per_query[c]['recall'].mean():.4f} | {agg_per_query[c]['mrr'].mean():.4f} | "
            f"{agg_per_query[c]['ndcg'].mean():.4f} |"
        )
    lines.append("")

    for e in embedders:
        lines.append(f"## {e}")
        lines.append("")
        for metric_key, metric_label in metric_labels.items():
            rng = np.random.default_rng(args.seed)
            pairs = []
            for a, b in itertools.combinations(CHUNKERS, 2):
                diffs = per_query[e][a][metric_key] - per_query[e][b][metric_key]
                observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
                pairs.append((a, b, observed, p, ci))
            corrected = holm_correct(pairs, alpha=args.alpha)
            lines.append(f"### {metric_label}")
            lines.append("")
            lines.append("| A | B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |")
            lines.append("|---|---|---|---|---|---|---|")
            for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
                mark = "**yes**" if sig else "no"
                lines.append(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |")
            lines.append("")

        lines.append(f"### Per-chunker mean, {e}")
        lines.append("")
        lines.append("| chunker | recall@{0} | mrr | ndcg@{0} |".format(args.k))
        lines.append("|---|---|---|---|")
        for c in sorted(CHUNKERS, key=lambda c: -per_query[e][c]["recall"].mean()):
            lines.append(
                f"| {c} | {per_query[e][c]['recall'].mean():.4f} | {per_query[e][c]['mrr'].mean():.4f} | "
                f"{per_query[e][c]['ndcg'].mean():.4f} |"
            )
        lines.append("")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
