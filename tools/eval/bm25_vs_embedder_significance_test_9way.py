"""9-embedder follow-up to bm25_vs_embedder_significance_test.py: BM25
lexical baseline vs each of the 9 dense embedders, Holm-corrected within
this 9-comparison family per metric.

Reuses the label/exclusion logic from embedder_matrix_9way.py (fixes the
same type-only-label collision risk and superseded-combo exclusion the
original 6-embedder script didn't need to handle).

Run with:
    .venv/Scripts/python.exe tools/eval/bm25_vs_embedder_significance_test_9way.py
"""
from __future__ import annotations

import argparse
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
    _RESULTS_DIR as _DENSE_RESULTS_DIR,
    build_combo_to_embedder,
    bootstrap_pvalue,
    holm_correct,
)

_BM25_RESULTS_DIR = REPO / "data" / "results" / "gold_bm25_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "bm25_vs_embedder_significance_test_9way.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    dense_combo_to_embedder = build_combo_to_embedder(_INDEX_DIR)
    query_set = load_gold_query_set(str(_GOLD_QUERY_SET))
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    queries = list(qrels.keys())
    query_idx = {q: i for i, q in enumerate(queries)}
    n_q = len(queries)

    dense_persisted = [load_retrieval_result(p) for p in _DENSE_RESULTS_DIR.glob("*.json")]
    bm25_persisted = [load_retrieval_result(p) for p in _BM25_RESULTS_DIR.glob("*.json")]
    print(f"loaded {len(dense_persisted)} dense results, {len(bm25_persisted)} bm25 results")

    embedders = sorted(set(dense_combo_to_embedder.values()))
    systems = embedders + ["bm25"]
    sums = {m: {s: np.zeros(n_q) for s in systems} for m in ("recall", "mrr", "ndcg")}
    counts = {m: {s: np.zeros(n_q) for s in systems} for m in ("recall", "mrr", "ndcg")}

    for r in dense_persisted:
        embedder = dense_combo_to_embedder.get(r.combination_id)
        if embedder is None:
            continue
        qi = query_idx.get(r.query)
        if qi is None:
            continue
        relevant = qrels[r.query]
        sums["recall"][embedder][qi] += recall_at_k(r, relevant, args.k)
        sums["mrr"][embedder][qi] += reciprocal_rank(r, relevant)
        sums["ndcg"][embedder][qi] += ndcg_at_k(r, relevant, args.k)
        counts["recall"][embedder][qi] += 1
        counts["mrr"][embedder][qi] += 1
        counts["ndcg"][embedder][qi] += 1

    for r in bm25_persisted:
        qi = query_idx.get(r.query)
        if qi is None:
            continue
        relevant = qrels[r.query]
        sums["recall"]["bm25"][qi] += recall_at_k(r, relevant, args.k)
        sums["mrr"]["bm25"][qi] += reciprocal_rank(r, relevant)
        sums["ndcg"]["bm25"][qi] += ndcg_at_k(r, relevant, args.k)
        counts["recall"]["bm25"][qi] += 1
        counts["mrr"]["bm25"][qi] += 1
        counts["ndcg"]["bm25"][qi] += 1

    per_query = {
        m: {s: np.divide(sums[m][s], counts[m][s], out=np.zeros(n_q), where=counts[m][s] > 0) for s in systems}
        for m in ("recall", "mrr", "ndcg")
    }
    for m in ("recall", "mrr", "ndcg"):
        for s in systems:
            missing = int((counts[m][s] == 0).sum())
            if missing:
                print(f"WARNING: {s} missing {missing}/{n_q} queries for {m}")

    rng = np.random.default_rng(args.seed)
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}

    report_lines = [
        "# BM25 vs each dense embedder -- 9-embedder matrix: paired bootstrap significance test (73-det Gold set)",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}). Each "
        "system's per-query score averaged across the 4 chunker strategies first. "
        f"Holm-Bonferroni correction within this {len(embedders)}-test family per metric "
        f"(alpha={args.alpha}).",
        "",
        "New vs the original 6-embedder comparison: `e5_small`, `qwen3_0.6b`, `sct` "
        "(at its corrected max_seq_length=510).",
        "",
    ]

    for metric_key, metric_label in metric_labels.items():
        pairs = []
        for e in embedders:
            diffs = per_query[metric_key]["bm25"] - per_query[metric_key][e]
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append(("bm25", e, observed, p, ci))
        corrected = holm_correct(pairs, alpha=args.alpha)

        report_lines.append(f"## {metric_label}")
        report_lines.append("")
        report_lines.append("| A | B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |")
        report_lines.append("|---|---|---|---|---|---|---|")
        for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: -x[2]):
            mark = "**yes**" if sig else "no"
            report_lines.append(
                f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |"
            )
        report_lines.append("")

    report_lines.append("## Per-system mean (for reference)")
    report_lines.append("")
    report_lines.append("| system | recall@{0} | mrr | ndcg@{0} |".format(args.k))
    report_lines.append("|---|---|---|---|")
    for s in sorted(systems, key=lambda s: -per_query["recall"][s].mean()):
        report_lines.append(
            f"| {s} | {per_query['recall'][s].mean():.4f} | {per_query['mrr'][s].mean():.4f} | {per_query['ndcg'][s].mean():.4f} |"
        )
    report_lines.append("")

    report = "\n".join(report_lines)
    _OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
