"""9-embedder follow-up to hybrid_significance_test.py: does Hybrid (RRF,
BM25+Dense) beat each of its two components alone, for all 9 embedders?
Two families of 9 tests each (one per embedder), Holm-corrected within each
family per metric: (1) hybrid_E vs BM25-alone, (2) hybrid_E vs dense-alone_E.

Reuses the label/exclusion logic from embedder_matrix_9way.py (fixes the
same type-only-label collision risk and superseded-combo exclusion the
original 6-embedder script didn't need to handle).

Run with:
    .venv/Scripts/python.exe tools/eval/hybrid_significance_test_9way.py
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
_HYBRID_RESULTS_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "hybrid_significance_test_9way.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    base_combo_to_embedder = build_combo_to_embedder(_INDEX_DIR)  # keys end in __dense
    base_combo_to_embedder = {k[: -len("__dense")]: v for k, v in base_combo_to_embedder.items()}

    query_set = load_gold_query_set(str(_GOLD_QUERY_SET))
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    queries = list(qrels.keys())
    query_idx = {q: i for i, q in enumerate(queries)}
    n_q = len(queries)

    dense_persisted = [load_retrieval_result(p) for p in _DENSE_RESULTS_DIR.glob("*.json")]
    bm25_persisted = [load_retrieval_result(p) for p in _BM25_RESULTS_DIR.glob("*.json")]
    hybrid_persisted = [load_retrieval_result(p) for p in _HYBRID_RESULTS_DIR.glob("*.json")]
    print(f"loaded dense={len(dense_persisted)} bm25={len(bm25_persisted)} hybrid={len(hybrid_persisted)}")

    embedders = sorted(set(base_combo_to_embedder.values()))

    def new_table():
        return {m: {e: np.zeros(n_q) for e in embedders + ["bm25"]} for m in ("recall", "mrr", "ndcg")}

    dense_sums, dense_counts = new_table(), new_table()
    hybrid_sums, hybrid_counts = new_table(), new_table()
    bm25_sums, bm25_counts = new_table(), new_table()

    def accumulate(persisted, sums, counts, key_fn, suffix):
        for r in persisted:
            base = r.combination_id[: -len(suffix)] if r.combination_id.endswith(suffix) else None
            if base is None:
                continue
            key = key_fn(base)
            if key is None:
                continue
            qi = query_idx.get(r.query)
            if qi is None:
                continue
            relevant = qrels[r.query]
            sums["recall"][key][qi] += recall_at_k(r, relevant, args.k)
            sums["mrr"][key][qi] += reciprocal_rank(r, relevant)
            sums["ndcg"][key][qi] += ndcg_at_k(r, relevant, args.k)
            counts["recall"][key][qi] += 1
            counts["mrr"][key][qi] += 1
            counts["ndcg"][key][qi] += 1

    accumulate(dense_persisted, dense_sums, dense_counts, lambda base: base_combo_to_embedder.get(base), "__dense")
    accumulate(hybrid_persisted, hybrid_sums, hybrid_counts, lambda base: base_combo_to_embedder.get(base), "__hybrid")
    accumulate(bm25_persisted, bm25_sums, bm25_counts, lambda base: "bm25", "__bm25")

    def per_query(sums, counts):
        return {
            m: {k: np.divide(sums[m][k], counts[m][k], out=np.zeros(n_q), where=counts[m][k] > 0) for k in sums[m]}
            for m in ("recall", "mrr", "ndcg")
        }

    dense_pq = per_query(dense_sums, dense_counts)
    hybrid_pq = per_query(hybrid_sums, hybrid_counts)
    bm25_pq = per_query(bm25_sums, bm25_counts)

    for m in ("recall", "mrr", "ndcg"):
        for e in embedders:
            missing_h = int((hybrid_counts[m][e] == 0).sum())
            missing_d = int((dense_counts[m][e] == 0).sum())
            if missing_h or missing_d:
                print(f"WARNING: {e} missing hybrid={missing_h} dense={missing_d} queries for {m}")

    rng = np.random.default_rng(args.seed)
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}

    report_lines = [
        "# Hybrid vs its components -- 9-embedder matrix: paired bootstrap significance test (73-det Gold set)",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}), each "
        f"system's per-query score averaged across the 4 chunkers first. Two separate "
        f"{len(embedders)}-test families (one per embedder), Holm-corrected independently "
        f"per metric: (1) hybrid_E vs BM25-alone, (2) hybrid_E vs dense-alone_E (alpha={args.alpha}).",
        "",
        "New vs the original 6-embedder hybrid matrix: `e5_small`, `qwen3_0.6b`, `sct` "
        "(at its corrected max_seq_length=510).",
        "",
    ]

    for metric_key, metric_label in metric_labels.items():
        report_lines.append(f"## {metric_label}: hybrid_E vs BM25-alone")
        report_lines.append("")
        report_lines.append("| embedder | mean(hybrid-bm25) | 95% CI | raw p | Holm-adj p | significant |")
        report_lines.append("|---|---|---|---|---|---|")
        pairs = []
        for e in embedders:
            diffs = hybrid_pq[metric_key][e] - bm25_pq[metric_key]["bm25"]
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((e, "bm25", observed, p, ci))
        corrected = holm_correct(pairs, alpha=args.alpha)
        for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: -x[2]):
            mark = "**yes**" if sig else "no"
            report_lines.append(f"| {a} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |")
        report_lines.append("")

        report_lines.append(f"## {metric_label}: hybrid_E vs dense-alone_E (same embedder)")
        report_lines.append("")
        report_lines.append("| embedder | mean(hybrid-dense) | 95% CI | raw p | Holm-adj p | significant |")
        report_lines.append("|---|---|---|---|---|---|")
        pairs = []
        for e in embedders:
            diffs = hybrid_pq[metric_key][e] - dense_pq[metric_key][e]
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((e, "dense", observed, p, ci))
        corrected = holm_correct(pairs, alpha=args.alpha)
        for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: -x[2]):
            mark = "**yes**" if sig else "no"
            report_lines.append(f"| {a} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |")
        report_lines.append("")

    report_lines.append("## Per-system mean (for reference)")
    report_lines.append("")
    report_lines.append("| embedder | hybrid recall@{0} | dense recall@{0} | bm25 recall@{0} |".format(args.k))
    report_lines.append("|---|---|---|---|")
    for e in sorted(embedders, key=lambda e: -hybrid_pq["recall"][e].mean()):
        report_lines.append(
            f"| {e} | {hybrid_pq['recall'][e].mean():.4f} | {dense_pq['recall'][e].mean():.4f} | {bm25_pq['recall']['bm25'].mean():.4f} |"
        )
    report_lines.append("")

    report = "\n".join(report_lines)
    _OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
