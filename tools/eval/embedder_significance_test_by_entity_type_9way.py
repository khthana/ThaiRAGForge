"""Paired bootstrap significance test across the 9-embedder matrix, split by
entity_type, Holm-corrected within each (entity_type, metric) family of 36
pairwise tests.

9-embedder follow-up to embedder_significance_test_by_entity_type.py (the
original 6-embedder version). Reuses the label/exclusion logic from
embedder_matrix_9way.py (same module, same combo-dir exclusion list) rather
than duplicating it, since that logic fixes two real bugs the original
script had: (1) labeling by `type` alone would silently pool e5-large with
e5-small and Qwen3-4B with Qwen3-0.6B under one label; (2) no exclusion list
would double-count the superseded 128-cap sct and rejected 510-cap congen
combos against their correct counterparts.

Same design otherwise: resample unit = query, per-query score averaged
across the 4 chunkers first, paired bootstrap (n_boot=10000), Holm-Bonferroni
correction applied within each (entity_type, metric) group of 36 pairwise
tests separately (not pooled across entity_types).

Run with:
    .venv/Scripts/python.exe tools/eval/embedder_significance_test_by_entity_type_9way.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from embedder_matrix_9way import (  # noqa: E402
    _INDEX_DIR,
    _RESULTS_DIR,
    build_combo_to_embedder,
    bootstrap_pvalue,
    holm_correct,
)

_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "embedder_significance_test_by_entity_type_9way.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    combo_to_embedder = build_combo_to_embedder(_INDEX_DIR)
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}

    entries_raw = yaml.safe_load(_GOLD_QUERY_SET.read_text(encoding="utf-8"))
    entity_type_by_query = {e["query"]: e.get("entity_type", "unknown") for e in entries_raw}
    queries_by_type: dict[str, list[str]] = defaultdict(list)
    for q, et in entity_type_by_query.items():
        queries_by_type[et].append(q)

    persisted = [load_retrieval_result(p) for p in _RESULTS_DIR.glob("*.json")]
    persisted = [r for r in persisted if r.combination_id in combo_to_embedder]
    print(f"loaded {len(persisted)} persisted retrieval results for the 9-embedder matrix")

    embedders = sorted(set(combo_to_embedder.values()))
    n_pairs = len(list(itertools.combinations(embedders, 2)))
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}

    report_lines = [
        "# Embedder pairwise significance test, split by entity_type -- 9-embedder matrix (Gold 73-det)",
        "",
        f"Paired bootstrap (n_boot={args.n_boot}, seed={args.seed}), per-query score "
        "averaged across the 4 chunkers first, resample unit = query WITHIN each "
        "entity_type. Holm-Bonferroni correction applied within each (entity_type, "
        f"metric) group of {n_pairs} pairwise tests separately (alpha={args.alpha}). Query "
        f"counts: {', '.join(f'{et}={len(qs)}' for et, qs in sorted(queries_by_type.items()))}.",
        "",
        "Embedders: " + ", ".join(embedders),
        "",
    ]

    rng = np.random.default_rng(args.seed)

    for etype in sorted(queries_by_type):
        queries = queries_by_type[etype]
        n_q = len(queries)
        query_idx = {q: i for i, q in enumerate(queries)}

        sums = {m: {e: np.zeros(n_q) for e in embedders} for m in ("recall", "mrr", "ndcg")}
        counts = {m: {e: np.zeros(n_q) for e in embedders} for m in ("recall", "mrr", "ndcg")}
        for r in persisted:
            embedder = combo_to_embedder.get(r.combination_id)
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

        for m in ("recall", "mrr", "ndcg"):
            for e in embedders:
                missing = int((counts[m][e] == 0).sum())
                if missing:
                    print(f"WARNING: entity_type={etype} embedder={e} missing {missing}/{n_q} queries for {m}")

        per_query = {
            m: {e: np.divide(sums[m][e], counts[m][e], out=np.zeros(n_q), where=counts[m][e] > 0) for e in embedders}
            for m in ("recall", "mrr", "ndcg")
        }

        report_lines.append(f"# entity_type = {etype} (n={n_q})")
        report_lines.append("")

        for metric_key, metric_label in metric_labels.items():
            pairs = []
            for a, b in itertools.combinations(embedders, 2):
                diffs = per_query[metric_key][a] - per_query[metric_key][b]
                observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
                pairs.append((a, b, observed, p, ci))
            corrected = holm_correct(pairs, alpha=args.alpha)

            report_lines.append(f"## {etype} / {metric_label}")
            report_lines.append("")
            report_lines.append("| A | B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |")
            report_lines.append("|---|---|---|---|---|---|---|")
            for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
                mark = "**yes**" if sig else "no"
                report_lines.append(
                    f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |"
                )
            report_lines.append("")

        report_lines.append(f"### {etype}: per-embedder mean")
        report_lines.append("")
        report_lines.append("| embedder | recall@{0} | mrr | ndcg@{0} |".format(args.k))
        report_lines.append("|---|---|---|---|")
        for e in sorted(embedders, key=lambda e: -per_query["recall"][e].mean()):
            report_lines.append(
                f"| {e} | {per_query['recall'][e].mean():.4f} | {per_query['mrr'][e].mean():.4f} | {per_query['ndcg'][e].mean():.4f} |"
            )
        report_lines.append("")

    report = "\n".join(report_lines)
    _OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
