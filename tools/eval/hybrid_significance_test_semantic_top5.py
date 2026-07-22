"""Per-chunker (semantic-only) pairwise significance test among the top 5
hybrid combos identified in the 9-embedder matrix (qwen3_0.6b, bge_m3,
e5_small, qwen3, jina_v5 -- recall@10 range 0.6796-0.6935, previously an
untested cluster; see docs/paper-results-summary.md "Open items").

Unlike hybrid_significance_test_9way.py (which averages each embedder across
all 4 chunkers and compares hybrid_E vs its own BM25/dense components), this
script isolates the semantic chunker only and does embedder-vs-embedder
pairwise comparisons under hybrid retrieval, to check whether any of the top
5 is actually better than the others rather than a tied cluster.

Run with:
    .venv/Scripts/python.exe tools/eval/hybrid_significance_test_semantic_top5.py
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
    build_combo_to_chunker_embedder,
    bootstrap_pvalue,
    holm_correct,
)

_HYBRID_RESULTS_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "hybrid_significance_test_semantic_top5.md"

CHUNKER = "semantic"
TOP5 = ["qwen3_0.6b", "bge_m3", "e5_small", "qwen3", "jina_v5"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)  # keys end in __dense
    combo_ce = {k[: -len("__dense")]: v for k, v in combo_ce.items()}
    semantic_top5_combos = {
        base for base, (chunker, embedder) in combo_ce.items()
        if chunker == CHUNKER and embedder in TOP5
    }
    found_embedders = {combo_ce[b][1] for b in semantic_top5_combos}
    missing = set(TOP5) - found_embedders
    if missing:
        print(f"WARNING: no semantic-chunker combo dirs found for: {missing}")

    query_set = load_gold_query_set(str(_GOLD_QUERY_SET))
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    queries = list(qrels.keys())
    query_idx = {q: i for i, q in enumerate(queries)}
    n_q = len(queries)

    hybrid_persisted = [load_retrieval_result(p) for p in _HYBRID_RESULTS_DIR.glob("*.json")]
    print(f"loaded hybrid={len(hybrid_persisted)} persisted results")

    embedders = [e for e in TOP5 if e in found_embedders]
    sums = {m: {e: np.zeros(n_q) for e in embedders} for m in ("recall", "mrr", "ndcg")}
    counts = {m: {e: np.zeros(n_q) for e in embedders} for m in ("recall", "mrr", "ndcg")}
    for r in hybrid_persisted:
        base = r.combination_id[: -len("__hybrid")] if r.combination_id.endswith("__hybrid") else None
        if base is None or base not in semantic_top5_combos:
            continue
        embedder = combo_ce[base][1]
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

    per_query = {
        m: {e: np.divide(sums[m][e], counts[m][e], out=np.zeros(n_q), where=counts[m][e] > 0) for e in embedders}
        for m in ("recall", "mrr", "ndcg")
    }
    for m in ("recall", "mrr", "ndcg"):
        for e in embedders:
            missing_q = int((counts[m][e] == 0).sum())
            if missing_q:
                print(f"WARNING: {e} missing {missing_q}/{n_q} queries for {m}")

    rng = np.random.default_rng(args.seed)
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}
    n_pairs = len(list(itertools.combinations(embedders, 2)))

    lines = [
        "# Hybrid top-5 embedders, semantic chunker only -- pairwise significance test (Gold 73-det)",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}), "
        f"semantic chunker + hybrid (RRF) retrieval only -- no cross-chunker averaging. "
        f"Holm-Bonferroni correction within each metric's {n_pairs} pairwise tests (alpha={args.alpha}). "
        "Answers: is any of the 5 numerically-top hybrid combos "
        "(qwen3_0.6b, bge_m3, e5_small, qwen3, jina_v5) actually better than the others, "
        "or are they a statistically tied cluster?",
        "",
    ]
    for metric_key, metric_label in metric_labels.items():
        pairs = []
        for a, b in itertools.combinations(embedders, 2):
            diffs = per_query[metric_key][a] - per_query[metric_key][b]
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((a, b, observed, p, ci))
        corrected = holm_correct(pairs, alpha=args.alpha)
        lines.append(f"## {metric_label}")
        lines.append("")
        lines.append("| A | B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |")
        lines.append("|---|---|---|---|---|---|---|")
        for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
            mark = "**yes**" if sig else "no"
            lines.append(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |")
        lines.append("")

    lines.append("## Per-embedder mean (semantic + hybrid only)")
    lines.append("")
    lines.append("| embedder | recall@{0} | mrr | ndcg@{0} |".format(args.k))
    lines.append("|---|---|---|---|")
    for e in sorted(embedders, key=lambda e: -per_query["recall"][e].mean()):
        lines.append(f"| {e} | {per_query['recall'][e].mean():.4f} | {per_query['mrr'][e].mean():.4f} | {per_query['ndcg'][e].mean():.4f} |")
    lines.append("")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
