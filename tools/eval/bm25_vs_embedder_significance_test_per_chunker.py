"""Per-chunker follow-up to bm25_vs_embedder_significance_test_9way.py: that
script averages each system's per-query score across all 4 chunkers before
testing BM25 vs. each embedder. This script instead tests BM25 vs. each
embedder **within each chunker separately** (no cross-chunker averaging) --
resolves paper-results-summary.md Open item #1 ("raw numbers there look more
favorable to BM25 than the aggregate view; worth checking if that's real or
a chunker-selection artifact").

Four independent 9-test families (one per chunker), each Holm-corrected
separately per metric -- mirrors the "family per natural grouping" Holm
convention used throughout this project (see docs/paper-results-summary.md
Methodology section).

Run with:
    .venv/Scripts/python.exe tools/eval/bm25_vs_embedder_significance_test_per_chunker.py
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
    EMBEDDER_ORDER,
    build_combo_to_chunker_embedder,
    bootstrap_pvalue,
    holm_correct,
)

_BM25_RESULTS_DIR = REPO / "data" / "results" / "gold_bm25_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "bm25_vs_embedder_significance_test_per_chunker.md"


def _strip(cid: str, suffix: str) -> str | None:
    return cid[: -len(suffix)] if cid.endswith(suffix) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)  # keys end in __dense
    chunker_by_base = {k[: -len("__dense")]: v[0] for k, v in combo_ce.items()}
    embedder_by_base = {k[: -len("__dense")]: v[1] for k, v in combo_ce.items()}
    chunkers = sorted(set(chunker_by_base.values()))
    embedders = [e for e in EMBEDDER_ORDER if e in set(embedder_by_base.values())]

    query_set = load_gold_query_set(str(_GOLD_QUERY_SET))
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    queries = list(qrels.keys())
    query_idx = {q: i for i, q in enumerate(queries)}
    n_q = len(queries)

    dense_persisted = [load_retrieval_result(p) for p in _DENSE_RESULTS_DIR.glob("*.json")]
    bm25_persisted = [load_retrieval_result(p) for p in _BM25_RESULTS_DIR.glob("*.json")]
    print(f"loaded {len(dense_persisted)} dense results, {len(bm25_persisted)} bm25 results")

    # per_query[chunker][metric][system] -> np.array over queries
    systems = embedders + ["bm25"]
    per_query = {
        c: {m: {s: np.zeros(n_q) for s in systems} for m in ("recall", "mrr", "ndcg")}
        for c in chunkers
    }
    counts = {
        c: {m: {s: np.zeros(n_q) for s in systems} for m in ("recall", "mrr", "ndcg")}
        for c in chunkers
    }

    for r in dense_persisted:
        base = _strip(r.combination_id, "__dense")
        if base is None:
            continue
        chunker = chunker_by_base.get(base)
        embedder = embedder_by_base.get(base)
        qi = query_idx.get(r.query)
        if chunker is None or embedder is None or qi is None:
            continue
        relevant = qrels[r.query]
        per_query[chunker]["recall"][embedder][qi] += recall_at_k(r, relevant, args.k)
        per_query[chunker]["mrr"][embedder][qi] += reciprocal_rank(r, relevant)
        per_query[chunker]["ndcg"][embedder][qi] += ndcg_at_k(r, relevant, args.k)
        counts[chunker]["recall"][embedder][qi] += 1
        counts[chunker]["mrr"][embedder][qi] += 1
        counts[chunker]["ndcg"][embedder][qi] += 1

    for r in bm25_persisted:
        base = _strip(r.combination_id, "__bm25")
        if base is None:
            continue
        chunker = chunker_by_base.get(base)
        qi = query_idx.get(r.query)
        if chunker is None or qi is None:
            continue
        relevant = qrels[r.query]
        per_query[chunker]["recall"]["bm25"][qi] += recall_at_k(r, relevant, args.k)
        per_query[chunker]["mrr"]["bm25"][qi] += reciprocal_rank(r, relevant)
        per_query[chunker]["ndcg"]["bm25"][qi] += ndcg_at_k(r, relevant, args.k)
        counts[chunker]["recall"]["bm25"][qi] += 1
        counts[chunker]["mrr"]["bm25"][qi] += 1
        counts[chunker]["ndcg"]["bm25"][qi] += 1

    for c in chunkers:
        for m in ("recall", "mrr", "ndcg"):
            for s in systems:
                cnt = counts[c][m][s]
                per_query[c][m][s] = np.divide(per_query[c][m][s], cnt, out=np.zeros(n_q), where=cnt > 0)
                missing = int((cnt == 0).sum())
                if missing:
                    print(f"WARNING: chunker={c} system={s} missing {missing}/{n_q} queries for {m}")

    rng = np.random.default_rng(args.seed)
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}

    report_lines = [
        "# BM25 vs each dense embedder, PER CHUNKER (no cross-chunker averaging) -- Gold 73-det",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}). "
        "Unlike `bm25_vs_embedder_significance_test_9way.py` (which averages each system's "
        "per-query score across all 4 chunkers first), this script keeps each chunker separate: "
        f"4 independent {len(embedders)}-test families (one per chunker), each Holm-corrected "
        f"independently per metric (alpha={args.alpha}). Resolves paper-results-summary.md "
        "Open item #1.",
        "",
    ]

    any_sig_summary = []
    for chunker in chunkers:
        report_lines.append(f"## Chunker: {chunker}")
        report_lines.append("")
        for metric_key, metric_label in metric_labels.items():
            pairs = []
            for e in embedders:
                diffs = per_query[chunker][metric_key]["bm25"] - per_query[chunker][metric_key][e]
                observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
                pairs.append(("bm25", e, observed, p, ci))
            corrected = holm_correct(pairs, alpha=args.alpha)

            report_lines.append(f"### {metric_label}")
            report_lines.append("")
            report_lines.append("| A | B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |")
            report_lines.append("|---|---|---|---|---|---|---|")
            for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: -x[2]):
                mark = "**yes**" if sig else "no"
                report_lines.append(
                    f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |"
                )
                if metric_key == "recall":
                    any_sig_summary.append((chunker, b, diff, holm_p, sig))
            report_lines.append("")

        report_lines.append(f"### Per-system mean, {chunker} only")
        report_lines.append("")
        report_lines.append("| system | recall@{0} | mrr | ndcg@{0} |".format(args.k))
        report_lines.append("|---|---|---|---|")
        for s in sorted(systems, key=lambda s: -per_query[chunker]["recall"][s].mean()):
            report_lines.append(
                f"| {s} | {per_query[chunker]['recall'][s].mean():.4f} | "
                f"{per_query[chunker]['mrr'][s].mean():.4f} | {per_query[chunker]['ndcg'][s].mean():.4f} |"
            )
        report_lines.append("")

    report_lines.append("## Cross-chunker summary: BM25 vs embedder on recall@10 (Holm-adj p, per chunker)")
    report_lines.append("")
    report_lines.append("| embedder | " + " | ".join(chunkers) + " | aggregate (9way script) |")
    report_lines.append("|---|" + "---|" * (len(chunkers) + 1))
    from collections import defaultdict
    by_embedder = defaultdict(dict)
    for chunker, e, diff, holm_p, sig in any_sig_summary:
        by_embedder[e][chunker] = (diff, holm_p, sig)
    for e in embedders:
        cells = []
        for c in chunkers:
            diff, holm_p, sig = by_embedder[e].get(c, (float("nan"), float("nan"), False))
            mark = "**sig**" if sig else "ns"
            cells.append(f"{diff:+.4f} ({mark})")
        report_lines.append(f"| {e} | " + " | ".join(cells) + " | see aggregate script |")
    report_lines.append("")

    report = "\n".join(report_lines)
    _OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
