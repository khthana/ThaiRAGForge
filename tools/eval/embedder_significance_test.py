"""Paired bootstrap significance test across the 6 embedders, Holm-corrected.

Follow-up to the 73-deterministic Gold-set embedder matrix
(docs/chunker-embedder-comparison-log.md, "Re-run บน 73-deterministic"):
that run found bge-m3 and Qwen3-Embedding-4B ~0.005 apart on recall@10 with
the rank order flipped from the 252-query (thematic-diluted) run, and flagged
"still needs a significance test" before either ranking can be cited.

Design (see docs/research-framework-gap-analysis.md Tier 1):
  - Resampling unit is the QUERY (73 of them), not the (query, chunker) pair,
    matching standard IR practice of treating the query set as a sample from a
    population of queries. For each embedder, a query's score is first
    averaged across the 4 chunker strategies -- this matches the per-embedder
    averages already reported in the comparison log.
  - Paired bootstrap (resample query indices with replacement, apply the same
    resampled indices to both systems in a pair) -- 10,000 resamples.
  - Two-sided bootstrap p-value: 2 * min(P(diff_boot <= 0), P(diff_boot >= 0)),
    clipped to 1.0.
  - Holm-Bonferroni correction applied within each metric family separately
    (15 pairwise tests for recall@10, 15 for mrr, 15 for ndcg@10) -- per the
    gap-analysis note that 6 embedders means 15 simultaneous comparisons.

Run with:
    .venv/Scripts/python.exe tools/eval/embedder_significance_test.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
_RESULTS_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "embedder_significance_test.md"

_MODEL_LABELS = {
    "kornwtp/ConGen-BGE_M3-model-phayathaibert": "congen",
    "BAAI/bge-m3": "bge_m3",
    "Thaweewat/jina-embedding-v3-m2v-1024": "m2v",
}


def _embedder_label(combo: dict) -> str:
    etype = combo["embedder"]["type"]
    if etype != "local":
        return etype
    model_name = combo["embedder"]["params"]["model_name"]
    return _MODEL_LABELS.get(model_name, model_name)


def build_combo_to_embedder(index_dir: Path) -> dict[str, str]:
    """combination_id (includes the __dense retriever suffix) -> embedder label."""
    mapping = {}
    for d in sorted(index_dir.iterdir()):
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        mapping[f"{d.name}__dense"] = _embedder_label(manifest["combo"])
    return mapping


def bootstrap_pvalue(diffs_by_query: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, tuple[float, float]]:
    """diffs_by_query: shape (n_queries,), one value per query (already averaged
    across chunkers) for metric_A - metric_B. Returns (observed_mean_diff, p_value, 95%_CI).
    """
    n = len(diffs_by_query)
    observed = diffs_by_query.mean()
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs_by_query[idx].mean(axis=1)
    p_le = float((boot_means <= 0).mean())
    p_ge = float((boot_means >= 0).mean())
    p_value = min(2 * min(p_le, p_ge), 1.0)
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    return observed, p_value, ci


def holm_correct(pairs: list[tuple], alpha: float = 0.05) -> list[tuple]:
    """pairs: list of (a, b, observed_diff, p_value, ci). Returns same list with
    an added holm_p (adjusted) and reject_at_alpha fields, sorted by raw p ascending
    during correction but returned in the original order.
    """
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][3])
    m = len(pairs)
    adjusted = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        raw_p = pairs[i][3]
        holm_p = (m - rank) * raw_p
        running_max = max(running_max, holm_p)
        adjusted[i] = min(running_max, 1.0)
    out = []
    for i, (a, b, diff, p, ci) in enumerate(pairs):
        out.append((a, b, diff, p, ci, adjusted[i], adjusted[i] < alpha))
    return out


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
    queries = list(qrels.keys())
    query_idx = {q: i for i, q in enumerate(queries)}

    persisted = [load_retrieval_result(p) for p in _RESULTS_DIR.glob("*.json")]
    print(f"loaded {len(persisted)} persisted retrieval results, {len(queries)} queries")

    embedders = sorted(set(combo_to_embedder.values()))
    n_q = len(queries)
    # per_embedder[metric][embedder] -> np.array shape (n_q,), averaged across chunkers
    sums = {m: {e: np.zeros(n_q) for e in embedders} for m in ("recall", "mrr", "ndcg")}
    counts = {m: {e: np.zeros(n_q) for e in embedders} for m in ("recall", "mrr", "ndcg")}

    unmatched = set()
    for r in persisted:
        embedder = combo_to_embedder.get(r.combination_id)
        if embedder is None:
            unmatched.add(r.combination_id)
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

    if unmatched:
        print(f"WARNING: {len(unmatched)} combination_ids had no embedder mapping: {sorted(unmatched)[:5]}...")

    per_query = {
        m: {e: np.divide(sums[m][e], counts[m][e], out=np.zeros(n_q), where=counts[m][e] > 0) for e in embedders}
        for m in ("recall", "mrr", "ndcg")
    }
    for m in ("recall", "mrr", "ndcg"):
        for e in embedders:
            missing = (counts[m][e] == 0).sum()
            if missing:
                print(f"WARNING: {e} missing {missing}/{n_q} queries for {m} (scored 0)")

    rng = np.random.default_rng(args.seed)
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}

    report_lines = [
        "# Embedder pairwise significance test (73-deterministic Gold set)",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}), "
        "each embedder's per-query score averaged across the 4 chunker strategies first. "
        f"Holm-Bonferroni correction applied within each metric's 15 pairwise tests (alpha={args.alpha}).",
        "",
    ]

    all_results = {}
    for metric_key, metric_label in metric_labels.items():
        pairs = []
        for a, b in itertools.combinations(embedders, 2):
            diffs = per_query[metric_key][a] - per_query[metric_key][b]
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((a, b, observed, p, ci))
        corrected = holm_correct(pairs, alpha=args.alpha)
        all_results[metric_key] = corrected

        report_lines.append(f"## {metric_label}")
        report_lines.append("")
        report_lines.append("| A | B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |")
        report_lines.append("|---|---|---|---|---|---|---|")
        for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
            mark = "**yes**" if sig else "no"
            report_lines.append(
                f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |"
            )
        report_lines.append("")

    report_lines.append("## Per-embedder mean (for reference, matches the comparison-log table)")
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
