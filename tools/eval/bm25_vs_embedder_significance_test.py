"""Paired bootstrap significance test: BM25 lexical baseline vs each of the 6
dense embedders, Holm-corrected within this 6-comparison family.

Follow-up to run_gold_bm25_eval.py, which found BM25 beats the average dense
embedder on every chunker by 0.10-0.20 recall@10 -- large enough gaps that
they're very likely real, but not yet bootstrap-verified with the same rigor
used for the embedder-vs-embedder comparisons (embedder_significance_test.py).

Same design: resample unit = query (73 of them), each system's per-query
score averaged across the 4 chunker strategies first (BM25 has its own
per-chunker score just like a dense embedder does -- it's just chunker-only,
not chunker x embedder). Paired bootstrap, n_boot=10000, two-sided
percentile p-value, Holm-Bonferroni correction within this family of 6
BM25-vs-embedder tests per metric (kept separate from the original 15
embedder-vs-embedder family -- this is a different question).

Run with:
    .venv/Scripts/python.exe tools/eval/bm25_vs_embedder_significance_test.py
"""
from __future__ import annotations

import argparse
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
_DENSE_RESULTS_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_BM25_RESULTS_DIR = REPO / "data" / "results" / "gold_bm25_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "bm25_vs_embedder_significance_test.md"

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


def build_dense_combo_to_embedder(index_dir: Path) -> dict[str, str]:
    mapping = {}
    for d in sorted(index_dir.iterdir()):
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        mapping[f"{d.name}__dense"] = _embedder_label(manifest["combo"])
    return mapping


def bootstrap_pvalue(diffs: np.ndarray, rng: np.random.Generator, n_boot: int):
    n = len(diffs)
    observed = diffs.mean()
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    p_le = float((boot_means <= 0).mean())
    p_ge = float((boot_means >= 0).mean())
    p_value = min(2 * min(p_le, p_ge), 1.0)
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    return observed, p_value, ci


def holm_correct(pairs: list[tuple], alpha: float) -> list[tuple]:
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][3])
    m = len(pairs)
    adjusted = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        raw_p = pairs[i][3]
        holm_p = (m - rank) * raw_p
        running_max = max(running_max, holm_p)
        adjusted[i] = min(running_max, 1.0)
    return [(a, b, diff, p, ci, adjusted[i], adjusted[i] < alpha) for i, (a, b, diff, p, ci) in enumerate(pairs)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    dense_combo_to_embedder = build_dense_combo_to_embedder(_INDEX_DIR)
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
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
        # combination_id looks like plain__<chunker>__e5__<hash>__bm25 -- any
        # BM25 result counts, one per chunker, four total per query.
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
            missing = (counts[m][s] == 0).sum()
            if missing:
                print(f"WARNING: {s} missing {missing}/{n_q} queries for {m}")

    rng = np.random.default_rng(args.seed)
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}

    report_lines = [
        "# BM25 vs each dense embedder: paired bootstrap significance test (73-det Gold set)",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}). Each "
        "system's per-query score averaged across the 4 chunker strategies first (BM25's "
        "'chunker axis' is its only axis; dense embedders' chunker x embedder cells are "
        "averaged over chunker the same way as in embedder_significance_test.py). "
        f"Holm-Bonferroni correction within this 6-test family per metric (alpha={args.alpha}) "
        "-- kept separate from the original 15-pair embedder-vs-embedder family.",
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
