"""Paired bootstrap significance test: does Hybrid (RRF, BM25+Dense) actually
beat each of its two components alone? Two separate families of 6 tests
(one per embedder), Holm-corrected within each family per metric:
  (1) hybrid_E vs BM25-alone, for each embedder E
  (2) hybrid_E vs dense-alone_E, for each embedder E (paired same embedder)

Follow-up to run_gold_hybrid_eval.py, which found hybrid beats both BM25
alone and dense alone by a wide margin (+0.10 to +0.23 recall@10) for every
embedder except m2v (where hybrid is much WORSE than BM25 alone -- a
suspected RRF failure mode when one signal is near-random). Confirms whether
these large descriptive gaps are real before citing "hybrid helps" in the
paper.

Run with:
    .venv/Scripts/python.exe tools/eval/hybrid_significance_test.py
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
_HYBRID_RESULTS_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "hybrid_significance_test.md"

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
        mapping[d.name] = _embedder_label(manifest["combo"])
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

    base_combo_to_embedder = build_dense_combo_to_embedder(_INDEX_DIR)
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    queries = list(qrels.keys())
    query_idx = {q: i for i, q in enumerate(queries)}
    n_q = len(queries)

    dense_persisted = [load_retrieval_result(p) for p in _DENSE_RESULTS_DIR.glob("*.json")]
    bm25_persisted = [load_retrieval_result(p) for p in _BM25_RESULTS_DIR.glob("*.json")]
    hybrid_persisted = [load_retrieval_result(p) for p in _HYBRID_RESULTS_DIR.glob("*.json")]
    print(f"loaded dense={len(dense_persisted)} bm25={len(bm25_persisted)} hybrid={len(hybrid_persisted)}")

    embedders = sorted(set(base_combo_to_embedder.values()))
    # score[system][embedder-or-'bm25'][query_idx] ; system in {"dense","hybrid","bm25"}
    def new_table():
        return {m: {e: np.zeros(n_q) for e in embedders + ["bm25"]} for m in ("recall", "mrr", "ndcg")}

    def new_counts():
        return {m: {e: np.zeros(n_q) for e in embedders + ["bm25"]} for m in ("recall", "mrr", "ndcg")}

    dense_sums, dense_counts = new_table(), new_counts()
    hybrid_sums, hybrid_counts = new_table(), new_counts()
    bm25_sums, bm25_counts = new_table(), new_counts()

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

    rng = np.random.default_rng(args.seed)
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}

    report_lines = [
        "# Hybrid vs its components: paired bootstrap significance test (73-det Gold set)",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}), each "
        "system's per-query score averaged across the 4 chunkers first. Two separate "
        "6-test families (one per embedder), Holm-corrected independently per metric: "
        f"(1) hybrid_E vs BM25-alone, (2) hybrid_E vs dense-alone_E (alpha={args.alpha}).",
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
