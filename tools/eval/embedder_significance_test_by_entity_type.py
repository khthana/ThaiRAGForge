"""Paired bootstrap significance test across the 6 embedders, split by
entity_type, Holm-corrected within each (entity_type, metric) family.

Follow-up to embedder_significance_test.py (which tested cross-chunker,
cross-entity_type aggregates and found bge-m3 vs Qwen3-Embedding-4B not
significant) and gold_embedder_breakdown_73det.py (which found bge-m3 and
Qwen3 have opposite per-entity_type profiles -- bge-m3 wins person, Qwen3 is
the strongest generalist -- that happen to average out to the same overall
score). This answers the natural next question: are those per-entity_type
differences themselves real, or still noise given the smaller per-type
sample sizes (person=30, program=30, faculty_adjunct_aggregate=13)?

Same design as embedder_significance_test.py (resample unit = query,
per-query score averaged across the 4 chunkers first, paired bootstrap,
n_boot=10000, two-sided percentile p-value) but run separately per
entity_type, with Holm-Bonferroni correction applied within each
(entity_type, metric) group of 15 pairwise tests rather than pooling
everything into one correction.

Run with:
    .venv/Scripts/python.exe tools/eval/embedder_significance_test_by_entity_type.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
_RESULTS_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "embedder_significance_test_by_entity_type.md"

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

    combo_to_embedder = build_combo_to_embedder(_INDEX_DIR)
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}

    entries_raw = yaml.safe_load(_GOLD_QUERY_SET.read_text(encoding="utf-8"))
    entity_type_by_query = {e["query"]: e.get("entity_type", "unknown") for e in entries_raw}
    queries_by_type: dict[str, list[str]] = defaultdict(list)
    for q, et in entity_type_by_query.items():
        queries_by_type[et].append(q)

    persisted = [load_retrieval_result(p) for p in _RESULTS_DIR.glob("*.json")]
    print(f"loaded {len(persisted)} persisted retrieval results")

    embedders = sorted(set(combo_to_embedder.values()))
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}

    report_lines = [
        "# Embedder pairwise significance test, split by entity_type (73-det Gold set)",
        "",
        f"Paired bootstrap (n_boot={args.n_boot}, seed={args.seed}), per-query score "
        "averaged across the 4 chunkers first, resample unit = query WITHIN each "
        "entity_type. Holm-Bonferroni correction applied within each (entity_type, "
        f"metric) group of 15 pairwise tests separately (alpha={args.alpha}). Query "
        f"counts: {', '.join(f'{et}={len(qs)}' for et, qs in sorted(queries_by_type.items()))}.",
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
