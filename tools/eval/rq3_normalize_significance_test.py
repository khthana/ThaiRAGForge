"""RQ3 ablation #1: does Thai text normalization help retrieval?

Isolates ONE variable: rag_lab.text_normalize.normalize_thai_text() (digit
conversion + pythainlp.util.normalize() -- see that module's docstring for
the exact rule list). Everything else is held at the `semantic + bge-m3`
combo already used in docs/paper-results-summary.md as this study's winning
system (chunker: semantic, breakpoint_threshold=0.5; embedder: local /
BAAI/bge-m3 -- verified to match data/index/chunker_compare_full's
plain__semantic__local__8aae9bcd manifest.json exactly).

Symmetric normalization: the query is normalized here with the SAME function
used to build the corpus (via the `normalized` loader). Normalizing only the
corpus would make exact-match retrieval (BM25, and therefore `hybrid`, which
fuses BM25) look artificially worse purely from losing lexical overlap
between a digit form used in the corpus and a different one still used in
the query -- an asymmetry artifact, not a normalization effect.

Join-key safety: normalizing a query changes the string that keys its
persisted RetrievalResult (rag_lab.results hashes `result.query`), so
looking baseline and treatment results up by matching stored query text
would silently mismatch. Instead this script always re-derives the correct
lookup key per arm from the ORIGINAL gold query set, in a fixed iteration
order: raw text for the baseline arm, `normalize_thai_text(raw)` for the
treatment arm -- never a dict keyed by whichever text happens to be stored.

Baseline ("raw") arm is NOT rebuilt -- reuses already-persisted dense/hybrid
results for plain__semantic__local__8aae9bcd in
data/results/gold_73det_full_embedder_matrix and data/results/gold_hybrid_73det.

Hybrid-arm comparability: the treatment side calls StrategySpec(type="hybrid")
with no params, i.e. HybridRetriever defaults (rrf_k=60, method=rrf) -- the
same call run_gold_hybrid_eval_9way_new.py used to produce the persisted
gold_hybrid_73det baseline, so both arms share the same fusion params and the
hybrid comparison isn't confounded by an RRF-setting mismatch.

Prerequisite (build the treatment arm's index first):
    PYTHONPATH=src python -m rag_lab.cli run --config config/experiments/rq3_normalize_ablation.yaml

Run with:
    .venv/Scripts/python.exe tools/eval/rq3_normalize_significance_test.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import QuerySetEntry, load_gold_query_set, run_query_set  # noqa: E402
from rag_lab.query_service import discover_indices  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from rag_lab.text_normalize import normalize_thai_text  # noqa: E402
from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402

_BASELINE_DENSE_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_BASELINE_HYBRID_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_BASELINE_COMBO_ID = "plain__semantic__local__8aae9bcd"
_TREATMENT_INDEX_DIR = REPO / "data" / "index" / "rq3_normalize_ablation"
_TREATMENT_RESULTS_DIR = REPO / "data" / "results" / "rq3_normalize_ablation"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "rq3_normalize_significance_test.md"
K = 10
N_BOOT = 10_000
SEED = 42

_METRICS = {
    f"recall@{K}": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    f"ndcg@{K}": lambda r, rel: ndcg_at_k(r, rel, K),
}


def _load_by_query(results_dir: Path, combination_id: str) -> dict[str, object]:
    by_query = {}
    for p in results_dir.glob(f"{combination_id}__*.json"):
        r = load_retrieval_result(p)
        by_query[r.query] = r
    return by_query


def _require_built(index_dir: Path, config_name: str) -> None:
    if not index_dir.is_dir():
        raise RuntimeError(
            f"{index_dir} does not exist -- build it first: PYTHONPATH=src python -m rag_lab.cli run "
            f"--config config/experiments/{config_name}"
        )


def _run_treatment_retrieval(query_set: list[QuerySetEntry], k: int) -> str:
    _require_built(_TREATMENT_INDEX_DIR, "rq3_normalize_ablation.yaml")
    indices = discover_indices(str(_TREATMENT_INDEX_DIR))
    if len(indices) != 1:
        raise RuntimeError(
            f"expected exactly 1 built combo under {_TREATMENT_INDEX_DIR}, found {len(indices)} -- "
            "build it first: PYTHONPATH=src python -m rag_lab.cli run "
            "--config config/experiments/rq3_normalize_ablation.yaml"
        )
    index_dir = indices[0].dir
    normalized_query_set = [
        QuerySetEntry(query=normalize_thai_text(e.query), relevant_resolution_ids=e.relevant_resolution_ids)
        for e in query_set
    ]
    for retriever_type in ("dense", "hybrid"):
        results_dir = _TREATMENT_RESULTS_DIR / retriever_type
        run_query_set(normalized_query_set, [index_dir], StrategySpec(type=retriever_type), k=k, results_dir=str(results_dir))
    return indices[0].combo_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--skip-retrieval", action="store_true", help="reuse already-persisted treatment results only")
    args = parser.parse_args()

    query_set = load_gold_query_set(_GOLD_QUERY_SET)

    if not args.skip_retrieval:
        treatment_combo_id = _run_treatment_retrieval(query_set, args.k)
    else:
        _require_built(_TREATMENT_INDEX_DIR, "rq3_normalize_ablation.yaml")
        treatment_combo_id = discover_indices(str(_TREATMENT_INDEX_DIR))[0].combo_id

    baseline_dense = _load_by_query(_BASELINE_DENSE_DIR, f"{_BASELINE_COMBO_ID}__dense")
    baseline_hybrid = _load_by_query(_BASELINE_HYBRID_DIR, f"{_BASELINE_COMBO_ID}__hybrid")
    treatment_dense = _load_by_query(_TREATMENT_RESULTS_DIR / "dense", f"{treatment_combo_id}__dense")
    treatment_hybrid = _load_by_query(_TREATMENT_RESULTS_DIR / "hybrid", f"{treatment_combo_id}__hybrid")

    n = len(query_set)
    rng = np.random.default_rng(args.seed)
    pairs = []  # (retriever, metric, raw_scores, norm_scores) built up, then bootstrapped
    raw_means: dict[tuple[str, str], float] = {}
    norm_means: dict[tuple[str, str], float] = {}

    for retriever_label, baseline_map, treatment_map in (
        ("dense", baseline_dense, treatment_dense),
        ("hybrid", baseline_hybrid, treatment_hybrid),
    ):
        for metric_name, metric_fn in _METRICS.items():
            raw_scores = np.zeros(n)
            norm_scores = np.zeros(n)
            for i, e in enumerate(query_set):
                b = baseline_map.get(e.query)
                t = treatment_map.get(normalize_thai_text(e.query))
                if b is None or t is None:
                    raise RuntimeError(
                        f"missing result for query index {i} ({retriever_label}): "
                        f"baseline={'ok' if b else 'MISSING'} treatment={'ok' if t else 'MISSING'}"
                    )
                raw_scores[i] = metric_fn(b, e.relevant_resolution_ids)
                norm_scores[i] = metric_fn(t, e.relevant_resolution_ids)
            raw_means[(retriever_label, metric_name)] = float(raw_scores.mean())
            norm_means[(retriever_label, metric_name)] = float(norm_scores.mean())
            diffs = norm_scores - raw_scores
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((retriever_label, metric_name, observed, p, ci))

    corrected = holm_correct(pairs, alpha=args.alpha)

    lines = [
        "# RQ3 ablation 1: Thai text normalization (paired bootstrap, Gold 73-det)",
        "",
        f"mean(normalized) - mean(raw) per (retriever, metric), paired bootstrap over {n} queries "
        f"(n_boot={args.n_boot}, seed={args.seed}), Holm-Bonferroni correction across all "
        f"{len(pairs)} tests (alpha={args.alpha}). Baseline = plain__semantic__local__8aae9bcd "
        "(not rebuilt); treatment = same chunker+embedder, `normalized` loader, query normalized "
        "identically at retrieval time.",
        "",
        "| retriever | metric | mean(raw) | mean(normalized) | diff | 95% CI | raw p | Holm-adj p | significant |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
        mark = "**yes**" if sig else "no"
        lines.append(
            f"| {a} | {b} | {raw_means[(a, b)]:.4f} | {norm_means[(a, b)]:.4f} | {diff:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |"
        )
    lines.append("")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
