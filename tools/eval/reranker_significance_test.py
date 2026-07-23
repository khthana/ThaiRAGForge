"""Cross-encoder reranker (gap-analysis Tier 3, item 8): does reranking the
top candidates from dense/hybrid retrieval with a cross-encoder
(BAAI/bge-reranker-v2-m3) improve ranking quality over the retriever's own
score alone?

Baseline (no rerank) is NOT rebuilt -- reuses already-persisted dense/hybrid
results for plain__fixed_size__local__ceea7536 in
data/results/gold_73det_full_embedder_matrix and data/results/gold_hybrid_73det.

Treatment retrieves against the SAME index (no new build: reranking is a
query-time-only stage, the corpus/embeddings never change) but widens the
retriever's candidate pool to rerank_pool_size before reranking down to the
final k. Because retrieval is deterministic, the baseline's top-k is exactly
the top-k-by-retriever-score slice of the treatment's wider pool, so the
paired diff (treatment - baseline) isolates the reranker's re-ordering effect
alone -- no separate "wide pool, no rerank" control arm is needed.

A reranker can only re-order what the retriever already found -- it cannot
improve recall@k (the *set* of resolutions retrieved into the widened pool is
unchanged by reranking a subset of it back down to k... unless the reranked
top-k drops a resolution the untouched top-k would have kept, which the
comparison below measures rather than assumes away). It targets ranking
quality (MRR, nDCG@10) where the RQ3 chunk-size ablation found the least
existing effect.

Also measures the reranker's own added latency (rerank() call only, model
load excluded) at rerank_pool_size, since this project reports cost/latency
alongside every quality result (see cost_latency_pareto.py).

Run with (no index build needed first -- reuses the existing baseline index):
    .venv/Scripts/python.exe tools/eval/reranker_significance_test.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from pythainlp.tokenize import word_tokenize  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder, build_reranker, build_retriever  # noqa: E402
from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set, run_query_set  # noqa: E402
from rag_lab.query_service import discover_indices  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from rag_lab.schema import Query  # noqa: E402
from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402

_BASELINE_DENSE_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_BASELINE_HYBRID_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_BASELINE_COMBO_ID = "plain__fixed_size__local__ceea7536"
_BASELINE_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full" / _BASELINE_COMBO_ID
_TREATMENT_RESULTS_DIR = REPO / "data" / "results" / "reranker_significance_test"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "reranker_significance_test.md"
K = 10
RERANK_POOL_SIZE = 50
N_BOOT = 10_000
SEED = 42

_METRICS = {
    f"recall@{K}": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    f"ndcg@{K}": lambda r, rel: ndcg_at_k(r, rel, K),
}


def _load_by_query(results_dir: Path, combination_id: str) -> dict[str, object]:
    return {r.query: r for r in (load_retrieval_result(p) for p in results_dir.glob(f"{combination_id}__*.json"))}


def _require_built(index_dir: Path) -> None:
    if not index_dir.is_dir():
        raise RuntimeError(
            f"{index_dir} does not exist -- this script reuses the existing "
            "chunker_compare_full baseline index and does not build a new one; "
            "run the main chunker/embedder comparison first if it's missing."
        )


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95))}


def _run_treatment_retrieval(k: int, pool_size: int) -> str:
    _require_built(_BASELINE_INDEX_DIR)
    indices = discover_indices(str(_BASELINE_INDEX_DIR.parent))
    [info] = [i for i in indices if i.combo_id == _BASELINE_COMBO_ID]
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    reranker_spec = StrategySpec(type="cross_encoder")
    for retriever_type in ("dense", "hybrid"):
        results_dir = _TREATMENT_RESULTS_DIR / retriever_type
        run_query_set(
            query_set, [info.dir], StrategySpec(type=retriever_type), k=k, results_dir=str(results_dir),
            reranker_spec=reranker_spec, rerank_pool_size=pool_size,
        )
    return info.combo_id


def _measure_rerank_latency(pool_size: int, k: int) -> dict[str, float]:
    """Time only the reranker's rerank() call, model load excluded, over a
    dense-retriever candidate pool for every Gold query. The reranker sees
    only text -- its cost doesn't depend on which retriever produced the
    pool, so measuring once against dense is representative of hybrid too."""
    manifest = json.loads((_BASELINE_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    index = ArtifactStore().load(str(_BASELINE_INDEX_DIR))
    retriever = build_retriever(StrategySpec(type="dense"))
    reranker = build_reranker(StrategySpec(type="cross_encoder"))

    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    durations_ms = []
    for entry in query_set:
        prepared = Query(text=entry.query, vector=embedder.embed_query(entry.query), tokens=word_tokenize(entry.query))
        pool = retriever.retrieve(prepared, index, pool_size)
        t0 = time.perf_counter()
        reranker.rerank(prepared, pool, k)
        durations_ms.append((time.perf_counter() - t0) * 1000)
    reranker.release()
    return _percentiles(durations_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--rerank-pool-size", type=int, default=RERANK_POOL_SIZE)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--skip-retrieval", action="store_true", help="reuse already-persisted treatment results only")
    parser.add_argument("--skip-latency", action="store_true", help="skip the separate latency-measurement pass")
    args = parser.parse_args()

    query_set = load_gold_query_set(_GOLD_QUERY_SET)

    if not args.skip_retrieval:
        treatment_combo_id = _run_treatment_retrieval(args.k, args.rerank_pool_size)
    else:
        _require_built(_BASELINE_INDEX_DIR)
        treatment_combo_id = _BASELINE_COMBO_ID

    reranker_name = "cross_encoder"
    baseline_dense = _load_by_query(_BASELINE_DENSE_DIR, f"{_BASELINE_COMBO_ID}__dense")
    baseline_hybrid = _load_by_query(_BASELINE_HYBRID_DIR, f"{_BASELINE_COMBO_ID}__hybrid")
    treatment_dense = _load_by_query(_TREATMENT_RESULTS_DIR / "dense", f"{treatment_combo_id}__dense__{reranker_name}")
    treatment_hybrid = _load_by_query(_TREATMENT_RESULTS_DIR / "hybrid", f"{treatment_combo_id}__hybrid__{reranker_name}")

    n = len(query_set)
    rng = np.random.default_rng(args.seed)
    pairs = []
    baseline_means: dict[tuple[str, str], float] = {}
    treatment_means: dict[tuple[str, str], float] = {}

    for retriever_label, baseline_map, treatment_map in (
        ("dense", baseline_dense, treatment_dense),
        ("hybrid", baseline_hybrid, treatment_hybrid),
    ):
        for metric_name, metric_fn in _METRICS.items():
            base_scores = np.zeros(n)
            treat_scores = np.zeros(n)
            for i, e in enumerate(query_set):
                b = baseline_map.get(e.query)
                t = treatment_map.get(e.query)
                if b is None or t is None:
                    raise RuntimeError(
                        f"missing result for query index {i} ({retriever_label}): "
                        f"baseline={'ok' if b else 'MISSING'} treatment={'ok' if t else 'MISSING'}"
                    )
                base_scores[i] = metric_fn(b, e.relevant_resolution_ids)
                treat_scores[i] = metric_fn(t, e.relevant_resolution_ids)
            baseline_means[(retriever_label, metric_name)] = float(base_scores.mean())
            treatment_means[(retriever_label, metric_name)] = float(treat_scores.mean())
            diffs = treat_scores - base_scores
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((retriever_label, metric_name, observed, p, ci))

    corrected = holm_correct(pairs, alpha=args.alpha)

    lines = [
        "# Cross-encoder reranker significance test (paired bootstrap, Gold 73-det)",
        "",
        f"mean(reranked) - mean(no-rerank) per (retriever, metric), paired bootstrap over {n} queries "
        f"(n_boot={args.n_boot}, seed={args.seed}), Holm-Bonferroni correction across all "
        f"{len(pairs)} tests (alpha={args.alpha}). Baseline = plain__fixed_size__local__ceea7536 "
        f"at k={args.k} (not rebuilt); treatment = same index, retriever pool widened to "
        f"rerank_pool_size={args.rerank_pool_size}, reranked by BAAI/bge-reranker-v2-m3 down to k={args.k}.",
        "",
        "| retriever | metric | mean(no-rerank) | mean(reranked) | diff | 95% CI | raw p | Holm-adj p | significant |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
        mark = "**yes**" if sig else "no"
        lines.append(
            f"| {a} | {b} | {baseline_means[(a, b)]:.4f} | {treatment_means[(a, b)]:.4f} | {diff:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |"
        )
    lines.append("")

    if not args.skip_latency:
        lat = _measure_rerank_latency(args.rerank_pool_size, args.k)
        lines.append("## Cost / quality trade-off")
        lines.append("")
        lines.append(
            f"Reranker `rerank()` call latency alone (model load excluded), "
            f"rerank_pool_size={args.rerank_pool_size}, over {n} Gold queries:"
        )
        lines.append("")
        lines.append("| p50 (ms) | p95 (ms) | mean (ms) |")
        lines.append("|---|---|---|")
        lines.append(f"| {lat['p50']:.1f} | {lat['p95']:.1f} | {lat['mean']:.1f} |")
        lines.append("")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
