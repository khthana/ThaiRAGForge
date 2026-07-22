"""RQ3 ablation #3: does chunk_size (fixed at 512 everywhere else in this
study) actually matter for retrieval quality?

Sweeps FixedSizeChunker at 256 / 512 / 1024 (chunk_overlap=50, embedder=
bge-m3 held fixed throughout). 512 is NOT rebuilt -- it's the already-
persisted plain__fixed_size__local__ceea7536 combo in
data/index/chunker_compare_full (verified against its manifest.json), so
this is a free 3rd point on top of the 2 new builds (256, 1024) from
config/experiments/rq3_chunksize_sweep.yaml. Query is untouched (only the
corpus side changes), so results join directly on query text.

For each (retriever, metric), Holm-corrects across the 3 pairwise
comparisons among {256, 512, 1024} -- same convention as
embedder_matrix_9way.py's per-metric pairwise family.

Also reports each size's actual chunk-count and chunk-length distribution
(read from chunks.parquet only, not the embeddings array) as a sanity check
that the configured chunk_size produced the expected typical chunk length.

Hybrid-arm comparability: sweep arms call StrategySpec(type="hybrid") with no
params, i.e. HybridRetriever defaults (rrf_k=60, method=rrf) -- the same call
run_gold_hybrid_eval_9way_new.py used for the persisted 512-size baseline in
gold_hybrid_73det, so all 3 sizes share the same fusion params.

Prerequisite (build the 256/1024 arms first):
    PYTHONPATH=src python -m rag_lab.cli run --config config/experiments/rq3_chunksize_sweep.yaml

Run with:
    .venv/Scripts/python.exe tools/eval/rq3_chunksize_sweep_report.py
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set, run_query_set  # noqa: E402
from rag_lab.query_service import discover_indices  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402

_BASELINE_DENSE_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_BASELINE_HYBRID_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_BASELINE_512_COMBO_ID = "plain__fixed_size__local__ceea7536"
_BASELINE_512_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full" / _BASELINE_512_COMBO_ID
_SWEEP_INDEX_DIR = REPO / "data" / "index" / "rq3_chunksize_sweep"
_SWEEP_RESULTS_DIR = REPO / "data" / "results" / "rq3_chunksize_sweep"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "rq3_chunksize_sweep_report.md"
K = 10
N_BOOT = 10_000
SEED = 42
SIZES = (256, 512, 1024)

_METRICS = {
    f"recall@{K}": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    f"ndcg@{K}": lambda r, rel: ndcg_at_k(r, rel, K),
}


def _load_by_query(results_dir: Path, combination_id: str) -> dict[str, object]:
    return {r.query: r for r in (load_retrieval_result(p) for p in results_dir.glob(f"{combination_id}__*.json"))}


def _chunk_length_stats(index_dir: Path) -> dict[str, float]:
    lengths = [len(t) for t in pq.read_table(index_dir / "chunks.parquet", columns=["text"]).column("text").to_pylist()]
    return {
        "n_chunks": len(lengths),
        "mean_len": statistics.mean(lengths),
        "median_len": statistics.median(lengths),
    }


def _sweep_indices() -> dict[int, object]:
    """chunk_size -> IndexInfo, for the two built (256, 1024) sweep combos."""
    if not _SWEEP_INDEX_DIR.is_dir():
        return {}
    infos: dict[int, object] = {}
    for info in discover_indices(str(_SWEEP_INDEX_DIR)):
        chunk_size = info.chunker.params.get("chunk_size")
        if chunk_size in SIZES:
            infos[chunk_size] = info
    return infos


def _run_sweep_retrieval(k: int) -> dict[int, str]:
    infos = _sweep_indices()
    missing = set(SIZES) - {512} - set(infos)
    if missing:
        raise RuntimeError(
            f"missing built combo(s) for chunk_size in {sorted(missing)} under {_SWEEP_INDEX_DIR} -- "
            "build them first: PYTHONPATH=src python -m rag_lab.cli run "
            "--config config/experiments/rq3_chunksize_sweep.yaml"
        )
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    combo_ids: dict[int, str] = {}
    for size, info in infos.items():
        combo_ids[size] = info.combo_id
        for retriever_type in ("dense", "hybrid"):
            results_dir = _SWEEP_RESULTS_DIR / retriever_type
            run_query_set(query_set, [info.dir], StrategySpec(type=retriever_type), k=k, results_dir=str(results_dir))
    return combo_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--skip-retrieval", action="store_true", help="reuse already-persisted sweep results only")
    args = parser.parse_args()

    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    n = len(query_set)

    if not args.skip_retrieval:
        combo_ids = _run_sweep_retrieval(args.k)
    else:
        combo_ids = {size: info.combo_id for size, info in _sweep_indices().items()}

    # per-size, per-retriever result lookup
    by_size_retriever: dict[tuple[int, str], dict[str, object]] = {}
    by_size_retriever[(512, "dense")] = _load_by_query(_BASELINE_DENSE_DIR, f"{_BASELINE_512_COMBO_ID}__dense")
    by_size_retriever[(512, "hybrid")] = _load_by_query(_BASELINE_HYBRID_DIR, f"{_BASELINE_512_COMBO_ID}__hybrid")
    for size, combo_id in combo_ids.items():
        by_size_retriever[(size, "dense")] = _load_by_query(_SWEEP_RESULTS_DIR / "dense", f"{combo_id}__dense")
        by_size_retriever[(size, "hybrid")] = _load_by_query(_SWEEP_RESULTS_DIR / "hybrid", f"{combo_id}__hybrid")

    rng = np.random.default_rng(args.seed)
    per_size_mean: dict[tuple[int, str, str], float] = {}
    lines = [
        "# RQ3 ablation 3: chunk_size sweep, FixedSizeChunker (paired bootstrap, Gold 73-det)",
        "",
        f"256 / 1024 built fresh; 512 reused from the existing plain__fixed_size__local__ceea7536 "
        f"combo. Paired bootstrap over {n} queries (n_boot={args.n_boot}, seed={args.seed}); "
        "Holm-Bonferroni correction within each (retriever, metric)'s 3 pairwise comparisons "
        f"among {{256, 512, 1024}} (alpha={args.alpha}).",
        "",
    ]

    for retriever_label in ("dense", "hybrid"):
        for metric_name, metric_fn in _METRICS.items():
            scores: dict[int, np.ndarray] = {}
            for size in SIZES:
                results = by_size_retriever[(size, retriever_label)]
                arr = np.zeros(n)
                for i, e in enumerate(query_set):
                    r = results.get(e.query)
                    if r is None:
                        raise RuntimeError(f"missing result for query index {i}, size={size}, retriever={retriever_label}")
                    arr[i] = metric_fn(r, e.relevant_resolution_ids)
                scores[size] = arr
                per_size_mean[(size, retriever_label, metric_name)] = float(arr.mean())

            pairs = []
            for a, b in itertools.combinations(SIZES, 2):
                diffs = scores[a] - scores[b]
                observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
                pairs.append((a, b, observed, p, ci))
            corrected = holm_correct(pairs, alpha=args.alpha)

            lines.append(f"## {retriever_label} / {metric_name}")
            lines.append("")
            lines.append("| size A | size B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |")
            lines.append("|---|---|---|---|---|---|---|")
            for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
                mark = "**yes**" if sig else "no"
                lines.append(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |")
            lines.append("")
            lines.append("| size | mean |")
            lines.append("|---|---|")
            for size in SIZES:
                lines.append(f"| {size} | {per_size_mean[(size, retriever_label, metric_name)]:.4f} |")
            lines.append("")

    lines.append("## Actual chunk-size stats per arm (read from chunks.parquet)")
    lines.append("")
    lines.append("| chunk_size (config) | n_chunks | mean_len | median_len |")
    lines.append("|---|---|---|---|")
    dirs = {512: _BASELINE_512_INDEX_DIR, **{size: Path(info.dir) for size, info in _sweep_indices().items()}}
    for size in SIZES:
        stats = _chunk_length_stats(dirs[size])
        lines.append(f"| {size} | {stats['n_chunks']} | {stats['mean_len']:.1f} | {stats['median_len']:.1f} |")
    lines.append("")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
