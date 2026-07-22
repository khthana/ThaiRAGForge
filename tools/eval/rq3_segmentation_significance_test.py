"""RQ3 ablation #2: does word-boundary-aware chunking beat raw-character
slicing?

Isolates ONE variable: WordAwareFixedSizeChunker (pythainlp `newmm`
segmentation snaps chunk boundaries to word edges) vs FixedSizeChunker
(blind `text[start:start+chunk_size]`, which can and does cut a Thai word in
half since Thai has no inter-word spaces). chunk_size=512, chunk_overlap=50,
embedder=bge-m3 held identical to the plain__fixed_size__local__ceea7536
baseline in data/index/chunker_compare_full (verified against its
manifest.json). Unlike the normalization ablation, the query is untouched
here -- only the corpus side changes -- so results join directly on query
text with no join-key risk.

Also reports each arm's chunk-count and chunk-length distribution (read
straight from chunks.parquet, not the embeddings array, so this stays cheap):
a boundary change that also happens to shift the *typical* chunk size would
confound "segmentation-aware" with "slightly different chunk_size", so this
is reported for transparency rather than assumed away.

Baseline (raw-char) arm is NOT rebuilt -- reuses already-persisted dense/hybrid
results for plain__fixed_size__local__ceea7536 in
data/results/gold_73det_full_embedder_matrix and data/results/gold_hybrid_73det.

Hybrid-arm comparability: the treatment side calls StrategySpec(type="hybrid")
with no params, i.e. HybridRetriever defaults (rrf_k=60, method=rrf) -- the
same call run_gold_hybrid_eval_9way_new.py used to produce the persisted
gold_hybrid_73det baseline, so both arms share the same fusion params.

Prerequisite (build the treatment arm's index first):
    PYTHONPATH=src python -m rag_lab.cli run --config config/experiments/rq3_segmentation_ablation.yaml

Run with:
    .venv/Scripts/python.exe tools/eval/rq3_segmentation_significance_test.py
"""
from __future__ import annotations

import argparse
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
_BASELINE_COMBO_ID = "plain__fixed_size__local__ceea7536"
_BASELINE_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full" / _BASELINE_COMBO_ID
_TREATMENT_INDEX_DIR = REPO / "data" / "index" / "rq3_segmentation_ablation"
_TREATMENT_RESULTS_DIR = REPO / "data" / "results" / "rq3_segmentation_ablation"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "rq3_segmentation_significance_test.md"
K = 10
N_BOOT = 10_000
SEED = 42

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


def _require_built(index_dir: Path, config_name: str) -> None:
    if not index_dir.is_dir():
        raise RuntimeError(
            f"{index_dir} does not exist -- build it first: PYTHONPATH=src python -m rag_lab.cli run "
            f"--config config/experiments/{config_name}"
        )


def _run_treatment_retrieval(k: int) -> str:
    _require_built(_TREATMENT_INDEX_DIR, "rq3_segmentation_ablation.yaml")
    indices = discover_indices(str(_TREATMENT_INDEX_DIR))
    if len(indices) != 1:
        raise RuntimeError(
            f"expected exactly 1 built combo under {_TREATMENT_INDEX_DIR}, found {len(indices)} -- "
            "build it first: PYTHONPATH=src python -m rag_lab.cli run "
            "--config config/experiments/rq3_segmentation_ablation.yaml"
        )
    index_dir = indices[0].dir
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    for retriever_type in ("dense", "hybrid"):
        results_dir = _TREATMENT_RESULTS_DIR / retriever_type
        run_query_set(query_set, [index_dir], StrategySpec(type=retriever_type), k=k, results_dir=str(results_dir))
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
        treatment_combo_id = _run_treatment_retrieval(args.k)
    else:
        _require_built(_TREATMENT_INDEX_DIR, "rq3_segmentation_ablation.yaml")
        treatment_combo_id = discover_indices(str(_TREATMENT_INDEX_DIR))[0].combo_id

    baseline_dense = _load_by_query(_BASELINE_DENSE_DIR, f"{_BASELINE_COMBO_ID}__dense")
    baseline_hybrid = _load_by_query(_BASELINE_HYBRID_DIR, f"{_BASELINE_COMBO_ID}__hybrid")
    treatment_dense = _load_by_query(_TREATMENT_RESULTS_DIR / "dense", f"{treatment_combo_id}__dense")
    treatment_hybrid = _load_by_query(_TREATMENT_RESULTS_DIR / "hybrid", f"{treatment_combo_id}__hybrid")

    n = len(query_set)
    rng = np.random.default_rng(args.seed)
    pairs = []
    raw_means: dict[tuple[str, str], float] = {}
    wa_means: dict[tuple[str, str], float] = {}

    for retriever_label, baseline_map, treatment_map in (
        ("dense", baseline_dense, treatment_dense),
        ("hybrid", baseline_hybrid, treatment_hybrid),
    ):
        for metric_name, metric_fn in _METRICS.items():
            raw_scores = np.zeros(n)
            wa_scores = np.zeros(n)
            for i, e in enumerate(query_set):
                b = baseline_map.get(e.query)
                t = treatment_map.get(e.query)
                if b is None or t is None:
                    raise RuntimeError(
                        f"missing result for query index {i} ({retriever_label}): "
                        f"baseline={'ok' if b else 'MISSING'} treatment={'ok' if t else 'MISSING'}"
                    )
                raw_scores[i] = metric_fn(b, e.relevant_resolution_ids)
                wa_scores[i] = metric_fn(t, e.relevant_resolution_ids)
            raw_means[(retriever_label, metric_name)] = float(raw_scores.mean())
            wa_means[(retriever_label, metric_name)] = float(wa_scores.mean())
            diffs = wa_scores - raw_scores
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((retriever_label, metric_name, observed, p, ci))

    corrected = holm_correct(pairs, alpha=args.alpha)

    lines = [
        "# RQ3 ablation 2: word-aware vs raw-character chunk boundaries (paired bootstrap, Gold 73-det)",
        "",
        f"mean(word-aware) - mean(raw-char) per (retriever, metric), paired bootstrap over {n} queries "
        f"(n_boot={args.n_boot}, seed={args.seed}), Holm-Bonferroni correction across all "
        f"{len(pairs)} tests (alpha={args.alpha}). Baseline = plain__fixed_size__local__ceea7536 "
        "(not rebuilt); treatment = same chunk_size/chunk_overlap/embedder, WordAwareFixedSizeChunker.",
        "",
        "| retriever | metric | mean(raw-char) | mean(word-aware) | diff | 95% CI | raw p | Holm-adj p | significant |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
        mark = "**yes**" if sig else "no"
        lines.append(
            f"| {a} | {b} | {raw_means[(a, b)]:.4f} | {wa_means[(a, b)]:.4f} | {diff:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |"
        )
    lines.append("")

    lines.append("## Chunk-size drift check (is the boundary change secretly a size change?)")
    lines.append("")
    lines.append("| arm | n_chunks | mean_len | median_len |")
    lines.append("|---|---|---|---|")
    treatment_index_dir = Path(discover_indices(str(_TREATMENT_INDEX_DIR))[0].dir)
    for label, d in (("raw-char (baseline)", _BASELINE_INDEX_DIR), ("word-aware (treatment)", treatment_index_dir)):
        stats = _chunk_length_stats(d)
        lines.append(f"| {label} | {stats['n_chunks']} | {stats['mean_len']:.1f} | {stats['median_len']:.1f} |")
    lines.append("")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
