"""Score the Hybrid retriever (RRF fusion of Dense + BM25 over the SAME
index, src/rag_lab/retrievers/hybrid.py) against the Gold query set, across
the full chunker x embedder matrix.

Motivated by the BM25 baseline finding (docs/chunker-embedder-comparison-log.md,
tools/eval/bm25_vs_embedder_significance_test.py): BM25 alone already
statistically ties the best dense embedders (bge-m3, Qwen3-4B). The open
question this answers is whether RRF-fusing BM25 with dense adds anything on
top of either signal alone, or whether the two are too correlated (both
keying off literal entity-name matches in this entity-anchored query set) to
be complementary.

No index rebuild needed -- every already-built combo in
data/index/chunker_compare_full/ carries both embeddings.npy AND
lexical.json (pipeline.build_index always computes both), which is exactly
what HybridRetriever needs from a single Index.

Run with:
    .venv/Scripts/python.exe tools/eval/run_gold_hybrid_eval.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO / "src"))
from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.metrics import evaluate  # noqa: E402
from rag_lab.query_sets import load_gold_query_set, run_query_set  # noqa: E402
from rag_lab.query_service import discover_indices  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

_CHUNKER_COMPARE_DIR = REPO / "data" / "index" / "chunker_compare_full"
_GOLD_QUERY_SET_PATH = REPO / "config" / "eval" / "gold_query_set_73det.yaml"


def render_report(scores: dict[str, dict[str, float]], k: int, n_queries: int) -> str:
    lines = [
        "# Gold query-set eval: Hybrid (RRF, BM25 + Dense) across the full matrix",
        "",
        f"- Query set: Gold 73-deterministic, {n_queries} queries",
        f"- k = {k}, retriever = hybrid (rrf, rrf_k=60 default) over each chunker x embedder combo",
        "",
        "| combination_id | recall@{0} | precision@{0} | mrr | ndcg@{0} | map |".format(k),
        "|---|---|---|---|---|---|",
    ]
    for combo_id in sorted(scores):
        s = scores[combo_id]
        lines.append(
            f"| {combo_id} | {s[f'recall@{k}']:.4f} | {s[f'precision@{k}']:.4f} | "
            f"{s['mrr']:.4f} | {s[f'ndcg@{k}']:.4f} | {s['map']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--gold-query-set", type=str, default=str(_GOLD_QUERY_SET_PATH))
    parser.add_argument("--index-dir", type=str, default=str(_CHUNKER_COMPARE_DIR))
    parser.add_argument(
        "--embedder-filter", type=str, default="",
        help="Substring filter on combo_id; empty = every combo (full 24-combo matrix)",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default=str(REPO / "data" / "results" / "gold_hybrid_73det"),
    )
    parser.add_argument(
        "--output", type=str,
        default=str(REPO / "data" / "results" / "gold_hybrid_73det_report.md"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap query count (smoke testing)")
    args = parser.parse_args()

    query_set = load_gold_query_set(args.gold_query_set)
    if args.limit:
        query_set = query_set[: args.limit]
    print(f"gold query set: {len(query_set)} queries")

    all_indices = discover_indices(args.index_dir)
    index_dirs = [i.dir for i in all_indices if args.embedder_filter in i.combo_id]
    print(f"scoring against {len(index_dirs)} combos: {[Path(d).name for d in index_dirs]}")

    t0 = time.time()
    run_query_set(
        query_set, index_dirs, StrategySpec(type="hybrid"), k=args.k, results_dir=args.results_dir
    )
    print(f"retrieval done in {time.time() - t0:.1f}s")

    persisted = [load_retrieval_result(p) for p in Path(args.results_dir).glob("*.json")]
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    scores = evaluate(persisted, qrels, k=args.k)

    report = render_report(scores, args.k, len(query_set))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
