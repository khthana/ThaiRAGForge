"""Score the BM25 lexical baseline (rank_bm25.BM25Okapi over PyThaiNLP
word-level tokens, see src/rag_lab/retrievers/bm25.py) against the Gold
query set, one run per chunker -- BM25 doesn't depend on the embedder axis,
so this reuses the already-built e5-variant index per chunker (its
lexical.json is identical across embedders for the same chunker) rather than
rebuilding anything.

Mirrors run_gold_chunker_eval.py's wiring with retriever_spec swapped from
dense to bm25.

Run with:
    .venv/Scripts/python.exe tools/eval/run_gold_bm25_eval.py
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
        "# Gold query-set eval: BM25 lexical baseline",
        "",
        f"- Query set: Gold 73-deterministic, {n_queries} queries",
        f"- k = {k}, retriever = bm25 (rank_bm25.BM25Okapi, PyThaiNLP newmm word tokens) "
        "-- one run per chunker, embedder axis irrelevant to BM25",
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
        "--embedder-filter", type=str, default="e5",
        help="Pick one embedder variant per chunker (BM25 ignores it; only the "
        "chunker's lexical tokens matter)",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default=str(REPO / "data" / "results" / "gold_bm25_73det"),
    )
    parser.add_argument(
        "--output", type=str,
        default=str(REPO / "data" / "results" / "gold_bm25_73det_report.md"),
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
        query_set, index_dirs, StrategySpec(type="bm25"), k=args.k, results_dir=args.results_dir
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
