"""Score the 4 chunker strategies (fixed_size/recursive/semantic/sentence, all
e5-large-embedded) against the Silver query set -- each Resolution's own title
used as a query, relevant to itself (CONTEXT.md, ADR-0002). Free (no manual
labeling): this is the first quantitative signal on chunker choice, ahead of
building a hand-labeled Gold set from academic_resolutions/entity_tags/gold_candidates.json.

Reuses query_sets.run_query_set (loads each combo's Index/embedder once, loops
queries against it -- see its docstring for why that matters: the naive
one-reload-per-query path took ~46s/query across 4 combos, i.e. ~36h for the
full Silver set; the batched version takes ~1s/query, i.e. under an hour) and
metrics.evaluate, both already unit-tested. This script is only the corpus-wide
wiring: load real Resolutions -> build Silver set -> run -> score -> report.

Read-only against the corpus and the built indices. Writes retrieval results to
--results-dir and a report to --output (both gitignored, same convention as the
rest of academic_resolutions/entity_tags/).

Run with:
    .venv/Scripts/python.exe tools/eval/run_silver_chunker_eval.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"

sys.path.insert(0, str(REPO / "src"))
from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.loaders import PlainLoader  # noqa: E402
from rag_lab.metrics import evaluate  # noqa: E402
from rag_lab.query_sets import build_silver_query_set, run_query_set  # noqa: E402
from rag_lab.query_service import discover_indices  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

_CHUNKER_COMPARE_DIR = REPO / "data" / "index" / "chunker_compare_full"


def load_resolutions(corpus_root: Path) -> list:
    resolutions = []
    loader = PlainLoader()
    for f in sorted(corpus_root.rglob("*.md")):
        if f.name.endswith(".dup"):
            continue
        resolutions.append(loader.load(str(f)))
    return resolutions


def render_report(scores: dict[str, dict[str, float]], k: int, n_queries: int) -> str:
    lines = [
        "# Silver query-set eval: chunker comparison",
        "",
        f"- Query set: Silver (each Resolution's title, relevant to itself), {n_queries} queries",
        f"- k = {k}, embedder = e5-large (held fixed to isolate the chunker variable)",
        "",
        "| combination_id | recall@{0} | mrr | ndcg@{0} |".format(k),
        "|---|---|---|---|",
    ]
    for combo_id in sorted(scores):
        s = scores[combo_id]
        lines.append(
            f"| {combo_id} | {s[f'recall@{k}']:.4f} | {s['mrr']:.4f} | {s[f'ndcg@{k}']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--index-dir", type=str, default=str(_CHUNKER_COMPARE_DIR),
        help="Directory containing built combo indices (looks for manifest.json per subdir)",
    )
    parser.add_argument(
        "--embedder-filter", type=str, default="e5",
        help="Only run combos whose combo_id contains this substring (isolates the chunker axis)",
    )
    parser.add_argument(
        "--results-dir", type=str,
        default=str(REPO / "data" / "results" / "silver_chunker_compare"),
    )
    parser.add_argument(
        "--output", type=str,
        default=str(REPO / "data" / "results" / "silver_chunker_compare_report.md"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap query count (smoke testing)")
    args = parser.parse_args()

    print("loading resolutions from corpus...")
    resolutions = load_resolutions(CORPUS_ROOT)
    print(f"loaded {len(resolutions)} resolutions")

    query_set = build_silver_query_set(resolutions)
    if args.limit:
        query_set = query_set[: args.limit]
    print(f"silver query set: {len(query_set)} queries")

    all_indices = discover_indices(args.index_dir)
    index_dirs = [i.dir for i in all_indices if args.embedder_filter in i.combo_id]
    print(f"scoring against {len(index_dirs)} combos: {[Path(d).name for d in index_dirs]}")

    t0 = time.time()
    run_query_set(
        query_set, index_dirs, StrategySpec(type="dense"), k=args.k, results_dir=args.results_dir
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
