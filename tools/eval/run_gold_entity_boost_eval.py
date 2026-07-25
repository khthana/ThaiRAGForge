"""Score the entity_boost path (EntityFilter narrow -> hybrid rank, see
query_service.query_indices/query_sets.run_query_set's entity_boost=True)
against the Gold 73-deterministic query set, broken out by entity_type.

This is the user-facing counterpart to run_gold_entity_lookup_eval.py's
exhaustive/unranked numbers. entity_lookup's 0.9291 recall@1000 answers "can
the relevant resolutions be retrieved at all" -- it does not answer "would a
user seeing a top-10 list actually see them," because entity_lookup returns
matches in arbitrary corpus order. This script measures the metric a real
top-k UI actually delivers: detected entities narrow the index first, then
the existing hybrid retriever ranks the narrowed set by the full query text
(so e.g. a faculty_adjunct_aggregate query's "อาจารย์พิเศษ" topic words can
separate the relevant few resolutions from the hundreds that merely mention
the faculty).

Retrieves once at k=30 (comfortably above every cutoff reported) and scores
recall/precision/ndcg at k=[5, 10, 20] plus mrr/map in one pass, so the
report shows exactly how much of the entity_lookup ceiling survives ranking
and top-k truncation. Also reports, per entity_type, the max number of
relevant resolutions for any single query -- recall@10 is mathematically
capped below 1.0 for a query whose gold answer set is larger than 10 (e.g. a
"list every revision of program X" query shape), which is a query-shape
property, not a retrieval defect, and must be read alongside the recall
number rather than silently baked into it.

Run with:
    .venv/Scripts/python.exe tools/eval/run_gold_entity_boost_eval.py
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO / "src"))
import yaml  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from rag_lab.query_sets import load_gold_query_set, run_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

_ENTITY_INDEX_DIR = REPO / "data" / "index" / "entity_tags_full" / "entity_tags__semantic__local__e4fe19d6"
_GOLD_QUERY_SET_PATH = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_RETRIEVE_K = 30
_REPORT_KS = [5, 10, 20]


def render_report(
    per_type: dict[str, dict[str, list[float]]],
    etypes: list[str],
    max_relevant: dict[str, int],
    over_cap: dict[str, int],
    n_queries: int,
) -> str:
    lines = [
        "# Gold query-set eval: entity_boost (EntityFilter narrow -> hybrid rank)",
        "",
        f"- Query set: Gold 73-deterministic, {n_queries} queries",
        "- Index: entity_tags_full (semantic + bge-m3, 70,789 chunks, zero contamination)",
        f"- retriever = hybrid, entity_boost=True; retrieved at k={_RETRIEVE_K}, scored at "
        f"k={_REPORT_KS}",
        "- This is the ranked, top-k-truncated counterpart to "
        "gold_entity_lookup_73det_report.md's exhaustive/unranked recall@1000 -- "
        "the number a real top-k UI actually delivers.",
        "",
        "## Per entity_type",
        "",
    ]
    for k in _REPORT_KS:
        lines.append(f"### recall/precision/ndcg@{k}")
        lines.append("")
        lines.append("| entity_type | n | recall | precision | ndcg | max relevant for 1 query | queries with >k relevant |")
        lines.append("|---|---|---|---|---|---|---|")
        for et in etypes:
            row = per_type[et]
            n = len(row[f"recall@{k}"])
            lines.append(
                f"| {et} | {n} | {sum(row[f'recall@{k}'])/n:.4f} | "
                f"{sum(row[f'precision@{k}'])/n:.4f} | {sum(row[f'ndcg@{k}'])/n:.4f} | "
                f"{max_relevant[et]} | {over_cap[et].get(k, 0)} |"
            )
        lines.append("")

    lines.append("### mrr (rank-cutoff independent)")
    lines.append("")
    lines.append("| entity_type | n | mrr |")
    lines.append("|---|---|---|")
    for et in etypes:
        row = per_type[et]
        n = len(row["mrr"])
        lines.append(f"| {et} | {n} | {sum(row['mrr'])/n:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-query-set", type=str, default=str(_GOLD_QUERY_SET_PATH))
    parser.add_argument("--index-dir", type=str, default=str(_ENTITY_INDEX_DIR))
    parser.add_argument(
        "--results-dir", type=str,
        default=str(REPO / "data" / "results" / "gold_entity_boost_73det"),
    )
    parser.add_argument(
        "--output", type=str,
        default=str(REPO / "data" / "results" / "gold_entity_boost_73det_report.md"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap query count (smoke testing)")
    args = parser.parse_args()

    query_set = load_gold_query_set(args.gold_query_set)
    if args.limit:
        query_set = query_set[: args.limit]
    print(f"gold query set: {len(query_set)} queries")

    entries_raw = yaml.safe_load(Path(args.gold_query_set).read_text(encoding="utf-8"))
    entity_type_by_query = {e["query"]: e.get("entity_type", "unknown") for e in entries_raw}

    t0 = time.time()
    run_query_set(
        query_set, [args.index_dir], StrategySpec(type="hybrid"), _RETRIEVE_K,
        results_dir=args.results_dir, entity_boost=True,
    )
    print(f"retrieval done in {time.time() - t0:.1f}s")

    persisted = [load_retrieval_result(p) for p in Path(args.results_dir).glob("*.json")]
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}

    by_query = {r.query: r for r in persisted}
    queries_by_type: dict[str, list[str]] = defaultdict(list)
    for q, et in entity_type_by_query.items():
        queries_by_type[et].append(q)
    etypes = sorted(queries_by_type)

    per_type: dict[str, dict[str, list[float]]] = {
        et: {**{f"recall@{k}": [] for k in _REPORT_KS},
             **{f"precision@{k}": [] for k in _REPORT_KS},
             **{f"ndcg@{k}": [] for k in _REPORT_KS},
             "mrr": []}
        for et in etypes
    }
    max_relevant: dict[str, int] = {}
    over_cap: dict[str, dict[int, int]] = {et: {k: 0 for k in _REPORT_KS} for et in etypes}

    for et, queries in queries_by_type.items():
        max_relevant[et] = max(len(qrels[q]) for q in queries)
        for q in queries:
            relevant = qrels[q]
            r = by_query.get(q)
            for k in _REPORT_KS:
                if len(relevant) > k:
                    over_cap[et][k] += 1
                if r is None:
                    per_type[et][f"recall@{k}"].append(0.0)
                    per_type[et][f"precision@{k}"].append(0.0)
                    per_type[et][f"ndcg@{k}"].append(0.0)
                else:
                    per_type[et][f"recall@{k}"].append(recall_at_k(r, relevant, k))
                    per_type[et][f"precision@{k}"].append(precision_at_k(r, relevant, k))
                    per_type[et][f"ndcg@{k}"].append(ndcg_at_k(r, relevant, k))
            per_type[et]["mrr"].append(reciprocal_rank(r, relevant) if r is not None else 0.0)

    report = render_report(per_type, etypes, max_relevant, over_cap, len(query_set))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
