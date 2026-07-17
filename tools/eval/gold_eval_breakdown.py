# -*- coding: utf-8 -*-
"""Per-entity_type breakdown of the Gold chunker eval, with a paired t-test
(fixed_size vs semantic, recall@10) per category -- mirrors the ad-hoc analysis
that produced the program/person finding in docs/chunker-embedder-comparison-log.md,
now generalized to cover the new `thematic` category too.

Reads the persisted per-(query, combo) RetrievalResults directly (same files
run_gold_chunker_eval.py writes) rather than only the already-aggregated
overall report, since the overall report has no entity_type axis.

Run with:
    .venv/Scripts/python.exe tools/eval/gold_eval_breakdown.py
"""
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from rag_lab.metrics import recall_at_k  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

RESULTS_DIR = REPO / "data" / "results" / "gold_chunker_compare"
GOLD_PATH = REPO / "config" / "eval" / "gold_query_set.yaml"
K = 10


def paired_t(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    mean = statistics.mean(diffs)
    if n < 2 or statistics.stdev(diffs) == 0:
        return mean, float("nan")
    se = statistics.stdev(diffs) / (n ** 0.5)
    return mean, mean / se


def main() -> None:
    entries = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))
    entity_type_by_query = {e["query"]: e.get("entity_type", "unknown") for e in entries}
    qrels = {e["query"]: e["relevant_resolution_ids"] for e in entries}

    results = [load_retrieval_result(p) for p in RESULTS_DIR.glob("*.json")]
    combo_ids = sorted({r.combination_id for r in results})
    by_combo_query = {(r.combination_id, r.query): r for r in results}

    # recall@10 per (entity_type, combo) -> list of per-query scores, query-aligned across combos
    recalls: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    queries_by_type: dict[str, list[str]] = defaultdict(list)
    for query in qrels:
        etype = entity_type_by_query.get(query, "unknown")
        queries_by_type[etype].append(query)

    for etype, queries in queries_by_type.items():
        for combo in combo_ids:
            for q in queries:
                r = by_combo_query.get((combo, q))
                score = recall_at_k(r, qrels[q], K) if r is not None else 0.0
                recalls[etype][combo].append(score)

    print(f"{'entity_type':<22} {'n':>4}  " + "  ".join(f"{c.split('__')[1]:>12}" for c in combo_ids))
    for etype in sorted(recalls):
        n = len(queries_by_type[etype])
        row = f"{etype:<22} {n:>4}  "
        row += "  ".join(f"{statistics.mean(recalls[etype][c]):>12.4f}" for c in combo_ids)
        print(row)

    fixed_combo = next(c for c in combo_ids if "fixed_size" in c)
    semantic_combo = next(c for c in combo_ids if "semantic" in c)
    print()
    print(f"paired t-test (recall@{K}, fixed_size - semantic) per entity_type:")
    for etype in sorted(recalls):
        fixed_scores = recalls[etype][fixed_combo]
        semantic_scores = recalls[etype][semantic_combo]
        diffs = [f - s for f, s in zip(fixed_scores, semantic_scores)]
        mean_diff, t = paired_t(diffs)
        wins_fixed = sum(1 for d in diffs if d > 0)
        wins_semantic = sum(1 for d in diffs if d < 0)
        ties = sum(1 for d in diffs if d == 0)
        n = len(diffs)
        print(
            f"  {etype:<22} n={n:<4} mean_diff={mean_diff:+.4f} t={t:+.2f} "
            f"wins fixed/semantic/tie = {wins_fixed}/{wins_semantic}/{ties}"
        )


if __name__ == "__main__":
    main()
