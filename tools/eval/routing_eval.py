# -*- coding: utf-8 -*-
"""Offline validation of the routing design (src/rag_lab/router.py +
query_service.resolve_index/route_query) against the Gold query set, reusing
retrieval results already persisted by run_gold_chunker_eval.py (e5 combos)
and the embedder-comparison run (local combos) -- no new retrieval/embedding
calls needed. Resolves routes via the real resolve_index(target,
discover_indices(...)) path (not a hardcoded hash table) so this doubles as
an integration check of the production wiring.

Two things to check, both requested by the user before committing to the
routing design:
1. Classification accuracy: does classify_query's live-regex/dictionary
   split actually agree with the hand-labeled entity_type in the yaml, or
   does e.g. a thematic query about "หลักสูตรควบสองปริญญา" leak into the
   program fallback by accident?
2. The unmatched-bucket fallback strategy: single default combo vs RRF
   merge of (person combo, program combo, default combo) -- run both, keep
   whichever wins on this data; RRF is not assumed correct just because it
   sounds safer.

Run with:
    .venv/Scripts/python.exe tools/eval/routing_eval.py
"""
from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from rag_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from rag_lab.query_service import discover_indices, resolve_index  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from rag_lab.router import ROUTE_COMBO, ROUTE_UNMATCHED, RouteTarget, classify_query, rrf_merge  # noqa: E402

GOLD_PATH = REPO / "config" / "eval" / "gold_query_set.yaml"
RESULT_DIRS = [
    REPO / "data" / "results" / "gold_chunker_compare",
    REPO / "data" / "results" / "gold_embedder_compare",
]
INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
K = 10


def main() -> None:
    entries = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))
    qrels = {e["query"]: e["relevant_resolution_ids"] for e in entries}
    entity_type = {e["query"]: e.get("entity_type", "unknown") for e in entries}

    results = []
    for d in RESULT_DIRS:
        results.extend(load_retrieval_result(p) for p in d.glob("*.json"))
    by_combo_query = {(r.combination_id, r.query): r for r in results}

    indices = discover_indices(INDEX_DIR)
    default_combo = resolve_index(ROUTE_COMBO[ROUTE_UNMATCHED], indices).combo_id + "__dense"
    person_combo = resolve_index(ROUTE_COMBO["person"], indices).combo_id + "__dense"
    program_combo = resolve_index(ROUTE_COMBO["program"], indices).combo_id + "__dense"

    # 1. Classification accuracy vs hand-labeled entity_type
    print("== classification vs ground-truth entity_type ==")
    confusion: dict[str, Counter] = defaultdict(Counter)
    predicted_route = {}
    for q, etype in entity_type.items():
        route = classify_query(q)
        predicted_route[q] = route
        confusion[etype][route] += 1
    for etype in sorted(confusion):
        row = ", ".join(f"{route}={n}" for route, n in confusion[etype].most_common())
        print(f"  {etype:<26} n={sum(confusion[etype].values()):<4} -> {row}")

    # 2. Unmatched-bucket fallback: single default combo vs RRF merge
    unmatched_queries = [q for q, r in predicted_route.items() if r == ROUTE_UNMATCHED]
    print(f"\n== unmatched bucket (n={len(unmatched_queries)}): default-only vs RRF merge ==")

    default_recalls, default_mrrs, default_ndcgs = [], [], []
    rrf_recalls, rrf_mrrs, rrf_ndcgs = [], [], []
    for q in unmatched_queries:
        rel = qrels[q]
        default_result = by_combo_query.get((default_combo, q))
        default_recalls.append(recall_at_k(default_result, rel, K) if default_result else 0.0)
        default_mrrs.append(reciprocal_rank(default_result, rel) if default_result else 0.0)
        default_ndcgs.append(ndcg_at_k(default_result, rel, K) if default_result else 0.0)

        candidates = [
            by_combo_query[(combo, q)]
            for combo in (default_combo, person_combo, program_combo)
            if (combo, q) in by_combo_query
        ]
        merged = rrf_merge(candidates, top_k=K)
        rrf_recalls.append(recall_at_k(merged, rel, K))
        rrf_mrrs.append(reciprocal_rank(merged, rel))
        rrf_ndcgs.append(ndcg_at_k(merged, rel, K))

    print(f"  default-only : recall@10={statistics.mean(default_recalls):.4f}  mrr={statistics.mean(default_mrrs):.4f}  ndcg@10={statistics.mean(default_ndcgs):.4f}")
    print(f"  RRF merge    : recall@10={statistics.mean(rrf_recalls):.4f}  mrr={statistics.mean(rrf_mrrs):.4f}  ndcg@10={statistics.mean(rrf_ndcgs):.4f}")

    diffs = [rrf - d for rrf, d in zip(rrf_recalls, default_recalls)]
    n = len(diffs)
    mean_diff = statistics.mean(diffs)
    if n > 1 and statistics.stdev(diffs) > 0:
        t = mean_diff / (statistics.stdev(diffs) / (n ** 0.5))
    else:
        t = float("nan")
    wins_rrf = sum(1 for d in diffs if d > 0)
    wins_default = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)
    print(f"  paired diff (RRF-default), recall@10: mean={mean_diff:+.4f} t={t:+.2f} wins RRF/default/tie = {wins_rrf}/{wins_default}/{ties}")

    # 3. End-to-end: routing system vs a single fixed combo baseline, over all 252
    print("\n== full routing system vs single-combo baselines (all 252 queries) ==")
    baseline_combos = {
        "fixed_size+e5 (naive baseline)": resolve_index(
            RouteTarget("fixed_size", "e5"), indices,
        ).combo_id + "__dense",
        "semantic+e5 (prior 'best overall')": resolve_index(
            RouteTarget("semantic", "e5"), indices,
        ).combo_id + "__dense",
    }
    for label, combo in baseline_combos.items():
        recalls = [recall_at_k(by_combo_query.get((combo, q)), qrels[q], K) if (combo, q) in by_combo_query else 0.0 for q in qrels]
        print(f"  {label:<38} recall@10={statistics.mean(recalls):.4f}")

    route_combo_id = {ROUTE_UNMATCHED: default_combo, "person": person_combo, "program": program_combo}
    routed_recalls = []
    for q in qrels:
        route = predicted_route[q]
        rel = qrels[q]
        if route == ROUTE_UNMATCHED:
            candidates = [
                by_combo_query[(combo, q)]
                for combo in (default_combo, person_combo, program_combo)
                if (combo, q) in by_combo_query
            ]
            result = rrf_merge(candidates, top_k=K)
        else:
            result = by_combo_query.get((route_combo_id[route], q))
        routed_recalls.append(recall_at_k(result, rel, K) if result else 0.0)
    print(f"  {'routed (person/program/RRF-fallback)':<38} recall@10={statistics.mean(routed_recalls):.4f}")


if __name__ == "__main__":
    main()
