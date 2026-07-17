# -*- coding: utf-8 -*-
"""Per-entity_type, per-(chunker, embedder) breakdown of the Gold eval --
mirrors gold_eval_breakdown.py's chunker analysis, but holds chunker fixed and
varies embedder instead, to answer "does embedder choice matter, independent
of the chunker question already answered?"

Reads both result dirs (e5 combos from the original chunker-comparison run,
local combos i.e. bge-m3/ConGen-PhayaThaiBERT from the embedder-comparison
run) since they were persisted separately.

Run with:
    .venv/Scripts/python.exe tools/eval/gold_embedder_breakdown.py
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

RESULT_DIRS = [
    REPO / "data" / "results" / "gold_chunker_compare",
    REPO / "data" / "results" / "gold_embedder_compare",
]
GOLD_PATH = REPO / "config" / "eval" / "gold_query_set.yaml"
K = 10

EMBEDDER_LABEL = {
    "e5": "e5-large",
    "ceea7536": "bge-m3",
    "7cceab27": "phayathaibert-congen",
    "e05efbb8": "bge-m3",
    "d04f22ee": "phayathaibert-congen",
    "8aae9bcd": "bge-m3",
    "87fee2dc": "phayathaibert-congen",
    "bf8b7ebb": "bge-m3",
    "5f573c4f": "phayathaibert-congen",
}


def parse_combo(combo_id: str) -> tuple[str, str]:
    # e.g. plain__fixed_size__e5__0638916c__dense -> ("fixed_size", "e5-large")
    parts = combo_id.split("__")
    chunker = parts[1]
    embedder_key = parts[3] if parts[2] == "local" else parts[2]
    return chunker, EMBEDDER_LABEL[embedder_key]


def main() -> None:
    entries = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))
    qrels = {e["query"]: e["relevant_resolution_ids"] for e in entries}
    entity_type_by_query = {e["query"]: e.get("entity_type", "unknown") for e in entries}

    results = []
    for d in RESULT_DIRS:
        results.extend(load_retrieval_result(p) for p in d.glob("*.json"))
    by_combo_query = {(r.combination_id, r.query): r for r in results}
    combos = sorted({r.combination_id for r in results})

    queries_by_type: dict[str, list[str]] = defaultdict(list)
    for q in qrels:
        queries_by_type[entity_type_by_query.get(q, "unknown")].append(q)

    # recall per (chunker, embedder, entity_type)
    table: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for combo in combos:
        chunker, embedder = parse_combo(combo)
        for etype, queries in queries_by_type.items():
            for q in queries:
                r = by_combo_query.get((combo, q))
                score = recall_at_k(r, qrels[q], K) if r is not None else 0.0
                table[(chunker, embedder)][etype].append(score)

    etypes = sorted(queries_by_type)
    for chunker in ["fixed_size", "recursive", "semantic", "sentence"]:
        print(f"\n== chunker={chunker} ==")
        print(f"{'embedder':<24} " + "  ".join(f"{et:>26}" for et in etypes) + f"  {'overall':>10}")
        for embedder in ["e5-large", "bge-m3", "phayathaibert-congen"]:
            row = table.get((chunker, embedder))
            if not row:
                continue
            cells = "  ".join(f"{statistics.mean(row[et]):>26.4f}" for et in etypes)
            overall = statistics.mean([s for et in etypes for s in row[et]])
            print(f"{embedder:<24} {cells}  {overall:>10.4f}")


if __name__ == "__main__":
    main()
