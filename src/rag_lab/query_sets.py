"""Silver and Gold query sets (CONTEXT.md): Query -> relevant resolution_ids.

Silver is free (derived from the corpus itself) but "easy" — title wording
overlaps the document. Gold is hand-written, harder, and truer to real use.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_lab.config import StrategySpec
from rag_lab.query_service import query_indices
from rag_lab.schema import Resolution


@dataclass
class QuerySetEntry:
    query: str
    relevant_resolution_ids: list[str]


def build_silver_query_set(resolutions: list[Resolution]) -> list[QuerySetEntry]:
    return [
        QuerySetEntry(query=r.title, relevant_resolution_ids=[r.resolution_id])
        for r in resolutions
        if r.title
    ]


def load_gold_query_set(path: str | Path) -> list[QuerySetEntry]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        QuerySetEntry(
            query=entry["query"],
            relevant_resolution_ids=entry["relevant_resolution_ids"],
        )
        for entry in data
    ]


def run_query_set(
    query_set: list[QuerySetEntry],
    index_dirs: list[str],
    retriever_spec: StrategySpec,
    k: int,
    results_dir: str | Path,
) -> None:
    """Retrieve every query in the set against every index and persist a
    RetrievalResult per (query, index) pair, so eval can score them later
    without re-indexing (results.py is the shared writer for #4 and here)."""
    for entry in query_set:
        query_indices(entry.query, index_dirs, retriever_spec, k, results_dir=results_dir)
