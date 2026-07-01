"""Persist / load a RetrievalResult.

A standalone reusable writer: Mode B (#4) persists on an interactive query, and
the batch eval step (#9) will call the *same* function so the on-disk format is
identical by construction.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from rag_lab.schema import RetrievalResult


def save_retrieval_result(result: RetrievalResult, results_dir: str | Path) -> Path:
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    query_hash = hashlib.sha256(result.query.encode("utf-8")).hexdigest()[:8]
    path = d / f"{result.combination_id}__{query_hash}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_retrieval_result(path: str | Path) -> RetrievalResult:
    return RetrievalResult.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
