"""Central data schema shared by every pipeline stage.

Kept deliberately minimal for the walking skeleton — fields are added only when
a test demands them. ``Chunk.resolution_id`` is load-bearing for evaluation:
relevance is judged at the Resolution level, not the chunk level (ADR-0002).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Resolution(BaseModel):
    """One academic-council resolution (มติ); the atomic source document."""

    resolution_id: str
    source_path: str
    raw_text: str
    source_url: str | None = None
    year: str | None = None
    session: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable sub-segment of a Resolution."""

    chunk_id: str
    resolution_id: str
    text: str
    chunk_index: int
    page: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedChunk(BaseModel):
    """A Chunk placed at a rank by a Retriever, with its score."""

    chunk_id: str
    resolution_id: str
    page: int
    score: float
    rank: int
    text: str


class RetrievalResult(BaseModel):
    """The persisted outcome of one query against one Index artifact."""

    query: str
    combination_id: str
    results: list[RankedChunk]
    top_k: int
    retriever: str


@dataclass
class Index:
    """An in-memory Index artifact: chunks aligned row-for-row to an embeddings
    matrix, plus the metadata identifying how it was built."""

    chunks: list[Chunk]
    embeddings: np.ndarray  # shape (len(chunks), dim)
    meta: dict[str, Any] = field(default_factory=dict)
