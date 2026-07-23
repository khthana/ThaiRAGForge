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
    reranker: str | None = None


@dataclass
class Query:
    """A prepared query representation handed to a Retriever. Different retrievers
    use different parts: dense uses `vector`, lexical (BM25) uses `tokens`,
    a DB-backed retriever (e.g. QdrantRetriever) uses `filters`."""

    text: str
    vector: np.ndarray | None = None
    tokens: list[str] | None = None
    filters: dict[str, Any] | None = None


@dataclass
class Index:
    """An in-memory Index artifact: chunks aligned row-for-row to an embeddings
    matrix (and, when present, to per-chunk lexical tokens), plus the metadata
    identifying how it was built."""

    chunks: list[Chunk]
    embeddings: np.ndarray  # shape (len(chunks), dim)
    meta: dict[str, Any] = field(default_factory=dict)
    lexical: list[list[str]] | None = None  # per-chunk tokens, row-aligned to chunks

    def select(self, row_indices: list[int]) -> "Index":
        """Return a sub-Index of the given rows, slicing chunks, embeddings and
        lexical tokens by the *same* indices so all arrays stay aligned."""
        rows = list(row_indices)
        return Index(
            chunks=[self.chunks[i] for i in rows],
            embeddings=self.embeddings[rows] if len(self.embeddings) else self.embeddings,
            meta=self.meta,
            lexical=None if self.lexical is None else [self.lexical[i] for i in rows],
        )
