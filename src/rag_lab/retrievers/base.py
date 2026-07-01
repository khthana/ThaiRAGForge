from __future__ import annotations

from abc import ABC, abstractmethod

from rag_lab.schema import Index, Query, RankedChunk


class BaseRetriever(ABC):
    """Ranks Chunks from a prepared Index for a prepared Query.

    Retrievers receive a prepared Query (vector and/or tokens), never a raw
    string — query preparation happens once in orchestration so retrievers never
    re-embed (ADR-0001) and dense/BM25/hybrid share this signature.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        ...
