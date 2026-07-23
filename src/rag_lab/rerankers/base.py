from __future__ import annotations

from abc import ABC, abstractmethod

from rag_lab.schema import Query, RankedChunk


class BaseReranker(ABC):
    """Re-scores and re-orders an already-retrieved candidate list for a
    prepared Query.

    Rerankers receive RankedChunks, never an Index -- they refine a
    Retriever's output, they don't search the corpus themselves (ADR-0001:
    retrieval strategies never re-embed or re-index).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def rerank(self, query: Query, candidates: list[RankedChunk], k: int) -> list[RankedChunk]:
        ...
