from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rag_lab.schema import Index, RankedChunk


class BaseRetriever(ABC):
    """Ranks Chunks from a prepared Index for a prepared query representation.

    Retrievers receive an already-computed query vector, never a raw string —
    query embedding happens once in orchestration so retrievers never re-embed
    (ADR-0001) and dense/BM25/hybrid can share this signature.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def retrieve(
        self, query_vector: np.ndarray, index: Index, k: int
    ) -> list[RankedChunk]:
        ...
