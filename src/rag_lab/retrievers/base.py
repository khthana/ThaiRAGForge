from __future__ import annotations

from abc import ABC, abstractmethod

from rag_lab.schema import Index, Query, RankedChunk


class BaseRetriever(ABC):
    """Ranks Chunks from a prepared Index for a prepared Query.

    Retrievers receive a prepared Query (vector and/or tokens), never a raw
    string — query preparation happens once in orchestration so retrievers never
    re-embed (ADR-0001) and dense/BM25/hybrid share this signature.
    """

    #: True for a retriever whose retrieve() returns every match rather than
    #: a top-k ranked slice (e.g. EntityLookupRetriever) -- k is ignored in
    #: that case. pipeline.retrieve() reads this to size RetrievalResult.top_k
    #: by actual result count instead of the ignored k.
    exhaustive: bool = False

    #: False for a retriever that serves from an external store (Qdrant) and
    #: reads NOTHING off the Index -- neither `embeddings` nor `chunks`. It is
    #: one flag rather than two because both consequences follow from that one
    #: fact: query_service may skip the ~234MB `embeddings.npy` load (the cost
    #: an engine-served path exists to avoid), and it must REFUSE a row-level
    #: filter/boost, because narrowing the in-process Index cannot narrow what
    #: the engine returns -- the failure would otherwise be silent and wrong,
    #: not loud.
    reads_index_rows: bool = True

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        ...
