from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from rag_lab.registries import retriever_registry
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.schema import Index, Query, RankedChunk


@retriever_registry.register("bm25")
class BM25Retriever(BaseRetriever):
    """Lexical BM25 over the index's per-chunk tokens. Corpus-relative: when run
    over a filtered sub-index it scores against that subset (Index.select carries
    the aligned lexical tokens along)."""

    @property
    def name(self) -> str:
        return "bm25"

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        if index.lexical is None:
            raise ValueError(
                "BM25Retriever needs a lexical index; rebuild the index for BM25"
            )
        if query.tokens is None:
            raise ValueError("BM25Retriever requires query.tokens")
        if not index.chunks:
            return []

        bm25 = BM25Okapi(index.lexical)
        scores = bm25.get_scores(query.tokens)
        order = np.argsort(-scores)[:k]
        return [
            RankedChunk(
                chunk_id=index.chunks[i].chunk_id,
                resolution_id=index.chunks[i].resolution_id,
                page=index.chunks[i].page,
                score=float(scores[i]),
                rank=rank + 1,
                text=index.chunks[i].text,
            )
            for rank, i in enumerate(order)
        ]
