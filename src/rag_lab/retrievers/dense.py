from __future__ import annotations

import numpy as np

from rag_lab.retrievers.base import BaseRetriever
from rag_lab.schema import Index, RankedChunk


class DenseRetriever(BaseRetriever):
    """Ranks chunks by cosine similarity between the query vector and each
    chunk embedding."""

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(
        self, query_vector: np.ndarray, index: Index, k: int
    ) -> list[RankedChunk]:
        embeddings = index.embeddings
        if len(index.chunks) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        row_norms = np.linalg.norm(embeddings, axis=1)
        denom = row_norms * q_norm
        dots = embeddings @ q
        scores = np.divide(
            dots, denom, out=np.zeros_like(dots, dtype=np.float64), where=denom > 0
        )

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
