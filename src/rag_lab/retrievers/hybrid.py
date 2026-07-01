from __future__ import annotations

from rag_lab.registries import retriever_registry
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.dense import DenseRetriever
from rag_lab.schema import Index, Query, RankedChunk


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    top = max(scores.values(), default=0.0)
    if top <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / top for k, v in scores.items()}


@retriever_registry.register("hybrid")
class HybridRetriever(BaseRetriever):
    """Fuse Dense and BM25 rankings. Default `rrf` fuses *ranks* (Reciprocal Rank
    Fusion), sidestepping the incomparable dense-cosine vs BM25 score scales;
    `weighted` fuses max-normalized scores with configurable weights."""

    def __init__(
        self,
        method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> None:
        self.method = method
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._dense = DenseRetriever()
        self._bm25 = BM25Retriever()

    @property
    def name(self) -> str:
        return "hybrid"

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        n = len(index.chunks)
        dense = self._dense.retrieve(query, index, n)
        bm25 = self._bm25.retrieve(query, index, n)
        by_id = {r.chunk_id: r for r in dense}
        by_id.update({r.chunk_id: r for r in bm25})

        if self.method == "rrf":
            fused: dict[str, float] = {}
            for ranking in (dense, bm25):
                for r in ranking:
                    fused[r.chunk_id] = fused.get(r.chunk_id, 0.0) + 1.0 / (
                        self.rrf_k + r.rank
                    )
        elif self.method == "weighted":
            dn = _normalize({r.chunk_id: r.score for r in dense})
            bn = _normalize({r.chunk_id: r.score for r in bm25})
            fused = {
                cid: self.dense_weight * dn.get(cid, 0.0)
                + self.bm25_weight * bn.get(cid, 0.0)
                for cid in by_id
            }
        else:
            raise ValueError(f"unknown hybrid method: {self.method!r}")

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            RankedChunk(
                chunk_id=cid,
                resolution_id=by_id[cid].resolution_id,
                page=by_id[cid].page,
                score=score,
                rank=rank + 1,
                text=by_id[cid].text,
            )
            for rank, (cid, score) in enumerate(ordered)
        ]
