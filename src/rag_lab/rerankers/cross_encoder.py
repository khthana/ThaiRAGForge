from __future__ import annotations

import numpy as np

from rag_lab.registries import reranker_registry
from rag_lab.rerankers.base import BaseReranker
from rag_lab.schema import Query, RankedChunk


@reranker_registry.register("cross_encoder")
class CrossEncoderReranker(BaseReranker):
    """Local sentence-transformers CrossEncoder reranker (default
    BAAI/bge-reranker-v2-m3), GPU if available.

    Unlike a bi-encoder (LocalSTEmbedder), a cross-encoder scores a
    (query, candidate) pair jointly in one forward pass instead of comparing
    pre-computed vectors -- more accurate, but only affordable over an
    already-narrowed candidate list, not the full corpus. The model is
    loaded lazily so importing this module stays cheap and unit tests never
    pull in the real model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        model=None,
        batch_size: int = 8,
        max_length: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = model
        self._batch_size = batch_size
        self._max_length = max_length

    @property
    def name(self) -> str:
        return "cross_encoder"

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name, device=self._device, max_length=self._max_length
            )
        return self._model

    def rerank(self, query: Query, candidates: list[RankedChunk], k: int) -> list[RankedChunk]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query.text, c.text) for c in candidates]
        scores = np.asarray(
            model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        )
        order = np.argsort(-scores)[:k]
        return [
            RankedChunk(
                chunk_id=candidates[i].chunk_id,
                resolution_id=candidates[i].resolution_id,
                page=candidates[i].page,
                score=float(scores[i]),
                rank=rank + 1,
                text=candidates[i].text,
            )
            for rank, i in enumerate(order)
        ]

    def release(self) -> None:
        """Free the loaded model. Not load-bearing at the default model size
        (~2.3GB, co-resident with bge-m3 comfortably within a 12GB card) --
        exists for interface parity with LocalSTEmbedder.release() and as an
        escape hatch if a much larger reranker model is configured later."""
        if self._model is None:
            return
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
