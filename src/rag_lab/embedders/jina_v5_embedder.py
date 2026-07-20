from __future__ import annotations

import numpy as np

from rag_lab.embedders.local_st_embedder import LocalSTEmbedder
from rag_lab.registries import embedder_registry


@embedder_registry.register("jina_v5")
class JinaV5Embedder(LocalSTEmbedder):
    """jina-embeddings-v5-text-small-retrieval, local on GPU.

    Unlike bge-m3/Qwen3, both sides need a named prompt here: the model card
    encodes queries with the "query" prompt and documents with the
    "document" prompt (both baked into the model's own sentence-transformers
    config) to pick the right instruction template for the retrieval task --
    plain .encode() with no prompt_name uses neither.

    model_id is suffixed for the same cache-identity reason as E5Embedder /
    Qwen3Embedder.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v5-text-small-retrieval",
        device: str | None = None,
        model=None,
        batch_size: int = 8,
    ) -> None:
        super().__init__(model_name=model_name, device=device, model=model, batch_size=batch_size)

    @property
    def model_id(self) -> str:
        return f"{self._model_name}+queryDocPrompt"

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return model.encode(
            texts,
            prompt_name="document",
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        return model.encode(
            [text], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
