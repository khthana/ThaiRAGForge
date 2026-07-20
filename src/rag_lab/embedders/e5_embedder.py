from __future__ import annotations

import numpy as np

from rag_lab.embedders.local_st_embedder import LocalSTEmbedder
from rag_lab.registries import embedder_registry


@embedder_registry.register("e5")
class E5Embedder(LocalSTEmbedder):
    """multilingual-e5-large, local on GPU, with e5's required passage:/query:
    prefixes (unlike bge-m3, e5 encodes passages and queries differently).

    model_id is suffixed so a config swap between "local" and "e5" on the same
    underlying model name still gets distinct cache entries (unprefixed vs
    prefixed vectors are not interchangeable — same collision class as #6's
    loader-identity fix).
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | None = None,
        model=None,
        batch_size: int = 8,
    ) -> None:
        super().__init__(model_name=model_name, device=device, model=model, batch_size=batch_size)

    @property
    def model_id(self) -> str:
        return f"{self._model_name}+e5prefix"

    def embed(self, texts: list[str]) -> np.ndarray:
        return super().embed([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return super().embed([f"query: {text}"])[0]
