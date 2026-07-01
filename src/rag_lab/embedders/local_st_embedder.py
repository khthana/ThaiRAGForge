from __future__ import annotations

import numpy as np

from rag_lab.embedders.base import BaseEmbedder
from rag_lab.registries import embedder_registry


@embedder_registry.register("local")
class LocalSTEmbedder(BaseEmbedder):
    """Local sentence-transformers embedder (default BAAI/bge-m3), GPU if available.

    The model is loaded lazily so importing this module (and the rest of the
    package) stays cheap and unit tests never pull in a 2 GB model. Embeddings
    are L2-normalized so cosine reduces to a dot product.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None

    @property
    def model_id(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
