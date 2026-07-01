from __future__ import annotations

import hashlib

import numpy as np

from rag_lab.embedders.base import BaseEmbedder
from rag_lab.registries import embedder_registry


@embedder_registry.register("hashing")
class HashingEmbedder(BaseEmbedder):
    """A cheap, deterministic bag-of-words baseline embedder (no model, no GPU).

    Whitespace tokens are hashed into a fixed-width vector. Useful as a fast
    lexical baseline and for reproducible experiments without loading bge-m3.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @property
    def model_id(self) -> str:
        return f"hashing-{self.dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                vecs[i, int(digest, 16) % self.dim] += 1.0
        return vecs
