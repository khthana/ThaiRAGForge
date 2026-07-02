from __future__ import annotations

from typing import Any

import numpy as np
from pythainlp.tokenize import sent_tokenize

from rag_lab.chunkers.base import BaseChunker
from rag_lab.embedders.base import BaseEmbedder
from rag_lab.embedders.local_st_embedder import LocalSTEmbedder
from rag_lab.registries import chunker_registry


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


@chunker_registry.register("semantic")
class SemanticChunker(BaseChunker):
    """Split into Thai sentences (pythainlp crfcut), then group consecutive
    sentences whose embedding similarity stays >= breakpoint_threshold into
    one chunk; a drop below it starts a new chunk.

    Uses a FIXED embedding model (bge-m3), deliberately decoupled from the
    Embedder axis under test: letting breakpoints shift with whichever
    Embedder a combo happens to use would confound the two axes (ADR-0001 —
    cross-embedder scores aren't comparable either, same root reason).
    """

    _MODEL_NAME = "BAAI/bge-m3"

    def __init__(
        self,
        breakpoint_threshold: float = 0.5,
        engine: str = "crfcut",
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self.breakpoint_threshold = breakpoint_threshold
        self.engine = engine
        # lazy default (LocalSTEmbedder only loads the model on first .embed())
        # keeps import/construction cheap; inject a fake for unit tests.
        self._embedder = embedder if embedder is not None else LocalSTEmbedder(self._MODEL_NAME)

    def params(self) -> dict[str, Any]:
        return {
            "type": "semantic",
            "breakpoint_threshold": self.breakpoint_threshold,
            "engine": self.engine,
            # the fixed model name, NOT self._embedder.model_id — an injected
            # test double must never change this chunker's cache identity.
            "embedding_model": self._MODEL_NAME,
        }

    def _split(self, text: str) -> list[str]:
        sentences = sent_tokenize(text, engine=self.engine)
        if len(sentences) <= 1:
            return list(sentences)

        vectors = self._embedder.embed(sentences)
        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            similarity = _cosine_similarity(vectors[i - 1], vectors[i])
            if similarity < self.breakpoint_threshold:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])
        return ["".join(group) for group in groups]
