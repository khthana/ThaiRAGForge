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


def _merge_short_fragments(sentences: list[str], min_length: int) -> list[str]:
    """Fold consecutive sentences into a running buffer until it reaches
    min_length, then flush. Any leftover tail joins the previous flushed
    sentence rather than surviving as its own tiny fragment.

    crfcut treats the period in Thai academic-title abbreviations (ผศ. ดร.
    ภ.สถ.ม. วศ.บ.) as a sentence boundary, and has no awareness of embedded
    HTML table markup (</td><td>, common in this corpus's OCR'd committee
    tables) either -- both produce degenerate 2-8 character "sentences" that
    would otherwise become standalone chunks with almost no content, and then
    score anomalously high on cosine similarity for any query sharing that
    fragment (nothing else in the vector to dilute the match).
    """
    if not sentences:
        return []
    merged: list[str] = []
    buffer = ""
    for s in sentences:
        buffer += s
        if len(buffer) >= min_length:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] += buffer
        else:
            merged.append(buffer)
    return merged


@chunker_registry.register("semantic")
class SemanticChunker(BaseChunker):
    """Split into Thai sentences (pythainlp crfcut), merge degenerate
    fragments back together (see _merge_short_fragments), then group
    consecutive sentences whose embedding similarity stays >=
    breakpoint_threshold into one chunk; a drop below it starts a new chunk.

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
        min_sentence_chars: int = 15,
    ) -> None:
        self.breakpoint_threshold = breakpoint_threshold
        self.engine = engine
        self.min_sentence_chars = min_sentence_chars
        # lazy default (LocalSTEmbedder only loads the model on first .embed())
        # keeps import/construction cheap; inject a fake for unit tests.
        self._embedder = embedder if embedder is not None else LocalSTEmbedder(self._MODEL_NAME)

    def params(self) -> dict[str, Any]:
        return {
            "type": "semantic",
            "breakpoint_threshold": self.breakpoint_threshold,
            "engine": self.engine,
            "min_sentence_chars": self.min_sentence_chars,
            # the fixed model name, NOT self._embedder.model_id — an injected
            # test double must never change this chunker's cache identity.
            "embedding_model": self._MODEL_NAME,
        }

    def _split(self, text: str) -> list[str]:
        raw_sentences = sent_tokenize(text, engine=self.engine)
        sentences = _merge_short_fragments(list(raw_sentences), self.min_sentence_chars)
        if len(sentences) <= 1:
            return sentences

        vectors = self._embedder.embed(sentences)
        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            similarity = _cosine_similarity(vectors[i - 1], vectors[i])
            if similarity < self.breakpoint_threshold:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])
        return ["".join(group) for group in groups]

    def release(self) -> None:
        """Free the internal bge-m3 model's GPU memory. Chunking (this
        model) and the axis embedder are two separate model instances that
        would otherwise both stay resident in VRAM -- observed to OOM a
        12GB card once the axis embedder is large (Qwen3-Embedding-4B).

        getattr-guarded: test doubles (BagOfWordsEmbedder etc.) are plain
        duck-typed stand-ins, not BaseEmbedder subclasses, so they don't
        carry the default no-op release().
        """
        release = getattr(self._embedder, "release", None)
        if release is not None:
            release()
