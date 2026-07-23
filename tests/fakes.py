"""Deterministic test doubles for the pipeline.

Keeping these out of `src/` makes clear they are test-only: the real embedder
(bge-m3) is heavy and non-deterministic, so unit tests drive the pipeline with a
hand-computable stand-in instead.
"""
from __future__ import annotations

import hashlib

import numpy as np


class BagOfWordsEmbedder:
    """Whitespace bag-of-words hashed into a fixed-width vector.

    Deterministic (stable hash, not Python's salted ``hash()``) so tests can
    reason about cosine similarity by hand: two texts sharing a token share a
    non-zero dimension. Also a *spy* — it records how many times ``embed`` was
    called, so the cache test can assert the embedder was not re-invoked.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim
        self.call_count = 0
        self.embedded_texts: list[str] = []

    @property
    def model_id(self) -> str:
        return f"bow-{self.dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        self.call_count += 1
        self.embedded_texts.extend(texts)
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.split():
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                vecs[i, int(digest, 16) % self.dim] += 1.0
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class FakeReranker:
    """Deterministic reranker stand-in: scores each candidate by how many
    whitespace tokens it shares with the query, so tests can predict the
    resulting order by hand without loading a real cross-encoder. Also a
    spy -- records the candidate list and k it was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []  # (query_text, n_candidates, k)

    @property
    def name(self) -> str:
        return "fake_reranker"

    def rerank(self, query, candidates, k):
        self.calls.append((query.text, len(candidates), k))
        q_tokens = set(query.text.split())

        def overlap(chunk) -> int:
            return len(q_tokens & set(chunk.text.split()))

        ordered = sorted(candidates, key=overlap, reverse=True)[:k]
        return [
            c.model_copy(update={"score": float(overlap(c)), "rank": rank + 1})
            for rank, c in enumerate(ordered)
        ]
