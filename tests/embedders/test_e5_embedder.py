"""Cycle 12 — E5Embedder: multilingual-e5-large needs asymmetric passage:/query:
prefixes (unlike bge-m3/hashing, which encode queries and passages the same way).
A fake ST-model is injected so the test never loads the real 2GB model.
"""
from __future__ import annotations

import numpy as np

from rag_lab.embedders.e5_embedder import E5Embedder


class _FakeSTModel:
    def __init__(self) -> None:
        self.encoded_batches: list[list[str]] = []

    def encode(
        self,
        texts,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        self.encoded_batches.append(list(texts))
        return np.zeros((len(texts), 3), dtype=np.float32)


def test_embed_prefixes_each_text_as_a_passage():
    fake_model = _FakeSTModel()
    embedder = E5Embedder(model=fake_model)

    embedder.embed(["สวัสดี", "ทดสอบ"])

    assert fake_model.encoded_batches == [["passage: สวัสดี", "passage: ทดสอบ"]]


def test_model_id_differs_from_plain_local_for_the_same_model_name():
    """Guards the index cache: unprefixed (local) and prefixed (e5) vectors for
    the same underlying model must never share a cache entry (#6's loader-identity
    collision, recurring for embedders)."""
    from rag_lab.embedders.local_st_embedder import LocalSTEmbedder

    model_name = "intfloat/multilingual-e5-large"
    plain = LocalSTEmbedder(model_name=model_name, model=_FakeSTModel())
    prefixed = E5Embedder(model_name=model_name, model=_FakeSTModel())

    assert plain.model_id != prefixed.model_id


def test_embed_query_prefixes_as_a_query_not_a_passage():
    """The load-bearing case: the inherited embed_query default (embed([text])[0])
    would wrongly apply the passage: prefix. E5 must override it."""
    fake_model = _FakeSTModel()
    embedder = E5Embedder(model=fake_model)

    embedder.embed_query("คำถาม")

    assert fake_model.encoded_batches == [["query: คำถาม"]]
