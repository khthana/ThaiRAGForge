"""Qwen3Embedder: documents encode plain (same as bge-m3), queries need the
model's built-in "query" prompt_name -- skipping it costs recall per the
model card. A fake ST-model is injected so the test never loads the real
model.
"""
from __future__ import annotations

import numpy as np

from rag_lab.embedders.qwen3_embedder import Qwen3Embedder


class _FakeSTModel:
    def __init__(self) -> None:
        self.encode_calls: list[dict] = []

    def encode(self, texts, prompt_name=None, normalize_embeddings=True, convert_to_numpy=True):
        self.encode_calls.append({"texts": list(texts), "prompt_name": prompt_name})
        return np.zeros((len(texts), 3), dtype=np.float32)


def test_embed_encodes_passages_plain_with_no_prompt():
    fake_model = _FakeSTModel()
    embedder = Qwen3Embedder(model=fake_model)

    embedder.embed(["สวัสดี", "ทดสอบ"])

    assert fake_model.encode_calls == [{"texts": ["สวัสดี", "ทดสอบ"], "prompt_name": None}]


def test_embed_query_uses_the_built_in_query_prompt():
    fake_model = _FakeSTModel()
    embedder = Qwen3Embedder(model=fake_model)

    embedder.embed_query("คำถาม")

    assert fake_model.encode_calls == [{"texts": ["คำถาม"], "prompt_name": "query"}]


def test_model_id_differs_from_plain_local_for_the_same_model_name():
    """Guards the index cache: unprefixed (local) and query-prompted (qwen3)
    vectors for the same underlying model must never share a cache entry."""
    from rag_lab.embedders.local_st_embedder import LocalSTEmbedder

    model_name = "Qwen/Qwen3-Embedding-4B"
    plain = LocalSTEmbedder(model_name=model_name, model=_FakeSTModel())
    prompted = Qwen3Embedder(model_name=model_name, model=_FakeSTModel())

    assert plain.model_id != prompted.model_id
