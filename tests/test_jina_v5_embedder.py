"""JinaV5Embedder: unlike Qwen3/bge-m3, both queries and documents need a
named prompt (the model card's "query"/"document" prompt_name values) to
pick the retrieval instruction template. A fake ST-model is injected so the
test never loads the real model.
"""
from __future__ import annotations

import numpy as np

from rag_lab.embedders.jina_v5_embedder import JinaV5Embedder


class _FakeSTModel:
    def __init__(self) -> None:
        self.encode_calls: list[dict] = []

    def encode(
        self,
        texts,
        prompt_name=None,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        self.encode_calls.append({"texts": list(texts), "prompt_name": prompt_name})
        return np.zeros((len(texts), 3), dtype=np.float32)


def test_embed_uses_the_document_prompt():
    fake_model = _FakeSTModel()
    embedder = JinaV5Embedder(model=fake_model)

    embedder.embed(["สวัสดี", "ทดสอบ"])

    assert fake_model.encode_calls == [
        {"texts": ["สวัสดี", "ทดสอบ"], "prompt_name": "document"}
    ]


def test_embed_query_uses_the_query_prompt():
    fake_model = _FakeSTModel()
    embedder = JinaV5Embedder(model=fake_model)

    embedder.embed_query("คำถาม")

    assert fake_model.encode_calls == [{"texts": ["คำถาม"], "prompt_name": "query"}]


def test_model_id_differs_from_plain_local_for_the_same_model_name():
    """Guards the index cache: unprefixed (local) and prompted (jina_v5)
    vectors for the same underlying model must never share a cache entry."""
    from rag_lab.embedders.local_st_embedder import LocalSTEmbedder

    model_name = "jinaai/jina-embeddings-v5-text-small-retrieval"
    plain = LocalSTEmbedder(model_name=model_name, model=_FakeSTModel())
    prompted = JinaV5Embedder(model_name=model_name, model=_FakeSTModel())

    assert plain.model_id != prompted.model_id
