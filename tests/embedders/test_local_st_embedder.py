"""LocalSTEmbedder batch_size: sentence-transformers pads a whole batch to
its longest member, so a chunker that leaves very large outliers in the
chunk-length tail (semantic chunker, observed 10-18k char chunks on the real
corpus) can OOM a 12GB card at the sentence-transformers default batch_size
(32) even though most chunks are short. A small default batch_size bounds
that multiplier. A fake ST-model is injected so the test never loads a real
model.
"""
from __future__ import annotations

import numpy as np

from rag_lab.embedders.local_st_embedder import LocalSTEmbedder


class _FakeSTModel:
    def __init__(self) -> None:
        self.encode_calls: list[dict] = []

    def encode(
        self,
        texts,
        batch_size=None,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        self.encode_calls.append({"texts": list(texts), "batch_size": batch_size})
        return np.zeros((len(texts), 3), dtype=np.float32)


def test_embed_passes_a_bounded_default_batch_size():
    fake_model = _FakeSTModel()
    embedder = LocalSTEmbedder(model=fake_model)

    embedder.embed(["a", "b"])

    assert fake_model.encode_calls[0]["batch_size"] == 8


def test_batch_size_is_configurable():
    fake_model = _FakeSTModel()
    embedder = LocalSTEmbedder(model=fake_model, batch_size=2)

    embedder.embed(["a", "b"])

    assert fake_model.encode_calls[0]["batch_size"] == 2


def test_release_drops_the_loaded_model_so_it_reloads_on_next_use():
    fake_model = _FakeSTModel()
    embedder = LocalSTEmbedder(model=fake_model)

    embedder.release()

    assert embedder._model is None


def test_release_on_a_never_loaded_embedder_is_a_no_op():
    embedder = LocalSTEmbedder()  # never loaded (no injected model, .embed() never called)

    embedder.release()  # must not raise

    assert embedder._model is None


def test_max_seq_length_is_applied_to_the_model_when_set():
    fake_model = _FakeSTModel()
    embedder = LocalSTEmbedder(model=fake_model, max_seq_length=2048)

    embedder.embed(["a"])

    assert fake_model.max_seq_length == 2048


def test_max_seq_length_left_untouched_when_not_set():
    """Default None must not stomp a model's own max_seq_length -- only
    embedders with the long-outlier-chunk OOM failure mode (Qwen3) opt in."""
    fake_model = _FakeSTModel()
    fake_model.max_seq_length = "untouched"
    embedder = LocalSTEmbedder(model=fake_model)

    embedder.embed(["a"])

    assert fake_model.max_seq_length == "untouched"
