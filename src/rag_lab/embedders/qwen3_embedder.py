from __future__ import annotations

import numpy as np

from rag_lab.embedders.local_st_embedder import LocalSTEmbedder
from rag_lab.registries import embedder_registry


@embedder_registry.register("qwen3")
class Qwen3Embedder(LocalSTEmbedder):
    """Qwen3-Embedding-4B, local on GPU.

    Passages are encoded plain, same as bge-m3. Queries use the model's
    built-in "query" prompt (a named prompt baked into the model's own
    sentence-transformers config, not a hand-rolled string prefix) --
    skipping it costs ~1-5% recall per the model card.

    model_id is suffixed so a config swap to plain "local" on the same
    model_name (unlikely but possible) still gets a distinct cache entry,
    same convention as E5Embedder.

    max_seq_length defaults to 2048 (well under the model's own 40,960-token
    ceiling): without flash attention, a single very long chunk needs
    O(seq_len^2) attention memory regardless of batch_size -- observed to
    OOM this model (13.87 GiB for one ~18k-char outlier chunk alone, at
    batch_size=1) on a 12GB card even though the same chunk embeds fine on
    smaller models. Only the long tail of chunks is affected (<0.1% of this
    project's real corpus chunks exceed ~5k chars).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        device: str | None = None,
        model=None,
        batch_size: int = 8,
        max_seq_length: int = 2048,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            model=model,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        )

    @property
    def model_id(self) -> str:
        return f"{self._model_name}+queryprompt"

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        return model.encode(
            [text], prompt_name="query", normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)
