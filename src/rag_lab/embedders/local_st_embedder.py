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

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        model=None,
        batch_size: int = 8,
        max_seq_length: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = model
        # A handful of chunkers (semantic especially -- see
        # docs/chunker-embedder-comparison-log.md) leave a long tail of very
        # large chunks (10-18k chars observed on the real corpus). Sentence-
        # transformers pads a whole batch to its longest member, so one huge
        # chunk sharing a batch with sentence-transformers' default batch_size
        # (32) multiplies that padded length by 32 in the attention
        # activations and reliably OOMs a 12GB card. A small default keeps
        # that multiplier bounded regardless of how long any single chunk is.
        self._batch_size = batch_size
        # Bounding batch_size alone isn't enough for a big model: without
        # flash attention, a single very long chunk needs O(seq_len^2)
        # attention memory regardless of batch size -- observed to OOM
        # Qwen3-Embedding-4B (13.87 GiB for one ~18k-char chunk alone,
        # batch_size=1) even though the same chunk embeds fine on smaller
        # models (jina-v5, bge-m3). None = use the model's own default (no
        # extra truncation) -- only models with this failure mode need it set.
        self._max_seq_length = max_seq_length

    @property
    def model_id(self) -> str:
        return self._model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self._device)
        if self._max_seq_length is not None:
            self._model.max_seq_length = self._max_seq_length
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            # A big model + small batch_size means many batches for a full
            # corpus -- without this, sentence-transformers stays silent for
            # the whole call, which looks indistinguishable from a hang to
            # anything watching for output (same reasoning as pipeline.py's
            # chunking-loop tqdm).
            show_progress_bar=True,
        ).astype(np.float32)

    def release(self) -> None:
        """Drop the loaded model and free its GPU memory.

        Needed when a chunker (semantic) and the axis embedder are both
        local models that would otherwise stay resident in VRAM
        simultaneously -- fine for small combos, but bge-m3 (chunking) +
        Qwen3-Embedding-4B (axis) together nearly saturate a 12GB card
        before a single batch is even encoded. Safe to call on an
        already-unloaded or never-loaded embedder (no-op).
        """
        if self._model is None:
            return
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
