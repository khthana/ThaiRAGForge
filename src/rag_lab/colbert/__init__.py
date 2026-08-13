"""Late-interaction (ColBERT) encoding and scoring.

Kept out of `embedders/` on purpose: `BaseEmbedder` returns one row per text and
the whole `Index` is row-aligned on that (audit check I1). ColBERT is many
vectors per chunk and needs its own artifact shape.
"""
from rag_lab.colbert.encoder import MODEL_NAME, ColbertConfig, ColbertEncoder
from rag_lab.colbert.scoring import maxsim, maxsim_reference, offsets_from_lengths

__all__ = [
    "MODEL_NAME",
    "ColbertConfig",
    "ColbertEncoder",
    "maxsim",
    "maxsim_reference",
    "offsets_from_lengths",
]
