from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Turns texts into a dense embedding matrix.

    Implementations must be swappable behind this interface (the real bge-m3
    embedder and any test double share it).
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier used in cache keys and manifests."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float matrix, one row per input text."""
