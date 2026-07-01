from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_lab.schema import Chunk, Resolution


class BaseChunker(ABC):
    """Splits a Resolution into retrievable Chunks."""

    @abstractmethod
    def chunk(self, resolution: Resolution) -> list[Chunk]:
        ...

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Parameters that identify this chunker for cache keys / manifests."""
