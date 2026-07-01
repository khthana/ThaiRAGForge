from __future__ import annotations

from typing import Any

from rag_lab.chunkers.base import BaseChunker
from rag_lab.registries import chunker_registry


@chunker_registry.register("fixed_size")
class FixedSizeChunker(BaseChunker):
    """Fixed-size sliding window over the text (character units)."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def params(self) -> dict[str, Any]:
        return {
            "type": "fixed_size",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _split(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        pieces: list[str] = []
        start = 0
        while start < len(text):
            pieces.append(text[start : start + self.chunk_size])
            start += step
        return pieces
