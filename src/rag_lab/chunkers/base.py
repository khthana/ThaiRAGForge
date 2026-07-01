from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_lab.chunkers.pages import segment_by_page
from rag_lab.schema import Chunk, Resolution


class BaseChunker(ABC):
    """Splits a Resolution into retrievable Chunks.

    Template method: the base handles `## Page` boundaries and chunk bookkeeping
    (chunk_index, resolution_id, page); a subclass implements only how to split a
    single page's text into pieces.
    """

    @abstractmethod
    def _split(self, text: str) -> list[str]:
        """Split one page's text into ordered pieces."""

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Parameters that identify this chunker for cache keys / manifests."""

    def chunk(self, resolution: Resolution) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for page, body in segment_by_page(resolution.raw_text):
            for piece in self._split(body):
                chunks.append(
                    Chunk(
                        chunk_id=f"{resolution.resolution_id}:{index}",
                        resolution_id=resolution.resolution_id,
                        text=piece,
                        chunk_index=index,
                        page=page,
                    )
                )
                index += 1
        return chunks
