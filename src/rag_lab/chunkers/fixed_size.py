from __future__ import annotations

import re
from typing import Any

from rag_lab.chunkers.base import BaseChunker
from rag_lab.schema import Chunk, Resolution

_PAGE_MARKER = re.compile(r"^\s*##\s*Page\s+(\d+)\s*$")


def _segment_by_page(text: str) -> list[tuple[int, str]]:
    """Split OCR text into (page_number, page_text) segments on `## Page N`
    marker lines. Text before any marker belongs to page 1."""
    segments: list[tuple[int, str]] = []
    current_page = 1
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            segments.append((current_page, body))

    for line in text.splitlines():
        marker = _PAGE_MARKER.match(line)
        if marker:
            flush()
            current_page = int(marker.group(1))
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return segments


class FixedSizeChunker(BaseChunker):
    """Fixed-size sliding window over the text (character units). `## Page`
    markers are hard boundaries: a window never spans two pages."""

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

    def _windows(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        pieces: list[str] = []
        start = 0
        while start < len(text):
            pieces.append(text[start : start + self.chunk_size])
            start += step
        return pieces

    def chunk(self, resolution: Resolution) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for page, body in _segment_by_page(resolution.raw_text):
            for piece in self._windows(body):
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
