from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_lab.chunkers.base import BaseChunker
from rag_lab.registries import chunker_registry

# Thai has no word spaces; fall back paragraph -> line -> space -> char.
_THAI_SEPARATORS = ["\n\n", "\n", " ", ""]


@chunker_registry.register("recursive")
class RecursiveChunker(BaseChunker):
    """Recursively split on a separator hierarchy (LangChain), keeping pieces
    under chunk_size and preferring higher-level separators."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or _THAI_SEPARATORS
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            keep_separator=False,
        )

    def params(self) -> dict[str, Any]:
        return {
            "type": "recursive",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)
