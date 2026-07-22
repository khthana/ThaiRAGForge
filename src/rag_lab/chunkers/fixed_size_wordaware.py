from __future__ import annotations

from typing import Any

from pythainlp.tokenize import word_tokenize

from rag_lab.chunkers.base import BaseChunker
from rag_lab.registries import chunker_registry


@chunker_registry.register("fixed_size_wordaware")
class WordAwareFixedSizeChunker(BaseChunker):
    """Same fixed-size sliding-window strategy as FixedSizeChunker, except a
    window is only allowed to end on a word boundary (pythainlp `newmm`
    segmentation) instead of wherever the raw character count happens to
    land -- Thai has no inter-word spaces, so FixedSizeChunker's blind
    `text[start:start+chunk_size]` slice routinely cuts a word in half.

    Isolates RQ3's segmentation ablation (docs/research-framework-gap-analysis.md):
    the ONLY difference from FixedSizeChunker is where a chunk is allowed to
    end; `chunk_size`/`chunk_overlap` keep the same character-budget meaning
    so the two chunkers stay comparable.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 0, engine: str = "newmm") -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.engine = engine

    def params(self) -> dict[str, Any]:
        return {
            "type": "fixed_size_wordaware",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "engine": self.engine,
        }

    def _split(self, text: str) -> list[str]:
        words = word_tokenize(text, engine=self.engine)
        if not words:
            return []

        pieces: list[str] = []
        n = len(words)
        start = 0
        while start < n:
            length = 0
            end = start
            while end < n and length + len(words[end]) <= self.chunk_size:
                length += len(words[end])
                end += 1
            if end == start:
                # a single word longer than chunk_size: it becomes its own
                # chunk rather than being silently dropped or split mid-word.
                end = start + 1
            pieces.append("".join(words[start:end]))
            if end >= n:
                break

            # back off by chunk_overlap worth of characters, in whole words,
            # so the next window's start also lands on a word boundary.
            back = end
            back_len = 0
            while back > start and back_len < self.chunk_overlap:
                back -= 1
                back_len += len(words[back])
            start = back if back > start else end
        return pieces
