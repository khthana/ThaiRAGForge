from __future__ import annotations

from typing import Any

from pythainlp.tokenize import sent_tokenize

from rag_lab.chunkers.base import BaseChunker
from rag_lab.registries import chunker_registry


@chunker_registry.register("sentence")
class SentenceChunker(BaseChunker):
    """Split into Thai sentences (pythainlp crfcut), then greedily group whole
    sentences up to chunk_size characters. A sentence is never split; one longer
    than chunk_size simply becomes its own chunk."""

    def __init__(self, chunk_size: int = 512, engine: str = "crfcut") -> None:
        self.chunk_size = chunk_size
        self.engine = engine

    def params(self) -> dict[str, Any]:
        return {"type": "sentence", "chunk_size": self.chunk_size, "engine": self.engine}

    def _split(self, text: str) -> list[str]:
        sentences = sent_tokenize(text, engine=self.engine)
        groups: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) > self.chunk_size:
                groups.append("".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence)
        if current:
            groups.append("".join(current))
        return groups
