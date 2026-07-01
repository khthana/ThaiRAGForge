from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_lab.schema import Index


@dataclass
class MetadataFilter:
    """An orthogonal pre-filter: keep only chunks whose metadata matches every
    criterion, then let any retriever rank the narrowed sub-index. Slicing goes
    through Index.select so embeddings and lexical tokens stay aligned."""

    criteria: dict[str, Any]

    def apply(self, index: Index) -> Index:
        rows = [
            i
            for i, chunk in enumerate(index.chunks)
            if all(chunk.metadata.get(key) == value for key, value in self.criteria.items())
        ]
        return index.select(rows)
