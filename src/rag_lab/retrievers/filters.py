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


@dataclass
class EntityFilter:
    """A pre-filter for list-valued entity metadata (metadata['people']/
    ['programs']/['courses'], each a list per chunk) -- MetadataFilter's `==`
    contract can't express list-membership, so this is a parallel class, not
    a modification (every existing MetadataFilter caller only needs scalar
    equality and stays untouched).

    entities: kind -> accepted canonical values, e.g. {"people": ["ธนา หงษ์สุวรรณ"]}.
    Semantics: OR within a kind (any accepted value tagged on the chunk is
    enough), AND across kinds. Because chunk metadata is resolution-level
    (ADR-0002: build_index merges Resolution.metadata onto every chunk of
    that resolution), an AND across kinds means "this Resolution mentions X
    AND Y" -- not that the same chunk co-mentions both.

    A missing key (an index built without a given tag, e.g. via PersonLoader
    alone) is indistinguishable from a genuinely empty match list -- callers
    must build this against an index whose loader wrote every kind being
    filtered on (see loaders/entity_loader.py's EntityTagLoader)."""

    entities: dict[str, list[str]]

    def apply(self, index: Index) -> Index:
        rows = [i for i, chunk in enumerate(index.chunks) if self._matches(chunk.metadata)]
        return index.select(rows)

    def _matches(self, metadata: dict[str, Any]) -> bool:
        for kind, accepted in self.entities.items():
            tagged = {self._key(kind, entry) for entry in (metadata.get(kind) or [])}
            if not (tagged & set(accepted)):
                return False
        return True

    @staticmethod
    def _key(kind: str, entry: Any) -> str:
        if kind == "people" and isinstance(entry, dict):
            return f"{entry.get('given_name', '')} {entry.get('surname', '')}"
        return entry
