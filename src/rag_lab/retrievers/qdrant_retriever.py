"""Dense retrieval backed by a Qdrant collection instead of index.embeddings.

Motivation: MetadataFilter (filters.py) is exact-match-only and can't express
"chunks belonging to a resolution in this set" -- the shape needed once
relevance is resolved via an entity join (see tools/corpus_prep/build_gold_candidates.py:
person/program -> set of resolution_ids), because a chunk's own metadata
doesn't carry which people/programs its resolution relates to (the four entity
loaders each replace `metadata` with one key; no index has all of them). Qdrant
does that filtering natively in the query itself instead of pre-slicing an
already-loaded Index.

Deliberately ignores `index.embeddings`/`index.chunks` -- the vectors and
payload live in the Qdrant collection (populated by a separate ingestion step,
see tools/eval/build_qdrant_person_slice.py), keyed by combo_id so one
collection corresponds to one built Index. `index` is still accepted (the
BaseRetriever signature requires it) but only its `meta['combo_id']`-adjacent
identity is implied by which collection this retriever was constructed to
point at -- it does not read index.chunks/index.embeddings at all.

Embedded local mode (QdrantClient(path=...), no server process) matches this
project's no-service ethos -- same reasoning as everywhere else in rag_lab
staying dependency-light (ADR-0001).
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from rag_lab.registries import retriever_registry
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.schema import Index, Query, RankedChunk


def _build_filter(filters: dict | None) -> Filter | None:
    """filters = {"resolution_id_in": [...]} -> a Qdrant match-any Filter.
    The only filter shape the vertical slice needs; extend here as more
    entity-anchored filter kinds (program, faculty) come online."""
    if not filters:
        return None
    conditions = []
    if resolution_ids := filters.get("resolution_id_in"):
        conditions.append(FieldCondition(key="resolution_id", match=MatchAny(any=resolution_ids)))
    return Filter(must=conditions) if conditions else None


@retriever_registry.register("qdrant")
class QdrantRetriever(BaseRetriever):
    def __init__(self, path: str, collection_name: str) -> None:
        self._client = QdrantClient(path=path)
        self._collection_name = collection_name

    @property
    def name(self) -> str:
        return "qdrant"

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        if query.vector is None:
            raise ValueError("QdrantRetriever requires query.vector")

        points = self._client.query_points(
            collection_name=self._collection_name,
            query=query.vector.tolist(),
            query_filter=_build_filter(query.filters),
            limit=k,
        ).points

        return [
            RankedChunk(
                chunk_id=p.payload["chunk_id"],
                resolution_id=p.payload["resolution_id"],
                page=p.payload["page"],
                score=float(p.score),
                rank=rank + 1,
                text=p.payload["text"],
            )
            for rank, p in enumerate(points)
        ]
