"""Exhaustive entity-lookup retrieval: when a query names a known person,
program, or course, return EVERY matching Resolution's chunks instead of a
ranked top-k subset. See docs/research-framework-gap-analysis.md's
entity-lookup next plan for why: recall@10 has a hard, structural ceiling
below 1.0 for "list all X" queries no matter how good the ranking is, and
name collisions in this corpus are rare enough that deterministic
dictionary/regex matching is trustworthy as a bypass, not just a ranking
signal.

Requires an index built with loaders.entity_loader.EntityTagLoader (or
another loader writing the same metadata['people']/['programs']/['courses']
keys) -- see query_service.check_entity_tags_loader for the loud-failure
guard callers should use before reaching this retriever.
"""
from __future__ import annotations

from typing import Callable

from rag_lab.registries import retriever_registry
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.retrievers.filters import EntityFilter
from rag_lab.router import detect_entities
from rag_lab.schema import Index, Query, RankedChunk


@retriever_registry.register("entity_lookup")
class EntityLookupRetriever(BaseRetriever):
    exhaustive = True

    def __init__(self, detector: Callable[[str], dict[str, list[str]]] = detect_entities) -> None:
        self._detect = detector

    @property
    def name(self) -> str:
        return "entity_lookup"

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        """`k` is intentionally ignored -- this retriever is exhaustive
        (see BaseRetriever.exhaustive); it returns every match, not a slice."""
        detected = self._detect(query.text)
        if not detected:
            return []
        matched = EntityFilter(detected).apply(index)

        seen: set[str] = set()
        deduped = []
        for chunk in matched.chunks:
            # One chunk per resolution_id (ADR-0002: relevance is judged at
            # the Resolution level, so a Resolution with many matching
            # chunks shouldn't flood the result with near-duplicate hits --
            # mirrors router.rrf_merge's existing dedup precedent).
            if chunk.resolution_id in seen:
                continue
            seen.add(chunk.resolution_id)
            deduped.append(chunk)

        return [
            RankedChunk(
                chunk_id=c.chunk_id,
                resolution_id=c.resolution_id,
                page=c.page,
                score=1.0,
                rank=i + 1,
                text=c.text,
            )
            for i, c in enumerate(deduped)
        ]
