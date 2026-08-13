"""Retrieval backed by a Qdrant collection instead of `index.embeddings`.

Two motivations, added at different times and both still live:

1. **Filtering.** `MetadataFilter` (filters.py) is exact-match-only and can't
   express "chunks belonging to a resolution in this set" -- the shape needed
   once relevance is resolved via an entity join (see
   tools/corpus_prep/build_gold_candidates.py: person/program -> set of
   resolution_ids), because a chunk's own metadata doesn't carry which
   people/programs its resolution relates to. Qdrant does that filtering
   natively inside the query.
2. **Serving.** The project is headed for a real deployment, where the two
   costs that dominate a routed hybrid query are `rank_bm25`'s single-threaded
   Python scoring and the k=n over-fetch -- neither of which is a property of
   the retrieval *method*. Both disappear if the engine holds the vectors.

Deliberately ignores `index.embeddings`/`index.chunks` -- the vectors and
payload live in the Qdrant collection (populated by a separate ingestion step,
see tools/eval/qdrant_pilot_ingest.py), keyed by combo_id so one collection
corresponds to one built Index. `index` is still accepted (the BaseRetriever
signature requires it) but is not read.

**Embedded mode (`path=`) is exact brute force, not ANN**, which is easy to
mistake: `QdrantClient(path=...)` reports an `HnswConfig` in collection info,
but `LocalCollection.search` computes `calculate_distance(...)` over every
vector and `np.argsort`s it. So the 2026-07-16 vertical slice measured a
*different algorithm* from a server, and any ANN-vs-exact question asked in
embedded mode is an instrument that can only answer "identical"
(cf. feedback_a_guards_precondition_biases_its_own_test). Pass `url=` for the
server. Both are supported: `path=` keeps the existing slice and its tests
working, `url=` is what a deployment uses.
"""
from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    SearchParams,
    SparseVector,
)

from rag_lab.registries import retriever_registry
from rag_lab.retrievers import bm25_sparse
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


def _make_client(
    path: str | None, url: str | None, api_key: str | None, timeout: int | None
) -> QdrantClient:
    if (path is None) == (url is None):
        raise ValueError("pass exactly one of path= (embedded, exact) or url= (server)")
    if path is not None:
        return QdrantClient(path=path)
    return QdrantClient(url=url, api_key=api_key, timeout=timeout)


def _to_ranked(points) -> list[RankedChunk]:
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


@retriever_registry.register("qdrant")
class QdrantRetriever(BaseRetriever):
    """Dense (cosine) retrieval from a Qdrant collection.

    `vector_name` is None for the single-unnamed-vector collections the 2026-07-16
    slice built, and a name for the pilot collections, which carry a dense and a
    sparse vector side by side.

    `exact=True` asks the *server* for a full scan. That is what makes the
    ANN question answerable within one engine: exact-vs-ANN then differs only by
    the HNSW traversal, with f32 storage, cosine normalisation and tie-breaking
    held identical -- so a measured gap cannot be blamed on the swap itself.
    """

    def __init__(
        self,
        path: str | None = None,
        collection_name: str = "",
        url: str | None = None,
        api_key: str | None = None,
        vector_name: str | None = None,
        hnsw_ef: int | None = None,
        exact: bool = False,
        timeout: int | None = None,
    ) -> None:
        self._client = _make_client(path, url, api_key, timeout)
        self._collection_name = collection_name
        self._vector_name = vector_name
        self._search_params = (
            SearchParams(hnsw_ef=hnsw_ef, exact=exact)
            if (hnsw_ef is not None or exact)
            else None
        )

    @property
    def name(self) -> str:
        return "qdrant"

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        if query.vector is None:
            raise ValueError("QdrantRetriever requires query.vector")

        points = self._client.query_points(
            collection_name=self._collection_name,
            query=query.vector.tolist(),
            using=self._vector_name,
            query_filter=_build_filter(query.filters),
            search_params=self._search_params,
            limit=k,
            with_payload=True,
        ).points
        return _to_ranked(points)


@retriever_registry.register("qdrant_sparse")
class QdrantSparseRetriever(BaseRetriever):
    """BM25 from a Qdrant sparse vector, scoring identically to `BM25Retriever`.

    The collection stores each chunk's *precomputed* BM25 document weights and
    this sends term counts, so Qdrant's plain dot product is `BM25Okapi.get_scores`
    (see `bm25_sparse` for the algebra and for why the engine's own IDF modifier
    must not be used). `vocab_path` is the token->id map written at ingestion:
    it must be the one built from the same fitted scorer, or every query scores
    against the wrong terms while still returning a plausible ranking.
    """

    def __init__(
        self,
        vocab_path: str,
        path: str | None = None,
        collection_name: str = "",
        url: str | None = None,
        api_key: str | None = None,
        vector_name: str = "bm25",
        timeout: int | None = None,
    ) -> None:
        self._client = _make_client(path, url, api_key, timeout)
        self._collection_name = collection_name
        self._vector_name = vector_name
        self._vocab: dict[str, int] = json.loads(
            Path(vocab_path).read_text(encoding="utf-8")
        )

    @property
    def name(self) -> str:
        return "qdrant_sparse"

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        if query.tokens is None:
            raise ValueError("QdrantSparseRetriever requires query.tokens")

        indices, values = bm25_sparse.query_sparse_vector(query.tokens, self._vocab)
        if not indices:
            return []

        points = self._client.query_points(
            collection_name=self._collection_name,
            query=SparseVector(indices=indices, values=values),
            using=self._vector_name,
            query_filter=_build_filter(query.filters),
            limit=k,
            with_payload=True,
        ).points
        return _to_ranked(points)
