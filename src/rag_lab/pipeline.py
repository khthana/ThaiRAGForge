"""Orchestration: build an Index from Resolutions, and run a query against it.

Query embedding happens here, once, before handing a vector to the retriever
(retrievers never re-embed — ADR-0001).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rag_lab.chunkers.base import BaseChunker
from rag_lab.embedders.base import BaseEmbedder
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.schema import Index, Resolution, RetrievalResult


def _cache_key(
    resolutions: list[Resolution], chunker: BaseChunker, embedder: BaseEmbedder
) -> str:
    """Identify an Index-build by its document set, chunker params, and embedder,
    so an unchanged (docset × chunker × embedder) reloads instead of re-embedding."""
    payload = json.dumps(
        {
            "docs": [[r.resolution_id, r.raw_text] for r in resolutions],
            "chunker": chunker.params(),
            "embedder": embedder.model_id,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_index(
    resolutions: list[Resolution],
    chunker: BaseChunker,
    embedder: BaseEmbedder,
    cache_dir: str | Path | None = None,
) -> Index:
    store = ArtifactStore()
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / _cache_key(resolutions, chunker, embedder)
        if (cache_path / "meta.json").exists():
            return store.load(cache_path)

    chunks = []
    for resolution in resolutions:
        chunks.extend(chunker.chunk(resolution))
    embeddings = embedder.embed([c.text for c in chunks])
    meta = {"chunker": chunker.params(), "embedder": embedder.model_id}
    index = Index(chunks=chunks, embeddings=embeddings, meta=meta)

    if cache_path is not None:
        store.save(index, cache_path)
    return index


def retrieve(
    query: str,
    index: Index,
    embedder: BaseEmbedder,
    retriever: BaseRetriever,
    k: int,
    combination_id: str | None = None,
) -> RetrievalResult:
    query_vector = embedder.embed([query])[0]
    ranked = retriever.retrieve(query_vector, index, k)
    if combination_id is None:
        combination_id = (
            f"{index.meta.get('chunker')}|{index.meta.get('embedder')}|{retriever.name}"
        )
    return RetrievalResult(
        query=query,
        combination_id=combination_id,
        results=ranked,
        top_k=k,
        retriever=retriever.name,
    )
