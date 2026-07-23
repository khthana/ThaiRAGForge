"""Orchestration: build an Index from Resolutions, and run a query against it.

Query embedding happens here, once, before handing a vector to the retriever
(retrievers never re-embed — ADR-0001).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from pythainlp.tokenize import word_tokenize
from tqdm import tqdm

from rag_lab.chunkers.base import BaseChunker
from rag_lab.embedders.base import BaseEmbedder
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.rerankers.base import BaseReranker
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.schema import Index, Query, Resolution, RetrievalResult


def _cache_key(
    resolutions: list[Resolution], chunker: BaseChunker, embedder: BaseEmbedder
) -> str:
    """Identify an Index-build by its document set, chunker params, and embedder,
    so an unchanged (docset × chunker × embedder) reloads instead of re-embedding."""
    payload = json.dumps(
        {
            # include the loader-produced fields/metadata, not just raw_text, so
            # two loaders that emit different Resolutions get different cache
            # entries (e.g. MetadataLoader vs NERLoader share raw_text but differ
            # in metadata) while identical outputs still share a cache entry.
            "docs": [
                [
                    r.resolution_id,
                    r.raw_text,
                    r.year,
                    r.session,
                    r.title,
                    r.source_url,
                    r.metadata,
                ]
                for r in resolutions
            ],
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

    t0 = time.perf_counter()
    chunks = []
    # A single-combo run (e.g. resuming just one leftover chunker x embedder
    # pair) has no outer per-combo tqdm tick to produce output -- chunking a
    # large corpus can run silent for a long stretch otherwise, which looks
    # indistinguishable from a hang to anything watching for output.
    for resolution in tqdm(resolutions, desc="chunking", leave=False):
        # propagate resolution-level metadata onto each chunk so the MetadataFilter
        # can narrow by year/session/faculty at query time (#6 only fills res.metadata)
        res_meta = {
            "year": resolution.year,
            "session": resolution.session,
            **resolution.metadata,
        }
        for chunk in chunker.chunk(resolution):
            chunk.metadata = {**res_meta, **chunk.metadata}
            chunks.append(chunk)
    t1 = time.perf_counter()
    # Chunking may have loaded its own GPU model (semantic's internal
    # bge-m3); free it before the axis embedder loads its own -- both
    # resident at once can exceed a 12GB card's VRAM for larger embedders.
    chunker.release()
    embeddings = embedder.embed([c.text for c in chunks])
    t2 = time.perf_counter()
    # BM25 tokens, row-aligned. Same silent-stretch risk as the chunking loop
    # above -- pythainlp tokenization over a full corpus's chunks with zero
    # output looks like a hang right as the (also silent-prone) embedding
    # phase finishes.
    lexical = [word_tokenize(c.text) for c in tqdm(chunks, desc="lexical", leave=False)]
    meta = {
        "chunker": chunker.params(),
        "embedder": embedder.model_id,
        # per-phase build timing (#10 Mode A metrics); absent on a cache hit above
        # since nothing was recomputed there.
        "timings": {"chunk_seconds": t1 - t0, "embed_seconds": t2 - t1},
    }
    index = Index(chunks=chunks, embeddings=embeddings, meta=meta, lexical=lexical)

    if cache_path is not None:
        store.save(index, cache_path)
    return index


def retrieve(
    query: str,
    index: Index,
    embedder: BaseEmbedder,
    retriever: BaseRetriever,
    k: int,
    *,
    reranker: BaseReranker | None = None,
    rerank_pool_size: int | None = None,
    combination_id: str | None = None,
) -> RetrievalResult:
    # An exhaustive retriever (e.g. EntityLookupRetriever) never reads
    # query.vector, so skip the embed call -- otherwise every entity-lookup
    # query would pay a real embedder's (possibly GPU) cost for nothing.
    exhaustive = getattr(retriever, "exhaustive", False)
    prepared = Query(
        text=query,
        vector=None if exhaustive else embedder.embed_query(query),
        tokens=word_tokenize(query),
    )
    # When reranking, fetch a wider candidate pool from the retriever so the
    # reranker has room to fix mistakes -- reranking only k already-good
    # candidates gives it little to work with. Both DenseRetriever and
    # BM25Retriever always score the full corpus and only slice at k, so
    # widening the pool costs nothing extra.
    pool_k = rerank_pool_size if (reranker is not None and rerank_pool_size) else k
    ranked = retriever.retrieve(prepared, index, pool_k)
    if reranker is not None:
        ranked = reranker.rerank(prepared, ranked, k)
    if combination_id is None:
        combination_id = (
            f"{index.meta.get('chunker')}|{index.meta.get('embedder')}|{retriever.name}"
        )
    return RetrievalResult(
        query=query,
        combination_id=combination_id,
        results=ranked,
        top_k=len(ranked) if exhaustive else k,
        retriever=retriever.name,
        reranker=reranker.name if reranker is not None else None,
    )
