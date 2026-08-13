"""QdrantHybridRetriever: the served counterpart of `HybridRetriever` -- both
arms answered by a Qdrant server, fused by the project's one RRF.

This is the retriever the serving layer reaches for. It exists as a sibling of
`HybridRetriever` rather than a flag on it because that class hardcodes
`DenseRetriever()`/`BM25Retriever()` and sizes its fetch depth from
`len(index.chunks)`, neither of which is available (or wanted) when the rows
live in an engine.

Three things it deliberately does NOT re-decide, because they were measured:

* **`exact=True` on the dense arm.** `data/results/qdrant_pilot.md`: HNSW costs
  -0.0199 recall@10 fused end to end at ef=512 and buys ~6ms against exact's
  17.8ms p50, and exact needs no `ef` retuning when `fetch_depth` moves. ANN is
  still reachable (`exact=False` + `hnsw_ef`), but `hnsw_ef` must then be >= the
  requested depth or the request is malformed rather than approximate.
* **`fetch_depth=200`.** `data/results/routed_fetch_depth_test.md` measured F=200
  against the shipped router: recall@10 +0.0005, all three metrics Holm 1.0000,
  latency 1193.9 -> 475.6 ms p50. That is the query-time default the Streamlit UI
  already sets; `HybridRetriever` keeps `None` so no eval silently re-ranks.
* **The fusion is imported**, never reimplemented -- see `fuse_rrf`.

The collection is resolved LAZILY from the Index it is handed, which is what
lets ONE spec serve all four routed collections through an unmodified
`route_query`: a collection is a copy of an `Index`'s rows and carries its name
(`ArtifactStore.load` stamps `Index.provenance["index_dir"]`, whose directory
name IS the combo id IS the collection name -- see
`tools/eval/qdrant_pilot_ingest.py`, `collection = args.collection or
args.index.name`). Passing `collection_name=` explicitly pins a single one.
"""
from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from rag_lab.registries import retriever_registry
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.retrievers.hybrid import fuse_rrf
from rag_lab.retrievers.qdrant_retriever import (
    QdrantRetriever,
    QdrantSparseRetriever,
    _make_client,
)
from rag_lab.schema import Index, Query, RankedChunk

#: Sidecar root written by tools/eval/qdrant_pilot_ingest.py: one
#: <collection>/vocab.json per collection. src/rag_lab/retrievers/ -> repo root.
_DEFAULT_VOCAB_ROOT = Path(__file__).resolve().parents[3] / "data" / "qdrant"


@retriever_registry.register("qdrant_hybrid")
class QdrantHybridRetriever(BaseRetriever):
    """Dense + sparse from one Qdrant collection, fused with weighted RRF.

    Reads nothing off the Index but its provenance, so `query_service` skips
    loading `embeddings.npy` -- see `BaseRetriever.reads_index_rows`."""

    reads_index_rows = False

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        path: str | None = None,
        collection_name: str | None = None,
        vocab_root: str | None = None,
        vocab_path: str | None = None,
        fetch_depth: int = 200,
        exact: bool = True,
        hnsw_ef: int | None = None,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "bm25",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        timeout: int | None = None,
    ) -> None:
        # `path` (embedded) and `url` (server) are mutually exclusive downstream;
        # passing path= wins so a test can run without a container.
        self.url = None if path else url
        self.path = path
        self.api_key = api_key
        self.collection_name = collection_name
        self.vocab_root = Path(vocab_root) if vocab_root else _DEFAULT_VOCAB_ROOT
        self.vocab_path = vocab_path
        self.fetch_depth = fetch_depth
        self.exact = exact
        self.hnsw_ef = hnsw_ef
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.timeout = timeout
        self._arms: dict[str, tuple[QdrantRetriever, QdrantSparseRetriever]] = {}
        self._client: QdrantClient | None = None

    @property
    def name(self) -> str:
        return "qdrant_hybrid"

    def _shared_client(self) -> QdrantClient:
        """ONE client for both arms and every collection. A client is per store,
        not per collection -- and in embedded mode a second client on the same
        directory raises outright, so sharing is what keeps `path=` usable at
        all. Built lazily: constructing the retriever must not require a
        reachable server (the registry instantiates it before any query, and
        query_service's refusal path only reads `reads_index_rows`)."""
        if self._client is None:
            self._client = _make_client(self.path, self.url, self.api_key, self.timeout)
        return self._client

    def _collection_for(self, index: Index) -> str:
        if self.collection_name:
            return self.collection_name
        index_dir = (index.provenance or {}).get("index_dir")
        if not index_dir:
            raise ValueError(
                "qdrant_hybrid could not resolve a collection: the Index carries no "
                "provenance['index_dir'] (ArtifactStore.load stamps it only when the "
                "index directory holds a manifest.json). Pass collection_name= to pin "
                "one explicitly."
            )
        return Path(index_dir).name

    def _arms_for(self, collection: str) -> tuple[QdrantRetriever, QdrantSparseRetriever]:
        """Cached per collection -- a routed session revisits the same four, and
        each pair re-reads a vocabulary sidecar off disk when built."""
        if collection not in self._arms:
            vocab = (
                Path(self.vocab_path)
                if self.vocab_path
                else self.vocab_root / collection / "vocab.json"
            )
            if not vocab.exists():
                raise FileNotFoundError(
                    f"qdrant_hybrid needs the ingest-time vocabulary sidecar at {vocab}; "
                    f"collection {collection!r} looks un-ingested. Run "
                    f"tools/eval/qdrant_pilot_ingest.py for it."
                )
            common = dict(client=self._shared_client(), collection_name=collection)
            self._arms[collection] = (
                QdrantRetriever(
                    vector_name=self.dense_vector_name,
                    exact=self.exact,
                    hnsw_ef=self.hnsw_ef,
                    **common,
                ),
                QdrantSparseRetriever(
                    vocab_path=str(vocab), vector_name=self.sparse_vector_name, **common
                ),
            )
        return self._arms[collection]

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        dense_arm, sparse_arm = self._arms_for(self._collection_for(index))
        # A depth below k would under-fetch the answer it is asked for; the
        # measured F=200 is a floor on the fetch, never a cap on the send.
        depth = max(self.fetch_depth, k)
        dense = dense_arm.retrieve(query, index, depth)
        sparse = sparse_arm.retrieve(query, index, depth)
        return fuse_rrf(
            dense,
            sparse,
            k,
            rrf_k=self.rrf_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )
