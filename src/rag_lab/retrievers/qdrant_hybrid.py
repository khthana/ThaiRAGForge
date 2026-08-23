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
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse

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

#: Rows sampled per collection to prove it is a copy of THIS build. Point id ==
#: row index at ingest, so a row's identity is checkable directly. Eight is not
#: a confidence level -- the row COUNT does most of the work and this catches the
#: same-count rewrite; it costs one round trip, once per collection per process.
_VERIFY_SAMPLE = 8


@retriever_registry.register("qdrant_hybrid")
class QdrantHybridRetriever(BaseRetriever):
    """Dense + sparse from one Qdrant collection, fused with weighted RRF.

    Reads no VECTORS off the Index, so `query_service` skips loading
    `embeddings.npy` -- see `BaseRetriever.reads_index_rows`, which controls
    exactly that and nothing else. It does read the Index's provenance (for the
    collection name) and, once per collection, its chunk ids: see `_verify`,
    which is what stops a collection left behind by a rebuild from answering.

    **The default url is `127.0.0.1`, not `localhost`, and the difference is
    2.0 seconds per request on this machine** (`data/results/serving_concurrency.md`
    section 3b): `docker run -p 6333:6333` publishes the port on IPv4 only,
    `getaddrinfo("localhost")` returns `::1` first, and the client stack spends
    ~2 s on that address before falling back -- 12.3 ms against 2,050 ms, a
    167x tax on a name. It went unnoticed because every eval script already
    passed `--url http://127.0.0.1:6333` while this class and the UI defaulted
    to `localhost`, so the published Qdrant latencies were measured on a path
    the shipped default did not take. Set `url=` explicitly for a real server;
    a host that is not on IPv4 loopback is a deployment decision, not a
    default."""

    reads_index_rows = False

    def __init__(
        self,
        # 127.0.0.1, NOT localhost, and that is a measured 167x -- see the
        # class docstring.
        url: str = "http://127.0.0.1:6333",
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
        verify_collection: bool = True,
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
        self.verify_collection = verify_collection
        self._arms: dict[str, tuple[QdrantRetriever, QdrantSparseRetriever]] = {}
        self._client: QdrantClient | None = None
        self._verified: set[str] = set()

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

    def _engine_context(self, collection: str, exc: Exception) -> str:
        """One message per cause, because they have different remedies.

        A 404 and a refused connection are both `ApiException`, and reporting
        the first as "the engine is unreachable" sends an operator to restart a
        server that is already running. The collection being absent is the
        rebuild-without-re-ingest case one step further along than the staleness
        guard: nothing to compare, because nothing is there.
        """
        where = self.url or self.path or "<default client>"
        if isinstance(exc, UnexpectedResponse) and getattr(exc, "status_code", None) == 404:
            return (
                f"qdrant collection {collection!r} does not exist on the engine at "
                f"{where}. The index it serves has been built but never ingested (or "
                f"the collection was dropped). Re-ingest it: python "
                f"tools/eval/qdrant_pilot_ingest.py --index <index dir>"
            )
        return (
            f"qdrant_hybrid could not reach the engine at {where} for collection "
            f"{collection!r}: {type(exc).__name__}: {exc}. Check the server is up "
            f"(docker start rag-qdrant) and that the url is reachable; nothing here "
            f"falls back to the in-process retrievers, because a silent switch of "
            f"retriever is a different answer, not a degraded one."
        )

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

    def _verify(self, collection: str, index: Index) -> None:
        """Refuse a collection that is a copy of a DIFFERENT build.

        **This is the engine-side counterpart of the index seal, and it exists
        because the two paths failed differently.** `index_cache._settle` refuses
        a directory whose artifacts disagree with the build its writer sealed;
        nothing made the equivalent claim about a collection. A collection is a
        copy of an `Index`'s rows (`qdrant_pilot_ingest.py`), so **any** index
        rebuild stales it -- and because `_to_ranked` builds results from the
        engine's stored PAYLOAD, a stale collection does not fail, it answers.
        Measured 2026-08-23 on a scratch index and a scratch collection: after a
        rebuild without a re-ingest, one `IndexInfo` and one query returned the
        CURRENT build in-process and the PREVIOUS build through this retriever,
        no error either side. That is this project's signature shape -- two
        artifacts produced at different times, no crash, just a wrong answer.

        Two signals, because neither alone is enough. The row COUNT does most of
        the work and costs one call. It cannot see a rebuild that preserves the
        count (a re-OCR that moves text without moving chunk boundaries), so a
        sample of rows is compared by identity: point id == row index at ingest,
        so row `i`'s `chunk_id` in the payload must be row `i`'s `chunk_id` here.

        Run ONCE per collection per retriever instance, and the serving layer
        caches retrievers -- so this is once per process, not per query. Failing
        is deliberately loud and names the remedy: the seal's trade, one layer
        up, is unavailability over a wrong answer.
        """
        if not self.verify_collection or collection in self._verified:
            return
        n_rows = len(index.chunks)
        client = self._shared_client()
        n_points = client.count(collection_name=collection, exact=True).count
        if n_points != n_rows:
            raise RuntimeError(
                f"qdrant collection {collection!r} holds {n_points:,} points but the "
                f"index it is serving for holds {n_rows:,} rows -- the collection is a "
                f"copy of a different build. Re-ingest it: "
                f"python tools/eval/qdrant_pilot_ingest.py --index "
                f"{(index.provenance or {}).get('index_dir', '<index dir>')}"
            )
        if n_rows:
            step = max(1, n_rows // _VERIFY_SAMPLE)
            ids = sorted({*range(0, n_rows, step), n_rows - 1})[:_VERIFY_SAMPLE]
            found = {
                r.id: (r.payload or {}).get("chunk_id")
                for r in client.retrieve(
                    collection_name=collection, ids=ids, with_payload=True
                )
            }
            for i in ids:
                if found.get(i) != index.chunks[i].chunk_id:
                    raise RuntimeError(
                        f"qdrant collection {collection!r} disagrees with the index it "
                        f"is serving for at row {i}: the collection has chunk_id "
                        f"{found.get(i)!r}, the index has "
                        f"{index.chunks[i].chunk_id!r}. The collection is a copy of a "
                        f"different build; re-ingest it: python "
                        f"tools/eval/qdrant_pilot_ingest.py --index "
                        f"{(index.provenance or {}).get('index_dir', '<index dir>')}"
                    )
        self._verified.add(collection)

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        collection = self._collection_for(index)
        try:
            self._verify(collection, index)
        except ApiException as exc:
            raise RuntimeError(self._engine_context(collection, exc)) from exc
        dense_arm, sparse_arm = self._arms_for(collection)
        # A depth below k would under-fetch the answer it is asked for; the
        # measured F=200 is a floor on the fetch, never a cap on the send.
        depth = max(self.fetch_depth, k)
        try:
            dense = dense_arm.retrieve(query, index, depth)
            sparse = sparse_arm.retrieve(query, index, depth)
        except ApiException as exc:
            # The raw client raises `[WinError 10061] No connection could be
            # made because the target machine actively refused it` -- measured,
            # and it names neither Qdrant, nor the url, nor the collection, nor
            # what to do about it. A served request deserves to know which
            # component is down.
            raise RuntimeError(self._engine_context(collection, exc)) from exc
        return fuse_rrf(
            dense,
            sparse,
            k,
            rrf_k=self.rrf_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )
