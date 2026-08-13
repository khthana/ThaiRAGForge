"""QdrantHybridRetriever: both arms answered by one Qdrant collection, fused
with the project's one RRF, against an embedded local store (no server).

Embedded mode is exact brute force rather than HNSW, which is exactly why it is
usable here: everything under test (collection resolution, fetch depth, the
fusion, the vocabulary sidecar) is independent of the search algorithm, and the
ANN question this retriever *doesn't* re-decide was already measured on a real
server (`data/results/qdrant_pilot.md`).
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from rag_lab.config import StrategySpec
from rag_lab.factory import build_retriever
from rag_lab.retrievers import QdrantHybridRetriever
from rag_lab.retrievers.hybrid import fuse_rrf
from rag_lab.schema import Index, Query

COLLECTION = "coll"

#: Dense order against the query [1, 0] is c0 > c1 > c2. The sparse arm is
#: seeded to say the *opposite* about c2, so a returned ranking that is merely
#: the dense one would be indistinguishable from a broken sparse arm; fusion
#: has to lift c2 to the top for the test to pass.
_POINTS = [
    {"chunk_id": "c0", "resolution_id": "r0", "dense": [1.0, 0.0], "sparse": ([1], [0.5])},
    {"chunk_id": "c1", "resolution_id": "r1", "dense": [0.9, 0.1], "sparse": ([1], [0.4])},
    {"chunk_id": "c2", "resolution_id": "r2", "dense": [0.8, 0.2], "sparse": ([0], [5.0])},
]

#: Term ids the seeded sparse vectors use. "ก" is the only term the query sends.
_VOCAB = {"ก": 0, "ข": 1}


def _seed(path: str) -> None:
    client = QdrantClient(path=path)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=2, distance=Distance.COSINE)},
        sparse_vectors_config={"bm25": SparseVectorParams()},
    )
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=i,
                vector={
                    "dense": p["dense"],
                    "bm25": SparseVector(indices=p["sparse"][0], values=p["sparse"][1]),
                },
                payload={
                    "chunk_id": p["chunk_id"],
                    "resolution_id": p["resolution_id"],
                    "page": 1,
                    "text": p["chunk_id"],
                },
            )
            for i, p in enumerate(_POINTS)
        ],
    )
    client.close()


def _vocab_file(tmp_path) -> str:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(_VOCAB, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _store(tmp_path):
    """Embedded Qdrant takes a file lock on its directory, so the store cannot
    live in tmp_path itself alongside the sidecars a test writes there."""
    d = tmp_path / "store"
    d.mkdir()
    _seed(str(d))
    return d


def _retriever(tmp_path, **kwargs) -> QdrantHybridRetriever:
    return QdrantHybridRetriever(
        path=str(_store(tmp_path)),
        collection_name=COLLECTION,
        vocab_path=_vocab_file(tmp_path),
        **kwargs,
    )


def _query() -> Query:
    return Query(text="ก", vector=np.array([1.0, 0.0]), tokens=["ก"])


def _index(provenance: dict | None = None) -> Index:
    # Reads no rows by construction -- see BaseRetriever.reads_index_rows.
    return Index(chunks=[], embeddings=np.zeros((0, 0)), provenance=provenance)


def test_fuses_both_arms_rather_than_returning_the_dense_ranking(tmp_path):
    ranked = _retriever(tmp_path).retrieve(_query(), _index(), k=3)

    assert [r.chunk_id for r in ranked] == ["c2", "c0", "c1"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert ranked[0].resolution_id == "r2"


def test_the_fusion_is_the_shared_fuse_rrf_not_a_reimplementation(tmp_path):
    """Scores must equal `fuse_rrf` over the two arms' own rankings -- the one
    assertion that a third copy of the RRF (with its own rrf_k or tie-break)
    would fail."""
    retriever = _retriever(tmp_path)
    dense_arm, sparse_arm = retriever._arms_for(COLLECTION)
    query, index = _query(), _index()

    expected = fuse_rrf(
        dense_arm.retrieve(query, index, 200),
        sparse_arm.retrieve(query, index, 200),
        3,
    )
    ranked = retriever.retrieve(query, index, k=3)

    assert [r.chunk_id for r in ranked] == [r.chunk_id for r in expected]
    assert [r.score for r in ranked] == [r.score for r in expected]


def test_fetch_depth_is_a_floor_on_the_fetch_not_a_cap_on_k(tmp_path):
    """`min(fetch_depth, k)` would under-fetch the answer it was asked for: at
    depth 1 each arm returns its own single best (c0 dense, c2 sparse) and the
    fusion could never deliver 3."""
    ranked = _retriever(tmp_path, fetch_depth=1).retrieve(_query(), _index(), k=3)

    assert len(ranked) == 3


def test_a_query_term_outside_the_vocabulary_degrades_to_the_dense_arm(tmp_path):
    """The sparse arm returns [] rather than raising, so an out-of-vocabulary
    query still answers -- with the dense ranking, unfused."""
    ranked = _retriever(tmp_path).retrieve(
        Query(text="zz", vector=np.array([1.0, 0.0]), tokens=["zz"]), _index(), k=3
    )

    assert [r.chunk_id for r in ranked] == ["c0", "c1", "c2"]


def test_the_collection_is_resolved_from_the_index_provenance(tmp_path):
    """What lets ONE spec serve all four routed collections: the collection name
    is the index directory's name, read off the Index at query time."""
    store = _store(tmp_path)
    retriever = QdrantHybridRetriever(path=str(store), vocab_path=_vocab_file(tmp_path))

    ranked = retriever.retrieve(
        _query(), _index({"index_dir": str(tmp_path / "somewhere" / COLLECTION)}), k=3
    )

    assert [r.chunk_id for r in ranked] == ["c2", "c0", "c1"]


def test_an_index_with_no_provenance_raises_rather_than_guessing_a_collection(tmp_path):
    retriever = QdrantHybridRetriever(
        path=str(_store(tmp_path)), vocab_path=_vocab_file(tmp_path)
    )

    with pytest.raises(ValueError, match="provenance"):
        retriever.retrieve(_query(), _index(), k=3)


def test_a_missing_vocabulary_sidecar_names_the_ingest_step(tmp_path):
    """An un-ingested route must fail loudly. The sparse arm reads the vocabulary
    written at ingestion; without it there is nothing to score against, and a
    silent dense-only answer would look like a working hybrid."""
    retriever = QdrantHybridRetriever(
        path=str(_store(tmp_path)),
        collection_name=COLLECTION,
        vocab_root=str(tmp_path / "not-ingested"),
    )

    with pytest.raises(FileNotFoundError, match="qdrant_pilot_ingest"):
        retriever.retrieve(_query(), _index(), k=3)


def test_it_reads_no_index_rows_so_query_service_can_skip_the_embeddings(tmp_path):
    assert QdrantHybridRetriever.reads_index_rows is False


def test_it_is_expressible_as_a_strategy_spec(tmp_path):
    """Every constructor argument stays a scalar, which is what lets the UI and
    `route_query` carry this retriever as a `{type, params}` spec."""
    retriever = build_retriever(
        StrategySpec(
            type="qdrant_hybrid",
            params={"url": "http://example.invalid:6333", "fetch_depth": 200},
        )
    )

    assert isinstance(retriever, QdrantHybridRetriever)
    assert retriever.name == "qdrant_hybrid"
    # Constructing it must not require a reachable server: the client is lazy.
    assert retriever._client is None


def test_the_measured_serving_defaults_are_the_class_defaults():
    """`exact=True` and `fetch_depth=200` are measurements
    (`data/results/qdrant_pilot.md`, `data/results/routed_fetch_depth_test.md`),
    not preferences -- changing either silently re-ranks the served path."""
    retriever = QdrantHybridRetriever()

    assert retriever.exact is True
    assert retriever.hnsw_ef is None
    assert retriever.fetch_depth == 200
    assert (retriever.dense_weight, retriever.bm25_weight, retriever.rrf_k) == (0.5, 0.5, 60)
