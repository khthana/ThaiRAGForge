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
from rag_lab.schema import Chunk, Index, Query

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


def _index(provenance: dict | None = None, chunk_ids: list[str] | None = None) -> Index:
    """The Index the collection is a copy of.

    It carries no EMBEDDINGS -- that is what `reads_index_rows = False` buys, and
    what query_service skips loading. It does carry the chunk rows, because
    query_service loads chunks either way (`with_embeddings` is the only thing
    the flag controls) and the staleness guard compares against them. Passing
    `chunk_ids` builds an Index that disagrees with the seeded collection, which
    is what the guard exists to catch.
    """
    ids = chunk_ids if chunk_ids is not None else [p["chunk_id"] for p in _POINTS]
    chunks = [
        Chunk(chunk_id=cid, resolution_id=f"r{i}", text=cid, chunk_index=i, page=1)
        for i, cid in enumerate(ids)
    ]
    return Index(chunks=chunks, embeddings=np.zeros((0, 0)), provenance=provenance)


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


# --------------------------------------------------------------------------- #
# The staleness guard (2026-08-23) -- the engine-side counterpart of the index
# seal. A collection is a copy of an Index's rows, so any rebuild stales it, and
# because results are built from the ENGINE's payload a stale collection does
# not fail, it ANSWERS. Measured end to end in
# tools/eval/serving_failure_modes.py; pinned here.
# --------------------------------------------------------------------------- #
def test_a_collection_holding_a_different_number_of_rows_is_refused(tmp_path):
    r = _retriever(tmp_path)
    with pytest.raises(RuntimeError, match="different build"):
        r.retrieve(_query(), _index(chunk_ids=["c0", "c1"]), 3)


def test_a_SAME_COUNT_collection_whose_rows_differ_is_refused(tmp_path):
    """The count does most of the work; this is the case it cannot see.

    A re-OCR that moves text without moving chunk boundaries rebuilds every
    vector and keeps the row count identical, so a count-only guard would pass
    it. Point id == row index at ingest, so row i's identity is checkable.
    """
    r = _retriever(tmp_path)
    with pytest.raises(RuntimeError, match="disagrees with the index"):
        r.retrieve(_query(), _index(chunk_ids=["c0", "c1", "OTHER"]), 3)


def test_the_refusal_names_the_collection_and_how_to_repair_it(tmp_path):
    """A message an operator can act on, or the guard just moves the confusion."""
    r = _retriever(tmp_path)
    with pytest.raises(RuntimeError) as exc:
        r.retrieve(_query(), _index(chunk_ids=["c0"]), 3)
    msg = str(exc.value)
    assert COLLECTION in msg
    assert "qdrant_pilot_ingest" in msg


def test_verify_collection_False_restores_the_unguarded_behaviour(tmp_path):
    """The escape hatch, and the proof that the defect it hides is real.

    With the guard off the retriever answers happily from a collection that
    disagrees with its index -- which is exactly what shipped before this guard
    and what `serving_failure_modes.md` records as SILENT.
    """
    r = _retriever(tmp_path, verify_collection=False)
    out = r.retrieve(_query(), _index(chunk_ids=["gone"]), 3)
    assert [c.chunk_id for c in out] == ["c2", "c0", "c1"]


def test_the_guard_costs_one_check_per_collection_not_one_per_query(tmp_path):
    """Once per collection per instance -- the serving layer caches instances."""
    r = _retriever(tmp_path)
    calls = {"n": 0}
    real_count = r._shared_client().count

    def counted(**kwargs):
        calls["n"] += 1
        return real_count(**kwargs)

    r._shared_client().count = counted
    for _ in range(3):
        r.retrieve(_query(), _index(), 3)
    assert calls["n"] == 1, "the guard must not re-verify on every query"
    assert COLLECTION in r._verified
