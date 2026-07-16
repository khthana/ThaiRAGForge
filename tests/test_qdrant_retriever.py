"""QdrantRetriever: dense retrieval + native resolution_id filtering against an
embedded local Qdrant collection (no server -- QdrantClient(path=tmp_path))."""
from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_lab.retrievers import QdrantRetriever
from rag_lab.schema import Index, Query


def _seed_collection(path: str, points: list[dict]) -> None:
    client = QdrantClient(path=path)
    client.create_collection(
        collection_name="test", vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    client.upsert(
        collection_name="test",
        points=[
            PointStruct(
                id=i,
                vector=p["vector"],
                payload={
                    "chunk_id": p["chunk_id"],
                    "resolution_id": p["resolution_id"],
                    "page": 1,
                    "text": p["chunk_id"],
                },
            )
            for i, p in enumerate(points)
        ],
    )
    client.close()


_POINTS = [
    {"chunk_id": "c0", "resolution_id": "r0", "vector": [1.0, 0.0]},
    {"chunk_id": "c1", "resolution_id": "r1", "vector": [0.9, 0.1]},
    {"chunk_id": "c2", "resolution_id": "r2", "vector": [0.8, 0.2]},
]


def test_ranks_by_cosine_like_dense_retriever(tmp_path):
    _seed_collection(str(tmp_path), _POINTS)
    retriever = QdrantRetriever(path=str(tmp_path), collection_name="test")

    ranked = retriever.retrieve(Query(text="q", vector=np.array([1.0, 0.0])), Index(
        chunks=[], embeddings=np.zeros((0, 2))
    ), k=3)

    assert [r.chunk_id for r in ranked] == ["c0", "c1", "c2"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert ranked[0].resolution_id == "r0"


def test_resolution_id_filter_excludes_non_matching_chunks(tmp_path):
    _seed_collection(str(tmp_path), _POINTS)
    retriever = QdrantRetriever(path=str(tmp_path), collection_name="test")

    ranked = retriever.retrieve(
        Query(
            text="q",
            vector=np.array([1.0, 0.0]),
            filters={"resolution_id_in": ["r1", "r2"]},
        ),
        Index(chunks=[], embeddings=np.zeros((0, 2))),
        k=3,
    )

    assert {r.resolution_id for r in ranked} == {"r1", "r2"}
    assert "c0" not in {r.chunk_id for r in ranked}


def test_no_filter_returns_everything_up_to_k(tmp_path):
    _seed_collection(str(tmp_path), _POINTS)
    retriever = QdrantRetriever(path=str(tmp_path), collection_name="test")

    ranked = retriever.retrieve(Query(text="q", vector=np.array([1.0, 0.0])), Index(
        chunks=[], embeddings=np.zeros((0, 2))
    ), k=2)

    assert len(ranked) == 2
