"""The serving-path retriever cache.

The third construction a served query pays for, and the last one to be priced.
`query_indices` built a fresh retriever on every call; for the engine topology
that instance owns the Qdrant client and a per-collection arm cache whose
construction parses a 78k-term vocabulary sidecar off disk, so
`data/results/serving_concurrency.md` section 4 measured **327 ms of a 433 ms
query** going on rebuilding what the previous query had just built.

Two properties carry the design and are pinned here rather than argued:

1. **Reuse cannot change an answer.** A retriever is a pure function of its
   spec -- everything a `retrieve()` reads comes from its arguments -- so a
   cached one must return exactly what a fresh one returns, including across
   two different indices in a row, which is the case a single-slot per-instance
   cache would get wrong.
2. **The eval path is deliberately excluded**, the same rule the embedder and
   index caches follow, so no published number can move.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from rag_lab.config import StrategySpec
from rag_lab.factory import (
    build_retriever,
    build_retriever_cached,
    clear_retriever_cache,
    retriever_cache_info,
)
from rag_lab.schema import Chunk, Index

SPEC = StrategySpec(type="hybrid", params={"fetch_depth": 200})


@pytest.fixture(autouse=True)
def _clean():
    clear_retriever_cache()
    yield
    clear_retriever_cache()


def _index(tag: str, n: int = 8) -> Index:
    chunks = [
        Chunk(
            chunk_id=f"{tag}-{i}",
            resolution_id=f"2568/1/เรื่อง {tag} {i}",
            text=f"ข้อความ {tag} หมายเลข {i}",
            chunk_index=i,
        )
        for i in range(n)
    ]
    emb = np.eye(n, 4, dtype=np.float32)
    return Index(
        chunks=chunks,
        embeddings=emb,
        meta={"combo_id": tag},
        lexical=[[tag, f"tok{i}"] for i in range(n)],
    )


class _Embedder:
    def embed_query(self, text: str):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def embed(self, texts):
        return np.stack([self.embed_query(t) for t in texts])


# ------------------------------------------------------------------ identity
def test_the_same_spec_returns_the_same_object():
    assert build_retriever_cached(SPEC) is build_retriever_cached(SPEC)


def test_a_different_spec_returns_a_different_object():
    other = StrategySpec(type="hybrid", params={"fetch_depth": 50})
    assert build_retriever_cached(SPEC) is not build_retriever_cached(other)


def test_key_order_in_params_is_not_a_different_spec():
    a = StrategySpec(type="hybrid", params={"fetch_depth": 200, "rrf_k": 60})
    b = StrategySpec(type="hybrid", params={"rrf_k": 60, "fetch_depth": 200})
    assert build_retriever_cached(a) is build_retriever_cached(b)


def test_the_uncached_builder_is_still_uncached():
    """The eval path must keep constructing fresh, or a 36-combo sweep starts
    sharing engine connections nobody asked for."""
    assert build_retriever(SPEC) is not build_retriever(SPEC)


def test_size_zero_disables_it(monkeypatch):
    monkeypatch.setenv("RAG_LAB_RETRIEVER_CACHE", "0")
    assert build_retriever_cached(SPEC) is not build_retriever_cached(SPEC)
    assert retriever_cache_info()["size"] == 0


def test_a_garbage_env_value_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("RAG_LAB_RETRIEVER_CACHE", "not-a-number")
    assert build_retriever_cached(SPEC) is build_retriever_cached(SPEC)


def test_it_evicts_the_least_recently_used(monkeypatch):
    monkeypatch.setenv("RAG_LAB_RETRIEVER_CACHE", "2")
    a = build_retriever_cached(StrategySpec(type="hybrid", params={"fetch_depth": 10}))
    build_retriever_cached(StrategySpec(type="hybrid", params={"fetch_depth": 20}))
    build_retriever_cached(StrategySpec(type="hybrid", params={"fetch_depth": 30}))
    assert retriever_cache_info()["size"] == 2
    assert build_retriever_cached(
        StrategySpec(type="hybrid", params={"fetch_depth": 10})
    ) is not a


# --------------------------------------------------------------- correctness
def test_a_cached_retriever_returns_what_a_fresh_one_returns():
    index, emb = _index("one"), _Embedder()
    from rag_lab.pipeline import retrieve

    fresh = retrieve("ข้อความ", index, emb, build_retriever(SPEC), 5, combination_id="f")
    cached = build_retriever_cached(SPEC)
    for _ in range(3):
        got = retrieve("ข้อความ", index, emb, cached, 5, combination_id="c")
        assert [r.chunk_id for r in got.results] == [r.chunk_id for r in fresh.results]
        assert [r.score for r in got.results] == [r.score for r in fresh.results]


def test_one_retriever_serving_two_indices_in_a_row_is_not_confused():
    """The routed case: four collections through one instance. A retriever that
    memoised anything per-Index in a single slot would answer the second index
    with the first one's state, which is the failure this cache could cause and
    the reason `QdrantHybridRetriever` keys its arms by collection."""
    from rag_lab.pipeline import retrieve

    emb = _Embedder()
    a, b = _index("aaa"), _index("bbb")
    fresh_a = retrieve("ข้อความ", a, emb, build_retriever(SPEC), 5, combination_id="f")
    fresh_b = retrieve("ข้อความ", b, emb, build_retriever(SPEC), 5, combination_id="f")

    shared = build_retriever_cached(SPEC)
    got_a = retrieve("ข้อความ", a, emb, shared, 5, combination_id="c")
    got_b = retrieve("ข้อความ", b, emb, shared, 5, combination_id="c")
    got_a2 = retrieve("ข้อความ", a, emb, shared, 5, combination_id="c")

    assert [r.chunk_id for r in got_a.results] == [r.chunk_id for r in fresh_a.results]
    assert [r.chunk_id for r in got_b.results] == [r.chunk_id for r in fresh_b.results]
    assert [r.chunk_id for r in got_a2.results] == [r.chunk_id for r in fresh_a.results]


def test_concurrent_builders_hand_everyone_one_object():
    """The object is now SHARED between concurrent callers where each used to
    build its own -- the same behaviour change the embedder cache made."""
    seen = []
    lock = threading.Lock()

    def work():
        r = build_retriever_cached(SPEC)
        with lock:
            seen.append(r)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(r) for r in seen}) == 1


def test_the_serving_path_uses_the_cache():
    """query_indices must call the cached builder, not the plain one -- the
    whole 327 ms lives on that one line."""
    import inspect

    from rag_lab import query_service

    src = inspect.getsource(query_service.query_indices)
    assert "build_retriever_cached(retriever_spec)" in src
    assert "build_retriever(retriever_spec)" not in src
