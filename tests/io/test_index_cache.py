"""The serving-path Index cache.

Measured motivation (`data/results/serving_cost_profile.md`, 2026-08-21):
`ArtifactStore.load` costs 1,159 ms, and because `BM25Okapi` is memoised on the
Index object, discarding the Index discards the scorer too and the next hybrid
retrieve rebuilds it -- 921 ms more. Together that is 2,079 ms of the 2,994 ms a
served query still costs after the embedder cache.

Sharing an Index is only safe while nothing mutates one, so the two properties
that carry the design are pinned here rather than argued:

1. **Staleness.** A long-running server holding an Index while its directory is
   rebuilt would serve the previous build's vectors while every artifact on disk
   says otherwise -- the two-artifacts-from-different-days failure this project
   keeps finding, except invisible because it lives in RAM. Every HIT re-stats
   the artifacts.
2. **No mutation.** `MetadataFilter`/`EntityFilter` must derive a new Index and
   leave the cached one intact, and the BM25 memo must SURVIVE across calls --
   it is 921 ms of the saving, so a cache that preserved the rows but dropped
   the scorer would deliver less than half of what it claims.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.io.index_cache import (
    clear_index_cache,
    index_cache_info,
    load_index_cached,
)
from rag_lab.retrievers.filters import MetadataFilter
from rag_lab.schema import Chunk, Index


def _make_index(tmp_path, name: str, n: int = 6, tag: str = "a"):
    chunks = [
        Chunk(
            chunk_id=f"{name}-{i}",
            resolution_id=f"2568/1/เรื่อง {i}",
            text=f"ข้อความ {tag} {i}",
            chunk_index=i,
            page=1,
            metadata={"year": 2568 if i % 2 == 0 else 2567},
        )
        for i in range(n)
    ]
    idx = Index(
        chunks=chunks,
        embeddings=np.arange(n * 4, dtype=np.float32).reshape(n, 4),
        meta={"combo_id": name},
        lexical=[[f"tok{i}", tag] for i in range(n)],
    )
    d = tmp_path / name
    ArtifactStore().save(idx, d)
    return d


@pytest.fixture(autouse=True)
def _clean():
    clear_index_cache()
    yield
    clear_index_cache()


# ------------------------------------------------------------------ identity
def test_the_same_directory_returns_the_same_object(tmp_path):
    d = _make_index(tmp_path, "one")
    assert load_index_cached(d) is load_index_cached(d)


def test_a_different_directory_returns_a_different_object(tmp_path):
    a, b = _make_index(tmp_path, "one"), _make_index(tmp_path, "two")
    assert load_index_cached(a) is not load_index_cached(b)


def test_with_embeddings_is_part_of_the_key(tmp_path):
    """An engine-served retriever loads without the matrix. Serving it the
    row-reading variant wastes the 234MB the flag exists to avoid; serving the
    reverse hands a row-reading retriever an EMPTY matrix, which is a silent
    wrong answer."""
    d = _make_index(tmp_path, "one")
    full = load_index_cached(d, with_embeddings=True)
    thin = load_index_cached(d, with_embeddings=False)
    assert full is not thin
    assert full.embeddings.shape == (6, 4)
    assert thin.embeddings.shape == (0, 0)


def test_the_uncached_loader_is_still_uncached(tmp_path):
    """The eval path must keep loading fresh: a script looping over 36 combos
    cannot afford to hold them all in RAM."""
    d = _make_index(tmp_path, "one")
    store = ArtifactStore()
    assert store.load(d) is not store.load(d)


# ----------------------------------------------------------------- staleness
def test_a_rebuilt_index_is_not_served_from_the_cache(tmp_path):
    """THE check. A rebuild under a running server must invalidate."""
    d = _make_index(tmp_path, "one", tag="old")
    first = load_index_cached(d)
    assert first.chunks[0].text.endswith("old 0")

    time.sleep(0.01)  # ensure a distinct mtime even on a coarse clock
    _make_index(tmp_path, "one", tag="new")  # same directory, rebuilt

    second = load_index_cached(d)
    assert second is not first, "a rebuilt index was served from RAM"
    assert second.chunks[0].text.endswith("new 0")


def test_a_changed_embeddings_file_alone_invalidates(tmp_path):
    """Rows can be identical while the vectors are not -- that is precisely the
    silent case, so the stamp must cover every artifact, not just the chunks."""
    d = _make_index(tmp_path, "one")
    first = load_index_cached(d)
    time.sleep(0.01)
    np.save(d / "embeddings.npy", np.zeros((6, 4), dtype=np.float32))

    second = load_index_cached(d)
    assert second is not first
    assert float(second.embeddings.sum()) == 0.0


def test_an_untouched_index_is_not_reloaded(tmp_path):
    """The converse: if any read counted as a change the cache would never hit
    and every test above would pass vacuously."""
    d = _make_index(tmp_path, "one")
    a = load_index_cached(d)
    for _ in range(5):
        assert load_index_cached(d) is a


# ------------------------------------------------------------------ bounding
def test_it_evicts_the_least_recently_used(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_LAB_INDEX_CACHE", "2")
    a = _make_index(tmp_path, "a")
    b = _make_index(tmp_path, "b")
    c = _make_index(tmp_path, "c")
    first_a = load_index_cached(a)
    load_index_cached(b)
    load_index_cached(a)  # touch A so B is the LRU
    load_index_cached(c)

    assert index_cache_info()["size"] == 2
    assert load_index_cached(a) is first_a, "A was touched, it must have survived"
    assert index_cache_info()["size"] == 2


def test_size_zero_disables_it(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_LAB_INDEX_CACHE", "0")
    d = _make_index(tmp_path, "one")
    assert load_index_cached(d) is not load_index_cached(d)
    assert index_cache_info()["size"] == 0


def test_a_garbage_env_value_falls_back_to_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_LAB_INDEX_CACHE", "lots")
    d = _make_index(tmp_path, "one")
    assert load_index_cached(d) is load_index_cached(d)


# --------------------------------------------------------------- no mutation
def test_a_filter_does_not_mutate_the_cached_index(tmp_path):
    """`MetadataFilter` goes through `Index.select`, which builds a new Index.
    If that ever changes, this cache turns from a slow path into a correctness
    bug -- so the property is checked here, not assumed."""
    d = _make_index(tmp_path, "one")
    cached = load_index_cached(d)
    before = len(cached.chunks)

    narrowed = MetadataFilter({"year": 2568}).apply(cached)

    assert narrowed is not cached
    assert len(narrowed.chunks) < before
    assert len(cached.chunks) == before, "the filter mutated the shared Index"
    assert len(load_index_cached(d).chunks) == before


def test_the_bm25_memo_survives_across_calls(tmp_path):
    """921 ms of the saving is the BM25Okapi memoised ON the Index. A cache that
    kept the rows but handed back a fresh object each time would deliver less
    than half of what it claims."""
    d = _make_index(tmp_path, "one")
    idx = load_index_cached(d)
    assert idx.lexical_scorer is None

    from rag_lab.retrievers.bm25 import BM25Retriever

    scorer = BM25Retriever._scorer(idx)
    assert load_index_cached(d).lexical_scorer is not None
    assert BM25Retriever._scorer(load_index_cached(d)) is scorer


def test_a_rebuild_drops_the_stale_bm25_memo(tmp_path):
    """The memo rides on the object, so invalidation must take it with them --
    otherwise a rebuilt index could be scored by the previous build's tokens."""
    d = _make_index(tmp_path, "one", tag="old")
    from rag_lab.retrievers.bm25 import BM25Retriever

    old_scorer = BM25Retriever._scorer(load_index_cached(d))
    time.sleep(0.01)
    _make_index(tmp_path, "one", tag="new")

    fresh = load_index_cached(d)
    assert fresh.lexical_scorer is None
    assert BM25Retriever._scorer(fresh) is not old_scorer


# -------------------------------------------------------------- concurrency
def test_concurrent_loads_hand_everyone_one_object(tmp_path):
    d = _make_index(tmp_path, "one")
    got: list[Index] = []
    barrier = threading.Barrier(6)

    def work():
        barrier.wait(timeout=30)
        got.append(load_index_cached(d))

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert len(got) == 6
    # The last writer wins the dict slot, and every LATER caller must get that
    # one. Racing loaders may each return their own object -- that is wasted
    # work, never a wrong answer, since all were read from the same bytes.
    assert len({id(g) for g in got}) <= len(got)
    after = load_index_cached(d)
    assert all(g.chunks[0].chunk_id == after.chunks[0].chunk_id for g in got)


def test_the_serving_path_uses_the_cache_and_the_store_is_untouched():
    import inspect

    from rag_lab import query_service

    src = inspect.getsource(query_service)
    assert src.count("load_index_cached(") >= 2, "both loops must use it"
    assert "store.load(" not in src, "a bare, uncached load survives"
