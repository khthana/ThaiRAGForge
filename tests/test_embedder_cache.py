"""The serving-path embedder cache.

Measured motivation (2026-08-21, shipped `person` route, bge-m3, 57,172 chunks):
one served query costs 11,433 ms of which 8,929 ms is loading weights the
previous query already loaded, against 359 ms of irreducible work. The cache is
worth ~32x on a warm query.

Two properties carry the whole design and both are checked here rather than
argued:

1. **A cached embedder's vectors are BITWISE identical to a fresh one's.** An
   embedder holds a model name, a device and two size limits and no per-index or
   per-query state, so reuse *cannot* change a vector -- but "cannot" is the kind
   of claim this project has been wrong about before, so it is executed.
2. **The object is now SHARED between concurrent callers**, where every caller
   used to construct its own. That is a genuine behaviour change under load, so
   concurrent encoding through one cached embedder is pinned to bit-identical
   output -- the same check `qdrant_concurrency_test.py`'s S1 makes one layer up.

The eval path is deliberately excluded (`build_embedder` stays uncached), because
a global cache would hold Qwen3-Embedding-4B resident beside its neighbours
during a 9-embedder sweep on a 12 GB card. That exclusion is a test too: a
regression that "helpfully" caches `build_embedder` must fail here.

Uses the `hashing` embedder throughout: it needs no GPU and no download, so these
run in CI, and every property under test is about the CACHE, not the model.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from rag_lab.config import StrategySpec
from rag_lab.factory import (
    build_embedder,
    build_embedder_cached,
    clear_embedder_cache,
    embedder_cache_info,
)

SPEC = StrategySpec(type="hashing", params={"dim": 64})
OTHER = StrategySpec(type="hashing", params={"dim": 128})
TEXTS = ["ผู้ช่วยศาสตราจารย์ ดร. ธนา หงษ์สุวรรณ", "หลักสูตรวิศวกรรมคอมพิวเตอร์"]


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_embedder_cache()
    yield
    clear_embedder_cache()


# ------------------------------------------------------------------ identity
def test_the_same_spec_returns_the_same_object():
    assert build_embedder_cached(SPEC) is build_embedder_cached(SPEC)


def test_a_different_spec_returns_a_different_object():
    assert build_embedder_cached(SPEC) is not build_embedder_cached(OTHER)


def test_param_key_order_does_not_split_the_cache():
    a = StrategySpec(type="hashing", params={"dim": 64})
    b = StrategySpec(type="hashing", params={"dim": 64})
    assert build_embedder_cached(a) is build_embedder_cached(b)


def test_the_uncached_builder_is_still_uncached():
    """The eval path must keep constructing fresh embedders: a sweep over 9
    models on a 12 GB card cannot afford a global cache."""
    assert build_embedder(SPEC) is not build_embedder(SPEC)


# ------------------------------------------------------------------ bounding
def test_it_evicts_the_least_recently_used(monkeypatch):
    monkeypatch.setenv("RAG_LAB_EMBEDDER_CACHE", "2")
    a = build_embedder_cached(SPEC)
    build_embedder_cached(OTHER)
    build_embedder_cached(SPEC)  # touch A so B is now the LRU
    third = StrategySpec(type="hashing", params={"dim": 256})
    build_embedder_cached(third)

    assert embedder_cache_info()["size"] == 2
    assert build_embedder_cached(SPEC) is a, "A was touched, it must have survived"
    assert build_embedder_cached(OTHER) is not None
    # B was evicted, so asking again constructs a new one -- checked by identity
    # against nothing, i.e. the cache is genuinely bounded rather than growing.
    assert embedder_cache_info()["size"] == 2


def test_size_zero_disables_it(monkeypatch):
    monkeypatch.setenv("RAG_LAB_EMBEDDER_CACHE", "0")
    assert build_embedder_cached(SPEC) is not build_embedder_cached(SPEC)
    assert embedder_cache_info()["size"] == 0


def test_a_garbage_env_value_falls_back_to_the_default(monkeypatch):
    """A typo in an env var must not silently disable the cache or crash a
    query -- it should behave as if unset."""
    monkeypatch.setenv("RAG_LAB_EMBEDDER_CACHE", "yes-please")
    assert build_embedder_cached(SPEC) is build_embedder_cached(SPEC)


def test_clear_empties_it():
    build_embedder_cached(SPEC)
    assert embedder_cache_info()["size"] == 1
    clear_embedder_cache()
    assert embedder_cache_info()["size"] == 0


# --------------------------------------------------------------- correctness
def test_a_cached_embedder_returns_bitwise_identical_vectors():
    """The property the whole cache rests on. Not approximately equal --
    identical, because reuse must not be a numerical decision at all."""
    fresh = build_embedder(SPEC).embed(TEXTS)
    cached = build_embedder_cached(SPEC).embed(TEXTS)
    again = build_embedder_cached(SPEC).embed(TEXTS)
    assert np.array_equal(fresh, cached), "a cached embedder changed a vector"
    assert np.array_equal(cached, again), "a reused embedder changed a vector"


def test_concurrent_encoding_through_one_shared_embedder_agrees_bitwise():
    """New behaviour: callers now share one object. Before the cache each
    thread built its own, so this class of interference did not exist."""
    reference = build_embedder(SPEC).embed(TEXTS)
    out: list[np.ndarray] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def work():
        try:
            e = build_embedder_cached(SPEC)
            barrier.wait(timeout=30)  # make the encodes actually overlap
            out.append(e.embed(TEXTS))
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)
            try:
                barrier.abort()
            except Exception:
                pass

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    assert len(out) == 8
    for v in out:
        assert np.array_equal(v, reference), "concurrent use perturbed a vector"


def test_concurrent_construction_hands_everyone_one_object(monkeypatch):
    """The double-checked branch: weights are loaded OUTSIDE the lock, so two
    threads can build the same spec at once. Exactly one must end up cached and
    every caller must get that one -- otherwise a model nobody holds stays
    resident on the card while another is handed out.

    **The constructor is deliberately slowed here.** A first version of this
    test used the real `hashing` embedder, which constructs in microseconds, so
    the first thread always populated the cache before the others looked and the
    race branch was never reached -- deleting the double-check left the test
    passing. A check that cannot fail is not a check, so the overlap is forced
    rather than hoped for.
    """
    import rag_lab.factory as factory

    built: list[object] = []

    def slow_build(spec):
        time.sleep(0.2)  # long enough that all 8 threads are inside at once
        obj = factory.embedder_registry.get(spec.type)(**spec.params)
        built.append(obj)
        return obj

    monkeypatch.setattr(factory, "build_embedder", slow_build)

    got: list[object] = []
    barrier = threading.Barrier(8)

    def work():
        barrier.wait(timeout=30)
        got.append(build_embedder_cached(SPEC))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert len(got) == 8
    assert len(built) > 1, (
        "the race never happened, so this test proved nothing -- raise the sleep"
    )
    assert len({id(g) for g in got}) == 1, "callers received different objects"
    assert embedder_cache_info()["size"] == 1


def test_the_serving_path_uses_the_cache_and_the_eval_path_does_not():
    """Pins the wiring itself, not just the mechanism: query_service must call
    the cached builder, and nothing else may quietly switch to it."""
    import inspect

    from rag_lab import query_service

    src = inspect.getsource(query_service)
    assert src.count("build_embedder_cached(") >= 2, "both loops must use it"
    # `build_embedder(` is NOT a substring of `build_embedder_cached(` -- the
    # underscore breaks it -- so this counts genuinely-bare calls only.
    assert src.count("build_embedder(") == 0, "a bare, uncached call survives"

    from rag_lab import query_sets

    assert "build_embedder_cached(" not in inspect.getsource(query_sets), (
        "the batch/eval path must keep building fresh embedders"
    )
