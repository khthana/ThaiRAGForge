"""Build strategy instances from {type, params} specs via the registries.

Imports the strategy packages so their decorator registrations fire (a
decorator-based registry is silent if the module is never imported).
"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict

import rag_lab.chunkers  # noqa: F401  (register strategies)
import rag_lab.embedders  # noqa: F401
import rag_lab.loaders  # noqa: F401
import rag_lab.rerankers  # noqa: F401
import rag_lab.retrievers  # noqa: F401
from rag_lab.config import StrategySpec
from rag_lab.registries import (
    chunker_registry,
    embedder_registry,
    loader_registry,
    reranker_registry,
    retriever_registry,
)


def build_loader(spec: StrategySpec):
    return loader_registry.get(spec.type)(**spec.params)


def build_chunker(spec: StrategySpec):
    return chunker_registry.get(spec.type)(**spec.params)


def build_embedder(spec: StrategySpec):
    """Always constructs a fresh embedder. See `build_embedder_cached` for the
    serving path, and note that this one is deliberately NOT the cached one."""
    return embedder_registry.get(spec.type)(**spec.params)


# --------------------------------------------------------------- embedder cache
#
# Measured 2026-08-21 on the shipped `person` route (bge-m3, 57,172 chunks),
# warm disk, one query decomposed into first-use vs warm-use:
#
#     one served query, nothing cached   11,433 ms
#       constructor                           0.0 ms   <- lazy, see below
#       first embed (weight load)         8,928.6 ms   <- what this cache removes
#       warm embed                           12.9 ms   <- irreducible GPU work
#       store.load + BM25 rebuild         2,159   ms   <- a separate, unbuilt cache
#       warm retrieve                       345.6 ms   <- irreducible
#
# So 78% of a served query is loading weights that the previous query already
# loaded. `LocalSTEmbedder._load()` is lazy, so this cost hides inside the first
# `embed()` rather than in the constructor -- which is why `build_embedder`
# timed at 0.0 ms and the first version of that measurement concluded a cache
# could win only 10%. Read a 9 s "encode" against the published 13-83 ms
# (data/results/cost_latency_pareto.md) as an instrument fault, not a finding.
#
# **Why this is a separate function rather than caching `build_embedder`.**
# Every eval script builds embedders in a loop over all 9 of them, and a global
# cache would hold Qwen3-Embedding-4B resident alongside its neighbours on a
# 12 GB card -- the OOM this project already lost five `semantic` x 4B runs to.
# The serving path opts in; nothing that produces a published number changes
# construction behaviour at all.
#
# **Bounded at 2 by default, and 2 is not arbitrary**: the shipped router's five
# routes resolve to exactly two distinct embedders (bge-m3 for
# person/faculty/unmatched, qwen3-0.6B for program/course), so 2 holds the whole
# serving set with nothing spare. `RAG_LAB_EMBEDDER_CACHE=0` disables it;
# any other integer sets the size.
_EMBEDDER_CACHE_ENV = "RAG_LAB_EMBEDDER_CACHE"
_DEFAULT_EMBEDDER_CACHE = 2

_embedder_cache: "OrderedDict[tuple, object]" = OrderedDict()
# Guards the dict only, NOT `embed()`. Construction under contention would
# otherwise load the same weights twice and cache the loser.
_embedder_lock = threading.Lock()


def _cache_size() -> int:
    raw = os.environ.get(_EMBEDDER_CACHE_ENV)
    if raw is None or raw == "":
        return _DEFAULT_EMBEDDER_CACHE
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_EMBEDDER_CACHE


def _spec_key(spec: StrategySpec) -> tuple:
    # params is a dict, so it cannot key a cache directly. Sorted JSON is a
    # canonical form: two specs that differ only in key order are the same
    # embedder and must hit, and any value difference (model_name, device,
    # batch_size, max_seq_length) must miss.
    return (spec.type, json.dumps(spec.params, sort_keys=True, default=str))


def _release(embedder) -> None:
    """Drop an evicted embedder's VRAM instead of waiting for the GC.

    Eviction that does not free is worse than no cache: the model is
    unreachable but still resident, so the card fills with dead weights.
    Guarded because torch is an optional import path here and a cache must not
    be the thing that makes a CPU-only environment fail.
    """
    try:
        model = getattr(embedder, "_model", None)
        if model is not None:
            setattr(embedder, "_model", None)
        del embedder
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def build_embedder_cached(spec: StrategySpec):
    """`build_embedder`, reusing an already-constructed embedder for the same spec.

    For the SERVING path (`query_service`), where consecutive queries hit the
    same two models and each construction otherwise reloads ~8.9 s of weights.

    An embedder is a pure function of its spec -- it holds a model name, a
    device and two size limits, and no per-index or per-query state -- so
    reuse cannot change a vector. That is checked rather than asserted:
    `tests/test_embedder_cache.py` requires a cached embedder's output to be
    BITWISE identical to a freshly built one.

    Consequence worth knowing before raising the size: the returned object is
    now SHARED between concurrent callers, where every caller used to get its
    own. `sentence_transformers.encode` is inference-only and mutates nothing
    per call, and concurrent encoding through one model is pinned to
    bit-identical output by the same test file -- the same check
    `qdrant_concurrency_test.py`'s S1 makes one layer up.
    """
    size = _cache_size()
    if size == 0:
        return build_embedder(spec)

    key = _spec_key(spec)
    with _embedder_lock:
        if key in _embedder_cache:
            _embedder_cache.move_to_end(key)
            return _embedder_cache[key]

    # Built OUTSIDE the lock: loading weights takes seconds, and holding the
    # lock across it would serialise every other route's cache hit behind it.
    embedder = build_embedder(spec)

    with _embedder_lock:
        if key in _embedder_cache:
            # Another thread won the race. Keep theirs so every caller shares
            # one object, and release ours rather than leaking it.
            _embedder_cache.move_to_end(key)
            winner = _embedder_cache[key]
            _release(embedder)
            return winner
        _embedder_cache[key] = embedder
        while len(_embedder_cache) > size:
            _, evicted = _embedder_cache.popitem(last=False)
            _release(evicted)
        return embedder


def clear_embedder_cache() -> None:
    """Drop every cached embedder and free its VRAM. For tests, and for a
    caller that needs the card back."""
    with _embedder_lock:
        while _embedder_cache:
            _, e = _embedder_cache.popitem()
            _release(e)


def embedder_cache_info() -> dict:
    """What is resident right now -- for a UI or a probe to report."""
    with _embedder_lock:
        return {
            "size": len(_embedder_cache),
            "max_size": _cache_size(),
            "keys": [k[0] + " " + k[1] for k in _embedder_cache],
        }


def build_retriever(spec: StrategySpec):
    return retriever_registry.get(spec.type)(**spec.params)


def build_reranker(spec: StrategySpec):
    return reranker_registry.get(spec.type)(**spec.params)
