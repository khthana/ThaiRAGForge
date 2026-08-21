"""A bounded, staleness-checked cache of loaded `Index` artifacts, for serving.

`ArtifactStore.load` re-reads `chunks.parquet`, the ~234MB `embeddings.npy` and
rebuilds ~57k `Chunk` objects on every call. Measured on the shipped `person`
route (`data/results/serving_cost_profile.md`, 2026-08-21) that is **1,159 ms**,
and because `BM25Okapi` is memoised *on the Index object*
(`Index.lexical_scorer`), throwing the Index away also throws the scorer away and
the next hybrid retrieve rebuilds it -- a further **921 ms**. Together 2,079 ms
of a 2,994 ms served query, i.e. everything left after the embedder cache except
the ~400 ms of real scoring work.

**Sharing an Index is only safe because nothing mutates one, and that was checked
rather than assumed.** Across `src/`, `tools/` and `app/` there is exactly ONE
write to an Index attribute: `bm25.py`'s `index.lexical_scorer = (...)`, the memo
this cache exists to preserve. `MetadataFilter` and `EntityFilter` both go
through `Index.select()`, which builds a NEW Index (fancy-indexing the embedding
matrix copies it), and no code assigns to `.chunks`/`.embeddings`/`.meta` or
mutates `meta` in place. If that ever stops being true this cache becomes a
correctness bug rather than a slow path, so
`tests/io/test_index_cache.py` pins the no-mutation property directly.

**Staleness is the whole risk, not memory.** A long-running server holding an
Index while the directory is rebuilt underneath it would serve the previous
build's vectors while every artifact on disk says otherwise -- the exact
two-artifacts-from-different-days shape `audit_pipeline_invariants.py` exists to
catch, except invisible because it lives in RAM. So every cache HIT re-stats the
four artifact files and serves the cached Index only if `(mtime_ns, size)` is
unchanged for all of them. That costs ~4 stat calls (microseconds) against a
1,159 ms reload, so there is no reason to make it optional.

Bounded at 4 by default: the five shipped routes resolve to four distinct index
directories (`faculty` and `unmatched` share one). `RAG_LAB_INDEX_CACHE=0`
disables it; any other integer sets the size. Unlike the embedder cache this
holds host RAM rather than VRAM -- roughly the size of `embeddings.npy` plus the
chunk objects per entry -- so raising it past the routed set is a RAM decision
and `index_cache_info()` reports what is resident.

**Serving path only.** `ArtifactStore.load` stays uncached, so an eval script
looping over 36 combos keeps its current memory profile and no published number
can move. Same rule, and the same reason, as `factory.build_embedder_cached`.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.schema import Index

_CACHE_ENV = "RAG_LAB_INDEX_CACHE"
_DEFAULT_SIZE = 4

#: The files whose (mtime_ns, size) make up an index's identity on disk. A
#: rebuild rewrites all of them; `lexical.json` is optional and its absence is
#: itself part of the stamp, so an index that gains one is treated as changed.
_ARTIFACTS = ("chunks.parquet", "embeddings.npy", "meta.json", "lexical.json")

_cache: "OrderedDict[tuple, tuple[Any, Index]]" = OrderedDict()
_lock = threading.Lock()


def _cache_size() -> int:
    raw = os.environ.get(_CACHE_ENV)
    if raw is None or raw == "":
        return _DEFAULT_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_SIZE


def _stamp(directory: Path) -> tuple:
    """What the index looks like on disk right now.

    `(mtime_ns, size)` per artifact, `None` for one that does not exist. Both
    halves matter: a rebuild that happens to land on the same nanosecond is
    absurd, but a same-second rewrite is not, and size catches the truncation
    case a coarse mtime would miss.
    """
    out = []
    for name in _ARTIFACTS:
        p = directory / name
        try:
            st = p.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


def load_index_cached(
    directory: str | Path,
    *,
    with_embeddings: bool = True,
    store: ArtifactStore | None = None,
) -> Index:
    """`ArtifactStore.load`, reusing an already-loaded Index for the same
    directory when nothing on disk has changed.

    `with_embeddings` is part of the key: an engine-served retriever loads
    without the matrix, and handing it a cached Index that has one would work
    but waste the ~234MB the flag exists to avoid -- while the reverse would
    hand a row-reading retriever an empty matrix, which is a silent wrong
    answer. Two different objects, two different keys.
    """
    store = store or ArtifactStore()
    size = _cache_size()
    d = Path(directory).resolve()
    if size == 0:
        return store.load(d, with_embeddings=with_embeddings)

    key = (str(d), bool(with_embeddings))
    stamp = _stamp(d)

    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            if hit[0] == stamp:
                _cache.move_to_end(key)
                return hit[1]
            # Rebuilt underneath us. Drop it rather than serving the previous
            # build's rows -- and drop it now, not on eviction, so the stale
            # object cannot be handed out by a concurrent hit.
            del _cache[key]

    # Loaded OUTSIDE the lock: ~1.2s of I/O would otherwise block every other
    # route's cache hit behind it.
    index = store.load(d, with_embeddings=with_embeddings)

    with _lock:
        # Re-stamp rather than trusting the pre-load one: if the directory was
        # rewritten *while* we were reading it, caching under the old stamp
        # would pin a torn read that no later check could ever invalidate.
        _cache[key] = (_stamp(d), index)
        _cache.move_to_end(key)
        while len(_cache) > size:
            _cache.popitem(last=False)
    return index


def clear_index_cache() -> None:
    """Drop every cached Index. For tests, and for a caller that needs the RAM."""
    with _lock:
        _cache.clear()


def index_cache_info() -> dict:
    """What is resident right now -- for a UI or a probe to report."""
    with _lock:
        return {
            "size": len(_cache),
            "max_size": _cache_size(),
            "entries": [
                {
                    "dir": Path(k[0]).name,
                    "with_embeddings": k[1],
                    "n_chunks": len(v[1].chunks),
                    "embeddings_shape": list(v[1].embeddings.shape),
                    "has_bm25_scorer": v[1].lexical_scorer is not None,
                }
                for k, v in _cache.items()
            ],
        }
