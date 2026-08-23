"""A bounded, staleness-checked cache of loaded `Index` artifacts, for serving.

`ArtifactStore.load` re-reads `chunks.parquet`, the ~234MB `embeddings.npy` and
rebuilds ~57k `Chunk` objects on every call. Measured on the shipped `person`
route (`data/results/serving_cost_profile.md`) that is **1,185 ms**,
and because `BM25Okapi` is memoised *on the Index object*
(`Index.lexical_scorer`), throwing the Index away also throws the scorer away and
the next hybrid retrieve rebuilds it -- a further **995 ms**. Together **2,180 ms**
off the embedder-cached arm's 3,069 ms steady state, i.e. everything left after
the embedder cache except the real scoring work.

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
~1.15 s reload, so there is no reason to make it optional.

**A rebuild that lands DURING a read is the case a hit-time check cannot see,
and it was got wrong first (fixed 2026-08-21).** The load used to be stamped
only *afterwards*, reasoning that stamping before would pin a torn read. That is
backwards: a rebuild overlapping the read leaves the post-load stamp equal to
what is now on disk, so the stale object is cached under the CURRENT stamp and
every later hit agrees with it -- pinned permanently, the exact failure this
cache exists to prevent. Measured, not argued: rewriting the directory mid-load
made the next two calls return the previous build's rows indefinitely. The load
is now stamped **before and after** and cached only if the two agree; otherwise
it is re-read (`_MAX_RELOADS`), and a read that keeps racing **raises**. The
object is not merely stale in that window -- `save` writes four files in
sequence and `Index` is row-aligned across two of them, so a read can pair one
build's chunks with another's vectors, which nothing downstream can detect.

**A rebuild that lands BETWEEN two of the writer's own files is a SECOND case,
and stamping the read at both ends does not see it either (found by measuring
it, 2026-08-21).** `save` writes `chunks.parquet` and then `embeddings.npy`, so
in between, the directory is *stably* inconsistent -- new chunks, previous
build's vectors, nothing moving. A reader whose whole load falls inside that
window stamps the same thing before and after, caches the pairing, and then
serves it to every later hit until the next write. Measured under load with a
150 ms inter-file gap, that was the MAJORITY of reads. It is undetectable
downstream (same row count, same dtype, wrong rows), so the writer now declares
when it is finished (`ArtifactStore.seal`) and `_settle` refuses a directory
whose artifacts do not match that declaration. A directory written before seals
existed is classified **unsealed** and gets the older, narrower guarantee --
reported per entry by `index_cache_info()` rather than assumed away.

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
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from rag_lab.io.artifact_store import ArtifactStore, artifact_stamp, read_seal
from rag_lab.schema import Index

_CACHE_ENV = "RAG_LAB_INDEX_CACHE"
_DEFAULT_SIZE = 4

#: How long to wait between looks at a directory whose seal does not match its
#: artifacts. A real save writes ~234MB, so the mismatching window is seconds;
#: this only has to be long enough that four looks are four looks and not one.
_SETTLE_SLEEP_S = 0.05

#: How many times to re-read an index whose directory changed *during* the read.
#: `ArtifactStore.save` writes four files in sequence, so a read that overlaps a
#: rebuild can pair new chunks with the previous build's vectors -- misaligned
#: rows, which `Index` cannot detect and which produce wrong answers rather than
#: an error. A rebuild's save is seconds, so a few re-reads outlast it; a read
#: that keeps racing raises instead of returning that pairing.
_MAX_RELOADS = 3

_cache: "OrderedDict[tuple, tuple[Any, Index, str]]" = OrderedDict()
_lock = threading.Lock()


def _cache_size() -> int:
    raw = os.environ.get(_CACHE_ENV)
    if raw is None or raw == "":
        return _DEFAULT_SIZE
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_SIZE


def _as_tuple(stamp) -> tuple:
    return tuple(tuple(e) if e is not None else None for e in stamp)


def _stamp(directory: Path) -> tuple:
    """What the index looks like on disk right now, as a hashable tuple.

    The rule itself lives in `artifact_store.artifact_stamp`, which is also
    what `seal` records -- one copy, so a reader and a writer cannot end up
    checking different files.
    """
    return _as_tuple(artifact_stamp(directory))


def _settle(d: Path) -> tuple[str, tuple]:
    """Decide what state the directory is in before reading a byte of it.

    Returns ("sealed"|"unsealed", stamp), or raises if the directory is
    mid-write.

    **This is the check the before/after stamping could not make.** Stamping a
    read at both ends detects a write that OVERLAPS the read; it cannot detect
    a directory that is stably inconsistent -- new chunks.parquet beside the
    previous build's embeddings.npy, nothing moving, because the writer is
    between its two writes. ArtifactStore.save leaves the previous seal
    standing while it writes, so that state is exactly "seal does not match
    artifacts", and it is refused here rather than read.

    A mismatch is NEVER downgraded to "probably an out-of-band edit, read it
    anyway". That was the tempting rule and it is unsound for a measurable
    reason: during the inter-file window the directory is *stable*, so
    stability cannot tell an edited directory from a half-written one. An
    index rewritten in place must re-seal (ArtifactStore.seal); that is a
    one-line obligation on a writer, against a silent wrong answer here.
    """
    for _ in range(_MAX_RELOADS + 1):
        stamp = _stamp(d)
        sealed = read_seal(d)
        if sealed is None:
            return "unsealed", stamp
        if _as_tuple(sealed) == stamp:
            return "sealed", stamp
        time.sleep(_SETTLE_SLEEP_S)
    raise RuntimeError(
        f"index directory {d} does not match the build its writer last sealed -- it is "
        f"being rebuilt, or an artifact was rewritten in place without re-sealing "
        f"(ArtifactStore.seal). Refusing to serve rows that may come from two builds."
    )


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
    #
    # STAMPED BEFORE AND AFTER, and cached only if the two agree. This version
    # of the cache stamped only *after* the load, reasoning that stamping before
    # "would pin a torn read that no later check could ever invalidate". The
    # reasoning was backwards and the measurement (2026-08-21) says so: a
    # rebuild landing mid-read leaves the post-load stamp equal to what is now
    # on disk, so the object is cached under the CURRENT stamp and every later
    # hit re-stats, agrees, and serves it -- the stale index is pinned
    # permanently, which is the exact failure this cache exists to prevent.
    # Stamping before would merely have caused a redundant reload. Doing both is
    # strictly better than either: an overlapping rebuild is detected rather
    # than reasoned about.
    last_exc: Exception | None = None
    for _ in range(_MAX_RELOADS):
        mode, before = _settle(d)
        try:
            index = store.load(d, with_embeddings=with_embeddings)
        except Exception as exc:
            # THE SAME RACE, THE OTHER OUTCOME. A write landing mid-read can
            # either hand back rows from two builds -- caught by the stamp
            # comparison below -- or truncate a file under the reader, in which
            # case `load` RAISES from inside pyarrow/numpy/json and, until
            # 2026-08-23, propagated straight past the check that already knew
            # how to handle it. Measured at real size rather than reasoned
            # about: a 305MB directory rewritten under a served load produced
            # ValueError("Failed to read all data for array") and
            # JSONDecodeError, never a mixed read, because files that large are
            # caught mid-copy by their own formats. Two consequences, and the
            # second is the dangerous one: the caller saw an exception the cache
            # could have retried, and it was NOT the RuntimeError a serving
            # layer retries on, so a torn read read as a corrupt index.
            #
            # A stable directory that still fails to load is genuinely corrupt,
            # and that exception is re-raised UNCHANGED. Only a directory that
            # moved under the read is retried.
            if _stamp(d) == before:
                raise
            last_exc = exc
            continue
        after = _stamp(d)
        if before == after:
            with _lock:
                _cache[key] = (after, index, mode)
                _cache.move_to_end(key)
                while len(_cache) > size:
                    _cache.popitem(last=False)
            return index
        # The read overlapped a write. The object is not merely stale, it may
        # pair one build's chunks with another's vectors (`save` writes four
        # files in sequence and `Index` is row-aligned across two of them), so
        # it is neither cached NOR returned.
    raise RuntimeError(
        f"index directory {d} changed during each of {_MAX_RELOADS} reads -- it is "
        f"being rebuilt. Refusing to serve a read that may pair one build's chunks "
        f"with another's vectors; retry once the build has finished."
    ) from last_exc


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
                    # "unsealed" is a REPORTED gap, not a passing grade: for
                    # such a directory only the overlapping-write race is
                    # detectable, not the mixed-build one.
                    "sealed": v[2] == "sealed",
                }
                for k, v in _cache.items()
            ],
        }
