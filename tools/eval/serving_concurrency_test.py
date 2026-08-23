"""The SHIPPED serving path under concurrent load: which topology, and does it scale?

`qdrant_concurrency.md` (2026-08-13) answered "which layer saturates" for a
**hand-assembled** Qdrant pipeline whose embedder and Index were built once
*outside* the loop. That was an idealisation then. It is the shipped path now:
`build_embedder_cached` and `load_index_cached` landed 2026-08-21, so a served
query really does reuse one model and one Index. What that report could not
answer is the question the deployment actually faces, because it never ran the
in-process path at all:

    the default `hybrid` retriever scores 57k x 1024 in numpy and runs
    `BM25Okapi.get_scores` in a Python loop, both IN the request thread.

So the two topologies this repo can serve have never been compared under load,
and only one of them is what `dense`/`hybrid` ship as the default. Four arms,
every real one driven through the **shipped `route_query`** rather than a
hand-assembled pipeline -- the point is what a deployment gets, including the
per-query costs an eval harness hoists out of the loop:

    null      the harness issuing nothing. A load generator that is itself the
              bottleneck measures Python's scheduler and reports it as a system
              limit.
    encode    embed_query alone, on the route's own embedder -- the GPU
              parameter. Does NOT transfer to another card, and is measured
              alone for exactly that reason: substitute a plateau and the
              composed arms' ceiling can be re-derived without re-running.
    inproc    route_query(retriever="hybrid", fetch_depth=200). The default.
    engine    route_query(retriever="qdrant_hybrid", exact=True, fetch_depth=200).

`inproc` and `engine` return the same fused answer over the same data (that is
`qdrant_routed_check.md`'s result, re-checked here as S6), so this is a pure
cost comparison between two ways of serving one system.

**Two design points that differ from the older harness, both forced by the
measurement rather than chosen.**

1. **Cells are time-boxed, not request-boxed.** That harness kept ~20 requests
   per worker at every level. Here the arms differ by an order of magnitude in
   per-query cost (`inproc` is 626.2 ms at C=1, the engine's retrieval half is
   ~27 ms), so a fixed request count either starves the fast arm's percentiles
   or spends minutes on one cell of the slow one. Each cell runs for
   `--budget` seconds OR `2 x C` requests, whichever is longer, and every table
   prints `n` and requests/worker so a thin percentile is visible rather than
   implied.

2. **The index cache is raised to 8 for the run, and that is a harness
   setting, not a deployment one.** `with_embeddings` is part of the cache key
   and `qdrant_hybrid` loads without the matrix, so holding both topologies
   resident needs two entries per routed directory. A deployment serves ONE
   topology and needs the shipped 4.

**A confound stated in advance, pointing the safe way.** The embedder runs
in-process, so GIL contention and CUDA serialisation are bundled with request
handling; the stated target has a separate GPU server. That makes this box look
worse at the app layer than the deployment will, so a clean scaling result is
conservative. It is also why `encode` is a separate arm.

Section 4 is a different kind of check and needs no GPU: **a rebuild landing
underneath a running server**, which `index_cache.py` was fixed for on
2026-08-21 and which its memory note still lists as unmeasured under real
concurrent load. Reader threads hammer `load_index_cached` on a scratch
directory while a writer alternates two builds through `ArtifactStore.save`.
Each build stamps its identity into BOTH the chunk text and the vectors, so a
read that pairs one build's chunks with another's rows is detectable -- which
is the failure mode `Index`'s row alignment (`I1`) cannot detect on its own.

Everything is cached to `data/results/serving_concurrency_raw.json` after every
cell, so a crash costs one cell and `--render` is free.

Run (Qdrant up and ingested; NOTHING else on the GPU -- stop Streamlit first):

    docker start rag-qdrant
    PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe \
        tools/eval/serving_concurrency_test.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Raised BEFORE the first cache read: the harness holds both topologies
# resident (with_embeddings is part of the key) where a deployment holds one.
os.environ.setdefault("RAG_LAB_INDEX_CACHE", "8")

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder_cached, embedder_cache_info  # noqa: E402
import rag_lab.io.index_cache as index_cache_mod  # noqa: E402
from rag_lab.io.artifact_store import ArtifactStore, seal  # noqa: E402
from rag_lab.io.index_cache import (  # noqa: E402
    clear_index_cache,
    index_cache_info,
    load_index_cached,
)
from rag_lab.query_service import (  # noqa: E402
    _read_manifest,
    discover_indices,
    resolve_index,
    route_query,
    warm_serving_caches,
)
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.router import classify_query, route_targets  # noqa: E402
from rag_lab.schema import Chunk, Index  # noqa: E402

INDEX_ROOT = REPO / "data/index/chunker_compare_full"
GOLD = REPO / "config/eval/gold_query_set_73det.yaml"
RAW = REPO / "data/results/serving_concurrency_raw.json"
REPORT = REPO / "data/results/serving_concurrency.md"
ROUTED_REPORT = REPO / "data/results/routed_fetch_depth_test.md"

K = 10
FETCH_DEPTH = 200                       # what the UI ships and what F=200 measured
CONCURRENCY = [1, 2, 5, 10, 25, 50]
ARMS = ["null", "encode", "inproc", "engine"]
REAL_ARMS = ["encode", "inproc", "engine"]

# S4's domain. Below CPython's switch interval a request's residence time is
# unresolvable by this instrument, not violating Little's law -- derived from
# the interpreter rather than from whichever arm happens to fail.
SWITCH_INTERVAL_S = sys.getswitchinterval()

DECISION_RULE = """
Let plateau(arm) = the highest throughput (q/s) that arm reaches at any level.

1. TOPOLOGY   ENGINE  if plateau(engine) >= 1.5 x plateau(inproc)
              INPROC  if plateau(inproc) >= 1.5 x plateau(engine)
              TIE     otherwise.
   1.5x is chosen in advance and the full curve is printed, so a near-call is
   visible rather than decided by the threshold.

2. SCALING (reported per arm, never instead of 1)
              SERIAL  if plateau <= 1.2 x throughput(C=1)
              SCALES  if plateau >= 1.5 x throughput(C=1)
              PARTIAL otherwise.
   An arm that is SERIAL cannot be sized by adding users; it must be sized by
   making one query cheaper, or by moving the work off the request thread.

3. HEADROOM   r = plateau(winner) / plateau(encode).
   A composed arm cannot exceed `encode`, which it contains. ENCODE-DOMINATED
   if r >= 0.8 -- the GPU is essentially the whole cost and substituting a
   faster one moves the system nearly 1:1. Otherwise the remainder is named
   and its size stated as a bound, never as a diagnosis.
""".strip()


# --------------------------------------------------------------------------- #
# anchors parsed from their own reports, never frozen as literals
# --------------------------------------------------------------------------- #
def published_routed_p50() -> float | None:
    """The shipped routed hybrid p50 at F=200, read from its own report.

    Fourteen frozen cross-artifact anchors have already been replaced in this
    repo for going on printing a number their source had moved past. A missing
    or renamed report yields None and the check says it could not be made.
    """
    if not ROUTED_REPORT.exists():
        return None
    for line in ROUTED_REPORT.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 2 and cells[1] == "F=200" and cells[2].endswith("ms"):
            try:
                return float(cells[2].removesuffix("ms").strip())
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------- #
# load generator
# --------------------------------------------------------------------------- #
def run_level(
    work, payloads: list, c: int, budget_s: float, max_requests: int = 200_000
) -> dict:
    """Closed loop: `c` threads, each taking the next payload the moment its
    previous request returns, until the budget expires AND every worker has
    issued at least 2 requests.

    Dispatch is `next(itertools.count())`, atomic under the GIL. A shared
    counter behind a `threading.Lock` is real untimed overhead outside the
    timed region -- a thread blocking on a contended lock can wait a full
    switch interval, which is nothing against a 626.2 ms retrieval but is several
    times a cheap arm's own work, and it shows up as Little's law failing for
    that arm alone.

    The deadline is set by the barrier's `action`, which runs in ONE thread
    while every other is still parked. Setting it in the first thread released
    would let a faster-scheduled peer read a deadline of 0 and quit after its
    minimum -- a harness bug that would report the fastest arm as the slowest.

    `max_requests` bounds memory rather than time: the `null` arm issues on the
    order of a million requests a second, and a per-request latency list of that
    length is the harness measuring its own allocator.
    """
    counter = itertools.count()
    min_requests = 2 * c
    lats: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    deadline = [0.0]

    def start() -> None:
        deadline[0] = time.perf_counter() + budget_s

    barrier = threading.Barrier(c, action=start)

    def worker(slot: int) -> None:
        local: list[float] = []
        local_err: list[str] = []
        barrier.wait()
        while True:
            i = next(counter)
            if i >= max_requests:
                break
            if i >= min_requests and time.perf_counter() >= deadline[0]:
                break
            payload = payloads[i % len(payloads)]
            t0 = time.perf_counter()
            try:
                work(payload, slot)
            except Exception as exc:  # recorded, not swallowed: an OOM at C=50
                local_err.append(f"{type(exc).__name__}: {exc}")  # is a finding
            local.append((time.perf_counter() - t0) * 1000.0)
        with lock:
            lats.extend(local)
            errors.extend(local_err)

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in range(c)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    lats.sort()
    return {
        "concurrency": c,
        "n": len(lats),
        "per_worker": len(lats) / c if c else 0.0,
        "wall_s": wall,
        "throughput_qps": len(lats) / wall if wall > 0 else 0.0,
        "p50_ms": statistics.median(lats) if lats else 0.0,
        "p95_ms": lats[int(0.95 * (len(lats) - 1))] if lats else 0.0,
        "mean_ms": statistics.fmean(lats) if lats else 0.0,
        "errors": len(errors),
        "error_sample": errors[:3],
    }


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def specs(url: str) -> dict[str, StrategySpec]:
    return {
        "inproc": StrategySpec(type="hybrid", params={"fetch_depth": FETCH_DEPTH}),
        "engine": StrategySpec(
            type="qdrant_hybrid",
            params={"url": url, "fetch_depth": FETCH_DEPTH, "exact": True},
        ),
    }


def warm_everything(indices, url: str) -> dict:
    """Warm both topologies, and time each one's FIRST query afterwards.

    The first query is timed because `warm_serving_caches` has a documented
    residue: everything resident is still not warm, the remainder being
    process-global CUDA/BLAS initialisation, which one probe retrieval removes.

    **The first version of this function found that the engine topology was not
    getting that probe at all** -- it was gated on `with_rows`, which is False
    for the engine shape -- and the engine arm's first query cost 657 ms against
    a ~160 ms steady state while the row-reading arm beside it had been probed.
    The gate is gone (`query_service.warm_serving_caches`), so both rows below
    are now probed and the comparison is between topologies rather than between
    one that was warmed and one that was not. The engine arm is still warmed
    FIRST, so its cell can never be flattered by a probe the other arm ran.
    """
    sp = specs(url)
    out: dict = {"warm": {}, "first_query_ms": {}}
    # Engine first: whichever arm is warmed first pays the process-global CUDA
    # init, so putting the arm under suspicion first is the conservative order.
    order = ["engine", "inproc"]
    for arm in order:
        t0 = time.perf_counter()
        res = warm_serving_caches(
            indices,
            sp[arm].type,
            with_rows=(arm == "inproc"),
            retriever_params=dict(sp[arm].params),
        )
        res["wall_ms"] = (time.perf_counter() - t0) * 1000
        out["warm"][arm] = res
    entries = load_gold_query_set(GOLD)
    for arm in order:
        t0 = time.perf_counter()
        route_query(entries[0].query, indices, sp[arm], K)
        out["first_query_ms"][arm] = (time.perf_counter() - t0) * 1000
    return out


def build_arms(indices, url: str, texts: list[str]):
    """arms[name] = (work_fn, payloads). Everything real goes through route_query."""
    sp = specs(url)

    # `encode` uses each query's OWN route embedder, so the arm carries the same
    # model mix the served arms do rather than one hand-picked model.
    targets = route_targets("hybrid")
    by_text: dict[str, object] = {}
    for text in texts:
        info = resolve_index(targets[classify_query(text)], indices)
        manifest = _read_manifest(info.dir)
        by_text[text] = build_embedder_cached(
            StrategySpec.model_validate(manifest["combo"]["embedder"])
        )

    def w_null(text, slot):
        return None

    def w_encode(text, slot):
        return by_text[text].embed_query(text)

    def w_inproc(text, slot):
        return route_query(text, indices, sp["inproc"], K)

    def w_engine(text, slot):
        return route_query(text, indices, sp["engine"], K)

    return {
        "null": (w_null, texts),
        "encode": (w_encode, texts),
        "inproc": (w_inproc, texts),
        "engine": (w_engine, texts),
    }


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #
def agreement(indices, url: str, texts: list[str]) -> dict:
    """S6: the two topologies must answer the same question.

    Without this the comparison could be between a fast arm and a wrong one.
    Compared as SCORE SEQUENCES, never as id sets: both arms fuse with the same
    RRF over the same rankings, and where two chunks carry the same fused score
    neither engine promises which one is returned -- a rule this project has now
    been bitten by three times.
    """
    sp = specs(url)
    rows = []
    for text in texts:
        a = route_query(text, indices, sp["inproc"], K).results
        b = route_query(text, indices, sp["engine"], K).results
        ids_a = [r.chunk_id for r in a]
        ids_b = [r.chunk_id for r in b]
        sa = [round(float(r.score), 12) for r in a]
        sb = [round(float(r.score), 12) for r in b]
        moved = [i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y]
        # A moved position is benign exactly when the two arms scored that rank
        # identically -- i.e. the ids swapped inside a tie group.
        out_of_tie = [i for i in moved if i < len(sa) and i < len(sb) and sa[i] != sb[i]]
        rows.append(
            {
                "query": text,
                "agree_at_10": len(ids_a) - len(moved),
                "moved": len(moved),
                "out_of_tie": len(out_of_tie),
                "max_score_gap": max((abs(x - y) for x, y in zip(sa, sb)), default=0.0),
            }
        )
    return {
        "rows": rows,
        "n": len(rows),
        "moved": sum(r["moved"] for r in rows),
        "out_of_tie": sum(r["out_of_tie"] for r in rows),
        "max_score_gap": max((r["max_score_gap"] for r in rows), default=0.0),
    }


def retriever_construction(indices, url: str, texts: list[str], reps: int = 12) -> dict:
    """What a served query pays to REBUILD its retriever, and what caching it
    removes.

    `query_indices` built a fresh retriever on every call until 2026-08-21. For
    the engine topology that instance owns the Qdrant client and a
    per-collection arm cache whose construction parses a 78k-term vocabulary
    sidecar off disk, so the whole thing was thrown away between queries --
    the embedder and the Index were cached, this was not.

    Three arms at C=1, everything else held: the retrieval is identical, only
    where the retriever comes from differs. `routed` is the shipped path, so it
    shows what a deployment now gets rather than what the two extremes imply.
    """
    from rag_lab.factory import build_retriever, build_retriever_cached
    from rag_lab.pipeline import retrieve

    sp = specs(url)
    targets = route_targets("qdrant_hybrid")
    store = ArtifactStore()
    held = build_retriever_cached(sp["engine"])
    fresh_ms, held_ms, routed_ms = [], [], []
    for text in texts[:reps]:
        info = resolve_index(targets[classify_query(text)], indices)
        manifest = _read_manifest(info.dir)
        embedder = build_embedder_cached(
            StrategySpec.model_validate(manifest["combo"]["embedder"])
        )
        index = load_index_cached(info.dir, with_embeddings=False, store=store)

        t0 = time.perf_counter()
        retrieve(text, index, embedder, build_retriever(sp["engine"]), K,
                 combination_id="fresh")
        fresh_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        retrieve(text, index, embedder, held, K, combination_id="held")
        held_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        route_query(text, indices, sp["engine"], K)
        routed_ms.append((time.perf_counter() - t0) * 1000)

    med = lambda xs: statistics.median(xs) if xs else 0.0  # noqa: E731
    return {
        "n": len(fresh_ms),
        "fresh_p50_ms": med(fresh_ms),
        "held_p50_ms": med(held_ms),
        "routed_p50_ms": med(routed_ms),
        "delta_ms": med(fresh_ms) - med(held_ms),
    }


def hostname_control(fast_url: str, slow_url: str, reps: int = 6) -> dict:
    """The same server, the same query, two spellings of the same host.

    Not a micro-benchmark: `QdrantHybridRetriever` and the Streamlit UI both
    defaulted to `localhost` while every eval script passed `127.0.0.1`, so
    every published Qdrant latency was measured on a path the shipped default
    did not take. Measured here rather than reasoned about, and the arm order
    is deliberately fast-then-slow-then-fast so a one-off stall cannot be read
    as the effect.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import SearchParams

    import socket

    out: dict = {"fast_url": fast_url, "slow_url": slow_url}
    try:
        out["resolves"] = {
            "slow": [
                [f[0].name, str(f[4][0])]
                for f in socket.getaddrinfo(
                    slow_url.split("//")[1].split(":")[0], None, type=socket.SOCK_STREAM
                )
            ]
        }
    except OSError as exc:  # noqa: BLE001
        out["resolves"] = {"error": str(exc)}

    rng = np.random.default_rng(0)
    vec = rng.normal(size=1024).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()

    def probe(url: str, collection: str) -> float | None:
        try:
            client = QdrantClient(url=url, timeout=60)
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                client.query_points(
                    collection_name=collection,
                    query=vec,
                    using="dense",
                    limit=FETCH_DEPTH,
                    search_params=SearchParams(exact=True),
                    with_payload=False,
                )
                times.append((time.perf_counter() - t0) * 1000)
            return statistics.median(times)
        except Exception as exc:  # noqa: BLE001
            out.setdefault("errors", []).append(f"{url}: {type(exc).__name__}")
            return None

    try:
        collection = QdrantClient(url=fast_url, timeout=60).get_collections().collections[0].name
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["collection"] = collection
    out["fast_p50_ms"] = probe(fast_url, collection)
    out["slow_p50_ms"] = probe(slow_url, collection)
    out["fast_again_p50_ms"] = probe(fast_url, collection)
    if out.get("fast_p50_ms"):
        out["ratio"] = (out.get("slow_p50_ms") or 0.0) / out["fast_p50_ms"]
    return out


# --------------------------------------------------------------------------- #
# section 4: a rebuild landing underneath a running server
# --------------------------------------------------------------------------- #
def _synthetic(tag: str, n: int, dim: int) -> Index:
    """A build whose identity is stamped in BOTH halves of the row alignment.

    The chunk text says which build it came from and so does every vector, so a
    read that pairs one build's chunks with another build's rows is DETECTABLE.
    That is the whole point: the two builds have the same shape, so the Index
    row-alignment invariant cannot see the mixing and neither can anything
    downstream -- which is why the cache has to refuse rather than return.
    """
    value = 1.0 if tag == "A" else 2.0
    chunks = [
        Chunk(
            chunk_id=f"{tag}-{i}",
            resolution_id=f"res/{i}",
            text=f"{tag} build row {i}",
            chunk_index=i,
        )
        for i in range(n)
    ]
    return Index(
        chunks=chunks,
        embeddings=np.full((n, dim), value, dtype=np.float32),
        meta={"build": tag},
    )


def _write_build(index: Index, d: Path, mid_gap_s: float) -> None:
    """ArtifactStore.save's writes, with a controllable gap between the two
    halves of the row alignment.

    mid_gap_s is the window between the two halves of the row alignment. Zero
    is the easiest case, not the realistic one: a real save writes ~234MB of
    embeddings.npy after the parquet, so a real rebuild leaves that window open
    for SECONDS. A non-zero gap is therefore closer to a deployment than the
    back-to-back arm is, and S9 requires it to have actually made the cache
    refuse -- a check that only *might* exercise its mechanism is a vacuous
    PASS.
    """
    store = ArtifactStore()
    tmp = d.parent / (d.name + "__staging")
    if tmp.exists():
        shutil.rmtree(tmp)
    store.save(index, tmp)  # built away from the readers
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tmp / "chunks.parquet", d / "chunks.parquet")
    if (tmp / "lexical.json").exists():
        # With the chunks, not after the gap: it is derived from chunk TEXT, so
        # it belongs to the same half of the row alignment. Absent for the
        # synthetic builds, so section 6's behaviour is unchanged.
        shutil.copyfile(tmp / "lexical.json", d / "lexical.json")
    if mid_gap_s:
        time.sleep(mid_gap_s)
    shutil.copyfile(tmp / "embeddings.npy", d / "embeddings.npy")
    shutil.copyfile(tmp / "meta.json", d / "meta.json")
    # LAST, and re-derived here rather than copied: the seal records mtimes,
    # and copying files gives them new ones. This is what an in-place writer
    # owes (ArtifactStore.save does it for its own writes).
    seal(d)
    shutil.rmtree(tmp)


def rebuild_under_load(tmp_root: Path, readers: int, seconds: float) -> dict:
    """Reader threads hammering load_index_cached while a writer rebuilds.

    Reported per writer mode. The decisive number is `mixed` -- a served Index
    whose chunks and vectors came from different builds -- which must be 0.
    `refused` is the designed behaviour, not a failure: a read that keeps racing
    raises rather than returning a pairing nothing downstream could detect.
    """
    out: dict = {"readers": readers, "seconds": seconds, "modes": {}}
    builds = {t: _synthetic(t, 200, 8) for t in ("A", "B")}
    real_read_seal = index_cache_mod.read_seal

    # The third arm is the NEGATIVE CONTROL, and the section is worth nothing
    # without it: it makes the reader treat the directory as unsealed, i.e.
    # exactly the behaviour that shipped before 2026-08-21, on this same rig.
    # If it does not produce mixed reads, the other two arms passing says
    # nothing about the fix -- it says the rig never exercised the race.
    modes = (
        ("back_to_back", 0.0, True),
        ("gap_150ms", 0.15, True),
        ("gap_150ms_unsealed", 0.15, False),
    )
    for mode, gap, use_seal in modes:
        index_cache_mod.read_seal = (
            real_read_seal if use_seal else (lambda _directory: None)
        )
        d = tmp_root / f"race_{mode}"
        if d.exists():
            shutil.rmtree(d)
        _write_build(builds["A"], d, 0.0)
        stop = threading.Event()
        stats = {"reads": 0, "mixed": 0, "refused": 0, "other_errors": 0, "writes": 0}
        lock = threading.Lock()

        def read_loop() -> None:
            reads = mixed = refused = other = 0
            while not stop.is_set():
                try:
                    idx = load_index_cached(d, with_embeddings=True)
                except RuntimeError:
                    refused += 1
                    continue
                except Exception:
                    other += 1
                    continue
                reads += 1
                tags = {c.text.split()[0] for c in idx.chunks}
                vals = set(np.unique(idx.embeddings).tolist())
                expected = {1.0} if tags == {"A"} else {2.0}
                if len(tags) != 1 or vals != expected:
                    mixed += 1
            with lock:
                stats["reads"] += reads
                stats["mixed"] += mixed
                stats["refused"] += refused
                stats["other_errors"] += other

        def write_loop() -> None:
            writes = 0
            for tag in itertools.cycle(("B", "A")):
                if stop.is_set():
                    break
                _write_build(builds[tag], d, gap)
                writes += 1
                time.sleep(0.02)
            with lock:
                stats["writes"] += writes

        threads = [threading.Thread(target=read_loop, daemon=True) for _ in range(readers)]
        writer = threading.Thread(target=write_loop, daemon=True)
        writer.start()
        for t in threads:
            t.start()
        time.sleep(seconds)
        stop.set()
        writer.join(timeout=30)
        for t in threads:
            t.join(timeout=30)

        # After the writer stops, the cache must converge on what is on disk.
        final = load_index_cached(d, with_embeddings=True)
        stats["final_build"] = final.meta.get("build")
        stats["final_consistent"] = (
            len({c.text.split()[0] for c in final.chunks}) == 1
            and len(set(np.unique(final.embeddings).tolist())) == 1
        )
        stats["sealed"] = use_seal
        stats["gap_s"] = gap
        out["modes"][mode] = stats
        shutil.rmtree(d, ignore_errors=True)
    index_cache_mod.read_seal = real_read_seal
    return out


# --------------------------------------------------------------------------- #
# section 6b: the same race, at REAL index size, through the shipped route_query
# --------------------------------------------------------------------------- #
def _stamp(base: Index, tag: str) -> Index:
    """A real index, stamped in BOTH halves of the row alignment.

    Section 6 uses a 200x8 synthetic because nothing else reads it. That leaves
    two things unmeasured, and they are the two that decide whether the seal
    holds in a deployment: the inter-file window is ARTIFICIAL there (0.15 s,
    chosen; here it is whatever writing 234 MB actually costs), and the reader
    calls `load_index_cached` DIRECTLY rather than going through the three
    caches a served query passes.

    The stamp has to be visible without disturbing what is being served, so it
    is deliberately minimal: a prefix on `chunk_id`, and column 0 of every
    vector. Text is untouched, so `lexical.json` stays valid for both builds and
    BM25 is not rebuilt per read; overwriting 1 of 1024 dimensions moves a
    cosine by ~0.1%, so the ranking a reader gets is still the real one.
    """
    value = 1.0 if tag == "A" else 2.0
    emb = base.embeddings.copy()
    emb[:, 0] = value
    return Index(
        chunks=[c.model_copy(update={"chunk_id": f"{tag}::{c.chunk_id}"})
                for c in base.chunks],
        embeddings=emb,
        meta={**base.meta, "build": tag},
        lexical=base.lexical,
    )


def rebuild_under_load_served(tmp_root: Path, indices, spec, texts: list[str],
                              readers: int, seconds: float,
                              pause_s: float = 25.0, mid_gap_s: float = 15.0) -> dict:
    """Section 6's race, but at real size and driven through `route_query`.

    THE SCRATCH COPY IS NOT A CONVENIENCE, IT IS THE SAFETY RULE. A writer runs
    in this loop, so pointing it at `data/index/` would destroy a real index
    that costs ~2 h of GPU to rebuild. `route_query` takes its `indices` list as
    an argument, so the redirect needs no monkeypatching and cannot leak: the
    IndexInfo handed to it is the real route target with `dir` swapped.

    Detection is honest about one seam. The LOAD under test is the one a served
    query performs; the CHECK reads the object the cache is holding immediately
    afterwards, through the same cache, because `RetrievalResult` does not carry
    the Index. A swap landing between the two shows up as a disagreement rather
    than as a miss, which is why `checked` is reported separately from `served`.
    """
    from rag_lab.query_service import IndexInfo

    target = route_targets(spec.type if spec.type in ("dense", "hybrid") else "hybrid")["person"]
    real = resolve_index(target, indices)
    src = Path(real.dir)

    base = load_index_cached(src, with_embeddings=True)
    builds = {t: _stamp(base, t) for t in ("A", "B")}
    del base
    clear_index_cache()

    out: dict = {"readers": readers, "seconds": seconds, "pause_s": pause_s,
                 "mid_gap_s": mid_gap_s, "modes": {},
                 "source_dir": str(src), "n_chunks": len(builds["A"].chunks),
                 "dim": int(builds["A"].embeddings.shape[1])}

    # Stage both builds ONCE. The writer loop then only copies, which is what a
    # real save's second half costs; re-serialising per cycle would time pyarrow
    # instead of the race.
    staging = {}
    for tag, idx in builds.items():
        st = tmp_root / f"staged_{tag}"
        if st.exists():
            shutil.rmtree(st)
        ArtifactStore().save(idx, st)
        # query_indices reads manifest.json, which ArtifactStore.save does not
        # write (the build pipeline does). It is NOT in ARTIFACT_FILES, so it is
        # outside both the row alignment and the seal -- copying it once is
        # faithful, and rewriting it per cycle would add nothing to the race.
        shutil.copyfile(src / "manifest.json", st / "manifest.json")
        staging[tag] = st
    out["staged_bytes"] = sum(
        f.stat().st_size for f in staging["A"].iterdir() if f.is_file())

    real_read_seal = index_cache_mod.read_seal
    modes = (("real_rebuild", True), ("real_rebuild_unsealed", False))
    for mode, use_seal in modes:
        index_cache_mod.read_seal = (
            real_read_seal if use_seal else (lambda _directory: None))
        d = tmp_root / f"served_{mode}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for f in staging["A"].iterdir():
            shutil.copyfile(f, d / f.name)
        seal(d)

        info = IndexInfo(combo_id=real.combo_id, dir=str(d), loader=real.loader,
                         chunker=real.chunker, embedder=real.embedder)
        clear_index_cache()
        stop = threading.Event()
        stats = {"served": 0, "checked": 0, "mixed": 0, "refused": 0,
                 "other_errors": 0, "writes": 0, "gap_s": [], "error_kinds": {}}
        lock = threading.Lock()

        def read_loop(worker: int) -> None:
            served = checked = mixed = refused = other = 0
            kinds: dict[str, int] = {}
            for n in itertools.count():
                if stop.is_set():
                    break
                text = texts[(worker * 7 + n) % len(texts)]
                try:
                    route_query(text, [info], spec, K)
                    served += 1
                except RuntimeError:
                    refused += 1
                    continue
                except Exception as exc:
                    other += 1
                    kinds[f"{type(exc).__name__}: {str(exc)[:90]}"] = (
                        kinds.get(f"{type(exc).__name__}: {str(exc)[:90]}", 0) + 1)
                    continue
                try:
                    idx = load_index_cached(Path(info.dir), with_embeddings=True)
                except Exception as exc:
                    k = f"check/{type(exc).__name__}: {str(exc)[:80]}"
                    kinds[k] = kinds.get(k, 0) + 1
                    continue
                checked += 1
                tags = {c.chunk_id.split("::")[0] for c in idx.chunks}
                vals = set(np.unique(idx.embeddings[:, 0]).tolist())
                if len(tags) != 1 or vals != ({1.0} if tags == {"A"} else {2.0}):
                    mixed += 1
            with lock:
                stats["served"] += served
                stats["checked"] += checked
                stats["mixed"] += mixed
                stats["refused"] += refused
                stats["other_errors"] += other
                for k, v in kinds.items():
                    stats["error_kinds"][k] = stats["error_kinds"].get(k, 0) + v

        def write_loop() -> None:
            writes, gaps = 0, []
            for tag in itertools.cycle(("B", "A")):
                if stop.is_set():
                    break
                st = staging[tag]
                shutil.copyfile(st / "chunks.parquet", d / "chunks.parquet")
                shutil.copyfile(st / "lexical.json", d / "lexical.json")
                # THE STABLE WINDOW IS CHOSEN, AND IT MUST EXCEED A LOAD.
                # Two measurements put it here, neither of them taste. (1) With
                # no gap at all BOTH modes returned zero mixed reads: `save`
                # goes from `pq.write_table` straight into `np.save`, so at
                # 305MB the exposure is dominated by a TRUNCATED file (loud,
                # and the format itself catches it) rather than by two complete
                # mismatched ones (silent). (2) At section 6's 0.15s it was
                # still zero, and that is structural rather than unlucky: a read
                # of this index takes ~1.5s, so a window shorter than a load
                # cannot contain one, the reader's own before/after stamps
                # straddle the writer's transition, and the 2026-08-21 stamping
                # fix catches it WITHOUT the seal. So the seal's unique job --
                # a directory that is stably inconsistent for longer than a read
                # -- is only reachable above that line, which is precisely the
                # in-place rewrite it was built for (relabel_index_resolution_ids
                # rewrites for minutes). The sealed arm faces the identical gap,
                # which is what keeps the comparison fair.
                if mid_gap_s:
                    time.sleep(mid_gap_s)
                t0 = time.perf_counter()
                # THE WINDOW IS MEASURED, NOT CHOSEN. Section 6 had to invent
                # 0.15 s; here it is what writing embeddings.npy really costs,
                # which is the number a deployment is exposed to.
                shutil.copyfile(st / "embeddings.npy", d / "embeddings.npy")
                gaps.append(time.perf_counter() - t0)
                shutil.copyfile(st / "meta.json", d / "meta.json")
                seal(d)
                writes += 1
                # A DEPLOYMENT REBUILDS OCCASIONALLY, NOT IN A LOOP. Without this
                # the directory is mid-write most of the time, so the cache
                # correctly refuses nearly every read and the run measures the
                # refusal path instead of the race -- the first smoke served 2
                # queries against 52 refusals, and the negative control could not
                # fire at all, which would have made S10 vacuous.
                if stop.wait(pause_s):
                    break
            with lock:
                stats["writes"] += writes
                stats["gap_s"].extend(gaps)

        threads = [threading.Thread(target=read_loop, args=(i,), daemon=True)
                   for i in range(readers)]
        writer = threading.Thread(target=write_loop, daemon=True)
        writer.start()
        for t in threads:
            t.start()
        time.sleep(seconds)
        stop.set()
        writer.join(timeout=120)
        for t in threads:
            t.join(timeout=120)

        final = load_index_cached(d, with_embeddings=True)
        ftags = {c.chunk_id.split("::")[0] for c in final.chunks}
        stats["final_consistent"] = (
            len(ftags) == 1
            and set(np.unique(final.embeddings[:, 0]).tolist())
            == ({1.0} if ftags == {"A"} else {2.0}))
        stats["sealed"] = use_seal
        stats["median_gap_s"] = (
            sorted(stats["gap_s"])[len(stats["gap_s"]) // 2] if stats["gap_s"] else 0.0)
        stats.pop("gap_s")
        out["modes"][mode] = stats
        clear_index_cache()
        shutil.rmtree(d, ignore_errors=True)

    index_cache_mod.read_seal = real_read_seal
    for st in staging.values():
        shutil.rmtree(st, ignore_errors=True)
    return out


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def plateau(levels: list[dict]) -> tuple[float, int]:
    best = max(levels, key=lambda r: r["throughput_qps"])
    return best["throughput_qps"], best["concurrency"]


def verdict(data: dict) -> dict:
    curves = data["curves"]
    out: dict = {"plateau": {}, "scaling": {}}
    for arm in ARMS:
        if arm not in curves or not curves[arm]:
            continue
        qps, at_c = plateau(curves[arm])
        first = next(
            (r for r in curves[arm] if r["concurrency"] == 1), curves[arm][0]
        )
        out["plateau"][arm] = {"qps": qps, "at_c": at_c, "c1_qps": first["throughput_qps"]}
        ratio = qps / first["throughput_qps"] if first["throughput_qps"] else 0.0
        out["scaling"][arm] = {
            "ratio": ratio,
            "label": "SERIAL" if ratio <= 1.2 else ("SCALES" if ratio >= 1.5 else "PARTIAL"),
        }

    pin = out["plateau"].get("inproc", {}).get("qps", 0.0)
    pen = out["plateau"].get("engine", {}).get("qps", 0.0)
    if pen >= 1.5 * pin:
        topology, winner = "ENGINE", "engine"
    elif pin >= 1.5 * pen:
        topology, winner = "INPROC", "inproc"
    else:
        topology = "TIE"
        winner = "engine" if pen >= pin else "inproc"
    out["topology"] = topology
    out["winner"] = winner
    out["engine_over_inproc"] = (pen / pin) if pin else 0.0

    penc = out["plateau"].get("encode", {}).get("qps", 0.0)
    r = (out["plateau"].get(winner, {}).get("qps", 0.0) / penc) if penc else 0.0
    out["headroom_ratio"] = r
    out["headroom"] = "ENCODE-DOMINATED" if r >= 0.8 else "NOT ENCODE-DOMINATED"
    return out


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def _f(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def render(data: dict) -> tuple[str, list[tuple[str, bool, str]]]:
    v = verdict(data)
    L: list[str] = []
    checks: list[tuple[str, bool, str]] = []

    L.append("# Serving under concurrent load — which topology, and does it scale?")
    L.append("")
    L.append(
        f"Generated by `tools/eval/serving_concurrency_test.py` on {data['generated']}."
    )
    L.append("")
    L.append(
        f"Every real arm is driven through the shipped `route_query` over the "
        f"{data['n_queries']} queries of `gold_query_set_73det.yaml` (so the route mix is "
        f"the real one), k={K}, `fetch_depth`={FETCH_DEPTH}, both serving caches warm. "
        f"Cells are time-boxed at {data['budget_s']:.0f}s or 2xC requests, whichever is "
        f"longer."
    )
    L.append("")

    # ---- 1. curves ----
    L.append("## 1. Throughput and latency")
    L.append("")
    L.append("| arm | C | q/s | p50 ms | p95 ms | n | req/worker | errors |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for arm in ARMS:
        for row in data["curves"].get(arm, []):
            L.append(
                f"| `{arm}` | {row['concurrency']} | {_f(row['throughput_qps'], 2)} | "
                f"{_f(row['p50_ms'])} | {_f(row['p95_ms'])} | {row['n']} | "
                f"{_f(row['per_worker'])} | {row['errors']} |"
            )
    L.append("")
    L.append("| arm | plateau q/s | at C | C=1 q/s | plateau / C=1 | scaling |")
    L.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for arm in ARMS:
        if arm not in v["plateau"]:
            continue
        p = v["plateau"][arm]
        s = v["scaling"][arm]
        L.append(
            f"| `{arm}` | {_f(p['qps'], 2)} | {p['at_c']} | {_f(p['c1_qps'], 2)} | "
            f"{_f(s['ratio'], 2)}x | **{s['label']}** |"
        )
    L.append("")

    # ---- 2. verdict ----
    L.append("## 2. Verdict, against the rule frozen before the run")
    L.append("")
    L.append("```")
    L.append(DECISION_RULE)
    L.append("```")
    L.append("")
    L.append(
        f"- **TOPOLOGY = {v['topology']}** — engine plateau is "
        f"{_f(v['engine_over_inproc'], 2)}x the in-process one."
    )
    for arm in REAL_ARMS:
        if arm in v["scaling"]:
            L.append(
                f"- `{arm}` **{v['scaling'][arm]['label']}** "
                f"({_f(v['scaling'][arm]['ratio'], 2)}x from C=1 to its plateau)"
            )
    L.append(
        f"- **{v['headroom']}** — the winning arm (`{v['winner']}`) reaches "
        f"{_f(100 * v['headroom_ratio'])}% of the `encode` ceiling it contains."
    )
    L.append("")
    L.append(
        "Answered by inversion, since the arrival rate is unknown: N users each "
        "issuing a query every T seconds need N/T q/s, so a plateau of X q/s says "
        "T >= N/X."
    )
    L.append("")
    L.append("| arm | 50 users need T >= | 10 users need T >= |")
    L.append("| --- | ---: | ---: |")
    for arm in REAL_ARMS:
        if arm in v["plateau"]:
            q = v["plateau"][arm]["qps"]
            L.append(f"| `{arm}` | {_f(50 / q, 2)} s | {_f(10 / q, 2)} s |")
    L.append("")

    # ---- 3. warm-up ----
    w = data.get("warmup", {})
    L.append("## 3. Warm-up, and the residue it leaves per topology")
    L.append("")
    L.append("| topology | warm-up ms | first query after it, ms |")
    L.append("| --- | ---: | ---: |")
    for arm in ("engine", "inproc"):
        entry = w.get("warm", {}).get(arm, {})
        L.append(
            f"| `{arm}` | {_f(entry.get('wall_ms', 0.0))} | "
            f"{_f(w.get('first_query_ms', {}).get(arm, 0.0))} |"
        )
    L.append("")
    L.append(
        "Both rows are probed. They were not when this section was first written: "
        "`warm_serving_caches` gated its probe retrieval on `with_rows`, which is "
        "False for the engine shape, so an engine-only process got none — and the "
        "probe's job is process-global CUDA/BLAS initialisation, which has nothing "
        "to do with which rows are resident. The engine arm is warmed **first** "
        "here, so its first-query cell can never be flattered by a probe the other "
        "arm ran."
    )
    L.append("")

    hc = data.get("hostname", {})
    if hc.get("fast_p50_ms"):
        L.append("## 3b. The same server under two spellings of its own hostname")
        L.append("")
        L.append(
            f"One dense `exact=True` search at depth {FETCH_DEPTH} against "
            f"`{hc.get('collection', '?')}`, arms ordered fast/slow/fast so a one-off "
            f"stall cannot be read as the effect."
        )
        L.append("")
        L.append("| url | p50 ms |")
        L.append("| --- | ---: |")
        L.append(f"| `{hc['fast_url']}` | {_f(hc['fast_p50_ms'])} |")
        L.append(
            f"| `{hc['slow_url']}` | "
            + (f"{_f(hc['slow_p50_ms'])}" if hc.get("slow_p50_ms") else "n/a")
            + " |"
        )
        L.append(f"| `{hc['fast_url']}` again | {_f(hc.get('fast_again_p50_ms') or 0.0)} |")
        L.append("")
        if hc.get("ratio"):
            L.append(
                f"**{_f(hc['ratio'])}x on a name.** `docker run -p 6333:6333` publishes "
                f"the port on IPv4 only, and `getaddrinfo` returns "
                f"{hc.get('resolves', {}).get('slow', [['?', '?']])[0][1]} first for the "
                f"slow spelling, so the client stack spends that time on an address the "
                f"server is not on before falling back. Every eval script in this repo "
                f"already passed the fast spelling; `QdrantHybridRetriever` and the "
                f"Streamlit UI defaulted to the slow one, so the published Qdrant "
                f"latencies were measured on a path the shipped default did not take. "
                f"Both defaults are now the fast spelling."
            )
            L.append("")

    # ---- 4. retriever construction ----
    rc = data.get("retriever_construction", {})
    if rc:
        L.append("## 4. The third construction nobody caches")
        L.append("")
        L.append(
            "`query_indices` built a fresh retriever on every call until this run. For "
            "the engine topology that instance owns the Qdrant client and a "
            "per-collection arm cache whose construction parses a 78k-term vocabulary "
            "sidecar off disk, so the whole thing was thrown away between queries — the "
            "embedder and the Index were cached, this was not. Three arms at C=1, the "
            "retrieval identical in all three; only where the retriever comes from "
            "differs."
        )
        L.append("")
        L.append("| retriever | p50 ms |")
        L.append("| --- | ---: |")
        L.append(f"| built fresh per query (the old shipped path) | {_f(rc['fresh_p50_ms'])} |")
        L.append(f"| one instance held across queries | {_f(rc['held_p50_ms'])} |")
        L.append(f"| **what the cache removes** | **{_f(rc['delta_ms'])}** |")
        if rc.get("routed_p50_ms"):
            L.append(
                f"| `route_query` as it now ships (`build_retriever_cached`) | "
                f"{_f(rc['routed_p50_ms'])} |"
            )
        L.append("")
        L.append(
            "`build_retriever_cached` is bounded at 4 and lives on the serving path "
            "only, the same rule the embedder and index caches follow — eval scripts "
            "keep constructing fresh, so no published number can move."
        )
        L.append("")

    # ---- 5. agreement ----
    ag = data.get("agreement", {})
    if ag:
        L.append("## 5. How far apart the two topologies rank (descriptive)")
        L.append("")
        L.append(
            f"Over {ag['n']} queries: **{ag['moved']}** of {10 * ag['n']} top-10 "
            f"positions hold a different chunk id, **{ag['out_of_tie']}** of them at a "
            f"rank where the two fused scores also differ. Largest fused-score gap at "
            f"any rank: {ag['max_score_gap']:.3e}."
        )
        L.append("")
        L.append(
            "**This is deliberately NOT a gate, and the reason is the point.** RRF "
            "consumes *ranks*, so a tie settled differently inside either arm -- which "
            "neither engine promises anything about, and which `qdrant_routed_check.md` "
            "measured directly (167 of 1,060 top-10 positions moved, every one of them "
            "inside a tie group) -- comes out of the fusion as a genuinely different "
            "fused score. Exactness is a claim about scores at the layer that computes "
            "them; here that layer is the dense and sparse arms, and the check that the "
            "two topologies serve the same data lives there, in `qdrant_routed_check.py` "
            "(C4/C4b/C5), not in this table."
        )
        L.append("")

    # ---- 6. rebuild under load ----
    race = data.get("race", {})
    if race:
        L.append("## 6. A rebuild landing underneath a running server")
        L.append("")
        L.append(
            f"{race['readers']} reader threads calling `load_index_cached` in a loop "
            f"for {race['seconds']:.0f}s per mode while a writer alternates two builds "
            f"of the same shape through the same four files. Each build stamps its "
            f"identity into the chunk text **and** into every vector, so a read that "
            f"pairs one build's chunks with another's rows is detectable — which the "
            f"row-alignment invariant itself cannot do."
        )
        L.append("")
        L.append(
            "| writer | inter-file gap | seal | reads | **mixed** | refused | torn | "
            "writes | final consistent |"
        )
        L.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for mode, st in race["modes"].items():
            L.append(
                f"| `{mode}` | {st.get('gap_s', 0.0):.2f}s | "
                f"{'yes' if st.get('sealed') else '**no (control)**'} | "
                f"{st['reads']:,} | **{st['mixed']:,}** | {st['refused']:,} | "
                f"{st['other_errors']:,} | {st['writes']:,} | "
                f"{'yes' if st['final_consistent'] else 'NO'} |"
            )
        L.append("")
        L.append(
            "**Read the third row first.** It is the negative control: the same rig with "
            "the reader made to treat the directory as unsealed, i.e. the behaviour that "
            "shipped before the seal. Its `mixed` count is what the other two rows would "
            "read without the fix — a check that cannot fail is not evidence, and this "
            "section would otherwise be a rig that never exercised the race."
        )
        L.append("")
        L.append(
            "`refused` is the designed behaviour, not a failure: a read that lands in a "
            "half-written directory raises rather than returning a pairing nothing "
            "downstream could detect. `torn` counts reads that hit a partially written "
            "file and raised on their own — always safe, never silent. The 0.15s gap is "
            "not a pessimistic invention: a real save writes ~234MB of `embeddings.npy` "
            "after the parquet, so the window a real rebuild leaves open is **seconds**, "
            "and the back-to-back row is the unrealistically easy one."
        )
        L.append("")

    served = data.get("race_served")
    if served:
        L.append("## 6b. The same race at REAL index size, through `route_query`")
        L.append("")
        L.append(
            f"Section 6 answers the mechanism on a 200x8 synthetic with readers calling "
            f"`load_index_cached` directly. This one answers the DEPLOYMENT question: "
            f"{served['readers']} reader threads issuing real `person` queries through the "
            f"shipped `route_query` (hybrid, `fetch_depth`={FETCH_DEPTH}, all three serving "
            f"caches live) against a scratch copy of "
            f"`{Path(served['source_dir']).name}` — **{served['n_chunks']:,} chunks, "
            f"dim {served['dim']}, {served['staged_bytes'] / 1024 / 1024:.0f} MB on disk** — "
            f"while a writer alternates two builds through the same four files, "
            f"{served['seconds']:.0f}s per mode."
        )
        L.append("")
        L.append(
            "| writer | seal | served | checked | **mixed** | refused | errors | writes "
            "| write cost | final consistent |"
        )
        L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for mode, st in served["modes"].items():
            L.append(
                f"| `{mode}` | {'yes' if st['sealed'] else '**no (control)**'} | "
                f"{st['served']:,} | {st['checked']:,} | **{st['mixed']:,}** | "
                f"{st['refused']:,} | {st['other_errors']:,} | {st['writes']:,} | "
                f"{st['median_gap_s']:.2f}s | "
                f"{'yes' if st['final_consistent'] else 'NO'} |"
            )
        L.append("")
        L.append(
            f"**Two windows, and only one of them is measured.** `write cost` is what "
            f"copying `embeddings.npy` actually costs at this size — the *truncated* "
            f"exposure, during which a reader's own before/after stamps already catch "
            f"the race. `stable window` ({served['mid_gap_s']:.0f}s) is **chosen**, and "
            f"it is the one the seal alone can see: both files complete, mismatched, "
            f"nothing moving."
        )
        L.append("")
        L.append(
            "**Why it had to be chosen, and why it had to be this large — measured in "
            "three steps rather than picked.** With **no** gap both modes returned "
            "**0 mixed**: `ArtifactStore.save` goes from `pq.write_table` straight into "
            "`np.save`, so at this size the exposure is dominated by a *truncated* file, "
            "which the formats themselves catch (`ValueError: Failed to read all data for "
            "array`, `JSONDecodeError`) — loud, not silent. At section 6's **0.15s** it "
            "was still 0, and that is structural: **a window shorter than a load cannot "
            "contain one**, so the reader's own stamps straddle the transition and the "
            "2026-08-21 stamping fix catches it *without* the seal. It fires only once "
            "the window exceeds a **contended** load — ~13s with several readers "
            "reloading 305MB against a writer copying the same — which is why the "
            "default is not the 1.5s an uncontended load costs."
        )
        L.append("")
        _sm = served["modes"]
        _sealed, _ctl = _sm.get("real_rebuild", {}), _sm.get("real_rebuild_unsealed", {})
        L.append(
            # Stated as PAIRS in prose, not only as table cells: a count is
            # invisible to D2 and traceable only to D5, which asks for the pair.
            f"At that window the control mixes **{_ctl.get('mixed', 0):,} of "
            f"{_ctl.get('checked', 0):,}** checked reads, while the sealed arm mixes "
            f"**{_sealed.get('mixed', 0):,} of {_sealed.get('checked', 0):,}**."
        )
        L.append("")
        L.append(
            "**So the seal's unique job is a real but NARROW one, and this section is "
            "what narrows it.** Against an overlapping rebuild the stamp comparison is "
            "sufficient on its own. What only the seal sees is a directory left stably "
            "inconsistent for **longer than a read** — which is not what `save` leaves "
            "at this size, but is exactly what an **in-place rewrite** leaves "
            "(`relabel_index_resolution_ids.py` rewrites for minutes), and what a small "
            "index leaves under section 6's conditions, where loads are fast enough to "
            "fit inside a short window."
        )
        L.append("")
        L.append(
            "**The scratch copy is the safety rule, not a convenience.** A writer runs in "
            "this loop, so pointing it at `data/index/` would destroy an index costing ~2h "
            "of GPU. `route_query` takes its `indices` list as an argument, so the redirect "
            "is a swapped `IndexInfo.dir` — no monkeypatching, and nothing can leak into "
            "the real tree."
        )
        L.append("")
        L.append(
            "**One seam, stated rather than hidden.** The *load* under test is the one a "
            "served query performs; the *check* reads the object the cache holds "
            "immediately afterwards, because `RetrievalResult` does not carry the Index. A "
            "write landing between the two shows up as a disagreement, not as a miss — "
            "which is why `checked` is reported separately from `served`."
        )
        L.append("")
        L.append(
            "**What the stamp costs the realism.** Both builds share their chunk TEXT, so "
            "`lexical.json` stays valid and BM25 is not rebuilt per read; the identity "
            "lives in a `chunk_id` prefix and in column 0 of every vector, i.e. 1 of "
            f"{served['dim']} dimensions, so the ranking a reader gets is still the real one."
        )
        L.append("")

    # ---- self-checks ----
    L.append("## 7. Self-checks")
    L.append("")

    def add(name: str, ok: bool | None, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    fastest_real = max(
        (v["plateau"][a]["qps"] for a in REAL_ARMS if a in v["plateau"]), default=0.0
    )
    null_q = v["plateau"].get("null", {}).get("qps", 0.0)
    add(
        "S1 the harness is not the bottleneck",
        null_q >= 100 * fastest_real,
        f"null {_f(null_q)} q/s vs fastest real arm {_f(fastest_real, 2)} q/s "
        f"({_f(null_q / fastest_real if fastest_real else 0.0)}x)",
    )

    pub = data.get("published_routed_p50")
    c1 = next(
        (r["p50_ms"] for r in data["curves"].get("inproc", []) if r["concurrency"] == 1),
        None,
    )
    if pub is None or c1 is None:
        add("S2 anchored against routed_fetch_depth_test.md", False, "UNPARSED — the cross-check could not be made")
    else:
        rel = abs(c1 - pub) / pub
        add(
            "S2 anchored against routed_fetch_depth_test.md",
            rel <= 0.40,
            f"inproc C=1 p50 {_f(c1)} ms vs published F=200 {_f(pub)} ms ({_f(100 * rel)}%)",
        )

    cachestat = data.get("cache_state", {})
    n_idx = cachestat.get("index", {}).get("size", 0)
    n_emb = cachestat.get("embedder", {}).get("size", 0)
    add(
        "S3 both caches resident, no errors in any cell",
        n_idx >= 4 and n_emb >= 1 and data.get("total_errors", 1) == 0,
        f"index cache {n_idx} entries, embedder cache {n_emb}, "
        f"{data.get('total_errors', '?')} errors across every cell",
    )

    # S4's domain has two halves, and BOTH are derived rather than chosen to
    # clear a failure.
    #  (a) latency floor: below ~10x CPython's switch interval a request's
    #      residence time is unresolvable by this instrument.
    #  (b) requests per worker: the dispatch counter is shared, so with r
    #      requests per worker the split is uneven by up to one request and the
    #      wall clock runs to ~(r+1)/r of the mean residence time. Little's
    #      ratio is therefore bounded ABOVE by r/(r+1) -- 0.67 at r=2 -- which
    #      is below the tolerance for arithmetic reasons, not physical ones.
    #      Inverting the tolerance itself gives the domain: r/(r+1) >= 0.85
    #      requires r >= 5.67, hence 6. (This is what caught inproc@C=50, which
    #      the budget leaves at exactly 2 requests per worker.)
    floor_ms = 10_000 * SWITCH_INTERVAL_S  # 10x the switch interval, in ms
    tol_lo, tol_hi = 0.85, 1.15
    min_per_worker = math.ceil(tol_lo / (1 - tol_lo))
    bad, thin = [], []
    tested = 0
    for arm in ARMS:
        for row in data["curves"].get(arm, []):
            if row["mean_ms"] < floor_ms:
                continue
            if row["per_worker"] < min_per_worker:
                thin.append(f"{arm}@C={row['concurrency']} r={row['per_worker']:.1f}")
                continue
            tested += 1
            little = row["throughput_qps"] * row["mean_ms"] / 1000.0 / row["concurrency"]
            if not tol_lo <= little <= tol_hi:
                bad.append(f"{arm}@C={row['concurrency']} {little:.2f}")
    add(
        "S4 Little's law holds where it is resolvable",
        not bad and tested > 0,
        f"{tested} cells in domain (mean >= {floor_ms:.0f} ms and >= {min_per_worker} "
        f"requests/worker); violations: {', '.join(bad) if bad else 'none'}"
        + (f"; below the request floor, not tested: {', '.join(thin)}" if thin else ""),
    )

    rep = data.get("repeat", {})
    drift = []
    for arm, row in rep.items():
        base = next(
            (r["throughput_qps"] for r in data["curves"].get(arm, []) if r["concurrency"] == 1),
            None,
        )
        if base:
            d = abs(row["throughput_qps"] - base) / base
            drift.append((arm, d))
    add(
        "S5 no machine drift across the run",
        bool(drift) and all(d <= 0.20 for _, d in drift),
        ", ".join(f"{a} {_f(100 * d)}%" for a, d in drift) or "not measured",
    )

    if ag:
        add(
            "S6 both topologies answered every query",
            ag["n"] > 0 and all(r["agree_at_10"] + r["moved"] == 10 for r in ag["rows"]),
            f"{ag['n']} queries, 10 results from each arm every time; ranking agreement "
            f"is descriptive (section 5), the exactness gate is qdrant_routed_check.py",
        )

    served = data.get("race_served")
    if served:
        sm = served["modes"]
        sealed_m = sm.get("real_rebuild", {})
        ctl = sm.get("real_rebuild_unsealed", {})
        add(
            "S10 a real-size rebuild under a served load is never mixed",
            bool(sealed_m)
            # checked > 0 IS THE CHECK. Its first run passed at "0 mixed of 0
            # checked": a contended load outlasted the pause between rebuilds,
            # so every read straddled the next cycle and the arm served nothing.
            # An arm that served no query cannot evidence that serving is safe.
            and sealed_m.get("checked", 0) > 0
            and sealed_m.get("mixed", 0) == 0
            and all(m["final_consistent"] for m in sm.values()),
            f"{sealed_m.get('mixed', 0)} mixed of {sealed_m.get('checked', 0):,} checked "
            f"reads ({sealed_m.get('served', 0):,} served) at {served['n_chunks']:,} chunks, "
            f"through the shipped route_query",
        )
        add(
            "S11 the real-size negative control reproduces the defect",
            ctl.get("mixed", 0) > 0,
            f"unsealed control served {ctl.get('mixed', 0):,} mixed of "
            f"{ctl.get('checked', 0):,} checked (0 would make S10 vacuous)",
        )
        add(
            "S12 the seal actually had to fire (the refusals are not incidental)",
            sealed_m.get("refused", 0) > 0 and sealed_m.get("writes", 0) > 0,
            f"{sealed_m.get('refused', 0):,} reads refused over "
            f"{sealed_m.get('writes', 0)} rebuilds, median {sealed_m.get('median_gap_s', 0.0):.2f}s "
            f"to write embeddings.npy -- a 0-mixed arm that never refused would mean the "
            f"window was never open, not that the seal held",
        )

    if race:
        modes = race["modes"]
        sealed = {k: m for k, m in modes.items() if m.get("sealed")}
        control = modes.get("gap_150ms_unsealed", {})
        mixed = sum(m["mixed"] for m in sealed.values())
        add(
            "S7 a rebuild under load is never served as a mixed Index",
            bool(sealed)
            and mixed == 0
            and all(m["final_consistent"] for m in modes.values()),
            f"{mixed} mixed reads over {sum(m['reads'] for m in sealed.values()):,} "
            f"sealed reads; the final read is consistent in every mode",
        )
        add(
            "S8 the negative control reproduces the defect",
            control.get("mixed", 0) > 0,
            f"unsealed control served {control.get('mixed', 0):,} mixed reads of "
            f"{control.get('reads', 0):,} (0 would mean the rig never exercised the race, "
            f"which would make S7 vacuous)",
        )
        add(
            "S9 the seal is what refuses, and it fired",
            sealed.get("gap_150ms", {}).get("refused", 0) > 0,
            f"{sealed.get('gap_150ms', {}).get('refused', 0):,} reads refused while the "
            f"writer was between its two files",
        )

    L.append("| check | verdict | detail |")
    L.append("| --- | --- | --- |")
    for name, ok, detail in checks:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    L.append("")
    return "\n".join(L) + "\n", checks


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:6333")
    ap.add_argument(
        "--slow-host",
        default="http://localhost:6333",
        help="the same server under a different hostname, for the section 3b control",
    )
    ap.add_argument("--budget", type=float, default=20.0, help="seconds per cell")
    ap.add_argument("--levels", default=",".join(str(c) for c in CONCURRENCY))
    ap.add_argument("--readers", type=int, default=8)
    ap.add_argument("--race-seconds", type=float, default=15.0)
    ap.add_argument(
        "--race-pause",
        type=float,
        default=25.0,
        help="seconds between rebuild cycles in section 6b. Must EXCEED one "
             "contended load or the sealed arm serves nothing between rebuilds and "
             "its 0-mixed result is vacuous; a deployment rebuilds occasionally, so "
             "a tight loop measures the refusal path instead",
    )
    ap.add_argument(
        "--race-mid-gap",
        type=float,
        default=15.0,
        help="section 6b's stable inconsistent window, i.e. the time both files "
             "are complete and mismatched. Must EXCEED one load (~1.5s here) or "
             "the before/after stamps catch the race without the seal and the "
             "negative control cannot fire",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny slice, to check it runs")
    ap.add_argument("--render", action="store_true", help="re-render from the raw cache")
    ap.add_argument(
        "--race-served",
        action="store_true",
        help="run ONLY section 6b (the real-size race through route_query) and "
             "re-render from the cached sweep, so the published throughput "
             "numbers are not re-measured to add a section",
    )
    args = ap.parse_args()

    if args.render or args.race_served:
        data = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        levels = [int(x) for x in args.levels.split(",")]
        entries = load_gold_query_set(GOLD)
        texts = [e.query for e in entries]
        if args.smoke:
            levels = [1, 2]
            texts = texts[:8]
            args.budget = 3.0
            args.race_seconds = 4.0
        indices = discover_indices(INDEX_ROOT)

        print(f"warming both topologies over {len(indices)} discovered indices ...")
        warmup = warm_everything(indices, args.url)
        for arm, ms in warmup["first_query_ms"].items():
            print(f"  first {arm} query after warm-up: {ms:,.1f} ms")

        data = {
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "budget_s": args.budget,
            "levels": levels,
            "n_queries": len(texts),
            "url": args.url,
            "warmup": warmup,
            "curves": {},
            "published_routed_p50": published_routed_p50(),
        }
        RAW.parent.mkdir(parents=True, exist_ok=True)

        print("hostname control ...")
        data["hostname"] = hostname_control(args.url, args.slow_host)
        print("checking the two topologies agree ...")
        data["agreement"] = agreement(indices, args.url, texts[: (8 if args.smoke else 24)])
        data["retriever_construction"] = retriever_construction(
            indices, args.url, texts, reps=(4 if args.smoke else 12)
        )

        arms = build_arms(indices, args.url, texts)
        for arm in ARMS:
            work, payloads = arms[arm]
            data["curves"][arm] = []
            for c in levels:
                # Warm AT THIS LEVEL before timing it. Added after S5 caught the
                # harness: only the repeat control did this, so `engine`'s C=1
                # cell -- the first sustained engine load of the whole run --
                # measured 6.30 q/s against the warmed repeat's 8.20 (30%),
                # while `encode` and `inproc` agreed to 6% because earlier
                # phases had already warmed them. A control warmed differently
                # from the treatment is not a control.
                run_level(work, payloads[:4], c, min(2.0, args.budget))
                row = run_level(work, payloads, c, args.budget)
                data["curves"][arm].append(row)
                print(
                    f"  {arm:8s} C={c:<3d} {row['throughput_qps']:8.2f} q/s  "
                    f"p50 {row['p50_ms']:8.1f} ms  n={row['n']:<5d} err={row['errors']}"
                )
                RAW.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

        # Drift control: C=1 again, at the END, warmed the same way every sweep
        # level was. Comparing a warm level against a cold repeat would report
        # the missing warm-up as drift.
        print("repeat control (C=1, at the end) ...")
        data["repeat"] = {}
        for arm in REAL_ARMS:
            work, payloads = arms[arm]
            run_level(work, payloads[:4], 1, 2.0)  # warm at this level first
            data["repeat"][arm] = run_level(work, payloads, 1, args.budget)

        data["cache_state"] = {"index": index_cache_info(), "embedder": embedder_cache_info()}
        data["total_errors"] = sum(
            r["errors"] for rows in data["curves"].values() for r in rows
        )

        tmp_root = Path(
            os.environ.get("TEMP", "/tmp")
        ) / "rag_lab_race"
        tmp_root.mkdir(parents=True, exist_ok=True)
        print("rebuild-under-load control ...")
        data["race"] = rebuild_under_load(tmp_root, args.readers, args.race_seconds)
        shutil.rmtree(tmp_root, ignore_errors=True)

        RAW.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.race_served:
        # Only section 6b. Everything else is re-rendered from the cached sweep:
        # adding a section must not silently re-measure the published throughput
        # numbers, which move ~6% run to run on this rig.
        data = json.loads(RAW.read_text(encoding="utf-8"))
        entries = load_gold_query_set(GOLD)
        texts = [e.query for e in entries if classify_query(e.query) == "person"]
        indices = discover_indices(INDEX_ROOT)
        spec = StrategySpec(type="hybrid", params={"fetch_depth": FETCH_DEPTH})
        tmp_root = Path(os.environ.get("TEMP", "/tmp")) / "rag_lab_race_served"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        print(f"real-size race under load ({len(texts)} person queries) ...")
        try:
            data["race_served"] = rebuild_under_load_served(
                tmp_root, indices, spec, texts, args.readers, args.race_seconds,
                args.race_pause, args.race_mid_gap)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
        if args.smoke:
            # A smoke slice may not publish. Its counts are a code check, not a
            # small version of the answer -- the same rule hybrid_weighted_
            # fetch_depth.py learned when its smoke reversed a headline's sign.
            print(json.dumps(data["race_served"], indent=1))
            print("smoke run -- nothing written")
            return
        RAW.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    text, checks = render(data)
    REPORT.write_text(text, encoding="utf-8")
    print(f"\nwrote {REPORT}")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


if __name__ == "__main__":
    main()
