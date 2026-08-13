"""Qdrant serving: which layer saturates first under concurrent load?

The pilot (`qdrant_pilot_test.py`) answered *does serving change the answers*
and left the question the stated deployment target actually needs unmeasured:
it ran **one query at a time in one process**, so it says nothing about
throughput, contention, or what happens when 50 people press Enter at once.

**Everything here is measured on the wrong machine on purpose, so the design
has to say which numbers survive the move.** The target is a faculty VM with a
*separate* GPU server; this box runs the embedder in-process on an RTX 3060 and
talks to Qdrant over loopback. So the deliverable is deliberately not "this
machine serves N users" -- that dies the moment the hardware changes. It is
**which layer saturates first, at what level, and what the other layer's cost
would have to become for the crossover to move**, which survives, plus an
`encode` curve that is a *substitutable parameter* rather than a result.

Four curves, measured separately, because one end-to-end number cannot be
recomposed for a different GPU:

    qdrant       dense (exact) + sparse, vector and tokens PRECOMPUTED
                 -> engine work alone. Transfers best: same container, same
                    data, same version; only CPU count differs.
    encode       embedder.embed_query alone
                 -> does NOT transfer (RTX 3060 vs the faculty GPU server), and
                    is exactly why it is measured on its own: substitute a new
                    plateau and the crossover can be re-derived without
                    re-running anything else.
    glue         tokenize + this repo's RRF over CACHED rankings
                 -> the pure-Python work `end_to_end` does and neither arm above
                    performs. Measured rather than assumed negligible, so the
                    composition check has no unexplained residue to absorb.
    end_to_end   encode + tokenize + both retrievals + this repo's RRF
                 -> the composition check. If it is materially below what the
                    three component curves allow, there is a layer none shows.

plus a `null` arm (the harness doing nothing), because a load generator that is
itself the bottleneck measures Python's thread scheduler and reports it as an
engine limit.

**A known confound, stated before the run and pointing the safe way.** The
embedder here lives in the *same process* that issues the requests, so GIL
contention and CUDA serialisation are bundled with request handling; in the
target topology a network hop separates them. That makes this machine look
*worse* at the app layer than the deployment will, so a clean result is
conservative and a dirty one needs decomposing before it is believed.

One control is worth naming here because it corrects a published figure rather
than guarding this run: **control 4** re-measures `encode` at C=1 with a fixed
idle gap before each call. `cost_latency_pareto.md`'s 82.94 ms for the same
model on this box is ~6x this arm's C=1 cost, and the cause is that an idle GPU
downclocks -- that report's loop leaves ~0.26 s and ~1.46 s of retrieval between
its two encodes. So 82.94 ms is encode-after-idle (the low-load regime) and this
arm is encode-under-load (the serving regime); both are right about different
questions, and §5b prints the curve that says so.

Closed loop: C worker threads, each issuing the next query the moment its
previous one returns. That measures capacity, not a realistic arrival process --
which is the point, since the arrival rate is exactly the thing nobody knows
yet. The target is answered by inversion instead: 50 users at one query every
T seconds needs 50/T q/s, so a measured plateau of X q/s says T >= 50/X.

Everything is cached to `data/results/qdrant_concurrency_raw.json` after every
(arm, level), so a crash costs one cell and `--render` is free.

Run (server up and ingested by tools/eval/qdrant_pilot_ingest.py; nothing else
on the GPU):

    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/qdrant_concurrency_test.py
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qdrant_client import QdrantClient  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.retrievers.qdrant_retriever import (  # noqa: E402
    QdrantRetriever,
    QdrantSparseRetriever,
)
from rag_lab.schema import Query  # noqa: E402

sys.path.insert(0, str(REPO / "tools/eval"))
from qdrant_pilot_test import rrf_fuse  # noqa: E402  -- one copy of the fusion

INDEX_DIR = REPO / "data/index/chunker_compare_full/plain__sentence__local__bf8b7ebb"
COMBO_ID = INDEX_DIR.name
GOLD = REPO / "config/eval/gold_query_set_73det.yaml"
RAW = REPO / "data/results/qdrant_concurrency_raw.json"
REPORT = REPO / "data/results/qdrant_concurrency.md"
PILOT_RAW = REPO / "data/results/qdrant_pilot_raw.json"

K = 10
FETCH_DEPTH = 200          # what the UI ships, and what the pilot measured at
CONCURRENCY = [1, 2, 5, 10, 25, 50]
ARMS = ["null", "qdrant", "encode", "glue", "end_to_end"]
COMPONENTS = ["encode", "qdrant", "glue"]      # what end_to_end is composed of

# Control 4. Encode the same queries at C=1 with a fixed sleep before each one.
# `cost_latency_pareto.md` publishes bge_m3 encode p50 = 82.94 ms and this
# script's C=1 `encode` arm is ~6x faster on the same model, same box, same
# query set; the difference is that the pareto loop leaves ~0.26 s and ~1.46 s
# of retrieval between its two encodes, and an idle GPU downclocks. The gaps
# below bracket that loop, so the disagreement is a measurement here rather
# than a story. 1.8 s is the pareto iteration's own total non-encode work.
IDLE_GAPS = [0.0, 0.5, 1.0, 1.8]

# Pre-registered before the run. Frozen here rather than in prose so the verdict
# is computed from the numbers instead of read off them afterwards.
DECISION_RULE = """
Let plateau(arm) = the highest throughput (q/s) that arm reaches at any
concurrency level measured.

1. ENCODE-BOUND   if plateau(encode) < plateau(qdrant).
       -> the GPU server is the binding constraint; one Qdrant is plenty and
          its remaining headroom is stated as a bound.
2. ENGINE-BOUND   if plateau(qdrant) < plateau(encode).
       -> sizing/sharding Qdrant matters more than the GPU; the engine curve is
          the one to re-measure on the VM, since it is the one that transfers.
3. APP-BOUND      if end_to_end falls materially below what its own three
       components allow. Reported IN ADDITION to 1/2, never instead of.

       predicted(C) = 1 / (1/encode(C) + 1/qdrant(C) + 1/glue(C))

       i.e. the HARMONIC combination at the SAME concurrency, because a worker
       runs the three stages back to back: by Little's law its residence times
       add, and throughput is C divided by the sum. APP-BOUND if the ratio
       end_to_end(C*)/predicted(C*) < 0.85, where C* is the level at which
       end_to_end peaks. 0.85 is chosen in advance and the full ratio column is
       printed at every level, so the threshold is not load-bearing.
""".strip()

# Correction, recorded rather than quietly fixed. The rule first written here
# used `0.85 * min(plateau(encode), plateau(qdrant))`, which is wrong by
# construction: two stages a worker runs SERIALLY cannot both run at their solo
# plateau, so min() is not a ceiling any composing system reaches. The smoke
# slice exposed it -- it declared APP-BOUND at ratio 0.443 on a system that in
# fact composes almost exactly (harmonic prediction 28.4 q/s vs 29.96 measured
# at C=4, 28.98 vs 30.81 at C=2). The `glue` arm exists for the same reason: the
# residual the harmonic law could not explain at C=1 was `end_to_end`'s
# tokenize + RRF fuse, work that neither component arm performs, so it is
# measured as a component instead of being absorbed by loosening 0.85 after
# seeing the number. No real number existed when either change was made.

# n grows with C so every level keeps ~20 requests per worker; below that a p95
# is being read off a handful of samples.
def n_requests(c: int, n_queries: int) -> int:
    return max(2 * n_queries, 20 * c)


# --------------------------------------------------------------------------- #
# load generator
# --------------------------------------------------------------------------- #
def run_level(work, payloads: list, c: int, n: int) -> dict:
    """Closed loop: c threads, n requests total, each thread taking the next
    index the moment its previous request returns.

    `work(payload, slot)` gets a `slot` in [0, c) so an arm can hand each thread
    its own client without the arm knowing how the loop is scheduled.

    Dispatch is `next(itertools.count())`, which is atomic under CPython's GIL,
    rather than a shared counter behind a `threading.Lock`. The lock version was
    real untimed overhead *outside* the timed region: a thread that blocks on a
    contended lock can wait a full GIL switch interval (5 ms by default), which
    is nothing against a 25 ms retrieval but is several times the `glue` arm's
    own 0.3 ms of work. It showed up as S4 failing for `glue` alone -- throughput
    x mean latency came out 1.30 at C=4 -- i.e. the harness was reporting its own
    dispatch cost as concurrency that never happened. Counter dispatch keeps the
    work-stealing that makes this a closed loop and removes the mutex.
    """
    counter = itertools.count()
    lats: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()          # results only, once per worker, not per request
    barrier = threading.Barrier(c)

    def worker(slot: int) -> None:
        local: list[float] = []
        local_err: list[str] = []
        barrier.wait()
        while True:
            i = next(counter)
            if i >= n:
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
        "wall_s": wall,
        "throughput_qps": len(lats) / wall if wall > 0 else 0.0,
        "p50_ms": statistics.median(lats) if lats else 0.0,
        "p95_ms": lats[int(0.95 * (len(lats) - 1))] if lats else 0.0,
        "p99_ms": lats[int(0.99 * (len(lats) - 1))] if lats else 0.0,
        "mean_ms": statistics.fmean(lats) if lats else 0.0,
        "errors": len(errors),
        "error_sample": errors[:3],
    }


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def build_arms(args, index, entries, embedder, tokenize):
    """Returns (arms, precomputed) where arms[name] = (work_fn, payloads).

    Each worker slot gets its **own** Qdrant client. That isolates the engine
    from client-pool contention, which is a separate, app-layer question -- and
    it is measured separately as the `shared client` control rather than left
    as an assumption. The sparse retriever is cloned shallowly so all slots
    share one parsed 78k-term vocabulary (a per-slot parse would cost ~12 MB and
    a JSON parse each, measuring the harness).
    """
    cmax = max(CONCURRENCY)
    vocab_path = REPO / "data/qdrant" / args.collection / "vocab.json"

    dense_slots = [
        QdrantRetriever(
            url=args.url,
            collection_name=args.collection,
            vector_name="dense",
            exact=True,          # the pilot's serving recommendation
            timeout=args.timeout,
        )
        for _ in range(cmax)
    ]
    sparse_proto = QdrantSparseRetriever(
        vocab_path=str(vocab_path),
        url=args.url,
        collection_name=args.collection,
        timeout=args.timeout,
    )
    sparse_slots = [sparse_proto]
    for _ in range(cmax - 1):
        clone = copy.copy(sparse_proto)          # shares _vocab, read-only here
        clone._client = QdrantClient(url=args.url, timeout=args.timeout)
        sparse_slots.append(clone)

    texts = [e.query for e in entries]
    vectors = [embedder.embed_query(t) for t in texts]        # also S1's reference
    tokens = [tokenize(t) for t in texts]
    prepared = [
        Query(text=t, vector=v, tokens=tk) for t, v, tk in zip(texts, vectors, tokens)
    ]

    # One serial pass, reused twice: it is `glue`'s input and S2's anchor, so
    # the rankings the fusion runs over are the same ones checked against the
    # pilot's published cache rather than a second, unverified retrieval.
    cached = [
        (dense_slots[0].retrieve(q, index, FETCH_DEPTH),
         sparse_slots[0].retrieve(q, index, FETCH_DEPTH))
        for q in prepared
    ]

    def w_null(payload, slot):
        return None

    def w_qdrant(q, slot):
        d = dense_slots[slot].retrieve(q, index, FETCH_DEPTH)
        l = sparse_slots[slot].retrieve(q, index, FETCH_DEPTH)
        return d, l

    def w_encode(text, slot):
        return embedder.embed_query(text)

    def w_glue(payload, slot):
        text, d, l = payload
        tokenize(text)
        return rrf_fuse(d, l, K)

    def w_end_to_end(text, slot):
        q = Query(text=text, vector=embedder.embed_query(text), tokens=tokenize(text))
        d = dense_slots[slot].retrieve(q, index, FETCH_DEPTH)
        l = sparse_slots[slot].retrieve(q, index, FETCH_DEPTH)
        return rrf_fuse(d, l, K)

    glue_payloads = [(t, d, l) for t, (d, l) in zip(texts, cached)]

    arms = {
        "null": (w_null, texts),
        "qdrant": (w_qdrant, prepared),
        "encode": (w_encode, texts),
        "glue": (w_glue, glue_payloads),
        "end_to_end": (w_end_to_end, texts),
    }
    return arms, {"texts": texts, "vectors": vectors, "prepared": prepared,
                  "cached": cached,
                  "dense_slots": dense_slots, "sparse_slots": sparse_slots}


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
def collect(args) -> dict:
    from pythainlp.tokenize import word_tokenize

    from rag_lab.factory import build_embedder

    index = ArtifactStore().load(INDEX_DIR)
    entries = load_gold_query_set(GOLD)
    if args.limit:
        entries = entries[: args.limit]
    manifest = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))

    def tokenize(t: str) -> list[str]:
        return word_tokenize(t)

    tokenize("อุ่นเครื่อง")   # pythainlp builds its trie on first call
    print("preparing payloads (encoding every query once, sequentially)...", flush=True)
    arms, pre = build_arms(args, index, entries, embedder, tokenize)
    nq = len(entries)

    raw: dict = {
        "smoke": bool(args.smoke),
        "combo_id": COMBO_ID,
        "collection": args.collection,
        "url": args.url,
        "k": K,
        "fetch_depth": FETCH_DEPTH,
        "n_chunks": len(index.chunks),
        "n_queries": nq,
        "concurrency": CONCURRENCY,
        "decision_rule": DECISION_RULE,
        "levels": [],
        "controls": {},
        "verify": {},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def flush() -> None:
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    # --- S2's anchor: the same top-10s the pilot published, from this path ---
    # Read off the rankings `glue` actually fuses, not a second retrieval: a
    # fresh pass would leave the fused arm's input unchecked, which is the half
    # that can silently differ.
    print("verifying the served arm against the pilot's cache...", flush=True)
    served_top10 = [[r.chunk_id for r in d[:K]] for d, _ in pre["cached"]]
    raw["verify"]["dense_top10"] = served_top10
    raw["verify"]["queries"] = pre["texts"]

    # --- the load sweep ---
    for arm in ARMS:
        work, payloads = arms[arm]
        for c in CONCURRENCY:
            n = n_requests(c, nq)
            run_level(work, payloads, min(c, 4), min(n, 4 * 8))       # warm-up
            row = run_level(work, payloads, c, n)
            row["arm"] = arm
            raw["levels"].append(row)
            flush()
            print(
                f"  {arm:11s} C={c:<3d} {row['throughput_qps']:8.2f} q/s  "
                f"p50 {row['p50_ms']:8.1f} ms  p95 {row['p95_ms']:8.1f} ms"
                f"{'  ERRORS ' + str(row['errors']) if row['errors'] else ''}",
                flush=True,
            )

    # --- control 1: repeat C=1 at the end (drift over the whole run) ---
    # Warmed exactly as the sweep warms every level. Without it this control
    # compares a warmed measurement against a cold one and calls the difference
    # drift: `encode` landed 35.5% high here, because the preceding arm leaves
    # the GPU idle for seconds and an idle GPU downclocks -- the very effect
    # control 4 below measures. Two things being compared have to be measured
    # the same way, or the control is reporting its own asymmetry.
    for arm in ARMS:
        work, payloads = arms[arm]
        run_level(work, payloads, 1, min(n_requests(1, nq), 32))          # warm-up
        row = run_level(work, payloads, 1, n_requests(1, nq))
        row["arm"] = arm
        raw["controls"].setdefault("repeat_c1", []).append(row)
    flush()

    # --- control 2: one shared client at max C (client-pool contention) ---
    shared_dense = QdrantRetriever(
        url=args.url, collection_name=args.collection, vector_name="dense",
        exact=True, timeout=args.timeout,
    )
    shared_sparse = pre["sparse_slots"][0]

    def w_shared(q, slot):
        return (
            shared_dense.retrieve(q, index, FETCH_DEPTH),
            shared_sparse.retrieve(q, index, FETCH_DEPTH),
        )

    cmax = max(CONCURRENCY)
    row = run_level(w_shared, pre["prepared"], cmax, n_requests(cmax, nq))
    row["arm"] = "qdrant (shared client)"
    raw["controls"]["shared_client"] = row
    flush()

    # --- control 3: does concurrent encoding return the same vectors? ---
    out: list = [None] * nq
    lock = threading.Lock()
    state = {"i": 0}

    def enc_worker():
        while True:
            with lock:
                i = state["i"]
                if i >= nq:
                    return
                state["i"] = i + 1
            out[i] = embedder.embed_query(pre["texts"][i])

    ts = [threading.Thread(target=enc_worker, daemon=True) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    deltas = [float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
              for a, b in zip(out, pre["vectors"])]
    raw["controls"]["concurrent_encode_max_abs_delta"] = max(deltas)
    flush()

    # --- control 4: encode cost as a function of how long the GPU sat idle ---
    # Not a curiosity: it is why this script's `encode` arm disagrees with
    # `cost_latency_pareto.md` by ~6x for the same model on the same box, and it
    # is measured here rather than argued in prose.
    # Each row is warmed AT ITS OWN GAP, discarded, before it is timed. The clock
    # state this control exists to measure is a property of the recent duty cycle,
    # not of the call being timed, so an unwarmed row measures the transition from
    # whatever ran before it rather than the steady state at that gap. Skipping
    # this is what made the zero-gap row disagree with the `encode` arm by 38.5%
    # on a smoke slice (S7's own subject) -- the same asymmetry the repeat-C=1
    # control hit above, for the same reason.
    gap_rows = []
    for gap in IDLE_GAPS:
        ts_ms = []
        for text in pre["texts"][:3]:                                   # warm-up
            if gap:
                time.sleep(gap)
            embedder.embed_query(text)
        for text in pre["texts"][: min(nq, 25)]:
            if gap:
                time.sleep(gap)
            t0 = time.perf_counter()
            embedder.embed_query(text)
            ts_ms.append((time.perf_counter() - t0) * 1000)
        gap_rows.append({
            "idle_gap_s": gap,
            "n": len(ts_ms),
            "p50_ms": statistics.median(ts_ms),
            "p95_ms": float(np.percentile(ts_ms, 95)),
            "mean_ms": statistics.fmean(ts_ms),
        })
        flush()
    raw["controls"]["encode_idle_gap"] = gap_rows
    flush()
    return raw


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def plateau(raw: dict, arm: str) -> float:
    vals = [r["throughput_qps"] for r in raw["levels"] if r["arm"] == arm]
    return max(vals) if vals else 0.0


def at(raw: dict, arm: str, c: int) -> float:
    for r in raw["levels"]:
        if r["arm"] == arm and r["concurrency"] == c:
            return r["throughput_qps"]
    return 0.0


def composition(raw: dict) -> list[dict]:
    """Per-C harmonic prediction for `end_to_end` from its three components.

    A worker runs encode -> retrieve -> fuse back to back, so at a fixed C its
    residence times add and throughputs combine reciprocally. Compared at the
    SAME C, never against each arm's own best level.
    """
    rows = []
    for c in raw["concurrency"]:
        parts = {a: at(raw, a, c) for a in COMPONENTS}
        if all(parts.values()):
            predicted = 1.0 / sum(1.0 / v for v in parts.values())
        else:
            predicted = 0.0
        measured = at(raw, "end_to_end", c)
        rows.append({
            "concurrency": c,
            **{f"{a}_qps": parts[a] for a in COMPONENTS},
            "predicted_qps": predicted,
            "measured_qps": measured,
            "ratio": (measured / predicted) if predicted else 0.0,
        })
    return rows


def verdict(raw: dict) -> tuple[str, str, float, int]:
    p_enc, p_qd = plateau(raw, "encode"), plateau(raw, "qdrant")
    primary = "ENCODE-BOUND" if p_enc < p_qd else "ENGINE-BOUND"
    rows = composition(raw)
    # C* = where end_to_end peaks, fixed by the rule rather than chosen to suit
    # the ratio.
    best = max(rows, key=lambda r: r["measured_qps"], default=None)
    ratio = best["ratio"] if best else 0.0
    app = "APP-BOUND" if ratio < 0.85 else "composition holds"
    return primary, app, ratio, (best["concurrency"] if best else 0)


def render(raw: dict) -> str:
    L = raw["levels"]
    lines = [
        "# Qdrant serving under concurrent load: which layer saturates first?",
        "",
        *(
            [
                "> **SMOKE RUN -- NOT A RESULT.** A handful of queries at low "
                "concurrency checks that the harness runs; it does not measure a "
                "plateau. §3's verdict is rendered from meaningless numbers and "
                "must not be read or cited.",
                "",
            ]
            if raw.get("smoke")
            else []
        ),
        "Generated by `tools/eval/qdrant_concurrency_test.py`"
        f" ({raw['generated_at']}); raw cache `data/results/qdrant_concurrency_raw.json`.",
        "",
        f"- Collection `{raw['collection']}` ({raw['n_chunks']:,} chunks) on `{raw['url']}`,"
        f" dense arm at `exact=True`",
        f"- {raw['n_queries']} Gold 73det queries replayed closed-loop, "
        f"k={raw['k']}, fetch_depth={raw['fetch_depth']}",
        "- Each worker slot holds its **own** Qdrant client; the shared-client case "
        "is a separate control below.",
        "",
        "**This machine is not the deployment target.** The embedder runs "
        "in-process on an RTX 3060 and Qdrant is on loopback; the target is a "
        "faculty VM with a separate GPU server. What transfers is *which layer "
        "saturates first* and the `qdrant` curve (same container, same data, "
        "same version -- only CPU count differs). The `encode` curve does not "
        "transfer and is measured alone so a different GPU's plateau can be "
        "substituted without re-running anything else. In-process encoding also "
        "bundles GIL contention with request handling, which a network hop would "
        "separate -- so this rig understates the app layer's real headroom.",
        "",
        "## 1. Pre-registered decision rule",
        "",
        "```",
        raw["decision_rule"],
        "```",
        "",
        "## 2. Throughput and latency by concurrency",
        "",
        "| arm | C | q/s | p50 ms | p95 ms | p99 ms | requests | errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in L:
        lines.append(
            f"| {r['arm']} | {r['concurrency']} | {r['throughput_qps']:.2f} | "
            f"{r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['p99_ms']:.1f} | "
            f"{r['n']} | {r['errors']} |"
        )

    p_enc, p_qd, p_e2e = (plateau(raw, a) for a in ("encode", "qdrant", "end_to_end"))
    p_glue = plateau(raw, "glue")
    primary, app, ratio, cstar = verdict(raw)
    lines += [
        "",
        "## 3. Verdict, by the rule above",
        "",
        f"- plateau(`qdrant`) = **{p_qd:.2f} q/s**",
        f"- plateau(`encode`) = **{p_enc:.2f} q/s**",
        f"- plateau(`glue`) = **{p_glue:.2f} q/s**",
        f"- plateau(`end_to_end`) = **{p_e2e:.2f} q/s**",
        "",
        "Composition is checked **at matched concurrency**, not against each "
        "arm's own best level: a worker runs the three stages serially, so its "
        "residence times add and the throughputs combine reciprocally. The whole "
        "column is printed so the 0.85 line is not load-bearing.",
        "",
        "| C | encode q/s | qdrant q/s | glue q/s | predicted | measured | ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in composition(raw):
        star = " *" if r["concurrency"] == cstar else ""
        lines.append(
            f"| {r['concurrency']}{star} | {r['encode_qps']:.2f} | {r['qdrant_qps']:.2f} | "
            f"{r['glue_qps']:.2f} | {r['predicted_qps']:.2f} | "
            f"{r['measured_qps']:.2f} | {r['ratio']:.3f} |"
        )
    lines += [
        "",
        f"`*` = C\\* , the level at which `end_to_end` peaks; the rule reads the "
        f"ratio there: **{ratio:.3f}**.",
        "",
        f"**{primary}** -- and the composition check says **{app}**.",
        "",
        "## 4. What that says about the stated target",
        "",
        "The arrival rate is the one thing nobody knows yet, so the target is "
        "answered by inversion rather than simulated: **U users each issuing a "
        "query every T seconds need U/T q/s.**",
        "",
        "| users | q/s needed at T=5 s | at T=10 s | at T=30 s |",
        "|---|---|---|---|",
    ]
    for u in (5, 25, 50):
        lines.append(f"| {u} | {u / 5:.1f} | {u / 10:.1f} | {u / 30:.2f} |")
    lines += [
        "",
        f"Measured end-to-end plateau is **{p_e2e:.2f} q/s** on this hardware, so "
        f"a sustained {50} concurrent users need T >= "
        f"**{50 / p_e2e:.1f} s** per user between queries at this plateau. "
        "Read that as the shape of the constraint, not as a capacity promise for "
        "the VM.",
        "",
        "## 5. Controls",
        "",
        "| control | value |",
        "|---|---|",
    ]
    sc = raw["controls"].get("shared_client")
    if sc:
        qd_max = next(
            (r for r in L if r["arm"] == "qdrant" and r["concurrency"] == sc["concurrency"]),
            None,
        )
        lines.append(
            f"| one shared client at C={sc['concurrency']} | {sc['throughput_qps']:.2f} q/s "
            f"vs {qd_max['throughput_qps']:.2f} q/s per-thread clients |"
        )
    for r in raw["controls"].get("repeat_c1", []):
        first = next(
            (x for x in L if x["arm"] == r["arm"] and x["concurrency"] == 1), None
        )
        if first:
            drift = (r["p50_ms"] - first["p50_ms"]) / first["p50_ms"] * 100 if first["p50_ms"] else 0
            lines.append(
                f"| `{r['arm']}` C=1 re-measured at the end | {r['p50_ms']:.1f} ms "
                f"vs {first['p50_ms']:.1f} ms ({drift:+.1f}%) |"
            )
    d = raw["controls"].get("concurrent_encode_max_abs_delta")
    if d is not None:
        lines.append(f"| concurrent encode vs sequential, max abs delta | {d:.3e} |")

    gaps = raw["controls"].get("encode_idle_gap") or []
    if gaps:
        enc1 = next((r for r in L if r["arm"] == "encode" and r["concurrency"] == 1), None)
        slowest = max(gaps, key=lambda r: r["p50_ms"])
        fastest = min(gaps, key=lambda r: r["p50_ms"])
        lines += [
            "",
            "### 5b. Why this script's `encode` arm disagrees with "
            "`cost_latency_pareto.md`",
            "",
            "That report publishes `bge_m3` encode p50 = **82.94 ms**; the `encode` "
            "arm above is several times faster, same model, same box, same query "
            "set. Both are correct and the difference is not noise -- **encode cost "
            "here is a function of how long the GPU sat idle beforehand**, and the "
            "pareto loop leaves ~0.26 s and ~1.46 s of retrieval between its two "
            "encodes per iteration. Control 4 measures that directly, at C=1, by "
            "inserting a fixed sleep before each encode:",
            "",
            "| idle gap before each encode | n | p50 ms | p95 ms | mean ms |",
            "|---|---|---|---|---|",
        ]
        for g in gaps:
            lines.append(
                f"| {g['idle_gap_s']:.1f} s | {g['n']} | {g['p50_ms']:.2f} | "
                f"{g['p95_ms']:.2f} | {g['mean_ms']:.2f} |"
            )
        ratio_gap = slowest["p50_ms"] / fastest["p50_ms"] if fastest["p50_ms"] else 0.0
        lines += [
            "",
            f"Across the grid p50 moves **{fastest['p50_ms']:.2f} -> "
            f"{slowest['p50_ms']:.2f} ms ({ratio_gap:.1f}x)** with nothing changed "
            "but the wait. So **82.94 ms is not a property of the embedder** -- it "
            "is encode-after-an-idle-GPU, i.e. the low-load regime, while this "
            "script's arm measures encode under sustained load, which is the "
            "serving regime. Cite the pareto figure for a lightly-loaded system "
            "and this one for a busy one; neither supersedes the other."
            + (
                f" The gap=0 row ({fastest['p50_ms']:.2f} ms p50) is the one that "
                f"should agree with the C=1 `encode` arm "
                f"({enc1['p50_ms']:.2f} ms), and S7 gates that."
                if enc1
                else ""
            ),
        ]

    lines += ["", "## 6. Self-checks", "", "| check | result | detail |", "|---|---|---|"]
    for c in self_checks(raw):
        lines.append(
            f"| {c['check']} | {'PASS' if c['ok'] else 'FAIL'} | {c['detail']} |"
        )
    lines += [
        "",
        "## 7. What this does NOT establish",
        "",
        "- One collection, one combo, one route. The other three routed "
        "collections are not ingested, and a real deployment holds all four -- "
        "which changes the engine's memory footprint, not its per-query cost.",
        "- No network between the app and either the embedder or the engine.",
        "- Closed loop, so there is no queueing behaviour under a bursty arrival "
        "process, and no measurement of what a request waits behind.",
        "- Nothing about answer quality: every arm here returns what the pilot "
        "already scored.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# self-checks
# --------------------------------------------------------------------------- #
def self_checks(raw: dict) -> list[dict]:
    checks: list[dict] = []

    d = raw["controls"].get("concurrent_encode_max_abs_delta")
    checks.append(
        {
            "check": "S1 concurrent encoding returns the same vectors as sequential",
            "ok": d is not None and d <= 1e-3,
            "detail": f"max abs delta {d:.3e} over {raw['n_queries']} queries"
            if d is not None
            else "not measured",
        }
    )

    same = miss = 0
    if PILOT_RAW.exists() and raw["verify"].get("dense_top10"):
        pilot = json.loads(PILOT_RAW.read_text(encoding="utf-8"))
        cached = {
            r["query"]: [x["chunk_id"] for x in r["arms"]["qdrant_exact"][: raw["k"]]]
            for r in pilot["rows"]
        }
        for q, top in zip(raw["verify"]["queries"], raw["verify"]["dense_top10"]):
            if q in cached and cached[q] == top:
                same += 1
            elif q in cached:
                miss += 1
    checks.append(
        {
            "check": "S2 the served dense arm reproduces the pilot's cached top-10s",
            "ok": miss == 0 and same > 0,
            "detail": f"{same} identical, {miss} differ (of {raw['n_queries']})",
        }
    )

    worst_arm, worst = "", 0.0
    for r in raw["controls"].get("repeat_c1", []):
        first = next(
            (x for x in raw["levels"] if x["arm"] == r["arm"] and x["concurrency"] == 1),
            None,
        )
        if first and first["p50_ms"] > 0 and r["arm"] != "null":
            drift = abs(r["p50_ms"] - first["p50_ms"]) / first["p50_ms"]
            if drift > worst:
                worst_arm, worst = r["arm"], drift
    checks.append(
        {
            "check": "S3 no drift across the run (C=1 re-measured at the end)",
            "ok": worst <= 0.25,
            "detail": f"worst |drift| {worst * 100:.1f}% (`{worst_arm}`)",
        }
    )

    # Little's law: in a closed loop with C workers, X * R == C. It is a check on
    # the HARNESS, not on the system, so it needs a stated domain -- a thread's
    # residence can only be measured to the granularity of the interpreter's own
    # scheduler, and CPython preempts on a switch interval (5 ms by default).
    # `glue` costs ~0.3 ms per request, i.e. an order of magnitude BELOW one
    # quantum, so at C>1 its measured residence is scheduler noise and the
    # identity is unresolvable there by construction. The floor is taken from
    # sys.getswitchinterval() rather than picked to clear the arm that failed,
    # and the skipped rows are named. S9 shows the exemption is safe: `glue`'s
    # reciprocal contributes ~0.1% of the harmonic prediction, so mismeasuring
    # it cannot move the verdict -- which is itself the finding about that layer.
    quantum_ms = sys.getswitchinterval() * 1000.0
    bad, skipped = [], []
    for r in raw["levels"]:
        if r["concurrency"] == 1 or r["arm"] == "null" or r["mean_ms"] <= 0:
            continue
        if r["mean_ms"] < quantum_ms:
            skipped.append(f"{r['arm']}@C={r['concurrency']}")
            continue
        implied = r["throughput_qps"] * r["mean_ms"] / 1000.0
        if abs(implied - r["concurrency"]) / r["concurrency"] > 0.10:
            bad.append(f"{r['arm']}@C={r['concurrency']}:{implied:.2f}")
    note = (
        f"; {len(skipped)} rows below the {quantum_ms:.0f} ms GIL quantum not "
        f"checkable ({', '.join(sorted({s.split('@')[0] for s in skipped}))})"
        if skipped
        else ""
    )
    checks.append(
        {
            "check": "S4 closed-loop identity holds (throughput x mean latency == C)",
            "ok": not bad,
            "detail": ("all within 10%" if not bad else "; ".join(bad[:4])) + note,
        }
    )

    p_null, p_e2e = plateau(raw, "null"), plateau(raw, "end_to_end")
    checks.append(
        {
            "check": "S5 the harness is not the bottleneck",
            "ok": p_e2e > 0 and p_null >= 20 * p_e2e,
            "detail": f"null {p_null:,.0f} q/s vs end_to_end {p_e2e:.2f} q/s "
            f"({p_null / p_e2e:.0f}x)" if p_e2e else "n/a",
        }
    )

    errs = sum(r["errors"] for r in raw["levels"])
    checks.append(
        {
            "check": "S6 no request errored at any concurrency level",
            "ok": errs == 0,
            "detail": f"{errs} errors across {len(raw['levels'])} levels",
        }
    )

    # The idle-gap curve is only evidence about the `encode` arm if its own
    # zero-gap point IS that arm: same model, same queries, same C=1, differing
    # in nothing but the sleep. Anchored rather than assumed, because a control
    # measuring something adjacent would explain a discrepancy it never touched.
    gaps = raw["controls"].get("encode_idle_gap") or []
    g0 = next((g for g in gaps if g["idle_gap_s"] == 0.0), None)
    enc1 = next(
        (r for r in raw["levels"] if r["arm"] == "encode" and r["concurrency"] == 1), None
    )
    rel = (
        abs(g0["p50_ms"] - enc1["p50_ms"]) / enc1["p50_ms"]
        if g0 and enc1 and enc1["p50_ms"] > 0
        else None
    )
    checks.append(
        {
            "check": "S7 the idle-gap control's zero-gap point is the C=1 `encode` arm",
            "ok": rel is not None and rel <= 0.35,
            "detail": f"gap=0 p50 {g0['p50_ms']:.2f} ms vs encode C=1 "
            f"{enc1['p50_ms']:.2f} ms ({rel * 100:.1f}%)"
            if rel is not None
            else "not measured",
        }
    )

    # The whole point of control 4 is that the gap matters. If it did not, the
    # section explaining the 6x disagreement would be resting on nothing.
    slow = max((g["p50_ms"] for g in gaps), default=0.0)
    fast = min((g["p50_ms"] for g in gaps), default=0.0)
    checks.append(
        {
            "check": "S8 idle gap actually moves encode cost (else 5b explains nothing)",
            "ok": fast > 0 and slow / fast >= 2.0,
            "detail": f"p50 {fast:.2f} -> {slow:.2f} ms ({slow / fast:.1f}x)"
            if fast
            else "not measured",
        }
    )

    # S4 skips `glue` because its residence is below the scheduler's own
    # granularity. That exemption is only safe if a mismeasured `glue` cannot
    # move anything downstream -- so measure that instead of asserting it. In
    # the harmonic law the arms enter as reciprocals, so an arm two orders of
    # magnitude faster than the others contributes ~0.1% of the prediction:
    # even a 2x error in it moves `predicted_qps` by less than the run-to-run
    # drift S3 tolerates. If `glue` ever grows into a real layer this check
    # fails and S4's exemption has to be revisited rather than inherited.
    share = 0.0
    for row in composition(raw):
        parts = [row.get(f"{a}_qps", 0.0) for a in COMPONENTS]
        if all(parts) and row.get("glue_qps"):
            total = sum(1.0 / v for v in parts)
            share = max(share, (1.0 / row["glue_qps"]) / total)
    checks.append(
        {
            "check": "S9 `glue` is negligible in the composition (S4's exemption is safe)",
            "ok": 0.0 < share <= 0.02,
            "detail": f"worst share of the harmonic sum {share * 100:.2f}% "
            f"(limit 2%)" if share else "not measured",
        }
    )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default=COMBO_ID)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render", action="store_true", help="re-render from the raw cache")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="wiring check only: 8 queries, C in {1,2,4}. Writes to its own "
        "*_smoke.* paths, because a smoke slice is not a small version of the "
        "answer -- its plateaus are meaningless and must not be readable as a "
        "verdict (cf. hybrid_weighted_fetch_depth.py, where a smoke slice "
        "reversed the headline's sign).",
    )
    args = ap.parse_args()

    global CONCURRENCY, RAW, REPORT
    if args.smoke:
        CONCURRENCY = [1, 2, 4]
        args.limit = args.limit or 8
        RAW = RAW.with_name("qdrant_concurrency_raw_smoke.json")
        REPORT = REPORT.with_name("qdrant_concurrency_smoke.md")

    if args.render:
        raw = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        raw = collect(args)
        print(f"raw -> {RAW}")

    REPORT.write_text(render(raw), encoding="utf-8")
    print(f"report -> {REPORT}")

    checks = self_checks(raw)
    print("\nself-checks")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']} -- {c['detail']}")
    primary, app, ratio, cstar = verdict(raw)
    print(f"\nverdict: {primary}; composition {app} (ratio {ratio:.3f} at C={cstar})")
    return 1 if any(not c["ok"] for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
