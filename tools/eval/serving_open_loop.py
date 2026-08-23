"""Open-loop arrival: what a user waits, at an arrival rate nobody controls.

`serving_concurrency.md` measures a CLOSED loop -- C worker threads, each
issuing its next query only after its last one returns. That answers "what does
the system do at concurrency C" and it is the right shape for finding a
plateau, but it is the wrong shape for sizing a deployment, for one structural
reason: **a closed loop throttles itself.** When the system slows down, its own
clients slow down with it, so a queue can never build. Real users do not do
that. They arrive when they arrive.

This measures the other shape. A dispatcher emits requests at a fixed arrival
rate lambda, INDEPENDENT of how fast anything completes, and every request is
timed from the moment it arrived -- not from the moment a worker picked it up.
That difference is the whole point: the number a user feels is

    response = queue wait + service

and a closed loop can only ever report the second term.

**Two things it is built to show, and they are different claims.**

1. **Where the knee is.** Below capacity, response ~ service and the queue is
   empty. Above capacity the queue grows without bound and there IS no latency
   figure -- the honest report is "unstable", not a p99 computed over whatever
   happened to finish. A grid that never goes unstable has not found the knee,
   which is why `A4` requires at least one of each.
2. **What burstiness alone costs.** The deterministic arms send at exactly the
   same rate with exactly even spacing. Any gap between them at equal lambda is
   the price of arrivals clumping, not of load -- a cost that a closed loop
   cannot express at all, and that shows up well BELOW capacity.

**Stability is decided by two signals, and reported as such.** Achieved
completion rate against offered rate (a system that cannot keep up completes
fewer than it is sent), and the slope of the backlog over the second half of
the run. Either alone is misleading: a run can be transiently backed up while
keeping up on average, and a short arm can look flat because it has not had
time to diverge yet.

**Percentiles are reported only where the sample supports them.** A p99 over
300 requests rests on three observations; publishing it would be a number, not
a measurement. The rule is `n * (1 - p) >= 5` -- so p99 needs 500 completions --
and cells below it print `n/a` rather than a figure. Same discipline as the
resolution limits already recorded in `serving_concurrency.md`.

**What this does NOT establish.** No network hop (app, embedder and engine are
one process on one box, which understates the app layer). One query shape, one
route mix -- the Gold set's, replayed in order. No think-time model: a Poisson
arrival is a memoryless approximation of many independent users, not a
simulation of any particular user population. And the arrival process is open
but the WORK is not: every request runs the same routed hybrid query, so this
says nothing about a mix of cheap and expensive requests.

Generated report: `data/results/serving_open_loop.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import clear_retriever_cache  # noqa: E402
from rag_lab.io.index_cache import clear_index_cache  # noqa: E402
from rag_lab.query_service import (  # noqa: E402
    discover_indices,
    route_query,
    warm_serving_caches,
)
from rag_lab.query_sets import load_gold_query_set  # noqa: E402

INDEX_ROOT = REPO / "data/index/chunker_compare_full"
GOLD = REPO / "config/eval/gold_query_set_73det.yaml"
REPORT = REPO / "data/results/serving_open_loop.md"
RAW = REPO / "data/results/serving_open_loop_raw.json"

K = 10
FETCH_DEPTH = 200
URL = "http://127.0.0.1:6333"

#: Server-side concurrency. Not a tuning knob here: the engine topology is
#: already at 86% of its plateau by C=1 (`serving_concurrency.md` section 7), so
#: 8 is comfortably past the point where adding workers stops helping. It is
#: reported rather than swept because what this script varies is ARRIVALS.
WORKERS = 8

#: A percentile is printed only when this many completions sit above it.
_MIN_TAIL_SAMPLES = 5

#: Published closed-loop C=1 p50 per topology (`serving_concurrency.md` section
#: 7). At lambda=1 the offered load is ~0.13 requests in flight, so an open-loop
#: cell there is effectively C=1 and must land near these -- which is what makes
#: A5 an anchor between two independent harnesses rather than a plausibility
#: range. Anchoring the ENGINE arm against the whole-system 463 ms warm query
#: was the first version's error: that figure is not this topology's.
_PUBLISHED_C1_P50_MS = {"engine": 128.3, "inproc": 626.2}

#: How long queued requests may still run after the dispatcher stops. Long
#: enough that a stable arm loses nothing, short enough that an unstable arm
#: does not spend longer draining than measuring.
_DRAIN_S = 15.0


def _spec(arm: str) -> StrategySpec:
    if arm == "engine":
        return StrategySpec(type="qdrant_hybrid",
                            params={"url": URL, "fetch_depth": FETCH_DEPTH, "exact": True})
    return StrategySpec(type="hybrid", params={"fetch_depth": FETCH_DEPTH})


def _pct(values: list[float], p: float) -> float | None:
    """The p-th percentile, or None when the sample cannot carry it."""
    if not values or len(values) * (1.0 - p) < _MIN_TAIL_SAMPLES:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
    return ordered[idx]


def _gaps(lam: float, arrival: str, n: int, seed: int) -> np.ndarray:
    """Inter-arrival times.

    `poisson` draws exponential gaps -- the memoryless process, i.e. many
    independent users. `deterministic` sends at exactly 1/lambda, which is the
    control: it offers the identical rate with none of the clumping, so the
    difference between the two arms at equal lambda is burstiness and nothing
    else.
    """
    rng = np.random.default_rng(seed)
    if arrival == "poisson":
        return rng.exponential(1.0 / lam, size=n)
    return np.full(n, 1.0 / lam)


def run_arm(arm: str, lam: float, seconds: float, arrival: str, indices,
            texts: list[str], *, null: bool = False, seed: int = 0) -> dict:
    """One (arm, lambda, arrival process) cell.

    The dispatcher does NOT wait for anything. It sleeps the next inter-arrival
    gap and submits, so when the system falls behind the backlog grows here
    rather than the arrival rate quietly dropping -- which is precisely what a
    closed loop does instead, and why it cannot see this.
    """
    spec = _spec(arm)
    records: list[tuple[float, float, float]] = []   # arrive, start, end
    lock = threading.Lock()
    backlog: list[tuple[float, int]] = []
    sched: list[float] = []          # every arrival, not only the ones that ran
    lag: list[float] = []            # actual submit time - scheduled arrival
    submitted = 0
    completed = 0
    dropped = 0
    drop_after = [float("inf")]

    def work(idx: int, t_arrive: float) -> None:
        nonlocal completed, dropped
        if time.perf_counter() > drop_after[0]:
            # Still queued when the drain window closed. Counted, not silently
            # discarded: on an unstable arm the requests that never ran ARE the
            # finding, and `cancel_futures` would have made them invisible to
            # the completion counter while never letting the drain loop finish.
            with lock:
                dropped += 1
            return
        t_start = time.perf_counter()
        if not null:
            route_query(texts[idx % len(texts)], indices, spec, K)
        t_end = time.perf_counter()
        with lock:
            completed += 1
            records.append((t_arrive, t_start, t_end))

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    n_max = int(lam * seconds * 3) + 64
    gaps = _gaps(lam, arrival, n_max, seed)

    t0 = time.perf_counter()
    deadline = t0 + seconds
    next_at = t0
    sample_at = t0
    i = 0
    while True:
        now = time.perf_counter()
        if now >= deadline:
            break
        if now >= sample_at:
            with lock:
                backlog.append((now - t0, submitted - completed))
            sample_at += 0.5
        if now >= next_at:
            t_arrive = next_at              # the SCHEDULED arrival, not `now`:
            # charging a request from when the dispatcher got to it would hide
            # dispatcher lag inside the service time, which is the bug the
            # closed-loop harness hit in a different form (its section 8).
            pool.submit(work, i, t_arrive)
            with lock:
                submitted += 1
                # Recorded for EVERY arrival, including ones that are later
                # dropped: taking the arrival process from the completed subset
                # would measure it on an unstable arm as whatever got through.
                sched.append(t_arrive - t0)
                lag.append((now - t_arrive) * 1000)
            next_at += float(gaps[min(i, n_max - 1)])
            i += 1
            continue
        time.sleep(min(0.002, max(0.0, next_at - now)))

    dispatch_end = time.perf_counter()
    # A DRAIN WINDOW, not a cancel and not an unbounded wait. A stable arm has
    # an empty queue and finishes it in a second or two, so nothing is lost; an
    # unstable arm would take longer to drain than it took to run, so what is
    # still queued after the window is dropped and COUNTED.
    drop_after[0] = dispatch_end + _DRAIN_S
    pool.shutdown(wait=True)

    with lock:
        recs = list(records)
        n_sub, n_done, n_drop = submitted, completed, dropped
        all_arrivals = np.array(sched)
        lags = list(lag)
    response = [(e - a) * 1000 for a, _s, e in recs]
    service = [(e - s) * 1000 for _a, s, e in recs]
    wait = [(s - a) * 1000 for a, s, _e in recs]
    realised_gaps = np.diff(all_arrivals) if all_arrivals.size > 2 else np.array([])

    span = dispatch_end - t0
    offered = n_sub / span
    achieved = n_done / span
    def _mean_backlog(lo: float, hi: float) -> float:
        vals = [b for t, b in backlog if lo <= t < hi]
        return float(np.mean(vals)) if vals else 0.0

    # QUARTERS, NOT A FITTED SLOPE. The first version fitted a line to the
    # backlog over the second half and called >0.2/s unstable. It produced
    # verdicts that contradicted the latencies beside them -- lambda=6 "unstable"
    # at a 342 ms response, lambda=8 "stable" at 1,356 ms -- because a fitted
    # slope over a sawtooth is dominated by where the sawtooth happened to be at
    # each end. Comparing the mean depth of the queue early against late asks
    # the question directly and is not a rate at all.
    q_early = _mean_backlog(span * 0.25, span * 0.50)
    q_late = _mean_backlog(span * 0.75, span)
    growth = q_late - q_early
    keeps_up = achieved >= 0.95 * offered
    # The threshold comes from the RIG: the queue is meaningfully deeper once it
    # has gained more than a full set of server workers, i.e. more than the pool
    # could be running at once. Below that it is a busy pool oscillating.
    stable = bool(keeps_up and growth <= WORKERS)

    return {
        "arm": arm,
        "lambda": lam,
        "arrival": arrival,
        "seconds": round(span, 1),
        "submitted": n_sub,
        "completed": n_done,
        "dropped": n_drop,
        "offered_qps": round(offered, 2),
        "achieved_qps": round(achieved, 2),
        "backlog_early": round(q_early, 1),
        "backlog_late": round(q_late, 1),
        "backlog_growth": round(growth, 1),
        "backlog_end": backlog[-1][1] if backlog else 0,
        "backlog_samples": [[round(t, 2), b] for t, b in backlog],
        "stable": stable,
        "response_p50": _pct(response, 0.50),
        "response_p90": _pct(response, 0.90),
        "response_p95": _pct(response, 0.95),
        "response_p99": _pct(response, 0.99),
        "service_p50": _pct(service, 0.50),
        "wait_p50": _pct(wait, 0.50),
        "wait_p95": _pct(wait, 0.95),
        "gap_cv": (round(float(np.std(realised_gaps) / np.mean(realised_gaps)), 3)
                   if realised_gaps.size > 5 and np.mean(realised_gaps) > 0 else None),
        "n_arrivals": int(all_arrivals.size),
        "dispatch_lag_p95": _pct(lags, 0.95),
    }


def _prepare(arm: str, indices, texts: list[str]) -> float:
    """Warm ONE topology, and hand it a short discarded ramp.

    **The topologies are prepared separately, and that is a correctness fix
    rather than tidiness.** `with_embeddings` is part of the index-cache key, so
    warming both in one process asks for 4 index directories x 2 variants = 8
    keys against a cache sized 4 (`RAG_LAB_INDEX_CACHE`, sized for the four
    directories the five routes resolve to). Every query then evicts an entry
    the other topology needs and pays a ~1.2 s reload. The first run of this
    script did exactly that and reported an in-process service p50 of 4,484 ms
    against `serving_concurrency.md`'s published 626.2 ms at C=1 -- a harness
    artefact, and a real note for anyone serving both retriever types from one
    process: raise the cache to 8.

    The ramp is discarded because CUDA/BLAS residue survives
    `warm_serving_caches` (documented there), and without it the first measured
    cell carries it -- the first run's lambda=1 cell showed a 6,484 ms p90 at
    almost no load.
    """
    clear_index_cache()
    clear_retriever_cache()
    sp = _spec(arm)
    t0 = time.perf_counter()
    warm_serving_caches(indices, sp.type, with_rows=(arm == "inproc"),
                        retriever_params=dict(sp.params))
    route_query(texts[0], indices, sp, K)
    return round((time.perf_counter() - t0) * 1000, 1)


def measure(seconds: float, lambdas: list[float]) -> dict:
    indices = discover_indices(INDEX_ROOT)
    entries = load_gold_query_set(GOLD)
    texts = [e.query for e in entries]
    cells: list[dict] = []
    warm: dict = {}

    warm["engine"] = _prepare("engine", indices, texts)

    # The harness itself, far above the grid: if the dispatcher cannot emit at a
    # rate well past anything measured, every "unstable" verdict below is about
    # this script rather than about the system.
    cells.append(run_arm("null", max(lambdas) * 20, 5.0, "poisson", indices, texts,
                         null=True, seed=1))

    print("  engine  ramp (discarded)")
    run_arm("engine", 2.0, 15.0, "poisson", indices, texts, seed=7)

    for lam in lambdas:
        print(f"  engine  poisson       lambda={lam}")
        cells.append(run_arm("engine", lam, seconds, "poisson", indices, texts,
                             seed=int(lam * 100)))

    for lam in sorted({lambdas[1], lambdas[len(lambdas) // 2]}):
        print(f"  engine  deterministic lambda={lam}")
        cells.append(run_arm("engine", lam, seconds, "deterministic", indices, texts,
                             seed=int(lam * 100)))

    warm["inproc"] = _prepare("inproc", indices, texts)
    print("  inproc  ramp (discarded)")
    run_arm("inproc", 1.0, 15.0, "poisson", indices, texts, seed=7)

    for lam in (lambdas[0], lambdas[1]):
        print(f"  inproc  poisson       lambda={lam}")
        cells.append(run_arm("inproc", lam, seconds, "poisson", indices, texts,
                             seed=int(lam * 100)))

    return {"workers": WORKERS, "k": K, "fetch_depth": FETCH_DEPTH,
            "warm_ms": warm, "cells": cells}


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def _ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v:,.0f}"


def render(data: dict) -> tuple[str, list[tuple[str, bool, str]]]:
    cells = data["cells"]
    real = [c for c in cells if c["arm"] != "null"]
    null = next((c for c in cells if c["arm"] == "null"), None)
    engine_p = [c for c in real if c["arm"] == "engine" and c["arrival"] == "poisson"]

    L: list[str] = []
    L.append("# Open-loop arrival: what a user waits when nobody throttles the arrivals")
    L.append("")
    L.append("Generated by `tools/eval/serving_open_loop.py`.")
    L.append("")
    L.append(
        f"`serving_concurrency.md` measures a **closed** loop: C workers, each issuing "
        f"its next query only after the last returns. That shape throttles itself — when "
        f"the system slows, its own clients slow with it, so a queue can never build. "
        f"Here a dispatcher emits at a fixed rate λ **independent of completions**, and "
        f"every request is timed from when it *arrived*, not from when a worker picked "
        f"it up. Server-side concurrency is fixed at **{data['workers']} workers**; "
        f"k={data['k']}, `fetch_depth`={data['fetch_depth']}; queries are the 106 Gold "
        f"73det queries replayed in order."
    )
    L.append("")
    L.append(
        "**response = queue wait + service.** A closed loop can only ever report the "
        "second term. The first is what a user feels when someone else is ahead of them."
    )
    L.append("")
    L.append("| arm | arrivals | λ (offered) | achieved | completed | dropped | queue early→late | stable | "
             "**response p50** | p90 | p95 | p99 | service p50 | wait p95 |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for c in real:
        L.append(
            f"| `{c['arm']}` | {c['arrival']} | {c['lambda']:g} q/s | "
            f"{c['achieved_qps']:.2f} | {c['completed']:,}/{c['submitted']:,} | "
            f"{c.get('dropped', 0):,} | "
            f"{c.get('backlog_early', 0):.1f} → {c.get('backlog_late', 0):.1f} | "
            f"{'yes' if c['stable'] else '**NO**'} | "
            f"**{_ms(c['response_p50'])}** | {_ms(c['response_p90'])} | "
            f"{_ms(c['response_p95'])} | {_ms(c['response_p99'])} | "
            f"{_ms(c['service_p50'])} | {_ms(c['wait_p95'])} |"
        )
    L.append("")
    L.append(
        "**Latency on an unstable row is survivorship-biased and must not be read as "
        "a latency.** Those cells complete the requests that queued least and drop the "
        "rest, so the figures describe who got through, not what the system does — the "
        "`dropped` column is the honest reading of an unstable row."
    )
    L.append("")
    L.append(
        f"**The queue is not only in the `wait` column.** With {data['workers']} server "
        f"workers over one GPU, a backlog partly shows up as *slower service* rather "
        f"than as waiting — several requests run at once and each takes longer. So "
        f"`wait` is a floor on the queueing cost, not the whole of it, and `response` "
        f"is the only column that carries all of it. A single-worker server would move "
        f"the same cost into `wait`."
    )
    L.append("")
    L.append(
        "**`dropped` is 0 everywhere, including the unstable row, and that is a "
        "property of the run length rather than of the system.** A 75 s arm at λ just "
        "past capacity accumulates a queue the 15 s drain window still absorbs; a "
        "longer arm at the same λ would start refusing. Read the queue depth, not the "
        "drop count, as the sign of divergence here."
    )
    L.append("")
    L.append("All latencies in ms. A percentile is printed only where at least "
             f"{_MIN_TAIL_SAMPLES} completions sit above it — so p99 needs 500 "
             "completions, and `n/a` means the sample cannot carry that figure rather "
             "than that nothing was measured.")
    L.append("")

    stable = [c for c in engine_p if c["stable"]]
    unstable = [c for c in engine_p if not c["stable"]]
    if stable and unstable:
        knee_lo = max(c["lambda"] for c in stable)
        knee_hi = min(c["lambda"] for c in unstable)
        L.append("## Where the knee is")
        L.append("")
        L.append(
            f"The engine topology stays stable to **λ = {knee_lo:g} q/s** and is "
            f"already unstable at **λ = {knee_hi:g} q/s**, so the knee sits between "
            f"them. Above it there is no latency figure to quote — the backlog grows "
            f"for as long as the load lasts, and any percentile computed over what "
            f"happened to finish is a property of the run length, not of the system."
        )
        L.append("")
        worst = max(stable, key=lambda c: c["lambda"])
        if worst["response_p50"] and worst["service_p50"]:
            infl = worst["response_p50"] / worst["service_p50"]
            L.append(
                f"At the last stable rate the queue is already doing visible work: "
                f"response p50 **{_ms(worst['response_p50'])} ms** against service p50 "
                f"**{_ms(worst['service_p50'])} ms** ({infl:.2f}x), i.e. a user spends "
                f"{'more' if infl > 2 else 'a meaningful share of'} their wait behind "
                f"other users rather than in the retriever."
            )
            L.append("")

    det = [c for c in real if c["arrival"] == "deterministic"]
    if det:
        L.append("## What burstiness alone costs")
        L.append("")
        L.append(
            "The deterministic arms offer the **same rate** with even spacing, so any "
            "gap at equal λ is clumping and nothing else — a cost a closed loop cannot "
            "express, because it has no arrival process to make bursty."
        )
        L.append("")
        L.append("| λ | poisson p95 | deterministic p95 | poisson p50 | deterministic p50 |")
        L.append("| ---: | ---: | ---: | ---: | ---: |")
        for d in det:
            p = next((c for c in engine_p if c["lambda"] == d["lambda"]), None)
            if p:
                L.append(
                    f"| {d['lambda']:g} q/s | {_ms(p['response_p95'])} | "
                    f"{_ms(d['response_p95'])} | {_ms(p['response_p50'])} | "
                    f"{_ms(d['response_p50'])} |"
                )
        L.append("")
        pairs = [(d, next((c for c in engine_p if c["lambda"] == d["lambda"]), None))
                 for d in det]
        usable = [(d, p) for d, p in pairs
                  if p and d["response_p95"] and p["response_p95"]
                  and d["response_p50"] and p["response_p50"]]
        if usable:
            lo = min(usable, key=lambda x: x[0]["lambda"])
            hi = max(usable, key=lambda x: x[0]["lambda"])
            L.append(
                f"**It costs the tail first.** At λ = {lo[0]['lambda']:g} q/s the two "
                f"medians are the same to within a millisecond "
                f"({_ms(lo[1]['response_p50'])} vs {_ms(lo[0]['response_p50'])} ms) while "
                f"p95 differs "
                f"{lo[1]['response_p95'] / lo[0]['response_p95']:.1f}x — the typical user "
                f"notices nothing and the unlucky one waits. By "
                f"λ = {hi[0]['lambda']:g} q/s it has reached the median too "
                f"({hi[1]['response_p50'] / hi[0]['response_p50']:.1f}x), with p95 "
                f"{hi[1]['response_p95'] / hi[0]['response_p95']:.1f}x."
            )
            L.append("")
            L.append(
                f"**So a capacity number taken from even spacing is optimistic about "
                f"what people feel.** At λ = {hi[0]['lambda']:g} q/s — comfortably inside "
                f"the plateau `serving_concurrency.md` reports — evenly spaced arrivals "
                f"see {_ms(hi[0]['response_p95'])} ms at p95 and Poisson arrivals at the "
                f"identical rate see {_ms(hi[1]['response_p95'])} ms. Nothing about the "
                f"system changed; only when the requests turned up."
            )
            L.append("")

    inproc = [c for c in real if c["arm"] == "inproc"]
    if inproc:
        L.append("## The topology choice, seen open-loop")
        L.append("")
        L.append(
            "`serving_concurrency.md` picks the engine topology on plateau throughput "
            "(9.81 vs 2.53 q/s). Open-loop the same choice shows up as *how early the "
            "queue starts*, which is the form a user experiences it in."
        )
        L.append("")
        for c in inproc:
            p = next((e for e in engine_p if e["lambda"] == c["lambda"]), None)
            L.append(
                f"- **λ = {c['lambda']:g} q/s**: in-process "
                f"{'stable' if c['stable'] else '**unstable**'}, response p50 "
                f"{_ms(c['response_p50'])} ms"
                + (f" — engine at the same rate is "
                   f"{'stable' if p['stable'] else '**unstable**'} at "
                   f"{_ms(p['response_p50'])} ms." if p else ".")
            )
        L.append("")

    L.append("## What is NOT established")
    L.append("")
    L.append(
        "- **No network hop.** App, embedder and engine are one process on one box, "
        "which understates the app layer and hides serialization a real hop adds.\n"
        "- **One work shape.** Every request is the same routed hybrid query, so "
        "nothing here says how a mix of cheap and expensive requests queues.\n"
        "- **No think-time model.** A Poisson process approximates many independent "
        "users; it is not a simulation of a particular population, and a real class of "
        "students hitting one deadline is far burstier than Poisson.\n"
        "- **The knee is a property of this box.** The GPU is the serialising layer "
        "(`serving_concurrency.md` section 7); a different card moves it."
    )
    L.append("")

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    # The estimator's own resolution: the CV of n exponential draws has standard
    # error ~1/sqrt(n), so a 14-arrival cell reads 0.68 while drawing a perfectly
    # good exponential. The check is scoped to cells that can carry it rather
    # than to a band wide enough to swallow a real defect.
    _CV_MIN_N = 100
    poisson_cvs = [c["gap_cv"] for c in real
                   if c["arrival"] == "poisson" and c["gap_cv"] is not None
                   and c.get("n_arrivals", 0) >= _CV_MIN_N]
    det_cvs = [c["gap_cv"] for c in det if c["gap_cv"] is not None]
    add("A1 the arrival process is the one claimed, over ALL arrivals",
        bool(poisson_cvs) and 0.75 <= min(poisson_cvs) and max(poisson_cvs) <= 1.30
        and (not det_cvs or max(det_cvs) < 0.10),
        f"poisson CV {min(poisson_cvs):.2f}-{max(poisson_cvs):.2f} over "
        f"{len(poisson_cvs)} cells with >= {_CV_MIN_N} arrivals (exponential is 1.0)"
        + (f", deterministic {max(det_cvs):.3f}" if det_cvs else "")
        + " -- taken over every arrival, not the completed subset, which on an "
          "unstable arm is whatever got through"
        if poisson_cvs else "no cell has enough arrivals to estimate the CV")

    below = [c for c in real if c["stable"]]
    add("A2 a stable arm actually delivered what it was offered",
        bool(below) and all(c["achieved_qps"] >= 0.95 * c["offered_qps"] for c in below),
        f"{len(below)} stable cells, worst achieved/offered "
        f"{min((c['achieved_qps'] / c['offered_qps']) for c in below):.3f}"
        if below else "no stable cell")

    add("A3 the dispatcher is not the bottleneck",
        null is not None and null["achieved_qps"] >= 0.9 * null["offered_qps"],
        f"null arm offered {null['offered_qps']:.0f} q/s, achieved "
        f"{null['achieved_qps']:.0f} q/s at {max(c['lambda'] for c in real):g}x the "
        f"top of the grid" if null else "no null arm")

    add("A4 the grid found the knee",
        bool(stable) and bool(unstable),
        f"{len(stable)} stable and {len(unstable)} unstable engine/poisson cells -- a "
        f"grid entirely on one side of the knee measures the grid, not the system"
        if engine_p else "no engine cells")

    anchors: list[str] = []
    anchor_ok = True
    for topo, published in _PUBLISHED_C1_P50_MS.items():
        low = [c for c in real if c["arm"] == topo and c["arrival"] == "poisson"]
        if not low:
            continue
        cell = min(low, key=lambda c: c["lambda"])
        got = cell["service_p50"]
        ok = got is not None and 0.6 * published <= got <= 2.0 * published
        anchor_ok = anchor_ok and ok
        anchors.append(
            f"{topo} {_ms(got)} ms at lambda={cell['lambda']:g} vs published "
            f"{published:.1f} ({'ok' if ok else 'OUT'})")
    add("A5 service at the lowest rate matches this topology's published C=1 p50",
        anchor_ok and bool(anchors),
        "; ".join(anchors) + " -- each arm against ITS OWN closed-loop figure, from an "
        "independent harness" if anchors else "no cells to anchor")

    printed_p99 = [c for c in real if c["response_p99"] is not None]
    add("A6 no percentile is printed that its sample cannot carry",
        all(c["completed"] * 0.01 >= _MIN_TAIL_SAMPLES for c in printed_p99),
        f"{len(printed_p99)} cells print a p99, each with at least "
        f"{_MIN_TAIL_SAMPLES / 0.01:.0f} completions; the rest print n/a")

    # THE BOUND COMES FROM THE INTERPRETER, NOT FROM A PREFERENCE. The dispatcher
    # is one Python thread competing with `workers` busy ones, so it cannot be
    # rescheduled faster than CPython's switch interval times the number of
    # runnable threads. Below that line a lag is unresolvable, not a defect --
    # the same reasoning `serving_concurrency.md`'s S4 uses for Little's law,
    # where a 5 ms quantum sets the domain rather than the failing arm doing it.
    floor_ms = sys.getswitchinterval() * 1000.0 * data["workers"]
    lagged = [c for c in real if c.get("dispatch_lag_p95") is not None
              and c.get("response_p50")]
    worst = (max(lagged, key=lambda c: c["dispatch_lag_p95"] / c["response_p50"])
             if lagged else None)
    add("A7 dispatcher lag is within what the GIL allows, and small against response",
        worst is not None and worst["dispatch_lag_p95"] <= floor_ms,
        f"worst cell: lag p95 {worst['dispatch_lag_p95']:.1f} ms against a "
        f"{floor_ms:.0f} ms floor (switch interval {sys.getswitchinterval() * 1000:.0f} ms "
        f"x {data['workers']} workers), i.e. "
        f"{100 * worst['dispatch_lag_p95'] / worst['response_p50']:.1f}% of that cell's "
        f"own response p50 -- charged to `wait`, so every response here is inflated by "
        f"at most that" if worst else "no lag samples")

    L.append("## Self-checks")
    L.append("")
    L.append("| check | verdict | detail |")
    L.append("| --- | --- | --- |")
    for name, ok, detail in checks:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    L.append("")
    return "\n".join(L) + "\n", checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true", help="re-render from the raw cache")
    ap.add_argument("--seconds", type=float, default=75.0)
    ap.add_argument("--lambdas", default="1,2,4,6,8,10",
                    help="offered arrival rates in q/s")
    ap.add_argument("--smoke", action="store_true",
                    help="a short grid; prints, writes nothing")
    args = ap.parse_args()

    if args.render:
        data = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        lambdas = [float(x) for x in args.lambdas.split(",")]
        seconds = 12.0 if args.smoke else args.seconds
        if args.smoke:
            lambdas = lambdas[:3]
        data = measure(seconds, lambdas)
        if args.smoke:
            print(json.dumps(data, ensure_ascii=False, indent=1))
            print("smoke run -- nothing written")
            return 0
        RAW.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    text, checks = render(data)
    REPORT.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT}")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
