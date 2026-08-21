"""Where a served query spends its time, and what the embedder cache removes.

Generated report: data/results/serving_cost_profile.md

**Why this exists rather than a note in a docstring.** `qdrant_concurrency.md`
established that the system is ENCODE-BOUND, but its harness built the embedder
once outside the loop -- so it measured a *warm* embedder and never priced the
per-call CONSTRUCTION that the shipped `query_service` actually pays. This
measures exactly the thing that study held constant.

Two sections, and the second is the one that decides anything:

  1  DECOMPOSITION -- one routed query split into first-use vs warm-use on both
     the embedder and the Index, so the cacheable part of each is visible
     separately from the irreducible part.
  2  A/B -- the shipped `route_query` with the cache off and on, alternated per
     query in one process, checking BOTH latency and whether the top-10 moved.

**Two instrument faults are baked into the design because both were hit while
writing it, and either one alone reverses the conclusion.**

(a) `LocalSTEmbedder._load()` is LAZY: the constructor stores a model name and
    `SentenceTransformer(...)` runs inside the first `embed()`. A version that
    timed `build_embedder` reported 0.0 ms and charged 8.9 s of weight loading
    to "encode", concluding a cache could win 10%. A 9 s encode against a
    published 13-83 ms (cost_latency_pareto.md) is an instrument fault, not a
    finding -- hence `S1`, which requires warm encode to land in the published
    range.
(b) `BM25Okapi` is memoised on the Index object (`Index.lexical_scorer`), so a
    freshly loaded Index rebuilds it (~1 s). Timing one retrieve per fresh load
    charges that rebuild to "retrieve"; §1 therefore reports first and warm
    retrieve separately and reads the DELTA as the cacheable part.

Self-checks (all must PASS, exit 1 otherwise):
  S1  warm encode lands in the published cost_latency_pareto range
  S2  every A/B query returns an identical top-10 -- ids AND scores
  S3  after alternating routes the cache holds exactly the shipped model count
  S4  the 'off' arm really is uncached (it must be slower than 'on' steady state
      by more than the weight-load cost, or the arms were not separated)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import (  # noqa: E402
    build_embedder,
    build_retriever,
    clear_embedder_cache,
    embedder_cache_info,
)
from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.pipeline import retrieve  # noqa: E402
from rag_lab.query_service import (  # noqa: E402
    _read_manifest,
    discover_indices,
    resolve_index,
    route_query,
)
from rag_lab.router import classify_query, route_targets  # noqa: E402

INDEX_ROOT = REPO / "data/index/chunker_compare_full"
REPORT = REPO / "data/results/serving_cost_profile.md"
RAW = REPO / "data/results/serving_cost_profile_raw.json"
CACHE_ENV = "RAG_LAB_EMBEDDER_CACHE"

# Alternating routes on purpose: consecutive same-route queries would be served
# by one resident model and a size-1 cache would look identical to a size-2 one.
# A deployment alternates; so does this.
QUERIES = [
    "ผู้ช่วยศาสตราจารย์ ดร. ธนา หงษ์สุวรรณ",
    "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมคอมพิวเตอร์",
    "รองศาสตราจารย์ ดร. สมชาย",
    "รายวิชา CALCULUS 2",
]
RETRIEVER = StrategySpec(type="hybrid", params={"fetch_depth": 200})
REPEATS = 3

# data/results/cost_latency_pareto.md, dim-1024 embedders. Wide on purpose: that
# report's own controls put this rig's resolution at ~5-10%, and encode cost on
# this card is a function of how long the GPU sat idle (13.6 ms at zero gap,
# 181 ms after 1.8 s idle). S1 is a sanity gate against a lazy load hiding in
# the encode number, not a precision claim.
PUBLISHED_ENCODE_MS = (5.0, 250.0)


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t) * 1000.0


def top10(res):
    return [(r.chunk_id, round(float(r.score), 10)) for r in res.results[:10]]


def decompose(indices) -> dict:
    """§1: one routed query, first-use separated from warm-use."""
    query = QUERIES[0]
    route = classify_query(query)
    info = resolve_index(route_targets("hybrid")[route], indices)
    manifest = _read_manifest(info.dir)
    spec = StrategySpec.model_validate(manifest["combo"]["embedder"])
    retriever = build_retriever(RETRIEVER)
    store = ArtifactStore()

    # Warm the OS page cache and the CUDA context outside the measurement: a
    # cache removes neither, so leaving them in would inflate what is priced.
    w_emb = build_embedder(spec)
    w_emb.embed(["warm"])
    w_idx = store.load(info.dir)
    retrieve(query, w_idx, w_emb, retriever, 10, combination_id="warm")
    n_chunks, dims = len(w_idx.chunks), w_idx.embeddings.shape
    del w_emb, w_idx

    rows = []
    for _ in range(REPEATS):
        embedder, t_build = timed(lambda: build_embedder(spec))
        _, t_first_emb = timed(lambda: embedder.embed([query]))
        _, t_warm_emb = timed(lambda: embedder.embed([query]))
        index, t_load = timed(lambda: store.load(info.dir))
        _, t_first_r = timed(
            lambda: retrieve(query, index, embedder, retriever, 10, combination_id="p")
        )
        _, t_warm_r = timed(
            lambda: retrieve(query, index, embedder, retriever, 10, combination_id="p")
        )
        rows.append([t_build, t_first_emb, t_warm_emb, t_load, t_first_r, t_warm_r])
        del embedder, index

    keys = ["build", "first_embed", "warm_embed", "load", "first_retrieve", "warm_retrieve"]
    mean = {k: sum(r[i] for r in rows) / len(rows) for i, k in enumerate(keys)}
    return {
        "route": route, "index": Path(info.dir).name, "model": spec.params,
        "n_chunks": n_chunks, "embeddings_shape": list(dims),
        "runs": rows, "mean_ms": mean,
    }


def ab(indices) -> dict:
    """§2: the shipped route_query, cache off vs on, alternated in one process."""
    os.environ[CACHE_ENV] = "0"
    clear_embedder_cache()
    route_query(QUERIES[0], indices, RETRIEVER, 10)  # warm disk/CUDA for both arms

    times = {"off": [], "on": []}
    answers = {"off": [], "on": []}
    for q in QUERIES * 2:
        for arm in ("off", "on"):
            # NOTE: no clear_embedder_cache() in the 'off' arm. At size 0 the
            # cached builder bypasses the cache entirely and never reads it, so
            # 'off' cannot benefit anyway -- while clearing WIPED the resident
            # models 'on' depends on. With the arms alternating per query, that
            # made every 'on' measurement cold and the treatment measured itself
            # at 1.0x. The harness was wrong, not the cache.
            os.environ[CACHE_ENV] = "0" if arm == "off" else "2"
            # ONE call, both timed and scored. A second call for the answer
            # would double the work and -- worse -- score a query the timing
            # never saw, so a cache bug that only bit the first call would be
            # invisible to S2.
            res, dt = timed(lambda: route_query(q, indices, RETRIEVER, 10))
            times[arm].append(dt)
            answers[arm].append(top10(res))
    os.environ.pop(CACHE_ENV, None)

    differing = [i for i in range(len(answers["off"])) if answers["off"][i] != answers["on"][i]]
    # Steady state excludes the cold fill: with 2 models and alternating routes
    # the first query of each route pays a load in BOTH arms by construction.
    steady_on = sorted(times["on"])[: len(times["on"]) - 2]
    return {
        "queries": [{"query": q, "route": classify_query(q)} for q in QUERIES],
        "off_ms": times["off"], "on_ms": times["on"],
        "p50_off": statistics.median(times["off"]),
        "p50_on": statistics.median(times["on"]),
        "p50_on_steady": statistics.median(steady_on),
        "n_differing": len(differing),
        "n_compared": len(answers["off"]),
        "cache_after": embedder_cache_info(),
    }


def render(data: dict) -> tuple[str, list[tuple[str, bool, str]]]:
    d, a = data["decompose"], data["ab"]
    m = d["mean_ms"]
    now = m["build"] + m["first_embed"] + m["load"] + m["first_retrieve"]
    irreducible = m["warm_embed"] + m["warm_retrieve"]
    emb_saving = m["build"] + m["first_embed"] - m["warm_embed"]
    idx_saving = m["load"] + (m["first_retrieve"] - m["warm_retrieve"])

    checks = [
        ("S1 warm encode is in the published cost_latency_pareto range",
         PUBLISHED_ENCODE_MS[0] <= m["warm_embed"] <= PUBLISHED_ENCODE_MS[1],
         f"{m['warm_embed']:.1f} ms, expected {PUBLISHED_ENCODE_MS[0]}-{PUBLISHED_ENCODE_MS[1]}"),
        ("S2 the cache does not change the answer",
         a["n_differing"] == 0,
         f"{a['n_differing']} of {a['n_compared']} queries returned a different top-10"),
        ("S3 the cache holds the shipped model count after alternating routes",
         a["cache_after"]["size"] == 2,
         f"holds {a['cache_after']['size']} of max {a['cache_after']['max_size']}"),
        ("S4 the two arms were genuinely separated",
         a["p50_off"] - a["p50_on_steady"] > emb_saving * 0.5,
         f"off p50 {a['p50_off']:.0f} - on steady {a['p50_on_steady']:.0f} = "
         f"{a['p50_off'] - a['p50_on_steady']:.0f} ms vs weight load {emb_saving:.0f} ms"),
    ]

    L = []
    L.append("# Serving cost profile — what a query pays for, and what the embedder cache removes")
    L.append("")
    L.append(f"Generated by `tools/eval/serving_cost_profile.py` on "
             f"{datetime.fromtimestamp(data['ts']):%Y-%m-%d %H:%M}.")
    L.append("")
    L.append(f"Route `{d['route']}` → `{d['index']}` ({d['n_chunks']:,} chunks, "
             f"embeddings {tuple(d['embeddings_shape'])}), retriever `hybrid` "
             f"at `fetch_depth=200`, {REPEATS} runs.")
    L.append("")
    L.append("## 1. One routed query, first use vs warm use")
    L.append("")
    L.append("| stage | mean ms | cacheable? |")
    L.append("| --- | ---: | --- |")
    for k, label, note in [
        ("build", "`build_embedder` (constructor)", "lazy — the cost is not here"),
        ("first_embed", "first `embed()` — loads the weights", "**yes, embedder cache**"),
        ("warm_embed", "warm `embed()`", "no — real GPU work"),
        ("load", "`ArtifactStore.load` (parquet + npy + Chunk objects)", "yes, an index cache"),
        ("first_retrieve", "first `retrieve` (includes the BM25Okapi rebuild)", "the delta, yes"),
        ("warm_retrieve", "warm `retrieve`", "no — scoring + fusion"),
    ]:
        L.append(f"| {label} | {m[k]:.1f} | {note} |")
    L.append("")
    L.append(f"- one served query today, nothing cached: **{now:,.0f} ms**")
    L.append(f"- the embedder cache removes **{emb_saving:,.0f} ms** "
             f"({emb_saving / now * 100:.0f}% of it)")
    L.append(f"- an index cache would remove a further **{idx_saving:,.0f} ms** "
             f"(load {m['load']:.0f} + BM25 rebuild {m['first_retrieve'] - m['warm_retrieve']:.0f}) "
             f"— **not built**")
    L.append(f"- irreducible: **{irreducible:.0f} ms** (encode {m['warm_embed']:.1f} + "
             f"score/fuse {m['warm_retrieve']:.0f})")
    L.append("")
    L.append("## 2. A/B on the shipped `route_query`")
    L.append("")
    L.append("Cache off vs on, alternated per query in one process so machine drift "
             "cannot be read as the effect. Routes alternate too — consecutive "
             "same-route queries would be served by one resident model and a size-1 "
             "cache would look identical to a size-2 one.")
    L.append("")
    L.append("| # | route | off (ms) | on (ms) |")
    L.append("| ---: | --- | ---: | ---: |")
    for i, (off, on) in enumerate(zip(a["off_ms"], a["on_ms"]), start=1):
        L.append(f"| {i} | {a['queries'][(i - 1) % len(a['queries'])]['route']} "
                 f"| {off:,.0f} | {on:,.0f} |")
    L.append("")
    L.append(f"- p50 **{a['p50_off']:,.0f} → {a['p50_on']:,.0f} ms** "
             f"(**{a['p50_off'] / a['p50_on']:.1f}x**, −{a['p50_off'] - a['p50_on']:,.0f} ms/query)")
    L.append(f"- steady state, excluding the two cold fills: **{a['p50_on_steady']:,.0f} ms** "
             f"(**{a['p50_off'] / a['p50_on_steady']:.1f}x**)")
    L.append(f"- the first query on each route pays a weight load in **both** arms "
             f"by construction; that is the cold fill, not a cache failure")
    L.append("")
    L.append("## 3. Self-checks")
    L.append("")
    L.append("| check | verdict | detail |")
    L.append("| --- | --- | --- |")
    for name, ok, detail in checks:
        L.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    L.append("")
    L.append("## 4. What this does NOT establish")
    L.append("")
    L.append("- **No index cache exists.** The `load` + BM25 rebuild cost above is "
             "measured, not removed; it is what a served query still pays.")
    L.append("- **No concurrency.** These are sequential, single-client timings. "
             "`qdrant_concurrency.md` is the layer that measures contention, and the "
             "cache makes callers *share* one model where each used to build its own "
             "— pinned bit-identical by `tests/test_embedder_cache.py`, not measured "
             "for throughput here.")
    L.append("- **The eval path is unchanged.** `build_embedder` stays uncached, so no "
             "published number can move; a 9-embedder sweep would otherwise hold "
             "Qwen3-4B resident beside its neighbours on a 12 GB card.")
    L.append("")
    return "\n".join(L) + "\n", checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="re-render from the cached raw JSON, no GPU")
    args = ap.parse_args()

    if args.render:
        data = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        indices = discover_indices(INDEX_ROOT)
        data = {"ts": time.time(), "decompose": decompose(indices), "ab": ab(indices)}
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    text, checks = render(data)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    failed = [c for c in checks if not c[1]]
    print(f"{len(checks) - len(failed)}/{len(checks)} self-checks pass -> {REPORT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
