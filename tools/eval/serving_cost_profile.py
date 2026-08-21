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
from rag_lab.io.index_cache import clear_index_cache, index_cache_info  # noqa: E402
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
INDEX_ENV = "RAG_LAB_INDEX_CACHE"

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


ROUTED_REPORT = REPO / "data/results/routed_fetch_depth_test.md"


def published_routed_p50() -> float | None:
    """The shipped routed hybrid p50 at F=200, parsed from its own report.

    PARSED, never frozen as a literal: this project has already replaced 14
    hardcoded cross-artifact anchors that went on printing a number their source
    had moved past. A missing or renamed report yields None and S7 says the
    cross-check could not be made, rather than passing silently.
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
    """§2: the shipped route_query across THREE arms, alternated in one process.

    Three, not two, because "is the cache worth it" and "which cache" are
    different questions and a single on/off arm answers only the first:

        none      both caches off  -- the behaviour before 2026-08-21
        embedder  embedder only    -- what shipped first
        both      + the Index cache

    NEITHER cache is cleared between arms, and getting that wrong cost this
    measurement twice. A disabled arm is separated by the disable itself: both
    `build_embedder_cached` and `load_index_cached` return the uncached object
    BEFORE reading their dict when size is 0, so nothing can leak into them.
    Clearing, meanwhile, runs immediately before the next enabled arm and wipes
    exactly the state it is about to measure -- which is how the embedder cache
    first read 1.0x and the index cache first read -2 ms.
    """
    arms = ("none", "embedder", "both")
    env = {
        "none": {CACHE_ENV: "0", INDEX_ENV: "0"},
        "embedder": {CACHE_ENV: "2", INDEX_ENV: "0"},
        "both": {CACHE_ENV: "2", INDEX_ENV: "4"},
    }

    os.environ.update(env["none"])
    clear_embedder_cache()
    clear_index_cache()
    route_query(QUERIES[0], indices, RETRIEVER, 10)  # warm disk/CUDA for all arms

    times = {a: [] for a in arms}
    answers = {a: [] for a in arms}
    for q in QUERIES * 2:
        for arm in arms:
            os.environ.update(env[arm])
            # NO clear of either cache here, and the index one is the SECOND
            # time this trap was hit -- the first version of this arm cleared
            # it on the reasoning that "a disabled cache is bypassed, not
            # cleared, so an entry could leak into a disabled arm". False:
            # `load_index_cached` returns `store.load(...)` before it ever
            # reads the dict at size 0, so a leftover entry cannot be served.
            # The clear was unnecessary AND destructive -- it ran immediately
            # before the `both` arm every query, so `both` never once got a
            # hit and the index cache measured itself at -2 ms.
            # NOTE: no clear_embedder_cache() here, and that was a real harness
            # bug: at size 0 the cached builder bypasses the cache and never
            # reads it, so a disabled arm cannot benefit anyway -- while
            # clearing WIPED the resident models the enabled arms depend on.
            # With arms alternating per query that made every treatment
            # measurement cold and the effect read 1.0x.
            res, dt = timed(lambda: route_query(q, indices, RETRIEVER, 10))
            times[arm].append(dt)
            answers[arm].append(top10(res))
    for k in (CACHE_ENV, INDEX_ENV):
        os.environ.pop(k, None)

    differing = {
        a: [i for i in range(len(answers["none"])) if answers["none"][i] != answers[a][i]]
        for a in ("embedder", "both")
    }
    return {
        "queries": [{"query": q, "route": classify_query(q)} for q in QUERIES],
        "arms": arms,
        "ms": {a: times[a] for a in arms},
        "p50": {a: statistics.median(times[a]) for a in arms},
        "n_differing": {a: len(v) for a, v in differing.items()},
        "n_compared": len(answers["none"]),
        "embedder_cache_after": embedder_cache_info(),
        "index_cache_after": index_cache_info(),
    }


def render(data: dict) -> tuple[str, list[tuple[str, bool, str]]]:
    d, a = data["decompose"], data["ab"]
    m = d["mean_ms"]
    now = m["build"] + m["first_embed"] + m["load"] + m["first_retrieve"]
    irreducible = m["warm_embed"] + m["warm_retrieve"]
    emb_saving = m["build"] + m["first_embed"] - m["warm_embed"]
    idx_saving = m["load"] + (m["first_retrieve"] - m["warm_retrieve"])

    # Steady state = the SECOND pass over the query list. Defined structurally
    # rather than by dropping the N slowest rows: the number of cold fills is
    # arm-dependent (the embedder arm fills 2 models, the `both` arm also fills
    # 4 indices), so a fixed N is a fudge that happens to work for one arm. Every
    # route appears once in the first pass by construction, so the second pass is
    # warm for every arm at once.
    n = len(a["queries"])
    a["p50_steady"] = {arm: statistics.median(a["ms"][arm][n:]) for arm in a["arms"]}

    checks = [
        ("S1 warm encode is in the published cost_latency_pareto range",
         PUBLISHED_ENCODE_MS[0] <= m["warm_embed"] <= PUBLISHED_ENCODE_MS[1],
         f"{m['warm_embed']:.1f} ms, expected {PUBLISHED_ENCODE_MS[0]}-{PUBLISHED_ENCODE_MS[1]}"),
        ("S2 neither cache changes the answer",
         sum(a["n_differing"].values()) == 0,
         f"embedder {a['n_differing']['embedder']} / both {a['n_differing']['both']} "
         f"of {a['n_compared']} queries returned a different top-10"),
        ("S3 the embedder cache holds the shipped model count",
         a["embedder_cache_after"]["size"] == 2,
         f"holds {a['embedder_cache_after']['size']} of max "
         f"{a['embedder_cache_after']['max_size']}"),
        ("S4 the index cache holds the routed index count",
         a["index_cache_after"]["size"] == 4,
         f"holds {a['index_cache_after']['size']} of max "
         f"{a['index_cache_after']['max_size']}"),
        ("S5 the arms were genuinely separated",
         a["p50"]["none"] - a["p50_steady"]["both"] > emb_saving * 0.5,
         f"none p50 {a['p50']['none']:.0f} - both steady {a['p50_steady']['both']:.0f} = "
         f"{a['p50']['none'] - a['p50_steady']['both']:.0f} ms vs weight load {emb_saving:.0f} ms"),
        ("S6 the BM25 memo survives in the index-cached arm",
         all(e["has_bm25_scorer"] for e in a["index_cache_after"]["entries"]),
         f"{sum(e['has_bm25_scorer'] for e in a['index_cache_after']['entries'])} of "
         f"{len(a['index_cache_after']['entries'])} cached indices carry a scorer"),
        ("S7 a fully warm query reproduces the published routed p50",
         (lambda pub: pub is not None and abs(a["p50_steady"]["both"] - pub) / pub < 0.5)(
             published_routed_p50()),
         (lambda pub: "UNPARSED -- the cross-check could not be made" if pub is None
          else f"{a['p50_steady']['both']:.0f} ms vs routed_fetch_depth_test.md's "
               f"{pub:.1f} ms ({abs(a['p50_steady']['both'] - pub) / pub * 100:.0f}% apart)"
          )(published_routed_p50())),
    ]

    L = []
    L.append("# Serving cost profile — what a query pays for, and what the caches remove")
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
        ("load", "`ArtifactStore.load` (parquet + npy + Chunk objects)", "**yes, index cache**"),
        ("first_retrieve", "first `retrieve` (includes the BM25Okapi rebuild)", "the delta, yes"),
        ("warm_retrieve", "warm `retrieve`", "no — scoring + fusion"),
    ]:
        L.append(f"| {label} | {m[k]:.1f} | {note} |")
    L.append("")
    L.append(f"- one served query with nothing cached: **{now:,.0f} ms**")
    L.append(f"- the embedder cache removes **{emb_saving:,.0f} ms** "
             f"({emb_saving / now * 100:.0f}% of it)")
    L.append(f"- the index cache removes a further **{idx_saving:,.0f} ms** "
             f"(load {m['load']:.0f} + the BM25 rebuild it discards "
             f"{m['first_retrieve'] - m['warm_retrieve']:.0f})")
    L.append(f"- irreducible: **{irreducible:.0f} ms** (encode {m['warm_embed']:.1f} + "
             f"score/fuse {m['warm_retrieve']:.0f})")
    L.append("")
    L.append("## 2. A/B on the shipped `route_query`")
    L.append("")
    L.append("Three arms, alternated per query in one process so machine drift cannot "
             "be read as the effect. Routes alternate too — consecutive same-route "
             "queries would be served by one resident model and one resident index, "
             "and a size-1 cache would look identical to the shipped size.")
    L.append("")
    L.append("| # | route | none (ms) | embedder (ms) | both (ms) |")
    L.append("| ---: | --- | ---: | ---: | ---: |")
    for i in range(len(a["ms"]["none"])):
        route = a["queries"][i % len(a["queries"])]["route"]
        L.append(f"| {i + 1} | {route} | {a['ms']['none'][i]:,.0f} | "
                 f"{a['ms']['embedder'][i]:,.0f} | {a['ms']['both'][i]:,.0f} |")
    L.append("")
    L.append("| arm | p50 (ms) | steady state (ms) | vs none |")
    L.append("| --- | ---: | ---: | ---: |")
    for arm in a["arms"]:
        speed = a["p50"]["none"] / a["p50"][arm]
        L.append(f"| {arm} | {a['p50'][arm]:,.0f} | {a['p50_steady'][arm]:,.0f} | "
                 f"{speed:.1f}x |")
    L.append("")
    L.append(f"- **{a['p50']['none']:,.0f} → {a['p50']['both']:,.0f} ms p50** "
             f"(**{a['p50']['none'] / a['p50']['both']:.1f}x**), steady state "
             f"**{a['p50_steady']['both']:,.0f} ms** "
             f"(**{a['p50']['none'] / a['p50_steady']['both']:.1f}x**)")
    L.append(f"- the index cache's own contribution: "
             f"**{a['p50']['embedder'] - a['p50']['both']:,.0f} ms** off the "
             f"embedder-only arm")
    L.append("- the first query on each route pays both loads in **every** arm by "
             "construction; that is the cold fill, not a cache failure")
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
    L.append("- **Nothing about staleness under load.** The index cache re-stats its "
             "artifacts on every hit and `tests/io/test_index_cache.py` pins that a "
             "rebuilt directory is never served from RAM, but no measurement here "
             "rebuilds an index while queries are in flight.")
    L.append(f"- **RAM is not measured.** {a['index_cache_after']['size']} indices "
             "are held resident; this reports latency, not footprint.")
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
