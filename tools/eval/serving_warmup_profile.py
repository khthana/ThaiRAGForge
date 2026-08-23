"""What a startup warm-up buys, measured per process state.

`serving_cost_profile.md` prices the two caches and `serving_concurrency.md`
reports one warm-up cost per topology. Neither answers the question a deployment
actually asks -- **how much does the FIRST user pay, and how much of it does
`warm_serving_caches` remove** -- and the figures for it have been quoted in
CLAUDE.md, the journey doc and code-explained since 2026-08-21 while living in no
report at all. `D7` flagged 26 such serving figures on its first run; this script
is what turns the warm-up half of them into an artifact.

A warm-up state is a property of a FRESH PROCESS, so it cannot be measured in a
loop: every arm runs in its own child, and the parent only aggregates. Arms:

    cold            nothing warmed -- what the first user pays today with
                    RAG_LAB_WARM_ON_START unset
    warm_no_probe   warm_serving_caches(probe_retrieval=False): every index and
                    embedder resident, and nothing else
    warm_probe      warm_serving_caches(): the shipped configuration, which adds
                    exactly one throwaway retrieval

The third arm exists because of a finding that is easy to disbelieve:
**everything resident is still not warm.** The residue is process-global
CUDA/BLAS initialisation, not per-index, so one probe fixes all four routes --
and a fourth child measures what that probe costs at the shipped
`fetch_depth=200` against the class default `None`, because a probe left at the
default fuses over the whole corpus and charges a slower code path than any user
query takes.

Self-checks, all reported in the output:

    S1  every arm returns the same top-10 for every query. Warming changes
        speed, not answers; if it changed an answer this whole file would be
        measuring two different systems.
    S2  the four queries resolve to four DISTINCT routes. Consecutive same-route
        queries would be served by one resident model and one resident index,
        and the cold arm would understate what a first user pays.
    S3  the steady-state figure is anchored against serving_cost_profile.md,
        PARSED from the report rather than frozen as a literal.
    S4  the probe at fetch_depth=None really is slower than at 200 -- otherwise
        the knob is inert and the claim about it is vacuous.
    S5  each arm's caches hold what the arm claims (entry counts), so a warm arm
        that silently failed to warm cannot read as a fast cold one.

Run:
    PYTHONPATH=src python tools/eval/serving_warmup_profile.py
    PYTHONPATH=src python tools/eval/serving_warmup_profile.py --render
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

REPORT = REPO / "data/results/serving_warmup_profile.md"
RAW = REPO / "data/results/serving_warmup_profile_raw.json"
COST_REPORT = REPO / "data/results/serving_cost_profile.md"
INDEX_ROOT = REPO / "data/index/chunker_compare_full"

# The same four queries `serving_cost_profile.py` uses, and for the same reason:
# alternating routes. A deployment alternates; so does this.
QUERIES = [
    "ผู้ช่วยศาสตราจารย์ ดร. ธนา หงษ์สุวรรณ",
    "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมคอมพิวเตอร์",
    "รองศาสตราจารย์ ดร. สมชาย",
    "รายวิชา CALCULUS 2",
]
ARMS = ["cold", "warm_no_probe", "warm_probe"]
K = 10
FETCH_DEPTH = 200
# Three passes per arm, each in its own process, because a single reading cannot
# state its own resolution -- and this rig's is coarse: `serving_cost_profile.md`
# reads ~6% run to run on an UNCHANGED script, which is more than several of the
# effects other serving work has argued about.
REPEATS = 3
LF = chr(10)


def steady_ms(arm: str, per_query: list[dict]) -> float | None:
    """The best of the 2nd-4th queries, or None when the arm has no steady state.

    `cold` has NO steady state and the column must not pretend otherwise: its
    four queries are four FIRST callers, one per route, each loading an index and
    possibly an embedder of its own. Printing the minimum anyway would put a
    plausible small number in the table and hide the shape of the problem -- a
    deployment without a warm-up does not get one slow query and then fast ones,
    it gets one slow query per route.

    A free function rather than three lines inside `run_child` so it can be
    tested without a GPU: `run_child` loads four indices and two embedders, so a
    test that had to call it could not pin this rule at all.
    """
    if arm == "cold":
        return None
    return round(min(r["ms"] for r in per_query[1:]), 1)


# --------------------------------------------------------------------- child
def run_child(arm: str) -> dict:
    """One arm, in its own process. Prints a single JSON line on stdout."""
    from rag_lab.config import StrategySpec
    from rag_lab.factory import embedder_cache_info
    from rag_lab.io.index_cache import index_cache_info
    from rag_lab.query_service import discover_indices, route_query, warm_serving_caches
    from rag_lab.router import classify_query

    spec = StrategySpec(type="hybrid", params={"fetch_depth": FETCH_DEPTH})
    indices = discover_indices(INDEX_ROOT)

    warm_ms = 0.0
    if arm != "cold":
        t0 = time.perf_counter()
        warm_serving_caches(
            indices,
            retriever_type="hybrid",
            retriever_params={"fetch_depth": FETCH_DEPTH},
            probe_retrieval=(arm == "warm_probe"),
        )
        warm_ms = (time.perf_counter() - t0) * 1000

    per_query = []
    for q in QUERIES:
        t0 = time.perf_counter()
        res = route_query(q, indices, spec, K)
        ms = (time.perf_counter() - t0) * 1000
        per_query.append({
            "query": q,
            "route": classify_query(q),
            "ms": round(ms, 1),
            "ids": [c.chunk_id for c in res.results[:K]],
        })

    return {
        "arm": arm,
        "warm_ms": round(warm_ms, 1),
        "queries": per_query,
        "first_ms": per_query[0]["ms"],
        "four_ms": round(sum(r["ms"] for r in per_query), 1),
        "steady_ms": steady_ms(arm, per_query),
        "index_cache_entries": index_cache_info()["size"],
        "embedder_cache_entries": embedder_cache_info()["size"],
    }


def run_probe_child() -> dict:
    """What the warm-up's own probe costs at two fetch depths.

    Not a user-visible latency: it is what `warm_serving_caches` pays, and the
    point is that leaving `fetch_depth` at the class default warms a code path
    (fusion over the whole corpus) that no shipped query takes.
    """
    from rag_lab.config import StrategySpec
    from rag_lab.query_service import discover_indices, route_query, warm_serving_caches

    indices = discover_indices(INDEX_ROOT)
    # Warm everything first: this measures the PROBE, not the loading under it.
    warm_serving_caches(
        indices, retriever_type="hybrid",
        retriever_params={"fetch_depth": FETCH_DEPTH}, probe_retrieval=True,
    )
    out = {}
    for label, params in (("F=200", {"fetch_depth": FETCH_DEPTH}),
                          ("F=None (class default)", {})):
        spec = StrategySpec(type="hybrid", params=params)
        route_query(QUERIES[0], indices, spec, K)          # settle
        t0 = time.perf_counter()
        route_query(QUERIES[0], indices, spec, K)
        out[label] = round((time.perf_counter() - t0) * 1000, 1)
    return {"probe": out}


# -------------------------------------------------------------------- parent
def median_arm(arm: str, runs: list[dict]) -> dict:
    """Collapse an arm's passes to a median, carrying the spread with it.

    Median rather than mean because three readings on a machine that sometimes
    stalls should not be dragged by one of them, and the spread is reported
    beside every figure so a reader can see when two arms are inside the noise.
    """
    mine = [r for r in runs if r["arm"] == arm]
    def med(key):
        vals = sorted(r[key] for r in mine if r[key] is not None)
        return None if not vals else round(vals[len(vals) // 2], 1)
    def spread(key):
        vals = [r[key] for r in mine if r[key] is not None]
        return None if not vals else (round(min(vals), 1), round(max(vals), 1))
    keys = ("warm_ms", "first_ms", "four_ms", "steady_ms")
    out = {"arm": arm, "n": len(mine),
           "queries": mine[0]["queries"],
           "index_cache_entries": mine[0]["index_cache_entries"],
           "embedder_cache_entries": mine[0]["embedder_cache_entries"]}
    out.update({k: med(k) for k in keys})
    out["spread"] = {k: spread(k) for k in keys}
    return out


def spawn(what: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", what],
        capture_output=True, text=True, env=env, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise SystemExit(f"child {what} failed:\n{r.stdout[-2000:]}\n{r.stderr[-4000:]}")
    line = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
    return json.loads(line)


def parse_cost_steady() -> float | None:
    """serving_cost_profile.md's `both` steady state, parsed not frozen.

    A frozen literal in a cross-artifact anchor is how 14 wrong numbers got
    printed elsewhere in this repo; the anchor has to read the report.
    """
    if not COST_REPORT.exists():
        return None
    for ln in COST_REPORT.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] == "both":
            try:
                return float(cells[2].replace(",", ""))
            except ValueError:
                return None
    return None


def render(data: dict) -> str:
    arms = {a["arm"]: a for a in data["arms"]}
    probe = data["probe"]
    f = lambda x: "—" if x is None else f"{x:,.1f}"

    def cell(r, key):
        lo_hi = r.get("spread", {}).get(key)
        if r[key] is None:
            return "n/a"
        if not lo_hi or lo_hi[0] == lo_hi[1]:
            return f(r[key])
        return f"{f(r[key])} <br><sub>{f(lo_hi[0])}–{f(lo_hi[1])}</sub>"

    n_routes = len({q["route"] for q in arms["cold"]["queries"]})
    n = arms["cold"]["n"]
    L = ["# What a startup warm-up buys — measured per process state", ""]
    L.append("Generated by `tools/eval/serving_warmup_profile.py`.")
    L.append("")
    L.append(f"{len(QUERIES)} routed queries over {n_routes} distinct routes, `hybrid` at "
             f"`fetch_depth={FETCH_DEPTH}`, k={K}. Each arm runs in its **own process** — a "
             "warm-up state is a property of a fresh process and cannot be measured in a "
             f"loop — and each is run **{n} times**, so every figure below is a median with "
             "its own min–max underneath it. Read the spread before reading a gap: this rig "
             "moves several percent on an unchanged script, which is more than some of the "
             "differences other serving work has argued about.")
    L.append("")
    L.append("## 1. What the first user pays")
    L.append("")
    L.append("| arm | warm-up | 1st query | 4 queries | steady (best of 2nd–4th) |")
    L.append("|---|---:|---:|---:|---:|")
    for a in ARMS:
        r = arms[a]
        L.append(f"| `{a}` | {cell(r, 'warm_ms')} | **{cell(r, 'first_ms')}** | "
                 f"{cell(r, 'four_ms')} | {cell(r, 'steady_ms')} |")
    L.append("")
    L.append("All figures in ms.")
    L.append("")
    cold, nop, prb = arms["cold"], arms["warm_no_probe"], arms["warm_probe"]
    L.append(f"- front-loading takes the first {len(QUERIES)} routed queries from "
             f"**{f(cold['four_ms'])} ms to {f(prb['four_ms'])} ms**, after a "
             f"**{f(prb['warm_ms'])} ms** warm-up")
    L.append(f"- **`cold` has no steady state, and that is the shape of the problem, not a "
             f"gap in the table**: its {len(QUERIES)} queries are {len(QUERIES)} *first* "
             f"callers, one per route, each loading an index and possibly an embedder of "
             f"its own. A deployment does not get one slow query and then fast ones — it "
             f"gets one slow query per route.")
    L.append(f"- **everything resident is still not warm**: with every index and embedder "
             f"loaded but no probe, the first real query still costs "
             f"**{f(nop['first_ms'])} ms** against {f(nop['steady_ms'])} ms for the ones "
             f"after it; one throwaway retrieval takes it to **{f(prb['first_ms'])} ms**. "
             f"The residue is process-global CUDA/BLAS initialisation, **not per-index** — "
             f"one probe fixes all four routes, which is why the warm-up does exactly one")
    # The probe's cost is a DIFFERENCE OF TWO MEDIANS, so it is only quotable when
    # it is larger than the noise the two arms carry. At n=3 it often is not, and
    # saying so is the honest reading -- section 2 measures the same retrieval
    # directly and does not depend on this subtraction.
    delta = round(prb["warm_ms"] - nop["warm_ms"], 1)
    noise = max(prb["spread"]["warm_ms"][1] - prb["spread"]["warm_ms"][0],
                nop["spread"]["warm_ms"][1] - nop["spread"]["warm_ms"][0])
    if abs(delta) > noise:
        L.append(f"- that probe costs **{f(delta)} ms** of the warm-up, against the "
                 f"{f(probe['F=200'])} ms a warm query at the same depth takes — the "
                 f"excess *is* the process-global initialisation it exists to pay")
    else:
        L.append(f"- **what the probe adds to the warm-up is not separable from noise at "
                 f"n={n}**: the two warm arms differ by {f(delta)} ms while their own "
                 f"passes spread {f(noise)} ms. Quote §2's {f(probe['F=200'])} ms — the "
                 f"same retrieval measured directly — for what one probe costs, not this "
                 f"subtraction")
    L.append("")
    L.append("## 2. The probe's own params are part of the measurement")
    L.append("")
    L.append("Both rows below are a **fully warm** query, so the only difference between "
             "them is the fusion depth:")
    L.append("")
    L.append("| query configuration | ms |")
    L.append("|---|---:|")
    for k, v in probe.items():
        L.append(f"| `{k}` | {f(v)} |")
    L.append("")
    L.append("A probe left at the class default fuses over the **whole corpus**, so it "
             "warms a slower code path than any shipped query takes and charges the "
             "difference to startup. `warm_serving_caches` is therefore given the params "
             "the deployment serves, rather than the class defaults.")
    L.append("")
    L.append("## 3. Self-checks")
    L.append("")
    L.append("| check | verdict | detail |")
    L.append("|---|---|---|")
    for c, v, d in data["checks"]:
        L.append(f"| {c} | {v} | {d} |")
    L.append("")
    L.append("**Off by default.** The warm-up holds the whole serving footprint "
             "(`serving_cache_memory.md`: host RAM for four indices plus VRAM for two "
             "embedders) on a card the eval scripts share, so an automatic grab at UI "
             "start is how a GPU run dies. A deployment sets `RAG_LAB_WARM_ON_START=1`.")
    L.append("")
    return LF.join(L)


def checks(data: dict) -> list[tuple[str, str, str]]:
    arms = {a["arm"]: a for a in data["arms"]}
    out = []

    # S1 -- the correctness gate. Warming may change speed, never answers. Run
    # over every PASS, not over the medians: the medians are one pass's ids by
    # construction, so checking them would compare three arms and miss the case
    # where one arm disagreed with itself between passes.
    runs = data["runs"]
    diffs = 0
    for i in range(len(QUERIES)):
        diffs += len({tuple(r["queries"][i]["ids"]) for r in runs}) - 1
    out.append(("S1 every arm returns the same top-10", "PASS" if diffs == 0 else "FAIL",
                f"{diffs} disagreements over {len(QUERIES)} queries x {len(runs)} runs"))

    # S2 -- four distinct routes, or the cold arm understates the first user's cost.
    routes = [q["route"] for q in arms["cold"]["queries"]]
    out.append(("S2 the queries cover distinct routes",
                "PASS" if len(set(routes)) == len(routes) else "FAIL",
                f"{len(set(routes))} distinct of {len(routes)}: {', '.join(routes)}"))

    # S3 -- anchored against another report, parsed rather than frozen.
    published = parse_cost_steady()
    got = arms["warm_probe"]["steady_ms"]
    if published is None:
        out.append(("S3 steady state anchored to serving_cost_profile.md", "WARN",
                    "UNPARSED -- the cross-check could not be made"))
    else:
        rel = abs(got - published) / published
        out.append(("S3 steady state anchored to serving_cost_profile.md",
                    "PASS" if rel <= 0.35 else "FAIL",
                    f"{got:,.1f} ms here vs {published:,.1f} ms published ({rel:.1%}; "
                    f"that report's own run-to-run spread is ~6%, and this is a median "
                    f"of per-pass minima rather than a p50 over 8 queries)"))

    # S4 -- the mechanism under test must be live, or section 2 is vacuous.
    p = data["probe"]
    deep, shallow = p["F=None (class default)"], p["F=200"]
    out.append(("S4 the probe's fetch_depth is not inert",
                "PASS" if deep > shallow else "FAIL",
                f"F=None {deep:,.1f} ms vs F=200 {shallow:,.1f} ms"))

    # S5 -- a warm arm that silently failed to warm would read as a fast cold one.
    bad = [a for a in ("warm_no_probe", "warm_probe")
           if arms[a]["index_cache_entries"] < 4 or arms[a]["embedder_cache_entries"] < 2]
    # `cold` is deliberately not checked: its caches end up full too, because the
    # four queries fill them. What it does not have is them full BEFOREHAND, which
    # is what the first-query column measures.
    out.append(("S5 each warmed arm's caches hold what it claims",
                "PASS" if not bad else "FAIL",
                "; ".join(f"{a}: {arms[a]['index_cache_entries']} indices / "
                          f"{arms[a]['embedder_cache_entries']} embedders"
                          for a in ("warm_no_probe", "warm_probe"))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--render", action="store_true",
                    help="re-render the report from the cached raw JSON (no GPU)")
    args = ap.parse_args()

    if args.child:
        print(json.dumps(run_probe_child() if args.child == "probe"
                         else run_child(args.child), ensure_ascii=False))
        return 0

    if args.render:
        data = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        data = {"arms": [], "runs": []}
        for rep in range(REPEATS):
            for arm in ARMS:
                print(f"  {arm} (pass {rep + 1}/{REPEATS}) ...", flush=True)
                r = spawn(arm)
                r["pass"] = rep + 1
                data["runs"].append(r)
        data["arms"] = [median_arm(a, data["runs"]) for a in ARMS]
        print("  probe ...", flush=True)
        data.update(spawn("probe"))
        data["checks"] = checks(data)
        RAW.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    REPORT.write_text(render(data), encoding="utf-8")
    for c, v, d in data["checks"]:
        print(f"  [{v}] {c}: {d}")
    print(f"wrote {REPORT.relative_to(REPO)}")
    return 1 if any(v == "FAIL" for _, v, _ in data["checks"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
