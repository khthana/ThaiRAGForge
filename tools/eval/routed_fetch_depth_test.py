"""Does the hybrid fetch-depth cut still cost only -0.0033 once the router ships?

`hybrid_fetch_depth_sweep.py` (2026-08-09) measured the `k=n` over-fetch and left
one thing open before the knob could be shipped: its headline
**-0.0033 macro recall@10 for -0.67s/query at F=200** is an average over 36
combos retrieving without a router, and hard routing has shipped since
2026-08-08. This project has now been burned twice by exactly that pairing --
per-`entity_type` alpha and the rrf4 reranker both looked worth wiring against
an unrouted baseline and were worth nothing against the shipped one -- so the
sweep's number is not yet a reason to ship anything, in either direction.

Note the asymmetry with those two cases, because it changes what a null MEANS
here. There, the intervention had to *win* and a null killed it. Here truncation
only has to not *lose*: a null is the outcome that licenses shipping. A null is
weak evidence on its own, so every row is reported as a **bound** -- "rules out a
loss worse than X" -- per this project's own rule, and the pre-registered family
is the one depth a decision would actually be made on.

PRE-REGISTERED (fixed before the run)
-------------------------------------
F=200 vs k=n on the routed system, 3 metrics, one Holm family of m=3. F=200 is
the sweep's own candidate: it is the shallowest depth whose macro damage was
already inside the non-monotonic noise (-0.0033, against F=500's -0.0018 and
F=1000's -0.0026). Every other depth in the grid is descriptive.

PREDICTION (registered before the run, so an aggregate null cannot be read as
confirming any mechanism it did not test). The unrouted sweep found `person`
queries *gain* at shallow F (+0.0212 at F=50) -- the only entity_type that does
-- because BM25 carries `person` (0.8147) and the cut deletes a weak dense arm's
tail. Hard routing already hands `person` its dense specialist, which has no such
tail. So that gain should **shrink** under routing. If it grows instead, the
mechanism recorded in CLAUDE.md for three separate results is wrong.

BUDGET (stated on every row, per the project's rule): every arm **sends k=10**.
The arms differ only in how many candidates they **fetch** before fusing -- F per
arm, or the whole corpus at k=n. No arm sees documents another cannot.

ANCHORS
-------
Two of the four corners are already published and are checked, not assumed:
S2 reproduces `routing_eval.md`'s `routed (shipped)` hybrid **0.6831** and S3
reproduces the unrouted single-combo **0.6281**, both from this independent code
path. S4 is the one that makes the truncated columns mean anything -- it checks
the numpy fusion against a real `HybridRetriever(fetch_depth=F)`, because the
anchors above exercise only F=n, where the mechanism under test is inert.

The fusion itself is **imported** from `hybrid_fetch_depth_sweep.py` rather than
re-implemented: two copies of a tie-break this fiddly would eventually disagree,
and that module's S5 already pins it against the shipped retriever.

Run:
    .venv/Scripts/python.exe tools/eval/routed_fetch_depth_test.py --smoke
    .venv/Scripts/python.exe tools/eval/routed_fetch_depth_test.py
    .venv/Scripts/python.exe tools/eval/routed_fetch_depth_test.py --latency  # idle machine
    .venv/Scripts/python.exe tools/eval/routed_fetch_depth_test.py --render   # no GPU
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from rank_bm25 import BM25Okapi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from pythainlp.tokenize import word_tokenize  # noqa: E402

from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402
from hybrid_fetch_depth_sweep import fuse_at_depth  # noqa: E402
from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder  # noqa: E402
from rag_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from rag_lab.query_service import discover_indices, resolve_index  # noqa: E402
from rag_lab.router import classify_query, route_targets  # noqa: E402
from rag_lab.schema import RankedChunk, RetrievalResult  # noqa: E402

INDEX_ROOT = REPO / "data" / "index" / "chunker_compare_full"
UNROUTED_COMBO = "plain__sentence__qwen3__ff8f6c49"
HYB_RES = REPO / "data" / "results" / "gold_hybrid_73det"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
OUT = REPO / "data" / "results" / "routed_fetch_depth_test.md"
RAW = REPO / "data" / "results" / "routed_fetch_depth_raw.json"
LAT = REPO / "data" / "results" / "routed_fetch_depth_latency.json"

K = 10
DEPTHS = [10, 20, 50, 100, 200, 500, 1000, 5000]
F_REGISTERED = 200
N_BOOT = 10_000
SEED = 42
# S2/S3 anchor on routing_eval.md's hybrid section, and the two values are
# PARSED from it rather than frozen here. They were literals (0.6831 / 0.6281)
# until 2026-08-18, when the rebuild-#4 refresh legitimately moved both to
# 0.6811 / 0.6229 and the checks went red against a report the live code in
# fact reproduced exactly -- a cross-artifact anchor written as a constant
# stops anchoring the moment the artifact moves, and then blames the wrong
# side. An unparseable anchor FAILS rather than skipping: a check that cannot
# find its counterpart must not pass quietly.
_ROUTING_EVAL = REPO / "data" / "results" / "routing_eval.md"


def published_hybrid_anchors() -> dict[str, float | None]:
    """`routed (shipped)` and `best single combo` recall@10 from routing_eval.md."""
    if not _ROUTING_EVAL.exists():
        return {"routed": None, "unrouted": None}
    return parse_hybrid_anchors(_ROUTING_EVAL.read_text(encoding="utf-8"))


def parse_hybrid_anchors(txt: str) -> dict[str, float | None]:
    """Same, from the report's text -- split out so a test can pin the scoping.

    Scoped to the *hybrid* half of the report: the dense half carries rows with
    identical labels (`recall@10 | routed (shipped) | ...`), so an unscoped
    search silently anchors on the wrong retriever and the whole check compares
    the right numbers against the wrong published ones.
    """
    out: dict[str, float | None] = {"routed": None, "unrouted": None}
    start = txt.find("## 3. Routed system vs single-combo baselines -- hybrid")
    if start < 0:
        return out
    section = txt[start:]
    end = section.find("\n## ", 1)
    if end > 0:
        section = section[:end]
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] != "recall@10":
            continue
        try:
            value = float(cells[2])
        except ValueError:
            continue
        if cells[1] == "routed (shipped)":
            out["routed"] = value
        elif cells[1].startswith("best single combo ="):
            out["unrouted"] = value
    return out


PUBLISHED = published_hybrid_anchors()
# paper-results-summary.md: mean(min(1, k/n_relevant)) over the 106-query set
QRELS_CEILING = 0.8856

_METRICS = {
    "recall@10": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    "ndcg@10": lambda r, rel: ndcg_at_k(r, rel, K),
}


def as_result(query: str, rows, cid, rid, page, label: str) -> RetrievalResult:
    return RetrievalResult(
        query=query, combination_id=label, top_k=len(rows), retriever=label,
        results=[
            RankedChunk(chunk_id=cid[r], resolution_id=rid[r], page=int(page[r]),
                        score=0.0, rank=i + 1, text="")
            for i, r in enumerate(rows)
        ],
    )


def persisted_hybrid_top10(combo: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in HYB_RES.glob(f"{combo}__hybrid__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["query"]] = [r["chunk_id"] for r in sorted(d["results"], key=lambda r: r["rank"])]
    return out


def sweep_one_index(index_dir: Path, qs: list[str], qrels, depths: list[int]):
    """Every depth's top-10 for the queries routed to this index.

    Returns per-query metric values keyed by depth (-1 == k=n, the shipped path),
    the top-10 chunk_ids at k=n so the caller can gate them against the persisted
    results, and per-depth agreement counts with that k=n ranking.

    One index at a time and released before the next: four resident embedding
    matrices plus their models is what the GPU here cannot hold, and the same
    constraint is why `reranker_rrf_routed_test.py` is written this way.
    """
    cols = pq.read_table(
        index_dir / "chunks.parquet", columns=["chunk_id", "resolution_id", "page"]
    ).to_pydict()
    cid, rid, page = cols["chunk_id"], cols["resolution_id"], cols["page"]
    rid_arr = np.array(rid, dtype=object)
    n = len(cid)

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    emb_model = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    qvecs = {q: np.asarray(emb_model.embed_query(q), dtype=np.float64) for q in qs}
    emb_model.release()

    emb = np.load(index_dir / "embeddings.npy")
    norms = np.linalg.norm(emb, axis=1)
    bm = BM25Okapi(json.loads((index_dir / "lexical.json").read_text(encoding="utf-8")))

    scores: dict[int, dict[str, dict[str, float]]] = {
        F: {m: {} for m in _METRICS} for F in [*depths, -1]
    }
    full_top10: dict[str, list[str]] = {}
    same_order = collections.Counter()
    same_set = collections.Counter()

    for q in qs:
        qq = qvecs[q]
        den = norms * np.linalg.norm(qq)
        ds = np.divide(emb @ qq, den, out=np.zeros(n), where=den > 0)
        dorder = np.argsort(-ds)
        dpos = np.empty(n, dtype=np.int64)
        dpos[dorder] = np.arange(n)
        border = np.argsort(-bm.get_scores(word_tokenize(q)))
        bpos = np.empty(n, dtype=np.int64)
        bpos[border] = np.arange(n)

        full = fuse_at_depth(dorder, dpos, border, bpos, n)
        full_top10[q] = [cid[i] for i in full]
        full_list = list(full)
        res = as_result(q, full_list, cid, rid_arr, page, "k=n")
        for m, fn in _METRICS.items():
            scores[-1][m][q] = fn(res, qrels[q])

        for F in depths:
            top = fuse_at_depth(dorder, dpos, border, bpos, F)
            same_order[F] += int(list(top) == full_list)
            same_set[F] += int(set(top.tolist()) == set(full_list))
            res = as_result(q, list(top), cid, rid_arr, page, f"F={F}")
            for m, fn in _METRICS.items():
                scores[F][m][q] = fn(res, qrels[q])

    del emb, bm, qvecs
    return scores, full_top10, same_order, same_set


def verify_against_retriever(index_dir: Path, qs: list[str], depths: list[int]):
    """S4: the numpy truncation vs a real `HybridRetriever(fetch_depth=F)`.

    S2/S3 anchor only F=n, where truncation collapses away -- they would pass
    unchanged if `fuse_at_depth` ignored F entirely. This is the check that
    exercises the mechanism the report exists to publish, on a routed index
    rather than the sweep's unrouted anchor.
    """
    from rag_lab.io.artifact_store import ArtifactStore
    from rag_lab.retrievers.hybrid import HybridRetriever
    from rag_lab.schema import Query

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    index = ArtifactStore().load(index_dir)
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    cid = [c.chunk_id for c in index.chunks]
    n = len(cid)
    emb = np.asarray(index.embeddings)
    norms = np.linalg.norm(emb, axis=1)
    bm = BM25Okapi(index.lexical)

    agree = differ = 0
    for q in qs:
        qq = np.asarray(embedder.embed_query(q), dtype=np.float64)
        den = norms * np.linalg.norm(qq)
        ds = np.divide(emb @ qq, den, out=np.zeros(n), where=den > 0)
        dorder = np.argsort(-ds)
        dpos = np.empty(n, dtype=np.int64)
        dpos[dorder] = np.arange(n)
        toks = word_tokenize(q)
        border = np.argsort(-bm.get_scores(toks))
        bpos = np.empty(n, dtype=np.int64)
        bpos[border] = np.arange(n)
        query = Query(text=q, vector=qq, tokens=toks)
        for F in depths:
            real = [r.chunk_id for r in HybridRetriever(fetch_depth=F).retrieve(query, index, K)]
            mine = [cid[i] for i in fuse_at_depth(dorder, dpos, border, bpos, F)]
            ok = mine[: len(real)] == real and len(real) <= len(mine)
            agree, differ = agree + ok, differ + (not ok)
    embedder.release()
    del index, emb, bm
    return differ == 0, f"{agree} (query, F) pairs reproduce, {differ} differ [F in {depths}]"


def run_latency(route_of, dir_of, combo_of, queries) -> int:
    """Time the shipped k=n path against F=200/F=1000 on the ROUTED indices.

    Each query is timed on the index the router actually sends it to, arms
    alternated per query inside one process on one loaded index, with the BM25
    scorer pre-warmed so its one-off build lands in neither arm. The sweep's
    -0.67s was measured on one unrouted combo; under routing the saving depends
    on each route's own corpus size, so it has to be re-measured rather than
    carried over.
    """
    from rag_lab.io.artifact_store import ArtifactStore
    from rag_lab.retrievers.bm25 import BM25Retriever
    from rag_lab.retrievers.hybrid import HybridRetriever
    from rag_lab.schema import Query

    by_combo: dict[str, list[str]] = collections.defaultdict(list)
    for q in queries:
        by_combo[combo_of[route_of[q]]].append(q)

    timings: dict[str, list[float]] = collections.defaultdict(list)
    per_combo: dict[str, dict[str, float]] = {}
    mismatched = 0
    for combo, qs in by_combo.items():
        d = dir_of[combo]
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        print(f"loading {combo} ({len(qs)} queries) ...", file=sys.stderr)
        index = ArtifactStore().load(d)
        embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
        prepared = [
            Query(text=q, vector=embedder.embed_query(q), tokens=word_tokenize(q))
            for q in qs
        ]
        embedder.release()
        BM25Retriever()._scorer(index)  # warm the memo; its build is not under test

        arms = [("k=n", HybridRetriever())] + [
            (f"F={F}", HybridRetriever(fetch_depth=F)) for F in (1000, 200)
        ]
        local: dict[str, list[float]] = collections.defaultdict(list)
        for q in prepared:
            baseline: list[str] = []
            for label, retr in arms:
                t0 = time.perf_counter()
                ranked = retr.retrieve(q, index, K)
                dt = time.perf_counter() - t0
                local[label].append(dt)
                timings[label].append(dt)
                if label == "k=n":
                    baseline = [r.chunk_id for r in ranked]
                elif label == "F=200" and [r.chunk_id for r in ranked] != baseline:
                    mismatched += 1
        per_combo[combo] = {
            "n_chunks": len(index.chunks),
            "n_queries": len(qs),
            **{f"{lab}_p50": statistics.median(v) * 1000 for lab, v in local.items()},
        }
        del index

    stats = {}
    for label, ts in timings.items():
        s = sorted(ts)
        stats[label] = {
            "p50": statistics.median(s) * 1000,
            "p95": s[int(0.95 * len(s))] * 1000,
            "mean": statistics.mean(s) * 1000,
        }
    LAT.write_text(json.dumps({
        "n_queries": len(queries), "mismatched_200": mismatched,
        "stats": stats, "per_combo": per_combo,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"{len(queries)} queries over {len(by_combo)} routed indices, paired in one process")
    print(f"{'arm':<10} {'p50':>10} {'p95':>10} {'mean':>10}")
    for label, s in stats.items():
        print(f"{label:<10} {s['p50']:>9.1f}ms {s['p95']:>9.1f}ms {s['mean']:>9.1f}ms")
    print(f"\nF=200 top-10 differs from k=n on {mismatched} of {len(queries)} queries")
    print(f"wrote {LAT.relative_to(REPO)} -- re-render the report to include it")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="fetch_depth measured against the hard router")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    if args.render:
        return render(json.loads(RAW.read_text(encoding="utf-8")))

    t0 = time.time()
    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: list(d["relevant_resolution_ids"]) for d in raw}
    etype = {d["query"]: d.get("entity_type", "?") for d in raw}
    depths = [F for F in DEPTHS if F in (50, 200, 1000)] if args.smoke else DEPTHS
    if args.smoke:
        queries = queries[:2] + queries[26:28] + queries[60:62] + queries[-2:]

    indices = discover_indices(INDEX_ROOT)
    targets = route_targets("hybrid")
    route_of = {q: classify_query(q) for q in queries}
    resolved = {r: resolve_index(t, indices) for r, t in targets.items()}
    combo_of = {r: i.combo_id for r, i in resolved.items()}
    dir_of = {i.combo_id: Path(i.dir) for i in resolved.values()}

    if args.latency:
        return run_latency(route_of, dir_of, combo_of, queries)

    checks: list[tuple[str, bool, str]] = []
    checks.append((
        "S0 every query routes to a target index (no query falls off the map)",
        all(route_of[q] in targets for q in queries),
        f"{len(queries)} queries over {len(set(route_of.values()))} routes, "
        f"{len({combo_of[r] for r in route_of.values()})} distinct indices",
    ))

    by_combo: dict[str, list[str]] = collections.defaultdict(list)
    for q in queries:
        by_combo[combo_of[route_of[q]]].append(q)

    # ---- routed arm, one index at a time -----------------------------------
    routed: dict[int, dict[str, dict[str, float]]] = {
        F: {m: {} for m in _METRICS} for F in [*depths, -1]
    }
    same_order, same_set = collections.Counter(), collections.Counter()
    persisted_ok = 0
    for combo, qs in by_combo.items():
        sc, top10, so, ss = sweep_one_index(dir_of[combo], qs, qrels, depths)
        for F in [*depths, -1]:
            for m in _METRICS:
                routed[F][m].update(sc[F][m])
        same_order.update(so)
        same_set.update(ss)
        persisted = persisted_hybrid_top10(combo)
        persisted_ok += sum(int(top10[q] == persisted.get(q, [])) for q in qs)
        print(f"  routed {combo}  {len(qs)} queries  {time.time()-t0:.0f}s", file=sys.stderr)
    checks.append((
        "S1 the routed k=n top-10 reproduces the persisted hybrid results",
        persisted_ok == len(queries), f"{persisted_ok} of {len(queries)}",
    ))

    # ---- unrouted arm, same code path, for the damage-curve comparison ------
    u_sc, u_top10, u_so, u_ss = sweep_one_index(
        dir_of.get(UNROUTED_COMBO, INDEX_ROOT / UNROUTED_COMBO), queries, qrels, depths)
    u_persisted = persisted_hybrid_top10(UNROUTED_COMBO)
    u_ok = sum(int(u_top10[q] == u_persisted.get(q, [])) for q in queries)
    checks.append((
        "S1b the unrouted k=n top-10 reproduces the persisted hybrid results",
        u_ok == len(queries), f"{u_ok} of {len(queries)}",
    ))
    print(f"  unrouted {UNROUTED_COMBO}  {time.time()-t0:.0f}s", file=sys.stderr)

    r_base = float(np.mean([routed[-1]["recall@10"][q] for q in queries]))
    u_base = float(np.mean([u_sc[-1]["recall@10"][q] for q in queries]))
    checks.append((
        "S2 the routed k=n arm is routing_eval.md's `routed (shipped)` hybrid",
        (PUBLISHED["routed"] is not None
         and abs(r_base - PUBLISHED["routed"]) < 5e-5) or args.smoke,
        f"{r_base:.4f} vs published "
        f"{PUBLISHED['routed']:.4f}" if PUBLISHED['routed'] is not None
        else f"{r_base:.4f} vs published UNPARSEABLE (routing_eval.md missing or its hybrid section changed shape)"
        + ("  [smoke: subset]" if args.smoke else ""),
    ))
    checks.append((
        "S3 the unrouted k=n arm reproduces the published single-combo hybrid",
        (PUBLISHED["unrouted"] is not None
         and abs(u_base - PUBLISHED["unrouted"]) < 5e-5) or args.smoke,
        f"{u_base:.4f} vs published "
        f"{PUBLISHED['unrouted']:.4f}" if PUBLISHED['unrouted'] is not None
        else f"{u_base:.4f} vs published UNPARSEABLE (routing_eval.md missing or its hybrid section changed shape)"
        + ("  [smoke: subset]" if args.smoke else ""),
    ))

    # S4: the live-mechanism anchor. Run on a routed index, not the sweep's.
    anchor_combo = sorted(by_combo, key=lambda c: -len(by_combo[c]))[0]
    ok4, detail4 = verify_against_retriever(
        dir_of[anchor_combo], by_combo[anchor_combo][:5], [5, 50, 200, 1000])
    checks.append((
        f"S4 truncated fusion reproduces HybridRetriever(fetch_depth=F) on `{anchor_combo}`",
        ok4, detail4,
    ))

    hi = max(
        float(np.mean([arm[F]["recall@10"][q] for q in queries]))
        for arm in (routed, u_sc) for F in [*depths, -1]
    )
    checks.append((
        "S5 no arm exceeds the qrels ceiling (all send k=10)",
        hi <= QRELS_CEILING + 1e-9, f"max recall@10 {hi:.4f} vs ceiling {QRELS_CEILING}",
    ))

    # ---- pre-registered family: F=200 vs k=n, routed, 3 metrics ------------
    rng = np.random.default_rng(args.seed)
    F_reg = F_REGISTERED if F_REGISTERED in depths else depths[len(depths) // 2]
    pairs = []
    for m in _METRICS:
        diffs = np.array([routed[F_reg][m][q] - routed[-1][m][q] for q in queries])
        obs, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
        pairs.append((f"F={F_reg}", "k=n", obs, p, ci))
    family = holm_correct(pairs, args.alpha)

    payload = {
        "queries": len(queries),
        "depths": depths,
        "f_registered": F_reg,
        "n_boot": args.n_boot,
        "routes": {q: route_of[q] for q in queries},
        "etype": {q: etype[q] for q in queries},
        "combo_of": combo_of,
        "n_indices": len({combo_of[r] for r in route_of.values()}),
        "routed": {str(F): {m: routed[F][m] for m in _METRICS} for F in [*depths, -1]},
        "unrouted": {str(F): {m: u_sc[F][m] for m in _METRICS} for F in [*depths, -1]},
        "unrouted_combo": UNROUTED_COMBO,
        "same_order": {str(F): same_order[F] for F in depths},
        "same_set": {str(F): same_set[F] for F in depths},
        "u_same_order": {str(F): u_so[F] for F in depths},
        "family": [[a, b, d, p, list(ci), adj, sig] for a, b, d, p, ci, adj, sig in family],
        "checks": [[n, ok, d] for n, ok, d in checks],
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }

    if args.smoke:
        for name, ok, detail in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        for F in depths:
            r = float(np.mean([routed[F]["recall@10"][q] for q in queries]))
            print(f"  F={F:<6} routed {r:.4f} ({r-r_base:+.4f})  "
                  f"same-order {same_order[F]}/{len(queries)}")
        print(f"\nsmoke run ({time.time()-t0:.0f}s) -- nothing written")
        return 0 if all(ok for _, ok, _ in checks) else 1

    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return render(payload)


def render(d: dict) -> int:
    queries = list(d["routed"]["-1"]["recall@10"])
    depths = d["depths"]
    routed = {int(F): v for F, v in d["routed"].items()}
    unrouted = {int(F): v for F, v in d["unrouted"].items()}
    same_order = {int(F): v for F, v in d["same_order"].items()}
    same_set = {int(F): v for F, v in d["same_set"].items()}
    u_same_order = {int(F): v for F, v in d["u_same_order"].items()}
    nq = len(queries)

    def mean(arm, F, m):
        return float(np.mean([arm[F][m][q] for q in queries]))

    r_base = mean(routed, -1, "recall@10")
    u_base = mean(unrouted, -1, "recall@10")
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# ตัดความลึกการดึงของ hybrid — วัดทับ hard router ที่ ship อยู่")
    w()
    w("Generated by `tools/eval/routed_fetch_depth_test.py`.")
    w()
    w(f"{nq} คำถาม · {d['n_indices']} index ตามเส้นทางของ router · k = {K} "
      f"(ทุก arm **ส่ง 10 เอกสารเท่ากัน** ต่างกันแค่ *ดึง* มา fuse กี่ตัว) · "
      f"bootstrap {d['n_boot']:,} ครั้ง")
    w()
    w("**คำถามที่ตอบ** — `hybrid_fetch_depth_sweep.md` วัดว่า F=200 เสีย macro recall@10 ")
    w("ไป −0.0033 แลกกับ ~0.67 วิ/query **แต่วัดแบบไม่มี router และเฉลี่ยข้าม 36 combo** ")
    w("ส่วนสิ่งที่ ship จริงตั้งแต่ 8 ส.ค. คือ hard router · โปรเจกต์นี้เจอมาแล้วสองครั้งว่า ")
    w("ของที่ดูคุ้มเมื่อวัดกับ baseline ที่ไม่มี router กลายเป็นศูนย์เมื่อวัดกับตัวที่ ship ")
    w("(per-`entity_type` alpha และ reranker rrf4) — แต่**ทิศทางของ null ตรงข้ามกันที่นี่**: ")
    w("สองอันนั้นต้อง*ชนะ* จึงจะคุ้ม ส่วนการตัดความลึกแค่ต้อง*ไม่แพ้* ดังนั้น null คือผลที่ ")
    w("อนุญาตให้ ship ได้ และต้องรายงานเป็น**ขอบเขต** ไม่ใช่ \"ไม่ต่างกัน\"")
    w()

    w("## 1. ระบบที่ ship จริง (routed) — ตัดที่ความลึกไหนเสียอะไรบ้าง")
    w()
    w("| F | top-10 เหมือน k=n เป๊ะ | recall@10 | Δ | MRR | Δ | nDCG@10 | Δ |")
    w("|---|---|---|---|---|---|---|---|")
    for F in depths:
        r = mean(routed, F, "recall@10")
        mr = mean(routed, F, "mrr")
        nd = mean(routed, F, "ndcg@10")
        w(f"| {F:,} | {same_order[F]}/{nq} ({100*same_order[F]/nq:.1f}%) | "
          f"{r:.4f} | {r-r_base:+.4f} | {mr:.4f} | {mr-mean(routed,-1,'mrr'):+.4f} | "
          f"{nd:.4f} | {nd-mean(routed,-1,'ndcg@10'):+.4f} |")
    w(f"| n (ทั้งคลัง) | {nq}/{nq} (100.0%) | {r_base:.4f} | — | "
      f"{mean(routed,-1,'mrr'):.4f} | — | {mean(routed,-1,'ndcg@10'):.4f} | — |")
    w()

    w(f"## 2. ครอบครัวที่ลงทะเบียนไว้ล่วงหน้า — F={d['f_registered']} เทียบ k=n (Holm, m=3)")
    w()
    w("| metric | Δ | 95% CI | p | Holm-adj | นัยสำคัญ |")
    w("|---|---|---|---|---|---|")
    for (a, b, diff, p, ci, adj, sig), m in zip(d["family"], _METRICS):
        w(f"| {m} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | "
          f"{adj:.4f} | {'ใช่' if sig else 'ไม่'} |")
    w()
    worst = min(d["family"], key=lambda r: r[4][0])
    w(f"**อ่านเป็นขอบเขต**: ที่ F={d['f_registered']} CI ตัดความเป็นไปได้ที่จะเสียมากกว่า ")
    w(f"**{abs(worst[4][0]):.4f}** บน metric ที่แย่ที่สุดในสามตัวออกไป")
    w()

    w("## 3. เทียบกับตอนไม่มี router (combo เดียว, code path เดียวกัน)")
    w()
    w(f"`{d['unrouted_combo']}` — คือ baseline ที่ตัวเลข −0.0033 เดิมถูกวัดเทียบ")
    w()
    w("| F | routed Δ recall@10 | unrouted Δ recall@10 | routed เหมือนเป๊ะ | unrouted เหมือนเป๊ะ |")
    w("|---|---|---|---|---|")
    for F in depths:
        w(f"| {F:,} | {mean(routed,F,'recall@10')-r_base:+.4f} | "
          f"{mean(unrouted,F,'recall@10')-u_base:+.4f} | "
          f"{100*same_order[F]/nq:.1f}% | {100*u_same_order[F]/nq:.1f}% |")
    w(f"| n | — ({r_base:.4f}) | — ({u_base:.4f}) | 100.0% | 100.0% |")
    w()

    w("## 4. แยกตามเส้นทาง (คำทำนายที่ลงทะเบียนไว้ก่อนรัน)")
    w()
    w("การกวาดแบบไม่มี router พบว่า `person` เป็นชนิดเดียวที่ **ดีขึ้น** ตอนตัดตื้น ")
    w("(+0.0212 ที่ F=50) เพราะ BM25 แบกคำถามชนิดนี้อยู่แล้ว การตัดจึงไปลบหางของ dense ")
    w("arm ที่อ่อน · ถ้ากลไกนี้ถูก routing ซึ่งยกดัชนีเฉพาะทางให้ `person` อยู่แล้ว ")
    w("ต้องทำให้กำไรนั้น**หดลง**")
    w()
    routes = d["routes"]
    rlist = sorted(set(routes.values()))
    shown = [F for F in depths if F in (50, 200, 1000)]
    w("| route | n | routed k=n | " + " | ".join(f"Δ F={F}" for F in shown)
      + " | " + " | ".join(f"unrouted Δ F={F}" for F in shown) + " |")
    w("|---" * (2 + 2 * len(shown) + 1) + "|")
    for rt in rlist:
        qs = [q for q in queries if routes[q] == rt]
        rb = float(np.mean([routed[-1]["recall@10"][q] for q in qs]))
        ub = float(np.mean([unrouted[-1]["recall@10"][q] for q in qs]))
        rc = " | ".join(
            f"{float(np.mean([routed[F]['recall@10'][q] for q in qs]))-rb:+.4f}" for F in shown)
        uc = " | ".join(
            f"{float(np.mean([unrouted[F]['recall@10'][q] for q in qs]))-ub:+.4f}" for F in shown)
        w(f"| {rt} | {len(qs)} | {rb:.4f} | {rc} | {uc} |")
    w()

    if LAT.exists():
        lat = json.loads(LAT.read_text(encoding="utf-8"))
        s = lat["stats"]
        w("## 5. ประหยัดเวลาได้เท่าไรบนเส้นทางจริง")
        w()
        w(f"{lat['n_queries']} คำถาม แต่ละข้อจับเวลาบน index ที่ router ส่งไปจริง · "
          "สลับ arm ทีละคำถามในโปรเซสเดียว · warm ตัว BM25 scorer ไว้ก่อน")
        w()
        w("| arm | p50 | p95 | mean | ประหยัดจาก k=n (p50) |")
        w("|---|---|---|---|---|")
        for label in ("k=n", "F=1000", "F=200"):
            if label not in s:
                continue
            saved = ("—" if label == "k=n" else
                     f"−{(s['k=n']['p50'] - s[label]['p50'])/1000:.3f} วิ "
                     f"({s['k=n']['p50']/s[label]['p50']:.2f}x)")
            w(f"| {label} | {s[label]['p50']:.1f} ms | {s[label]['p95']:.1f} ms | "
              f"{s[label]['mean']:.1f} ms | {saved} |")
        w()
        w("| index ของเส้นทาง | chunk | คำถาม | k=n p50 | F=200 p50 |")
        w("|---|---|---|---|---|")
        for combo, pc in lat["per_combo"].items():
            w(f"| `{combo}` | {pc['n_chunks']:,} | {pc['n_queries']} | "
              f"{pc.get('k=n_p50', float('nan')):.1f} ms | "
              f"{pc.get('F=200_p50', float('nan')):.1f} ms |")
        w()
        w(f"ที่ F=200 top-10 ต่างจาก k=n **{lat['mismatched_200']} จาก {lat['n_queries']} คำถาม**")
        w()

    w("## self-check")
    w()
    for name, ok, detail in d["checks"]:
        w(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    w()

    for name, ok, detail in d["checks"]:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
    if not all(ok for _, ok, _ in d["checks"]):
        print("\nself-check failed; refusing to publish numbers", file=sys.stderr)
        return 1

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
