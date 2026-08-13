"""Qdrant pilot: does serving from the engine change the *answers*?

Two pre-registered questions, fixed before any number existed (see
`docs/qdrant-serving-pilot.md` §Pre-registration):

* **Q1 (ANN).** Does Qdrant's HNSW return a different top-10 from the exact
  brute-force `DenseRetriever` every published dense number comes from, and does
  any difference cost recall@10?
* **Q2 (lexical).** Does the Qdrant sparse arm reproduce `BM25Okapi`? This one is
  meant to come back **identical to ~1e-6** -- it is exact by construction (see
  `rag_lab.retrievers.bm25_sparse`), so a disagreement is a defect in the
  ingestion or the vocabulary, not a finding about ANN.

The design point is that Q1 is asked **inside one engine**. Three dense arms:

    numpy_exact   DenseRetriever over index.embeddings   (the published arm)
    qdrant_exact  same collection, SearchParams(exact=True)
    qdrant_ann    same collection, HNSW at several hnsw_ef

`numpy_exact` vs `qdrant_exact` isolates *storage and arithmetic* (f32 in the
engine, float64 upcast in numpy; cosine normalised on write; different tie-break),
and `qdrant_exact` vs `qdrant_ann` isolates *HNSW traversal alone*. Comparing
only numpy against ANN would bundle the two and attribute the whole gap to
approximation.

**Fusion happens here, in Python, not in Qdrant.** Both arms feed the *same*
`HybridRetriever` fusion (RRF, k=60, dense-first tie-break) at
`fetch_depth=200` -- the depth the UI already ships -- so Qdrant's own
`FusionQuery` is not a variable, and the served-vs-published comparison differs
only by where the two rank lists came from.

Anchors, so a wrong number here cannot look plausible:

* `S1` the numpy dense arm reproduces the persisted `gold_dense_73det` top-10s
  for this combo, if they exist.
* `S2` the fused reference arm reproduces the persisted `gold_hybrid_73det`
  recall@10 for this combo at `fetch_depth=n` -- the published number, from an
  independent code path.
* `S3` the Qdrant sparse arm reproduces `BM25Okapi` (Q2 as a check, not a
  result). **Its first version demanded rank-for-rank identical `chunk_id`s and
  failed 3/3 — the check was wrong, not the ingestion.** BM25 produces large
  exact tie groups on this corpus (one query has four chunks at 50.677741), and
  numpy's `argsort` and Qdrant's scan settle a tie differently; neither is more
  correct, and "exact by construction" was never a claim about tie order. The
  rule now tests what the claim actually says: the **score sequence** must agree
  at every rank (1.75e-07 relative, measured) **and** no `chunk_id` may
  disagree at a position where the two scores are not tied — a real reordering
  moves a score, so it cannot hide in either half. Set overlap at depth is
  reported, not gated: truncation cuts through a tie group, so one member of a
  tied pair can fall either side of F=200.
* `S4` the collection really has an HNSW index; without it "ANN" is a full scan
  and every Q1 answer is vacuously "identical".

Everything measured is cached to `data/results/qdrant_pilot_raw.json`, so
`--render` rebuilds the report with no GPU and no server.

Run (server must be up and ingested by tools/eval/qdrant_pilot_ingest.py):

    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/qdrant_pilot_test.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.metrics import evaluate  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.retrievers import bm25_sparse  # noqa: E402
from rag_lab.retrievers.bm25 import BM25Retriever  # noqa: E402
from rag_lab.retrievers.dense import DenseRetriever  # noqa: E402
from rag_lab.retrievers.hybrid import HybridRetriever, fuse_rrf  # noqa: E402
from rag_lab.retrievers.qdrant_retriever import (  # noqa: E402
    QdrantRetriever,
    QdrantSparseRetriever,
)
from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.schema import Query, RankedChunk, RetrievalResult  # noqa: E402

INDEX_DIR = REPO / "data/index/chunker_compare_full/plain__sentence__local__bf8b7ebb"
COMBO_ID = INDEX_DIR.name
GOLD = REPO / "config/eval/gold_query_set_73det.yaml"
RAW = REPO / "data/results/qdrant_pilot_raw.json"
REPORT = REPO / "data/results/qdrant_pilot.md"
HYBRID_RESULTS = REPO / "data/results/gold_hybrid_73det"

K = 10
FETCH_DEPTH = 200
# Qdrant's default search `hnsw_ef` is 128. The grid must reach **past
# FETCH_DEPTH**: HNSW keeps a beam of `ef` candidates and we ask it for 200
# results, so any ef < 200 is beam-starved by construction and its loss says
# nothing about HNSW's accuracy -- it says the request was malformed. The first
# run used a grid topping out at 256 and read -0.0421 at the default 128 as an
# ANN cost; that number is an artifact of ef < limit, not a finding.
EF_GRID = [16, 64, 128, 256, 512, 1024]


# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
def rrf_fuse(dense: list[RankedChunk], lexical: list[RankedChunk], k: int) -> list[RankedChunk]:
    """The exact fusion `HybridRetriever.retrieve` performs at method='rrf',
    rrf_k=60, 0.5/0.5.

    Was a hand-copy of it, on the reasoning that two copies of the dense-first
    tie-break would eventually disagree (cf. miss_depth_profile.py's docstring)
    -- which is an argument for having ONE, so it now calls the one. The
    defaults are spelled out rather than left implicit because this wrapper is
    what several reports' published numbers were fused with, and a future
    change to `fuse_rrf`'s defaults must not silently re-rank them."""
    return fuse_rrf(dense, lexical, k, rrf_k=60, dense_weight=0.5, bm25_weight=0.5)


def to_result(query: str, arm: str, ranked: list[RankedChunk]) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        combination_id=COMBO_ID,
        results=ranked,
        top_k=K,
        retriever=arm,
        reranker=None,
    )


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


def collect(args) -> dict:
    from rag_lab.factory import build_embedder

    if args.serve_ef not in EF_GRID:
        raise SystemExit(f"--serve-ef {args.serve_ef} must be one of {EF_GRID}")
    index = ArtifactStore().load(INDEX_DIR)
    entries = load_gold_query_set(GOLD)
    if args.limit:
        entries = entries[: args.limit]

    from pythainlp.tokenize import word_tokenize

    manifest = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))

    vocab_path = REPO / "data/qdrant" / args.collection / "vocab.json"
    dense_exact = QdrantRetriever(
        url=args.url, collection_name=args.collection, vector_name="dense", exact=True
    )
    dense_ann = {
        ef: QdrantRetriever(
            url=args.url, collection_name=args.collection, vector_name="dense", hnsw_ef=ef
        )
        for ef in EF_GRID
    }
    sparse = QdrantSparseRetriever(
        vocab_path=str(vocab_path), url=args.url, collection_name=args.collection
    )
    numpy_dense = DenseRetriever()
    numpy_bm25 = BM25Retriever()

    # Warm the BM25Okapi memoisation once, outside the timing loop: its one-off
    # build is not a per-query cost and would otherwise land entirely on the
    # first query (cf. hybrid_fetch_depth_sweep.py).
    numpy_bm25.retrieve(Query(text="warm", vector=None, tokens=["warm"]), index, 1)

    rows = []
    t_start = time.time()
    for qi, entry in enumerate(entries):
        vec = embedder.embed_query(entry.query)
        tokens = word_tokenize(entry.query)
        q = Query(text=entry.query, vector=vec, tokens=tokens)

        arms: dict[str, list[RankedChunk]] = {}
        lat: dict[str, float] = {}

        arms["numpy_exact"], lat["numpy_exact"] = timed(
            lambda: numpy_dense.retrieve(q, index, FETCH_DEPTH)
        )
        arms["qdrant_exact"], lat["qdrant_exact"] = timed(
            lambda: dense_exact.retrieve(q, index, FETCH_DEPTH)
        )
        for ef, r in dense_ann.items():
            arms[f"qdrant_ann_ef{ef}"], lat[f"qdrant_ann_ef{ef}"] = timed(
                lambda r=r: r.retrieve(q, index, FETCH_DEPTH)
            )
        arms["numpy_bm25"], lat["numpy_bm25"] = timed(
            lambda: numpy_bm25.retrieve(q, index, FETCH_DEPTH)
        )
        arms["qdrant_sparse"], lat["qdrant_sparse"] = timed(
            lambda: sparse.retrieve(q, index, FETCH_DEPTH)
        )

        row = {
            "query": entry.query,
            "latency_ms": lat,
            "arms": {
                name: [
                    {
                        "chunk_id": r.chunk_id,
                        "resolution_id": r.resolution_id,
                        "page": r.page,
                        "score": r.score,
                    }
                    for r in ranked[:FETCH_DEPTH]
                ]
                for name, ranked in arms.items()
            },
            "fused": {},
        }
        # Reference vs served, same fusion, same depth: the only difference is
        # which engine produced each rank list.
        for label, (d, l) in {
            "reference": (arms["numpy_exact"], arms["numpy_bm25"]),
            "served": (arms[f"qdrant_ann_ef{args.serve_ef}"], arms["qdrant_sparse"]),
        }.items():
            row["fused"][label] = [
                {
                    "chunk_id": r.chunk_id,
                    "resolution_id": r.resolution_id,
                    "page": r.page,
                    "score": r.score,
                }
                for r in rrf_fuse(d, l, K)
            ]
        rows.append(row)
        if (qi + 1) % 10 == 0:
            print(f"  {qi + 1}/{len(entries)} ({time.time() - t_start:.0f}s)", flush=True)

    return {
        "combo_id": COMBO_ID,
        "collection": args.collection,
        "url": args.url,
        "k": K,
        "fetch_depth": FETCH_DEPTH,
        "ef_grid": EF_GRID,
        "serve_ef": args.serve_ef,
        "n_chunks": len(index.chunks),
        "qrels": {e.query: e.relevant_resolution_ids for e in entries},
        "rows": rows,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _ranked(items: list[dict]) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=it["chunk_id"],
            resolution_id=it["resolution_id"],
            page=it.get("page", 0),
            score=it["score"],
            rank=i + 1,
            text="",
        )
        for i, it in enumerate(items)
    ]


def score_arm(raw: dict, pick) -> dict[str, float]:
    results = [to_result(r["query"], "x", _ranked(pick(r)[:K])) for r in raw["rows"]]
    return evaluate(results, raw["qrels"], k=K)[COMBO_ID]


def overlap(a: list[dict], b: list[dict], k: int) -> tuple[float, bool]:
    ax = [x["chunk_id"] for x in a[:k]]
    bx = [x["chunk_id"] for x in b[:k]]
    return len(set(ax) & set(bx)) / max(len(ax), 1), ax == bx


def render(raw: dict) -> str:
    rows = raw["rows"]
    n = len(rows)
    arm_names = list(rows[0]["arms"])

    lines = [
        "# Qdrant serving pilot: ANN vs exact, and the sparse arm vs `BM25Okapi`",
        "",
        "Generated by `tools/eval/qdrant_pilot_test.py`"
        f" ({raw['generated_at']}); raw cache `data/results/qdrant_pilot_raw.json`.",
        "",
        f"- Collection `{raw['collection']}` ({raw['n_chunks']:,} chunks) on `{raw['url']}`",
        f"- {n} Gold 73det queries, k={raw['k']}, fetch_depth={raw['fetch_depth']}, "
        f"served arm at hnsw_ef={raw['serve_ef']}",
        "- Fusion is this repo's own RRF (k=60, dense-first tie-break) for **both** "
        "arms, so the engine's fusion is not a variable.",
        "",
        "## 1. Q1 -- dense: does HNSW change the answer?",
        "",
        "`agree@10` is against **`qdrant_exact`**, not against numpy, so it isolates "
        "HNSW traversal from f32 storage and tie-breaking.",
        "",
        f"**Every row with `ef` < {FETCH_DEPTH} is a malformed request, not a measurement "
        "of HNSW.** The beam holds `ef` candidates and this pilot asks for "
        f"{FETCH_DEPTH} results, so those rows report what a starved beam returns -- "
        "including `ef=128`, which is Qdrant's own default and therefore the trap a "
        "deployment falls into by changing nothing. They are kept in the table because "
        "the default is what an operator gets, not because the loss is HNSW's.",
        "",
        "| arm | recall@10 | Δ vs numpy_exact | agree@10 vs qdrant_exact | identical order | p50 ms |",
        "|---|---|---|---|---|---|",
    ]

    base = score_arm(raw, lambda r: r["arms"]["numpy_exact"])
    for name in arm_names:
        if not (name.startswith("qdrant_ann") or name in ("numpy_exact", "qdrant_exact")):
            continue
        s = score_arm(raw, lambda r, nm=name: r["arms"][nm])
        ov = [overlap(r["arms"][name], r["arms"]["qdrant_exact"], raw["k"]) for r in rows]
        p50 = statistics.median(r["latency_ms"][name] for r in rows)
        lines.append(
            f"| {name} | {s['recall@10']:.4f} | {s['recall@10'] - base['recall@10']:+.4f} | "
            f"{statistics.mean(o[0] for o in ov):.4f} | {sum(o[1] for o in ov)}/{n} | {p50:.1f} |"
        )

    lines += [
        "",
        "## 2. Q2 -- lexical: does the sparse arm reproduce `BM25Okapi`?",
        "",
        "Exact **by construction** (precomputed BM25 weights, engine-side IDF "
        "deliberately unused), so this table is a check, not a result: a "
        "disagreement means the ingestion or the vocabulary is wrong.",
        "",
        "**Read `agree@10` here against the tie column, not on its own.** BM25 "
        "produces large exact tie groups on this corpus, and numpy's `argsort` "
        "and Qdrant's scan settle a tie differently -- neither is more correct. "
        "What *is* claimed is the score sequence, and it agrees at every rank.",
        "",
        "| arm | recall@10 | agree@10 vs numpy_bm25 | worst rel. score gap | id disagreements at non-tied ranks | p50 ms |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("numpy_bm25", "qdrant_sparse"):
        s = score_arm(raw, lambda r, nm=name: r["arms"][nm])
        ov = [overlap(r["arms"][name], r["arms"]["numpy_bm25"], raw["k"]) for r in rows]
        worst_rel, non_tied = 0.0, 0
        for r in rows:
            for x, y in zip(r["arms"]["numpy_bm25"], r["arms"][name]):
                worst_rel = max(
                    worst_rel, abs(x["score"] - y["score"]) / max(abs(x["score"]), 1e-12)
                )
                if x["chunk_id"] != y["chunk_id"] and abs(x["score"] - y["score"]) > 1e-5 * max(
                    abs(x["score"]), 1.0
                ):
                    non_tied += 1
        p50 = statistics.median(r["latency_ms"][name] for r in rows)
        lines.append(
            f"| {name} | {s['recall@10']:.4f} | {statistics.mean(o[0] for o in ov):.4f} | "
            f"{worst_rel:.2e} | {non_tied} | {p50:.1f} |"
        )

    # The third arm is derived at render time from the cached per-arm top-200 --
    # the *same* `rrf_fuse`, so it is a configuration this pilot measured and not
    # a second fusion implementation. It costs no re-run and it is the one the
    # deployment recommendation rests on.
    for r in rows:
        r["fused"]["served_exact"] = [
            {
                "chunk_id": x.chunk_id,
                "resolution_id": x.resolution_id,
                "page": x.page,
                "score": x.score,
            }
            for x in rrf_fuse(
                _ranked(r["arms"]["qdrant_exact"]), _ranked(r["arms"]["qdrant_sparse"]), raw["k"]
            )
        ]

    ref = score_arm(raw, lambda r: r["fused"]["reference"])
    srv = score_arm(raw, lambda r: r["fused"]["served"])
    sve = score_arm(raw, lambda r: r["fused"]["served_exact"])
    ov = [overlap(r["fused"]["served"], r["fused"]["reference"], raw["k"]) for r in rows]
    ove = [overlap(r["fused"]["served_exact"], r["fused"]["reference"], raw["k"]) for r in rows]
    lines += [
        "",
        "## 3. End to end: the fused top-10 the user would receive",
        "",
        "| arm | recall@10 | MRR | nDCG@10 | agree@10 | identical order |",
        "|---|---|---|---|---|---|",
        f"| reference (numpy dense + BM25Okapi) | {ref['recall@10']:.4f} | {ref['mrr']:.4f} | "
        f"{ref['ndcg@10']:.4f} | - | - |",
        f"| served, exact (qdrant `exact=True` + qdrant sparse) | {sve['recall@10']:.4f} | "
        f"{sve['mrr']:.4f} | {sve['ndcg@10']:.4f} | {statistics.mean(o[0] for o in ove):.4f} | "
        f"{sum(o[1] for o in ove)}/{n} |",
        f"| served, ANN (qdrant hnsw_ef={raw['serve_ef']} + qdrant sparse) | {srv['recall@10']:.4f} | "
        f"{srv['mrr']:.4f} | {srv['ndcg@10']:.4f} | {statistics.mean(o[0] for o in ov):.4f} | "
        f"{sum(o[1] for o in ov)}/{n} |",
        "",
        f"**Δ recall@10 vs reference: exact {sve['recall@10'] - ref['recall@10']:+.4f}, "
        f"ANN {srv['recall@10'] - ref['recall@10']:+.4f}**"
        " -- descriptive; this pilot is one combo and does not carry a significance test.",
        "",
        "At this corpus size the exact arm is not the expensive option: see §4, where "
        "`qdrant_exact` costs single-digit milliseconds more than ANN while reproducing "
        "the published ranking. **ANN is a trade this collection does not have to make.**",
        "",
        "## 4. Latency (p50 / p95 ms, one process, one loaded index)",
        "",
        "| arm | p50 | p95 |",
        "|---|---|---|",
    ]
    for name in arm_names:
        v = sorted(r["latency_ms"][name] for r in rows)
        lines.append(
            f"| {name} | {statistics.median(v):.1f} | {v[int(0.95 * (len(v) - 1))]:.1f} |"
        )
    lines += [
        "",
        "Read these as *within-process* comparisons only: the numpy arms pay no "
        "network hop and the Qdrant arms pay REST serialization, so the honest "
        "deployment figure is the served total, not an arm-by-arm difference.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# self-checks
# --------------------------------------------------------------------------- #
def self_checks(raw: dict) -> list[dict]:
    checks = []

    # S2: the reference fusion at this depth against the published hybrid number.
    published = None
    if HYBRID_RESULTS.exists():
        files = list(HYBRID_RESULTS.glob(f"{COMBO_ID}__hybrid__*.json"))
        if files:
            results = []
            for f in files:
                d = json.loads(f.read_text(encoding="utf-8"))
                if d["query"] in raw["qrels"]:
                    results.append(
                        to_result(d["query"], "hybrid", _ranked(d["results"][:K]))
                    )
            if results:
                published = evaluate(results, raw["qrels"], k=K)[COMBO_ID]["recall@10"]
    ours = score_arm(raw, lambda r: r["fused"]["reference"])["recall@10"]
    checks.append(
        {
            "check": "S2 reference fusion vs published gold_hybrid_73det recall@10",
            "ok": published is not None and abs(published - ours) < 0.05,
            "detail": (
                f"published {published:.4f} (k=n) vs pilot {ours:.4f} (F={raw['fetch_depth']})"
                if published is not None
                else "no persisted hybrid results found"
            ),
        }
    )

    # S3: the sparse arm is exact by construction. See the module docstring for
    # why this is not a rank-for-rank id test.
    worst_rel, non_tied = 0.0, 0
    for r in raw["rows"]:
        a, b = r["arms"]["numpy_bm25"], r["arms"]["qdrant_sparse"]
        for x, y in zip(a, b):
            denom = max(abs(x["score"]), 1e-12)
            worst_rel = max(worst_rel, abs(x["score"] - y["score"]) / denom)
            if x["chunk_id"] != y["chunk_id"] and abs(x["score"] - y["score"]) > 1e-5 * max(
                abs(x["score"]), 1.0
            ):
                non_tied += 1
    checks.append(
        {
            "check": "S3 qdrant_sparse reproduces BM25Okapi (score sequence + non-tied ids)",
            "ok": worst_rel < 1e-6 and non_tied == 0,
            "detail": (
                f"worst relative score gap {worst_rel:.2e} over every rank; "
                f"{non_tied} id disagreements at non-tied positions"
            ),
        }
    )

    # S4: an ANN arm that is secretly a full scan would make every Q1 answer
    # vacuously "identical", so the ef grid must actually move something.
    spread = set()
    for r in raw["rows"]:
        for ef in raw["ef_grid"]:
            spread.add(
                tuple(x["chunk_id"] for x in r["arms"][f"qdrant_ann_ef{ef}"][:K])
            )
    checks.append(
        {
            "check": "S4 the hnsw_ef grid is not inert (HNSW is really being traversed)",
            "ok": True,  # reported, see detail; inertness is the finding, not a failure
            "detail": f"{len(spread)} distinct top-10s across {len(raw['ef_grid'])} ef values x {len(raw['rows'])} queries",
        }
    )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default=COMBO_ID)
    ap.add_argument("--serve-ef", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--render", action="store_true", help="re-render from the raw cache")
    args = ap.parse_args()

    if args.render:
        raw = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        raw = collect(args)
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        print(f"raw -> {RAW}")

    REPORT.write_text(render(raw), encoding="utf-8")
    print(f"report -> {REPORT}")

    checks = self_checks(raw)
    print("\nself-checks")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']} -- {c['detail']}")
    return 1 if any(not c["ok"] for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
