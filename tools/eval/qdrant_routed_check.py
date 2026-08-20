"""Do the four ingested Qdrant collections serve the SHIPPED router correctly?

This is a **completion check on the ingestion**, not a new experiment. Nothing
here is pre-registered and no verdict is claimed: `qdrant_pilot.py` already
answered the two questions (serve dense with `exact=True`; the sparse arm is
exact by construction) on **one** collection, and its stated gap was
"one collection / one combo / one route". The other three were ingested on
2026-08-13; this script asks whether the whole routed stack, served end to end
out of the engine, still returns the published answer.

The claim under test is therefore narrow and falsifiable:

    routing the 106 Gold queries through `classify_query`, retrieving dense
    (`exact=True`) + sparse from each route's own collection at
    `fetch_depth=200`, and fusing with this repo's own RRF, reproduces
    `routed_fetch_depth_test.md`'s **0.6835** macro recall@10.

Two arms, differing only in *where each rank list came from*:

    reference   DenseRetriever + BM25Retriever over the on-disk Index
    served      QdrantRetriever(exact=True) + QdrantSparseRetriever

**The anchor is per-query, not the macro.** `routed_fetch_depth_raw.json` holds
all 106 per-query recall@10 values at F=200, so C2 gates the reference arm
against every one of them rather than against their mean -- a macro can agree
while individual queries move in compensating directions. C2 is what makes C3
(served vs reference) mean anything: without it, a served arm agreeing with a
*wrong* reference would read as a pass.

C8 is the one check here that is about *code paths* rather than about the
ingestion. Everything else assembles the served arm by hand -- it names the
collection, builds both Qdrant retrievers and calls the fusion itself -- which
proves the collections are right and says nothing about whether an application
reaching for `route_query` gets the same thing. C8 runs the shipped path
(`route_query` + a `qdrant_hybrid` StrategySpec) and requires it to return this
script's own served top-10, id for id.

C6 records the mechanism behind the pilot's one unexplained observation. It
reported `indexed_vectors_count` at ~1.93x `points_count` and called it
unexplained; telemetry gives `num_vectors == 2 x num_points` on every
collection, so the counter sums **dense + sparse** (each point carries both) and
the shortfall below 2N is dense vectors sitting in a segment still under
`indexing_threshold`. So the bound is `N <= indexed <= 2N`, and the residue is
reported as a fraction rather than asserted away -- it is real: those rows are
plain-scanned rather than HNSW-traversed, which is immaterial only because the
recommendation is `exact=True`.

Everything measured is cached to `data/results/qdrant_routed_check_raw.json`, so
`--render` rebuilds the report with no GPU and no server.

Run (server up, all four collections ingested):

    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/qdrant_routed_check.py --smoke
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/qdrant_routed_check.py
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/qdrant_routed_check.py --render
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from pythainlp.tokenize import word_tokenize  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

# The fusion is IMPORTED, never reimplemented: two copies of the dense-first
# tie-break would eventually disagree (the rule qdrant_pilot_test.py,
# routed_fetch_depth_test.py and miss_depth_profile.py all follow).
from qdrant_pilot_test import rrf_fuse  # noqa: E402
from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder  # noqa: E402
from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.metrics import recall_at_k  # noqa: E402
from rag_lab.query_service import (  # noqa: E402
    discover_indices,
    resolve_index,
    route_query,
)
from rag_lab.retrievers.bm25 import BM25Retriever  # noqa: E402
from rag_lab.retrievers.dense import DenseRetriever  # noqa: E402
from rag_lab.retrievers.qdrant_retriever import (  # noqa: E402
    QdrantRetriever,
    QdrantSparseRetriever,
)
from rag_lab.router import classify_query, route_targets  # noqa: E402
from rag_lab.schema import Query, RankedChunk, RetrievalResult  # noqa: E402

INDEX_ROOT = REPO / "data" / "index" / "chunker_compare_full"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
QDRANT_DIR = REPO / "data" / "qdrant"
RAW = REPO / "data" / "results" / "qdrant_routed_check_raw.json"
REPORT = REPO / "data" / "results" / "qdrant_routed_check.md"
# routed_fetch_depth_test.py's cache: 106 per-query recall@10 at F=200.
ROUTED_RAW = REPO / "data" / "results" / "routed_fetch_depth_raw.json"

K = 10
FETCH_DEPTH = 200
# Parsed from routed_fetch_depth_test.md's §1 table rather than frozen here.
# It was the literal 0.6835 until 2026-08-18, when the rebuild-#4 refresh moved
# that report to 0.6815 and this file kept printing a figure its own source no
# longer contained -- the check itself was never wrong (C2 compares per-query
# values from routed_fetch_depth_raw.json), but the number it *reported* was.
# A cross-artifact figure has to be read from the artifact, every run.
_ROUTED_FETCH_REPORT = REPO / "data" / "results" / "routed_fetch_depth_test.md"


def published_f200() -> float | None:
    """Routed macro recall@10 at F=200, from routed_fetch_depth_test.md's §1 row."""
    if not _ROUTED_FETCH_REPORT.exists():
        return None
    return parse_f200(_ROUTED_FETCH_REPORT.read_text(encoding="utf-8"))


def parse_f200(text: str) -> float | None:
    """Same, from the report's text -- split out so a test can pin it.

    Keyed on the depth cell being exactly "200": the table carries a row per
    fetch depth, so a substring match would take whichever depth happens to be
    first. None means the row is gone, which the caller must surface rather
    than quietly anchoring on a stale literal.
    """
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0] == "200":
            try:
                return float(cells[2])
            except ValueError:
                return None
    return None


PUBLISHED_F200 = published_f200()
# The pilot measured served-vs-reference at +0.0017 end to end on one
# collection, attributed to tie-break convention (two exact engines settle an
# exact tie differently and neither is more correct). This is a tolerance on a
# *known* effect, chosen before the run and stated on the row.
SERVED_TOL = 0.0100
SPARSE_REL_TOL = 1e-5
# f32 storage on the server against float64 numpy arithmetic; ~1e-8 in practice.
DENSE_REL_TOL = 1e-5


def as_result(query: str, ranked: list[RankedChunk], label: str) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        combination_id=label,
        top_k=len(ranked),
        retriever=label,
        results=ranked,
    )


def server_stats(client: QdrantClient, name: str) -> dict:
    info = client.get_collection(name)
    return {
        "status": str(info.status),
        "points": int(info.points_count or 0),
        "indexed_vectors": int(info.indexed_vectors_count or 0),
        "segments": int(info.segments_count or 0),
    }


def run_one_combo(combo: str, index_dir: Path, qs: list[str], url: str) -> dict:
    """Both arms for the queries routed to one index.

    One index + one embedder resident at a time and released before the next:
    four embedding matrices plus their models is what this GPU cannot hold
    (the same constraint routed_fetch_depth_test.py and
    reranker_rrf_routed_test.py are written around).
    """
    index = ArtifactStore().load(index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))

    vocab_path = QDRANT_DIR / combo / "vocab.json"
    qd_dense = QdrantRetriever(
        url=url, collection_name=combo, vector_name="dense", exact=True
    )
    qd_sparse = QdrantSparseRetriever(
        vocab_path=str(vocab_path), url=url, collection_name=combo
    )
    np_dense = DenseRetriever()
    np_bm25 = BM25Retriever()
    # Warm the BM25Okapi memoisation outside the loop; its one-off build is not
    # a per-query cost and would otherwise land entirely on the first query.
    np_bm25.retrieve(Query(text="warm", vector=None, tokens=["warm"]), index, 1)

    rows = []
    for q in qs:
        vec = embedder.embed_query(q)
        tokens = word_tokenize(q)
        query = Query(text=q, vector=vec, tokens=tokens)

        d_np = np_dense.retrieve(query, index, FETCH_DEPTH)
        d_qd = qd_dense.retrieve(query, index, FETCH_DEPTH)
        l_np = np_bm25.retrieve(query, index, FETCH_DEPTH)
        l_qd = qd_sparse.retrieve(query, index, FETCH_DEPTH)

        rows.append({
            "query": q,
            "combo": combo,
            "reference": [
                {"chunk_id": r.chunk_id, "resolution_id": r.resolution_id}
                for r in rrf_fuse(d_np, l_np, K)
            ],
            "served": [
                {"chunk_id": r.chunk_id, "resolution_id": r.resolution_id}
                for r in rrf_fuse(d_qd, l_qd, K)
            ],
            # Score sequences, not ids: exactness is a claim about SCORES. Two
            # exact engines settle an exact tie differently and neither is more
            # correct (the S3 correction in qdrant_pilot_test.py). Kept to full
            # depth because a top-10 id disagreement has to be looked up in the
            # other arm's scores to show it lies inside a tie group.
            "dense_np": [{"chunk_id": r.chunk_id, "score": r.score} for r in d_np],
            "dense_qd": [{"chunk_id": r.chunk_id, "score": r.score} for r in d_qd],
            "sparse_np_scores": [r.score for r in l_np],
            "sparse_qd_scores": [r.score for r in l_qd],
        })

    n_rows = len(index.chunks)
    embedder.release()
    del index, np_bm25
    return {"combo": combo, "rows": rows, "n_rows": n_rows}


def run_wiring_check(
    queries: list[str],
    route_of: dict[str, str],
    indices,
    url: str,
    per_route: int = 2,
) -> dict:
    """C8's arm: the same answer, fetched through the SHIPPED serving path.

    `run_one_combo` above names the collection, builds both Qdrant retrievers
    and calls the fusion itself. That is the right shape for checking an
    *ingestion* and the wrong shape for checking a *deployment*: the route ->
    collection resolution, the skipped `embeddings.npy`, the lazily-derived
    collection name and the retriever's own fetch depth are all code it
    bypasses. Here nothing is assembled -- one `StrategySpec` goes into
    `route_query` and whatever comes back is the answer an application gets.

    Deliberately a SUBSET (2 queries per route, 8 of 106). `build_embedder`
    has no cache and `query_indices` never releases, so a 106-query loop loads
    an embedder 106 times and one of the routed embedders is the 4B qwen3 on a
    12 GB card. That is affordable here because the claim being anchored is an
    *identity between two code paths*, which a subset settles; the scores are
    C2's and C3's job.
    """
    # The shipped path resolves its own targets from the spec's retriever type,
    # and `qdrant_hybrid` is not in ROUTE_COMBO_BY_RETRIEVER -- it falls back to
    # ROUTE_COMBO (the hybrid map). Assert that rather than assume it: if the
    # two maps ever diverge, C8 would be comparing two different collections
    # and could still pass.
    if route_targets("qdrant_hybrid") != route_targets("hybrid"):
        raise AssertionError(
            "route_targets('qdrant_hybrid') no longer resolves to the hybrid map, "
            "so the shipped path and this script's served arm would be reading "
            "different collections. Add an explicit entry to "
            "ROUTE_COMBO_BY_RETRIEVER and re-point this check."
        )

    by_route: dict[str, list[str]] = collections.defaultdict(list)
    for q in queries:
        by_route[route_of[q]].append(q)
    chosen = [q for r in sorted(by_route) for q in by_route[r][:per_route]]

    spec = StrategySpec(
        type="qdrant_hybrid",
        params={"url": url, "fetch_depth": FETCH_DEPTH, "exact": True},
    )
    out = {}
    for i, q in enumerate(chosen, 1):
        print(f"    wiring [{i}/{len(chosen)}] {route_of[q]}", flush=True)
        res = route_query(q, indices, spec, K)
        out[q] = {
            "route": route_of[q],
            "combination_id": res.combination_id,
            "chunk_ids": [r.chunk_id for r in res.results],
        }
    return out


def collect(args) -> dict:
    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: list(d["relevant_resolution_ids"]) for d in raw}
    etype = {d["query"]: d.get("entity_type", "?") for d in raw}

    indices = discover_indices(INDEX_ROOT)
    targets = route_targets("hybrid")
    route_of = {q: classify_query(q) for q in queries}
    resolved = {r: resolve_index(t, indices) for r, t in targets.items()}
    combo_of = {r: i.combo_id for r, i in resolved.items()}
    dir_of = {i.combo_id: Path(i.dir) for i in resolved.values()}

    if args.smoke:
        # Two queries per route, so every collection is exercised.
        by_route = collections.defaultdict(list)
        for q in queries:
            by_route[route_of[q]].append(q)
        queries = [q for r in targets for q in by_route[r][:2]]

    by_combo: dict[str, list[str]] = collections.defaultdict(list)
    for q in queries:
        by_combo[combo_of[route_of[q]]].append(q)

    client = QdrantClient(url=args.url)
    live = {c.name for c in client.get_collections().collections}
    stats = {}
    for combo in by_combo:
        stats[combo] = server_stats(client, combo) if combo in live else None

    t0 = time.time()
    per_combo = {}
    for i, (combo, qs) in enumerate(by_combo.items(), 1):
        print(f"[{i}/{len(by_combo)}] {combo}: {len(qs)} queries", flush=True)
        per_combo[combo] = run_one_combo(combo, dir_of[combo], qs, args.url)
        print(f"    done ({time.time() - t0:.0f}s)", flush=True)

    n_rows = {combo: pc["n_rows"] for combo, pc in per_combo.items()}

    # After the loop above, never inside it: `route_query` builds its own
    # embedder and this box holds one at a time.
    print("wiring check (shipped route_query path)", flush=True)
    wiring = run_wiring_check(queries, route_of, indices, args.url)

    return {
        "queries": queries,
        "qrels": qrels,
        "etype": etype,
        "route_of": route_of,
        "combo_of": combo_of,
        "n_rows": n_rows,
        "server": stats,
        "live_collections": sorted(live),
        "k": K,
        "fetch_depth": FETCH_DEPTH,
        "url": args.url,
        "smoke": bool(args.smoke),
        "per_combo": per_combo,
        "wiring": wiring,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _ranked(items: list[dict]) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=it["chunk_id"],
            resolution_id=it["resolution_id"],
            page=0,
            score=0.0,
            rank=i + 1,
            text="",
        )
        for i, it in enumerate(items)
    ]


def per_query_recall(raw: dict, arm: str) -> dict[str, float]:
    out = {}
    for pc in raw["per_combo"].values():
        for row in pc["rows"]:
            res = as_result(row["query"], _ranked(row[arm]), arm)
            out[row["query"]] = recall_at_k(res, raw["qrels"][row["query"]], K)
    return out


def _worst_rel(pairs) -> tuple[float, int]:
    worst, n = 0.0, 0
    for x, y in pairs:
        n += 1
        worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-12))
    return worst, n


def sparse_agreement(rows: list[dict]) -> tuple[float, int]:
    """Worst relative score disagreement at any rank, and how many ranks compared."""
    return _worst_rel(
        (x, y)
        for row in rows
        for x, y in zip(row["sparse_np_scores"], row["sparse_qd_scores"])
    )


def dense_agreement(rows: list[dict]) -> dict:
    """Dense exactness, as a claim about the SCORE SEQUENCE, plus what the id
    disagreements are made of.

    The first smoke run failed a set-identity check on `recursive x qwen3` and
    the ingestion was fine: this corpus repeats a course table verbatim across
    curriculum revisions, those chunks embed identically, and one query's entire
    top-12 sat at a single score (0.626616248932). Inside a tie group of that
    size the *set* returned is not defined by either engine, so a set test
    asserts an order nobody promised. What is testable: the score at each rank
    agrees, and every id that moved carries the tied score.
    """
    worst, n_ranks = _worst_rel(
        (a["score"], b["score"])
        for row in rows
        for a, b in zip(row["dense_np"], row["dense_qd"])
    )
    in_tie = out_tie = unresolved = moved = 0
    agree, identical, biggest_tie = [], 0, 1
    for row in rows:
        np_score = {r["chunk_id"]: r["score"] for r in row["dense_np"]}
        a = [r["chunk_id"] for r in row["dense_np"][:K]]
        b = [r["chunk_id"] for r in row["dense_qd"][:K]]
        agree.append(len(set(a) & set(b)) / max(len(a), 1))
        identical += int(a == b)
        counts = collections.Counter(round(r["score"], 9) for r in row["dense_np"][:50])
        biggest_tie = max(biggest_tie, max(counts.values(), default=1))
        for i, (x, y) in enumerate(zip(a, b)):
            if x == y:
                continue
            moved += 1
            got = np_score.get(y)
            if got is None:
                unresolved += 1
            elif abs(got - np_score[x]) <= 1e-6 * max(abs(got), 1e-12):
                in_tie += 1
            else:
                out_tie += 1
    return {
        "worst_rel": worst,
        "n_ranks": n_ranks,
        "agree_at_k": statistics.mean(agree),
        "identical_order": identical,
        "moved": moved,
        "moved_in_tie": in_tie,
        "moved_out_of_tie": out_tie,
        "moved_unresolved": unresolved,
        "biggest_tie_group": biggest_tie,
    }


def analyse(raw: dict) -> dict:
    ref = per_query_recall(raw, "reference")
    srv = per_query_recall(raw, "served")

    published = None
    if ROUTED_RAW.exists():
        pub_raw = json.loads(ROUTED_RAW.read_text(encoding="utf-8"))
        published = pub_raw["routed"][str(FETCH_DEPTH)]["recall@10"]

    per_combo = {}
    for combo, pc in raw["per_combo"].items():
        rows = pc["rows"]
        dense = dense_agreement(rows)
        worst, n_ranks = sparse_agreement(rows)
        per_combo[combo] = {
            "n_queries": len(rows),
            "dense": dense,
            "dense_agree_at_k": dense["agree_at_k"],
            "dense_identical_order": dense["identical_order"],
            "sparse_worst_rel": worst,
            "sparse_ranks": n_ranks,
            "ref_recall": statistics.mean(ref[r["query"]] for r in rows),
            "srv_recall": statistics.mean(srv[r["query"]] for r in rows),
        }

    per_route = {}
    for q in raw["queries"]:
        per_route.setdefault(raw["route_of"][q], []).append(q)
    routes = {
        r: {
            "n": len(qs),
            "ref": statistics.mean(ref[q] for q in qs),
            "srv": statistics.mean(srv[q] for q in qs),
        }
        for r, qs in per_route.items()
    }

    return {
        "ref_macro": statistics.mean(ref.values()),
        "srv_macro": statistics.mean(srv.values()),
        "ref": ref,
        "srv": srv,
        "published": published,
        "per_combo": per_combo,
        "routes": routes,
    }


def checks(raw: dict, an: dict) -> list[tuple[str, bool, str]]:
    out: list[tuple[str, bool, str]] = []

    missing = [c for c, s in raw["server"].items() if s is None]
    out.append((
        "C1 every routed query's collection exists on the server, with "
        "points_count == index rows",
        not missing
        and all(
            raw["server"][c]["points"] == raw["n_rows"][c] for c in raw["server"]
        ),
        f"{len(raw['server'])} collections routed to, "
        + ", ".join(
            f"{c}: {raw['server'][c]['points']:,}/{raw['n_rows'][c]:,}"
            for c in sorted(raw["server"])
            if raw["server"][c]
        )
        + (f"; MISSING {missing}" if missing else ""),
    ))

    pub = an["published"]
    if pub is None:
        out.append(("C2 reference arm vs published per-query F=200 recall@10",
                    False, "routed_fetch_depth_raw.json not found"))
    elif raw["smoke"]:
        differ = [q for q in an["ref"] if q in pub and abs(an["ref"][q] - pub[q]) > 1e-9]
        out.append((
            "C2 reference arm reproduces the published per-query F=200 recall@10",
            not differ,
            f"{len(an['ref']) - len(differ)}/{len(an['ref'])} queries identical "
            f"(smoke slice)",
        ))
    else:
        shared = [q for q in an["ref"] if q in pub]
        differ = [q for q in shared if abs(an["ref"][q] - pub[q]) > 1e-9]
        out.append((
            "C2 reference arm reproduces the published per-query F=200 recall@10 "
            "(routed_fetch_depth_raw.json), all 106",
            len(shared) == len(an["ref"]) and not differ,
            f"{len(shared) - len(differ)}/{len(shared)} queries identical; "
            f"macro {an['ref_macro']:.4f} vs published "
            + (f"{PUBLISHED_F200:.4f}" if PUBLISHED_F200 is not None else "UNPARSEABLE")
            + (f"; {len(differ)} differ" if differ else ""),
        ))

    d = an["srv_macro"] - an["ref_macro"]
    out.append((
        f"C3 served (Qdrant exact + sparse) reproduces the reference within "
        f"{SERVED_TOL:.4f}",
        abs(d) <= SERVED_TOL,
        f"served {an['srv_macro']:.4f} vs reference {an['ref_macro']:.4f} "
        f"({d:+.4f}); tolerance is the pilot's measured tie-break gap (+0.0017)",
    ))

    worst_dense = max(v["dense"]["worst_rel"] for v in an["per_combo"].values())
    out.append((
        f"C4 served dense (`exact=True`) score sequence agrees with numpy dense at "
        f"every rank (< {DENSE_REL_TOL:g} relative, f32 storage)",
        worst_dense < DENSE_REL_TOL,
        "; ".join(
            f"{c}: {v['dense']['worst_rel']:.2e} over {v['dense']['n_ranks']:,} ranks"
            for c, v in sorted(an["per_combo"].items())
        ),
    ))

    out_of_tie = sum(v["dense"]["moved_out_of_tie"] for v in an["per_combo"].values())
    unresolved = sum(v["dense"]["moved_unresolved"] for v in an["per_combo"].values())
    moved = sum(v["dense"]["moved"] for v in an["per_combo"].values())
    out.append((
        "C4b every top-10 id that differs between the two exact engines carries an "
        "identical reference score, i.e. sits inside a tie group",
        out_of_tie == 0 and unresolved == 0,
        f"{moved} of {len(raw['queries']) * K} top-10 positions moved; "
        f"{moved - out_of_tie - unresolved} inside a tie group, {out_of_tie} outside, "
        f"{unresolved} not found in the reference's top-{raw['fetch_depth']}; "
        "largest tie group "
        + ", ".join(
            f"{c.split('__')[1]} {v['dense']['biggest_tie_group']}"
            for c, v in sorted(an["per_combo"].items())
        ),
    ))

    worst_sparse = max(v["sparse_worst_rel"] for v in an["per_combo"].values())
    out.append((
        f"C5 sparse score sequence agrees with BM25Okapi at every rank "
        f"(< {SPARSE_REL_TOL:g} relative)",
        worst_sparse < SPARSE_REL_TOL,
        "; ".join(
            f"{c}: {v['sparse_worst_rel']:.2e} over {v['sparse_ranks']:,} ranks"
            for c, v in sorted(an["per_combo"].items())
        ),
    ))

    ok_bound = all(
        s["points"] <= s["indexed_vectors"] <= 2 * s["points"]
        for s in raw["server"].values()
        if s
    )
    out.append((
        "C6 indexed_vectors_count is within [N, 2N] -- it counts dense AND "
        "sparse, one of each per point",
        ok_bound,
        "; ".join(
            f"{c}: {s['indexed_vectors']:,} = 2x{s['points']:,} - "
            f"{2 * s['points'] - s['indexed_vectors']:,} dense unindexed "
            f"({(2 * s['points'] - s['indexed_vectors']) / max(s['points'], 1):.2%})"
            for c, s in sorted(raw["server"].items())
            if s
        ),
    ))

    # C8 is about code paths, not about the engine. Every check above builds
    # the served arm by hand; this one asks whether an application calling
    # `route_query` with a `qdrant_hybrid` spec gets that same answer, which is
    # the only form in which the wiring is testable at all.
    wiring = raw.get("wiring") or {}
    served_by_q = {
        row["query"]: [it["chunk_id"] for it in row["served"]]
        for pc in raw["per_combo"].values()
        for row in pc["rows"]
    }
    compared = [q for q in wiring if q in served_by_q]
    differ = [q for q in compared if wiring[q]["chunk_ids"] != served_by_q[q]]
    wrong_combo = [
        q for q in compared
        if not wiring[q]["combination_id"].startswith(raw["combo_of"][raw["route_of"][q]])
    ]
    out.append((
        "C8 the SHIPPED path (`route_query` + a `qdrant_hybrid` StrategySpec) returns "
        "this script's own served top-10, id for id",
        # `bool(compared)` is the denominator, not decoration: an empty wiring
        # run would make both `not differ` and `not wrong_combo` vacuously true.
        # Re-rendering a cache written before this check existed must FAIL, and
        # the fix for that is a re-run, never a loosened check (the E3 rule:
        # 0 is ambiguous between examined-and-clean and nothing-to-examine).
        bool(compared) and not differ and not wrong_combo,
        f"{len(compared) - len(differ)}/{len(compared)} queries identical; "
        f"{len(compared) - len(wrong_combo)}/{len(compared)} resolved the same "
        f"collection as the hand-assembled arm"
        + ("" if compared else "; NO wiring run in the cache (re-run, do not --render)"),
    ))
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def render(raw: dict) -> str:
    an = analyse(raw)
    ck = checks(raw, an)
    n = len(raw["queries"])

    L = [
        "# Qdrant: do all four routed collections serve the shipped router?",
        "",
        f"Generated by `tools/eval/qdrant_routed_check.py` ({raw['generated_at']}); "
        f"raw cache `data/results/qdrant_routed_check_raw.json`.",
        "",
        f"- {n} Gold 73det queries routed by `classify_query`, k={raw['k']}, "
        f"fetch_depth={raw['fetch_depth']}, dense served with `exact=True`",
        f"- {len(raw['per_combo'])} collections on `{raw['url']}` "
        f"(the shipped hybrid router's 5 routes resolve to 4 distinct indices: "
        "`faculty` and `unmatched` share one)",
        "- Fusion is this repo's own RRF (k=60, dense-first tie-break), **imported** "
        "from `qdrant_pilot_test.py`, for both arms -- so the engine's own fusion "
        "is not a variable",
        "",
        "**This is a completion check on the ingestion, not an experiment.** No "
        "pre-registration, no significance test, no verdict: the question is "
        "whether the served stack returns the published answer.",
        "",
        "## 1. The deliverable",
        "",
        "| arm | macro recall@10 | vs published |",
        "|---|---|---|",
    ]
    L.append(
        f"| published (`routed_fetch_depth_test.md`, F=200) | "
        + (f"{PUBLISHED_F200:.4f}" if PUBLISHED_F200 is not None else "n/a") + " | -- |"
    )
    L.append(
        f"| reference (numpy dense + `BM25Okapi`, this code path) | "
        f"{an['ref_macro']:.4f} | {an['ref_macro'] - PUBLISHED_F200:+.4f} |"
    )
    L.append(
        f"| **served (Qdrant `exact=True` + sparse)** | **{an['srv_macro']:.4f}** | "
        f"{an['srv_macro'] - PUBLISHED_F200:+.4f} |"
    )
    L += [
        "",
        "## 2. Per collection",
        "",
        "| collection | queries | rows | points | indexed vectors | dense worst rel. "
        "err | sparse worst rel. err | dense agree@10 | largest tie group | ref | "
        "served |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c, v in sorted(an["per_combo"].items()):
        s = raw["server"].get(c) or {}
        L.append(
            f"| `{c}` | {v['n_queries']} | {raw['n_rows'][c]:,} | "
            f"{s.get('points', 0):,} | {s.get('indexed_vectors', 0):,} | "
            f"{v['dense']['worst_rel']:.2e} | {v['sparse_worst_rel']:.2e} | "
            f"{v['dense_agree_at_k']:.4f} | {v['dense']['biggest_tie_group']} | "
            f"{v['ref_recall']:.4f} | {v['srv_recall']:.4f} |"
        )
    L += [
        "",
        "**`dense agree@10` is descriptive, and the tie-group column is why.** Both "
        "engines are exact, so they agree on every *score* (C4); where they return "
        "different ids they are choosing among chunks whose scores are equal, and "
        "neither choice is more correct. This corpus makes that common rather than "
        "exotic -- a course table repeated verbatim across curriculum revisions "
        "embeds identically, and one `course` query's entire top-12 sits at a single "
        "score. C4b is the testable form: every moved id carries the tied score.",
        "",
        "`indexed vectors` counts **two per point** (dense + sparse); anything short "
        "of 2x `points` is dense rows in a segment still under "
        "`indexing_threshold=20000`, i.e. plain-scanned rather than HNSW-traversed. "
        "That residue is immaterial here only because the recommendation is "
        "`exact=True` -- it would matter to an ANN deployment. This resolves the "
        "\"unexplained ~1.93x\" left open by `qdrant_pilot.md`.",
        "",
        "## 3. Per route",
        "",
        "| route | queries | reference | served | delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for r, v in sorted(an["routes"].items(), key=lambda kv: -kv[1]["n"]):
        L.append(
            f"| `{r}` | {v['n']} | {v['ref']:.4f} | {v['srv']:.4f} | "
            f"{v['srv'] - v['ref']:+.4f} |"
        )
    L += [
        "",
        "## 4. Self-checks",
        "",
        "| check | verdict | detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in ck:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")

    n_pass = sum(1 for _, ok, _ in ck if ok)
    L += [
        "",
        f"**{n_pass}/{len(ck)} checks pass.**",
        "",
        "## 5. What this does and does not establish",
        "",
        "- **Does**: all four routed collections serve the shipped router's answer, "
        "end to end, out of the engine; the sparse arm is exact on every one of "
        "them, not only the pilot's; `points_count` matches the index row count "
        "everywhere.",
        "- **Does not**: this is one query set, one fetch depth, one fusion, and "
        "**no network hop** -- client and server are on the same box. It says "
        "nothing about ANN (deliberately: the pilot's recommendation is `exact=True`), "
        "nothing about concurrency (`qdrant_concurrency.md` covers that, and found "
        "the GPU binds first), and nothing about a rebuild: a re-ingest after any "
        "index rebuild has to re-run this.",
        "- **It is wired** (2026-08-13): `qdrant_hybrid` is a registered retriever, "
        "`query_service.query_indices` skips `embeddings.npy` for any retriever "
        "declaring `reads_index_rows = False`, and C8 above runs the shipped "
        "`route_query` path against the arm this script assembles by hand. One "
        "spec serves all four collections because the collection name is resolved "
        "from `Index.provenance` at query time. Nothing *defaults* to it: the "
        "in-process retrievers are still what every eval and the UI select unless "
        "a `qdrant_hybrid` spec is passed.",
        "- **A known serving gap, not introduced here**: `query_indices` builds an "
        "embedder per call and never releases it, so a served deployment reloads "
        "the query encoder on every request. That is why C8 is a subset rather "
        "than all 106, and it is pre-existing on the in-process path (the "
        "Streamlit UI has it too). `qdrant_concurrency.md` already measured that "
        "the encoder, not the engine, is what saturates.",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:6333")
    ap.add_argument("--smoke", action="store_true",
                    help="two queries per route, so every collection is touched")
    ap.add_argument("--render", action="store_true", help="re-render from the cache")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    if args.render:
        raw = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        raw = collect(args)
        if not args.smoke:
            RAW.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    md = render(raw)
    if args.smoke and not args.render:
        print(md)
        ok = all(o for _, o, _ in checks(raw, analyse(raw)))
        return 0 if ok else 1

    REPORT.write_text(md, encoding="utf-8")
    print(md)
    ok = all(o for _, o, _ in checks(raw, analyse(raw)))
    print(f"\nwrote {REPORT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
