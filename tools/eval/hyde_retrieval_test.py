"""Does HyDE help retrieval on this corpus? Pre-registered, then measured.

HyDE (Hypothetical Document Embeddings): ask an LLM to write the passage that
*would* answer the query, embed **that**, and retrieve with it. A pure query
transform -- no index rebuild, one generation serves every combo.

**The prediction was written before this script existed** and is frozen in
`docs/hyde-axis-notes.md` (2026-08-07), reproduced verbatim in `PREREGISTERED`
below and printed into every report so a reader never has to take the outcome's
word for what was expected:

  1. On **73det**, HyDE ties or degrades, and is **worst on `person`**.
  2. On **thematic**, it may improve, most for the weak embedders.
  3. An improvement on **73det** would falsify the reasoning and is the more
     interesting outcome -- record it as such if it happens.

Design, and why each choice is not free:

* **HyDE feeds the dense arm only; BM25 always gets the raw query.** Frozen in
  the notes. Mixing would confound "HyDE helps dense" with "HyDE poisons BM25".
  That second claim is a *premise* of the design and was unmeasured, so arm
  `poison` measures it directly on the primary combo (P3) instead of leaving it
  asserted -- the same move as measuring the `weighted` x `fetch_depth` guard
  rather than keeping it.
* **Four vector arms, because "HyDE" underdetermines the implementation.** This
  project's indices are built with `embed()` (passage side) while queries go
  through `embed_query()`, which for `e5`/`jina_v5`/`qwen3` is a *different
  encoding* (a `query: ` prefix or a `prompt_name="query"` prompt). A
  hypothetical **document** arguably belongs on the passage side. So: `raw`
  (baseline, the published path), `hyde` (document embedded as a passage, the
  canonical reading), `hyde_q` (document embedded as a query, the naive
  drop-in), `concat` (query + document, the standard robustness variant),
  `hyde_half` (a prefix of the same document). A null on one formulation is weak
  evidence; a null on four is the finding.
* **`hyde_half` exists because every generated document hit the 256-token cap**
  (285 of 285), so "the cap crippled the treatment" is a live objection to any
  null. Greedy decoding is a prefix process, so a prefix of the generation is
  *exactly* what a smaller cap would have produced -- this measures the slope
  with respect to length using text already on disk, no generation. It bounds
  the objection rather than settling it: it says which direction length pushes,
  not what an uncapped document would have scored.
* **The generation is read from a cache, never produced here.** `temperature=0`
  is not reproducible on this stack, so per-arm generation would unpair the
  comparison. See `hyde_generate.py`.
* **Unrouted, on one combo, and that is a stated limitation not an oversight.**
  Significance is tested on `sentence x qwen3_0.6b` (the shipped-best hybrid
  combo); the 36-combo macro is descriptive, because combos are not independent
  samples. **A positive result here would still have to be re-measured against
  the shipped hard router before it meant anything** -- per-`entity_type` alpha
  and rrf4 both beat their unrouted baseline and then died against the router,
  twice, for the same reason ([[feedback_per_type_repair_substitutes_for_routing]]).
  A *negative* result needs no such re-measurement, which is the asymmetry that
  makes an unrouted test the right first test for a prediction of failure.

Method. Retrieval is replicated in numpy from the shipped retrievers rather than
re-run through them, so eight arms cost about what two retrieval passes cost --
the same trick as `hybrid_alpha_sweep.py` and `hybrid_fetch_depth_sweep.py`,
whose `fuse_at_depth` is **imported** here rather than reimplemented (two copies
of that tie-break would eventually disagree). The three replication traps are
inherited with it: per-query gemv rather than a batched matmul, hybrid's
dense-first stable tie-break, and anchoring at F = n.

Self-checks refuse to publish. S2/S3/S4 gate the **`raw`** arm against the
persisted results -- if the baseline is not the published ranking, no treatment
column means anything. Those exercise only the untreated path and would pass
unchanged if the HyDE vectors were silently identical to the raw ones, so **S5**
checks that the mechanism is live ([[feedback_anchor_a_check_where_the_mechanism_is_live]]),
and **S7** checks the inline metrics against `rag_lab.metrics` itself.

Read-only: consumes indices, persisted results, the gold sets and the generation
cache; writes one report and one raw cache, and no index.

Usage:
    python tools/eval/hyde_retrieval_test.py --set 73det --smoke   # ~2 min, writes nothing
    python tools/eval/hyde_retrieval_test.py --set 73det
    python tools/eval/hyde_retrieval_test.py --set thematic
    python tools/eval/hyde_retrieval_test.py --set 73det --render  # no GPU
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from rank_bm25 import BM25Okapi

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

from embedder_matrix_9way import (  # noqa: E402
    _EXCLUDED_COMBO_DIRS,
    _embedder_label,
    bootstrap_pvalue,
    holm_correct,
)
from hybrid_fetch_depth_sweep import fuse_at_depth, persisted_top10  # noqa: E402
from hyde_generate import load_cache, load_set  # noqa: E402
from pythainlp.tokenize import word_tokenize  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder  # noqa: E402

INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
RES = REPO / "data" / "results"

SETS = {
    "73det": {
        "dense": RES / "gold_73det_full_embedder_matrix",
        "bm25": RES / "gold_bm25_73det",
        "hybrid": RES / "gold_hybrid_73det",
    },
    "thematic": {
        "dense": RES / "thematic_dense",
        "bm25": RES / "thematic_bm25",
        "hybrid": RES / "thematic_hybrid",
    },
}

K = 10
N_BOOT = 10_000
SEED = 42
ALPHA = 0.05

# The combo every significance test is run on: sentence x qwen3_0.6b, the
# shipped-best hybrid combo, so the result is quoted on the configuration anyone
# would deploy. The run prints the model it actually loaded rather than trusting
# the name -- a plausible-looking hash resolves to a real directory holding a
# different model, and nothing about the name says which.
PRIMARY_COMBO = "plain__sentence__qwen3__ff8f6c49"
PRIMARY_EXPECT = "Qwen/Qwen3-Embedding-0.6B"

# Vector arms. The value is how the dense query vector is built; BM25 always
# receives the raw query's tokens (the frozen design), except in `poison`.
VECTOR_ARMS = ["raw", "hyde", "hyde_q", "concat", "hyde_half"]
METRICS = ["recall@10", "mrr", "ndcg@10"]

PREREGISTERED = {
    "source": "docs/hyde-axis-notes.md, written 2026-08-07, before any of this was built",
    "P1": "On 73det, HyDE ties or degrades against the raw query, and is worst on `person`.",
    "P2": "On thematic, HyDE may improve, most for the weak embedders.",
    "P3": "Feeding HyDE into BM25 as well is worse than feeding the dense arm only "
          "(hallucinated tokens into a lexical matcher should be actively harmful). "
          "This is a premise of the design, measured here rather than assumed.",
    "P4": "An improvement on 73det would falsify the reasoning and is the MORE "
          "INTERESTING outcome -- to be recorded as such, not explained away.",
    "primary family": "Family 1, m=6: `hyde` vs `raw` on the primary combo, "
                      "{dense, hybrid} x {recall@10, MRR, nDCG@10}, paired bootstrap "
                      f"({N_BOOT} resamples, seed {SEED}) + Holm.",
    "decision rule": "P1 stands unless a Holm-adjusted p < 0.05 shows `hyde` BEATING "
                     "`raw` on recall@10 in family 1 on 73det. A tie is reported as a "
                     "BOUND (what it rules out), never as 'no difference'. The other "
                     "formulations (`hyde_q`, `concat`, `hyde_half`), the 36-combo "
                     "macro and the per-embedder table are EXPLORATORY and cannot "
                     "promote a null to a win.",
    "known limitation": "Unrouted. A positive result would need re-measuring against "
                        "the shipped hard router before it meant anything; a negative "
                        "one would not.",
}


def score_top(top_rows, rid_arr, gold: set[str]) -> tuple[float, float, float]:
    """recall@K, reciprocal rank, nDCG@K at the resolution level (ADR-0002).

    Replicates `rag_lab.metrics` over a top-K chunk window: the window is taken
    over CHUNKS first and deduped to resolutions after slicing, so redundant
    chunks from one resolution can crowd another out. S7 gates this against the
    real functions rather than against the comment.
    """
    seen: set[str] = set()
    dcg = 0.0
    rr = 0.0
    for rank, row in enumerate(top_rows[:K], 1):
        r = rid_arr[row]
        if r in gold:
            if rr == 0.0:
                rr = 1.0 / rank
            if r not in seen:
                seen.add(r)
                dcg += 1.0 / math.log2(rank + 1)
    ideal = min(len(gold), K)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return len(seen) / len(gold), rr, (dcg / idcg if idcg > 0 else 0.0)


def check_metrics_against_library(rid_arr, cid, gold: set[str], rows) -> tuple[bool, str]:
    """S7: the inline metrics must equal `rag_lab.metrics` on real rankings."""
    from rag_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
    from rag_lab.schema import RankedChunk, RetrievalResult

    res = RetrievalResult(
        query="s7",
        combination_id="s7",
        top_k=K,
        retriever="dense",
        results=[
            RankedChunk(
                chunk_id=cid[row], resolution_id=rid_arr[row], page=1,
                score=1.0 / rank, rank=rank, text="",
            )
            for rank, row in enumerate(rows[:K], 1)
        ],
    )
    want = (
        recall_at_k(res, sorted(gold), K),
        reciprocal_rank(res, sorted(gold)),
        ndcg_at_k(res, sorted(gold), K),
    )
    got = score_top(rows, rid_arr, gold)
    ok = all(abs(a - b) < 1e-12 for a, b in zip(want, got))
    return ok, f"library {tuple(round(x, 6) for x in want)} vs inline {tuple(round(x, 6) for x in got)}"


def build_query_vectors(spec_json: str, queries: list[str], docs: dict[str, str]) -> dict:
    """One embedder, all four vector arms.

    `hyde` goes through `embed()` (the passage side, what every chunk in the
    index went through) and `hyde_q` through `embed_query()`. For `local`
    embedders those are the same call; for `e5`/`jina_v5`/`qwen3` they are not,
    which is the whole reason both arms exist.
    """
    emb_obj = build_embedder(StrategySpec.model_validate(json.loads(spec_json)))
    hyde_texts = [docs[q] for q in queries]
    # `hyde_half` measures the LENGTH axis for free, and it is not a proxy:
    # greedy decoding is a prefix process, so a prefix of the generation is
    # exactly the text a smaller `num_predict` would have produced. Every
    # document hit the 256-token cap, so "would a longer document have helped?"
    # is a live objection; this answers the slope of it without generating
    # anything. Halved by characters, so which cap it corresponds to is
    # approximate even though the prefix property is exact.
    half_texts = [t[: len(t) // 2] for t in hyde_texts]
    out = {
        "raw": [emb_obj.embed_query(q) for q in queries],
        "hyde": list(emb_obj.embed(hyde_texts)),
        "hyde_q": [emb_obj.embed_query(t) for t in hyde_texts],
        "concat": [emb_obj.embed_query(q + "\n" + docs[q]) for q in queries],
        "hyde_half": list(emb_obj.embed(half_texts)),
    }
    release = getattr(emb_obj, "release", None)
    if callable(release):
        release()
    del emb_obj
    return out


def run(set_name: str, smoke: bool, arms: list[str], do_poison: bool) -> dict:
    t_start = time.time()
    rows = load_set(set_name)
    docs_cache = load_cache()

    queries = [r["query"] for r in rows]
    qrels = {r["query"]: set(r["relevant_resolution_ids"]) for r in rows}
    etype = {r["query"]: r.get("entity_type", "?") for r in rows}
    if smoke:
        # Spread across the set rather than taking a prefix (entity types are
        # grouped), but keep only what the cache already holds, so a smoke run
        # can be done while generation is still in flight. It writes nothing,
        # and S1/S6 are scoped out of its exit code for exactly that reason.
        have = [q for q in queries if q in docs_cache]
        stride = max(1, len(have) // 8)
        queries = have[::stride][:8]
        print(f"smoke: {len(queries)} queries of {len(have)} cached", file=sys.stderr)

    checks: list[tuple[str, bool, str]] = []
    missing = [q for q in queries if q not in docs_cache]
    settings = sorted({(e["model"], e["num_predict"], e["num_ctx"])
                       for q, e in docs_cache.items() if q in set(queries)})
    checks.append((
        "S6 every query has a cached hypothetical document, from one generator",
        not missing and len(settings) == 1,
        f"{len(queries) - len(missing)} of {len(queries)} cached; "
        f"settings {settings if settings else 'none'}",
    ))
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(queries)} queries have no cached document. "
            "Run tools/eval/hyde_generate.py first."
        )
    docs = {q: docs_cache[q]["doc"] for q in queries}
    q_tokens = {q: word_tokenize(q) for q in queries}

    with_results = sorted(
        {"__".join(f.stem.split("__")[:4]) for f in SETS[set_name]["hybrid"].glob("*.json")}
    )
    combos = [c for c in with_results if (INDEX_DIR / c).is_dir() and c not in _EXCLUDED_COMBO_DIRS]
    checks.append((
        "S1 combo set derived from existing index dirs, not a bare glob",
        len(combos) == 36, f"{len(combos)} kept of {len(with_results)} with results",
    ))
    if smoke:
        combos = [PRIMARY_COMBO] + [c for c in sorted(combos) if c != PRIMARY_COMBO][:1]

    manifests = {
        c: json.loads((INDEX_DIR / c / "manifest.json").read_text(encoding="utf-8"))
        for c in combos
    }
    labels = {c: _embedder_label(manifests[c]["combo"]) for c in combos}
    chunker_of = {c: manifests[c]["combo"]["chunker"]["type"] for c in combos}
    primary_model = manifests[PRIMARY_COMBO]["combo"]["embedder"]["params"].get("model_name")
    checks.append((
        "S0 the primary combo really holds the model the report names",
        primary_model == PRIMARY_EXPECT, f"{PRIMARY_COMBO} -> {primary_model}",
    ))

    by_embedder: dict[str, list[str]] = collections.defaultdict(list)
    for c in combos:
        by_embedder[json.dumps(manifests[c]["combo"]["embedder"], sort_keys=True)].append(c)
    qvecs: dict[str, dict] = {}
    for spec_json in sorted(by_embedder):
        qvecs[spec_json] = build_query_vectors(spec_json, queries, docs)
        print(f"  encoded {len(queries)} x {len(arms)} arms for "
              f"{json.loads(spec_json).get('params', {}).get('model_name', '?')}"
              f"  {time.time() - t_start:.0f}s", file=sys.stderr, flush=True)

    # S5: the treatment must actually differ from the baseline. A null with the
    # HyDE vector silently equal to the raw one would look identical to a null
    # with a live transform, and only this check can tell them apart.
    sims = []
    for spec_json, v in qvecs.items():
        for j in range(len(queries)):
            a = np.asarray(v["raw"][j], dtype=np.float64)
            b = np.asarray(v["hyde"][j], dtype=np.float64)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            sims.append(float(a @ b / (na * nb)) if na and nb else 0.0)

    # scores[combo][arm][metric] -> one float per query, in `queries` order
    scores: dict[str, dict[str, dict[str, list[float]]]] = {}
    poison: dict[str, dict[str, list[float]]] = {}
    dense_ok = dense_bad = bm_ok = bm_bad = hyb_ok = hyb_bad = 0
    bm_combos = 0
    changed = same = 0
    s7: list[tuple[bool, str]] = []
    bm25_cache: dict[str, tuple[list[str], np.ndarray]] = {}
    cache_misaligned: list[str] = []

    for ci, combo in enumerate(sorted(combos), 1):
        d = INDEX_DIR / combo
        cols = pq.read_table(d / "chunks.parquet", columns=["chunk_id", "resolution_id"]).to_pydict()
        cid, rid = cols["chunk_id"], cols["resolution_id"]
        rid_arr = np.array(rid, dtype=object)
        n = len(cid)

        emb32 = np.load(d / "embeddings.npy")
        row_norms = np.linalg.norm(emb32, axis=1)
        # Promote once instead of once per gemv. `f32 @ f64` promotes internally
        # to exactly this array, so the arithmetic -- and therefore the tie
        # structure S2 gates on -- is unchanged; only the repeated 300 MB
        # conversion goes away.
        emb = emb32.astype(np.float64)
        del emb32
        qv = qvecs[json.dumps(manifests[combo]["combo"]["embedder"], sort_keys=True)]

        ck = chunker_of[combo]
        bpos_all = None
        if ck in bm25_cache:
            cached_cid, cached = bm25_cache[ck]
            if cached_cid == cid:
                bpos_all = cached
            else:
                cache_misaligned.append(combo)
        lex = None
        if bpos_all is None:
            lex = json.loads((d / "lexical.json").read_text(encoding="utf-8"))
            bm = BM25Okapi(lex)
            bpos_all = np.empty((len(queries), n), dtype=np.int64)
            for j, q in enumerate(queries):
                border = np.argsort(-bm.get_scores(q_tokens[q]))
                bpos_all[j][border] = np.arange(n)
            bm25_cache[ck] = (cid, bpos_all)
            del bm

        ptop_d = persisted_top10(SETS[set_name]["dense"], combo, "dense")
        ptop_b = persisted_top10(SETS[set_name]["bm25"], combo, "bm25")
        ptop_h = persisted_top10(SETS[set_name]["hybrid"], combo, "hybrid")
        bm_combos += bool(ptop_b)

        per_arm: dict[str, dict[str, list[float]]] = {
            f"{r}_{a}": {m: [] for m in METRICS} for a in arms for r in ("dense", "hybrid")
        }

        want_poison = do_poison and combo == PRIMARY_COMBO
        if want_poison:
            if lex is None:
                lex = json.loads((d / "lexical.json").read_text(encoding="utf-8"))
            bm_p = BM25Okapi(lex)
            poison = {m: [] for m in METRICS}

        for j, q in enumerate(queries):
            gold = qrels[q]
            bpos = bpos_all[j]
            border = np.argsort(bpos)
            if q in ptop_b:
                ok = [cid[i] for i in border[:K]] == ptop_b[q]
                bm_ok, bm_bad = bm_ok + ok, bm_bad + (not ok)

            raw_top: list[int] = []
            for arm in arms:
                qq = np.asarray(qv[arm][j], dtype=np.float64)
                denom = row_norms * np.linalg.norm(qq)
                dots = emb @ qq
                dscore = np.divide(
                    dots, denom, out=np.zeros_like(dots, dtype=np.float64), where=denom > 0
                )
                dorder = np.argsort(-dscore)
                dpos = np.empty(n, dtype=np.int64)
                dpos[dorder] = np.arange(n)

                dtop = dorder[:K]
                if arm == "raw":
                    raw_top = list(dtop)
                    if q in ptop_d:
                        ok = [cid[i] for i in dtop] == ptop_d[q]
                        dense_ok, dense_bad = dense_ok + ok, dense_bad + (not ok)
                    if len(s7) < 3:
                        s7.append(check_metrics_against_library(rid_arr, cid, gold, list(dtop)))
                elif arm == "hyde":
                    changed += int(list(dtop) != raw_top)
                    same += int(list(dtop) == raw_top)

                for m, v in zip(METRICS, score_top(list(dtop), rid_arr, gold)):
                    per_arm[f"dense_{arm}"][m].append(v)

                htop = fuse_at_depth(dorder, dpos, border, bpos, n)
                if arm == "raw" and q in ptop_h:
                    ok = [cid[i] for i in htop] == ptop_h[q]
                    hyb_ok, hyb_bad = hyb_ok + ok, hyb_bad + (not ok)
                for m, v in zip(METRICS, score_top(list(htop), rid_arr, gold)):
                    per_arm[f"hybrid_{arm}"][m].append(v)

                if want_poison and arm == "hyde":
                    pb = np.argsort(-bm_p.get_scores(word_tokenize(docs[q])))
                    pbpos = np.empty(n, dtype=np.int64)
                    pbpos[pb] = np.arange(n)
                    ptop = fuse_at_depth(dorder, dpos, pb, pbpos, n)
                    for m, v in zip(METRICS, score_top(list(ptop), rid_arr, gold)):
                        poison[m].append(v)

        scores[combo] = per_arm
        del emb, lex
        print(f"  [{ci}/{len(combos)}] {combo}  {time.time() - t_start:.0f}s",
              file=sys.stderr, flush=True)

    # Denominators are printed because 0 is ambiguous between "examined and
    # clean" and "nothing was there to examine" -- the E3 rule. It bites here:
    # `thematic_bm25` persists only 4 of the 36 combos, so a bare "0 differ"
    # would hide that S3 saw an eighth of the pairs S2 did.
    possible = len(queries) * len(combos)
    checks.append((
        "S2 the `raw` dense top-10 reproduces the persisted results",
        dense_bad == 0 and dense_ok > 0,
        f"{dense_ok} reproduce, {dense_bad} differ, of {possible} query-combo pairs",
    ))
    checks.append((
        "S3 BM25 top-10 reproduces the persisted results",
        bm_bad == 0 and bm_ok > 0,
        f"{bm_ok} reproduce, {bm_bad} differ, of {len(queries) * bm_combos} query-combo "
        f"pairs ({bm_combos} of {len(combos)} combos persist a BM25 arm; BM25 depends "
        "only on the chunker, so the missing ones are duplicates, not gaps)",
    ))
    checks.append((
        "S3b combos sharing a chunker share chunk rows (licenses the BM25 cache)",
        not cache_misaligned,
        f"{len(bm25_cache)} lexical indices for {len(combos)} combos; "
        f"{len(cache_misaligned)} misaligned",
    ))
    checks.append((
        "S4 the `raw` hybrid top-10 reproduces the persisted results",
        hyb_bad == 0 and hyb_ok > 0,
        f"{hyb_ok} reproduce, {hyb_bad} differ, of {possible} query-combo pairs",
    ))
    checks.append((
        "S5 the treatment is live: HyDE vectors differ, and move the ranking",
        max(sims) < 0.999 and changed > 0,
        f"max cos(raw, hyde) {max(sims):.4f} over {len(sims)} query-embedder pairs; "
        f"dense top-10 changed on {changed} of {changed + same} query-combo pairs",
    ))
    checks.append((
        "S7 the inline metrics equal rag_lab.metrics on real rankings",
        all(ok for ok, _ in s7), "; ".join(det for _, det in s7) or "not sampled",
    ))

    return {
        "set": set_name,
        "smoke": smoke,
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_s": time.time() - t_start,
        "arms": arms,
        "combos": sorted(combos),
        "labels": labels,
        "chunker_of": chunker_of,
        "queries": queries,
        "etype": etype,
        "n_relevant": {q: len(qrels[q]) for q in queries},
        "primary": {"combo": PRIMARY_COMBO, "model": primary_model},
        "scores": scores,
        "poison": poison,
        "checks": [list(c) for c in checks],
        "cap_hit": sum(1 for q in queries if docs_cache[q]["hit_cap"]),
        "doc_settings": settings,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def test_family(
    pairs: list[tuple[str, str, list[float], list[float]]], rng
) -> list[tuple]:
    """Paired bootstrap + Holm over one family. `pairs` are (label_a, label_b, a, b)."""
    raw = []
    for la, lb, a, b in pairs:
        diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        observed, p, ci = bootstrap_pvalue(diffs, rng, N_BOOT)
        raw.append((la, lb, observed, p, ci))
    return holm_correct(raw, ALPHA)


def verdict(row) -> str:
    _, _, diff, _, _, adj, sig = row
    if not sig:
        return "ns"
    return "treatment better" if diff > 0 else "treatment WORSE"


def fam_table(L: list[str], rows: list[tuple], m: int) -> None:
    L.append(f"Holm family size m={m}.")
    L.append("")
    L.append("| comparison | diff | 95% CI | p | Holm-adj | verdict |")
    L.append("|---|---|---|---|---|---|")
    for la, lb, diff, p, ci, adj, sig in rows:
        L.append(
            f"| {la} vs {lb} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{p:.4f} | {adj:.4f} | {verdict((la, lb, diff, p, ci, adj, sig))} |"
        )
    L.append("")


def render(d: dict) -> str:
    rng = np.random.default_rng(SEED)
    set_name = d["set"]
    queries = d["queries"]
    scores = d["scores"]
    primary = d["primary"]["combo"]
    arms = d["arms"]
    L: list[str] = []

    L.append(f"# HyDE on the {set_name} query set")
    L.append("")
    L.append(
        f"`tools/eval/hyde_retrieval_test.py`; run {d['at']}, "
        f"{d['elapsed_s'] / 60:.1f} min, {len(d['combos'])} combos x {len(queries)} queries."
    )
    L.append("")

    L.append("## 0. What was predicted, before any of this ran")
    L.append("")
    for k, v in PREREGISTERED.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L.append("## 1. Self-checks")
    L.append("")
    L.append("| check | result | detail |")
    L.append("|---|---|---|")
    for name, ok, det in d["checks"]:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {det} |")
    L.append("")

    L.append("## 2. Primary result (family 1, pre-registered)")
    L.append("")
    L.append(
        f"Combo `{primary}` = {d['labels'][primary]}, model `{d['primary']['model']}`. "
        "`hyde` is the hypothetical document embedded on the passage side; BM25 "
        "receives the raw query in both arms."
    )
    L.append("")
    L.append("| arm | " + " | ".join(METRICS) + " |")
    L.append("|---|" + "---|" * len(METRICS))
    for r in ("dense", "hybrid"):
        for a in arms:
            key = f"{r}_{a}"
            L.append(f"| {key} | " + " | ".join(
                f"{np.mean(scores[primary][key][m]):.4f}" for m in METRICS) + " |")
    L.append("")
    fam1 = [
        (f"{r}_hyde", f"{r}_raw", scores[primary][f"{r}_hyde"][m], scores[primary][f"{r}_raw"][m])
        for r in ("dense", "hybrid")
        for m in METRICS
    ]
    rows1 = test_family(fam1, rng)
    labelled = [(f"{la} [{m}]", lb, *rest) for (la, lb, *rest), m in
                zip(rows1, [m for _ in ("dense", "hybrid") for m in METRICS])]
    fam_table(L, labelled, len(fam1))
    r10 = rows1[0]
    L.append(
        f"**Bound on the primary metric.** dense recall@10 moves {r10[2]:+.4f}, "
        f"CI [{r10[4][0]:+.4f}, {r10[4][1]:+.4f}], Holm-adj {r10[5]:.4f}. "
        + ("HyDE significantly beats the raw query here, which FALSIFIES the "
           "pre-registered reasoning (P4) -- record it as the interesting outcome, "
           "and re-measure against the shipped hard router before acting on it."
           if r10[6] and r10[2] > 0 else
           ("HyDE is significantly WORSE than the raw query." if r10[6] else
            f"Not significant -- state it as a bound, never as 'no difference': it "
            f"rules out HyDE gaining more than {abs(r10[4][1]):.4f} recall@10, and "
            f"rules out it losing more than {abs(r10[4][0]):.4f}."))
    )
    L.append("")

    types = sorted({d["etype"][q] for q in queries})
    if len(types) > 1:
        L.append("## 3. Per entity_type (family 2, pre-registered: worst on `person`)")
        L.append("")
        idx = {t: [i for i, q in enumerate(queries) if d["etype"][q] == t] for t in types}
        L.append("| entity_type | n | dense_raw | dense_hyde | diff |")
        L.append("|---|---|---|---|---|")
        fam2 = []
        for t in types:
            ii = idx[t]
            a = [scores[primary]["dense_hyde"]["recall@10"][i] for i in ii]
            b = [scores[primary]["dense_raw"]["recall@10"][i] for i in ii]
            L.append(f"| {t} | {len(ii)} | {np.mean(b):.4f} | {np.mean(a):.4f} | "
                     f"{np.mean(a) - np.mean(b):+.4f} |")
            fam2.append((f"dense_hyde [{t}]", "dense_raw", a, b))
        L.append("")
        fam_table(L, test_family(fam2, rng), len(fam2))

    L.append("## 4. The other formulations (exploratory)")
    L.append("")
    L.append(
        "`hyde_q` embeds the same document through `embed_query()` and `concat` "
        "embeds query + document. These cannot promote a null in family 1 to a "
        "win; they exist so a null is a null about **HyDE**, not about one way of "
        "wiring it."
    )
    L.append("")
    fam3 = [
        (f"{r}_{a}", f"{r}_raw", scores[primary][f"{r}_{a}"]["recall@10"],
         scores[primary][f"{r}_raw"]["recall@10"])
        for r in ("dense", "hybrid") for a in arms if a not in ("raw", "hyde")
    ]
    if fam3:
        fam_table(L, test_family(fam3, rng), len(fam3))

    L.append("## 5. Macro over every combo (descriptive)")
    L.append("")
    L.append(
        "Combos are not independent samples, so nothing here is significance-tested. "
        "It answers a different question: is the primary combo representative, or is "
        "HyDE helping somewhere else?"
    )
    L.append("")
    L.append("| arm | " + " | ".join(METRICS) + " |")
    L.append("|---|" + "---|" * len(METRICS))
    for r in ("dense", "hybrid"):
        for a in arms:
            key = f"{r}_{a}"
            vals = {m: float(np.mean([np.mean(scores[c][key][m]) for c in d["combos"]]))
                    for m in METRICS}
            L.append(f"| {key} | " + " | ".join(f"{vals[m]:.4f}" for m in METRICS) + " |")
    L.append("")
    L.append("Per embedder, dense recall@10 (averaged over the 4 chunkers):")
    L.append("")
    L.append("| embedder | raw | hyde | diff |")
    L.append("|---|---|---|---|")
    by_lab: dict[str, list[str]] = collections.defaultdict(list)
    for c in d["combos"]:
        by_lab[d["labels"][c]].append(c)
    per_lab = []
    for lab in sorted(by_lab):
        cs = by_lab[lab]
        b = float(np.mean([np.mean(scores[c]["dense_raw"]["recall@10"]) for c in cs]))
        a = float(np.mean([np.mean(scores[c]["dense_hyde"]["recall@10"]) for c in cs]))
        per_lab.append((lab, b, a))
    for lab, b, a in sorted(per_lab, key=lambda x: -x[1]):
        L.append(f"| {lab} | {b:.4f} | {a:.4f} | {a - b:+.4f} |")
    L.append("")
    if len(per_lab) > 2:
        xs = np.array([b for _, b, _ in per_lab])
        ys = np.array([a - b for _, b, a in per_lab])
        r = float(np.corrcoef(xs, ys)[0, 1])
        L.append(
            f"Correlation between an embedder's baseline strength and what HyDE does "
            f"to it: **r = {r:+.3f}**. P2 predicts help concentrates on the weak "
            f"embedders, i.e. a negative r."
        )
        L.append("")

    if d.get("poison"):
        L.append("## 6. P3: what happens if BM25 gets the hypothetical document too")
        L.append("")
        L.append(
            "The frozen design gives BM25 the raw query. That is a **premise**, not a "
            "measurement, so here it is measured on the primary combo: same dense "
            "HyDE vector, BM25 fed the generated document's tokens instead of the "
            "query's."
        )
        L.append("")
        L.append("| arm | " + " | ".join(METRICS) + " |")
        L.append("|---|" + "---|" * len(METRICS))
        for key, src in (("hybrid_raw", scores[primary]["hybrid_raw"]),
                         ("hybrid_hyde (BM25 raw, shipped design)", scores[primary]["hybrid_hyde"]),
                         ("hybrid_hyde + BM25 poisoned", d["poison"])):
            L.append(f"| {key} | " + " | ".join(f"{np.mean(src[m]):.4f}" for m in METRICS) + " |")
        L.append("")
        fam_table(L, test_family([
            ("hybrid_hyde+poison", "hybrid_hyde", d["poison"][m], scores[primary]["hybrid_hyde"][m])
            for m in METRICS
        ], rng), len(METRICS))

    L.append("## 7. Verdict against the pre-registration")
    L.append("")
    L.append(
        f"Generation: {d['cap_hit']} of {len(queries)} documents hit the "
        f"`num_predict` cap (settings {d['doc_settings']}); see "
        "`data/results/hyde_generation.md` for what the generator wrote."
    )
    L.append("")
    L.append(
        "_Written by hand after reading the tables above -- deliberately not "
        "auto-generated, because the point of a pre-registration is that a human "
        "states whether it held._"
    )
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=sorted(SETS), default="73det")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--arms", default=",".join(VECTOR_ARMS))
    ap.add_argument("--no-poison", action="store_true")
    args = ap.parse_args()

    raw_path = RES / f"hyde_retrieval_{args.set}_raw.json"
    out_path = RES / f"hyde_retrieval_{args.set}.md"

    if args.render:
        d = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        arms = [a for a in args.arms.split(",") if a]
        if arms[0] != "raw" or "hyde" not in arms:
            raise SystemExit("--arms must start with `raw` and include `hyde`")
        d = run(args.set, args.smoke, arms, not args.no_poison)
        for name, ok, det in d["checks"]:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}: {det}")
        if args.smoke:
            print("smoke: writing nothing (S1 is scoped to the full combo set)")
            return 0 if all(c[1] for c in d["checks"] if not c[0].startswith("S1")) else 1
        if not all(ok for _, ok, _ in d["checks"]):
            print("self-check failed; refusing to publish numbers")
            return 1
        raw_path.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    out_path.write_text(render(d), encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
