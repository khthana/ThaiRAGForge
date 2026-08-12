"""Does `weighted` fusion survive a truncated fetch the way `rrf` does?

`HybridRetriever.__init__` refuses `method="weighted"` with a non-`None`
`fetch_depth`. That guard is **containment, not a verdict**: the sweep
(`hybrid_fetch_depth_sweep.py`) and the routed test both measured `rrf` only,
and under `weighted` a chunk past one arm's cut has that arm's *normalized
score* read as 0 rather than merely losing an RRF term. Nobody had measured
whether that matters, and a harsher fusion returns a plausible ranking rather
than an error. The guard's own docstring names its exit condition -- "measure
it and lift it" -- and this script is that measurement.

**Two questions, deliberately kept apart.** `weighted` has never been scored on
the Gold set at all, so "how much does truncation cost `weighted`" is
meaningless without "where does `weighted` sit to begin with". Both columns are
reported, each against its own F=n baseline, and the cross-method row is stated
separately.

Method. The fusion is replicated in numpy from the shipped retriever, exactly
as the `rrf` sweep does, so 11 depths x 2 methods cost about what one retrieval
pass costs. The `rrf` half is not reimplemented -- `fuse_at_depth` is
**imported** from `hybrid_fetch_depth_sweep.py`, because two copies of that
tie-break would eventually disagree -- which also makes this run's `rrf`
columns a cross-artifact anchor against the published sweep (S7).

One lemma the replication rests on, stated rather than self-checked because it
is true by construction and a check that cannot fail is a vacuous PASS: **the
`weighted` normalizers are invariant to F.** `_normalize` divides by the max of
the list it is given, each arm's list is sorted by score descending, so the max
over the top-F is the max over all n for every F >= 1. Truncation therefore
changes *which* terms are present, never the scale they are measured on.

Three replication traps, all inherited from where they were found the hard way:

  1. Per-query gemv, not a batched matmul (`miss_depth_profile.py`).
  2. The fusion dict is insertion-ordered `dense[:F]` first, then the BM25-only
     remainder in BM25 rank order, and `sorted` is stable -- so equal scores
     keep dense ahead. `weighted` produces exact ties far more often than `rrf`
     does (every chunk cut from both arms scores exactly 0.0), so this matters
     more here, not less.
  3. S5/S6 check the numpy fusion against the real `HybridRetriever`, S6 at the
     truncated depths where the mechanism under test is actually live. S5 alone
     would pass unchanged if `fuse_weighted_at_depth` ignored F entirely
     (feedback_anchor_a_check_where_the_mechanism_is_live).

Whether the guard survives this measurement is decided by the pre-registered
rule below, which is frozen in this file and committed before the run.

Historical note, because it explains this file's own commit history: S6 has to
construct the very combination the guard forbade, so the first run went through
`allow_unmeasured_truncation`, a named escape hatch of the same shape as
`rq4_generate.py`'s `--allow-small-ctx` rather than a silent removal of the
guard. The rule then came out **LIFT** (see `VERDICT`), which retired both the
guard and the hatch -- so re-running this script today constructs the pair
directly.

Read-only: consumes indices, persisted results and the gold set; writes one
report and one cache, and no index.
"""
from __future__ import annotations

import collections
import json
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

from embedder_matrix_9way import _EXCLUDED_COMBO_DIRS, _embedder_label  # noqa: E402
from hybrid_fetch_depth_sweep import (  # noqa: E402
    BM25_RES,
    DENSE_RES,
    DEPTHS,
    GOLD,
    HYB_RES,
    INDEX_DIR,
    K,
    fuse_at_depth,
    persisted_top10,
)
from hybrid_fetch_depth_sweep import RAW as RRF_RAW  # noqa: E402
from pythainlp.tokenize import word_tokenize  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder  # noqa: E402

OUT = REPO / "data" / "results" / "hybrid_weighted_fetch_depth.md"
RAW = REPO / "data" / "results" / "hybrid_weighted_fetch_depth_raw.json"

# The shipped defaults. Sweeping the weights as well would confound the fusion
# method with the weight, which is the mistake hybrid_alpha_sweep.py was
# designed around.
DW = BW = 0.5

# Frozen before the run and rendered into the report verbatim, so a reader can
# see which of these the numbers refuted. This project has twice learned more
# from a refuted prediction than from a confirmed one (rq4 follow-up (a), the
# fetch-depth sweep's "F=1000 will be identical").
PREDICTIONS = [
    ("P1", "`weighted` needs a deeper F than `rrf` to reproduce its own F=n "
           "top-10 -- i.e. same-order agreement is lower at every depth."),
    ("P2", "`weighted`'s recall@10 damage at a given F is larger in magnitude "
           "than `rrf`'s at the same F."),
    ("P3", "The BM25 side of the cut is nearly free and the dense side is not. "
           "`BM25Okapi` floors its IDF at `eps * average_idf > 0`, so a BM25 "
           "score is >= 0 and the tail of the ranking is *exactly* 0 -- zeroing "
           "a term that was already 0 changes nothing. Dense is cosine, so its "
           "tail can be negative, and there truncation is a **promotion**, not "
           "a demotion. So the perturbation is two-sided under `weighted` where "
           "it is uniformly downward under `rrf`."),
    ("P4", "`weighted` at F=n scores below `rrf` at F=n on macro recall@10 -- "
           "fusing incomparable score scales is the thing RRF exists to avoid."),
]

# Pre-registered before the numbers existed. Stated in terms of *kind* rather
# than an arbitrary threshold: the guard exists because the behaviour was
# unknown, so what licenses lifting it is that the behaviour can be written
# down, not that it is good.
DECISION_RULE = (
    "**LIFT** the guard if (i) S5 and S6 pass at every measured depth, and "
    "(ii) `weighted`'s damage curve is the same *kind* of thing as `rrf`'s -- "
    "monotone-ish in F, bounded, and statable as a number a docstring can "
    "carry. **KEEP** it if the damage is different in kind: non-monotone in a "
    "way that makes \"deep enough\" undefined, or a depth at which `weighted` "
    "beats its own F=n baseline by more than the family's own noise (which "
    "would mean truncation is not an approximation of the untruncated "
    "ranking at all). Either way the docstring stops saying \"nobody has "
    "measured it\"."
)

# Written AFTER the numbers, unlike everything above it. Kept in the source
# rather than only in CLAUDE.md so `--render` reproduces the verdict along with
# the table it rests on.
OUTCOMES = [
    ("P1", "CONFIRMED", "`weighted` agrees with its own F=n top-10 far less "
     "often at every depth -- 33.57% vs `rrf`'s 56.29% at F=200, and still "
     "75.55% vs 88.00% at F=10,000."),
    ("P2", "CONFIRMED",
     "On the full set -- **and refuted by the smoke slice**, "
     "which is the part worth keeping. On 2 combos x 8 queries `weighted` "
     "*gained* from truncation, peaking at 0.7708 (F=100) against 0.5938 at "
     "F=n, i.e. the smoke reversed the sign of the headline and would have "
     "sent the decision down the KEEP branch. A smoke run checks that the "
     "code runs; it is not a small version of the answer."),
    ("P3", "REFUTED",
     "Both halves. The BM25 side is **not** nearly free: only "
     "0.1% of the BM25 terms a cut zeroes were already 0, so BM25 carries "
     "73% of dense's zeroed mass at F=50 (88,301 vs 121,437). A chunk scores "
     "exactly 0 only when it matches *no* query term, and a 20-token query "
     "has common tokens that reach almost every chunk -- the IDF floor makes "
     "each of those contributions positive rather than negative. The "
     "promotion half is real but negligible in the other direction: 2 of "
     "157,717 zeroed dense terms at F=50, 990 of 3,086,381 at F=1,000 "
     "(0.0%). So the perturbation is effectively one-sided after all, just "
     "for the opposite reason to `rrf`'s."),
    ("P4", "REFUTED",
     "In the direction that is worth a follow-up: at F=n "
     "`weighted` scores **above** `rrf` (0.5442 vs 0.5204, +0.0239 macro "
     "recall@10). Descriptive only -- no significance test, macro over 36 "
     "combos, unrouted, and nothing ships `weighted`. Read it as a "
     "hypothesis, never as a result."),
]

VERDICT = (
    "**LIFT.** (i) S5 and S6 reproduce the real `HybridRetriever` exactly at "
    "F=n and at all four truncated depths. (ii) The damage is bounded, "
    "statable and monotone-ish: it shrinks from -0.1161 (F=10) to -0.0112 "
    "(F=10,000) with one ~0.010-wide dip in the 100-500 band, on a 0.116 "
    "span. **Neither KEEP trigger fired** -- no depth beats its own F=n "
    "baseline (every delta is negative; the smoke's apparent baseline-beating "
    "did not survive the full set), and \"deep enough\" is not undefined, it "
    "is ~= n.\n\n"
    "**LIFT is not a recommendation, and the number is the whole point.** At "
    "F=200 `weighted` loses **-0.0609** macro recall@10 against its own F=n, "
    "about **18x** `rrf`'s -0.0033 at the same depth, and `person` alone "
    "loses **-0.1965**. The guard existed because the pair was *unmeasured*, "
    "not because it was bad; this codebase does not ban a measured-but-worse "
    "configuration (nothing bans `m2v`), it bans an unmeasured one from "
    "passing as measured. So the raise goes and the docstring carries the "
    "cost."
)


def candidates_at_depth(
    dorder: np.ndarray, dpos: np.ndarray, border: np.ndarray, F: int
) -> np.ndarray:
    """The fusion dict's key order: `dense[:F]`, then the BM25-only remainder.

    Identical three lines to the ones inside the imported `fuse_at_depth` --
    duplicated rather than factored out of it because that function is the
    published `rrf` path and must not be edited by this script. S6 pins this
    copy against the real retriever, which is what would catch a divergence.
    """
    n = len(dpos)
    dsel = dorder[:F]
    in_dense = np.zeros(n, dtype=bool)
    in_dense[dsel] = True
    bsel = border[:F]
    return np.concatenate([dsel, bsel[~in_dense[bsel]]])


def fuse_weighted_at_depth(
    cand: np.ndarray,
    dpos: np.ndarray,
    dscore: np.ndarray,
    bpos: np.ndarray,
    bscore: np.ndarray,
    F: int,
    dmax: float,
    bmax: float,
) -> np.ndarray:
    """Top-K chunk rows from `weighted` fusion over each arm's top-F only.

    Replicates `HybridRetriever.retrieve`'s `weighted` branch: each arm's score
    is divided by that arm's own maximum (0.0 for every row if the maximum is
    <= 0, which is `_normalize`'s guard and fires whenever a query matches no
    token at all), a row outside an arm's top-F contributes that arm's
    `.get(cid, 0.0)` -- a literal zero, not a floor value -- and the tie-break
    is the stable sort over the dense-first key order.
    """
    dn = dscore[cand] / dmax if dmax > 0 else np.zeros(len(cand))
    bn = bscore[cand] / bmax if bmax > 0 else np.zeros(len(cand))
    fused = DW * np.where(dpos[cand] < F, dn, 0.0) + BW * np.where(
        bpos[cand] < F, bn, 0.0
    )
    return cand[np.argsort(-fused, kind="stable")][:K]


def zeroing_profile(
    cand: np.ndarray,
    dpos: np.ndarray,
    dscore: np.ndarray,
    bpos: np.ndarray,
    bscore: np.ndarray,
    F: int,
    dmax: float,
    bmax: float,
) -> dict[str, tuple[int, int, int, float]]:
    """What truncation actually replaced with 0, per arm, by sign.

    P3 is a claim about the *shape* of the perturbation, not its size, and the
    aggregate recall column cannot distinguish "a term worth 0.4 was deleted"
    from "a term that was already 0 was deleted". Returns, per arm:
    (# strictly positive terms zeroed = demotion, # already-zero = free,
    # negative = promotion, sum of |value| = total perturbation mass).
    """
    out: dict[str, tuple[int, int, int, float]] = {}
    for arm, pos, score, mx in (
        ("dense", dpos, dscore, dmax),
        ("bm25", bpos, bscore, bmax),
    ):
        cut = cand[pos[cand] >= F]
        v = score[cut] / mx if mx > 0 else np.zeros(len(cut))
        out[arm] = (
            int((v > 0).sum()),
            int((v == 0).sum()),
            int((v < 0).sum()),
            float(np.abs(v).sum()),
        )
    return out


def verify_weighted_against_retriever(
    queries: list[str], q_tokens: dict[str, list[str]], combo: str, depths: list[int]
) -> tuple[tuple[bool, str], tuple[bool, str]]:
    """Check the numpy `weighted` fusion against `HybridRetriever` itself.

    Two checks, not one. S5 anchors F=n, where the truncation collapses away and
    only the score-fusion arithmetic is under test -- it is the check that the
    reimplementation of `_normalize` is right. S6 anchors the truncated depths,
    where the mechanism this whole report exists to measure is live; without it
    every published column would rest on reasoning rather than on the code.
    """
    from rag_lab.io.artifact_store import ArtifactStore
    from rag_lab.retrievers.hybrid import HybridRetriever
    from rag_lab.schema import Query

    d = INDEX_DIR / combo
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    index = ArtifactStore().load(d)
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    cid = [c.chunk_id for c in index.chunks]
    n = len(cid)
    emb = np.asarray(index.embeddings)
    row_norms = np.linalg.norm(emb, axis=1)
    bm = BM25Okapi(index.lexical)

    full_ok = full_bad = cut_ok = cut_bad = 0
    for q in queries:
        qq = np.asarray(embedder.embed_query(q), dtype=np.float64)
        denom = row_norms * np.linalg.norm(qq)
        dots = emb @ qq
        dscore = np.divide(
            dots, denom, out=np.zeros_like(dots, dtype=np.float64), where=denom > 0
        )
        dorder = np.argsort(-dscore)
        dpos = np.empty(n, dtype=np.int64)
        dpos[dorder] = np.arange(n)
        bscore = bm.get_scores(q_tokens[q])
        border = np.argsort(-bscore)
        bpos = np.empty(n, dtype=np.int64)
        bpos[border] = np.arange(n)
        dmax, bmax = float(dscore[dorder[0]]), float(bscore[border[0]])

        query = Query(text=q, vector=qq, tokens=q_tokens[q])
        for F in [None, *depths]:
            eff = n if F is None else F
            retr = HybridRetriever(method="weighted", fetch_depth=F)
            real = [r.chunk_id for r in retr.retrieve(query, index, K)]
            cand = candidates_at_depth(dorder, dpos, border, eff)
            mine = [
                cid[i]
                for i in fuse_weighted_at_depth(
                    cand, dpos, dscore, bpos, bscore, eff, dmax, bmax
                )
            ]
            # the retriever cannot return more than it fetched, so at F < K its
            # output is legitimately shorter -- compare on the shared prefix
            ok = mine[: len(real)] == real and len(real) <= len(mine)
            if F is None:
                full_ok, full_bad = full_ok + ok, full_bad + (not ok)
            else:
                cut_ok, cut_bad = cut_ok + ok, cut_bad + (not ok)
    del embedder, index, emb, bm
    return (
        (full_bad == 0, f"{full_ok} queries reproduce, {full_bad} differ [{combo}, F=n]"),
        (
            cut_bad == 0,
            f"{cut_ok} (query, F) pairs reproduce, {cut_bad} differ "
            f"[{combo}, F in {depths}]",
        ),
    )


def check_against_rrf_sweep(
    same_order: dict[int, int], recall: dict[int, list[float]], n_pairs: int
) -> tuple[bool, str]:
    """S7 -- this run's `rrf` columns must reproduce the published sweep.

    Not a formality: it re-derives another report's headline figures from a
    second run over the same indices, so a mismatch says either the fusion was
    edited or `chunker_compare_full` moved since 2026-08-09. Compared against
    the sweep's own cache rather than against numbers typed from its report.
    """
    if not RRF_RAW.exists():
        return False, f"{RRF_RAW.name} missing -- cannot anchor the rrf columns"
    d = json.loads(RRF_RAW.read_text(encoding="utf-8"))
    if d["n_pairs"] != n_pairs:
        return False, f"sweep has {d['n_pairs']} pairs, this run has {n_pairs}"
    bad = []
    for F in DEPTHS:
        if d["same_order"][str(F)] != same_order[F]:
            bad.append(f"F={F} same-order {d['same_order'][str(F)]} vs {same_order[F]}")
    for F in [*DEPTHS, -1]:
        a = float(np.mean(d["recall"][str(F)]))
        b = float(np.mean(recall[F]))
        if abs(a - b) > 5e-5:
            bad.append(f"F={F} recall {a:.4f} vs {b:.4f}")
    return not bad, (
        f"{len(DEPTHS)+1} depths reproduce the sweep exactly"
        if not bad
        else "; ".join(bad[:4])
    )


def render_from_cache() -> int:
    """Rebuild the report from the persisted raw results, no GPU, no retrieval."""
    d = json.loads(RAW.read_text(encoding="utf-8"))
    by_type = {}
    for key, v in d["recall_by_type"].items():
        m, F, t = key.split("|", 2)
        by_type[(m, int(F), t)] = v
    by_combo = {}
    for key, v in d["recall_combo"].items():
        m, c, F = key.split("|")
        by_combo[(m, c, int(F))] = v
    return render(
        d["n_pairs"],
        d["n_desc"],
        d["combos"],
        d["queries"],
        d["labels"],
        d["chunker_of"],
        {m: {int(F): c for F, c in v.items()} for m, v in d["same_order"].items()},
        {m: {int(F): c for F, c in v.items()} for m, v in d["same_set"].items()},
        {m: unpack_depth_inner(v) for m, v in d["recall"].items()},
        by_type,
        by_combo,
        {int(F): v for F, v in d["zeroing"].items()},
        {m: {int(F): c for F, c in v.items()} for m, v in d["both_arms"].items()},
        d["flat"],
        [tuple(c) for c in d["checks"]],
        sorted({t for _, _, t in by_type}),
    )


def unpack_depth_inner(v: dict[str, list[float]]) -> dict[int, list[float]]:
    return {int(F): x for F, x in v.items()}


def main() -> int:
    if "--render" in sys.argv:
        return render_from_cache()

    # --smoke exercises every path on 2 combos x 8 queries so the replication is
    # verified before committing to the full run. It cannot publish: the
    # self-checks are scoped to the full combo set and it returns without
    # writing.
    smoke = "--smoke" in sys.argv
    t_start = time.time()
    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]
    qrels = {d["query"]: set(d["relevant_resolution_ids"]) for d in raw}
    etype = {d["query"]: d.get("entity_type", "?") for d in raw}
    if smoke:
        queries = queries[:4] + queries[-4:]
    q_tokens = {q: word_tokenize(q) for q in queries}

    checks: list[tuple[str, bool, str]] = []

    with_results = sorted(
        {"__".join(f.stem.split("__")[:4]) for f in HYB_RES.glob("*.json")}
    )
    combos = [
        c
        for c in with_results
        if (INDEX_DIR / c).is_dir() and c not in _EXCLUDED_COMBO_DIRS
    ]
    checks.append((
        "S1 combo set derived from existing index dirs, not a bare glob",
        len(combos) == 36,
        f"{len(combos)} kept of {len(with_results)} with results",
    ))
    if smoke:
        combos = sorted(combos)[:2]

    manifests = {
        c: json.loads((INDEX_DIR / c / "manifest.json").read_text(encoding="utf-8"))
        for c in combos
    }
    labels = {c: _embedder_label(manifests[c]["combo"]) for c in combos}
    chunker_of = {c: manifests[c]["combo"]["chunker"]["type"] for c in combos}

    by_embedder: dict[str, list[str]] = collections.defaultdict(list)
    for c in combos:
        by_embedder[json.dumps(manifests[c]["combo"]["embedder"], sort_keys=True)].append(c)
    qvecs: dict[str, list] = {}
    for spec_json in sorted(by_embedder):
        emb_obj = build_embedder(StrategySpec.model_validate(json.loads(spec_json)))
        qvecs[spec_json] = [emb_obj.embed_query(q) for q in queries]
        del emb_obj
        print(
            f"  encoded {len(queries)} queries for "
            f"{json.loads(spec_json).get('model_name', '?')}",
            file=sys.stderr,
        )

    METHODS = ("rrf", "weighted")
    same_order: dict[str, collections.Counter] = {m: collections.Counter() for m in METHODS}
    same_set: dict[str, collections.Counter] = {m: collections.Counter() for m in METHODS}
    recall: dict[str, dict[int, list[float]]] = {
        m: collections.defaultdict(list) for m in METHODS
    }
    recall_by_type: dict[tuple[str, int, str], list[float]] = collections.defaultdict(list)
    recall_combo: dict[tuple[str, str, int], list[float]] = collections.defaultdict(list)
    # per depth: [dense(+,0,-,mass), bm25(+,0,-,mass)] summed over every pair
    zeroing: dict[int, dict[str, list[float]]] = {
        F: {"dense": [0, 0, 0, 0.0], "bm25": [0, 0, 0, 0.0]} for F in DEPTHS
    }
    # How many of the 10 returned chunks sit inside *both* arms' top-F. The smoke
    # run refuted P2 in the opposite direction -- `weighted` gets *better* as F
    # shrinks -- and a recall column alone cannot say why. This counter can: at
    # F=n "also in the other arm's list" is true of every chunk and therefore
    # carries no information, so the hypothesis is that truncation is what hands
    # `weighted` an intersection signal it structurally lacks, i.e. the thing RRF
    # gets for free from ranks.
    both_arms: dict[str, dict[int, int]] = {
        m: collections.defaultdict(int) for m in METHODS
    }
    # Why that signal is missing: max-normalized cosine is nearly flat across the
    # whole corpus while max-normalized BM25 collapses to exactly 0. Recorded as
    # each arm's normalized score at dense/BM25 rank 10 and at the very bottom.
    flat: dict[str, list[float]] = {k: [] for k in ("d10", "dmin", "b10", "bmin")}

    n_pairs = 0
    dense_ok = dense_bad = bm_ok = bm_bad = hyb_ok = hyb_bad = 0
    bm25_cache: dict[str, tuple[list[str], np.ndarray]] = {}
    cache_misaligned: list[str] = []
    n_chunks: set[int] = set()

    for ci, combo in enumerate(sorted(combos), 1):
        d = INDEX_DIR / combo
        cols = pq.read_table(
            d / "chunks.parquet", columns=["chunk_id", "resolution_id"]
        ).to_pydict()
        cid, rid = cols["chunk_id"], cols["resolution_id"]
        rid_arr = np.array(rid, dtype=object)
        n = len(cid)
        n_chunks.add(n)

        emb = np.load(d / "embeddings.npy")
        row_norms = np.linalg.norm(emb, axis=1)
        qv = qvecs[json.dumps(manifests[combo]["combo"]["embedder"], sort_keys=True)]

        # BM25 reads only the lexical index, a function of loader + chunker, so
        # combos sharing a chunker share it. Cached only after checking the
        # condition that licenses the cache -- identical chunk rows -- because a
        # cached score vector against different rows is silent misalignment.
        # Scores are cached here, not ranks: `weighted` needs the values, and
        # ranks are recovered from them per query for pennies.
        ck = chunker_of[combo]
        bscore_all = None
        if ck in bm25_cache:
            cached_cid, cached = bm25_cache[ck]
            if cached_cid == cid:
                bscore_all = cached
            else:
                cache_misaligned.append(combo)
        if bscore_all is None:
            lex = json.loads((d / "lexical.json").read_text(encoding="utf-8"))
            bm = BM25Okapi(lex)
            bscore_all = np.empty((len(queries), n), dtype=np.float64)
            for j, q in enumerate(queries):
                bscore_all[j] = bm.get_scores(q_tokens[q])
            bm25_cache[ck] = (cid, bscore_all)
            del lex, bm

        ptop_d = persisted_top10(DENSE_RES, combo, "dense")
        ptop_b = persisted_top10(BM25_RES, combo, "bm25")
        ptop_h = persisted_top10(HYB_RES, combo, "hybrid")

        for j, q in enumerate(queries):
            qq = np.asarray(qv[j], dtype=np.float64)
            denom = row_norms * np.linalg.norm(qq)
            dots = emb @ qq
            dscore = np.divide(
                dots, denom, out=np.zeros_like(dots, dtype=np.float64), where=denom > 0
            )
            dorder = np.argsort(-dscore)
            dpos = np.empty(n, dtype=np.int64)
            dpos[dorder] = np.arange(n)
            if q in ptop_d:
                ok = [cid[i] for i in dorder[:K]] == ptop_d[q]
                dense_ok, dense_bad = dense_ok + ok, dense_bad + (not ok)

            bscore = bscore_all[j]
            border = np.argsort(-bscore)
            bpos = np.empty(n, dtype=np.int64)
            bpos[border] = np.arange(n)
            if q in ptop_b:
                ok = [cid[i] for i in border[:K]] == ptop_b[q]
                bm_ok, bm_bad = bm_ok + ok, bm_bad + (not ok)

            dmax, bmax = float(dscore[dorder[0]]), float(bscore[border[0]])
            gold = qrels[q]
            n_pairs += 1

            # F = n: each method's own baseline. The rrf one additionally
            # anchors against the persisted hybrid results.
            full: dict[str, list[int]] = {}
            for m in METHODS:
                if m == "rrf":
                    top = fuse_at_depth(dorder, dpos, border, bpos, n)
                else:
                    cand = candidates_at_depth(dorder, dpos, border, n)
                    top = fuse_weighted_at_depth(
                        cand, dpos, dscore, bpos, bscore, n, dmax, bmax
                    )
                full[m] = list(top)
                r = len(gold & set(rid_arr[full[m]])) / len(gold)
                recall[m][-1].append(r)
                recall_by_type[(m, -1, etype[q])].append(r)
                recall_combo[(m, combo, -1)].append(r)
                both_arms[m][-1] += len(top)  # at F=n every chunk is in both
            if dmax > 0:
                flat["d10"].append(float(dscore[dorder[K - 1]] / dmax))
                flat["dmin"].append(float(dscore[dorder[-1]] / dmax))
            if bmax > 0:
                flat["b10"].append(float(bscore[border[K - 1]] / bmax))
                flat["bmin"].append(float(bscore[border[-1]] / bmax))
            if q in ptop_h:
                ok = [cid[i] for i in full["rrf"]] == ptop_h[q]
                hyb_ok, hyb_bad = hyb_ok + ok, hyb_bad + (not ok)

            for F in DEPTHS:
                cand = candidates_at_depth(dorder, dpos, border, F)
                prof = zeroing_profile(
                    cand, dpos, dscore, bpos, bscore, F, dmax, bmax
                )
                for arm, vals in prof.items():
                    acc = zeroing[F][arm]
                    for x in range(4):
                        acc[x] += vals[x]
                for m in METHODS:
                    if m == "rrf":
                        top = fuse_at_depth(dorder, dpos, border, bpos, F)
                    else:
                        top = fuse_weighted_at_depth(
                            cand, dpos, dscore, bpos, bscore, F, dmax, bmax
                        )
                    same_order[m][F] += int(list(top) == full[m])
                    same_set[m][F] += int(set(top.tolist()) == set(full[m]))
                    both_arms[m][F] += int(
                        ((dpos[top] < F) & (bpos[top] < F)).sum()
                    )
                    r = len(gold & set(rid_arr[list(top)])) / len(gold)
                    recall[m][F].append(r)
                    recall_by_type[(m, F, etype[q])].append(r)
                    recall_combo[(m, combo, F)].append(r)

        del emb
        print(
            f"  [{ci}/{len(combos)}] {combo}  {time.time()-t_start:.0f}s", file=sys.stderr
        )

    checks.append((
        "S2 dense top-10 reproduces the persisted results",
        dense_bad == 0, f"{dense_ok} reproduce, {dense_bad} differ",
    ))
    checks.append((
        "S3 BM25 top-10 reproduces the persisted results",
        bm_bad == 0, f"{bm_ok} reproduce, {bm_bad} differ",
    ))
    checks.append((
        "S3b combos sharing a chunker share chunk rows (licenses the BM25 cache)",
        not cache_misaligned,
        f"{len(bm25_cache)} score matrices built for {len(combos)} combos; "
        f"{len(cache_misaligned)} misaligned",
    ))
    checks.append((
        "S4 the rrf F=n column reproduces the persisted hybrid top-10",
        hyb_bad == 0, f"{hyb_ok} reproduce, {hyb_bad} differ",
    ))

    biggest, smallest = max(n_chunks), min(n_chunks)
    n_desc = f"{smallest:,}" if biggest == smallest else f"{smallest:,}–{biggest:,}"

    print("verifying the weighted fusion against the real retriever ...", file=sys.stderr)
    anchor = "plain__fixed_size__local__ceea7536"
    s5, s6 = verify_weighted_against_retriever(
        queries[:6], q_tokens, anchor, [5, 50, 200, 1000]
    )
    checks.append((
        "S5 weighted at F=n reproduces HybridRetriever(method='weighted') exactly",
        s5[0], s5[1],
    ))
    checks.append((
        "S6 weighted at F<n reproduces "
        "HybridRetriever(method='weighted', fetch_depth=F) exactly",
        s6[0], s6[1],
    ))

    if not smoke:
        ok7, det7 = check_against_rrf_sweep(same_order["rrf"], recall["rrf"], n_pairs)
        checks.append((
            "S7 this run's rrf columns reproduce hybrid_fetch_depth_sweep.md",
            ok7, det7,
        ))

    if smoke:
        for name, ok, detail in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        for F in DEPTHS:
            print(
                f"  F={F:<6} rrf same-order {same_order['rrf'][F]}/{n_pairs} "
                f"recall {np.mean(recall['rrf'][F]):.4f}  |  "
                f"weighted same-order {same_order['weighted'][F]}/{n_pairs} "
                f"recall {np.mean(recall['weighted'][F]):.4f}"
            )
        print(
            f"  F=n     rrf {np.mean(recall['rrf'][-1]):.4f}  |  "
            f"weighted {np.mean(recall['weighted'][-1]):.4f}"
        )
        print(
            f"\nsmoke run ({len(combos)} combos x {len(queries)} queries, "
            f"{time.time()-t_start:.0f}s) -- nothing written"
        )
        return 0 if all(ok for _, ok, _ in checks) else 1

    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps({
        "n_pairs": n_pairs,
        "n_desc": n_desc,
        "combos": sorted(combos),
        "queries": len(queries),
        "depths": DEPTHS,
        "labels": labels,
        "chunker_of": chunker_of,
        "same_order": {m: {str(F): same_order[m][F] for F in DEPTHS} for m in METHODS},
        "same_set": {m: {str(F): same_set[m][F] for F in DEPTHS} for m in METHODS},
        "recall": {
            m: {str(F): recall[m][F] for F in [*DEPTHS, -1]} for m in METHODS
        },
        "recall_by_type": {
            f"{m}|{F}|{t}": v for (m, F, t), v in recall_by_type.items()
        },
        "recall_combo": {f"{m}|{c}|{F}": v for (m, c, F), v in recall_combo.items()},
        "zeroing": {str(F): v for F, v in zeroing.items()},
        "both_arms": {
            m: {str(F): both_arms[m][F] for F in [*DEPTHS, -1]} for m in METHODS
        },
        "flat": {k: float(np.mean(v)) for k, v in flat.items()},
        "checks": [[name, ok, detail] for name, ok, detail in checks],
    }, ensure_ascii=False), encoding="utf-8")

    return render(
        n_pairs, n_desc, sorted(combos), len(queries), labels, chunker_of,
        {m: dict(same_order[m]) for m in METHODS},
        {m: dict(same_set[m]) for m in METHODS},
        {m: dict(recall[m]) for m in METHODS},
        dict(recall_by_type), dict(recall_combo), zeroing,
        {m: dict(both_arms[m]) for m in METHODS},
        {k: float(np.mean(v)) for k, v in flat.items()}, checks,
        sorted({etype[q] for q in queries}),
    )


def render(
    n_pairs, n_desc, combos, n_queries, labels, chunker_of, same_order, same_set,
    recall, recall_by_type, recall_combo, zeroing, both_arms, flat, checks, types,
) -> int:
    base = {m: float(np.mean(recall[m][-1])) for m in ("rrf", "weighted")}
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# `weighted` × `fetch_depth` — วัดคู่ที่ guard ห้ามไว้")
    w()
    w(f"Generated by `tools/eval/hybrid_weighted_fetch_depth.py` · {len(combos)} combo × ")
    w(f"{n_queries} คำถาม = {n_pairs:,} คู่ (combo, คำถาม) · n = {n_desc} chunk "
      f"(แล้วแต่ chunker) · k = {K} · น้ำหนัก {DW}/{BW}")
    w()
    w("`HybridRetriever.__init__` ปฏิเสธ `method=\"weighted\"` คู่กับ `fetch_depth` "
      "ที่ไม่ใช่ `None` มาตั้งแต่ 2026-08-11 — **เป็นการกันไว้ ไม่ใช่คำตัดสิน** "
      "เพราะทั้ง sweep และ routed test วัดแต่ `rrf` และภายใต้ `weighted` chunk ที่หลุด ")
    w("จุดตัดของ arm ไหน จะถูกอ่านคะแนน (ที่ normalize แล้ว) ของ arm นั้นเป็น **0** ")
    w("ไม่ใช่แค่เสียเทอม RRF ไป — หยาบกว่า และไม่เคยมีใครวัด รายงานนี้คือการวัดนั้น")
    w()
    w("**สองคำถามที่ต้องแยกกัน** — `weighted` ไม่เคยถูกให้คะแนนบน Gold set มาก่อนเลย ")
    w("ดังนั้น “ตัดแล้วเสียเท่าไร” ไม่มีความหมายถ้าไม่รู้ว่า “ตั้งต้นอยู่ตรงไหน” "
      "ตารางข้างล่างจึงเทียบแต่ละวิธีกับ F=n **ของตัวมันเอง** และแยกแถวข้ามวิธีออกมาต่างหาก")
    w()
    w("**ข้อสังเกตเชิงโครงสร้าง (จริงโดยนิยาม จึงเขียนไว้เฉย ๆ ไม่ใช่ self-check)** — "
      "ตัวหารของ `weighted` ไม่ขึ้นกับ F: `_normalize` หารด้วยค่าสูงสุดของลิสต์ที่ได้รับ ")
    w("และลิสต์ของแต่ละ arm เรียงจากมากไปน้อยอยู่แล้ว ค่าสูงสุดของ top-F จึงเท่ากับของทั้ง n "
      "ทุก F ≥ 1 — การตัดจึงเปลี่ยน *ว่ามีเทอมไหนอยู่บ้าง* ไม่ใช่เปลี่ยนสเกล")
    w()

    w("## 0. คำทำนายที่ลงทะเบียนก่อนรัน")
    w()
    w("อยู่ในซอร์ส (`PREDICTIONS`) และ commit ก่อนรันจริง ดูที่ *ข้อไหนถูกหักล้าง* ")
    w("ไม่ใช่ดูแค่ข้อที่ถูก")
    w()
    for pid, text in PREDICTIONS:
        w(f"- **{pid}** — {text}")
    w()
    w("**กติกาตัดสินที่ลงทะเบียนไว้** — " + DECISION_RULE)
    w()

    w("## 1. ตัดที่ความลึกไหน อันดับยังเหมือนเดิม (เทียบกับ F=n ของวิธีเดียวกัน)")
    w()
    w("| F | rrf: top-10 เหมือนเป๊ะ | weighted: top-10 เหมือนเป๊ะ | rrf recall@10 | Δ | "
      "weighted recall@10 | Δ |")
    w("|---|---|---|---|---|---|---|")
    for F in DEPTHS:
        cells = []
        for m in ("rrf", "weighted"):
            so = same_order[m][F]
            cells.append(f"{so:,}/{n_pairs:,} ({100*so/n_pairs:.2f}%)")
        rr = float(np.mean(recall["rrf"][F]))
        rw = float(np.mean(recall["weighted"][F]))
        w(f"| {F:,} | {cells[0]} | {cells[1]} | {rr:.4f} | {rr-base['rrf']:+.4f} | "
          f"{rw:.4f} | {rw-base['weighted']:+.4f} |")
    w(f"| n (ทั้งคลัง) | {n_pairs:,}/{n_pairs:,} (100.00%) | "
      f"{n_pairs:,}/{n_pairs:,} (100.00%) | {base['rrf']:.4f} | — | "
      f"{base['weighted']:.4f} | — |")
    w()
    w("`recall@10` เป็น macro เฉลี่ยข้ามทั้ง 36 combo — ใช้ดู *ขนาดความเสียหาย* "
      "ของทั้งตระกูล ไม่ใช่ผลของระบบที่ส่งจริง (ไม่มีระบบไหนใช้ `weighted`)")
    w()

    w("## 2. ข้ามวิธี — `weighted` ตั้งต้นอยู่ตรงไหน")
    w()
    w("| เทียบ | rrf | weighted | ต่าง |")
    w("|---|---|---|---|")
    for F in (-1, 200, 1000):
        lab = "F=n" if F == -1 else f"F={F:,}"
        a = float(np.mean(recall["rrf"][F]))
        b = float(np.mean(recall["weighted"][F]))
        w(f"| recall@10 ที่ {lab} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")
    w()

    w("## 3. การตัดไปแทนที่ *อะไร* ด้วย 0 (กลไก, ไม่ใช่ขนาด)")
    w()
    w("แถวที่หลุดจุดตัดของ arm ไหน จะถูกอ่านเทอมของ arm นั้นเป็น 0 — คอลัมน์นี้บอกว่า ")
    w("ค่าจริงที่ถูกแทนคือบวก (ลดอันดับ) ศูนย์อยู่แล้ว (ฟรี) หรือลบ (**เพิ่ม**อันดับ) "
      "ตัวเลขเป็นผลรวมทุกคู่ (combo, คำถาม)")
    w()
    w("| F | arm | ถูกลด (ค่า>0) | ฟรี (ค่า=0) | ถูกเพิ่ม (ค่า<0) | มวลรวม |Δ| |")
    w("|---|---|---|---|---|---|")
    for F in [f for f in DEPTHS if f in (50, 200, 1000, 5000)]:
        for arm in ("dense", "bm25"):
            pos, zer, neg, mass = zeroing[F][arm]
            tot = pos + zer + neg
            w(f"| {F:,} | {arm} | {pos:,} ({100*pos/tot:.1f}%) | "
              f"{zer:,} ({100*zer/tot:.1f}%) | {neg:,} ({100*neg/tot:.1f}%) | "
              f"{mass:,.1f} |")
    w()

    w("## 3b. การตัดทำอะไรกับ `weighted` — สัญญาณ intersection ที่ทายผิดทาง")
    w()
    w("สมมติฐานตอนเขียนสคริปต์: ที่ F=n ประโยค “chunk นี้อยู่ในลิสต์ของอีก arm ด้วย” "
      "เป็นจริงกับ **ทุก** chunk จึงไม่ได้บอกอะไรเลย — `weighted` ไม่มีสัญญาณ *ตัดกัน* "
      "(intersection) แบบที่ `rrf` ได้มาฟรีจากอันดับ และการตัดคือสิ่งที่สร้างสัญญาณนั้นขึ้นมา "
      "จึงคาดว่าตัดแล้วน่าจะ *ดีขึ้น* (smoke 2 combo × 8 คำถามก็ออกมาแบบนั้นจริง ๆ)")
    w()
    w("**ชุดเต็มหักล้างข้อสรุปนั้น แต่ยืนยันกลไก** — Δ ติดลบทุกความลึก (§1) "
      "สิ่งที่การตัดทำไม่ใช่ *เพิ่ม* สัญญาณ intersection พอประมาณ แต่ทำให้การอยู่ใน "
      "intersection **เกือบชี้ขาด**: 10 อันดับแรกของ `weighted` เป็น chunk ที่อยู่ใน "
      "top-F ทั้งสอง arm 8.25/10 ที่ F=200 และ 9.99/10 ที่ F=1,000 เทียบกับ `rrf` "
      "ที่ 7.41 และ 8.30 — พูดอีกอย่างคือ `weighted` ที่ถูกตัดกลายเป็นตัวจัดอันดับแบบ "
      "*เอาเฉพาะที่ทั้งสอง arm เห็นตรงกัน* และเขี่ยแถวที่ arm เดียวเจอทิ้ง")
    w()
    w("| F | rrf: จาก 10 อันดับแรก อยู่ใน top-F ทั้งสอง arm | weighted: เท่าไร |")
    w("|---|---|---|")
    for F in DEPTHS:
        cells = [
            f"{both_arms[m][F]/n_pairs:.2f} / 10" for m in ("rrf", "weighted")
        ]
        w(f"| {F:,} | {cells[0]} | {cells[1]} |")
    w(f"| n (ทั้งคลัง) | {both_arms['rrf'][-1]/n_pairs:.2f} / 10 (จริงโดยนิยาม) | "
      f"{both_arms['weighted'][-1]/n_pairs:.2f} / 10 (จริงโดยนิยาม) |")
    w()
    w("เหตุผลอยู่ที่สเกลของคะแนน — cosine ที่ normalize ด้วยค่าสูงสุดแล้วยัง **แบนมาก** "
      "ทั้งคลัง")
    w()
    w("| arm | คะแนน normalize ที่อันดับ 10 | ที่อันดับสุดท้าย (n) |")
    w("|---|---|---|")
    w(f"| dense (cosine) | {flat['d10']:.4f} | {flat['dmin']:.4f} |")
    w(f"| BM25 | {flat['b10']:.4f} | {flat['bmin']:.4f} |")
    w()
    w("แถวที่หลุดจุดตัดของ arm หนึ่งจึงเสียคะแนนราว `0.5 × 0.27..0.95` ทันที ขณะที่ภายใต้ "
      "`rrf` แถวที่หลุดที่อันดับ 1,000 เสียแค่ `0.5/1060 ≈ 0.0005` เทียบกับ `0.5/61 ≈ 0.0082` "
      "ที่ arm ที่เหลือให้ — **ภายใต้ `rrf` การถูกตัดคือการถูกลดอันดับ ภายใต้ `weighted` "
      "มันเกือบเท่ากับถูกตัดสิทธิ์** และนั่นคือสิ่งที่ §4 วัดได้: `person` ซึ่ง BM25 เป็น "
      "arm ที่แบกอยู่ ตกไป -0.1965 ที่ F=200 ส่วน `program` ที่ dense แบก กลับ +0.0212")
    w()
    w("**ข้อควรระวังกับตารางข้างบนนี้ — อย่าอ่าน `BM25 = 0.0000` ว่าหางเป็นศูนย์ทั้งหาง** "
      "`BM25Okapi` ยก IDF ที่ติดลบขึ้นเป็น `epsilon × average_idf > 0` คะแนนจึง ≥ 0 เสมอ "
      "และแถว *สุดท้าย* ของอันดับก็เป็น 0 จริง แต่ตารางที่ 3 บอกว่าเทอมที่การตัดแทนด้วย 0 "
      "นั้นเป็น 0 อยู่แล้วเพียง **0.1%** เพราะ chunk จะได้ 0 พอดีก็ต่อเมื่อไม่ตรงกับ "
      "*คำใดเลย* ในคำถาม และคำถามยาว ~20 token ย่อมมีคำพบบ่อยที่แตะเกือบทุก chunk "
      "(คำทำนาย P3 ที่ว่า “ฝั่ง BM25 แทบฟรี” จึงถูกหักล้าง — ที่ F=50 ฝั่ง BM25 "
      "แบกมวลที่ถูกลบไป 73% ของฝั่ง dense: 88,301 ต่อ 121,437)")
    w()

    w("## 4. แยกตามชนิดคำถาม (Δ จาก F=n ของวิธีเดียวกัน)")
    w()
    shown = [F for F in DEPTHS if F in (50, 200, 1000)]
    w("| entity_type | วิธี | F=n | " + " | ".join(f"F={F:,}" for F in shown) + " |")
    w("|---" * (len(shown) + 3) + "|")
    for et in types:
        for m in ("rrf", "weighted"):
            b = float(np.mean(recall_by_type[(m, -1, et)]))
            cells = " | ".join(
                f"{float(np.mean(recall_by_type[(m, F, et)]))-b:+.4f}" for F in shown
            )
            w(f"| {et} | {m} | {b:.4f} | {cells} |")
    w()

    w("## 5. combo ที่โดนหนักที่สุด")
    w()
    w("| F | วิธี | combo ที่แย่ที่สุด | Δ recall@10 |")
    w("|---|---|---|---|")
    for F in shown:
        for m in ("rrf", "weighted"):
            deltas = {
                c: float(
                    np.mean(recall_combo[(m, c, F)]) - np.mean(recall_combo[(m, c, -1)])
                )
                for c in combos
            }
            worst = min(deltas, key=lambda c: deltas[c])
            w(f"| {F:,} | {m} | `{chunker_of[worst]} × {labels[worst]}` | "
              f"{deltas[worst]:+.4f} |")
    w()
    w("**ไม่มีเฟสวัดเวลาในสคริปต์นี้โดยตั้งใจ** — เวลาที่ประหยัดได้มาจากการ *ดึงน้อยลง* "
      "ไม่ใช่จากการ fuse ดังนั้นตัวเลขก็คือของ `hybrid_fetch_depth_sweep.py` "
      "(k=n 1089.5 ms → F=200 417.9 ms) และการรันซ้ำบนเครื่องที่ไม่ว่างจะให้ตัวเลขที่แย่กว่า ")
    w("โดยไม่ได้ตอบอะไรใหม่ — ข้อนี้เป็นการให้เหตุผล ไม่ใช่การวัด จึงไม่ตีพิมพ์เป็นตัวเลขใหม่")
    w()

    w("## 6. คำทำนายข้อไหนถูกหักล้าง และคำตัดสิน")
    w()
    w("ส่วนนี้เขียน *หลัง* เห็นตัวเลข ต่างจาก §0 ที่ freeze ไว้ก่อนรัน")
    w()
    for pid, status, text in OUTCOMES:
        w(f"- **{pid} — {status}** · {text}")
    w()
    w("### คำตัดสินตามกติกาที่ลงทะเบียนไว้")
    w()
    w(VERDICT)
    w()

    w("## self-check")
    w()
    for name, ok, detail in checks:
        w(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    w()

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
    if not all(ok for _, ok, _ in checks):
        print("\nself-check failed; refusing to publish numbers", file=sys.stderr)
        return 1

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
