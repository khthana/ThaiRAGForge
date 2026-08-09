"""Is the hybrid `k=n` over-fetch free to remove? Sweep the fetch depth F.

`HybridRetriever.retrieve` asks each arm for the **entire corpus** (`n = 74,816`
chunks) so RRF sees complete rankings, then keeps 10. After the BM25 scorer was
memoised on the Index (2026-08-09) this over-fetch is what is left of the hybrid
overhead: ~1.36s per query, of which the two arms' own scoring is only ~0.51s.
The rest is materialising ~150k `RankedChunk` objects, dict-merging them and
fusing in a Python loop -- work whose only purpose is to give RRF ranks nobody
reads.

**Unlike the memoisation, truncating is NOT free**, which is exactly why it needs
measuring rather than doing. Fusing only each arm's top-F changes the arithmetic:
a chunk in dense's top-F but past BM25's cut loses its BM25 term entirely instead
of contributing `w/(60+rank)`. That can reorder the top-10. So this measures the
damage as a function of F instead of assuming a "deep enough" value:

  * how often the top-10 is **byte-identical** to the k=n ranking (order and set)
  * what recall@10 does -- the number any decision has to be made on
  * what it saves in wall-clock, measured paired in one process

Method. The fusion is replicated in numpy from the shipped retriever rather than
re-run through it, so 11 depths cost about what one retrieval pass costs (the
same trick as `hybrid_alpha_sweep.py`). Three replication traps, all inherited
from `miss_depth_profile.py` where they were found the hard way:

  1. **Per-query gemv, not a batched matmul** -- BLAS accumulates a matrix-matrix
     product differently, and `np.argsort`'s quicksort is unstable, so exact
     score ties reorder and only 98 of 106 queries reproduce.
  2. **Hybrid's tie-break is insertion-ordered, dense-first.** `HybridRetriever`
     builds `fused` by iterating the dense ranking first and settles ties with a
     stable `sorted`. Under truncation that insertion order becomes
     `dense[:F]` followed by the BM25-only remainder **in BM25 rank order** --
     replicated here exactly, because a tie broken the other way would show up as
     a spurious "the ranking changed".
  3. At F = n the truncated formula must collapse to the shipped one. S2/S3/S4
     gate that against the persisted top-10s: if the F=n column is not the
     published ranking, no other column means anything. Those three are not
     sufficient on their own, though -- they exercise only the untruncated path
     and would pass unchanged if `fuse_at_depth` ignored F entirely, so **S5**
     checks the truncated depths against `HybridRetriever(fetch_depth=F)` itself.

Two phases, run separately on purpose. The default phase is the correctness
sweep. `--latency` times the real retrievers instead and must be run alone on an
idle machine -- see feedback_check_benchmark_position_drift: this project has
already published one timing table that was an artifact of loop position.

Read-only: consumes indices, persisted results and the gold set; writes one
report and no index.
"""
from __future__ import annotations

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

from embedder_matrix_9way import _EXCLUDED_COMBO_DIRS, _embedder_label  # noqa: E402
from pythainlp.tokenize import word_tokenize  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder  # noqa: E402

INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
DENSE_RES = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
BM25_RES = REPO / "data" / "results" / "gold_bm25_73det"
HYB_RES = REPO / "data" / "results" / "gold_hybrid_73det"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
OUT = REPO / "data" / "results" / "hybrid_fetch_depth_sweep.md"
RAW = REPO / "data" / "results" / "hybrid_fetch_depth_raw.json"
LAT = REPO / "data" / "results" / "hybrid_fetch_depth_latency.json"

K = 10
RRF_K = 60
DEPTHS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

# The combo the latency phase times: sentence x qwen3_0.6b, the shipped-best
# hybrid combo, so the saving is quoted on the configuration anyone would deploy.
# The run prints the model it actually loaded rather than trusting this name --
# a plausible-looking hash typed from memory resolved to a real directory holding
# a different model, and nothing about the name says which.
LATENCY_COMBO = "plain__sentence__qwen3__ff8f6c49"
LATENCY_EXPECT = "Qwen/Qwen3-Embedding-0.6B"


def persisted_top10(results_dir: Path, combo: str, arm: str) -> dict[str, list[str]]:
    """query -> the top-10 chunk_ids actually written to disk."""
    out: dict[str, list[str]] = {}
    for f in results_dir.glob(f"{combo}__{arm}__*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        out[d["query"]] = [
            r["chunk_id"] for r in sorted(d["results"], key=lambda r: r["rank"])
        ]
    return out


def fuse_at_depth(
    dorder: np.ndarray, dpos: np.ndarray, border: np.ndarray, bpos: np.ndarray, F: int
) -> np.ndarray:
    """Top-K chunk rows from RRF over each arm's top-F only.

    Replicates `HybridRetriever.retrieve` with both arms fetched at F instead of
    n, including its tie-break: `fused` is a dict filled dense-first, so equal
    scores keep dense order, and a BM25-only chunk sits after every dense one at
    the same score. A chunk past an arm's cut contributes nothing from that arm
    -- not a floor value, not `1/(k+F)`, but zero, which is what dropping it from
    the ranking actually does.
    """
    n = len(dpos)
    dsel = dorder[:F]
    in_dense = np.zeros(n, dtype=bool)
    in_dense[dsel] = True
    bsel = border[:F]
    cand = np.concatenate([dsel, bsel[~in_dense[bsel]]])

    dp, bp = dpos[cand], bpos[cand]
    fused = np.where(dp < F, 0.5 / (RRF_K + dp + 1), 0.0) + np.where(
        bp < F, 0.5 / (RRF_K + bp + 1), 0.0
    )
    return cand[np.argsort(-fused, kind="stable")][:K]


def verify_truncation_against_retriever(
    queries: list[str], q_tokens: dict[str, list[str]], combo: str, depths: list[int]
) -> tuple[bool, str]:
    """Check the numpy truncation against `HybridRetriever(fetch_depth=F)` itself.

    S4 anchors only F = n, where the truncation collapses away -- so on its own it
    says nothing about the columns this report exists to publish. The shipped
    retriever grew a `fetch_depth` parameter for the latency phase, which makes
    the intermediate depths checkable against real code instead of against the
    reasoning that produced them. Same discipline as `hybrid_alpha_sweep.py`
    pinning its vectorised fusion at the anchored grid points.
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

    agree = differ = 0
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
        border = np.argsort(-bm.get_scores(q_tokens[q]))
        bpos = np.empty(n, dtype=np.int64)
        bpos[border] = np.arange(n)

        query = Query(text=q, vector=qq, tokens=q_tokens[q])
        for F in depths:
            real = [
                r.chunk_id
                for r in HybridRetriever(fetch_depth=F).retrieve(query, index, K)
            ]
            mine = [cid[i] for i in fuse_at_depth(dorder, dpos, border, bpos, F)]
            # the retriever cannot return more than it fetched, so at F < K its
            # output is legitimately shorter -- compare on the shared prefix
            ok = mine[: len(real)] == real and len(real) <= len(mine)
            agree, differ = agree + ok, differ + (not ok)
    del embedder, index, emb, bm
    return differ == 0, (
        f"{agree} (query, F) pairs reproduce, {differ} differ "
        f"[{combo}, F in {depths}]"
    )


def run_latency() -> int:
    """Time the shipped k=n path against a truncated fetch, paired in one process.

    Both arms are timed against one loaded index inside one process, alternating
    per query, because this project has already been burned by a timing loop
    whose position mattered more than its subject. The BM25 scorer is warmed
    first so its one-off ~1.0s build lands in neither arm's numbers.
    """
    from rag_lab.io.artifact_store import ArtifactStore
    from rag_lab.retrievers.bm25 import BM25Retriever
    from rag_lab.retrievers.hybrid import HybridRetriever
    from rag_lab.schema import Query

    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    queries = [d["query"] for d in raw]

    d = INDEX_DIR / LATENCY_COMBO
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    model_name = manifest["combo"]["embedder"]["params"].get("model_name")
    if model_name != LATENCY_EXPECT:
        print(f"{LATENCY_COMBO} holds {model_name!r}, expected {LATENCY_EXPECT!r}",
              file=sys.stderr)
        return 1
    print(f"loading {LATENCY_COMBO} ({model_name}) ...", file=sys.stderr)
    index = ArtifactStore().load(d)
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    prepared = [
        Query(text=q, vector=embedder.embed_query(q), tokens=word_tokenize(q))
        for q in queries
    ]
    del embedder
    BM25Retriever()._scorer(index)  # warm the memo; its build is not under test

    arms = [("k=n", HybridRetriever())] + [
        (f"F={F}", HybridRetriever(fetch_depth=F)) for F in (1000, 200)
    ]
    timings: dict[str, list[float]] = collections.defaultdict(list)
    mismatched = 0
    for i, q in enumerate(prepared):
        for label, retr in arms:
            t0 = time.perf_counter()
            ranked = retr.retrieve(q, index, K)
            timings[label].append(time.perf_counter() - t0)
            if label == "k=n":
                baseline = [r.chunk_id for r in ranked]
            elif label == "F=1000" and [r.chunk_id for r in ranked] != baseline:
                mismatched += 1
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(prepared)}", file=sys.stderr)

    stats = {}
    for label, ts in timings.items():
        s = sorted(ts)
        stats[label] = {
            "p50": statistics.median(s) * 1000,
            "p95": s[int(0.95 * len(s))] * 1000,
            "mean": statistics.mean(s) * 1000,
        }
    LAT.write_text(json.dumps({
        "combo": LATENCY_COMBO, "model": model_name, "n_queries": len(prepared),
        "mismatched_1000": mismatched, "stats": stats,
    }, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"{len(prepared)} queries, paired in one process, {LATENCY_COMBO}")
    print(f"{'arm':<10} {'p50':>10} {'p95':>10} {'mean':>10}")
    for label, s in stats.items():
        print(f"{label:<10} {s['p50']:>9.1f}ms {s['p95']:>9.1f}ms {s['mean']:>9.1f}ms")
    print(f"\nF=1000 top-10 differs from k=n on {mismatched} of {len(prepared)} queries")
    print(f"wrote {LAT.relative_to(REPO)} -- re-render the report to include it")
    return 0


def render_from_cache() -> int:
    """Rebuild the report from the persisted raw results, no GPU and no retrieval."""
    d = json.loads(RAW.read_text(encoding="utf-8"))
    recall = {int(F): v for F, v in d["recall"].items()}
    by_type: dict[tuple[int, str], list[float]] = {}
    for key, v in d["recall_by_type"].items():
        F, t = key.rsplit("|", 1)
        by_type[(int(F), t)] = v
    by_combo: dict[tuple[str, int], list[float]] = {}
    for key, v in d["recall_combo"].items():
        c, F = key.rsplit("|", 1)
        by_combo[(c, int(F))] = v
    types = sorted({t for _, t in by_type})
    return render(
        d["n_pairs"], d["n_desc"], d["combos"], d["queries"], d["labels"],
        d["chunker_of"], {int(F): v for F, v in d["same_order"].items()},
        {int(F): v for F, v in d["same_set"].items()}, recall, by_type, by_combo,
        [tuple(c) for c in d["checks"]], types,
    )


def main() -> int:
    if "--latency" in sys.argv:
        return run_latency()
    if "--render" in sys.argv:
        return render_from_cache()

    # --smoke exercises every path on 2 combos x 8 queries (~40s) so the
    # replication is verified before committing to the full run. It cannot
    # publish: the self-checks are scoped to the full combo set and it returns
    # without writing.
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

    with_results = sorted({"__".join(f.stem.split("__")[:4]) for f in HYB_RES.glob("*.json")})
    combos = [
        c for c in with_results
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
        print(f"  encoded {len(queries)} queries for "
              f"{json.loads(spec_json).get('model_name', '?')}", file=sys.stderr)

    # per depth: ordered/set agreement with k=n, and macro recall@10
    same_order: dict[int, int] = collections.Counter()
    same_set: dict[int, int] = collections.Counter()
    recall: dict[int, list[float]] = collections.defaultdict(list)
    recall_by_type: dict[tuple[int, str], list[float]] = collections.defaultdict(list)
    # per (combo, depth) recall, so the report can name the worst-hit combo
    recall_combo: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
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
        # cached rank vector against different rows is silent misalignment.
        ck = chunker_of[combo]
        bpos_all = None
        if ck in bm25_cache:
            cached_cid, cached = bm25_cache[ck]
            if cached_cid == cid:
                bpos_all = cached
            else:
                cache_misaligned.append(combo)
        if bpos_all is None:
            lex = json.loads((d / "lexical.json").read_text(encoding="utf-8"))
            bm = BM25Okapi(lex)
            bpos_all = np.empty((len(queries), n), dtype=np.int64)
            for j, q in enumerate(queries):
                border = np.argsort(-bm.get_scores(q_tokens[q]))
                bpos_all[j][border] = np.arange(n)
            bm25_cache[ck] = (cid, bpos_all)
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

            bpos = bpos_all[j]
            border = np.argsort(bpos)
            if q in ptop_b:
                ok = [cid[i] for i in border[:K]] == ptop_b[q]
                bm_ok, bm_bad = bm_ok + ok, bm_bad + (not ok)

            # F = n: the shipped path. Anchors every other column.
            full_top = fuse_at_depth(dorder, dpos, border, bpos, n)
            if q in ptop_h:
                ok = [cid[i] for i in full_top] == ptop_h[q]
                hyb_ok, hyb_bad = hyb_ok + ok, hyb_bad + (not ok)

            # depth key -1 == the shipped k=n path, the baseline every column
            # below is a delta against.
            gold = qrels[q]
            full_ids = list(full_top)
            r_full = len(gold & set(rid_arr[full_ids])) / len(gold)
            recall[-1].append(r_full)
            recall_by_type[(-1, etype[q])].append(r_full)
            recall_combo[(combo, -1)].append(r_full)
            n_pairs += 1

            for F in DEPTHS:
                top = fuse_at_depth(dorder, dpos, border, bpos, F)
                same_order[F] += int(list(top) == full_ids)
                same_set[F] += int(set(top.tolist()) == set(full_ids))
                r = len(gold & set(rid_arr[list(top)])) / len(gold)
                recall[F].append(r)
                recall_by_type[(F, etype[q])].append(r)
                recall_combo[(combo, F)].append(r)

        del emb
        print(f"  [{ci}/{len(combos)}] {combo}  {time.time()-t_start:.0f}s", file=sys.stderr)

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
        f"{len(bm25_cache)} lexical indices built for {len(combos)} combos; "
        f"{len(cache_misaligned)} misaligned",
    ))
    checks.append((
        "S4 the F=n column reproduces the persisted hybrid top-10",
        hyb_bad == 0, f"{hyb_ok} reproduce, {hyb_bad} differ",
    ))
    # n is per-combo (the chunker decides how many chunks the corpus becomes), so
    # "F = n" in the tables below is each combo's own corpus size, not one number.
    # Quoting a single n would understate the deepest cut on offer. This is
    # reported as a fact rather than as a self-check: "F >= n cannot truncate" is
    # true by construction, and a check that cannot fail is a vacuous PASS
    # dressed up as evidence.
    biggest, smallest = max(n_chunks), min(n_chunks)
    n_desc = f"{smallest:,}" if biggest == smallest else f"{smallest:,}–{biggest:,}"

    # S5 is the check that makes the truncated columns citable at all. S2-S4 only
    # ever exercise the untruncated path, where the whole mechanism under test is
    # inert -- they would pass identically if fuse_at_depth ignored F.
    print("verifying truncation against the real retriever ...", file=sys.stderr)
    anchor = "plain__fixed_size__local__ceea7536"
    ok6, detail6 = verify_truncation_against_retriever(
        queries[:6], q_tokens, anchor, [5, 50, 200, 1000]
    )
    checks.append((
        "S5 truncated fusion reproduces HybridRetriever(fetch_depth=F) exactly", ok6, detail6,
    ))

    if smoke:
        for name, ok, detail in checks:
            print(f"[{'PASS' if ok else 'FAIL'}] {name} -- {detail}")
        for F in DEPTHS:
            print(f"  F={F:<6} same-order {same_order[F]}/{n_pairs}  "
                  f"recall {np.mean(recall[F]):.4f} vs {np.mean(recall[-1]):.4f}")
        print(f"\nsmoke run ({len(combos)} combos x {len(queries)} queries, "
              f"{time.time()-t_start:.0f}s) -- nothing written")
        return 0 if all(ok for _, ok, _ in checks) else 1

    # Persist everything the report is rendered from, so the published figures
    # can be reproduced without a 25-minute GPU pass -- the same reason
    # cost_latency_pareto.py caches its intrinsic-cost phase. `--render` rebuilds
    # the report from this file alone; if a number in the prose cannot be
    # reproduced from it, the number is wrong.
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps({
        "n_pairs": n_pairs,
        "n_desc": n_desc,
        "combos": sorted(combos),
        "queries": len(queries),
        "depths": DEPTHS,
        "labels": labels,
        "chunker_of": chunker_of,
        "same_order": {str(F): same_order[F] for F in DEPTHS},
        "same_set": {str(F): same_set[F] for F in DEPTHS},
        "recall": {str(F): recall[F] for F in [*DEPTHS, -1]},
        "recall_by_type": {
            f"{F}|{t}": v for (F, t), v in recall_by_type.items()
        },
        "recall_combo": {f"{c}|{F}": v for (c, F), v in recall_combo.items()},
        "checks": [[name, ok, detail] for name, ok, detail in checks],
    }, ensure_ascii=False), encoding="utf-8")

    return render(
        n_pairs, n_desc, sorted(combos), len(queries), labels, chunker_of,
        same_order, same_set, recall, recall_by_type, recall_combo, checks,
        sorted({etype[q] for q in queries}),
    )


def render(
    n_pairs, n_desc, combos, n_queries, labels, chunker_of, same_order, same_set,
    recall, recall_by_type, recall_combo, checks, types,
) -> int:
    base = float(np.mean(recall[-1]))
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# ตัดการดึงเกินของ hybrid — กวาดค่าความลึก F")
    w()
    w(f"Generated by `tools/eval/hybrid_fetch_depth_sweep.py` · {len(combos)} combo × ")
    w(f"{n_queries} คำถาม = {n_pairs:,} คู่ (combo, คำถาม) · n = {n_desc} chunk "
      f"(แล้วแต่ chunker) · k = {K}")
    w()
    w("**คำถามที่ตอบ** — `HybridRetriever` ขอ ranking ทั้งคลังจากทั้งสอง arm แล้วเก็บแค่ 10 ")
    w("หลังจาก memoise ตัว BM25 scorer แล้ว การดึงเกินนี้คือสิ่งที่เหลืออยู่ของ overhead ")
    w("(~1.36 วิ/query โดยที่การให้คะแนนของทั้งสอง arm รวมกันเป็นแค่ ~0.51 วิ) ")
    w("**แต่การตัดไม่ฟรีเหมือนตอน memoise** — chunk ที่อยู่ใน top-F ของ dense แต่หลุด ")
    w("จุดตัดของ BM25 จะเสียเทอม BM25 ไปทั้งก้อน ไม่ใช่ได้ค่าน้อย ๆ ดังนั้นอันดับเปลี่ยนได้จริง")
    w()

    w("## 1. ตัดที่ความลึกไหน อันดับยังเหมือนเดิม")
    w()
    w("| F | top-10 เหมือนเป๊ะ (ทั้งลำดับ) | เหมือนเป็นเซ็ต | recall@10 | Δ จาก k=n |")
    w("|---|---|---|---|---|")
    for F in DEPTHS:
        r = float(np.mean(recall[F]))
        w(f"| {F:,} | {same_order[F]:,}/{n_pairs:,} "
          f"({100*same_order[F]/n_pairs:.2f}%) | "
          f"{same_set[F]:,}/{n_pairs:,} ({100*same_set[F]/n_pairs:.2f}%) | "
          f"{r:.4f} | {r-base:+.4f} |")
    w(f"| n (ทั้งคลัง) | {n_pairs:,}/{n_pairs:,} (100.00%) | "
      f"{n_pairs:,}/{n_pairs:,} (100.00%) | {base:.4f} | — |")
    w()
    w("`recall@10` เป็น macro เฉลี่ยข้ามทั้ง 36 combo — ตัวเลขนี้จึงไม่ใช่ recall ของระบบที่ ")
    w("ส่งจริง แต่เป็นค่าเฉลี่ยของทั้งตระกูล ใช้ดู *ขนาดความเสียหาย* ไม่ใช่ใช้อ้างเป็นผลระบบ")
    w()

    # ---- 2. per entity_type at the depths that matter --------------------
    w("## 2. แยกตามชนิดคำถาม")
    w()
    shown = [F for F in DEPTHS if F in (50, 200, 1000, 5000)]
    w("| entity_type | k=n | " + " | ".join(f"F={F:,}" for F in shown) + " |")
    w("|---" * (len(shown) + 2) + "|")
    for et in types:
        b = float(np.mean(recall_by_type[(-1, et)]))
        cells = " | ".join(
            f"{float(np.mean(recall_by_type[(F, et)]))-b:+.4f}" for F in shown
        )
        w(f"| {et} | {b:.4f} | {cells} |")
    w()

    # ---- 3. the worst-hit combo ------------------------------------------
    w("## 3. combo ที่โดนหนักที่สุด (ตัวชี้ว่าความเสียหายกระจุกตรงไหน)")
    w()
    w("| F | combo ที่แย่ที่สุด | Δ recall@10 ของ combo นั้น |")
    w("|---|---|---|")
    for F in shown:
        deltas = {
            c: float(np.mean(recall_combo[(c, F)]) - np.mean(recall_combo[(c, -1)]))
            for c in combos
        }
        worst = min(deltas, key=lambda c: deltas[c])
        w(f"| {F:,} | `{chunker_of[worst]} × {labels[worst]}` | {deltas[worst]:+.4f} |")
    w()

    if LAT.exists():
        lat = json.loads(LAT.read_text(encoding="utf-8"))
        s = lat["stats"]
        w("## 4. ประหยัดเวลาได้เท่าไร")
        w()
        w(f"`{lat['combo']}` ({lat['model']}) · {lat['n_queries']} คำถาม · วัดแบบจับคู่ ")
        w("ในโปรเซสเดียว บน index ที่โหลดครั้งเดียว และ warm ตัว BM25 scorer ไว้ก่อน ")
        w("(การ build ครั้งเดียวของมันจึงไม่ตกอยู่ในฝั่งใดฝั่งหนึ่ง)")
        w()
        w("| arm | p50 | p95 | mean | ประหยัดจาก k=n (p50) |")
        w("|---|---|---|---|---|")
        for label in ("k=n", "F=1000", "F=200"):
            if label not in s:
                continue
            saved = (
                "—" if label == "k=n"
                else f"−{(s['k=n']['p50'] - s[label]['p50'])/1000:.3f} วิ "
                     f"({s['k=n']['p50']/s[label]['p50']:.2f}x)"
            )
            w(f"| {label} | {s[label]['p50']:.1f} ms | {s[label]['p95']:.1f} ms | "
              f"{s[label]['mean']:.1f} ms | {saved} |")
        w()
        w(f"ที่ F=1000 top-10 ต่างจาก k=n **{lat['mismatched_1000']} จาก "
          f"{lat['n_queries']} คำถาม** บน combo นี้ — ตัวเลขนี้เป็นของ combo เดียว ")
        w("ไม่ใช่ค่าเฉลี่ยในตารางที่ 1")
        w()
        w("**เครื่องต้องว่างตอนวัด** — โปรเจกต์นี้เคยตีพิมพ์ตารางเวลาที่กลายเป็นผลของ ")
        w("ตำแหน่งใน loop มาแล้ว การวัดนี้จึงสลับ arm ทีละคำถามแทนที่จะวัดทีละ arm จนจบ")
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
