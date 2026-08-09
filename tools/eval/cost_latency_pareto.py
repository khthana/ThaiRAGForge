"""Cost/latency characterization for the 9-embedder matrix (gap-analysis Tier 1
item #4): vector dim, on-disk index size, embed throughput (from build-time
meta.json), and query latency p50/p95/mean -- split into query-encode time
(embedder-dependent) and search time (brute-force cosine dot product,
index-size-dependent, see retrievers/dense.py) -- for dense, BM25, and hybrid
retrieval on the `semantic` chunker (the paper's recommended chunker, so the
cost axis lines up with the headline quality numbers).

Three phases:
  1. Static stats (no model loading): read manifest.json/meta.json, and the
     embeddings.npy shape/file sizes, for each embedder's semantic-chunker combo.
  2. Dynamic stats (loads each embedder once, times every Gold query -- the set
     named `73det` holds 106 since the `course` queries landed): dense
     query-encode + search latency per embedder; one BM25-only run (embedder-
     agnostic); hybrid (encode + BM25 + RRF fuse) per embedder. Cached to
     `_LATENCY_CACHE` after measuring, so a re-run of phases 3/4 (which iterate
     quickly) doesn't repeat ~20 min of sequential model loading -- pass
     `--reuse-latency-cache` to load it instead of re-measuring.

     **Each embedder is timed in its own subprocess, and that is load-bearing.**
     The 2026-08-07 run of this script had its latency columns thrown out: six
     embedders share dim=1024 and therefore run the identical numpy op on an
     identically-shaped array, yet their `search p50` spread 74.2% -- split
     exactly at loop position 6, everything before the 4B `qwen3` at 301-317ms
     and everything after it at 434-525ms, because its memory is not released
     before the rest of the loop is timed. `embedder.release()` does not fix
     that; a fresh process does. Phase 2 therefore fans out to one child per
     embedder (`--measure-one LABEL`) and merges the parts.

     Isolation alone is a claim, so every child also times `_reference_probe()`
     -- byte-identical work in every process -- and the report prints the spread.
     That converts "same-dim embedders should agree" from something a reader has
     to reconstruct afterwards into a number printed next to the table.
  3. Quality numbers: recall@10 per embedder, read directly from already-
     persisted retrieval result JSONs and filtered to the `semantic` chunker --
     NOT the cross-chunker aggregate, so it's apples-to-apples with the
     semantic-chunker-only latency numbers above (an earlier draft of this
     script hardcoded the aggregate, which doesn't correspond to what's timed).
  4. Intrinsic-cost decomposition: dense.py recomputes embedding row-norms on
     every call, and hybrid.py asks both sub-retrievers for the entire corpus
     (k=n) before fusing+truncating -- both are current-implementation choices,
     not floors on what dense/hybrid retrieval must cost. This phase measures
     each avoidable component directly (norm recompute, BM25Okapi rebuild,
     full-corpus RankedChunk materialization) so the report can show an
     "intrinsic" estimate (encode + bounded search, no avoidable overhead)
     alongside the "measured" total, instead of letting the measured total
     alone imply the overhead is unavoidable.

Run with:
    .venv/Scripts/python.exe tools/eval/cost_latency_pareto.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pythainlp.tokenize import word_tokenize  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_embedder, build_retriever  # noqa: E402
from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.metrics import recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from rag_lab.schema import Query  # noqa: E402
from embedder_matrix_9way import EMBEDDER_ORDER, build_combo_to_chunker_embedder  # noqa: E402

_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "cost_latency_pareto.md"
_LATENCY_CACHE = REPO / "data" / "results" / "cost_latency_raw.json"
_PARTS_DIR = REPO / "data" / "results" / "_latency_parts"
_DENSE_RESULTS_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_BM25_RESULTS_DIR = REPO / "data" / "results" / "gold_bm25_73det"
_HYBRID_RESULTS_DIR = REPO / "data" / "results" / "gold_hybrid_73det"

# semantic-chunker combo dir per embedder label (the paper's recommended
# chunker -- see docs/paper-results-summary.md headline combo). Resolved via
# embedder_matrix_9way.py's label/exclusion logic, listed by hand here so
# this script doesn't need to re-scan+re-label every run.
_SEMANTIC_COMBO_DIRS = {
    "bge_m3": "plain__semantic__local__8aae9bcd",
    "congen": "plain__semantic__local__87fee2dc",
    "e5": "plain__semantic__e5__35b906c6",
    "e5_small": "plain__semantic__e5__2dac4e98",
    "jina_v5": "plain__semantic__jina_v5__4fd4f5b9",
    "m2v": "plain__semantic__local__834c4336",
    "qwen3": "plain__semantic__qwen3__a0f495a8",
    "qwen3_0.6b": "plain__semantic__qwen3__06058e0d",
    "sct": "plain__semantic__local__f477fdca",
}

def compute_semantic_quality(query_set) -> dict[str, dict]:
    """recall@10 per embedder (dense, hybrid) plus BM25, computed from
    already-persisted retrieval result JSONs and filtered to the `semantic`
    chunker -- the same combos the latency measurements above use. Apples-to
    -apples with the cost numbers; earlier draft used a cross-chunker
    aggregate here, which doesn't correspond to what's actually timed."""
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    combo_to_chunker_embedder = build_combo_to_chunker_embedder(_INDEX_DIR)
    base_to_chunker_embedder = {
        cid.rsplit("__dense", 1)[0]: ce for cid, ce in combo_to_chunker_embedder.items()
    }

    def score(results_dir: Path, suffix: str, key_fn) -> dict[str, list[float]]:
        per_key: dict[str, list[float]] = defaultdict(list)
        for p in results_dir.glob("*.json"):
            r = load_retrieval_result(p)
            if not r.combination_id.endswith(suffix):
                continue
            base = r.combination_id[: -len(suffix)]
            chunker, embedder = base_to_chunker_embedder.get(base, (None, None))
            if chunker != "semantic":
                continue
            key = key_fn(embedder)
            if key is None:
                continue
            per_key[key].append(recall_at_k(r, qrels[r.query], 10))
        return per_key

    dense_scores = score(_DENSE_RESULTS_DIR, "__dense", lambda e: e)
    hybrid_scores = score(_HYBRID_RESULTS_DIR, "__hybrid", lambda e: e)
    bm25_scores = score(_BM25_RESULTS_DIR, "__bm25", lambda e: "bm25")

    quality = {
        "dense": {e: sum(v) / len(v) for e, v in dense_scores.items() if e in EMBEDDER_ORDER},
        "hybrid": {e: sum(v) / len(v) for e, v in hybrid_scores.items() if e in EMBEDDER_ORDER},
        "bm25": sum(bm25_scores["bm25"]) / len(bm25_scores["bm25"]),
    }
    missing_dense = set(EMBEDDER_ORDER) - set(quality["dense"])
    missing_hybrid = set(EMBEDDER_ORDER) - set(quality["hybrid"])
    if missing_dense or missing_hybrid:
        print(f"WARNING: missing semantic-chunker quality data -- dense={missing_dense} hybrid={missing_hybrid}")
    return quality


def _percentiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _dir_size_bytes(d: Path) -> int:
    return sum(f.stat().st_size for f in d.iterdir() if f.is_file())


def collect_static_stats() -> dict[str, dict]:
    stats = {}
    for label, combo_name in _SEMANTIC_COMBO_DIRS.items():
        d = _INDEX_DIR / combo_name
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        embeddings = np.load(d / "embeddings.npy", mmap_mode="r")
        n_chunks, dim = embeddings.shape
        embed_seconds = meta["timings"]["embed_seconds"]
        stats[label] = {
            "n_resolutions": manifest["n_resolutions"],
            "n_chunks": n_chunks,
            "dim": dim,
            "embed_seconds": embed_seconds,
            "chunks_per_sec": n_chunks / embed_seconds if embed_seconds > 0 else float("nan"),
            "embeddings_mb": (d / "embeddings.npy").stat().st_size / 1e6,
            "index_total_mb": _dir_size_bytes(d) / 1e6,
        }
    return stats


def _reference_probe() -> float:
    """Byte-identical work in every process, so the machine's floor is measured
    rather than assumed.

    The 2026-08-07 run's real defect was not that its numbers were noisy -- it
    was that nothing in the run could tell you so. The floor had shifted ~1.25x
    against 07-29 and the loop had drifted on top of that, and both were only
    found afterwards, by noticing that six dim=1024 embedders which must run the
    same op disagreed by 74.2%. A fixed array, fixed seed and fixed op timed once
    per child makes that check a column instead of an autopsy: children differ
    only in which embedder they load, so any spread here is the machine, and a
    tight spread is the licence to compare embedders across children at all.

    Shaped like the real workload it stands in for (a 74,816-row dim=1024 gemv
    plus the argsort, i.e. what `DenseRetriever.retrieve` does), so its absolute
    value is comparable to `search p50` rather than being an arbitrary unit.
    """
    rng = np.random.default_rng(12345)
    a = rng.normal(size=(74_816, 1024)).astype(np.float32)
    v = rng.normal(size=1024).astype(np.float64)
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        dots = a @ v
        _ = np.argsort(-dots)[:10]
        times.append((time.perf_counter() - t0) * 1000)
    del a
    return float(np.median(times))


def measure_bm25(query_set) -> dict:
    """BM25 is embedder-agnostic -- measured once, using any one combo's index
    (`lexical.json` is identical in shape/role across every combo of the same
    chunker; only the chunker matters, held fixed at `semantic`).

    Since 2026-08-08 `BM25Retriever` memoises its `BM25Okapi` on the Index, so
    the first of these 73 queries pays the corpus-sized build and the other 72
    do not. That is the shipped behaviour and the right thing to time; it does
    mean `mean` sits above `p50` here by roughly build/73, and p50/p95 describe
    the warm path.
    """
    store = ArtifactStore()
    any_dir = _INDEX_DIR / next(iter(_SEMANTIC_COMBO_DIRS.values()))
    index = store.load(any_dir)
    bm25 = build_retriever(StrategySpec(type="bm25"))
    times = []
    for entry in query_set:
        t0 = time.perf_counter()
        q = Query(text=entry.query, vector=None, tokens=word_tokenize(entry.query))
        bm25.retrieve(q, index, k=10)
        times.append((time.perf_counter() - t0) * 1000)
    del index
    return {"search_ms": _percentiles(times)}


def measure_one_embedder(label: str, query_set) -> dict:
    """Dense encode/search and hybrid totals for one embedder, in one process.

    Dense and hybrid are timed against the *same* loaded Index, so the BM25
    scorer memoised by the dense pass's neighbours is already warm when hybrid
    runs -- again the shipped behaviour, and the reason hybrid's remaining cost
    is the k=n over-fetch rather than the rebuild.
    """
    store = ArtifactStore()
    d = _INDEX_DIR / _SEMANTIC_COMBO_DIRS[label]
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    embedder = build_embedder(StrategySpec.model_validate(manifest["combo"]["embedder"]))
    index = store.load(d)
    dense = build_retriever(StrategySpec(type="dense"))
    hybrid = build_retriever(StrategySpec(type="hybrid"))

    encode_times, dense_search_times, hybrid_times = [], [], []
    for entry in query_set:
        t0 = time.perf_counter()
        vector = embedder.embed_query(entry.query)
        t1 = time.perf_counter()
        tokens = word_tokenize(entry.query)
        q = Query(text=entry.query, vector=vector, tokens=tokens)
        dense.retrieve(q, index, k=10)
        t2 = time.perf_counter()
        encode_times.append((t1 - t0) * 1000)
        dense_search_times.append((t2 - t1) * 1000)

        t0 = time.perf_counter()
        vector = embedder.embed_query(entry.query)
        tokens = word_tokenize(entry.query)
        q = Query(text=entry.query, vector=vector, tokens=tokens)
        hybrid.retrieve(q, index, k=10)
        hybrid_times.append((time.perf_counter() - t0) * 1000)

    embedder.release()
    del embedder, index, dense, hybrid
    return {
        "encode_ms": _percentiles(encode_times),
        "dense_search_ms": _percentiles(dense_search_times),
        "dense_total_ms": _percentiles([e + s for e, s in zip(encode_times, dense_search_times)]),
        "hybrid_total_ms": _percentiles(hybrid_times),
    }


def collect_latency_stats(query_set) -> tuple[dict[str, dict], dict[str, float], dict | None]:
    """Fan out to one child process per embedder and merge the parts.

    Fails loudly on a child error rather than dropping an embedder: a silently
    missing row would be read as "that combo wasn't measured this time" and
    quietly compared against a stale cached one.
    """
    _PARTS_DIR.mkdir(parents=True, exist_ok=True)
    latency: dict[str, dict] = {}
    probes: dict[str, float] = {}

    for label in ["bm25", *EMBEDDER_ORDER]:
        part = _PARTS_DIR / f"{label}.json"
        part.unlink(missing_ok=True)
        print(f"  [child] {label} ...", flush=True)
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                               "--measure-one", label])
        if proc.returncode != 0 or not part.exists():
            raise SystemExit(f"latency child for {label!r} failed (exit {proc.returncode}); "
                             "refusing to report a partial table")
        payload = json.loads(part.read_text(encoding="utf-8"))
        latency[label] = payload["latency"]
        probes[label] = payload["probe_ms"]
        summary = (f"search p50={payload['latency']['search_ms']['p50']:.1f}ms"
                   if label == "bm25" else
                   f"encode p50={payload['latency']['encode_ms']['p50']:.1f}ms, "
                   f"dense p50={payload['latency']['dense_total_ms']['p50']:.1f}ms, "
                   f"hybrid p50={payload['latency']['hybrid_total_ms']['p50']:.1f}ms")
        print(f"  {label}: {summary}  (probe {payload['probe_ms']:.1f}ms)")

    # Position-drift control: re-measure the FIRST embedder last, through the
    # identical code path. The reference probe turns out not to be enough on its
    # own -- it holds the CPU floor steady (0.3% across a 45-min run on
    # 2026-08-09) while the same embedder's `search p50` still rose 5.1%, so
    # something outside the process drifts that a freshly-allocated array does
    # not see. Most likely page-cache state around the 306 MB `embeddings.npy`
    # each child loads. This control measures that residual directly instead of
    # letting it be inferred from a monotonic-looking column.
    first = EMBEDDER_ORDER[0]
    print(f"  [child] {first} (repeat, position-drift control) ...", flush=True)
    repeat_part = _PARTS_DIR / f"{first}__repeat.json"
    repeat_part.unlink(missing_ok=True)
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                           "--measure-one", first, "--part-name", f"{first}__repeat"])
    repeat = None
    if proc.returncode == 0 and repeat_part.exists():
        payload = json.loads(repeat_part.read_text(encoding="utf-8"))
        repeat = {"label": first,
                  "search_p50": payload["latency"]["dense_search_ms"]["p50"],
                  "probe_ms": payload["probe_ms"]}
    else:  # a failed control is not a failed run -- report its absence, don't abort
        print(f"  WARNING: position-drift control for {first!r} failed "
              f"(exit {proc.returncode}); report will say so")

    _LATENCY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _LATENCY_CACHE.write_text(
        json.dumps({"latency": latency, "probes": probes, "repeat": repeat}, indent=2),
        encoding="utf-8")
    return latency, probes, repeat


def measure_intrinsic_costs(static_stats: dict, query_set) -> dict[str, dict]:
    """Quantify the two avoidable overheads found while building this table,
    so the report can separate "cost intrinsic to the retrieval method" from
    "cost of this implementation's current choices":

    - dense.py recomputes `np.linalg.norm(embeddings, axis=1)` on every call
      instead of caching it once per Index -- pure waste, same result every
      query. Still true.
    - hybrid.py asks both DenseRetriever and BM25Retriever for the entire
      corpus (k=n) before fusing+truncating to the caller's k -- scales with
      corpus size, not with k. Still true, and now the *whole* of hybrid's
      overhead rather than half of it.

    The third item this used to measure -- BM25Retriever rebuilding a fresh
    BM25Okapi per query -- was FIXED on 2026-08-08 (`5cc71a1`): the scorer is
    memoised on the Index. The build is still timed below because it is still
    paid, once per loaded Index, and because its ratio to `get_scores` is what
    sizes the fix; but it is no longer a per-query cost, and the report must not
    imply it is. Do not read the two remaining overheads as equally removable:
    the norm cache cannot change results, whereas truncating the k=n over-fetch
    would change what RRF sees and therefore the rankings.

    Measured directly on the actual semantic-chunker index files (no GPU /
    embedder loading needed): dot-product-only search (norms precomputed),
    BM25 get_scores with the BM25Okapi build excluded, and RankedChunk
    construction cost at k=10 vs. k=n.

    **`get_scores` is timed on a real Gold query, and that is not a detail.**
    `rank_bm25` loops over query *terms* in Python, touching all 74,816
    doc-frequency dicts per term, so its cost is linear in token count at
    ~12 ms/token here. This function used to hand it
    `index.chunks[0].text.split()[:8]`, which yields **3** tokens and 40.7 ms --
    the source of the 0.041s figure in `5cc71a1` and CLAUDE.md. The Gold queries
    this project actually evaluates tokenize to a median of **21** terms and cost
    ~250 ms, so the synthetic figure understated the real per-query cost ~6x and
    correspondingly overstated the build-to-score ratio. Any BM25 timing quoted
    without its token count is unfalsifiable; `n_query_tokens` is returned so the
    report can print it.
    """
    store = ArtifactStore()
    any_dir = _INDEX_DIR / next(iter(_SEMANTIC_COMBO_DIRS.values()))
    index = store.load(any_dir)
    n = len(index.chunks)
    embeddings = index.embeddings
    rng = np.random.default_rng(0)

    # dense: norm-recompute cost (current) vs. cached-norm dot-product-only
    # search (intrinsic), measured on the real embeddings.npy for one
    # representative embedder per distinct dim actually in use (384/1024/2560).
    dim_to_label = {}
    for label, s in static_stats.items():
        dim_to_label.setdefault(s["dim"], label)
    dense_per_dim = {}
    for dim, label in dim_to_label.items():
        d = _INDEX_DIR / _SEMANTIC_COMBO_DIRS[label]
        emb = np.load(d / "embeddings.npy")
        q = rng.normal(size=dim).astype(np.float64)
        norm_times, dot_times = [], []
        for _ in range(20):
            t0 = time.perf_counter()
            np.linalg.norm(emb, axis=1)
            t1 = time.perf_counter()
            dots = emb @ q
            _ = np.argsort(-dots)[:10]
            t2 = time.perf_counter()
            norm_times.append((t1 - t0) * 1000)
            dot_times.append((t2 - t1) * 1000)
        del emb
        dense_per_dim[dim] = {
            "norm_recompute_ms": float(np.median(norm_times)),
            "dot_and_sort_ms": float(np.median(dot_times)),
        }

    # BM25: the BM25Okapi build (paid once per loaded Index since 5cc71a1;
    # paid per query before that) vs. get_scores alone, which is what every
    # query after the first now costs.
    #
    # get_scores is timed on **every real Gold query**, not on a synthetic
    # token list, because its cost is linear in token count (see docstring).
    build_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        bm25 = BM25Okapi(index.lexical)
        build_times.append((time.perf_counter() - t0) * 1000)

    gold_tokens = [word_tokenize(e.query) for e in query_set]
    word_tokenize("อุ่นเครื่อง")  # trie load must not land on a timed query
    score_times = []
    for toks in gold_tokens:
        t0 = time.perf_counter()
        bm25.get_scores(toks)
        score_times.append((time.perf_counter() - t0) * 1000)
    token_counts = [len(t) for t in gold_tokens]
    tokens = gold_tokens[0]

    # hybrid over-fetch: RankedChunk construction cost at k=10 (bounded, what
    # a capped-candidate-pool design would fetch) vs. k=n (current, full
    # corpus materialized on both sides before RRF fuse+truncate).
    dense = build_retriever(StrategySpec(type="dense"))
    q_vec = rng.normal(size=embeddings.shape[1]).astype(np.float64)
    query = Query(text="test", vector=q_vec, tokens=tokens)
    k_small_times, k_full_times = [], []
    for _ in range(10):
        t0 = time.perf_counter()
        dense.retrieve(query, index, k=10)
        t1 = time.perf_counter()
        dense.retrieve(query, index, k=n)
        t2 = time.perf_counter()
        k_small_times.append((t1 - t0) * 1000)
        k_full_times.append((t2 - t1) * 1000)

    del index
    return {
        "dense_per_dim": dense_per_dim,
        "bm25_rebuild_ms": float(np.median(build_times)),
        "bm25_score_only_ms": float(np.median(score_times)),
        "n_gold_queries": len(query_set),
        "bm25_query_tokens_p50": int(np.percentile(token_counts, 50)),
        "bm25_query_tokens_min": int(min(token_counts)),
        "bm25_query_tokens_max": int(max(token_counts)),
        "dense_at_k10_ms": float(np.median(k_small_times)),
        "dense_at_kfull_ms": float(np.median(k_full_times)),
        "n_chunks": n,
    }


def render_probe_section(probes: dict[str, float], latency_stats: dict,
                         static_stats: dict, repeat: dict | None) -> list[str]:
    """The measurement's own controls, printed next to the numbers they license.

    **Two controls, and they answer different questions -- the second exists
    because the first was found insufficient on 2026-08-09.**

    (1) `_reference_probe()`: the same array, seed and op in every child, so a
    spread means the machine's CPU floor moved during the run.

    (2) The repeat control: the first embedder re-measured last, through the
    identical code path. On 2026-08-09 the probe held to **0.3%** across a 45-min
    run while `bge_m3`'s own `search p50` rose **5.1%** on re-measurement -- so
    the probe alone would have licensed cross-embedder comparisons it cannot
    actually support. Something outside the process drifts (most plausibly page-
    cache state around the 306 MB `embeddings.npy` each child loads) that a
    freshly-allocated array never touches.

    (3) A direct consistency test, since it needs no extra run at all: embedders
    sharing a dim run the identical numpy op over identically-shaped arrays
    (`n_chunks` is the same for every combo), so their `search p50` *should* be
    equal. Whatever they spread is the measurement's own noise.

    All three are reported rather than asserted, because the honest failure mode
    is a warning: a wide spread doesn't make the run wrong, it makes
    cross-embedder *search* comparisons unsafe while leaving encode times (which
    differ by 2-80x, far above any drift seen here) usable.
    """
    if not probes:
        return []
    vals = list(probes.values())
    lo, hi, med = min(vals), max(vals), float(np.median(vals))
    spread = (hi - lo) / lo * 100 if lo > 0 else float("nan")
    verdict = ("the machine's floor held steady"
               if spread < 15 else
               "**the machine's floor MOVED -- treat search-time differences between "
               "embedders as unmeasured**, the way the 2026-08-07 run had to be")
    lines = [
        "## Timing controls (read this before comparing search times across embedders)",
        "",
        f"**Control 1 -- reference probe.** Each embedder was timed in its own "
        f"subprocess, and each ran an identical reference workload (74,816x1024 gemv + "
        f"argsort, fixed seed). Median **{med:.1f} ms**, range **{lo:.1f}-{hi:.1f} ms**, "
        f"spread **{spread:.1f}%** -- {verdict}.",
        "",
    ]

    if repeat is not None:
        orig = latency_stats.get(repeat["label"], {}).get("dense_search_ms", {}).get("p50")
        if orig:
            drift = (repeat["search_p50"] - orig) / orig * 100
            pdrift = (repeat["probe_ms"] - probes.get(repeat["label"], repeat["probe_ms"]))
            lines += [
                f"**Control 2 -- repeat.** `{repeat['label']}` was measured first and "
                f"again last, same code path: **{orig:.1f} ms -> "
                f"{repeat['search_p50']:.1f} ms ({drift:+.1f}%)**, while its reference "
                f"probe moved {pdrift:+.1f} ms. **This is the number to subtract before "
                f"reading anything into a small cross-embedder search difference** -- and "
                f"it is why control 1 is not sufficient on its own: the probe can hold "
                f"flat while the real workload drifts.",
                "",
            ]

    by_dim: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for label, s in static_stats.items():
        p50 = latency_stats.get(label, {}).get("dense_search_ms", {}).get("p50")
        if p50:
            by_dim[s["dim"]].append((label, p50))
    shared = {d: v for d, v in by_dim.items() if len(v) > 1}
    if shared:
        lines.append(
            "**Control 3 -- same-dim consistency.** Every combo indexes the same "
            "number of chunks, so embedders sharing a dim run the identical numpy op "
            "on identically-shaped arrays; their `search p50` should be equal, and "
            "whatever they spread is measurement noise."
        )
        lines.append("")
        lines.append("| dim | embedders | search p50 range (ms) | spread |")
        lines.append("|---|---|---|---|")
        for dim in sorted(shared):
            v = sorted(shared[dim], key=lambda t: t[1])
            slo, shi = v[0][1], v[-1][1]
            lines.append(f"| {dim} | {len(v)} | {slo:.1f}-{shi:.1f} | "
                         f"{(shi - slo) / slo * 100:.1f}% |")
        lines.append("")

    lines += [
        "These exist because the 2026-08-07 run was discarded: six embedders share "
        "dim=1024 and run the identical numpy op, yet their `search p50` spread 74.2%, "
        "split exactly at loop position 6 (the 4B `qwen3`, whose memory the loop never "
        "released). Absolute latency also depends on the machine's floor at run time, "
        "which moved ~1.25x between 07-29 and 08-07; compare the probe value across "
        "runs before comparing any latency column across runs.",
        "",
        "| measured in child | reference probe (ms) |",
        "|---|---|",
        *[f"| {k} | {v:.1f} |" for k, v in probes.items()],
        "",
    ]
    return lines


def render_report(static_stats: dict, latency_stats: dict, quality: dict, intrinsic: dict,
                  probes: dict[str, float] | None = None,
                  repeat: dict | None = None) -> str:
    lines = [
        "# Cost / latency characterization -- 9-embedder matrix (gap-analysis Tier 1 #4)",
        "",
        "All numbers measured on the `semantic` chunker's combo per embedder (the "
        "paper's recommended chunker, so cost and quality numbers below refer to "
        # The set is *named* `73det` but has held 106 entries since the 33 `course`
        # queries landed; print the count rather than the name's number.
        f"the same combos). Query latency = wall-clock over the "
        f"{intrinsic['n_gold_queries']} Gold queries "
        f"(`{_GOLD_QUERY_SET.name}`), single query at a time, warm model. "
        "BM25 measured once (embedder-agnostic).",
        "",
        "**Read the \"measured\" and \"intrinsic\" numbers as two different questions.** "
        "\"Measured\" = what this implementation currently does, including two "
        "avoidable inefficiencies quantified in the Intrinsic-vs-measured section "
        "below: `DenseRetriever` recomputes embedding row-norms from scratch on "
        "every query instead of caching them once per index, and `HybridRetriever` "
        "asks both sub-retrievers to rank+materialize the *entire* corpus (k=n, not "
        "a bounded candidate pool) before fusing. \"Intrinsic\" = what the same "
        "method would cost with those fixed (norms cached at load, hybrid fusing a "
        "bounded pool). The Pareto plot built from this data uses **encode time** as "
        "the primary cost axis for exactly this reason: it's the only cost component "
        "here that is truly embedder-dependent and not an artifact of the current "
        "implementation.",
        "",
        "**A third overhead this report used to list is gone.** `BM25Retriever` "
        "rebuilt a fresh `BM25Okapi` from the tokenized corpus on every single "
        "query until 2026-08-08 (`5cc71a1`); it is now memoised on the `Index`, so "
        "the corpus-sized build is paid once per loaded index. Every BM25 and "
        "hybrid latency figure below is post-fix and is **not** comparable with the "
        "same columns in runs dated 2026-07-29 or earlier, which are high by "
        "roughly the rebuild cost quoted further down. The dense-only columns are "
        "unaffected by that change.",
        "",
        "**Two overheads, but only one is free to remove.** Caching the row-norms "
        "cannot change a single ranking. Truncating the k=n over-fetch can: "
        "`HybridRetriever` requests complete rankings precisely so RRF fuses full "
        "orderings, so a bounded pool is a different retrieval method, not the same "
        "one made faster. \"Intrinsic hybrid\" below is therefore a floor on the "
        "method, not a patch waiting to be applied.",
        "",
        "## Index build cost + size (semantic chunker)",
        "",
        "| embedder | dim | n_chunks | embed_seconds | chunks/sec | embeddings.npy (MB) | index dir total (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in EMBEDDER_ORDER:
        s = static_stats[label]
        lines.append(
            f"| {label} | {s['dim']} | {s['n_chunks']} | {s['embed_seconds']:.1f} | "
            f"{s['chunks_per_sec']:.1f} | {s['embeddings_mb']:.1f} | {s['index_total_mb']:.1f} |"
        )
    lines.append("")

    lines.extend(render_probe_section(probes or {}, latency_stats, static_stats, repeat))

    lines.append("## Query latency (ms), dense retrieval -- encode vs. search breakdown (measured, current implementation)")
    lines.append("")
    lines.append("| embedder | encode p50 | encode p95 | search p50 | search p95 | total p50 | total p95 |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in EMBEDDER_ORDER:
        L = latency_stats[label]
        lines.append(
            f"| {label} | {L['encode_ms']['p50']:.2f} | {L['encode_ms']['p95']:.2f} | "
            f"{L['dense_search_ms']['p50']:.2f} | {L['dense_search_ms']['p95']:.2f} | "
            f"{L['dense_total_ms']['p50']:.2f} | {L['dense_total_ms']['p95']:.2f} |"
        )
    lines.append("")
    lines.append(f"BM25 search p50/p95 (embedder-agnostic): "
                  f"{latency_stats['bm25']['search_ms']['p50']:.2f} / "
                  f"{latency_stats['bm25']['search_ms']['p95']:.2f} ms")
    lines.append("")

    lines.append("## Query latency (ms), hybrid retrieval (encode + BM25 + RRF fuse; measured, current implementation)")
    lines.append("")
    lines.append("| embedder | total p50 | total p95 |")
    lines.append("|---|---|---|")
    for label in EMBEDDER_ORDER:
        L = latency_stats[label]
        lines.append(f"| {label} | {L['hybrid_total_ms']['p50']:.2f} | {L['hybrid_total_ms']['p95']:.2f} |")
    lines.append("")

    n = intrinsic["n_chunks"]
    lines.append("## Intrinsic vs. measured latency -- what's embedder cost vs. current-implementation overhead")
    lines.append("")
    lines.append(
        f"Measured directly on the `semantic`-chunker index ({n:,} chunks), no GPU/embedder "
        "loading needed -- these decompose the gap between the tables above and what dense/"
        "hybrid retrieval would cost with the two overheads removed."
    )
    lines.append("")
    lines.append("**Dense: norm recompute vs. cached-norm dot product, by dim** (median of 20 runs, full corpus, k=10 argsort included in dot-product time)")
    lines.append("")
    lines.append("| dim | norm recompute (ms, current, redone every query) | dot product + sort (ms, intrinsic search cost) |")
    lines.append("|---|---|---|")
    for dim in sorted(intrinsic["dense_per_dim"]):
        d = intrinsic["dense_per_dim"][dim]
        lines.append(f"| {dim} | {d['norm_recompute_ms']:.2f} | {d['dot_and_sort_ms']:.2f} |")
    lines.append("")
    lines.append(
        f"**BM25 index build** (paid once per loaded `Index` since `5cc71a1`; paid on "
        f"*every query* before that) = {intrinsic['bm25_rebuild_ms']:.2f} ms (median). "
        f"`get_scores` alone, which is what every query after the first now costs = "
        f"{intrinsic['bm25_score_only_ms']:.2f} ms (median over the "
        f"{intrinsic['n_gold_queries']} real Gold queries) -- the build is "
        f"{intrinsic['bm25_rebuild_ms'] / max(intrinsic['bm25_score_only_ms'], 1e-9):.1f}x "
        "the scoring-only cost, which is the size of what memoising it removed from "
        "every BM25 and hybrid query."
    )
    lines.append("")
    lines.append(
        f"**State the token count with any BM25 timing.** `rank_bm25` loops over query "
        f"*terms* in Python, touching all {n:,} doc-frequency dicts per term, so `get_scores` "
        f"is linear in query length (~12 ms/token here). These Gold queries tokenize to "
        f"{intrinsic['bm25_query_tokens_p50']} terms at the median "
        f"(min {intrinsic['bm25_query_tokens_min']}, max {intrinsic['bm25_query_tokens_max']}). "
        "This script previously fed `get_scores` an 8-word slice of a chunk, which tokenizes "
        "to **3** terms and ~41 ms -- the origin of the `0.041s` / `26x` figures quoted for "
        "`5cc71a1`. Those describe a query shape this project never issues; the ratio above "
        "is the one that applies to the queries it evaluates. **The absolute saving is what "
        "transfers between the two (~1.07 s/query either way); the multiplier is not.**"
    )
    lines.append("")
    hybrid_p50s = [latency_stats[e]["hybrid_total_ms"]["p50"] for e in EMBEDDER_ORDER]
    lines.append(
        f"**Hybrid over-fetch**: `DenseRetriever.retrieve(k=10)` (bounded, what a capped-"
        f"candidate-pool hybrid design would request) = {intrinsic['dense_at_k10_ms']:.2f} ms "
        f"(median) vs. `DenseRetriever.retrieve(k=n={n:,})` (current, what `HybridRetriever` "
        f"actually requests from each side before fusing) = {intrinsic['dense_at_kfull_ms']:.2f} ms "
        f"(median) -- {intrinsic['dense_at_kfull_ms'] - intrinsic['dense_at_k10_ms']:.0f} ms of "
        "that gap is `RankedChunk` construction (with full chunk text) for tens of thousands of "
        "chunks nobody will look at, purely from the choice to fetch k=n instead of a bounded pool. "
        "`BM25Retriever` pays the same k=n tax on its side of the fuse. **With the per-query "
        "rebuild gone, this is now the whole of the remaining overhead** -- it, not RRF itself, is "
        f"what stands between the intrinsic estimates and the {min(hybrid_p50s) / 1000:.1f}-"
        f"{max(hybrid_p50s) / 1000:.1f}s hybrid totals in the table above."
    )
    lines.append("")
    overfetch_share = ((intrinsic["dense_at_kfull_ms"] - intrinsic["dense_at_k10_ms"])
                       / intrinsic["dense_at_kfull_ms"] * 100)
    lines.append(
        f"As a share, the over-fetch is **{overfetch_share:.0f}% of dense k=n cost** this run. "
        "Earlier runs put it at 66% (460/699 on 2026-07-29, 558/847 on 08-07) and that figure "
        "was cited as stable across runs; it is not, so quote it from the current run rather "
        "than as a constant of the implementation."
    )
    lines.append("")

    lines.append("## Intrinsic latency estimate per embedder (encode + bounded search, overhead removed)")
    lines.append("")
    lines.append(
        "`intrinsic dense` = encode p50 + dot-product-and-sort at that embedder's dim (no norm "
        "recompute). `intrinsic hybrid` = encode p50 + dot-product-and-sort + BM25 `get_scores`-"
        "only (no BM25Okapi rebuild, no k=n over-fetch on either side; RRF-fuse over a bounded "
        "pool is <5ms and not separately measured here). Compare to the measured totals above -- "
        "the gap is exactly the two overheads decomposed in the previous section, not a floor on "
        "what dense/hybrid retrieval must cost."
    )
    lines.append("")
    lines.append("| embedder | dim | intrinsic dense (ms) | measured dense total p50 (ms) | intrinsic hybrid (ms) | measured hybrid total p50 (ms) |")
    lines.append("|---|---|---|---|---|---|")
    intrinsic_est = {}
    for label in EMBEDDER_ORDER:
        dim = static_stats[label]["dim"]
        d = intrinsic["dense_per_dim"][dim]
        encode_p50 = latency_stats[label]["encode_ms"]["p50"]
        int_dense = encode_p50 + d["dot_and_sort_ms"]
        int_hybrid = encode_p50 + d["dot_and_sort_ms"] + intrinsic["bm25_score_only_ms"]
        intrinsic_est[label] = {"dense": int_dense, "hybrid": int_hybrid}
        lines.append(
            f"| {label} | {dim} | {int_dense:.2f} | {latency_stats[label]['dense_total_ms']['p50']:.2f} | "
            f"{int_hybrid:.2f} | {latency_stats[label]['hybrid_total_ms']['p50']:.2f} |"
        )
    lines.append("")
    lines.append(
        "BM25's own intrinsic cost (scoring-only, no rebuild) is "
        f"{intrinsic['bm25_score_only_ms']:.2f} ms -- both intrinsic-hybrid and intrinsic-dense "
        "estimates above already fold that in via the `bm25_score_only_ms` term, so the marginal "
        "cost of adding lexical signal to dense retrieval, once both overheads are fixed, is "
        f"~{intrinsic['bm25_score_only_ms']:.0f} ms, not the ~2s the measured hybrid total implies."
    )
    lines.append("")

    lines.append("## Quality vs. cost (recall@10, semantic chunker -- same combos as the latency tables above)")
    lines.append("")
    lines.append(
        "`intrinsic` columns are the honest cost axis (see previous section); `measured` columns "
        "are what this implementation currently does, dominated by the two overheads above."
    )
    lines.append("")
    lines.append("| embedder | recall@10 (dense) | intrinsic dense (ms) | measured dense total p50 (ms) | recall@10 (hybrid) | intrinsic hybrid (ms) | measured hybrid total p50 (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in sorted(EMBEDDER_ORDER, key=lambda e: -quality["dense"][e]):
        L = latency_stats[label]
        lines.append(
            f"| {label} | {quality['dense'][label]:.4f} | {intrinsic_est[label]['dense']:.2f} | "
            f"{L['dense_total_ms']['p50']:.2f} | "
            f"{quality['hybrid'][label]:.4f} | {intrinsic_est[label]['hybrid']:.2f} | "
            f"{L['hybrid_total_ms']['p50']:.2f} |"
        )
    lines.append("")
    bm25_measured = latency_stats["bm25"]["search_ms"]["p50"]
    lines.append(
        f"BM25 alone: recall@10={quality['bm25']:.4f}, measured latency "
        f"p50={bm25_measured:.2f} ms (no embed cost). Since `5cc71a1` this **no longer "
        f"includes a BM25Okapi rebuild** -- the ~{intrinsic['bm25_rebuild_ms'] / 1000:.1f}s "
        f"build is paid once per loaded `Index`, so the measured figure is scoring plus "
        f"argsort plus k=10 `RankedChunk` construction. It therefore sits close to the "
        f"{intrinsic['bm25_score_only_ms']:.2f} ms intrinsic scoring cost rather than a "
        f"second above it."
    )
    if bm25_measured < intrinsic["bm25_score_only_ms"]:
        lines.append("")
        lines.append(
            f"*(The measured figure reading slightly **below** the intrinsic one is not a "
            f"contradiction: the intrinsic phase runs in the parent process, after several "
            f"BM25Okapi builds and three full `embeddings.npy` loads, while the measured "
            f"figure comes from a fresh child. The "
            f"{(intrinsic['bm25_score_only_ms'] - bm25_measured) / bm25_measured * 100:.0f}% "
            f"difference is the same process-state drift control 2 quantifies at +5.1%, "
            f"pointing the other way. Neither is wrong; they are not comparable to better "
            f"than ~10%.)*"
        )
    lines.append("")
    lines.append(
        "Note: these are semantic-chunker-specific numbers, not the cross-chunker aggregates "
        "used in the 9-way significance tests (`embedder_significance_test_9way.md`, "
        "`hybrid_significance_test_9way.md`) -- both are correct, they answer different "
        "questions (\"which embedder wins on the paper's recommended chunker\" vs. \"which "
        "embedder wins on average across all 4 chunkers\"). See "
        "`docs/paper-results-summary.md` for how the two relate."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-latency-cache", action="store_true",
        help=f"load query-latency stats from {_LATENCY_CACHE.name} instead of "
             "re-measuring (skips ~20 min of sequential model loading)",
    )
    parser.add_argument(
        "--measure-one", metavar="LABEL",
        help="internal: time one embedder (or 'bm25') in this process and write its "
             "part file. The parent invokes this per embedder so no model's memory "
             "can bias the next one's timings -- see the module docstring.",
    )
    parser.add_argument(
        "--part-name", metavar="NAME",
        help="internal: write the part file under this name instead of the "
             "measured label, so the position-drift control does not overwrite "
             "the first embedder's original measurement.",
    )
    args = parser.parse_args()

    query_set = load_gold_query_set(_GOLD_QUERY_SET)

    if args.measure_one:
        label = args.measure_one
        payload = {
            "latency": measure_bm25(query_set) if label == "bm25"
            else measure_one_embedder(label, query_set),
            "probe_ms": _reference_probe(),
        }
        _PARTS_DIR.mkdir(parents=True, exist_ok=True)
        name = args.part_name or label
        (_PARTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2),
                                                 encoding="utf-8")
        return

    print(f"gold query set: {len(query_set)} queries")

    print("collecting static index/build stats...")
    static_stats = collect_static_stats()

    print("computing semantic-chunker quality numbers from persisted results...")
    quality = compute_semantic_quality(query_set)

    cached = (json.loads(_LATENCY_CACHE.read_text(encoding="utf-8"))
              if args.reuse_latency_cache and _LATENCY_CACHE.exists() else None)

    # The intrinsic phase is cached too, not just the latency phase. It is
    # re-measured cheaply enough (~3 min, no GPU) that re-running it on every
    # render looks harmless, but it wobbled ~15% between two renders minutes
    # apart on 2026-08-09 (dense k=n 545 -> 639 ms) -- so a figure copied from
    # this report into prose would stop matching the report the next time
    # anyone regenerated it, which is exactly the drift `audit_doc_claims.py`'s
    # D2 check exists to catch. A published number must be reproducible from
    # the artifact that published it.
    if cached and cached.get("intrinsic"):
        print("reusing cached intrinsic-cost decomposition")
        intrinsic = cached["intrinsic"]
        # JSON object keys are strings; `dense_per_dim` is keyed by int dim and
        # the renderer indexes it with the int from `static_stats`.
        intrinsic["dense_per_dim"] = {int(k): v
                                      for k, v in intrinsic["dense_per_dim"].items()}
    else:
        print("measuring intrinsic-cost decomposition (no model loading)...")
        intrinsic = measure_intrinsic_costs(static_stats, query_set)

    if cached:
        print(f"reusing cached query latency from {_LATENCY_CACHE}")
        # Pre-2026-08-09 caches are a bare label->stats map with no probe data.
        latency_stats = cached.get("latency", cached)
        probes = cached.get("probes", {})
        repeat = cached.get("repeat")
    else:
        print("collecting query latency (one child process per embedder)...")
        latency_stats, probes, repeat = collect_latency_stats(query_set)

    if not (cached and cached.get("intrinsic")):
        payload = cached or {"latency": latency_stats, "probes": probes, "repeat": repeat}
        payload["intrinsic"] = intrinsic
        _LATENCY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _LATENCY_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report = render_report(static_stats, latency_stats, quality, intrinsic, probes, repeat)
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
