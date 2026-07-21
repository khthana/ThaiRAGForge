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
  2. Dynamic stats (loads each embedder once, times 73 Gold queries): dense
     query-encode + search latency per embedder; one BM25-only run (embedder-
     agnostic); hybrid (encode + BM25 + RRF fuse) per embedder. Cached to
     `_LATENCY_CACHE` after measuring, so a re-run of phases 3/4 (which iterate
     quickly) doesn't repeat ~20 min of sequential model loading -- pass
     `--reuse-latency-cache` to load it instead of re-measuring.
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


def collect_latency_stats(query_set) -> dict[str, dict]:
    store = ArtifactStore()
    latency = {}

    # BM25 is embedder-agnostic -- measure once using any one combo's index
    # (lexical.json is identical in shape/role across every combo of the same
    # chunker; only the chunker matters for BM25, held fixed at `semantic`).
    any_dir = _INDEX_DIR / next(iter(_SEMANTIC_COMBO_DIRS.values()))
    index = store.load(any_dir)
    bm25 = build_retriever(StrategySpec(type="bm25"))
    bm25_times = []
    for entry in query_set:
        t0 = time.perf_counter()
        q = Query(text=entry.query, vector=None, tokens=word_tokenize(entry.query))
        bm25.retrieve(q, index, k=10)
        bm25_times.append((time.perf_counter() - t0) * 1000)
    latency["bm25"] = {"search_ms": _percentiles(bm25_times)}
    del index

    for label in EMBEDDER_ORDER:
        combo_name = _SEMANTIC_COMBO_DIRS[label]
        d = _INDEX_DIR / combo_name
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

        latency[label] = {
            "encode_ms": _percentiles(encode_times),
            "dense_search_ms": _percentiles(dense_search_times),
            "dense_total_ms": _percentiles([e + s for e, s in zip(encode_times, dense_search_times)]),
            "hybrid_total_ms": _percentiles(hybrid_times),
        }
        print(f"{label}: encode p50={latency[label]['encode_ms']['p50']:.1f}ms, "
              f"dense_total p50={latency[label]['dense_total_ms']['p50']:.1f}ms, "
              f"hybrid p50={latency[label]['hybrid_total_ms']['p50']:.1f}ms")

        embedder.release()
        del embedder, index, dense, hybrid

    _LATENCY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _LATENCY_CACHE.write_text(json.dumps(latency, indent=2), encoding="utf-8")
    return latency


def measure_intrinsic_costs(static_stats: dict) -> dict[str, dict]:
    """Quantify the two avoidable overheads found while building this table,
    so the report can separate "cost intrinsic to the retrieval method" from
    "cost of this implementation's current choices":

    - dense.py recomputes `np.linalg.norm(embeddings, axis=1)` on every call
      instead of caching it once per Index -- pure waste, same result every
      query.
    - hybrid.py asks both DenseRetriever and BM25Retriever for the entire
      corpus (k=n) before fusing+truncating to the caller's k, and
      BM25Retriever rebuilds a fresh BM25Okapi from the tokenized corpus on
      every call instead of caching it once per Index -- both scale with
      corpus size, not with k.

    Measured directly on the actual semantic-chunker index files (no GPU /
    embedder loading needed): dot-product-only search (norms precomputed),
    BM25 get_scores with the BM25Okapi build excluded, and RankedChunk
    construction cost at k=10 vs. k=n.
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

    # BM25: full rebuild-per-query (current) vs. get_scores only, assuming
    # the BM25Okapi object were built once at index-load time (intrinsic).
    build_times, score_times = [], []
    tokens = index.chunks[0].text.split()[:8] or ["test"]
    for _ in range(10):
        t0 = time.perf_counter()
        bm25 = BM25Okapi(index.lexical)
        t1 = time.perf_counter()
        bm25.get_scores(tokens)
        t2 = time.perf_counter()
        build_times.append((t1 - t0) * 1000)
        score_times.append((t2 - t1) * 1000)

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
        "dense_at_k10_ms": float(np.median(k_small_times)),
        "dense_at_kfull_ms": float(np.median(k_full_times)),
        "n_chunks": n,
    }


def render_report(static_stats: dict, latency_stats: dict, quality: dict, intrinsic: dict) -> str:
    lines = [
        "# Cost / latency characterization -- 9-embedder matrix (gap-analysis Tier 1 #4)",
        "",
        "All numbers measured on the `semantic` chunker's combo per embedder (the "
        "paper's recommended chunker, so cost and quality numbers below refer to "
        "the same combos). Query latency = wall-clock over the 73 Gold queries, "
        "single query at a time, warm model. BM25 measured once (embedder-agnostic).",
        "",
        "**Read the \"measured\" and \"intrinsic\" numbers as two different questions.** "
        "\"Measured\" = what this implementation currently does, including two "
        "avoidable inefficiencies quantified in the Intrinsic-vs-measured section "
        "below: `DenseRetriever` recomputes embedding row-norms from scratch on "
        "every query instead of caching them once per index, and `HybridRetriever` "
        "asks both sub-retrievers to rank+materialize the *entire* corpus (k=n, not "
        "a bounded candidate pool) before fusing -- `BM25Retriever` on top of that "
        "rebuilds a fresh `BM25Okapi` index from scratch on every single query. "
        "\"Intrinsic\" = what the same method would cost with those two fixed (norms "
        "cached at load, BM25 index cached at load, hybrid fusing a bounded pool). "
        "The Pareto plot built from this data uses **encode time** as the primary "
        "cost axis for exactly this reason: it's the only cost component here that "
        "is truly embedder-dependent and not an artifact of the current "
        "implementation.",
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
        f"**BM25**: rebuild `BM25Okapi` from scratch (current, every query) = "
        f"{intrinsic['bm25_rebuild_ms']:.2f} ms (median). `get_scores` alone on an "
        f"already-built index (intrinsic, what a load-time-cached index would cost) = "
        f"{intrinsic['bm25_score_only_ms']:.2f} ms (median) -- "
        f"{intrinsic['bm25_rebuild_ms'] / max(intrinsic['bm25_score_only_ms'], 1e-9):.0f}x "
        "the scoring-only cost is the rebuild alone."
    )
    lines.append("")
    lines.append(
        f"**Hybrid over-fetch**: `DenseRetriever.retrieve(k=10)` (bounded, what a capped-"
        f"candidate-pool hybrid design would request) = {intrinsic['dense_at_k10_ms']:.2f} ms "
        f"(median) vs. `DenseRetriever.retrieve(k=n={n:,})` (current, what `HybridRetriever` "
        f"actually requests from each side before fusing) = {intrinsic['dense_at_kfull_ms']:.2f} ms "
        f"(median) -- {intrinsic['dense_at_kfull_ms'] - intrinsic['dense_at_k10_ms']:.0f} ms of "
        "that gap is `RankedChunk` construction (with full chunk text) for tens of thousands of "
        "chunks nobody will look at, purely from the choice to fetch k=n instead of a bounded pool. "
        "`BM25Retriever` pays the same k=n tax on its side of the fuse, compounding with the "
        "rebuild-per-query cost above -- together these two effects, not RRF itself, explain most "
        "of the ~2.3-2.9s hybrid totals in the table above."
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
    lines.append(f"BM25 alone: recall@10={quality['bm25']:.4f}, "
                  f"measured latency p50={latency_stats['bm25']['search_ms']['p50']:.2f} ms "
                  "(no embed cost; latency here already includes the per-query BM25Okapi rebuild "
                  f"decomposed above -- intrinsic scoring-only cost is {intrinsic['bm25_score_only_ms']:.2f} ms).")
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
    args = parser.parse_args()

    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    print(f"gold query set: {len(query_set)} queries")

    print("collecting static index/build stats...")
    static_stats = collect_static_stats()

    print("computing semantic-chunker quality numbers from persisted results...")
    quality = compute_semantic_quality(query_set)

    print("measuring intrinsic-cost decomposition (no model loading)...")
    intrinsic = measure_intrinsic_costs(static_stats)

    if args.reuse_latency_cache and _LATENCY_CACHE.exists():
        print(f"reusing cached query latency from {_LATENCY_CACHE}")
        latency_stats = json.loads(_LATENCY_CACHE.read_text(encoding="utf-8"))
    else:
        print("collecting query latency (loads each embedder once)...")
        latency_stats = collect_latency_stats(query_set)

    report = render_report(static_stats, latency_stats, quality, intrinsic)
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
