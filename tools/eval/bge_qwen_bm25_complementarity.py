"""Investigates paper-results-summary.md Open item #2: why does bge_m3
overtake qwen3(4B) specifically under hybrid (0.6472 vs 0.6235 aggregate
recall@10), despite the two tying on dense-alone (0.5107 vs 0.5155)?
Working hypothesis, never verified: bge_m3's dense-alone errors are more
complementary with BM25's errors than qwen3's are.

Two proxies computed directly from already-persisted results (no new
retrieval): for each embedder, among the relevant resolutions its own
dense-alone top-10 misses, what fraction does BM25 (same chunker) already
contain in its own top-10 ("rescue rate")? And the union-coverage ceiling
(dense hits UNION BM25 hits) / relevant, an approximate upper bound on what
RRF fusion could recover.

Caveat: the current HybridRetriever over-fetches the full corpus (k=n) from
both sub-retrievers before RRF-fusing, so a chunk doesn't strictly need to
be in a retriever's own top-10 to survive fusion into the final top-10 --
but only top-10 is persisted per eval run (top_k=10 everywhere), so
"present in a retriever's own top-10" is used here as an approximation of
"ranked well enough to plausibly survive RRF fusion," not an exact
reconstruction of the RRF math on full corpus ranks. Directional evidence,
not a proof.

Rescue rate and union coverage below are micro-averaged (pooled counts
across all queries), not a mean of per-query rates -- avoids noise from
queries with very few relevant docs dominating a simple average.

Run with:
    .venv/Scripts/python.exe tools/eval/bge_qwen_bm25_complementarity.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from embedder_matrix_9way import (  # noqa: E402
    _INDEX_DIR,
    _RESULTS_DIR as _DENSE_RESULTS_DIR,
    build_combo_to_chunker_embedder,
)

_BM25_RESULTS_DIR = REPO / "data" / "results" / "gold_bm25_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "bge_qwen_bm25_complementarity.md"

TARGET_EMBEDDERS = ["bge_m3", "qwen3"]


def hit_resolutions(result, relevant: set) -> set:
    return {rc.resolution_id for rc in result.results} & relevant


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / ((vx * vy) ** 0.5)


def main() -> None:
    query_set = load_gold_query_set(str(_GOLD_QUERY_SET))
    qrels = {e.query: set(e.relevant_resolution_ids) for e in query_set}
    queries = list(qrels.keys())

    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)
    chunker_by_base = {k[: -len("__dense")]: v[0] for k, v in combo_ce.items()}
    embedder_by_base = {k[: -len("__dense")]: v[1] for k, v in combo_ce.items()}
    chunkers = sorted(set(chunker_by_base.values()))

    dense_persisted = [load_retrieval_result(p) for p in _DENSE_RESULTS_DIR.glob("*.json")]
    bm25_persisted = [load_retrieval_result(p) for p in _BM25_RESULTS_DIR.glob("*.json")]
    print(f"loaded {len(dense_persisted)} dense, {len(bm25_persisted)} bm25 persisted results")

    dense_by_ceq = {}
    for r in dense_persisted:
        base = r.combination_id[: -len("__dense")] if r.combination_id.endswith("__dense") else None
        if base is None:
            continue
        chunker, embedder = chunker_by_base.get(base), embedder_by_base.get(base)
        if chunker is None or embedder is None:
            continue
        dense_by_ceq[(chunker, embedder, r.query)] = r

    bm25_by_cq = {}
    for r in bm25_persisted:
        base = r.combination_id[: -len("__bm25")] if r.combination_id.endswith("__bm25") else None
        if base is None:
            continue
        chunker = chunker_by_base.get(base)
        if chunker is None:
            continue
        bm25_by_cq[(chunker, r.query)] = r

    stats = {e: defaultdict(lambda: defaultdict(int)) for e in TARGET_EMBEDDERS}
    per_query_recall_bm25 = defaultdict(dict)
    per_query_recall_dense = {e: defaultdict(dict) for e in TARGET_EMBEDDERS}

    for chunker in chunkers:
        for query, relevant in qrels.items():
            bm25_r = bm25_by_cq.get((chunker, query))
            bm25_hits = hit_resolutions(bm25_r, relevant) if bm25_r else set()
            per_query_recall_bm25[chunker][query] = len(bm25_hits) / len(relevant)
            for embedder in TARGET_EMBEDDERS:
                dense_r = dense_by_ceq.get((chunker, embedder, query))
                dense_hits = hit_resolutions(dense_r, relevant) if dense_r else set()
                per_query_recall_dense[embedder][chunker][query] = len(dense_hits) / len(relevant)
                misses = relevant - dense_hits
                rescued = misses & bm25_hits
                union_hits = dense_hits | bm25_hits
                s = stats[embedder][chunker]
                s["dense_misses"] += len(misses)
                s["rescued"] += len(rescued)
                s["union_hits"] += len(union_hits)
                s["relevant_total"] += len(relevant)

    lines = [
        "# Why does bge_m3 overtake qwen3 under hybrid despite tying it dense-alone?",
        "",
        "Investigates paper-results-summary.md Open item #2. See script docstring for the "
        "'rescue rate' / 'union coverage' proxy definitions and their caveats (approximation "
        "from top-10-only persisted results, not exact RRF math). Micro-averaged (pooled counts).",
        "",
        "## Rescue rate: of the relevant resolutions dense-alone misses, what fraction does BM25 (same chunker) already contain?",
        "",
        "| chunker | embedder | dense misses (pooled) | rescued by BM25 | rescue rate |",
        "|---|---|---|---|---|",
    ]
    agg = {e: defaultdict(int) for e in TARGET_EMBEDDERS}
    for chunker in chunkers:
        for embedder in TARGET_EMBEDDERS:
            s = stats[embedder][chunker]
            rate = s["rescued"] / s["dense_misses"] if s["dense_misses"] else float("nan")
            lines.append(f"| {chunker} | {embedder} | {s['dense_misses']} | {s['rescued']} | {rate:.4f} |")
            agg[embedder]["dense_misses"] += s["dense_misses"]
            agg[embedder]["rescued"] += s["rescued"]
    lines.append("")
    lines.append("## Aggregate rescue rate (all 4 chunkers pooled)")
    lines.append("")
    lines.append("| embedder | dense misses | rescued by BM25 | rescue rate |")
    lines.append("|---|---|---|---|")
    for embedder in TARGET_EMBEDDERS:
        a = agg[embedder]
        rate = a["rescued"] / a["dense_misses"] if a["dense_misses"] else float("nan")
        lines.append(f"| {embedder} | {a['dense_misses']} | {a['rescued']} | {rate:.4f} |")
    lines.append("")

    lines.append("## Union coverage: (dense hits UNION BM25 hits) / relevant -- approx. ceiling hybrid could reach")
    lines.append("")
    lines.append("| chunker | embedder | union hits | relevant total | union coverage |")
    lines.append("|---|---|---|---|---|")
    agg_u = {e: defaultdict(int) for e in TARGET_EMBEDDERS}
    for chunker in chunkers:
        for embedder in TARGET_EMBEDDERS:
            s = stats[embedder][chunker]
            cov = s["union_hits"] / s["relevant_total"] if s["relevant_total"] else float("nan")
            lines.append(f"| {chunker} | {embedder} | {s['union_hits']} | {s['relevant_total']} | {cov:.4f} |")
            agg_u[embedder]["union_hits"] += s["union_hits"]
            agg_u[embedder]["relevant_total"] += s["relevant_total"]
    lines.append("")
    lines.append("## Aggregate union coverage")
    lines.append("")
    lines.append("| embedder | union hits | relevant total | union coverage |")
    lines.append("|---|---|---|---|")
    for embedder in TARGET_EMBEDDERS:
        a = agg_u[embedder]
        cov = a["union_hits"] / a["relevant_total"] if a["relevant_total"] else float("nan")
        lines.append(f"| {embedder} | {a['union_hits']} | {a['relevant_total']} | {cov:.4f} |")
    lines.append("")

    lines.append("## Per-query recall correlation with BM25 (Pearson r; lower = more complementary / less redundant)")
    lines.append("")
    lines.append("| chunker | corr(BM25, bge_m3) | corr(BM25, qwen3) |")
    lines.append("|---|---|---|")
    for chunker in chunkers:
        bm25_vals = [per_query_recall_bm25[chunker][q] for q in queries]
        row = []
        for embedder in TARGET_EMBEDDERS:
            dense_vals = [per_query_recall_dense[embedder][chunker][q] for q in queries]
            row.append(pearson(bm25_vals, dense_vals))
        lines.append(f"| {chunker} | {row[0]:.4f} | {row[1]:.4f} |")
    lines.append("")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
