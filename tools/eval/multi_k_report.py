"""Multi-k report (k=1,3,5,10): recall/precision/nDCG@k + MAP + MRR, for the
9-embedder dense matrix, BM25, and hybrid (RRF) -- aggregated across the 4
chunkers, matching the convention every other headline table in
docs/paper-results-summary.md already uses.

Resolves the remaining "not yet done" tail of gap-analysis Tier 1 item #1:
`src/rag_lab/metrics.py` already supports multi-k (`evaluate(..., k=[1,3,5,10])`),
but no eval script had actually been re-run with it -- every citable number was
still k=10 only. This is a pure recompute over already-persisted retrieval
results: every combo was retrieved at top_k=10 (`RetrievalResult.results` has
exactly 10 ranked chunks per query), so k in {1,3,5,10} needs no new retrieval,
no GPU, no embedding calls -- runs in seconds against JSON already on disk.

Run with:
    .venv/Scripts/python.exe tools/eval/multi_k_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.metrics import evaluate  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from embedder_matrix_9way import (  # noqa: E402
    _INDEX_DIR,
    _RESULTS_DIR as _DENSE_RESULTS_DIR,
    EMBEDDER_ORDER,
    build_combo_to_chunker_embedder,
)

_BM25_RESULTS_DIR = REPO / "data" / "results" / "gold_bm25_73det"
_HYBRID_RESULTS_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "multi_k_report.md"
KS = [1, 3, 5, 10]


def _strip(cid: str, suffix: str) -> str | None:
    return cid[: -len(suffix)] if cid.endswith(suffix) else None


def aggregate_across_chunkers(scores_by_base, base_to_label, labels):
    by_label = {lbl: [] for lbl in labels}
    for base, scores in scores_by_base.items():
        label = base_to_label.get(base)
        if label in by_label:
            by_label[label].append(scores)
    out = {}
    for lbl, dicts in by_label.items():
        if not dicts:
            continue
        out[lbl] = {m: sum(d[m] for d in dicts) / len(dicts) for m in dicts[0]}
    return out


def render_table(agg: dict, title: str) -> list[str]:
    lines = [f"## {title}", ""]
    cols = ["map", "mrr"] + [f"recall@{k}" for k in KS] + [f"precision@{k}" for k in KS] + [f"ndcg@{k}" for k in KS]
    lines.append("| system | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for lbl in sorted(agg, key=lambda l: -agg[l]["recall@10"]):
        s = agg[lbl]
        cells = [lbl] + [f"{s[c]:.4f}" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> None:
    query_set = load_gold_query_set(str(_GOLD_QUERY_SET))
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    print(f"gold query set: {len(query_set)} queries")

    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)  # keys end in __dense
    chunker_by_base = {k[: -len("__dense")]: v[0] for k, v in combo_ce.items()}
    embedder_by_base = {k[: -len("__dense")]: v[1] for k, v in combo_ce.items()}
    embedders = [e for e in EMBEDDER_ORDER if e in set(embedder_by_base.values())]
    chunkers = sorted(set(chunker_by_base.values()))

    dense_persisted = [load_retrieval_result(p) for p in _DENSE_RESULTS_DIR.glob("*.json")]
    dense_scores = evaluate(dense_persisted, qrels, k=KS)
    dense_by_base = {b: s for cid, s in dense_scores.items() if (b := _strip(cid, "__dense"))}
    dense_agg = aggregate_across_chunkers(dense_by_base, embedder_by_base, embedders)
    print(f"dense: {len(dense_persisted)} persisted results, {len(dense_by_base)} combos scored")

    hybrid_persisted = [load_retrieval_result(p) for p in _HYBRID_RESULTS_DIR.glob("*.json")]
    hybrid_scores = evaluate(hybrid_persisted, qrels, k=KS)
    hybrid_by_base = {b: s for cid, s in hybrid_scores.items() if (b := _strip(cid, "__hybrid"))}
    hybrid_agg = aggregate_across_chunkers(hybrid_by_base, embedder_by_base, embedders)
    print(f"hybrid: {len(hybrid_persisted)} persisted results, {len(hybrid_by_base)} combos scored")

    bm25_persisted = [load_retrieval_result(p) for p in _BM25_RESULTS_DIR.glob("*.json")]
    bm25_scores = evaluate(bm25_persisted, qrels, k=KS)
    bm25_by_base = {b: s for cid, s in bm25_scores.items() if (b := _strip(cid, "__bm25"))}
    bm25_per_chunker = aggregate_across_chunkers(bm25_by_base, chunker_by_base, chunkers)
    bm25_overall = aggregate_across_chunkers(bm25_by_base, {b: "bm25" for b in bm25_by_base}, ["bm25"])
    print(f"bm25: {len(bm25_persisted)} persisted results, {len(bm25_by_base)} combos scored")

    lines = [
        "# Multi-k report (k=1,3,5,10): recall / precision / nDCG@k + MAP + MRR",
        "",
        f"Gold 73-det, {len(query_set)} queries. Recomputed purely from already-persisted "
        "top-10 retrieval results (no re-retrieval, no GPU, no embedding calls) -- closes "
        "the remaining 'not yet done' tail of gap-analysis Tier 1 item #1. Dense and hybrid "
        "tables average each embedder's per-query score across the 4 chunker strategies "
        "first, matching every other aggregate table in docs/paper-results-summary.md.",
        "",
    ]
    lines += render_table(dense_agg, "Dense-alone, 9 embedders (aggregated across 4 chunkers)")
    lines += render_table(hybrid_agg, "Hybrid (RRF), 9 embedders (aggregated across 4 chunkers)")
    lines += render_table(bm25_per_chunker, "BM25, per chunker")
    lines += render_table(bm25_overall, "BM25, aggregated across 4 chunkers")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
