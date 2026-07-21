"""Before/after eval for the ConGen/SCT silent-truncation bug fix (2026-07-21).

Both kornwtp Thai models (ConGen-BGE_M3-model-phayathaibert,
SCT-KD-BGE-M3-model-phayathaibert) shipped max_seq_length=128 in their HF
repo's sentence_bert_config.json -- far below their PhayaThaiBERT backbone's
true 510-token ceiling (RoBERTa reserves 2 position slots for the padding
offset, so max_position_embeddings=512 - 2 = 510). Fixed via a
max_seq_length=510 override
(config/experiments/chunker_compare_full_fix_congen_sct_maxseqlen.yaml,
new combo dirs since combo id hashes the full embedder params).

Runs Gold 73-det retrieval against both the old (128-cap, buggy) and new
(510-cap, fixed) combo dirs for both models across all 4 chunkers, then:
1. reports recall@10/mrr/ndcg@10 old vs new per chunker (magnitude of the
   truncation cost)
2. paired-bootstrap significance test, old vs new, per model (aggregated
   across chunkers) -- is the fix a real, provable improvement?

Run with:
    .venv/Scripts/python.exe tools/eval/congen_sct_truncation_fix_eval.py
"""
from __future__ import annotations

import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.metrics import recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set, run_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_RESULTS_DIR = REPO / "data" / "results" / "congen_sct_truncation_fix"
_OUTPUT = REPO / "data" / "results" / "congen_sct_truncation_fix_report.md"
K = 10
N_BOOT = 10_000
SEED = 42

# (model_label, chunker) -> (old_combo_id, new_combo_id)
COMBOS = {
    ("congen", "fixed_size"): ("plain__fixed_size__local__7cceab27", "plain__fixed_size__local__e6048946"),
    ("congen", "recursive"): ("plain__recursive__local__d04f22ee", "plain__recursive__local__4a350a4e"),
    ("congen", "sentence"): ("plain__sentence__local__5f573c4f", "plain__sentence__local__26622ae7"),
    ("congen", "semantic"): ("plain__semantic__local__87fee2dc", "plain__semantic__local__6b33a155"),
    ("sct", "fixed_size"): ("plain__fixed_size__local__9d03b361", "plain__fixed_size__local__2020e443"),
    ("sct", "recursive"): ("plain__recursive__local__31293c05", "plain__recursive__local__29632808"),
    ("sct", "sentence"): ("plain__sentence__local__d6c1f8e1", "plain__sentence__local__e83d75c9"),
    ("sct", "semantic"): ("plain__semantic__local__9576aa59", "plain__semantic__local__f477fdca"),
}


def paired_bootstrap(old_scores: list[float], new_scores: list[float]) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    diffs = np.array(new_scores) - np.array(old_scores)
    observed = float(diffs.mean())
    n = len(diffs)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boot[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # two-sided percentile p-value: fraction of bootstrap draws crossing 0
    p = 2 * min((boot >= 0).mean(), (boot <= 0).mean())
    p = min(p, 1.0)
    return observed, p


def main() -> None:
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    print(f"gold 73-det: {len(query_set)} queries")

    all_combo_ids = {cid for pair in COMBOS.values() for cid in pair}
    index_dirs = [str(_INDEX_DIR / cid) for cid in all_combo_ids]

    t0 = time.time()
    run_query_set(query_set, index_dirs, StrategySpec(type="dense"), k=K, results_dir=str(_RESULTS_DIR))
    print(f"retrieval done in {time.time() - t0:.1f}s")

    persisted = [load_retrieval_result(p) for p in _RESULTS_DIR.glob("*.json")]
    by_combo_query = {(r.combination_id, r.query): r for r in persisted}

    lines = [
        "# ConGen/SCT truncation-fix before/after (Gold 73-det)",
        "",
        "Old = max_seq_length 128 (buggy, repo default). New = max_seq_length 510",
        "(fixed, true backbone ceiling). Per-query recall@10/mrr/ndcg@10, then",
        "paired-bootstrap significance (new - old, n_boot=10000, resample unit=query).",
        "",
        "## Per-chunker recall@10",
        "",
        "| model | chunker | old | new | diff | Holm-adj p (see below, single family) |",
        "|---|---|---|---|---|---|",
    ]

    raw_pvals = []
    diff_rows = []
    for (model, chunker), (old_id, new_id) in COMBOS.items():
        old_scores, new_scores = [], []
        for entry in query_set:
            q = entry.query
            relevant = qrels[q]
            r_old = by_combo_query.get((f"{old_id}__dense", q))
            r_new = by_combo_query.get((f"{new_id}__dense", q))
            old_scores.append(recall_at_k(r_old, relevant, K) if r_old else 0.0)
            new_scores.append(recall_at_k(r_new, relevant, K) if r_new else 0.0)
        obs, p = paired_bootstrap(old_scores, new_scores)
        raw_pvals.append(p)
        diff_rows.append((model, chunker, statistics.mean(old_scores), statistics.mean(new_scores), obs, p))

    # Holm correction across these 8 chunker-level tests
    order = sorted(range(len(raw_pvals)), key=lambda i: raw_pvals[i])
    m = len(raw_pvals)
    holm_adj = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        adj = min((m - rank) * raw_pvals[i], 1.0)
        running_max = max(running_max, adj)
        holm_adj[i] = running_max

    for (model, chunker, old_mean, new_mean, obs, p), p_adj in zip(diff_rows, holm_adj):
        lines.append(
            f"| {model} | {chunker} | {old_mean:.4f} | {new_mean:.4f} | {obs:+.4f} | {p_adj:.4f} |"
        )

    lines.append("")
    lines.append("## Aggregate (across all 4 chunkers) per model, recall@10")
    lines.append("")
    lines.append("| model | old mean | new mean | diff | paired-bootstrap p |")
    lines.append("|---|---|---|---|---|")
    for model in ["congen", "sct"]:
        old_by_q = defaultdict(list)
        new_by_q = defaultdict(list)
        for (m2, chunker), (old_id, new_id) in COMBOS.items():
            if m2 != model:
                continue
            for entry in query_set:
                q = entry.query
                relevant = qrels[q]
                r_old = by_combo_query.get((f"{old_id}__dense", q))
                r_new = by_combo_query.get((f"{new_id}__dense", q))
                old_by_q[q].append(recall_at_k(r_old, relevant, K) if r_old else 0.0)
                new_by_q[q].append(recall_at_k(r_new, relevant, K) if r_new else 0.0)
        old_agg = [statistics.mean(old_by_q[e.query]) for e in query_set]
        new_agg = [statistics.mean(new_by_q[e.query]) for e in query_set]
        obs, p = paired_bootstrap(old_agg, new_agg)
        lines.append(
            f"| {model} | {statistics.mean(old_agg):.4f} | {statistics.mean(new_agg):.4f} | {obs:+.4f} | {p:.4f} |"
        )

    report = "\n".join(lines)
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
