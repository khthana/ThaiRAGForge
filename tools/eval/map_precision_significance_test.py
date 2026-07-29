"""MAP + precision@1 significance test -- closes the last untested-metric gap.

Every significance test in this project so far covers only recall@10, MRR, and
nDCG@10. MAP and precision@1 are reported in `data/results/multi_k_report.md`
and cited in docs/paper-results-summary.md ("Multi-k metrics"), but have never
been significance-tested -- so statements like "qwen3_0.6b leads every metric"
rest on raw means for those two, exactly the failure mode that retired the
"semantic chunking wins" headline (see docs/paper-results-summary.md Open item
#13). This script tests them.

The multi-k section also flagged a **scope mismatch**: the existing tied-cluster
finding (`hybrid_significance_test_semantic_top5.py`) is scoped to the `semantic`
chunker only, while the multi-k table aggregates across all 4 chunkers -- so the
table could not confirm or refute that tie even when the numbers agreed
directionally. This script therefore runs **both scopes**, so each claim can be
tested at the scope it was actually made at:

  * aggregate  -- all 9 embedders, per-query score averaged across the 4
                  chunkers first (the convention used by embedder_matrix_9way.py
                  and every cross-chunker table in the paper summary)
  * semantic   -- the 5 embedders of the top-5 tie test, `semantic` chunker only,
                  no cross-chunker averaging (matches that test's scope exactly)

crossed with both retrievers (dense-alone, hybrid). Holm-Bonferroni is applied
within each (retriever, scope, metric) family separately -- the same
"one family per set of simultaneous comparisons" convention used throughout.

MAP is `average_precision_at_k` at k=10 (resolution-level, divided by total
relevant count -- see src/rag_lab/metrics.py); precision@1 is `precision_at_k`
at k=1.

Pure recompute from already-persisted top-10 retrieval results -- no new
retrieval, no GPU, no embedding calls.

Run with:
    .venv/Scripts/python.exe tools/eval/map_precision_significance_test.py
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.metrics import average_precision_at_k, precision_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from embedder_matrix_9way import (  # noqa: E402
    _INDEX_DIR,
    _RESULTS_DIR as _DENSE_RESULTS_DIR,
    EMBEDDER_ORDER,
    bootstrap_pvalue,
    build_combo_to_chunker_embedder,
    holm_correct,
)

_HYBRID_RESULTS_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "map_precision_significance_test.md"

# Same 5 embedders as hybrid_significance_test_semantic_top5.py, so the semantic
# scope here lines up exactly with the tie that scope is meant to test.
TOP5 = ["qwen3_0.6b", "bge_m3", "e5_small", "qwen3", "jina_v5"]
SEMANTIC = "semantic"
METRICS = ("map", "precision@1")


def per_query_scores(persisted, combo_ce, suffix, qrels, query_idx, n_q, k, *, chunker_filter=None):
    """-> {embedder: {metric: np.ndarray over queries}}, averaged over the chunkers kept.

    `chunker_filter=None` keeps all 4 chunkers (aggregate scope); passing a
    chunker name keeps only that one (per-chunker scope). Averaging happens
    per query *before* any test, matching the convention used elsewhere.
    """
    embedders = sorted({v[1] for v in combo_ce.values()})
    sums = {e: {m: np.zeros(n_q) for m in METRICS} for e in embedders}
    counts = {e: {m: np.zeros(n_q) for m in METRICS} for e in embedders}

    for r in persisted:
        if not r.combination_id.endswith(suffix):
            continue
        base = r.combination_id[: -len(suffix)]
        if base not in combo_ce:
            continue
        chunker, embedder = combo_ce[base]
        if chunker_filter is not None and chunker != chunker_filter:
            continue
        qi = query_idx.get(r.query)
        if qi is None:
            continue
        relevant = qrels[r.query]
        sums[embedder]["map"][qi] += average_precision_at_k(r, relevant, k)
        sums[embedder]["precision@1"][qi] += precision_at_k(r, relevant, 1)
        for m in METRICS:
            counts[embedder][m][qi] += 1

    out = {}
    for e in embedders:
        if counts[e]["map"].sum() == 0:
            continue
        out[e] = {
            m: np.divide(sums[e][m], counts[e][m], out=np.zeros(n_q), where=counts[e][m] > 0)
            for m in METRICS
        }
        missing = int((counts[e]["map"] == 0).sum())
        if missing:
            print(f"  WARNING: {e} missing {missing}/{n_q} queries")
    return out


def run_family(per_query, embedders, metric, rng, n_boot, alpha):
    pairs = []
    for a, b in itertools.combinations(embedders, 2):
        diffs = per_query[a][metric] - per_query[b][metric]
        observed, p, ci = bootstrap_pvalue(diffs, rng, n_boot)
        pairs.append((a, b, observed, p, ci))
    return holm_correct(pairs, alpha)


def render(results, per_query, embedders, title, note, metric, alpha):
    lines = [f"### {title} -- {metric}", "", note, ""]
    means = {e: float(per_query[e][metric].mean()) for e in embedders}
    lines.append("| embedder | mean |")
    lines.append("|---|---|")
    for e in sorted(embedders, key=lambda x: -means[x]):
        lines.append(f"| {e} | {means[e]:.4f} |")
    lines += ["", "| A | B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |", "|---|---|---|---|---|---|---|"]
    for a, b, diff, p, ci, adj, sig in sorted(results, key=lambda r: r[5]):
        lines.append(
            f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{p:.4f} | {adj:.4f} | {'**yes**' if sig else 'no'} |"
        )
    n_sig = sum(1 for r in results if r[6])
    lines += ["", f"{n_sig}/{len(results)} pairs significant at alpha={alpha} after Holm correction.", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="window for MAP (precision@1 is always k=1)")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    queries = list(qrels)
    query_idx = {q: i for i, q in enumerate(queries)}
    n_q = len(queries)
    print(f"gold query set: {n_q} queries")

    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)  # keys end in __dense
    combo_ce = {k[: -len("__dense")]: v for k, v in combo_ce.items()}

    dense = [load_retrieval_result(p) for p in _DENSE_RESULTS_DIR.glob("*.json")]
    hybrid = [load_retrieval_result(p) for p in _HYBRID_RESULTS_DIR.glob("*.json")]
    print(f"loaded {len(dense)} dense + {len(hybrid)} hybrid persisted results")

    scopes = {}
    for rname, persisted, suffix in (("dense", dense, "__dense"), ("hybrid", hybrid, "__hybrid")):
        print(f"{rname}: aggregate scope")
        scopes[(rname, "aggregate")] = per_query_scores(
            persisted, combo_ce, suffix, qrels, query_idx, n_q, args.k
        )
        print(f"{rname}: semantic-only scope")
        scopes[(rname, "semantic")] = per_query_scores(
            persisted, combo_ce, suffix, qrels, query_idx, n_q, args.k, chunker_filter=SEMANTIC
        )

    lines = [
        "# MAP + precision@1 significance test (Gold 73-det)",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}), "
        f"Holm-Bonferroni within each (retriever, scope, metric) family separately "
        f"(alpha={args.alpha}). MAP = average_precision_at_k at k={args.k}; precision@1 = "
        "precision_at_k at k=1. Both are resolution-level (ADR-0002). Pure recompute from "
        "already-persisted top-10 retrieval results -- no new retrieval.",
        "",
        "Closes the standing open item that MAP and precision@1 were reported but never "
        "tested. Two scopes are run because the existing tied-cluster finding "
        "(`hybrid_significance_test_semantic_top5.py`) is scoped to the `semantic` chunker "
        "only while the multi-k table aggregates across all 4 -- testing both means each "
        "claim can be checked at the scope it was actually made at.",
        "",
    ]

    summary: list[str] = []
    for rname in ("dense", "hybrid"):
        for scope in ("aggregate", "semantic"):
            pq = scopes[(rname, scope)]
            if scope == "aggregate":
                embedders = [e for e in EMBEDDER_ORDER if e in pq]
                note = (
                    f"All {len(embedders)} embedders, per-query score averaged across the 4 "
                    "chunkers first (cross-chunker aggregate convention)."
                )
            else:
                embedders = [e for e in TOP5 if e in pq]
                note = (
                    f"Top-5 hybrid embedders, `{SEMANTIC}` chunker only, no cross-chunker "
                    "averaging -- same scope as `hybrid_significance_test_semantic_top5.py`."
                )
            if len(embedders) < 2:
                print(f"SKIP {rname}/{scope}: only {len(embedders)} embedders found")
                continue
            lines.append(f"## {rname} / {scope} scope")
            lines.append("")
            rng_seed = args.seed
            for metric in METRICS:
                rng = np.random.default_rng(rng_seed)
                res = run_family(pq, embedders, metric, rng, args.n_boot, args.alpha)
                lines += render(res, pq, embedders, f"{rname} / {scope}", note, metric, args.alpha)
                best = max(embedders, key=lambda e: pq[e][metric].mean())
                beaten = [b for a, b, *_, sig in res if a == best and sig] + [
                    a for a, b, *_, sig in res if b == best and sig
                ]
                tied = [e for e in embedders if e != best and e not in beaten]
                summary.append(
                    f"- **{rname} / {scope} / {metric}**: highest is `{best}` "
                    f"({pq[best][metric].mean():.4f}); significantly beats {len(beaten)}/"
                    f"{len(embedders) - 1} others; ties {sorted(tied)}"
                )

    lines += ["## Summary", "", *summary, ""]

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nwritten to {_OUTPUT}")


if __name__ == "__main__":
    main()
