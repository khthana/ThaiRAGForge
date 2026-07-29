"""BM25 and hybrid (RRF) recall@10 broken out by query entity_type, against the
structural ceiling -- closes a standing open item from docs/paper-results-summary.md.

Only *dense-alone* has ever had a per-entity_type breakdown (via
`embedder_matrix_9way.py`, which produced the "specialist vs generalist" finding).
BM25 and hybrid were only ever reported as cross-chunker aggregates, so two
questions raised by the "Structural recall@10 ceiling by entity_type" section
have been unanswerable:

  1. How close does hybrid -- the project's recommended system -- actually get to
     the `person` ceiling of 0.976?
  2. Does hybrid narrow the `faculty_adjunct_aggregate` gap, whose ceiling is a
     much lower 0.681?

The ceiling matters because the Gold set is not one-relevant-doc-per-query: a
query with 43 relevant resolutions can score at most 10/43 under a k=10 window
even with a perfect retriever. Raw recall@10 across entity types is therefore
not comparable without normalising by it -- this script reports both the raw
number and the fraction of the achievable ceiling attained, which is the
comparable quantity.

Pure recompute from already-persisted top-10 retrieval results -- no new
retrieval, no GPU, no embedding calls.

Run with:
    .venv/Scripts/python.exe tools/eval/bm25_hybrid_entity_type_breakdown.py
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.metrics import recall_at_k  # noqa: E402
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
_OUTPUT = REPO / "data" / "results" / "bm25_hybrid_entity_type_breakdown.md"


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def by_embedder(persisted, combo_ce, suffix, qrels, types_by_query, k):
    """-> {embedder: {entity_type: recall}}, averaged across the 4 chunkers."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in persisted:
        if not r.combination_id.endswith(suffix):
            continue
        base = r.combination_id[: -len(suffix)]
        if base not in combo_ce:
            continue
        _, embedder = combo_ce[base]
        etype = types_by_query.get(r.query)
        if etype is None:
            continue
        acc[embedder][etype].append(recall_at_k(r, qrels[r.query], k))
    return {e: {t: mean(v) for t, v in d.items()} for e, d in acc.items()}


def by_chunker(persisted, combo_ce, suffix, qrels, types_by_query, k):
    """-> {chunker: {entity_type: recall}} (BM25 is embedder-agnostic)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in persisted:
        if not r.combination_id.endswith(suffix):
            continue
        base = r.combination_id[: -len(suffix)]
        if base not in combo_ce:
            continue
        chunker, _ = combo_ce[base]
        etype = types_by_query.get(r.query)
        if etype is None:
            continue
        acc[chunker][etype].append(recall_at_k(r, qrels[r.query], k))
    return {c: {t: mean(v) for t, v in d.items()} for c, d in acc.items()}


def render(table, etypes, row_label, title, note, ceilings=None, order=None):
    lines = [f"## {title}", "", note, ""]
    lines.append(f"| {row_label} | " + " | ".join(etypes) + " | overall |")
    lines.append("|---|" + "---|" * (len(etypes) + 1))
    rows = order or sorted(table, key=lambda r: -mean(list(table[r].values())))
    for row in rows:
        if row not in table:
            continue
        cells = [f"{table[row].get(t, 0.0):.4f}" for t in etypes]
        lines.append(f"| {row} | " + " | ".join(cells) + f" | {mean(list(table[row].values())):.4f} |")
    if ceilings:
        lines.append("| *(ceiling)* | " + " | ".join(f"*{ceilings[t]:.4f}*" for t in etypes) + " | |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    k = args.k

    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    raw = yaml.safe_load(_GOLD_QUERY_SET.read_text(encoding="utf-8"))
    types_by_query = {e["query"]: e.get("entity_type", "unknown") for e in raw}
    etypes = sorted(set(types_by_query.values()))
    print(f"gold query set: {len(query_set)} queries, entity types: {etypes}")

    # Structural ceiling: mean over queries of min(1, k / n_relevant).
    ceil_acc = defaultdict(list)
    nrel_acc = defaultdict(list)
    for q, rel in qrels.items():
        t = types_by_query.get(q, "unknown")
        ceil_acc[t].append(min(1.0, k / len(rel)) if rel else 0.0)
        nrel_acc[t].append(len(rel))
    ceilings = {t: mean(v) for t, v in ceil_acc.items()}

    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)  # keys end in __dense
    combo_ce = {c[: -len("__dense")]: v for c, v in combo_ce.items()}

    dense = [load_retrieval_result(p) for p in _DENSE_RESULTS_DIR.glob("*.json")]
    hybrid = [load_retrieval_result(p) for p in _HYBRID_RESULTS_DIR.glob("*.json")]
    bm25 = [load_retrieval_result(p) for p in _BM25_RESULTS_DIR.glob("*.json")]
    print(f"loaded {len(dense)} dense + {len(hybrid)} hybrid + {len(bm25)} bm25 results")

    dense_tbl = by_embedder(dense, combo_ce, "__dense", qrels, types_by_query, k)
    hybrid_tbl = by_embedder(hybrid, combo_ce, "__hybrid", qrels, types_by_query, k)
    bm25_by_chunker = by_chunker(bm25, combo_ce, "__bm25", qrels, types_by_query, k)
    bm25_agg = {t: mean([bm25_by_chunker[c][t] for c in bm25_by_chunker if t in bm25_by_chunker[c]]) for t in etypes}

    order = [e for e in EMBEDDER_ORDER if e in hybrid_tbl]

    lines = [
        f"# BM25 and hybrid recall@{k} by entity_type, vs. the structural ceiling (Gold 73-det)",
        "",
        f"{len(query_set)} queries. Pure recompute from already-persisted top-{k} retrieval "
        "results -- no new retrieval. Closes the open item that only dense-alone had ever "
        "been broken out by entity_type, leaving it unknown how close the project's "
        "recommended system (hybrid) gets to each category's achievable ceiling.",
        "",
        "## Structural ceiling per entity_type",
        "",
        f"The Gold set is not one-relevant-doc-per-query, so recall@{k} is capped below 1.0: "
        f"a query with n relevant resolutions can score at most min(1, {k}/n). The ceiling "
        "below is the mean of that cap over each category's queries -- **raw recall is not "
        "comparable across categories without it**.",
        "",
        "| entity_type | n queries | avg relevant | max relevant | ceiling |",
        "|---|---|---|---|---|",
    ]
    for t in etypes:
        lines.append(
            f"| {t} | {len(nrel_acc[t])} | {mean(nrel_acc[t]):.1f} | "
            f"{max(nrel_acc[t])} | **{ceilings[t]:.4f}** |"
        )
    lines.append("")

    lines += render(
        bm25_by_chunker, etypes, "chunker", "BM25 by chunker",
        "BM25 is embedder-agnostic, so it varies only by chunker.", ceilings,
    )
    lines += render(
        {"bm25 (aggregate)": bm25_agg}, etypes, "system", "BM25 aggregated across 4 chunkers",
        "Same aggregation convention as every other headline table.", ceilings,
    )
    lines += render(
        hybrid_tbl, etypes, "embedder", "Hybrid (RRF) by embedder",
        "Per-query recall averaged across the 4 chunkers first.", ceilings, order,
    )
    lines += render(
        dense_tbl, etypes, "embedder", "Dense-alone by embedder (for side-by-side reference)",
        "Reproduces the existing `gold_embedder_breakdown_9way.md` numbers under the same "
        "code path, so the three retrievers can be compared directly.", ceilings, order,
    )

    # Headline: best system per entity_type and how much of the ceiling it reaches.
    lines += [
        "## Ceiling attainment -- best system per entity_type",
        "",
        "`% of ceiling` = recall / ceiling. This is the number that is actually comparable "
        "across categories; raw recall is not.",
        "",
        "| entity_type | ceiling | best hybrid | recall | % of ceiling | best dense | recall | % of ceiling | bm25 | % of ceiling |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    attainment = {}
    for t in etypes:
        bh = max(hybrid_tbl, key=lambda e: hybrid_tbl[e].get(t, 0.0))
        bd = max(dense_tbl, key=lambda e: dense_tbl[e].get(t, 0.0))
        hv, dv, bv = hybrid_tbl[bh].get(t, 0.0), dense_tbl[bd].get(t, 0.0), bm25_agg.get(t, 0.0)
        c = ceilings[t]
        attainment[t] = (bh, hv, hv / c if c else 0.0)
        lines.append(
            f"| {t} | {c:.4f} | {bh} | {hv:.4f} | **{hv / c:.1%}** | "
            f"{bd} | {dv:.4f} | {dv / c:.1%} | {bv:.4f} | {bv / c:.1%} |"
        )
    lines.append("")

    ranked = sorted(attainment.items(), key=lambda kv: -kv[1][2])
    lines += [
        "## Reading this",
        "",
        "Ranked by how much of its own achievable ceiling the best hybrid system reaches "
        "(highest = closest to solved, lowest = most real headroom left):",
        "",
    ]
    for t, (emb, val, frac) in ranked:
        lines.append(f"- **{t}**: {frac:.1%} of ceiling ({val:.4f} / {ceilings[t]:.4f}, best = `{emb}`)")
    lines.append("")

    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    for t, (emb, val, frac) in ranked:
        print(f"{t:32s} {frac:6.1%} of ceiling  ({val:.4f}/{ceilings[t]:.4f}, {emb})")
    print(f"\nwritten to {_OUTPUT}")


if __name__ == "__main__":
    main()
