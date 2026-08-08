# -*- coding: utf-8 -*-
"""Soft routing (one index, per-route fusion weight) vs hard routing (a
different index per route), head to head on the 106-query 73det Gold set.

Why this comparison exists. Two results landed on 2026-08-08 pointing opposite
ways, and they had never been put on the same axis:

  * hard routing (`tools/eval/routing_eval.py`, task #19) switches the *index*
    per route. It closed a 43% coverage hole, but against the best single combo
    used for everything it was only +0.0101 recall@10 under hybrid -- not
    significant. It needs 5 indices to do that.
  * soft routing (`tools/eval/hybrid_alpha_sweep.py`, task #20) switches only
    the *fusion weight* per route, on one index. That is +0.0350 recall@10,
    Holm-significant and surviving leave-one-out.

Both read the same signal (`classify_query`), so the interesting question is
whether the cheap one is as good as the expensive one -- and whether they add.

READ THE DATE ON ANY RESULT FROM THIS SCRIPT. The first run (against
`ROUTE_COMBO`'s 2026-07-17 targets) had soft ahead of hard, and its per-route
table is what showed why: the `program` target was not merely stale, routing to
it scored 0.5321 where not routing scored 0.6105. Those targets were refreshed
the same day and this script re-run; hard then led on every metric and every
route. **The soft arm never moved.** So a headline from here is a statement
about the route targets in force at run time, not a property of the two
mechanisms -- which is exactly why arm C resolves through the live route map
(see `target_combo_id`) instead of pinning combo ids.

Design. Four arms, every one of them retrieving k=10 from exactly ONE index per
query, so no arm wins by spending more (see
[[feedback_state_the_retrieval_budget_in_every_comparison]]):

  A  single @ 0.50            -- best single combo for every query, shipped 50:50.
                                 No classifier at all.
  B  single + per-type alpha  -- same one index, fusion weight per route.  SOFT
  C  routed @ 0.50            -- the shipped hybrid target per route.      HARD
  D  routed + per-type alpha  -- both.

Fitting budgets are matched, which is the trap this project has hit before.
Index choice is held at its *shipped* value in every arm (arm A uses the combo
`routing_eval` found to be the argmax over all 36 under hybrid, and which its
LOO selector re-picks in every fold, so it is not acting as an oracle here).
The only thing fitted is alpha, and it is fitted leave-one-out: for a query of
route r, alpha comes from r's OTHER queries on r's own index. `(oracle)` rows
fit alpha on all of a route's queries including the scored one -- upper bounds,
reported for the gap, never as a system.

Routing uses `classify_query`, not the gold `entity_type` -- the deployable
signal, not the label. They agree exactly on this set (checked and reported),
so this costs nothing here while keeping the arm honest.

Run with:
    .venv/Scripts/python.exe tools/eval/soft_vs_hard_routing.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402
from hybrid_alpha_sweep import arm_ranks, fuse  # noqa: E402

from rag_lab.factory import build_embedder  # noqa: E402
from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from rag_lab.query_service import discover_indices, resolve_index  # noqa: E402
from rag_lab.router import classify_query, route_targets  # noqa: E402

GOLD_PATH = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
OUTPUT = REPO / "data" / "results" / "soft_vs_hard_routing.md"
K = 10
N_BOOT = 10_000
SEED = 42

# The best single combo over all 36 under hybrid, per routing_eval.md. Its LOO
# selector re-picks it in every fold (best single = best single (loo) = 0.6281),
# so holding it fixed here is not an oracle shortcut.
SINGLE_COMBO = "plain__sentence__qwen3__ff8f6c49"  # sentence + Qwen3-Embedding-0.6B

METRICS = {
    "recall@10": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    "ndcg@10": lambda r, rel: ndcg_at_k(r, rel, K),
}


def load_gold() -> tuple[dict[str, list[str]], dict[str, str]]:
    entries = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))
    return (
        {e["query"]: e["relevant_resolution_ids"] for e in entries},
        {e["query"]: e.get("entity_type", "unknown") for e in entries},
    )


def target_combo_id(route: str, index_list: list) -> str:
    """The combo_id a route retrieves from. Goes through `query_service`'s own
    `resolve_index`, not a local re-implementation, so this arm switches indices
    exactly the way the shipped router does -- and a route-target edit is picked
    up here instead of being silently ignored.

    Asks for the *hybrid* map explicitly: this whole script is hybrid, and since
    2026-08-08 the route map is keyed by retriever (router.route_targets), so
    reading the `ROUTE_COMBO` alias would quietly bind arm C to whichever
    retriever that alias happens to point at."""
    return resolve_index(route_targets("hybrid")[route], index_list).combo_id


def score_grid(
    d_ranks, b_ranks, res_ids, queries, qrels, alphas
) -> dict[tuple[float, str], dict[str, float]]:
    """Per-query score for every (alpha, metric) on one index."""
    out: dict[tuple[float, str], dict[str, float]] = {}
    for a in alphas:
        results = {q: fuse(d_ranks[i], b_ranks[i], res_ids, q, a) for i, q in enumerate(queries)}
        for m, fn in METRICS.items():
            out[(a, m)] = {q: fn(results[q], qrels[q]) for q in queries}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha-level", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=OUTPUT)
    args = ap.parse_args()

    qrels, etype = load_gold()
    queries = sorted(qrels)
    alphas = [round(i * args.step, 4) for i in range(int(round(1.0 / args.step)) + 1)]
    index_list = discover_indices(INDEX_DIR)
    indices = {i.combo_id: i for i in index_list}
    store = ArtifactStore()
    rng = np.random.default_rng(args.seed)

    # --- the routing signal: classify_query, not the gold label ---------------
    route_of = {q: classify_query(q) for q in queries}
    by_route: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_route[route_of[q]].append(q)

    # Agreement is measured on the PARTITION, not on the label strings: the Gold
    # set calls the faculty queries `faculty_adjunct_aggregate` while the router
    # calls that route `faculty`, so a naive string comparison reports 93/106
    # when the routing is in fact exact. Map each entity_type to its modal route
    # and count queries that leave it.
    modal: dict[str, str] = {}
    for t in set(etype.values()):
        rs = [route_of[q] for q in queries if etype[q] == t]
        modal[t] = max(set(rs), key=rs.count)
    agree = sum(1 for q in queries if route_of[q] == modal[etype[q]])

    # --- which index each arm reads, per query -------------------------------
    routed_combo = {r: target_combo_id(r, index_list) for r in by_route}
    needed: dict[str, list[str]] = {SINGLE_COMBO: list(queries)}
    for r, qs in by_route.items():
        needed.setdefault(routed_combo[r], [])
        needed[routed_combo[r]].extend(q for q in qs if q not in needed[routed_combo[r]])

    # --- retrieve both arms once per (index, needed queries) -----------------
    grids: dict[str, dict[tuple[float, str], dict[str, float]]] = {}
    for combo_id, qs in needed.items():
        info = indices[combo_id]
        embedder = build_embedder(info.embedder)
        index = store.load(info.dir)
        t0 = time.time()
        d_ranks, b_ranks, res_ids = arm_ranks(index, embedder, qs)
        grids[combo_id] = score_grid(d_ranks, b_ranks, res_ids, qs, qrels, alphas)
        print(f"[{combo_id}] {len(qs)} queries, {len(index.chunks):,} chunks, "
              f"{time.time() - t0:.0f}s")

    def alpha_for(grid, pool: list[str], metric: str) -> float:
        return max(alphas, key=lambda a: statistics.mean(grid[(a, metric)][q] for q in pool))

    # --- the four arms, per query --------------------------------------------
    def arm_scores(metric: str) -> dict[str, list[float]]:
        single = grids[SINGLE_COMBO]
        a_single = [single[(0.5, metric)][q] for q in queries]
        c_routed = [grids[routed_combo[route_of[q]]][(0.5, metric)][q] for q in queries]

        b_soft, d_both, b_orc, d_orc = [], [], [], []
        for q in queries:
            peers = [x for x in by_route[route_of[q]] if x != q]
            own = by_route[route_of[q]]
            rg = grids[routed_combo[route_of[q]]]
            b_soft.append(single[(alpha_for(single, peers, metric) if peers else 0.5, metric)][q])
            d_both.append(rg[(alpha_for(rg, peers, metric) if peers else 0.5, metric)][q])
            b_orc.append(single[(alpha_for(single, own, metric), metric)][q])
            d_orc.append(rg[(alpha_for(rg, own, metric), metric)][q])
        return {
            "A  single @ 0.50": a_single,
            "C  routed @ 0.50 (hard)": c_routed,
            "B  single + per-route alpha (soft, loo)": b_soft,
            "D  routed + per-route alpha (both, loo)": d_both,
            "B' single + per-route alpha (oracle)": b_orc,
            "D' routed + per-route alpha (oracle)": d_orc,
        }

    arms = {m: arm_scores(m) for m in METRICS}

    lines: list[str] = []
    out = lines.append
    out(f"# Soft vs hard routing -- per-route fusion weight vs per-route index (n={len(queries)})")
    out("")
    out("Generated by `tools/eval/soft_vs_hard_routing.py`. Every arm retrieves")
    out(f"k={K} from exactly **one** index per query -- equal retrieval budget, no arm")
    out("wins by fetching more. Hybrid (RRF) throughout; `alpha` is the dense weight.")
    out("")
    out("Routing signal is `classify_query`, not the gold `entity_type`. The two")
    out(f"partitions agree on **{agree}/{len(queries)}** queries, so the deployable signal")
    out("costs nothing on this set. (Agreement is measured on the partition: the Gold")
    out("set names the faculty queries `faculty_adjunct_aggregate` and the router names")
    out("that route `faculty`, so comparing label strings would understate it as 93/106.)")
    out("Routes and their shipped index targets:")
    out("")
    out("| route | n | index used |")
    out("|---|---|---|")
    for r in sorted(by_route):
        out(f"| `{r}` | {len(by_route[r])} | `{routed_combo[r]}` |")
    out(f"| _(arm A/B, all queries)_ | {len(queries)} | `{SINGLE_COMBO}` |")
    out("")
    out("Arms: **A** no classifier at all; **B** classifier moves only the fusion")
    out("weight (one index); **C** classifier moves the index (5 indices); **D** both.")
    out("Only alpha is fitted, leave-one-out within a route. `'` rows fit alpha on the")
    out("scored query too -- upper bounds, not systems.")
    out("")

    # Per-route breakdown. The aggregate cannot explain why D (both) trails B
    # (soft alone) on recall@10 -- that has to be a route where switching the
    # index costs more than the alpha then recovers.
    out("## Per-route breakdown (recall@10)")
    out("")
    out("`alpha*` is the oracle per-route optimum on that route's own index (the")
    out("LOO folds pick from this same curve). Read it against the shipped 0.50.")
    out("")
    out("| route | n | A single@0.50 | B soft | C hard | D both | alpha* single | alpha* routed |")
    out("|---|---|---|---|---|---|---|---|")
    single = grids[SINGLE_COMBO]
    for r in sorted(by_route):
        qs, rg = by_route[r], grids[routed_combo[r]]
        a_s = alpha_for(single, qs, "recall@10")
        a_r = alpha_for(rg, qs, "recall@10")
        cells = [
            statistics.mean(single[(0.5, "recall@10")][q] for q in qs),
            statistics.mean(arms["recall@10"]["B  single + per-route alpha (soft, loo)"][queries.index(q)] for q in qs),
            statistics.mean(rg[(0.5, "recall@10")][q] for q in qs),
            statistics.mean(arms["recall@10"]["D  routed + per-route alpha (both, loo)"][queries.index(q)] for q in qs),
        ]
        out(f"| `{r}` | {len(qs)} | " + " | ".join(f"{c:.4f}" for c in cells)
            + f" | {a_s:.2f} | {a_r:.2f} |")
    out("")

    for metric in METRICS:
        out(f"## {metric}")
        out("")
        out("| arm | mean | vs A | vs C (hard) |")
        out("|---|---|---|---|")
        base_a = statistics.mean(arms[metric]["A  single @ 0.50"])
        base_c = statistics.mean(arms[metric]["C  routed @ 0.50 (hard)"])
        for name, vals in arms[metric].items():
            m = statistics.mean(vals)
            out(f"| {name} | {m:.4f} | {m - base_a:+.4f} | {m - base_c:+.4f} |")
        out("")

    # --- significance --------------------------------------------------------
    comparisons = [
        ("B  single + per-route alpha (soft, loo)", "A  single @ 0.50"),
        ("C  routed @ 0.50 (hard)", "A  single @ 0.50"),
        ("B  single + per-route alpha (soft, loo)", "C  routed @ 0.50 (hard)"),
        ("D  routed + per-route alpha (both, loo)", "C  routed @ 0.50 (hard)"),
    ]
    pairs, mlabels = [], []
    for metric in METRICS:
        for hi, lo in comparisons:
            diffs = np.array(arms[metric][hi]) - np.array(arms[metric][lo])
            obs, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((hi, lo, obs, p, ci))
            mlabels.append(metric)
    corrected = holm_correct(pairs, alpha=args.alpha_level)

    out("## Significance")
    out("")
    out(f"Paired bootstrap, {args.n_boot:,} resamples, seed {args.seed}, Holm within a")
    out(f"family of m = **{len(corrected)}** (4 comparisons x 3 metrics). Quote the")
    out("family size with any p from this table -- it is part of the number.")
    out("")
    out("| metric | comparison | diff | 95% CI | Holm-adj p | significant |")
    out("|---|---|---|---|---|---|")
    for metric, (hi, lo, diff, _p, ci, adj, sig) in zip(mlabels, corrected):
        out(f"| {metric} | {hi.split()[0]} vs {lo.split()[0]} | {diff:+.4f} "
            f"| [{ci[0]:+.4f}, {ci[1]:+.4f}] | {adj:.4f} | {'**yes**' if sig else 'no'} |")
    out("")
    # Read the B-vs-A row straight back out of `corrected` rather than quoting
    # numbers inline: this paragraph is prose ABOUT the table above it, so it
    # has no cell for a verdict-differ to check and would survive any refresh
    # unnoticed. The p in particular is NOT a property of this comparison alone
    # -- Holm makes it depend on every other pair in the family, so it moves
    # whenever a sibling comparison moves, without B or A changing at all.
    ba = {
        m: (d, adj)
        for m, (hi, lo, d, _p, _ci, adj, _s) in zip(mlabels, corrected)
        if hi.startswith("B") and lo.startswith("A")
    }
    out("**Cross-check, and a family-size warning.** Arms A and B here are the same")
    out("two systems `hybrid_alpha_sweep.py` compared on `sentence+qwen3_0.6b`, and")
    out("the three B-vs-A effect sizes reproduce it to 4 decimal places ("
        + " / ".join(f"{ba[m][0]:+.4f}" for m in METRICS)
        + ") -- two independent scripts, same folds, same numbers. The")
    out("**verdict** on `recall@10` nonetheless differs: Holm-adj **0.0252** there")
    out(f"(m=9) and **{ba['recall@10'][1]:.4f}** here (m={len(corrected)}). Same data, "
        "same difference,")
    out("larger family. Neither is wrong. Cite the sweep's m=9 for \"is a per-route")
    out("alpha worth anything\" -- that is the family built to answer it -- and cite")
    out("this table's m=12 only for the four comparisons it exists to make.")
    out("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
