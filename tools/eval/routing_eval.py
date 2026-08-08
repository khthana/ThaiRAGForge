# -*- coding: utf-8 -*-
"""Offline validation of the routing design (src/rag_lab/router.py +
query_service.resolve_index/route_query) against the Gold 73-det query set,
reusing retrieval results already persisted by the 9-embedder matrix run and
the hybrid run -- no new retrieval or embedding calls.

Rewritten 2026-08-08. The previous version scored the 252-query
`gold_query_set.yaml` against 3 routes; both halves of that are now wrong:

- the 252 set pools two query shapes that carry *opposite* chunker signal
  (see docs/chunker-embedder-comparison-log.md), so it is never the set to
  make a routing decision on. This reads the 106-query 73det set.
- `classify_query` had 3 routes while the Gold set has 4 entity types. The
  33 `course` queries were added eight days after the router shipped and
  nothing failed -- they simply fell to `unmatched`, along with the 13
  `faculty_adjunct_aggregate` ones: 46/106 = 43% of the set silently
  unrouted. That is what this run exists to measure and close.

Three routed variants are reported, because "route to the best combo per
entity type" is easy to measure dishonestly:

- `shipped`   : the ROUTE_COMBO literal in router.py. Fixed config, nothing
                fitted at eval time -- this is the number that describes
                what actually runs, and the one to cite.
- `oracle`    : per-type argmax over all 36 combos, chosen on all 106
                queries and scored on the same 106. An UPPER BOUND, not a
                system -- printed only so the gap to `shipped` is visible.
- `loo`       : leave-one-out. For each query, each route's target is chosen
                using the other 105 queries only, then the held-out query is
                scored under it. This is the honest constructible estimate:
                it costs nothing extra (all 36x106 cells are already on
                disk) and removes the "argmax was picked on the test set"
                objection that `oracle` invites.

Retrieval budget is identical across every arm: each fetches k=10 from
exactly one index and sends 10 (per the equal-budget rule -- a routed system
that queried several indices would be spending more, not routing better).

Run with:
    .venv/Scripts/python.exe tools/eval/routing_eval.py
    .venv/Scripts/python.exe tools/eval/routing_eval.py --retriever dense
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedder_matrix_9way import (  # noqa: E402
    bootstrap_pvalue,
    build_combo_to_chunker_embedder,
    holm_correct,
)

from rag_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402
from rag_lab.query_service import discover_indices, resolve_index  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from rag_lab.router import ROUTE_COMBO, classify_query  # noqa: E402

GOLD_PATH = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
RESULT_DIRS = {
    "dense": REPO / "data" / "results" / "gold_73det_full_embedder_matrix",
    "hybrid": REPO / "data" / "results" / "gold_hybrid_73det",
}
OUTPUT = REPO / "data" / "results" / "routing_eval.md"
K = 10
N_BOOT = 10_000
SEED = 42

METRICS = {
    "recall@10": lambda r, rel: recall_at_k(r, rel, K),
    "mrr": lambda r, rel: reciprocal_rank(r, rel),
    "ndcg@10": lambda r, rel: ndcg_at_k(r, rel, K),
}


def load_gold() -> tuple[dict[str, list[str]], dict[str, str]]:
    entries = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))
    qrels = {e["query"]: e["relevant_resolution_ids"] for e in entries}
    etype = {e["query"]: e.get("entity_type", "unknown") for e in entries}
    return qrels, etype


def load_scores(
    result_dir: Path, combo_map: dict[str, tuple[str, str]], qrels: dict[str, list[str]]
) -> dict[tuple[str, str], dict[str, float]]:
    """(combo_key, query) -> {metric: score}, where combo_key is the raw
    combination_id. Results for superseded combos (not in combo_map) and for
    queries outside the 73det set are dropped."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for p in result_dir.glob("*.json"):
        r = load_retrieval_result(p)
        if r.combination_id not in combo_map or r.query not in qrels:
            continue
        rel = qrels[r.query]
        out[(r.combination_id, r.query)] = {m: fn(r, rel) for m, fn in METRICS.items()}
    return out


def shipped_combo_keys(retriever: str) -> dict[str, str]:
    """route -> combination_id for the ROUTE_COMBO literal, resolved through
    the production resolve_index path (not a hardcoded table), so this
    doubles as an integration check that every shipped route points at a
    real, unambiguous built index."""
    indices = discover_indices(INDEX_DIR)
    suffix = "__dense" if retriever == "dense" else "__hybrid"
    return {
        route: resolve_index(target, indices).combo_id + suffix
        for route, target in ROUTE_COMBO.items()
    }


def per_type_means(
    scores: dict[tuple[str, str], dict[str, float]],
    queries_by_type: dict[str, list[str]],
    combos: list[str],
    metric: str,
) -> dict[str, dict[str, float]]:
    """entity_type -> combo_key -> mean metric over that type's queries."""
    out: dict[str, dict[str, float]] = {}
    for etype, qs in queries_by_type.items():
        out[etype] = {}
        for combo in combos:
            vals = [scores[(combo, q)][metric] for q in qs if (combo, q) in scores]
            if vals:
                out[etype][combo] = statistics.mean(vals)
    return out


def argmax_combo(
    scores: dict[tuple[str, str], dict[str, float]],
    queries: list[str],
    combos: list[str],
    metric: str,
) -> str:
    """Best combo over `queries`, ties broken by combo_key for determinism."""
    best, best_score = None, -1.0
    for combo in sorted(combos):
        vals = [scores[(combo, q)][metric] for q in queries if (combo, q) in scores]
        if not vals:
            continue
        m = statistics.mean(vals)
        if m > best_score:
            best, best_score = combo, m
    return best


def routed_per_query(
    scores: dict[tuple[str, str], dict[str, float]],
    queries: list[str],
    route_of: dict[str, str],
    combos: list[str],
    metric: str,
    mode: str,
    shipped: dict[str, str],
) -> list[float]:
    """Per-query metric under one routed variant. Order matches `queries`."""
    if mode == "shipped":
        return [scores[(shipped[route_of[q]], q)][metric] for q in queries]

    if mode == "prev3":
        # the router as it stood until 2026-08-08: person/program only, with
        # course and faculty queries falling through to the unmatched default
        prev = {q: (r if r in ("person", "program") else "unmatched") for q, r in route_of.items()}
        return [scores[(shipped[prev[q]], q)][metric] for q in queries]

    by_route: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_route[route_of[q]].append(q)

    if mode == "oracle":
        chosen = {r: argmax_combo(scores, qs, combos, metric) for r, qs in by_route.items()}
        return [scores[(chosen[route_of[q]], q)][metric] for q in queries]

    if mode == "loo":
        out = []
        for q in queries:
            route = route_of[q]
            train = [x for x in by_route[route] if x != q]
            # a route with a single query has no training data of its own;
            # fall back to the shipped target rather than fitting on the
            # held-out query itself
            combo = argmax_combo(scores, train, combos, metric) if train else shipped[route]
            out.append(scores[(combo, q)][metric])
        return out

    raise ValueError(f"unknown mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever", choices=["dense", "hybrid", "both"], default="both")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()

    qrels, etype = load_gold()
    queries = sorted(qrels)
    combo_map = build_combo_to_chunker_embedder(INDEX_DIR)

    route_of = {q: classify_query(q) for q in queries}
    queries_by_type: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        queries_by_type[etype[q]].append(q)

    lines: list[str] = []
    out = lines.append
    out("# Query routing evaluation (Gold 73-det, n=%d)" % len(queries))
    out("")
    out("Generated by `tools/eval/routing_eval.py`. Reuses persisted retrieval results; ")
    out("no new embedding or retrieval calls. Every arm fetches k=%d from exactly one " % K)
    out("index and sends %d documents -- identical retrieval budget across all arms." % K)
    out("")

    # --- 1. classification accuracy -------------------------------------
    out("## 1. Classification vs hand-labelled `entity_type`")
    out("")
    out("| entity_type | n | routes assigned |")
    out("|---|---|---|")
    confusion: dict[str, Counter] = defaultdict(Counter)
    for q in queries:
        confusion[etype[q]][route_of[q]] += 1
    for et in sorted(confusion):
        row = ", ".join(f"`{r}`={n}" for r, n in confusion[et].most_common())
        out(f"| {et} | {sum(confusion[et].values())} | {row} |")
    out("")
    n_unrouted = sum(1 for q in queries if route_of[q] == "unmatched")
    out(f"Unrouted (`unmatched`): **{n_unrouted}/{len(queries)}** "
        f"({100.0 * n_unrouted / len(queries):.0f}%).")
    out("")

    retrievers = ["dense", "hybrid"] if args.retriever == "both" else [args.retriever]
    rng = np.random.default_rng(args.seed)

    for retriever in retrievers:
        # build_combo_to_chunker_embedder keys already carry a "__dense"
        # suffix; strip it so the same map serves the hybrid results too
        combo_suffix = f"__{retriever}"
        combo_label = {
            c.removesuffix("__dense") + combo_suffix: label for c, label in combo_map.items()
        }
        combos = list(combo_label)
        scores = load_scores(RESULT_DIRS[retriever], {c: combo_label[c] for c in combos}, qrels)
        present = {c for c, _ in scores}
        combos = [c for c in combos if c in present]
        shipped = shipped_combo_keys(retriever)

        out(f"## 2. Per-`entity_type` best combo -- {retriever} (evidence behind `ROUTE_COMBO`)")
        out("")
        means = per_type_means(scores, queries_by_type, combos, "recall@10")
        out("| entity_type | n | best combo | recall@10 | shipped route target | recall@10 | gap |")
        out("|---|---|---|---|---|---|---|")
        for et in sorted(queries_by_type):
            best = max(means[et], key=means[et].get)
            route = Counter(route_of[q] for q in queries_by_type[et]).most_common(1)[0][0]
            ship = shipped[route]
            bc, be = combo_label[best]
            sc, se = combo_label[ship]
            gap = means[et][best] - means[et].get(ship, float("nan"))
            out(f"| {et} | {len(queries_by_type[et])} | {bc}+{be} | {means[et][best]:.4f} "
                f"| `{route}` -> {sc}+{se} | {means[et].get(ship, float('nan')):.4f} | {gap:+.4f} |")
        out("")

        # --- 2b. is a per-route target stable enough to adopt? ------------
        out(f"### Target stability under leave-one-out -- {retriever}")
        out("")
        out("How many *different* combos the LOO selector picks for a route across its")
        out("folds. One distinct target means the choice does not depend on any single")
        out("query, so adopting it into `ROUTE_COMBO` is a refresh rather than a fit; more")
        out("than one means the route's top combos are interchangeable and the argmax is")
        out("noise. `n_train` is the fold size, i.e. that route's query count minus one.")
        out("")
        out("| route | n | n_train | distinct LOO targets (recall@10) | modal target | shipped |")
        out("|---|---|---|---|---|---|")
        by_route_q: dict[str, list[str]] = defaultdict(list)
        for q in queries:
            by_route_q[route_of[q]].append(q)
        for route in sorted(by_route_q):
            qs = by_route_q[route]
            picks = Counter(
                argmax_combo(scores, [x for x in qs if x != q], combos, "recall@10") for q in qs
            )
            modal, _ = picks.most_common(1)[0]
            mc, me = combo_label[modal]
            sc, se = combo_label[shipped[route]]
            same = "same" if modal == shipped[route] else f"{sc}+{se}"
            out(f"| `{route}` | {len(qs)} | {len(qs) - 1} | {len(picks)} "
                f"| {mc}+{me} ({picks[modal]}/{len(qs)} folds) | {same} |")
        out("")

        # --- 3. routed vs baselines --------------------------------------
        out(f"## 3. Routed system vs single-combo baselines -- {retriever}")
        out("")
        out("Baselines are matched to the routed arm's fitting budget: `best single combo`")
        out("is argmax over all 106 (an oracle, like `routed (oracle)`), `best single combo")
        out("(loo)` re-picks per held-out query (like `routed (loo)`). Comparing a LOO")
        out("routed arm against an oracle baseline would understate routing, and comparing")
        out("an oracle routed arm against a fixed baseline would overstate it.")
        out("")
        out("| metric | arm | mean | vs best-single |")
        out("|---|---|---|---|")
        per_metric_vectors: dict[str, dict[str, list[float]]] = {}
        for metric in METRICS:
            best_single = argmax_combo(scores, queries, combos, metric)
            single_loo = []
            for q in queries:
                train = [x for x in queries if x != q]
                single_loo.append(scores[(argmax_combo(scores, train, combos, metric), q)][metric])
            vectors = {
                "best single combo": [scores[(best_single, q)][metric] for q in queries],
                "best single combo (loo)": single_loo,
                "shipped unmatched default": [
                    scores[(shipped["unmatched"], q)][metric] for q in queries
                ],
            }
            for mode in ("prev3", "shipped", "oracle", "loo"):
                vectors[f"routed ({mode})"] = routed_per_query(
                    scores, queries, route_of, combos, metric, mode, shipped
                )
            per_metric_vectors[metric] = vectors
            ref = statistics.mean(vectors["best single combo"])
            bc, be = combo_label[best_single]
            for name, vec in vectors.items():
                note = f" = {bc}+{be}" if name == "best single combo" else ""
                out(f"| {metric} | {name}{note} | {statistics.mean(vec):.4f} "
                    f"| {statistics.mean(vec) - ref:+.4f} |")
        out("")

        # --- 4. significance ---------------------------------------------
        out(f"## 4. Paired bootstrap, routed vs baseline -- {retriever}")
        out("")
        out(f"{args.n_boot:,} resamples, seed {args.seed}, Holm-corrected within one family of ")
        out("all (routed variant x baseline x metric) pairs.")
        out("")
        # each routed arm is tested against the baseline fitted the same way
        matched_baseline = {
            "prev3": "best single combo",
            "shipped": "best single combo",
            "oracle": "best single combo",
            "loo": "best single combo (loo)",
        }
        pairs = []
        labels = []
        for metric in METRICS:
            vectors = per_metric_vectors[metric]
            for mode in ("shipped", "oracle", "loo"):
                for base in (matched_baseline[mode], "routed (prev3)"):
                    diffs = np.array(vectors[f"routed ({mode})"]) - np.array(vectors[base])
                    observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
                    pairs.append((f"routed ({mode})", base, observed, p, ci))
                    labels.append(metric)
        corrected = holm_correct(pairs, alpha=args.alpha)
        out(f"Family size m = **{len(corrected)}**.")
        out("")
        out("| metric | arm | baseline | diff | 95% CI | Holm-adj p | significant |")
        out("|---|---|---|---|---|---|---|")
        for metric, (a, b, diff, _p, ci, adj, sig) in zip(labels, corrected):
            out(f"| {metric} | {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] "
                f"| {adj:.4f} | {'**yes**' if sig else 'no'} |")
        out("")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
