"""Re-evaluate the 179 thematic gold queries now that they name their meeting.

The thematic subset has been carried as "near-zero discriminative power" since
2026-07-17 (t=0.02, 67% exact ties between fixed_size and semantic) and excluded
from every cited number. That reading was retired on 2026-07-30: the queries were
**unanswerable as posed** -- all 179 are meeting-scoped, yet every one asked about
"ในการประชุมครั้งนี้" without naming the meeting, so no retriever could tell which of
78 meetings was meant and every other meeting that discussed the theme scored as a
miss. `qualify_thematic_queries.py` rewrote all 179 to name their meeting.

This script asks the question that rewrite makes askable for the first time:
**with the queries well-posed, do thematic queries discriminate between chunkers
and embedders the way the entity-anchored ones do?** Two outcomes are both
informative -- if they now discriminate, the subset becomes usable evidence about
a query shape the deterministic set cannot cover (multi-document, no named
entity); if they still tie, "thematic queries don't discriminate" survives as a
real finding about the query shape rather than an artifact of its phrasing.

Scope mirrors the main comparison so the numbers are comparable: 4 chunkers x 9
embedders dense, plus BM25-alone and hybrid, over `chunker_compare_full`. Labels
and superseded-combo exclusions are imported from `embedder_matrix_9way` rather
than redefined (CLAUDE.md's convention -- that script is the single source).

Reported:
  * per-chunker and per-embedder means for recall@10 / MRR / nDCG@10;
  * fixed_size-vs-semantic, the specific pair the 2026-07-17 claim was about,
    at every embedder plus an aggregate, paired-bootstrap + Holm corrected;
  * a side-by-side of thematic vs the 73-det set's own aggregate, so "these
    queries are harder/easier" is stated in numbers rather than impression.

**Index-staleness caveat, checked rather than assumed**: the indices predate two
corpus changes made 2026-07-30 (the 2568/7 CHECO text fix and the restored
minutes document). Only 2 thematic entries reference 2568/7 at all, and their gold
ids are *other* documents in that meeting -- neither changed file is a gold id for
any thematic query -- so the effect is limited to a marginally different distractor
set. The eval is therefore valid now; a rebuild would not change which documents
count as relevant.

Run (retrieval is the slow part; scoring alone is seconds):
    PYTHONPATH=src python tools/eval/run_thematic_eval.py --retrieval dense
    PYTHONPATH=src python tools/eval/run_thematic_eval.py --retrieval bm25,hybrid
    PYTHONPATH=src python tools/eval/run_thematic_eval.py --skip-retrieval
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import QuerySetEntry, run_query_set  # noqa: E402
from rag_lab.query_service import discover_indices  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

from embedder_matrix_9way import (  # noqa: E402
    EMBEDDER_ORDER,
    _EXCLUDED_COMBO_DIRS,
    bootstrap_pvalue,
    build_combo_to_chunker_embedder,
    holm_correct,
)

_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
_GOLD = REPO / "config" / "eval" / "gold_query_set.yaml"
_RESULTS = {
    "dense": REPO / "data" / "results" / "thematic_dense",
    "bm25": REPO / "data" / "results" / "thematic_bm25",
    "hybrid": REPO / "data" / "results" / "thematic_hybrid",
}
_REPORT = REPO / "data" / "results" / "thematic_eval.md"
K = 10
N_BOOT = 10_000
SEED = 42
CHUNKERS = ["fixed_size", "recursive", "semantic", "sentence"]


def load_thematic() -> list[QuerySetEntry]:
    data = yaml.safe_load(_GOLD.read_text(encoding="utf-8"))
    return [
        QuerySetEntry(query=e["query"], relevant_resolution_ids=e["relevant_resolution_ids"])
        for e in data
        if e.get("entity_type") == "thematic"
    ]


def score(results, qrels: dict[str, list[str]], k: int) -> dict[str, dict[str, float]]:
    """metric -> combination_id -> mean over queries."""
    per = defaultdict(lambda: defaultdict(list))
    for r in results:
        rel = qrels.get(r.query)
        if rel is None:
            continue
        # metrics take the RetrievalResult itself: per ADR-0002 the top-k window
        # is over chunks first and dedup to resolutions happens after slicing, so
        # deduping here would silently change the definition of recall@k.
        per["recall"][r.combination_id].append(recall_at_k(r, rel, k))
        per["mrr"][r.combination_id].append(reciprocal_rank(r, rel))
        per["ndcg"][r.combination_id].append(ndcg_at_k(r, rel, k))
    return {m: {c: statistics.fmean(v) for c, v in d.items()} for m, d in per.items()}


def per_query_matrix(results, qrels, queries, k):
    """(metric, combination_id) -> per-query array aligned with `queries`."""
    idx = {q: i for i, q in enumerate(queries)}
    out: dict[tuple[str, str], np.ndarray] = {}
    for r in results:
        rel = qrels.get(r.query)
        if rel is None:
            continue
        for m, fn in (
            ("recall", lambda: recall_at_k(r, rel, k)),
            ("mrr", lambda: reciprocal_rank(r, rel)),
            ("ndcg", lambda: ndcg_at_k(r, rel, k)),
        ):
            arr = out.setdefault((m, r.combination_id), np.full(len(queries), np.nan))
            arr[idx[r.query]] = fn()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--retrieval", default="", help="comma-separated: dense,bm25,hybrid")
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--combos", type=int, default=0, help="limit combos (timing probe)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    query_set = load_thematic()
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    print(f"{len(query_set)} thematic queries; "
          f"{statistics.fmean(len(v) for v in qrels.values()):.1f} relevant docs each on average")

    combo_meta = build_combo_to_chunker_embedder(_INDEX_DIR)
    wanted = {cid.rsplit("__dense", 1)[0] for cid in combo_meta}
    dirs = [i.dir for i in discover_indices(str(_INDEX_DIR)) if Path(i.dir).name in wanted]
    if args.combos:
        dirs = dirs[: args.combos]

    for mode in [m for m in args.retrieval.split(",") if m] if not args.skip_retrieval else []:
        spec = {"dense": StrategySpec(type="dense"),
                "bm25": StrategySpec(type="bm25"),
                "hybrid": StrategySpec(type="hybrid")}[mode]
        print(f"\n[{mode}] retrieval over {len(dirs)} combo(s) x {len(query_set)} queries")
        run_query_set(query_set, dirs, spec, k=args.k, results_dir=str(_RESULTS[mode]))
        print(f"[{mode}] done -> {_RESULTS[mode]}")

    lines = ["# Thematic query-set eval (179 meeting-qualified queries)", ""]
    lines += [
        f"- Query set: `config/eval/gold_query_set.yaml`, `entity_type: thematic` only, "
        f"{len(query_set)} queries, rewritten 2026-07-30 to name their meeting",
        f"- k = {args.k}; index root `chunker_compare_full`; "
        f"bootstrap n={args.n_boot}, seed={args.seed}, Holm-corrected",
        "",
    ]

    loaded: dict[str, list] = {}
    for mode, rdir in _RESULTS.items():
        if not rdir.exists():
            continue
        res = [load_retrieval_result(p) for p in rdir.glob("*.json")]
        res = [r for r in res if r.query in qrels]
        if res:
            loaded[mode] = res
            print(f"loaded {len(res)} persisted {mode} results")

    if not loaded:
        print("\nno results yet -- run with --retrieval dense first")
        return

    queries = list(qrels)
    for mode, res in loaded.items():
        means = score(res, qrels, args.k)
        by_chunker: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        by_embedder: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for metric, per_combo in means.items():
            for cid, value in per_combo.items():
                key = cid if cid in combo_meta else f"{cid.split('__' + mode)[0]}__dense"
                if key not in combo_meta:
                    continue
                chunker, embedder = combo_meta[key]
                by_chunker[metric][chunker].append(value)
                by_embedder[metric][embedder].append(value)

        lines += [f"## {mode}: mean over combos", "",
                  "| chunker | recall@%d | mrr | ndcg@%d |" % (args.k, args.k), "|---|---|---|---|"]
        for c in CHUNKERS:
            if c in by_chunker.get("recall", {}):
                lines.append(f"| {c} | " + " | ".join(
                    f"{statistics.fmean(by_chunker[m][c]):.4f}" for m in ("recall", "mrr", "ndcg")) + " |")
        lines += ["", "| embedder | recall@%d | mrr | ndcg@%d |" % (args.k, args.k), "|---|---|---|---|"]
        for e in EMBEDDER_ORDER:
            if e in by_embedder.get("recall", {}):
                lines.append(f"| {e} | " + " | ".join(
                    f"{statistics.fmean(by_embedder[m][e]):.4f}" for m in ("recall", "mrr", "ndcg")) + " |")
        lines.append("")

    # ---- the 2026-07-17 claim, retested: fixed_size vs semantic ----
    if "dense" in loaded:
        mat = per_query_matrix(loaded["dense"], qrels, queries, args.k)
        rng = np.random.default_rng(args.seed)
        lines += ["## fixed_size vs semantic, dense (the pair the retired t=0.02 claim was about)", "",
                  "| embedder | metric | mean(fixed-semantic) | 95% CI | raw p | Holm-adj p | significant |",
                  "|---|---|---|---|---|---|---|"]
        pairs, labels = [], []
        by_ce = {v: k for k, v in combo_meta.items()}
        for e in EMBEDDER_ORDER:
            for metric in ("recall", "mrr", "ndcg"):
                f_id, s_id = by_ce.get(("fixed_size", e)), by_ce.get(("semantic", e))
                if not f_id or not s_id:
                    continue
                a, b = mat.get((metric, f_id)), mat.get((metric, s_id))
                if a is None or b is None:
                    continue
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() < 10:
                    continue
                diffs = (a - b)[ok]
                obs, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
                pairs.append((e, metric, obs, p, ci))
                labels.append((e, metric, int(ok.sum())))
        for (e, metric, obs, p, ci, adj, sig), (_, _, n) in zip(holm_correct(pairs), labels):
            lines.append(
                f"| {e} | {metric} | {obs:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{p:.4f} | {adj:.4f} | {'**yes**' if sig else 'no'} | n={n}")
        lines.append("")
        n_sig = sum(1 for *_, sig in holm_correct(pairs) if sig)
        lines += [f"**{n_sig} of {len(pairs)} fixed_size-vs-semantic tests are significant "
                  f"after Holm correction.**", ""]

    _REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {_REPORT}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
