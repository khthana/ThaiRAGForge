"""9-embedder Gold 73-det matrix: retrieval + entity-type breakdown +
pairwise significance test, in one script.

Supersedes gold_embedder_breakdown_73det.py / embedder_significance_test.py
for this expanded matrix. Two bugs in those scripts' label logic would have
silently corrupted results if reused as-is for this expansion, both fixed
here:

1. `_embedder_label` returned the raw `type` string for non-"local"
   embedders ("e5", "qwen3") with no model_name disambiguation -- harmless
   with the original matrix (one model per type), but multilingual-e5-large
   vs -small and Qwen3-Embedding-4B vs -0.6B now share a type, so scores
   from two different models would silently pool into one label.
2. `build_combo_to_embedder` iterates every directory under
   data/index/chunker_compare_full with no exclusion list -- picks up BOTH
   the superseded 128-cap `sct` combos and the superseded (rejected) 510-cap
   `congen` combos (see docs/paper-results-summary.md "Resolved 2026-07-21"
   -- 510 helps sct, hurts congen, so the *correct* congen is the original
   128-cap and the *correct* sct is the new 510-cap). Without an exclusion
   list both would double up with their correct counterparts under the same
   label.

Run with:
    .venv/Scripts/python.exe tools/eval/embedder_matrix_9way.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set, run_query_set  # noqa: E402
from rag_lab.query_service import discover_indices  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
_RESULTS_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_SIG_OUTPUT = REPO / "data" / "results" / "embedder_significance_test_9way.md"
_BREAKDOWN_OUTPUT = REPO / "data" / "results" / "gold_embedder_breakdown_9way.md"
K = 10
N_BOOT = 10_000
SEED = 42

# superseded combo dirs: old 128-cap sct (fixed by the 510 rebuild) and the
# rejected 510-cap congen (510 significantly hurt congen, see
# docs/paper-results-summary.md "Resolved 2026-07-21"). Excluded so their
# labels don't collide with the correct counterpart below.
_EXCLUDED_COMBO_DIRS = {
    "plain__fixed_size__local__9d03b361", "plain__recursive__local__31293c05",
    "plain__sentence__local__d6c1f8e1", "plain__semantic__local__9576aa59",
    "plain__fixed_size__local__e6048946", "plain__recursive__local__4a350a4e",
    "plain__sentence__local__26622ae7", "plain__semantic__local__6b33a155",
}

_MODEL_LABELS = {
    ("local", "BAAI/bge-m3"): "bge_m3",
    ("local", "kornwtp/ConGen-BGE_M3-model-phayathaibert"): "congen",
    ("local", "kornwtp/SCT-KD-BGE-M3-model-phayathaibert"): "sct",
    ("local", "Thaweewat/jina-embedding-v3-m2v-1024"): "m2v",
    ("e5", "intfloat/multilingual-e5-large"): "e5",
    ("e5", "intfloat/multilingual-e5-small"): "e5_small",
    ("qwen3", "Qwen/Qwen3-Embedding-4B"): "qwen3",
    ("qwen3", "Qwen/Qwen3-Embedding-0.6B"): "qwen3_0.6b",
}
EMBEDDER_ORDER = ["e5", "e5_small", "bge_m3", "congen", "sct", "qwen3", "qwen3_0.6b", "jina_v5", "m2v"]


def _embedder_label(combo: dict) -> str:
    etype = combo["embedder"]["type"]
    if etype == "jina_v5":
        return "jina_v5"
    model_name = combo["embedder"]["params"]["model_name"]
    return _MODEL_LABELS.get((etype, model_name), f"{etype}:{model_name}")


def build_combo_to_embedder(index_dir: Path) -> dict[str, str]:
    mapping = {}
    for d in sorted(index_dir.iterdir()):
        if d.name in _EXCLUDED_COMBO_DIRS:
            continue
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        mapping[f"{d.name}__dense"] = _embedder_label(manifest["combo"])
    return mapping


def build_combo_to_chunker_embedder(index_dir: Path) -> dict[str, tuple[str, str]]:
    mapping = {}
    for d in sorted(index_dir.iterdir()):
        if d.name in _EXCLUDED_COMBO_DIRS:
            continue
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        combo = manifest["combo"]
        mapping[f"{d.name}__dense"] = (combo["chunker"]["type"], _embedder_label(combo))
    return mapping


def bootstrap_pvalue(diffs: np.ndarray, rng: np.random.Generator, n_boot: int):
    n = len(diffs)
    observed = diffs.mean()
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    p_le = float((boot_means <= 0).mean())
    p_ge = float((boot_means >= 0).mean())
    p_value = min(2 * min(p_le, p_ge), 1.0)
    ci = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    return observed, p_value, ci


def holm_correct(pairs: list[tuple], alpha: float = 0.05) -> list[tuple]:
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][3])
    m = len(pairs)
    adjusted = [None] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        holm_p = (m - rank) * pairs[i][3]
        running_max = max(running_max, holm_p)
        adjusted[i] = min(running_max, 1.0)
    return [(a, b, diff, p, ci, adjusted[i], adjusted[i] < alpha) for i, (a, b, diff, p, ci) in enumerate(pairs)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--skip-retrieval", action="store_true", help="reuse already-persisted results only")
    args = parser.parse_args()

    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}

    combo_to_embedder = build_combo_to_embedder(_INDEX_DIR)
    needed_labels = set(EMBEDDER_ORDER)
    present_labels = set(combo_to_embedder.values())
    missing = needed_labels - present_labels
    if missing:
        print(f"WARNING: no combo dirs found for labels: {missing}")

    if not args.skip_retrieval:
        # run_query_set has no dedup -- always re-runs and overwrites. Cheap
        # enough (query-time retrieval, not indexing) that re-running the
        # original 24 combos too is simpler and safer than trying to filter.
        all_indices = discover_indices(str(_INDEX_DIR))
        needed_dirs = [
            i.dir for i in all_indices
            if Path(i.dir).name in {cid.rsplit("__dense", 1)[0] for cid in combo_to_embedder}
        ]
        print(f"running retrieval for {len(needed_dirs)} combos")
        run_query_set(query_set, needed_dirs, StrategySpec(type="dense"), k=args.k, results_dir=str(_RESULTS_DIR))

    persisted = [load_retrieval_result(p) for p in _RESULTS_DIR.glob("*.json")]
    persisted = [r for r in persisted if r.combination_id in combo_to_embedder]
    print(f"loaded {len(persisted)} persisted retrieval results for the 9-embedder matrix")

    # ---- significance test (aggregate across chunkers) ----
    queries = list(qrels.keys())
    query_idx = {q: i for i, q in enumerate(queries)}
    n_q = len(queries)
    embedders = [e for e in EMBEDDER_ORDER if e in present_labels]
    sums = {m: {e: np.zeros(n_q) for e in embedders} for m in ("recall", "mrr", "ndcg")}
    counts = {m: {e: np.zeros(n_q) for e in embedders} for m in ("recall", "mrr", "ndcg")}
    for r in persisted:
        embedder = combo_to_embedder.get(r.combination_id)
        qi = query_idx.get(r.query)
        if embedder is None or qi is None:
            continue
        relevant = qrels[r.query]
        sums["recall"][embedder][qi] += recall_at_k(r, relevant, args.k)
        sums["mrr"][embedder][qi] += reciprocal_rank(r, relevant)
        sums["ndcg"][embedder][qi] += ndcg_at_k(r, relevant, args.k)
        counts["recall"][embedder][qi] += 1
        counts["mrr"][embedder][qi] += 1
        counts["ndcg"][embedder][qi] += 1

    per_query = {
        m: {e: np.divide(sums[m][e], counts[m][e], out=np.zeros(n_q), where=counts[m][e] > 0) for e in embedders}
        for m in ("recall", "mrr", "ndcg")
    }
    for m in ("recall", "mrr", "ndcg"):
        for e in embedders:
            missing_q = int((counts[m][e] == 0).sum())
            if missing_q:
                print(f"WARNING: {e} missing {missing_q}/{n_q} queries for {m}")

    rng = np.random.default_rng(args.seed)
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}
    n_pairs = len(list(itertools.combinations(embedders, 2)))

    sig_lines = [
        "# Embedder pairwise significance test -- 9-embedder matrix (Gold 73-det)",
        "",
        f"Paired bootstrap over {n_q} queries (n_boot={args.n_boot}, seed={args.seed}), each "
        f"embedder's per-query score averaged across the 4 chunker strategies first. "
        f"Holm-Bonferroni correction within each metric's {n_pairs} pairwise tests (alpha={args.alpha}).",
        "",
        "New vs the original 6-embedder matrix: `e5_small`, `qwen3_0.6b`, `sct` (re-added at its "
        "corrected max_seq_length=510; `congen` stays at its original, empirically-confirmed-correct 128).",
        "",
    ]
    for metric_key, metric_label in metric_labels.items():
        pairs = []
        for a, b in itertools.combinations(embedders, 2):
            diffs = per_query[metric_key][a] - per_query[metric_key][b]
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
            pairs.append((a, b, observed, p, ci))
        corrected = holm_correct(pairs, alpha=args.alpha)
        sig_lines.append(f"## {metric_label}")
        sig_lines.append("")
        sig_lines.append("| A | B | mean(A-B) | 95% CI | raw p | Holm-adj p | significant |")
        sig_lines.append("|---|---|---|---|---|---|---|")
        for a, b, diff, p, ci, holm_p, sig in sorted(corrected, key=lambda x: x[5]):
            mark = "**yes**" if sig else "no"
            sig_lines.append(f"| {a} | {b} | {diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |")
        sig_lines.append("")

    sig_lines.append("## Per-embedder mean")
    sig_lines.append("")
    sig_lines.append("| embedder | recall@{0} | mrr | ndcg@{0} |".format(args.k))
    sig_lines.append("|---|---|---|---|")
    for e in sorted(embedders, key=lambda e: -per_query["recall"][e].mean()):
        sig_lines.append(f"| {e} | {per_query['recall'][e].mean():.4f} | {per_query['mrr'][e].mean():.4f} | {per_query['ndcg'][e].mean():.4f} |")
    sig_lines.append("")

    _SIG_OUTPUT.write_text("\n".join(sig_lines), encoding="utf-8")
    print(f"written to {_SIG_OUTPUT}")

    # ---- entity-type breakdown ----
    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)
    entries_raw = __import__("yaml").safe_load(_GOLD_QUERY_SET.read_text(encoding="utf-8"))
    entity_type_by_query = {e["query"]: e.get("entity_type", "unknown") for e in entries_raw}
    queries_by_type: dict[str, list[str]] = defaultdict(list)
    for q, et in entity_type_by_query.items():
        queries_by_type[et].append(q)
    etypes = sorted(queries_by_type)

    by_combo_query = {(r.combination_id, r.query): r for r in persisted}
    table: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for combination_id, (chunker, embedder) in combo_ce.items():
        for etype, qs in queries_by_type.items():
            for q in qs:
                r = by_combo_query.get((combination_id, q))
                relevant = qrels[q]
                table[(embedder, etype)]["recall"].append(recall_at_k(r, relevant, args.k) if r else 0.0)

    bd_lines = [
        "# Cross-chunker average per embedder x entity_type -- 9-embedder matrix (recall@10, Gold 73-det)",
        "",
        "| embedder | " + " | ".join(etypes) + " | overall |",
        "|---|" + "---|" * (len(etypes) + 1),
    ]
    for embedder in embedders:
        cells, all_vals = [], []
        for et in etypes:
            vals = table.get((embedder, et), {}).get("recall", [])
            cells.append(f"{statistics.mean(vals):.4f}" if vals else "n/a")
            all_vals.extend(vals)
        overall = statistics.mean(all_vals) if all_vals else float("nan")
        bd_lines.append(f"| {embedder} | " + " | ".join(cells) + f" | {overall:.4f} |")
    bd_lines.append("")

    _BREAKDOWN_OUTPUT.write_text("\n".join(bd_lines), encoding="utf-8")
    print("\n".join(bd_lines))
    print(f"written to {_BREAKDOWN_OUTPUT}")


if __name__ == "__main__":
    main()
