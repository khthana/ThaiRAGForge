"""Per-entity_type, per-(chunker, embedder) breakdown of the Gold eval, for all
6 embedders, on the clean 73-deterministic query set (no thematic dilution).

Supersedes gold_embedder_breakdown.py for the "is ConGen really a program-
specialist / person-loser, and does that pattern hold for the 3 new
embedders?" question -- that script only covered e5/bge-m3/ConGen on the
252-query (thematic-diluted) set. This one reuses the combo->embedder
mapping approach from embedder_significance_test.py and reads from
data/results/gold_73det_full_embedder_matrix/ (all 24 combos, already
computed for the significance test).

Run with:
    .venv/Scripts/python.exe tools/eval/gold_embedder_breakdown_73det.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
_RESULTS_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "gold_embedder_breakdown_73det.md"
K = 10

_MODEL_LABELS = {
    "kornwtp/ConGen-BGE_M3-model-phayathaibert": "congen",
    "BAAI/bge-m3": "bge_m3",
    "Thaweewat/jina-embedding-v3-m2v-1024": "m2v",
}

CHUNKER_ORDER = ["fixed_size", "recursive", "semantic", "sentence"]
EMBEDDER_ORDER = ["e5", "bge_m3", "congen", "qwen3", "jina_v5", "m2v"]


def _embedder_label(combo: dict) -> str:
    etype = combo["embedder"]["type"]
    if etype != "local":
        return etype
    model_name = combo["embedder"]["params"]["model_name"]
    return _MODEL_LABELS.get(model_name, model_name)


def build_combo_map(index_dir: Path) -> dict[str, tuple[str, str]]:
    """combination_id (incl. __dense) -> (chunker, embedder)."""
    mapping = {}
    for d in sorted(index_dir.iterdir()):
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        combo = manifest["combo"]
        mapping[f"{d.name}__dense"] = (combo["chunker"]["type"], _embedder_label(combo))
    return mapping


def main() -> None:
    combo_map = build_combo_map(_INDEX_DIR)
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}

    entries_raw = __import__("yaml").safe_load(_GOLD_QUERY_SET.read_text(encoding="utf-8"))
    entity_type_by_query = {e["query"]: e.get("entity_type", "unknown") for e in entries_raw}
    queries_by_type: dict[str, list[str]] = defaultdict(list)
    for q, et in entity_type_by_query.items():
        queries_by_type[et].append(q)
    etypes = sorted(queries_by_type)

    persisted = [load_retrieval_result(p) for p in _RESULTS_DIR.glob("*.json")]
    by_combo_query = {(r.combination_id, r.query): r for r in persisted}

    # table[(chunker, embedder)][etype] -> {"recall": [...], "mrr": [...], "ndcg": [...]}
    table: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"recall": [], "mrr": [], "ndcg": []})
    )
    for combination_id, (chunker, embedder) in combo_map.items():
        for etype, queries in queries_by_type.items():
            for q in queries:
                r = by_combo_query.get((combination_id, q))
                relevant = qrels[q]
                if r is None:
                    table[(chunker, embedder)][etype]["recall"].append(0.0)
                    table[(chunker, embedder)][etype]["mrr"].append(0.0)
                    table[(chunker, embedder)][etype]["ndcg"].append(0.0)
                    continue
                table[(chunker, embedder)][etype]["recall"].append(recall_at_k(r, relevant, K))
                table[(chunker, embedder)][etype]["mrr"].append(reciprocal_rank(r, relevant))
                table[(chunker, embedder)][etype]["ndcg"].append(ndcg_at_k(r, relevant, K))

    lines = [
        "# Per-entity_type embedder breakdown (73-deterministic Gold set, all 6 embedders)",
        "",
        f"recall@{K}, per (chunker, embedder), broken out by entity_type. "
        f"Query counts: {', '.join(f'{et}={len(qs)}' for et, qs in sorted(queries_by_type.items()))}.",
        "",
    ]

    for chunker in CHUNKER_ORDER:
        lines.append(f"## chunker={chunker}")
        lines.append("")
        header = "| embedder | " + " | ".join(etypes) + " | overall |"
        sep = "|---|" + "---|" * (len(etypes) + 1)
        lines.append(header)
        lines.append(sep)
        for embedder in EMBEDDER_ORDER:
            row = table.get((chunker, embedder))
            if not row:
                continue
            cells = " | ".join(f"{statistics.mean(row[et]['recall']):.4f}" for et in etypes)
            overall = statistics.mean([s for et in etypes for s in row[et]["recall"]])
            lines.append(f"| {embedder} | {cells} | {overall:.4f} |")
        lines.append("")

    # Cross-chunker average per (embedder, entity_type) -- the "is it a specialist" summary
    lines.append("## Cross-chunker average per embedder x entity_type (recall@10)")
    lines.append("")
    lines.append("| embedder | " + " | ".join(etypes) + " | overall |")
    lines.append("|---|" + "---|" * (len(etypes) + 1))
    for embedder in EMBEDDER_ORDER:
        per_type_means = []
        for et in etypes:
            vals = [
                s
                for chunker in CHUNKER_ORDER
                for s in table.get((chunker, embedder), {}).get(et, {}).get("recall", [])
            ]
            per_type_means.append(statistics.mean(vals) if vals else float("nan"))
        overall_vals = [
            s
            for chunker in CHUNKER_ORDER
            for et in etypes
            for s in table.get((chunker, embedder), {}).get(et, {}).get("recall", [])
        ]
        overall = statistics.mean(overall_vals) if overall_vals else float("nan")
        cells = " | ".join(f"{v:.4f}" for v in per_type_means)
        lines.append(f"| {embedder} | {cells} | {overall:.4f} |")
    lines.append("")

    report = "\n".join(lines)
    _OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {_OUTPUT}")


if __name__ == "__main__":
    main()
