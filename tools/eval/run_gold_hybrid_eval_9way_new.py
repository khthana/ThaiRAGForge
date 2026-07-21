"""Run Hybrid (RRF, BM25+Dense) retrieval for the 3 embedders added after the
original 6-embedder hybrid matrix: e5_small, qwen3_0.6b, sct (at its fixed
max_seq_length=510 -- NOT the superseded 128-cap combo).

Targets the 12 new combo dirs explicitly (3 embedders x 4 chunkers) rather
than a substring --embedder-filter, since "sct" would otherwise also match
the old, superseded 128-cap sct combos still on disk (see
docs/paper-results-summary.md "Resolved 2026-07-21" for why those are kept
around but must not be scored). Writes into the SAME results dir as the
original 6-embedder hybrid run so downstream significance tests (which glob
that whole directory) pick these up too.

Run with:
    .venv/Scripts/python.exe tools/eval/run_gold_hybrid_eval_9way_new.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.query_sets import load_gold_query_set, run_query_set  # noqa: E402

_INDEX_DIR = REPO / "data" / "index" / "chunker_compare_full"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_RESULTS_DIR = REPO / "data" / "results" / "gold_hybrid_73det"

_NEW_COMBO_DIRS = [
    "plain__fixed_size__local__2020e443", "plain__recursive__local__29632808",
    "plain__sentence__local__e83d75c9", "plain__semantic__local__f477fdca",  # sct, fixed (510)
    "plain__fixed_size__qwen3__72147070", "plain__recursive__qwen3__77daa38d",
    "plain__sentence__qwen3__ff8f6c49", "plain__semantic__qwen3__06058e0d",  # qwen3_0.6b
    "plain__fixed_size__e5__bcd966c9", "plain__recursive__e5__b1207dcb",
    "plain__sentence__e5__3ce2741a", "plain__semantic__e5__2dac4e98",  # e5_small
]


def main() -> None:
    query_set = load_gold_query_set(str(_GOLD_QUERY_SET))
    print(f"gold query set: {len(query_set)} queries")

    index_dirs = [str(_INDEX_DIR / name) for name in _NEW_COMBO_DIRS]
    for d in index_dirs:
        if not Path(d).exists():
            raise FileNotFoundError(d)
    print(f"scoring hybrid retrieval against {len(index_dirs)} new combos")

    t0 = time.time()
    run_query_set(
        query_set, index_dirs, StrategySpec(type="hybrid"), k=10, results_dir=str(_RESULTS_DIR)
    )
    print(f"retrieval done in {time.time() - t0:.1f}s, written to {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
