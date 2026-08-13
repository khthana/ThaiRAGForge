"""The per-chunker bars a one-chunker ColBERT pilot has to clear.

`docs/colbert-late-interaction-notes.md` registers the prediction against two
numbers -- BM25 `person` **0.8147** and the best dense embedder's `program`
**0.6066** (`qwen3_0.6b`) -- and both are **cross-chunker aggregates**. The
pilot builds *one* chunker (7.3 GB for all four; the card holds one at a time),
so comparing it to either figure would be the wrong-pair trap that killed
per-`entity_type` alpha and rrf4: chunker choice alone moves the BM25 `person`
bar across **0.7998 - 0.8281**, i.e. 0.028 of swing that has nothing to do with
late interaction.

So the pilot is scored against **its own chunker's** baselines, computed here,
and the published aggregates are re-entered only when all four chunkers exist.

Two conventions worth stating because they decide the bar:

* **`person` is BM25's bar and `program` is dense's**, per the prediction --
  they are not the same retriever, and that asymmetry *is* the complementarity
  the axis exists to test.
* For `program` the bar is `max(qwen3_0.6b, argmax over embedders)` **at that
  chunker**. The prediction names `qwen3_0.6b` because it is the aggregate
  argmax, but the aggregate argmax need not be the argmax at every chunker, and
  taking the larger of the two means a ColBERT win cannot be bought by the
  comparator getting weaker where the pilot happens to run.

Pure recompute from already-persisted top-10 results -- no retrieval, no GPU.

Run with:
    .venv/Scripts/python.exe tools/eval/colbert_pilot_baselines.py
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
    build_combo_to_chunker_embedder,
)

_BM25_RESULTS_DIR = REPO / "data" / "results" / "gold_bm25_73det"
_GOLD_QUERY_SET = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUTPUT = REPO / "data" / "results" / "colbert_pilot_baselines.md"

# The two published aggregates this script must reproduce, or it is not
# measuring the same thing the prediction was registered against.
_PUBLISHED = {
    ("bm25", "person"): 0.8147,
    ("qwen3_0.6b", "program"): 0.6066,
}
_ANCHOR_TOL = 5e-5

# Named in the prediction; the `program` bar may only ever move *up* from it.
_PREDICTION_EMBEDDER = "qwen3_0.6b"

# Truncation at the checkpoint's own doc_maxlen=300, from
# data/results/colbert_length_profile.md -- reported beside each chunker
# because it is the confound that decides which chunker the pilot should use.
_TRUNCATED_AT_300 = {
    "fixed_size": 0.024,
    "recursive": 0.011,
    "semantic": 0.074,
    "sentence": 0.032,
}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def per_query(persisted, combo_ce, suffix, qrels, types_by_query, k):
    """-> {(chunker, embedder): {entity_type: {query: recall}}}.

    Keyed per query rather than pre-averaged so an aggregate can be built with
    the same "average across chunkers per query first" convention the published
    tables use, instead of averaging averages.
    """
    acc = defaultdict(lambda: defaultdict(dict))
    for r in persisted:
        if not r.combination_id.endswith(suffix):
            continue
        base = r.combination_id[: -len(suffix)]
        if base not in combo_ce:
            continue
        chunker, embedder = combo_ce[base]
        etype = types_by_query.get(r.query)
        if etype is None:
            continue
        acc[(chunker, embedder)][etype][r.query] = recall_at_k(r, qrels[r.query], k)
    return acc


def program_mean(cells, chunker, embedder):
    """`.get`, never `[]` -- `cells` is a defaultdict during a real run, so
    indexing a combo that does not exist would fabricate an empty cell scoring
    0.0 and quietly lower a bar instead of being noticed."""
    return mean(list(cells.get((chunker, embedder), {}).get("program", {}).values()))


def across_chunkers(cells, embedder, etype):
    """Aggregate the published way: mean over queries of the per-query mean
    across the chunkers that scored it."""
    per_q = defaultdict(list)
    for (_chunker, emb), by_type in cells.items():
        if emb != embedder:
            continue
        for q, v in by_type.get(etype, {}).items():
            per_q[q].append(v)
    return mean([mean(v) for v in per_q.values()])


def load_cells(k: int):
    """-> `(bm25_cells, dense_cells)` from the persisted top-10 results.

    Factored out of `main` so `colbert_pilot.py` scores its comparator arms from
    **this** path rather than a second copy of it: the two would eventually
    disagree about a suffix or an excluded combo, and the disagreement would look
    like a ColBERT effect.
    """
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    raw = yaml.safe_load(_GOLD_QUERY_SET.read_text(encoding="utf-8"))
    types_by_query = {e["query"]: e.get("entity_type", "unknown") for e in raw}

    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)
    combo_ce = {c[: -len("__dense")]: v for c, v in combo_ce.items()}

    dense = [load_retrieval_result(p) for p in _DENSE_RESULTS_DIR.glob("*.json")]
    bm25 = [load_retrieval_result(p) for p in _BM25_RESULTS_DIR.glob("*.json")]
    print(f"loaded {len(dense)} dense + {len(bm25)} bm25 results")

    dense_cells = per_query(dense, combo_ce, "__dense", qrels, types_by_query, k)
    # BM25 is embedder-agnostic; collapse the embedder axis so the two tables
    # share one shape. Every combo sharing a chunker holds the same rows, so any
    # of them carries the identical BM25 result -- keyed "-" to say so.
    bm25_raw = per_query(bm25, combo_ce, "__bm25", qrels, types_by_query, k)
    bm25_cells = defaultdict(lambda: defaultdict(dict))
    for (chunker, _emb), by_type in bm25_raw.items():
        for t, per_q in by_type.items():
            bm25_cells[(chunker, "-")][t].update(per_q)
    return bm25_cells, dense_cells


def self_checks(bm25_cells, dense_cells, chunkers, embedders):
    out = []

    agg_bm25 = across_chunkers(bm25_cells, "-", "person")
    ok = abs(agg_bm25 - _PUBLISHED[("bm25", "person")]) < _ANCHOR_TOL
    out.append((
        "S1 BM25 `person` aggregate reproduces the published 0.8147",
        ok,
        f"{agg_bm25:.4f} vs {_PUBLISHED[('bm25', 'person')]:.4f}",
    ))

    agg_dense = across_chunkers(dense_cells, _PREDICTION_EMBEDDER, "program")
    ok = abs(agg_dense - _PUBLISHED[(_PREDICTION_EMBEDDER, "program")]) < _ANCHOR_TOL
    out.append((
        f"S2 dense `{_PREDICTION_EMBEDDER}` `program` aggregate reproduces the published 0.6066",
        ok,
        f"{agg_dense:.4f} vs {_PUBLISHED[(_PREDICTION_EMBEDDER, 'program')]:.4f}",
    ))

    # A bar built from one chunker is only meaningful if every chunker was
    # actually scored on the same queries -- a missing combo would silently
    # lower a bar rather than fail.
    counts = {c: len(bm25_cells[(c, "-")].get("person", {})) for c in chunkers}
    ok = len(set(counts.values())) == 1 and next(iter(counts.values())) > 0
    out.append((
        "S3 every chunker scored the same number of `person` queries",
        ok,
        ", ".join(f"{c}={n}" for c, n in sorted(counts.items())),
    ))

    dcounts = {
        c: len({e for (ch, e) in dense_cells if ch == c}) for c in chunkers
    }
    ok = len(set(dcounts.values())) == 1 and next(iter(dcounts.values())) == len(embedders)
    out.append((
        "S4 every chunker has the same embedder set",
        ok,
        ", ".join(f"{c}={n}" for c, n in sorted(dcounts.items())),
    ))

    # The `program` bar takes a max, so it can never sit below the embedder the
    # prediction names -- if it did, the pilot would be scored against a weaker
    # comparator than the registered one. Stated over the realised numbers, not
    # assumed from the expression: a missing cell means 0.0 and a NaN makes every
    # comparison False, so `max` alone does not guarantee what it looks like.
    viol = []
    for c in chunkers:
        vals = [program_mean(dense_cells, c, e) for e in embedders]
        named = program_mean(dense_cells, c, _PREDICTION_EMBEDDER)
        has_named = bool(dense_cells.get((c, _PREDICTION_EMBEDDER), {}).get("program"))
        if not has_named or not all(v == v for v in vals) or max(vals) + 1e-12 < named:
            viol.append(c)
    out.append((
        f"S5 the `program` bar is finite and never below `{_PREDICTION_EMBEDDER}` at any chunker",
        not viol,
        "none" if not viol else ", ".join(viol),
    ))

    return out


def render(bm25_cells, dense_cells, chunkers, embedders, ceilings, nq, checks):
    lines = [
        "# ColBERT pilot: the per-chunker bars",
        "",
        "Generated by `tools/eval/colbert_pilot_baselines.py`.",
        "",
        "The registered prediction names two cross-chunker aggregates (BM25 `person` "
        "**0.8147**, dense `program` **0.6066**). A one-chunker pilot cannot be scored "
        "against either -- chunker choice alone moves the BM25 `person` bar by 0.028. "
        "These are the same two quantities computed **at each chunker**, which is what "
        "the pilot is scored against; the aggregates are re-entered only when all four "
        "chunkers exist.",
        "",
        "## 1. The two bars, per chunker",
        "",
        "`person` is BM25's bar and `program` is dense's, per the prediction -- the "
        "asymmetry is the complementarity being tested. The `program` bar is "
        f"`max({_PREDICTION_EMBEDDER}, argmax over embedders)` at that chunker, so a "
        "ColBERT win cannot be bought by the comparator being weak where the pilot ran. "
        "`trunc@300` is the share of that chunker's chunks that lose their tail at the "
        "checkpoint's own `doc_maxlen`, i.e. the confound, and it points **against** the "
        "treatment.",
        "",
        "| chunker | `person` bar (BM25) | `program` bar | (which embedder) | "
        f"`{_PREDICTION_EMBEDDER}` `program` | trunc@300 |",
        "|---|---:|---:|---|---:|---:|",
    ]
    bars = {}
    for c in chunkers:
        person = mean(list(bm25_cells.get((c, "-"), {}).get("person", {}).values()))
        prog = {e: program_mean(dense_cells, c, e) for e in embedders}
        best = max(prog, key=lambda e: prog[e])
        named = prog[_PREDICTION_EMBEDDER]
        bar = max(prog[best], named)
        which = best if prog[best] >= named else _PREDICTION_EMBEDDER
        bars[c] = (person, bar, which)
        lines.append(
            f"| {c} | **{person:.4f}** | **{bar:.4f}** | `{which}` | {named:.4f} | "
            f"{_TRUNCATED_AT_300.get(c, float('nan')):.1%} |"
        )
    lines += [
        "",
        f"*(aggregate, for reference only: BM25 `person` "
        f"{across_chunkers(bm25_cells, '-', 'person'):.4f}, "
        f"`{_PREDICTION_EMBEDDER}` `program` "
        f"{across_chunkers(dense_cells, _PREDICTION_EMBEDDER, 'program'):.4f})*",
        "",
        "## 2. Spread of each bar across chunkers",
        "",
        "This is the quantity that makes a one-chunker pilot ambiguous if it is scored "
        "against an aggregate, and it is also what bounds how much a chunker swap could "
        "rescue a loss.",
        "",
        "| bar | min | max | spread |",
        "|---|---:|---:|---:|",
    ]
    pv = [bars[c][0] for c in chunkers]
    gv = [bars[c][1] for c in chunkers]
    lines.append(f"| `person` (BM25) | {min(pv):.4f} | {max(pv):.4f} | **{max(pv) - min(pv):.4f}** |")
    lines.append(f"| `program` (dense) | {min(gv):.4f} | {max(gv):.4f} | **{max(gv) - min(gv):.4f}** |")

    lines += [
        "",
        "## 3. Structural ceilings (unchanged, for reading the bars)",
        "",
        "| entity_type | n queries | ceiling |",
        "|---|---:|---:|",
    ]
    for t in sorted(ceilings):
        lines.append(f"| {t} | {nq[t]} | {ceilings[t]:.4f} |")

    lines += ["", "## 4. Self-checks", "", "| check | verdict | detail |", "|---|---|---|"]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    npass = sum(1 for _, ok, _ in checks if ok)
    lines += ["", f"**{npass} pass / {len(checks) - npass} fail** over {len(checks)} checks.", ""]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    k = args.k

    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    raw = yaml.safe_load(_GOLD_QUERY_SET.read_text(encoding="utf-8"))
    types_by_query = {e["query"]: e.get("entity_type", "unknown") for e in raw}

    ceil_acc, nq = defaultdict(list), defaultdict(int)
    for q, rel in qrels.items():
        t = types_by_query.get(q, "unknown")
        ceil_acc[t].append(min(1.0, k / len(rel)) if rel else 0.0)
        nq[t] += 1
    ceilings = {t: mean(v) for t, v in ceil_acc.items()}

    bm25_cells, dense_cells = load_cells(k)

    chunkers = sorted({c for (c, _e) in dense_cells})
    embedders = sorted({e for (_c, e) in dense_cells})
    print(f"chunkers: {chunkers}")
    print(f"embedders: {len(embedders)}")

    checks = self_checks(bm25_cells, dense_cells, chunkers, embedders)
    lines = render(bm25_cells, dense_cells, chunkers, embedders, ceilings, nq, checks)
    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  -- {detail}")
    print(f"\nwritten to {_OUTPUT}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
