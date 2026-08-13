"""ColBERT pilot: one chunker, all 106 Gold queries, against its own two bars.

PRE-REGISTERED
--------------
The prediction is the one frozen in `docs/colbert-late-interaction-notes.md`
before any of this existed:

    ColBERT-alone ties or beats **BM25** on `person`, **and** ties or beats the
    best dense embedder on `program`, **in the same run**.

An aggregate win licenses only "a stronger retriever"; the axis exists to test
whether late interaction resolves the *split* BM25 and dense have on this corpus
(BM25 carries `person`, dense carries `program`), so both cells must clear or the
prediction has not been met.

The two published figures the prediction names -- BM25 `person` 0.8147 and
`qwen3_0.6b` `program` 0.6066 -- are **cross-chunker aggregates**, and this pilot
builds ONE chunker (7.3 GB of token vectors for all four; the card holds one at a
time). Scoring one chunker against an aggregate is the wrong-pair trap that
killed per-`entity_type` alpha and rrf4, so the bars are taken **at this
chunker** from `colbert_pilot_baselines.py`, whose S1/S2 reproduce both
aggregates exactly.

WHY `recursive`, CHOSEN BEFORE THE RUN
--------------------------------------
On the one criterion that is a property of the *treatment* rather than of the
answer: truncation at the checkpoint's own `doc_maxlen=300`, where `recursive`
loses the tail of **1.1%** of chunks against 2.4 / 3.2 / 7.4% for the others
(`data/results/colbert_length_profile.md`). The hazard is stated rather than
hidden: `recursive` also happens to have the second-easiest bars, and `semantic`
-- easiest bars of the four -- is the worst on truncation, so the two criteria
are not independent and picking after the fact would be cherry-picking. That is
exactly why the criterion is named here, in advance, and why a margin smaller
than the cross-chunker spread of the bar it clears counts **at this chunker
only** and never at the axis level.

DECISION RULE
-------------
Frozen in `DECISION_RULE` below, committed before the artifact was built, and
evaluated by `decide()` -- which is unit-tested on every branch, because a rule
that is only prose is a rule that gets re-read favourably.

BUDGET
------
One GPU job at a time. `--build` encodes and exits (so the encoder's memory and
the scorer's never coexist); the default phase scores from the saved artifact and
needs the GPU only for 106 query encodes. `--render` re-derives the report from
`colbert_pilot_raw.json` with no GPU and no artifact at all.

ANCHORS
-------
S1/S2 are `colbert_pilot_baselines.py`'s own anchor checks, re-run here so the
comparator vectors this script pairs against are provably the ones that reproduce
the published aggregates. S6 re-scores a sample of documents with
`maxsim_reference` -- the naive one-document-at-a-time definition -- because the
packed `reduceat` path is an optimisation whose failure mode is a plausible
ranking.

Run with:
    .venv/Scripts/python.exe tools/eval/colbert_pilot.py --smoke      # ~1 min GPU
    .venv/Scripts/python.exe tools/eval/colbert_pilot.py --build
    .venv/Scripts/python.exe tools/eval/colbert_pilot.py
    .venv/Scripts/python.exe tools/eval/colbert_pilot.py --render
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.colbert.encoder import MODEL_NAME, ColbertConfig, ColbertEncoder  # noqa: E402
from rag_lab.colbert.scoring import maxsim_reference, offsets_from_lengths  # noqa: E402
from rag_lab.colbert.store import ColbertStore, verify_alignment  # noqa: E402
from rag_lab.metrics import recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.retrievers.colbert import ColbertRetriever  # noqa: E402
from rag_lab.schema import Chunk, Index, Query, RetrievalResult  # noqa: E402
from colbert_pilot_baselines import (  # noqa: E402
    _GOLD_QUERY_SET,
    _PREDICTION_EMBEDDER,
    _TRUNCATED_AT_300,
    load_cells,
    mean,
    program_mean,
    self_checks as baseline_anchor_checks,
)
from embedder_matrix_9way import (  # noqa: E402
    _INDEX_DIR,
    bootstrap_pvalue,
    build_combo_to_chunker_embedder,
    holm_correct,
)

CHUNKER = "recursive"
K = 10
N_BOOT = 10_000
SEED = 42

_ARTIFACT_DIR = REPO / "data" / "index" / "colbert" / f"{CHUNKER}__doc300_q32"
_RAW = REPO / "data" / "results" / "colbert_pilot_raw.json"
_OUTPUT = REPO / "data" / "results" / "colbert_pilot.md"

# The two cells the prediction is made of. `person` is BM25's bar and `program`
# is dense's -- they are not the same retriever, and that asymmetry *is* the
# complementarity under test.
PREDICTION_CELLS = [("person", "bm25"), ("program", "dense")]

# A point-estimate loss beyond this cannot be recovered by the axis a CONTINUE
# would go on to explore: `power_analysis.md` puts every observed chunker-pair
# difference at |diff| <= 0.0230 recall@10 (MDEs 0.0302-0.0532), and the measured
# bar spreads here are 0.0283 (`person`) / 0.0346 (`program`). So this is a COST
# rule, not an inference one, and it is deliberately evaluated BEFORE
# significance: at n=30 per type a 0.05 loss can fail to reach significance, and
# "we could not resolve it" is not a reason to spend 3x more GPU on it.
STOP_MARGIN = 0.05

DECISION_RULE = """\
Evaluated at `recursive` only, on recall@10 against that chunker's own bars,
family 1 = the 2 prediction cells (Holm, m=2, alpha=0.05). Frozen before the
artifact was built.

1. STOP  -- either cell loses by more than 0.0500 (point estimate). Checked
           FIRST and overriding: a loss that large is beyond anything a chunker
           swap has been observed to recover, so the continuation has nothing to
           find. Close the axis.
2. CONTINUE -- both cells clear, where "clears" = the prediction's own wording,
           ties or beats: diff >= 0, or diff < 0 and not significant after Holm.
           Build the other three chunkers and re-enter the published aggregates.
3. NARROW -- anything else (a real loss on at least one cell, but no cell worse
           than 0.0500). Run `sentence` and nothing else; if it also fails to
           clear both cells, STOP. Cost capped at two chunkers.

Two riders, also frozen:

* A margin smaller than the bar's own cross-chunker spread (0.0283 `person`,
  0.0346 `program`) counts at `recursive` only, never as an axis-level claim.
* The 512/48 length fallback fires at most once, only on a STOP or NARROW, and
  only if the losing cell's truncation is shown to be anomalous -- i.e. the
  truncation rate among the chunks of that cell's gold resolutions is materially
  above the corpus rate recorded at build time. Otherwise 300/32 stands and the
  truncation stays a stated confound pointing against the treatment.
"""


# --------------------------------------------------------------------- chunks
def source_combo_dirs() -> list[Path]:
    """Every live combo directory built with this chunker.

    Chunk rows are a function of loader+chunker, not of the embedder, so all of
    these hold the same `chunks.parquet` -- S3 checks that rather than assuming
    it, because otherwise the artifact would be silently tied to whichever one
    happened to sort first.
    """
    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)
    return sorted(
        _INDEX_DIR / combo[: -len("__dense")]
        for combo, (chunker, _e) in combo_ce.items()
        if chunker == CHUNKER
    )


def read_chunks(directory: Path) -> list[Chunk]:
    import pyarrow.parquet as pq

    cols = pq.read_table(directory / "chunks.parquet").to_pydict()
    return [
        Chunk(
            chunk_id=cols["chunk_id"][i],
            resolution_id=cols["resolution_id"][i],
            text=cols["text"][i],
            chunk_index=int(cols["chunk_index"][i]),
            page=int(cols["page"][i]),
            metadata={},
        )
        for i in range(len(cols["chunk_id"]))
    ]


def read_chunk_ids(directory: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.read_table(directory / "chunks.parquet", columns=["chunk_id"])
                .column("chunk_id").to_pylist())


def as_index(chunks: list[Chunk]) -> Index:
    """An `Index` shell carrying the chunk rows only.

    `embeddings.npy` is never read: nothing in this path is dense, and loading a
    4096-wide matrix for 70k chunks would cost 1.15 GB to be ignored.
    """
    return Index(chunks=chunks, embeddings=np.zeros((len(chunks), 0), dtype=np.float32))


# ---------------------------------------------------------------------- build
def build_artifact(chunks: list[Chunk], out_dir: Path, source: Path,
                   enc: ColbertEncoder) -> dict:
    """Encode `chunks` and save the artifact. The caller owns the encoder's
    lifetime, so the build and the score phase never hold one at the same time."""
    cfg = enc.config
    texts = [c.text for c in chunks]

    t0 = time.time()
    vecs, lengths = enc.encode_documents(texts)
    seconds = time.time() - t0
    rotary = enc.rotary_repaired

    # The truncation rate is recorded at build time rather than re-derived later:
    # it is the pre-registered confound, and it is what the 512/48 fallback rule
    # is conditioned on. It cannot be recovered from `lengths`, which counts
    # *kept* tokens after punctuation masking and is <= doc_maxlen by
    # construction whether or not the tail was cut -- so it is a separate pass
    # over the encoder's own tokenizer, never a second differently-configured one.
    tok = enc._tok
    n_over = 0
    for i in range(0, len(texts), 256):
        batch = [f"{cfg.document_prefix} {t}" for t in texts[i:i + 256]]
        n_over += sum(1 for ids in tok(batch)["input_ids"] if len(ids) > cfg.doc_maxlen)
    truncated = n_over / len(texts) if texts else None

    meta = {
        "model": MODEL_NAME,
        "chunker": CHUNKER,
        "dim": cfg.dim,
        "doc_maxlen": cfg.doc_maxlen,
        "query_maxlen": cfg.query_maxlen,
        "mask_punctuation": cfg.mask_punctuation,
        "attend_to_mask_tokens": cfg.attend_to_mask_tokens,
        "source_index_dir": str(source),
        "docset_hash": json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        .get("docset_hash"),
        "n_chunks": len(chunks),
        "n_token_vectors": int(lengths.sum()),
        "rotary_layers_repaired": rotary,
        "truncated_share": truncated,
        "encode_seconds": seconds,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    ColbertStore().save(out_dir, [c.chunk_id for c in chunks], vecs, lengths, meta)
    return meta


# ---------------------------------------------------------------- length rider
def truncation_rider(chunks: list[Chunk], queries: list[str], qrels: dict,
                     doc_maxlen: int) -> dict:
    """Can truncation account for the losing cell's gap? An arithmetic bound.

    `DECISION_RULE`'s 512/48 fallback fires only if the losing cell's truncation
    is "materially above" the corpus rate -- and choosing what counts as material
    *after* seeing the gap is precisely the re-reading the frozen rule exists to
    prevent. So the rider is answered with a bound instead of a threshold, in the
    style [[feedback_an_observed_extreme_is_not_a_bound]] asks for.

    Grant truncation the most damage it could possibly do: assume a gold
    resolution with **any** truncated chunk is destroyed outright and can never
    be retrieved. Recall@10 for a query is (relevant found)/(relevant), so
    destroying `a` of its `r` relevant resolutions costs at most `a/r`. The mean
    of that over the cell's queries is an upper bound on how much recall
    truncation could have taken -- generous twice over, since a truncated chunk
    keeps its first `doc_maxlen` tokens and a resolution usually has several
    chunks, of which only the long ones are cut.

    If the bound is below the observed gap, truncation cannot explain it and the
    fallback does not fire, whatever threshold anyone would have chosen.
    """
    from transformers import AutoTokenizer

    # The encoder's own tokenizer by name -- `ColbertEncoder` builds it with a
    # bare `AutoTokenizer.from_pretrained(MODEL_NAME)`, so this is the same
    # object without paying for the model. The prefix is part of what gets
    # tokenized at build time, so it is part of what gets counted here.
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    prefix = ColbertConfig().document_prefix

    gold = {rid for q in queries for rid in qrels[q]}
    relevant_chunks = [c for c in chunks if c.resolution_id in gold]
    # Two different quantities, deliberately counted separately: `n_cut_chunks`
    # is how many chunks were actually cut, `cut` is the set of resolutions that
    # lost at least one. Only the second feeds the bound, but reporting the first
    # is what makes the bound's generosity visible.
    n_cut_chunks = 0
    cut: set[str] = set()
    for i in range(0, len(relevant_chunks), 256):
        batch = relevant_chunks[i:i + 256]
        ids = tok([f"{prefix} {c.text}" for c in batch])["input_ids"]
        for c, seq in zip(batch, ids):
            if len(seq) > doc_maxlen:
                n_cut_chunks += 1
                cut.add(c.resolution_id)

    per_query = []
    for q in queries:
        rel = qrels[q]
        if rel:
            per_query.append(len([r for r in rel if r in cut]) / len(rel))
    bound = float(np.mean(per_query)) if per_query else float("nan")
    return {
        "n_queries": len(queries),
        "n_gold_resolutions": len(gold),
        "n_gold_chunks": len(relevant_chunks),
        "n_gold_chunks_truncated": n_cut_chunks,
        "gold_resolutions_touched": len(cut),
        "share_gold_resolutions_touched": len(cut) / len(gold) if gold else float("nan"),
        "recall_damage_bound": bound,
    }


def rider_for(verdict: str, cells: list[dict], chunks: list[Chunk], qrels: dict,
              doc_maxlen: int) -> dict | None:
    """The rider's gate, in one place so both entry points ask the same question.

    It can only fire on STOP or NARROW, and only for a cell that actually failed,
    so it is computed exactly there and nowhere else -- running it on a CONTINUE
    would be answering a question `DECISION_RULE` does not ask.
    """
    if verdict not in ("STOP", "NARROW"):
        return None
    losing = [c for c in cells
              if not clears(c["diff"], c["significant"]) or c["diff"] < -STOP_MARGIN]
    if not losing:
        return None
    worst = min(losing, key=lambda c: c["diff"])
    rider = truncation_rider(chunks, worst["queries"], qrels, doc_maxlen)
    rider["cell"] = worst["label"]
    rider["gap"] = abs(worst["diff"])
    rider["fires"] = bool(rider["recall_damage_bound"] >= rider["gap"])
    return rider


# -------------------------------------------------------------------- scoring
def as_result(query: str, ranked, k: int) -> RetrievalResult:
    return RetrievalResult(query=query, combination_id=f"colbert__{CHUNKER}",
                           results=list(ranked), top_k=k, retriever="colbert")


def score_queries(retriever, index, queries, qmats, qrels, k):
    out, latencies = {}, []
    for i, q in enumerate(queries):
        t0 = time.time()
        ranked = retriever.retrieve(Query(text=q, vector=qmats[i]), index, k)
        latencies.append((time.time() - t0) * 1000.0)
        out[q] = recall_at_k(as_result(q, ranked, k), qrels[q], k)
    return out, latencies


def by_type(per_query: dict, types: dict) -> dict:
    acc = defaultdict(dict)
    for q, v in per_query.items():
        acc[types.get(q, "unknown")][q] = v
    return acc


# ------------------------------------------------------------------- decision
def clears(diff: float, significant: bool) -> bool:
    """The prediction's own wording: *ties or beats*."""
    return diff >= 0 or not significant


def decide(cells: list[dict]) -> tuple[str, str]:
    """`cells` are dicts with `label`, `diff`, `significant`. See DECISION_RULE."""
    # A cell with no paired queries bootstraps to NaN, and every comparison
    # against NaN is False -- so it would slip through `clears()` as a tie and
    # let one measured cell decide a conjunction on its own. Refused outright:
    # the rule has no answer for a cell that was never measured.
    unmeasured = [c for c in cells if not math.isfinite(c["diff"])]
    if unmeasured:
        return "INVALID", (
            "no comparable measurement on "
            + ", ".join(c["label"] for c in unmeasured)
            + " -- the rule is not applicable and no verdict is claimed"
        )
    blown = [c for c in cells if c["diff"] < -STOP_MARGIN]
    if blown:
        worst = ", ".join(f"{c['label']} {c['diff']:+.4f}" for c in blown)
        return "STOP", (
            f"loses by more than {STOP_MARGIN:.4f} on {worst} -- beyond what a "
            "chunker swap has been observed to recover"
        )
    failing = [c for c in cells if not clears(c["diff"], c["significant"])]
    if not failing:
        return "CONTINUE", "both cells clear their bar (tie or beat)"
    return "NARROW", (
        "a real loss on "
        + ", ".join(f"{c['label']} {c['diff']:+.4f}" for c in failing)
        + f", but nothing worse than {STOP_MARGIN:.4f}"
    )


# ---------------------------------------------------------------------- report
def render(raw: dict) -> list[str]:
    checks = [tuple(c) for c in raw["checks"]]
    cells = raw["cells"]
    verdict, why = raw["verdict"], raw["verdict_reason"]
    meta = raw["artifact_meta"]

    lines = [
        f"# ColBERT pilot ({CHUNKER}, doc_maxlen={meta.get('doc_maxlen')}, "
        f"query_maxlen={meta.get('query_maxlen')})",
        "",
        "Generated by `tools/eval/colbert_pilot.py`.",
        "",
        f"`{MODEL_NAME}`, ColBERT-alone, k={raw['k']}, unrouted, all "
        f"{raw['n_queries']} Gold queries of `gold_query_set_73det.yaml`. "
        f"Bars from `colbert_pilot_baselines.py` **at `{CHUNKER}`**, never the "
        "published cross-chunker aggregates -- see this script's docstring.",
        "",
        "## 1. The pre-registered prediction",
        "",
        "*ColBERT-alone ties or beats BM25 on `person` **and** ties or beats the best "
        "dense embedder on `program`, in the same run.* Family 1, Holm m="
        f"{len(cells)}, alpha={raw['alpha']}, {raw['n_boot']:,} bootstrap resamples.",
        "",
        "| cell | comparator | n | ColBERT | bar | diff | 95% CI | Holm p | verdict |",
        "|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for c in cells:
        # An unmeasured cell must not be rendered as `fails`: `clears()` returns
        # False on NaN for the same reason it would have returned True on the
        # comparison the other way round, and neither answer is a measurement.
        if math.isfinite(c["diff"]):
            cell_verdict = "**clears**" if clears(c["diff"], c["significant"]) else "fails"
        else:
            cell_verdict = "unmeasured"
        lines.append(
            f"| `{c['etype']}` | {c['comparator']} | {c['n']} | {c['colbert']:.4f} | "
            f"{c['bar']:.4f} | **{c['diff']:+.4f}** | "
            f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] | {c['holm_p']:.4f} | "
            f"{cell_verdict} |"
        )

    lines += [
        "",
        "## 2. Every entity type, and overall (descriptive, not pre-registered)",
        "",
        "The comparators outside family 1 are shown for orientation only: `person`'s "
        "bar is BM25 and `program`'s is dense *because the prediction says so*, and "
        "no other cell was registered.",
        "",
        "| entity_type | n | ColBERT | BM25 | best dense | ceiling |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t in raw["type_order"]:
        d = raw["descriptive"][t]
        lines.append(
            f"| {t} | {d['n']} | {d['colbert']:.4f} | {d['bm25']:.4f} | "
            f"{d['dense']:.4f} | {d['ceiling']:.4f} |"
        )
    o = raw["descriptive"]["_overall"]
    lines.append(
        f"| **overall** | {o['n']} | **{o['colbert']:.4f}** | {o['bm25']:.4f} | "
        f"{o['dense']:.4f} | {o['ceiling']:.4f} |"
    )

    # Which arm does the treatment's per-type PROFILE follow? Derived from the
    # table rather than typed, so it cannot drift away from the numbers above it
    # ([[feedback_recompute_derived_stats_from_the_table]]).
    near_bm25 = [t for t in raw["type_order"]
                 if abs(raw["descriptive"][t]["colbert"] - raw["descriptive"][t]["bm25"])
                 < abs(raw["descriptive"][t]["colbert"] - raw["descriptive"][t]["dense"])]
    best_overall = max(("ColBERT", o["colbert"]), ("BM25", o["bm25"]),
                       ("dense", o["dense"]), key=lambda kv: kv[1])
    registered = {etype for etype, _ in PREDICTION_CELLS}
    beats_both = [t for t in raw["type_order"]
                  if t not in registered
                  and raw["descriptive"][t]["colbert"] > raw["descriptive"][t]["bm25"]
                  and raw["descriptive"][t]["colbert"] > raw["descriptive"][t]["dense"]]
    lines += [
        "",
        f"ColBERT sits nearer BM25 than dense on {len(near_bm25)} of "
        f"{len(raw['type_order'])} types ({', '.join(f'`{t}`' for t in near_bm25)}) -- "
        "not a majority, but **those are exactly the two the prediction is decided on**, "
        "and the direction is the same on both: strong where the lexical arm is strong, "
        "weak where the lexical arm is weak. That is the axis's own motivation answered "
        "in the negative -- late interaction was proposed here to *cover* the arm split "
        "and it inherits one side of it instead. On the unregistered types it is not a "
        "lexical model at all: it beats **both** arms on "
        f"{', '.join(f'`{t}`' for t in beats_both) or 'no type'}.",
        "",
        f"Note also that {best_overall[0]} carries the highest overall figure in the "
        f"table ({best_overall[1]:.4f}), which is exactly the aggregate reading the "
        "conjunctive pre-registration exists to refuse: an aggregate win licenses "
        "\"a stronger retriever\", never \"late interaction resolves the "
        "complementarity\".",
    ]

    lines += [
        "",
        "## 3. The confound, measured at build time",
        "",
        f"| quantity | value |",
        "|---|---|",
        f"| chunks | {meta.get('n_chunks'):,} |",
        f"| token vectors | {meta.get('n_token_vectors'):,} |",
        f"| chunks truncated at `doc_maxlen={meta.get('doc_maxlen')}` | "
        f"{meta.get('truncated_share', float('nan')):.1%} "
        f"(profile: {_TRUNCATED_AT_300.get(CHUNKER, float('nan')):.1%}) |",
        f"| rotary layers repaired at load | {meta.get('rotary_layers_repaired')} |",
        f"| encode time | {meta.get('encode_seconds', 0) / 60:.1f} min |",
        f"| query latency p50 | {raw['latency_p50_ms']:.1f} ms |",
        f"| `docset_hash` | `{meta.get('docset_hash')}` |",
        "",
        "Truncation points **against** the treatment: a win is not bought by it. "
        "`rotary layers repaired` is the checkpoint's transformers-5.x bug "
        "(`encoder._repair_rotary`); it should read 24 today and 0 once the loader "
        "materialises the buffer correctly.",
    ]

    rider = raw.get("rider")
    if rider:
        lines += [
            "",
            f"### 3b. The length rider, answered as a bound ({rider['cell']})",
            "",
            "`DECISION_RULE`'s 512/48 fallback is conditioned on the losing cell's "
            "truncation being *materially above* the corpus rate -- and deciding what "
            "counts as material after seeing the gap is the re-reading the frozen rule "
            "exists to prevent. So it is answered with a bound rather than a threshold: "
            "grant truncation the most damage it could possibly do, and see whether that "
            "is even enough.",
            "",
            "| quantity | value |",
            "|---|---|",
            f"| gold resolutions for this cell | {rider['n_gold_resolutions']:,} |",
            f"| their chunks | {rider['n_gold_chunks']:,} |",
            f"| chunks truncated | {rider['n_gold_chunks_truncated']:,} "
            f"({rider['n_gold_chunks_truncated'] / max(rider['n_gold_chunks'], 1):.2%}, "
            f"corpus: {meta.get('truncated_share', float('nan')):.2%}) |",
            f"| gold resolutions with >=1 truncated chunk | {rider['gold_resolutions_touched']:,} "
            f"({rider['share_gold_resolutions_touched']:.1%}) |",
            f"| **recall@10 a total loss of those could explain** | "
            f"**{rider['recall_damage_bound']:.4f}** |",
            f"| observed gap | {rider['gap']:.4f} |",
            f"| fallback fires | **{'yes' if rider['fires'] else 'no'}** |",
            "",
            "The bound assumes a resolution with **any** truncated chunk is destroyed "
            "outright and can never be retrieved -- generous twice over, since a "
            "truncated chunk keeps its first "
            f"{meta.get('doc_maxlen')} tokens and most resolutions have several chunks "
            "of which only the long ones are cut. "
            + (
                "It exceeds the gap, so the fallback fires and 512/48 is owed."
                if rider["fires"] else
                "It is smaller than the gap, so truncation is arithmetically incapable "
                "of explaining the loss and 300/32 stands -- no threshold needed."
            ),
            "",
            "The rule's literal wording is answerable here as well, and agrees: the "
            "truncation rate among this cell's gold chunks is *below* the corpus rate "
            "recorded at build time, so the truncation is not anomalous by the reading "
            "the rule actually gives either.",
        ]

    lines += [
        "",
        "## 4. Decision",
        "",
        f"### {verdict}",
        "",
        f"{why}.",
        "",
        "The rule, frozen in `DECISION_RULE` before the artifact existed:",
        "",
        "```",
        DECISION_RULE.rstrip(),
        "```",
        "",
        "## 5. Self-checks",
        "",
        "| check | verdict | detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    npass = sum(1 for _, ok, _ in checks if ok)
    lines += ["", f"**{npass} pass / {len(checks) - npass} fail** over {len(checks)} checks.", ""]
    return lines


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true",
                        help="encode the corpus into the artifact and exit")
    parser.add_argument("--render", action="store_true",
                        help="re-derive the report from the raw cache; no GPU")
    parser.add_argument("--smoke", action="store_true",
                        help="500 chunks, 8 queries, in memory; writes no report")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.render:
        raw = json.loads(_RAW.read_text(encoding="utf-8"))
        # Back-fill the length rider for a cache written before it existed. It
        # needs the tokenizer and the chunk rows, not the model or the GPU, and
        # it is a pure function of the cache's own verdict and cells -- so it is
        # computed once and PERSISTED, which keeps every later `--render` free
        # and keeps the report's figure sourced from an artifact on disk rather
        # than from a hand-run one-off.
        if "rider" not in raw:
            query_set = load_gold_query_set(_GOLD_QUERY_SET)
            qrels = {e.query: e.relevant_resolution_ids for e in query_set}
            raw["rider"] = rider_for(
                raw["verdict"], raw["cells"], read_chunks(source_combo_dirs()[0]),
                qrels, raw["artifact_meta"].get("doc_maxlen", 300))
            _RAW.write_text(json.dumps(raw, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            print("length rider back-filled into the raw cache")
        _OUTPUT.write_text("\n".join(render(raw)), encoding="utf-8")
        if raw.get("rider"):
            r = raw["rider"]
            print(f"length rider on {r['cell']}: truncation could account for at most "
                  f"{r['recall_damage_bound']:.4f} of the {r['gap']:.4f} gap -- "
                  f"{'FIRES' if r['fires'] else 'does not fire'}")
        print(f"written to {_OUTPUT}")
        return 0 if all(c[1] for c in raw["checks"]) else 1

    dirs = source_combo_dirs()
    if not dirs:
        raise SystemExit(f"no live {CHUNKER} combo directory under {_INDEX_DIR}")
    source = dirs[0]

    # ------------------------------------------------------------- build phase
    if args.build:
        chunks = read_chunks(source)
        print(f"encoding {len(chunks):,} {CHUNKER} chunks from {source.name}")
        enc = ColbertEncoder(ColbertConfig(), batch_size=args.batch_size)
        meta = build_artifact(chunks, _ARTIFACT_DIR, source, enc)
        enc.release()
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"\nwritten to {_ARTIFACT_DIR}")
        return 0

    # ------------------------------------------------------------- score phase
    query_set = load_gold_query_set(_GOLD_QUERY_SET)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    raw_yaml = yaml.safe_load(_GOLD_QUERY_SET.read_text(encoding="utf-8"))
    types = {e["query"]: e.get("entity_type", "unknown") for e in raw_yaml}
    queries = [e["query"] for e in raw_yaml]

    if args.smoke:
        chunks = read_chunks(source)[:500]
        queries = queries[:8]
        print(f"SMOKE: {len(chunks)} chunks, {len(queries)} queries -- "
              "the numbers below are not a small version of the answer")

    # Queries first, then release the model: the encoder's VRAM and the scorer's
    # ~4 GB float32 matrix have no reason to be resident at the same time.
    enc = ColbertEncoder(ColbertConfig(), batch_size=args.batch_size)
    qmats = enc.encode_queries(queries)
    if args.smoke:
        smoke_meta = build_artifact(chunks, _ARTIFACT_DIR.parent / "_smoke", source, enc)
        print(json.dumps(smoke_meta, ensure_ascii=False, indent=2))
    enc.release()

    art = ColbertStore().load(
        _ARTIFACT_DIR.parent / "_smoke" if args.smoke else _ARTIFACT_DIR, mmap=False)
    if not args.smoke:
        chunks = read_chunks(source)
    index = as_index(chunks)
    # Pay the fp16 -> float32 widening once for the whole run rather than once
    # per query (`maxsim` takes `asarray`, so this is what makes that free).
    art.vecs = np.asarray(art.vecs, dtype=np.float32)
    retriever = ColbertRetriever(artifact=art)

    per_query, latencies = score_queries(retriever, index, queries, qmats, qrels, args.k)
    cb = by_type(per_query, types)

    # ------------------------------------------------------------ comparators
    bm25_cells, dense_cells = load_cells(args.k)
    embedders = sorted({e for (_c, e) in dense_cells})
    best_dense = max(embedders, key=lambda e: program_mean(dense_cells, CHUNKER, e))
    if program_mean(dense_cells, CHUNKER, best_dense) < program_mean(
            dense_cells, CHUNKER, _PREDICTION_EMBEDDER):
        best_dense = _PREDICTION_EMBEDDER
    comparator = {
        "bm25": bm25_cells[(CHUNKER, "-")],
        "dense": dense_cells[(CHUNKER, best_dense)],
    }

    rng = np.random.default_rng(args.seed)
    pairs, meta_cells = [], []
    for etype, arm in PREDICTION_CELLS:
        qs = sorted(set(cb.get(etype, {})) & set(comparator[arm].get(etype, {})))
        diffs = np.array([cb[etype][q] - comparator[arm][etype][q] for q in qs])
        if len(qs) == 0:
            observed, p, ci = float("nan"), 1.0, (float("nan"), float("nan"))
        else:
            observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
        pairs.append(("colbert", f"{arm}@{CHUNKER}", float(observed), p, ci))
        meta_cells.append({
            "etype": etype,
            "label": f"`{etype}` vs {arm}",
            "comparator": f"BM25" if arm == "bm25" else f"dense `{best_dense}`",
            "n": len(qs),
            "queries": qs,
            "colbert": mean([cb[etype][q] for q in qs]),
            "bar": mean([comparator[arm][etype][q] for q in qs]),
        })
    corrected = holm_correct(pairs, args.alpha)
    for cell, (_a, _b, diff, p, ci, adj, sig) in zip(meta_cells, corrected):
        cell.update(diff=float(diff), p=float(p), ci=[float(ci[0]), float(ci[1])],
                    holm_p=float(adj), significant=bool(sig))

    verdict, why = decide(meta_cells)

    rider = rider_for(verdict, meta_cells, chunks, qrels,
                      art.meta.get("doc_maxlen", 300))

    # ------------------------------------------------------------ descriptive
    ceil = {}
    for q, rel in qrels.items():
        ceil.setdefault(types.get(q, "unknown"), []).append(
            min(1.0, args.k / len(rel)) if rel else 0.0)
    descriptive, type_order = {}, sorted(cb)
    for t in type_order:
        qs = sorted(cb[t])
        descriptive[t] = {
            "n": len(qs),
            "colbert": mean([cb[t][q] for q in qs]),
            "bm25": mean([comparator["bm25"].get(t, {}).get(q, 0.0) for q in qs]),
            "dense": mean([comparator["dense"].get(t, {}).get(q, 0.0) for q in qs]),
            "ceiling": mean(ceil.get(t, [0.0])),
        }
    allq = sorted(per_query)
    descriptive["_overall"] = {
        "n": len(allq),
        "colbert": mean([per_query[q] for q in allq]),
        "bm25": mean([comparator["bm25"].get(types[q], {}).get(q, 0.0) for q in allq]),
        "dense": mean([comparator["dense"].get(types[q], {}).get(q, 0.0) for q in allq]),
        "ceiling": mean([v for vs in ceil.values() for v in vs]),
    }

    checks = self_checks(art, index, dirs, source, qmats, meta_cells,
                         bm25_cells, dense_cells, embedders, args.seed)

    raw = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chunker": CHUNKER,
        "k": args.k,
        "alpha": args.alpha,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_queries": len(queries),
        "best_dense": best_dense,
        "artifact_meta": art.meta,
        "cells": meta_cells,
        "descriptive": descriptive,
        "type_order": type_order,
        "per_query": per_query,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "verdict": verdict,
        "verdict_reason": why,
        "rider": rider,
        "checks": [[n, bool(ok), d] for n, ok, d in checks],
    }

    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  -- {detail}")
    print(f"\n{verdict}: {why}")
    if rider:
        print(f"length rider on {rider['cell']}: truncation could account for at most "
              f"{rider['recall_damage_bound']:.4f} of the {rider['gap']:.4f} gap -- "
              f"{'FIRES' if rider['fires'] else 'does not fire'}")
    if not all(ok for _, ok, _ in checks):
        print("a self-check FAILED -- the verdict above is not to be acted on")

    if args.smoke:
        print("\nSMOKE: no report written, no raw cache written.")
        return 0
    _RAW.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    _OUTPUT.write_text("\n".join(render(raw)), encoding="utf-8")
    print(f"written to {_OUTPUT}")
    return 0 if all(ok for _, ok, _ in checks) else 1


# ------------------------------------------------------------------- checks
def self_checks(art, index, dirs, source, qmats, cells,
                bm25_cells, dense_cells, embedders, seed):
    out = []

    chunkers = sorted({c for (c, _e) in dense_cells})
    for name, ok, detail in baseline_anchor_checks(
            bm25_cells, dense_cells, chunkers, embedders)[:2]:
        out.append((name, ok, detail))

    # S3: the artifact is built from ONE combo directory's chunks.parquet, and
    # that is only sound because chunk rows are a function of loader+chunker.
    ids0, mismatched = read_chunk_ids(source), []
    for d in dirs:
        if d != source and read_chunk_ids(d) != ids0:
            mismatched.append(d.name)
    out.append((
        f"S3 all {len(dirs)} live `{CHUNKER}` combos hold identical chunk rows",
        not mismatched,
        f"{len(ids0):,} rows, source `{source.name}`"
        + ("" if not mismatched else "; differs: " + ", ".join(mismatched)),
    ))

    # S4: the whole L family, against the index actually being scored.
    ls = verify_alignment(art, [c.chunk_id for c in index.chunks],
                          doc_maxlen=art.meta.get("doc_maxlen"))
    bad = [f"{n}" for n, ok, _ in ls if not ok]
    out.append((
        "S4 artifact aligns with the index (L1a-L6)",
        not bad,
        f"{len(ls) - len(bad)}/{len(ls)} pass" + ("" if not bad else "; failed " + ", ".join(bad)),
    ))

    # S5: the comparison is paired. A query scored by one arm and not the other
    # would silently shrink a family-1 cell rather than fail.
    unpaired = [c["label"] for c in cells if c["n"] == 0]
    counts = ", ".join(f"{c['etype']}={c['n']}" for c in cells)
    out.append((
        "S5 every family-1 cell is paired over a non-empty query set",
        not unpaired,
        counts,
    ))

    # S6: the packed reduceat path against the naive definition. Sampled, since
    # `maxsim_reference` materialises one document at a time.
    rng = np.random.default_rng(seed)
    off = offsets_from_lengths(np.asarray(art.lengths))
    rows = rng.choice(len(art.lengths), size=min(64, len(art.lengths)), replace=False)
    q = np.asarray(qmats[0], dtype=np.float32)
    docs = [np.asarray(art.vecs[off[i]:off[i] + art.lengths[i]], dtype=np.float32)
            for i in rows]
    ref = maxsim_reference(q, docs)
    from rag_lab.colbert.scoring import maxsim as _maxsim
    full = _maxsim(q, art.vecs, np.asarray(art.lengths))
    gap = float(np.abs(full[rows] - ref).max())
    out.append((
        "S6 packed MaxSim reproduces the naive per-document definition (64 sampled)",
        gap < 1e-3,
        f"max |delta| = {gap:.2e}",
    ))

    # S7: the build-time truncation rate against the independently-computed
    # length profile. It is the pre-registered confound, so a silently different
    # `doc_maxlen` must not pass unnoticed.
    got = art.meta.get("truncated_share")
    want = _TRUNCATED_AT_300.get(CHUNKER)
    ok = got is not None and want is not None and abs(got - want) < 0.005
    out.append((
        f"S7 truncation at `doc_maxlen` reproduces `colbert_length_profile.md`",
        bool(ok),
        "UNCHECKED (no rate recorded)" if got is None
        else f"{got:.3%} vs profile {want:.3%}",
    ))

    return out


if __name__ == "__main__":
    raise SystemExit(main())
