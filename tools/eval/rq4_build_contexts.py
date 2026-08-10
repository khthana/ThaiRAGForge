"""RQ4 step 1: assemble generation contexts from already-persisted retrieval results.

Deterministic and CPU-only -- no index load, no GPU, no model. Persisted
RetrievalResults already carry each hit's `text`, so every arm can be built
from what RQ1-RQ2 already produced; nothing needs re-retrieving.

Arms (docs/rq4-design.md 4c) vary *only* retrieval, holding generator and prompt
fixed, and each was chosen because its retrieval-level difference is already
significance-tested rather than assumed:

    hybrid_qwen3_0.6b_semantic   best-measured configuration
    dense_qwen3_0.6b_semantic    hybrid-vs-dense is RQ2's most robust finding
    bm25_semantic                free baseline; carries `person` outright
    hybrid_m2v_semantic          known RRF failure case
    closed_book                  floor: no context at all

Two entity arms were added 2026-08-10 as the pre-registered *upper bound* on the
relation-graph axis: both read the entity dictionaries the qrels were derived
from, so both are circular by construction and neither is a deployable result.
If a structurally advantaged arm cannot lift citation recall, a richer graph
built on the same dictionaries will not either.

    entity_lookup_semantic       exhaustive + circular, but UNRANKED
    entity_boost_semantic        the same candidate set, ordered by hybrid

**Read `entity_lookup_semantic` with its ordering caveat.** `EntityLookup` scores
every hit 1.0, so `rank` is corpus order, not relevance -- taking the first k is
an arbitrary slice wherever the arm returns more than k documents (49 of 106
queries; median 10, max 768). At k=10 that slice still carries >=1 gold document
for 97 of 106 queries, so the arm is not gutted, but a null from it alone would
be ambiguous between "the dictionary adds nothing" and "the slice threw the
evidence away". `entity_boost_semantic` is the disambiguator: same candidates,
ordered, 104 of 106 and macro recall@10 0.7646 vs 0.6578. Cite the pair.

**Citations are numbered, not spelled out.** The design doc left the citation
format open; numeric labels win on the only criterion that matters here, which is
that an unparseable citation makes 4a unmeasurable. A `resolution_id` in this
corpus is a full Thai document title ("2567/11/เรื่อง ขอความเห็นชอบ..."), and asking
a 4B local model to reproduce one verbatim would measure its copying accuracy
rather than its grounding. Each context block gets `[1]`..`[k]`, the script keeps
the label -> resolution_id map, and scoring maps back. The fabrication mode 4a
cares about survives: a citation to `[11]` when 10 blocks were supplied is
detectable exactly, because the context is known.

**Blocks are documents, not chunks.** Retrieval returns chunks and several can
share a `resolution_id`; relevance in this project is judged at the resolution
level (ADR-0002), so chunks are grouped per document, keeping best rank order.
Otherwise one document occupying 4 of 10 slots would look like 4 citable sources
and the citation-precision denominator would be wrong.

Also records, per (query, arm), whether *any* gold document made the context --
that is the 4b abstention 2x2's row variable, and it comes free here rather than
needing the generator.

Run:
    PYTHONPATH=src python tools/eval/rq4_build_contexts.py
    ... --k 5 --max-chars 1200   # if the generator's context window forces it
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.results import load_retrieval_result  # noqa: E402

_GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUT = REPO / "data" / "rq4" / "contexts"

# arm -> (results dir, combination_id) | None for the closed-book arm
ARMS: dict[str, tuple[str, str] | None] = {
    "hybrid_qwen3_0.6b_semantic": ("gold_hybrid_73det", "plain__semantic__qwen3__06058e0d__hybrid"),
    "dense_qwen3_0.6b_semantic": ("gold_73det_full_embedder_matrix", "plain__semantic__qwen3__06058e0d__dense"),
    # gold_bm25_73det holds two semantic entries (the --embedder-filter "e5" also
    # matched e5_small); BM25 ignores the embedder, so they are the same run and
    # either serves -- pinned explicitly so the arm can't silently change.
    "bm25_semantic": ("gold_bm25_73det", "plain__semantic__e5__35b906c6__bm25"),
    "hybrid_m2v_semantic": ("gold_hybrid_73det", "plain__semantic__local__834c4336__hybrid"),
    # circular by construction -- see the module docstring. Both read
    # `entity_tags_full`, rebuilt 2026-08-05.
    "entity_lookup_semantic": (
        "gold_entity_lookup_73det", "entity_tags__semantic__local__e4fe19d6__entity_lookup"),
    "entity_boost_semantic": (
        "gold_entity_boost_73det", "entity_tags__semantic__local__e4fe19d6__hybrid__entity_boost"),
    "closed_book": None,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10, help="context blocks (documents) per query")
    ap.add_argument("--max-chars", type=int, default=1500, help="cap per document block")
    ap.add_argument("--out", type=str, default=str(_OUT))
    ap.add_argument("--arms", default="", help="comma-separated; default all. Building a "
                    "subset leaves the other arms' context files untouched, which is how "
                    "an arm is added without rewriting the ones 530 answers are keyed to.")
    args = ap.parse_args()

    wanted = [a for a in args.arms.split(",") if a] or list(ARMS)
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; known: {sorted(ARMS)}")

    gold = yaml.safe_load(_GOLD.read_text(encoding="utf-8"))
    qrels = {e["query"]: e["relevant_resolution_ids"] for e in gold}
    entity_type = {e["query"]: e.get("entity_type", "") for e in gold}
    print(f"{len(qrels)} queries from {_GOLD.name}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    stats: dict[str, list[int]] = defaultdict(list)
    gold_present: dict[str, int] = defaultdict(int)

    for arm in wanted:
        source = ARMS[arm]
        arm_dir = out_root / arm
        arm_dir.mkdir(exist_ok=True)
        by_query: dict[str, list] = {}

        if source is not None:
            rdir, combo_id = source
            for path in (REPO / "data" / "results" / rdir).glob("*.json"):
                r = load_retrieval_result(path)
                if r.combination_id != combo_id or r.query not in qrels:
                    continue
                by_query[r.query] = r.results
            if len(by_query) != len(qrels):
                print(f"  !! {arm}: {len(by_query)}/{len(qrels)} queries found for {combo_id}")
                return 1

        for i, (query, relevant) in enumerate(qrels.items()):
            blocks: OrderedDict[str, list[str]] = OrderedDict()
            for hit in sorted(by_query.get(query, []), key=lambda h: h.rank):
                blocks.setdefault(hit.resolution_id, [])
                if len(blocks) > args.k:
                    blocks.popitem()
                    break
                blocks[hit.resolution_id].append(hit.text)

            labelled = []
            for n, (rid, texts) in enumerate(blocks.items(), start=1):
                body = "\n".join(texts)[: args.max_chars]
                labelled.append({"label": n, "resolution_id": rid, "text": body})

            has_gold = any(b["resolution_id"] in relevant for b in labelled)
            gold_present[arm] += int(has_gold)
            stats[arm].append(sum(len(b["text"]) for b in labelled))

            (arm_dir / f"q{i:03d}.json").write_text(json.dumps({
                "query": query,
                "arm": arm,
                "entity_type": entity_type[query],
                "relevant_resolution_ids": relevant,
                "blocks": labelled,
                # 4b's row variable: is the answer even *available* in this context?
                "context_has_gold": has_gold,
            }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\nwritten to {out_root}\n")
    print("| arm | queries | mean ctx chars | max | context has >=1 gold doc |")
    print("|---|---|---|---|---|")
    for arm in wanted:
        s = stats[arm]
        print(f"| {arm} | {len(s)} | {statistics.fmean(s):,.0f} | {max(s):,} | "
              f"{gold_present[arm]}/{len(s)} ({gold_present[arm] / len(s):.0%}) |")
    print("\nThe last column is 4b's ceiling: an arm can only answer where it is "
          "non-zero, and abstention is the *correct* behaviour everywhere else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
