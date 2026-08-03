"""Residual-relevance study: does the Gold qrels systematically under-credit dense retrieval?

**The validity threat this measures.** The Gold set's `relevant_resolution_ids`
were derived by *string containment* -- a resolution counts as relevant to a
program query when the canonical program string appears in its title, and to a
person query when the person is named in its body
(`tools/corpus_prep/build_gold_candidates.py`). That construction is exactly
what BM25 does at query time. So any document that is genuinely relevant but
phrases the entity differently is a **false negative in the qrels**, and a
retriever that finds it semantically is *penalised for being right*, while the
lexical retriever is graded against a key built the way it works.

This is textbook pooling bias, and it points straight at two of the study's
most-quoted findings: "BM25 significantly beats bge_m3" and "BM25 carries
`person` outright (0.8147)". If dense retrieval's top-10 contains relevant
documents the qrels never judged, at a higher rate than BM25's does, both
claims are inflated and must be restated.

**Design.** For a stratified sample of queries, collect every top-10 hit that
the qrels do not judge at all, sample a few per arm, and have a human judge
them. The review sheet is **blinded**: the arm that retrieved each candidate is
written to a separate key file, so a judgement cannot be biased toward the
hypothesis. Judging is per `resolution_id`, matching how relevance is defined
throughout the project (ADR-0002).

The outcome is a per-arm rate of relevant-but-unjudged hits. Symmetric rates
mean the qrels are merely incomplete (which harms all arms equally and is
normal in IR); a higher rate for dense means the qrels are *biased*, and the
BM25-vs-dense comparisons need a caveat or a correction.

**This script only builds the sheet and scores it back.** It cannot judge, and
deliberately makes no attempt to -- an LLM judgement here would reintroduce
exactly the automated-relevance risk the Gold set was designed to avoid
(docs/entity-extraction-and-gold-eval-log.md).

Build the sheet:
    .venv/Scripts/python.exe tools/eval/residual_relevance_sample.py

Then fill in `verdict:` for each item (`y` relevant / `n` not / `?` unsure).
**Recommended**: `residual_relevance_review_app.py`, a small Streamlit
reviewer (`.venv/Scripts/streamlit.exe run
tools/eval/residual_relevance_review_app.py`) that shows each candidate's
full, untruncated chunk text and writes verdicts back into
data/results/residual_relevance/review_sheet.yaml with one click -- editing
the sheet's `snippet:` field by hand caps it at `--snippet` chars (600 by
default), which can cut a chunk before the sentence that would establish
relevance, making some items unjudgeable from the raw YAML alone. Then score:
    .venv/Scripts/python.exe tools/eval/residual_relevance_sample.py --score
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402

_GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_OUT_DIR = REPO / "data" / "results" / "residual_relevance"
_SHEET = _OUT_DIR / "review_sheet.yaml"
_KEY = _OUT_DIR / "sample_key.yaml"
_REPORT = REPO / "data" / "results" / "residual_relevance.md"

# One representative combo per retrieval paradigm, all on `semantic` chunking so
# the chunker axis is held fixed. dense-vs-bm25 is the comparison under test;
# hybrid is included because it is the configuration the project recommends.
ARMS = {
    "dense": ("gold_73det_full_embedder_matrix", "plain__semantic__qwen3__06058e0d__dense"),
    "bm25": ("gold_bm25_73det", "plain__semantic__e5__35b906c6__bm25"),
    "hybrid": ("gold_hybrid_73det", "plain__semantic__qwen3__06058e0d__hybrid"),
}


def load_arms(qrels: dict[str, list[str]], k: int) -> dict[str, dict[str, list[str]]]:
    """arm -> query -> top-k resolution_ids, in rank order, deduped per document."""
    out: dict[str, dict[str, list[str]]] = {}
    for arm, (rdir, combo) in ARMS.items():
        by_query: dict[str, list[str]] = {}
        for path in (REPO / "data" / "results" / rdir).glob("*.json"):
            r = load_retrieval_result(path)
            if r.combination_id != combo or r.query not in qrels:
                continue
            seen: list[str] = []
            for hit in sorted(r.results, key=lambda h: h.rank):
                if hit.resolution_id not in seen:
                    seen.append(hit.resolution_id)
                if len(seen) >= k:
                    break
            by_query[r.query] = seen
        if len(by_query) != len(qrels):
            raise SystemExit(f"{arm}: {len(by_query)}/{len(qrels)} queries found for {combo}")
        out[arm] = by_query
    return out


def snippet_for(query: str, rid: str, k: int) -> str:
    """First chunk text for a document, from whichever arm retrieved it."""
    for rdir, combo in ARMS.values():
        for path in (REPO / "data" / "results" / rdir).glob("*.json"):
            r = load_retrieval_result(path)
            if r.combination_id != combo or r.query != query:
                continue
            for hit in sorted(r.results, key=lambda h: h.rank):
                if hit.resolution_id == rid:
                    return hit.text
    return ""


def build_text_index(qrels: dict[str, set[str]]) -> dict[tuple[str, str], str]:
    """(query, resolution_id) -> full, untruncated chunk text, for every hit
    across all three arms. Shared by `build()` (which then truncates to
    `--snippet` chars for the on-disk sheet) and
    `residual_relevance_review_app.py` (which doesn't truncate at all --
    the reviewer needs the whole chunk to judge relevance, and a fixed
    character cut sometimes lands before the sentence that would establish
    it)."""
    text_of: dict[tuple[str, str], str] = {}
    for rdir, combo in ARMS.values():
        for path in (REPO / "data" / "results" / rdir).glob("*.json"):
            r = load_retrieval_result(path)
            if r.combination_id != combo or r.query not in qrels:
                continue
            for hit in sorted(r.results, key=lambda h: h.rank):
                text_of.setdefault((r.query, hit.resolution_id), hit.text)
    return text_of


def load_qrels() -> dict[str, set[str]]:
    """query -> gold relevant_resolution_ids, from the 73det Gold set."""
    return {e.query: set(e.relevant_resolution_ids) for e in load_gold_query_set(_GOLD)}


def build_full_document_index() -> dict[str, str]:
    """resolution_id -> full document text, sourced directly from the corpus
    via `PlainLoader` -- NOT from persisted retrieval hits like
    `build_text_index`. Guarantees coverage of every resolution_id, including
    an `already_judged_relevant` calibration document that no arm's top-k
    happened to retrieve for that query (a real possibility -- that's the
    whole reason this study exists). A title alone can be useless for
    calibration: person qrels come from body mentions, not the title, so
    `residual_relevance_review_app.py` needs the actual text to show why a
    reference document counts as relevant. Cheap (~1.6s for the full
    ~2,850-file corpus, text-only, no chunking/embedding)."""
    from rag_lab.loaders.common import is_real_resolution_path
    from rag_lab.loaders.plain_loader import PlainLoader

    loader = PlainLoader()
    paths = [
        p for p in sorted((REPO / "academic_resolutions").rglob("*.md"))
        if is_real_resolution_path(p)
    ]
    index: dict[str, str] = {}
    for p in paths:
        r = loader.load(str(p))
        index[r.resolution_id] = r.raw_text
    return index


def build(args) -> None:
    qrels = load_qrels()
    # QuerySetEntry carries only query + qrels; entity_type lives in the yaml
    etype = {e["query"]: e.get("entity_type", "")
             for e in yaml.safe_load(_GOLD.read_text(encoding="utf-8"))}
    arms = load_arms(qrels, args.k)

    # Stratify by entity_type so a category cannot dominate the estimate; the
    # threat is not uniform across them (person qrels come from body mentions,
    # program qrels from title strings).
    by_type: dict[str, list[str]] = defaultdict(list)
    for q in qrels:
        by_type[etype[q]].append(q)
    rng = np.random.default_rng(args.seed)
    chosen: list[str] = []
    for t, qs in sorted(by_type.items()):
        share = max(1, round(args.queries * len(qs) / len(qrels)))
        pick = rng.choice(sorted(qs), size=min(share, len(qs)), replace=False)
        chosen.extend(pick.tolist())
    print(f"sampled {len(chosen)} queries across {len(by_type)} entity types")

    # cache one snippet per (query, rid) -- reading every result file per lookup
    # would be O(queries * files); build the map in one pass instead
    text_of = build_text_index(qrels)

    items: dict[tuple[str, str], dict] = {}
    unjudged_totals: dict[str, dict[str, int]] = {a: {} for a in ARMS}

    for q in chosen:
        for arm in ARMS:
            unjudged = [r for r in arms[arm][q] if r not in qrels[q]]
            unjudged_totals[arm][q] = len(unjudged)
            for rid in rng.permutation(len(unjudged))[: args.per_arm]:
                rid = unjudged[int(rid)]
                item = items.setdefault((q, rid), {"arms": [], "ranks": {}})
                if arm not in item["arms"]:
                    item["arms"].append(arm)
                item["ranks"][arm] = arms[arm][q].index(rid) + 1

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet, key = [], []
    order = rng.permutation(len(items))  # shuffled so arms don't cluster visually
    keyed = sorted(items.items())
    for n, i in enumerate(order, start=1):
        (q, rid), meta = keyed[int(i)]
        sheet.append({
            "id": n,
            "query": q,
            "entity_type": etype[q],
            "candidate": rid,
            "snippet": text_of.get((q, rid), "")[: args.snippet],
            "already_judged_relevant": sorted(qrels[q])[:3],
            "verdict": "",  # y = relevant / n = not relevant / ? = unsure
        })
        key.append({"id": n, "query": q, "candidate": rid,
                    "arms": meta["arms"], "ranks": meta["ranks"]})

    _SHEET.write_text(
        "# Residual-relevance review sheet. Fill `verdict` for every item:\n"
        "#   y = this document IS relevant to the query (the qrels missed it)\n"
        "#   n = not relevant\n"
        "#   ? = cannot tell from the snippet\n"
        "# `already_judged_relevant` shows up to 3 documents the qrels DO count,\n"
        "# as a calibration reference for what 'relevant' means for this query.\n"
        "# The retrieving arm is deliberately not shown -- it is in sample_key.yaml.\n\n"
        + yaml.safe_dump(sheet, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8")
    _KEY.write_text(yaml.safe_dump(
        {"items": key, "unjudged_totals": unjudged_totals, "queries": sorted(chosen),
         "k": args.k, "per_arm": args.per_arm, "seed": args.seed},
        allow_unicode=True, sort_keys=False), encoding="utf-8")

    per_arm_mean = {a: np.mean(list(v.values())) for a, v in unjudged_totals.items()}
    print(f"\n{len(items)} items to judge -> {_SHEET}")
    print(f"key (arm assignment, hidden from the sheet) -> {_KEY}")
    print("\nunjudged documents per query in top-10, by arm (the pool being sampled):")
    for a, m in per_arm_mean.items():
        print(f"  {a:7s} {m:.2f} of {args.k}")
    print("\nThose means are already informative: a much larger unjudged pool for one "
          "arm\nis where a bias would show up, though only the verdicts can say whether "
          "the\nunjudged documents are actually relevant.")


def score(args) -> None:
    sheet = yaml.safe_load(_SHEET.read_text(encoding="utf-8"))
    key = yaml.safe_load(_KEY.read_text(encoding="utf-8"))
    verdicts = {it["id"]: str(it.get("verdict", "")).strip().lower() for it in sheet}
    blank = [i for i, v in verdicts.items() if v not in {"y", "n", "?"}]
    if blank:
        raise SystemExit(f"{len(blank)} of {len(verdicts)} items unjudged "
                         f"(first: id {blank[0]}). Fill every `verdict` before scoring.")

    arms_of = {it["id"]: it["arms"] for it in key["items"]}
    totals = key["unjudged_totals"]
    n_q = len(key["queries"])

    rows = []
    for arm in ARMS:
        ids = [i for i, a in arms_of.items() if arm in a]
        y = sum(verdicts[i] == "y" for i in ids)
        n = sum(verdicts[i] == "n" for i in ids)
        unsure = sum(verdicts[i] == "?" for i in ids)
        judged = y + n
        rate = y / judged if judged else 0.0
        pool = float(np.mean([totals[arm][q] for q in key["queries"]]))
        # Wilson interval: the counts here are small, and a normal interval on a
        # proportion this size can run below 0 or above 1.
        if judged:
            z, p = 1.96, rate
            denom = 1 + z**2 / judged
            centre = (p + z**2 / (2 * judged)) / denom
            half = z * np.sqrt(p * (1 - p) / judged + z**2 / (4 * judged**2)) / denom
            ci = (max(0.0, centre - half), min(1.0, centre + half))
        else:
            ci = (0.0, 1.0)
        rows.append((arm, len(ids), y, n, unsure, rate, ci, pool, rate * pool))

    lines = [
        "# Residual relevance: relevant-but-unjudged documents in top-10",
        "",
        f"{len(verdicts)} candidate documents judged by hand across {n_q} sampled "
        f"queries, blinded to the retrieving arm. A candidate is a top-{key['k']} hit "
        "that the Gold qrels do not judge at all.",
        "",
        "**Why this matters**: the Gold qrels were built by string containment, which "
        "is how BM25 matches. If dense retrieval surfaces relevant documents the qrels "
        "never judged, at a higher rate than BM25 does, then every BM25-vs-dense "
        "comparison in this project is biased toward BM25 and the margin is overstated.",
        "",
        "| arm | judged | relevant | not | unsure | residual rate | 95% CI (Wilson) | "
        "unjudged/query | est. missed relevant/query |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, n_items, y, n, unsure, rate, ci, pool, est in rows:
        lines.append(
            f"| {arm} | {n_items} | {y} | {n} | {unsure} | {rate:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | {pool:.2f} | {est:.2f} |"
        )
    lines.append("")

    dense = next(r for r in rows if r[0] == "dense")
    bm25 = next(r for r in rows if r[0] == "bm25")
    gap = dense[8] - bm25[8]
    overlap = not (dense[6][1] < bm25[6][0] or bm25[6][1] < dense[6][0])
    lines += [
        "## Verdict on the pooling-bias threat",
        "",
        f"dense residual rate {dense[5]:.3f} vs BM25 {bm25[5]:.3f}; estimated relevant "
        f"documents missed per query {dense[8]:.2f} vs {bm25[8]:.2f} "
        f"(difference {gap:+.2f}).",
        "",
        ("The two Wilson intervals **overlap**, so this sample does not establish a "
         "difference in residual rate between the arms. Incompleteness that affects "
         "both arms alike depresses all absolute scores but leaves the *comparison* "
         "intact -- the BM25-vs-dense findings stand, with incompleteness reported as "
         "a limitation on absolute values."
         if overlap else
         "The two Wilson intervals **do not overlap**: the arms are missing credit at "
         "different rates, which is bias rather than mere incompleteness. Every "
         "BM25-vs-dense claim needs restating with this correction, and the per-query "
         "estimate above gives its rough size."),
        "",
        f"Scale for reading the estimate: mean relevant documents per query in the "
        f"qrels is 9.87, so {dense[8]:.2f} missed per query is "
        f"~{dense[8] / 9.87:.1%} of the judged pool.",
        "",
        "**Limits.** One sampled document per arm per query is a rate estimate, not a "
        "census; the interval above is the honest precision. Judgements are by a single "
        "annotator (the corpus owner), so this inherits the same single-annotator "
        "limitation as the Gold set itself.",
        "",
    ]
    _REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"written to {_REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score", action="store_true", help="score a filled-in review sheet")
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--per-arm", type=int, default=2, help="candidates sampled per arm per query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--snippet", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    (score if args.score else build)(args)


if __name__ == "__main__":
    main()
