"""Derive which published reports describe the CURRENT indices, and which do not.

CLAUDE.md carried this as a hand-written list for three days and it was wrong in
both directions inside four ("the entire reranker family" over-claimed one
morning and was current by that evening). A to-do list written into living
guidance is a claim that needs re-verifying like any other, and this one is
mechanically derivable: a report is stale when it is older than the newest index
build, because an index rebuild changes what every retrieval report measured
without touching a single generator (which is exactly why `audit_doc_claims.py`'s
D1a cannot see it -- no script moved, the *indices* did).

So the ENUMERATION lives here and the JUDGEMENT stays in CLAUDE.md. Those are
different questions and lumping them together is how a list nobody can ever clear
ends up being a list nobody reads:

  * "stale" is a fact about a timestamp -- this script's job;
  * "worth refreshing" is a decision about whether a verdict could flip or a
    published bound could sharpen -- a human's job.

Two exemption tables encode the judgements that are already made, each with its
reason, and both are REPORTED rather than silently dropped:

  * NOT_WORTH_REFRESHING -- stale and deliberately staying stale (HyDE: margins
    10x-27x larger than anything rebuild #4 moved, and both results are
    directional losses, so there is no bound a refresh could sharpen).
  * CORPUS_INDEPENDENT -- a rebuild cannot stale it at all (model qualification
    gates models on hand-written examples).

Two further classes are excluded before any of that, because a known-retired
artifact keeping a list red is how the list stops being read:

  * anything under a `_`-prefixed directory (`_pre_rebuild4_backup`,
    `_rq4_baseline_2026_08_10`, ...) -- those ARE the pre-refresh snapshots, so
    being old is what they are for;
  * `audit_doc_claims.RETIRED_REPORTS`, IMPORTED rather than re-listed. Two
    copies of a retirement rule diverge, and this project has the scar tissue:
    `RETIRED_RESULT_DIRS`, `RETIRED_RESULTS` and `RETIRED_REPORTS` already state
    the same idea at three layers and each had to be corrected separately.

An entry naming a file that no longer exists is a FAIL, not a silent pass: an
exemption list is the easiest way to make a check vacuous, the same rule D1c
already applies to RETIRED_REPORTS.

Usage:
    python tools/eval/report_currency.py            # write the report
    python tools/eval/report_currency.py --check    # exit 1 if a table is broken
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_doc_claims import RETIRED_REPORTS  # noqa: E402  (the one copy of that rule)

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data" / "results"
INDEX_ROOT = REPO / "data" / "index"
OUT = RESULTS / "report_currency.md"

# Stale on purpose. Key -> the reason a refresh would buy nothing, which is the
# half of the decision no timestamp can supply.
NOT_WORTH_REFRESHING = {
    # No 4-decimal figure appears in any reason below, deliberately. This report
    # lands in `audit_doc_claims.py`'s D2 haystack, so restating a published value
    # here would hand prose quoting it a fresh, current-looking alibi for exactly
    # the number this report is calling stale -- the alibi shape D2 already has.
    "hyde_retrieval_73det.md": "HyDE is CLOSED on a loss an order of magnitude larger "
                               "than anything rebuild #4 moved; a directional loss states "
                               "no bound a refresh could sharpen",
    "hyde_retrieval_thematic.md": "same axis, same direction; P2 refuted, verdict cannot flip",
    "hyde_generation.md": "an input to the two above, not a result of its own",
    "hyde_generation_cost.md": "prices generation, not retrieval; no index is involved",
    # The 6-embedder originals. Their `_9way` successors cover all nine and are
    # current; refreshing these would produce a second, narrower answer to a
    # question already answered, which is what "superseded but kept for
    # reference" means in CLAUDE.md.
    "embedder_significance_test.md": "6-embedder original; superseded by "
                                     "embedder_matrix_9way.py's tables",
    "embedder_significance_test_by_entity_type.md": "6-embedder original; superseded by "
                                                   "the _9way version",
    "bm25_vs_embedder_significance_test.md": "6-embedder original; superseded by the "
                                             "_9way version",
    "hybrid_significance_test.md": "6-embedder original; superseded by the _9way version",
    "gold_embedder_breakdown_73det.md": "6-embedder breakdown; superseded by "
                                        "embedder_significance_test_by_entity_type_9way.md",
    "qdrant_concurrency_smoke.md": "a smoke slice, kept to show the harness ran; a smoke "
                                   "slice is not a small version of the answer, so "
                                   "refreshing it would establish nothing",
    # --- triaged 2026-08-23. Each of these is stale AND staying stale; the two
    # that a re-run could settle cheaply were re-run instead of listed here.
    "bge_qwen_bm25_complementarity.md": "its own header calls its rescue-rate and "
                                        "union-coverage figures an approximation from "
                                        "top-10-only persisted results; "
                                        "bm25_hybrid_entity_type_breakdown.md is current "
                                        "and establishes the same complementarity "
                                        "directly, so a refresh would restate a proxy "
                                        "for a question already answered exactly",
    "residual_relevance.md": "the verdict is incomplete-not-biased, and it rests on three "
                             "Wilson intervals over ~42 judgements per arm that overlap "
                             "heavily; their width is two orders of magnitude larger than "
                             "anything rebuild #4 moved, so no re-sample can separate the "
                             "arms. Refreshing also means re-judging the 126-item sheet, "
                             "which is the dated evidence, not the rendering",
    "qdrant_pilot.md": "its two conclusions are a recommendation (serve dense exact) and a "
                       "structural finding (a beam narrower than the request is malformed). "
                       "The first is re-verified end to end by qdrant_routed_check.md, "
                       "current after the 08-20 re-ingest; the second is a property of HNSW, "
                       "not of this corpus. The ANN-vs-exact margin is an order of magnitude "
                       "wider than a rebuild moves recall",
    "qdrant_concurrency.md": "measures a HAND-ASSEMBLED pipeline whose embedder and Index "
                             "were built once outside the loop -- an idealisation the "
                             "shipped path no longer resembles. serving_concurrency.md "
                             "re-measures the real route_query and REVERSES its headline "
                             "for that path, so refreshing this one would re-measure a "
                             "topology nothing runs",
}

# A rebuild cannot stale these at all.
CORPUS_INDEPENDENT = {
    "reranker_model_qualification.md": "gates models on hand-written examples; no corpus, "
                                       "no index",
    "colbert_model_qualification.md": "same shape: 11 checks x 4 variants over hand-written "
                                      "probes, so no rebuild can move it",
    "colbert_pylate_crosscheck.md": "one fixed query and two fixed documents, re-derived "
                                    "from persisted .npz; no index is read",
    # Added 2026-08-23. Not the same shape as the three above -- it probes a
    # model rather than gating one -- but corpus-independent for the same reason
    # the class exists: what it measures is a fixed historical artifact.
    "rq4_prompt_fit_probes.md": "asks what ollama FED for prompts already on disk. Those "
                                "prompts are frozen artifacts, so no rebuild can change "
                                "their token counts; answers regenerated since carry a "
                                "recorded num_ctx and are covered by G1a instead. It is "
                                "also load-bearing rather than dormant: G1c passes BY "
                                "READING IT, and its universe is imported from the audit, "
                                "so a re-run today would probe the empty set",
}


def builds_by_root() -> dict[str, tuple[dt.datetime, str]]:
    """Newest build time per index root, plus which combo carries it.

    Read from each manifest's own recorded `timestamp`, never from the directory
    mtime: the RQ3 treatment folders still read 2026-08-08 while their contents
    are from 08-17, and taking the folder at face value nearly bought a 2.5-hour
    rebuild that was already done.
    """
    out: dict[str, tuple[dt.datetime, str]] = {}
    for m in INDEX_ROOT.glob("*/*/manifest.json"):
        try:
            when = dt.datetime.fromisoformat(
                json.loads(m.read_text(encoding="utf-8"))["timestamp"])
        except Exception:
            continue
        root = m.parent.parent.name
        if root not in out or when > out[root][0]:
            out[root] = (when, f"{root}/{m.parent.name}")
    if not out:
        raise SystemExit("no index manifest carries a timestamp")
    return out


def cutoff_for(report: Path, builds: dict[str, tuple[dt.datetime, str]]
               ) -> tuple[dt.datetime, str, bool]:
    """The build time a report must beat, and whether it was ATTRIBUTED or screened.

    **The global newest build over-flags, and it did so the day this script
    landed.** `entity_tags_full` was rebuilt 2026-08-12 against a corpus last
    edited 08-09, so it is current -- but rebuild #4 finished 08-17 on a
    *different* root, and comparing against that called both
    `gold_entity_*_73det_report.md` stale when the index they were scored on had
    not moved. An always-red check is one nobody reads.

    So: if a report names an index root in its own text, it is judged against
    THAT root's newest build (the strongest evidence available without the report
    recording its provenance, which is E0's job one layer down). If it names none,
    the global newest is used as a conservative SCREEN and the row says so --
    screened is not the same claim as attributed, and collapsing the two is how
    `undefined` gets reported as `zero`.
    """
    text = report.read_text(encoding="utf-8", errors="ignore")
    named = [r for r in builds if r in text]
    if named:
        when, who = max((builds[r] for r in named), key=lambda t: t[0])
        return when, who, True
    when, who = max(builds.values(), key=lambda t: t[0])
    return when, who, False


def classify(builds) -> dict[str, list[tuple]]:
    """Bucket every report. Rows carry (rel, mtime, cutoff, who, attributed)."""
    buckets: dict[str, list[tuple]] = {
        "current": [], "stale": [], "declined": [], "corpus_independent": [],
        "retired": [],
    }
    for r in sorted(RESULTS.rglob("*.md")):
        if r.name == OUT.name:
            continue
        when = dt.datetime.fromtimestamp(r.stat().st_mtime, tz=dt.timezone.utc)
        rel = r.relative_to(RESULTS).as_posix()
        if any(part.startswith("_") for part in r.relative_to(RESULTS).parts[:-1]) \
                or r.name in RETIRED_REPORTS:
            buckets["retired"].append((rel, when, None, "", False))
            continue
        if r.name in CORPUS_INDEPENDENT:
            buckets["corpus_independent"].append((rel, when, None, "", False))
            continue
        cutoff, who, attributed = cutoff_for(r, builds)
        row = (rel, when, cutoff, who, attributed)
        if when >= cutoff:
            buckets["current"].append(row)
        elif r.name in NOT_WORTH_REFRESHING:
            buckets["declined"].append(row)
        else:
            buckets["stale"].append(row)
    return buckets


def dead_entries() -> list[str]:
    live = {p.name for p in RESULTS.rglob("*.md")}
    return sorted((set(NOT_WORTH_REFRESHING) | set(CORPUS_INDEPENDENT)) - live)


def render(builds, buckets, dead) -> str:
    d = lambda w: w.astimezone().strftime("%Y-%m-%d")
    total = sum(len(v) for v in buckets.values())
    L = ["# Report currency against the newest index build", ""]
    L.append("Generated by `tools/eval/report_currency.py`.")
    L.append("")
    L.append(f"Newest build per index root, over "
             f"{len(list(INDEX_ROOT.glob('*/*/manifest.json')))} index manifests:")
    L.append("")
    L.append("| index root | newest build |")
    L.append("|---|---|")
    for root in sorted(builds, key=lambda r: builds[r][0], reverse=True):
        L.append(f"| `{root}` | {d(builds[root][0])} |")
    L.append("")
    L.append("A report older than the build it is judged against measured indices that "
             "no longer exist. **A report naming an index root in its own text is judged "
             "against THAT root** (*attributed*); one naming none is screened against the "
             "newest build anywhere (*screened*), which is conservative and can over-flag "
             "-- `entity_tags_full` was rebuilt after the corpus last changed, so its "
             "reports are current even though a different root was rebuilt later. This "
             "says nothing about whether refreshing is worth doing: that judgement lives "
             "in `CLAUDE.md` and in the two exemption tables below.")
    L.append("")
    L.append(f"| bucket | count | of |")
    L.append("|---|---:|---:|")
    for k, label in (("current", "current (>= the newest build)"),
                     ("stale", "stale — treat every figure as pre-rebuild"),
                     ("declined", "stale, deliberately not refreshed"),
                     ("corpus_independent", "a rebuild cannot stale it"),
                     ("retired", "superseded snapshot or retired report")):
        L.append(f"| {label} | {len(buckets[k])} | {total} |")
    L.append("")
    if dead:
        L.append(f"**BROKEN: {len(dead)} exemption entr"
                 f"{'y names a' if len(dead) == 1 else 'ies name'} missing file"
                 f"{'' if len(dead) == 1 else 's'}** — "
                 + ", ".join(f"`{n}`" for n in dead)
                 + ". An exemption for a file that does not exist is how a list goes "
                   "vacuous while still looking maintained.")
        L.append("")
    L.append("## Stale, and nothing says otherwise")
    L.append("")
    if buckets["stale"]:
        L.append("| report | last written | judged against | how |")
        L.append("|---|---|---|---|")
        for rel, when, cut, who, att in sorted(buckets["stale"], key=lambda t: t[1]):
            L.append(f"| `{rel}` | {d(when)} | `{who}` {d(cut)} | "
                     f"{'attributed' if att else 'screened'} |")
    else:
        L.append(f"None — all {total} reports are current or exempted.")
    L.append("")
    for key, title in (("declined", "Stale on purpose"),
                       ("corpus_independent", "Not stale-able by a rebuild")):
        L.append(f"## {title}")
        L.append("")
        table = NOT_WORTH_REFRESHING if key == "declined" else CORPUS_INDEPENDENT
        if buckets[key]:
            L.append("| report | last written | why |")
            L.append("|---|---|---|")
            for row in sorted(buckets[key], key=lambda t: t[1]):
                L.append(f"| `{row[0]}` | {d(row[1])} | {table[Path(row[0]).name]} |")
        else:
            L.append("None.")
        L.append("")
    L.append("## Superseded snapshots and retired reports")
    L.append("")
    L.append(f"{len(buckets['retired'])} excluded before the date test: a `_`-prefixed "
             "directory IS a pre-refresh snapshot, and `RETIRED_REPORTS` is imported from "
             "`audit_doc_claims.py` rather than re-listed here. Printed rather than "
             "dropped, so this bucket cannot quietly absorb a live report.")
    L.append("")
    L.append(", ".join(f"`{r[0]}`" for r in sorted(buckets["retired"])) or "None.")
    L.append("")
    L.append("## Current")
    L.append("")
    L.append(", ".join(f"`{r[0]}`" for r in sorted(buckets["current"])) or "None.")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if an exemption entry names a missing file")
    args = ap.parse_args()

    builds = builds_by_root()
    buckets = classify(builds)
    dead = dead_entries()
    OUT.write_text(render(builds, buckets, dead), encoding="utf-8")

    newest = max(builds.values(), key=lambda t: t[0])
    screened = sum(1 for r in buckets["stale"] if not r[4])
    print(f"newest index build {newest[0].astimezone():%Y-%m-%d} ({newest[1]}); "
          f"{len(buckets['stale']) - screened} of {len(buckets['stale'])} stale rows "
          f"attributed to a named root, {screened} screened against the global newest")
    for k in ("current", "stale", "declined", "corpus_independent", "retired"):
        print(f"  {k:20} {len(buckets[k]):3}")
    print(f"wrote {OUT.relative_to(REPO)}")
    if dead:
        print(f"BROKEN: exemption entries naming missing files: {', '.join(dead)}")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
