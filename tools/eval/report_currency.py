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
}

# A rebuild cannot stale these at all.
CORPUS_INDEPENDENT = {
    "reranker_model_qualification.md": "gates models on hand-written examples; no corpus, "
                                       "no index",
    "colbert_model_qualification.md": "same shape: 11 checks x 4 variants over hand-written "
                                      "probes, so no rebuild can move it",
    "colbert_pylate_crosscheck.md": "one fixed query and two fixed documents, re-derived "
                                    "from persisted .npz; no index is read",
}


def newest_index_build() -> tuple[dt.datetime, str]:
    """The newest `timestamp` across every index manifest, and which combo it is.

    Read from the manifest's own recorded field, never from the directory mtime:
    the RQ3 treatment folders still read 2026-08-08 while their contents are from
    08-17, and taking the folder at face value nearly bought a 2.5-hour rebuild
    that was already done.
    """
    best, who = None, ""
    for m in INDEX_ROOT.glob("*/*/manifest.json"):
        try:
            ts = json.loads(m.read_text(encoding="utf-8")).get("timestamp")
            when = dt.datetime.fromisoformat(ts)
        except Exception:
            continue
        if best is None or when > best:
            best, who = when, f"{m.parent.parent.name}/{m.parent.name}"
    if best is None:
        raise SystemExit("no index manifest carries a timestamp")
    return best, who


def classify(cutoff: dt.datetime) -> dict[str, list[tuple[str, dt.datetime]]]:
    buckets: dict[str, list[tuple[str, dt.datetime]]] = {
        "current": [], "stale": [], "declined": [], "corpus_independent": [],
        "retired": [],
    }
    for r in sorted(RESULTS.rglob("*.md")):
        if r.name == OUT.name:
            continue
        when = dt.datetime.fromtimestamp(r.stat().st_mtime, tz=dt.timezone.utc)
        rel = str(r.relative_to(RESULTS)).replace("\\", "/")
        if any(part.startswith("_") for part in r.relative_to(RESULTS).parts[:-1]) \
                or r.name in RETIRED_REPORTS:
            buckets["retired"].append((rel, when))
        elif r.name in CORPUS_INDEPENDENT:
            buckets["corpus_independent"].append((rel, when))
        elif when >= cutoff:
            buckets["current"].append((rel, when))
        elif r.name in NOT_WORTH_REFRESHING:
            buckets["declined"].append((rel, when))
        else:
            buckets["stale"].append((rel, when))
    return buckets


def dead_entries() -> list[str]:
    live = {p.name for p in RESULTS.rglob("*.md")}
    return sorted((set(NOT_WORTH_REFRESHING) | set(CORPUS_INDEPENDENT)) - live)


def render(cutoff: dt.datetime, who: str, buckets, dead) -> str:
    d = lambda w: w.astimezone().strftime("%Y-%m-%d")
    total = sum(len(v) for v in buckets.values())
    L = ["# Report currency against the newest index build", ""]
    L.append("Generated by `tools/eval/report_currency.py`.")
    L.append("")
    L.append(f"Newest index build: **{d(cutoff)}** (`{who}`), over "
             f"{len(list(INDEX_ROOT.glob('*/*/manifest.json')))} index manifests.")
    L.append("")
    L.append("A report older than that build measured indices that no longer exist. "
             "This says nothing about whether refreshing it is worth doing — that "
             "judgement lives in `CLAUDE.md`, and the two exemption tables below are "
             "where a judgement already made is recorded.")
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
        L.append("| report | last written |")
        L.append("|---|---|")
        for rel, when in sorted(buckets["stale"], key=lambda t: t[1]):
            L.append(f"| `{rel}` | {d(when)} |")
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
            for rel, when in sorted(buckets[key], key=lambda t: t[1]):
                L.append(f"| `{rel}` | {d(when)} | {table[Path(rel).name]} |")
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
    L.append(", ".join(f"`{rel}`" for rel, _ in sorted(buckets["retired"])) or "None.")
    L.append("")
    L.append("## Current")
    L.append("")
    L.append(", ".join(f"`{rel}`" for rel, _ in sorted(buckets["current"])) or "None.")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if an exemption entry names a missing file")
    args = ap.parse_args()

    cutoff, who = newest_index_build()
    buckets = classify(cutoff)
    dead = dead_entries()
    OUT.write_text(render(cutoff, who, buckets, dead), encoding="utf-8")

    print(f"newest index build {cutoff.astimezone():%Y-%m-%d} ({who})")
    for k in ("current", "stale", "declined", "corpus_independent", "retired"):
        print(f"  {k:20} {len(buckets[k]):3}")
    print(f"wrote {OUT.relative_to(REPO)}")
    if dead:
        print(f"BROKEN: exemption entries naming missing files: {', '.join(dead)}")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
