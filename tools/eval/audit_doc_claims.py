"""The docs layer of the invariant sweep: check what the prose *says* against
what the artifacts on disk actually hold.

`audit_pipeline_invariants.py` gates three layers (C = corpus, I = index,
E = eval) and `diff_significance_reports.py` gates report-vs-report. Nothing
gated report-vs-**prose**, and that is where this project's avoidable mistakes
have actually happened: a number written by hand into `CLAUDE.md` or
`docs/paper-results-summary.md`, correct on the day it was typed, that no
later refresh touches because a refresh re-runs scripts and diffs reports --
it never reads the prose.

The failure is invisible to every existing guard by construction. The case that
motivated this script: the per-chunker BM25-vs-embedder table in
`paper-results-summary.md` drifted from its report in the 2026-08-06 refresh
(`e5` reads +0.0675/+0.1188/+0.1017/+0.1223, the report says
+0.0674/+0.1161/+0.1036/+0.1233). Every *verdict* cell was identical, so
`diff_significance_reports.py` correctly reported 0 flips and nobody re-copied
the numbers.

Checks (D = docs):

    D1  a report older than the script that generates it (prose quoting it is
        quoting a stale run), plus reports whose generator can't be identified
    D2  a number in the prose that appears in no report -- the main check
    D3  a significant p-value (<0.05) quoted inside a sentence that calls the
        result not significant, or vice versa
    D4  an eval *input* (a constant a script reads) changed after the report
        that depends on it was generated -- the "editing ROUTE_COMBO silently
        re-scores soft_vs_hard_routing.md" failure

Read-only. Exits 1 if any check FAILs.

Run:
    python tools/eval/audit_doc_claims.py
    python tools/eval/audit_doc_claims.py --list        # every D2 hit, with context
    python tools/eval/audit_doc_claims.py --report docs/doc-claims-audit.md
"""
from __future__ import annotations

import argparse
import re

import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

if hasattr(sys.stdout, "reconfigure"):  # Thai + U+2212 vs the Windows console codepage
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data/results"
SCRIPTS = REPO / "tools/eval"
ALLOWLIST = REPO / "tools/eval/doc_claims_allowlist.yaml"

# The prose under audit. Only files that make *claims about measurements* belong
# here -- narrative logs (docs/*-log.md) are deliberately append-only records of
# what was believed at a point in time, so a stale number in one is the point.
DOCS = [
    Path("CLAUDE.md"),
    Path("docs/paper-results-summary.md"),
]

# Script outputs a number may legitimately come from. Deliberately **.md only**:
# including the per-query result JSON under data/results/ makes D2 vacuous rather
# than thorough -- 225 MB of per-query scores contains almost every 4-decimal
# value by coincidence, which took the untraceable count from 122 down to 27
# without a single one of those 95 being genuinely traceable to a reported
# figure. Same lesson as C4 in the pipeline audit: a check whose haystack is too
# big stops being a check.
ARTIFACT_GLOBS = ["data/results/**/*.md"]
# Reports that live under docs/ because they are read as documents, but are
# script output all the same.
ARTIFACT_FILES = [
    Path("docs/pipeline-invariant-audit.md"),
    Path("docs/title-body-agreement.md"),
    Path("docs/relation-graph.md"),
    Path("docs/program-matcher-absorption.md"),
    Path("docs/rq4-prompt-truncation.md"),
]

# A 4-decimal figure is this project's universal format for a metric, an effect
# size or a p-value. Deliberately not 2- or 3-decimal: those collide with dates,
# version numbers and ordinary prose, and every script here reports at 4.
NUM = re.compile(r"(?<![\d.])(\d\.\d{4})(?![\d])")

# D2 exemption 1: the prose is explicitly citing a number as *no longer current*.
# `paper-results-summary.md` keeps its own supersession history on purpose (see
# [[feedback_external_analysis_reads_a_stale_slice]]) -- a retired figure quoted
# as retired is the doc doing its job, not drift.
SUPERSEDED = re.compile(
    r"superseded|stale|retired|withdrawn|do not cite|pre-rebuild|predates|"
    r"no longer|dropped (?:from|to)|rose|was \*?\*?0\.|read \*?\*?0\.[\d]+ at that|"
    r"up from|originally|at that point|reversing|retract|the old\b|→|->",
    re.I,
)
# D2 exemption 2: the block is a dated snapshot -- a table explicitly labelled
# with the date it was measured, kept for the record.
DATED = re.compile(
    r"(?:refreshed|re-run|rerun|recorded|measured|as of|updated|added|caveat)\s*\**\s*20\d\d-\d\d-\d\d",
    re.I,
)

# D3 vocabulary.
P_VALUE = re.compile(r"(?:p\s*=\s*|Holm(?:-adj)?\.?\s*(?:p\s*=\s*)?)(\d\.\d+)", re.I)
NOT_SIG = re.compile(
    r"\bns\b|not significant|non-significant|ไม่มีนัยสำคัญ|"
    r"no (?:\w+ ){0,2}significant|no longer significant|flat tie",
    re.I,
)
IS_SIG = re.compile(r"\bsignificant(?:ly)?\b", re.I)

# D4: constants and config a script *reads*, so editing one silently re-scores
# every report below it without touching the script or the index. Curated
# rather than derived: an import graph would flag every unrelated core edit and
# the check would be ignored within a week.
EVAL_INPUTS: dict[str, list[str]] = {
    "src/rag_lab/router.py": ["routing_eval.md", "soft_vs_hard_routing.md"],
    "src/rag_lab/retrievers/hybrid.py": [
        "hybrid_alpha_sweep.md", "soft_vs_hard_routing.md", "hybrid_significance_test_9way.md",
    ],
    "config/eval/gold_query_set_73det.yaml": [
        "routing_eval.md", "soft_vs_hard_routing.md", "hybrid_alpha_sweep.md",
        "power_analysis.md", "map_precision_significance_test.md", "rq4_score.md",
    ],
    "tools/eval/rq4_generate.py": ["rq4_score.md", "rq4_score_guarded.md"],
}

findings: list[tuple[str, str, str]] = []


def record(check: str, ok: bool, detail: str, warn: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    findings.append((check, status, detail))
    print(f"[{status}] {check}: {detail}")


def _mtime(p: Path) -> datetime:
    """Filesystem mtime, same convention as `audit_pipeline_invariants.py`'s I6/E4.

    Deliberately *not* the git commit date, which is wrong here in both
    directions: `data/results/` is gitignored so a report has no commit date at
    all, and a script's commit lands *after* the run that produced the report
    (generate, read, then commit both) -- so taking the later of the two flagged
    all 10 report/script pairs on this script's first run, every one a false
    positive. The cost is that a fresh clone rewrites every mtime to checkout
    time and D1a/D4 go quiet; that is already true of the checks this mirrors.
    """
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)


def _blocks(lines: list[str]) -> list[list[tuple[int, str]]]:
    """Split into contiguous non-blank runs (a paragraph or a markdown table)."""
    out: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        if ln.strip():
            cur.append((i + 1, ln))
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _allowlist(section: str) -> list[dict]:
    """Triaged exemptions, each carrying the reason it was cleared.

    A reason field rather than a bare list on purpose: an exemption whose
    justification is not written down is indistinguishable from a check that was
    switched off, and the next person cannot re-audit it.
    """
    if not ALLOWLIST.exists():
        return []
    raw = yaml.safe_load(ALLOWLIST.read_text(encoding="utf-8")) or {}
    return raw.get(section) or []


# ------------------------------------------------------------------- D1
# Snapshots from a superseded pipeline. Their generator is still a live script
# that has moved on since, so "report older than its generator" is true of every
# one of them and says nothing: nothing re-runs these, and nothing reads them for
# a current number. Failing them would leave D1a permanently red, which is the
# mistake `audit_pipeline_invariants.py`'s RETIRED_RESULT_DIRS and
# `tools/archive_unused.py`'s RETIRED_RESULTS already avoid one layer down --
# a FAIL has to mean a *live* report has drifted. Classified, not skipped: the
# count is printed beside D1a's denominator, and each entry carries its reason
# the way `doc_claims_allowlist.yaml` does, so it can be re-audited rather than
# taken on trust. Each report also says so in its own first lines.
RETIRED_REPORTS = {
    "gold_73det_full_embedder_matrix_report.md":
        "6-embedder rollup; superseded by embedder_significance_test_9way.md",
    "gold_full_embedder_matrix_report.md":
        "retired 252-query set; superseded by the 9-way matrix on 73det",
    "gold_chunker_compare_report.md":
        "retired 252-query set; superseded by hybrid_chunker_significance_test.md",
    "gold_chunker_compare_73det_report.md":
        "superseded by hybrid_chunker_significance_test.md",
    "gold_embedder_compare_report.md":
        "retired 252-query set; superseded by the 9-way matrix",
    "silver_chunker_compare_report.md":
        "Silver query set, replaced by the Gold sets",
    "congen_sct_truncation_fix_report.md":
        "one-off before/after for the max_seq_length fix; both embedders are in the 9-way matrix since",
    "pipeline_invariant_audit.md":
        "2026-07-30 snapshot; the current report is docs/pipeline-invariant-audit.md",
    # The one entry here that must NOT be cleared by re-running its generator.
    # It is the record of which 81 cells were truncated at num_ctx=8192; those
    # cells were regenerated 2026-08-10, so a re-run would legitimately report
    # 0 and erase the only list of what was damaged. Its generator moved after
    # it was written (the unsound chars/token screen was replaced with a
    # provable UTF-8-byte bound), which is exactly the D1a shape -- but the
    # right response is retirement, not regeneration.
    "rq4_truncated_cells.md":
        "pre-repair snapshot of the 81 truncated cells; regenerating them 2026-08-10 "
        "is what makes it historical, and re-running the generator would erase it",
    # Not an eval report at all, and the only file here that declares no
    # generator: a one-off corpus diff from commit e1523b3 whose throwaway
    # script was not retained. It cannot be made to name a generator honestly,
    # so it is classified rather than left as a permanent D1b warn.
    "person_cross_cell_fix_review.md":
        "one-off person_loader cross-cell diff (e1523b3); generator not retained",
}


def _generator_for(report: Path) -> Path | None:
    """The script that writes this report: its own declaration, else by name."""
    head = report.read_text(encoding="utf-8", errors="ignore")[:2000]
    m = re.search(r"tools/eval/([a-z0-9_]+\.py)", head)
    if m and (SCRIPTS / m.group(1)).exists():
        return SCRIPTS / m.group(1)
    stem = report.stem
    for cand in (stem, stem.removesuffix("_report"), f"run_{stem}", f"run_{stem.removesuffix('_report')}"):
        if (SCRIPTS / f"{cand}.py").exists():
            return SCRIPTS / f"{cand}.py"
    return None


def audit_reports() -> None:
    reports = sorted(RESULTS.glob("*.md"))
    stale, orphan, retired = [], [], []
    for r in reports:
        if r.name in RETIRED_REPORTS:
            retired.append(r.name)
            continue
        gen = _generator_for(r)
        if gen is None:
            orphan.append(r.name)
            continue
        if _mtime(gen) > _mtime(r):
            stale.append(f"{r.name} (generator {gen.name} is newer)")
    live = len(reports) - len(retired)
    record(
        "D1a report older than its generator", not stale,
        f"{len(stale)} of {live - len(orphan)} live reports with an identified generator "
        f"({len(retired)} classified retired)",
    )
    for s in stale[:15]:
        print(f"        {s}")
    # Not a FAIL: many reports predate the naming convention. It is a WARN so the
    # denominator above is honest about how much D1a actually examined -- a check
    # that silently skips 40 of 45 files is the vacuous-PASS trap again.
    record(
        "D1b report declares its generator", not orphan,
        f"{len(orphan)} of {live} live reports have no identifiable generator "
        "(add a 'Generated by `tools/eval/x.py`' line)",
        warn=True,
    )
    # The exemption list itself can go stale: a retired report that is deleted
    # leaves an entry that quietly shrinks nothing, and one that is revived would
    # be exempt forever. Same shape as the cleanup-breaks-an-audit lesson.
    missing = sorted(set(RETIRED_REPORTS) - {r.name for r in reports})
    record(
        "D1c every classified-retired report still exists", not missing,
        f"{len(missing)} of {len(RETIRED_REPORTS)} RETIRED_REPORTS entries name a missing file",
        warn=True,
    )


# ------------------------------------------------------------------- D2
def audit_numbers(show_all: bool) -> None:
    artifacts: list[Path] = []
    for g in ARTIFACT_GLOBS:
        artifacts += sorted(REPO.glob(g))
    artifacts += [REPO / f for f in ARTIFACT_FILES if (REPO / f).exists()]

    known: dict[str, tuple[str, int]] = {}
    per_report: dict[str, list[tuple[float, str, int]]] = {}
    for a in artifacts:
        rel = str(a.relative_to(REPO)).replace("\\", "/")
        for i, ln in enumerate(a.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for n in NUM.findall(ln):
                known.setdefault(n, (a.name, i))
                per_report.setdefault(rel, []).append((float(n), n, i))

    CITED = re.compile(r"data/results/[\w/]+\.md")

    def nearest(n: str, context: str) -> str:
        """Distinguish 'drifted' from 'absent' -- the triage step that decides
        whether a hit is a stale copy of a live table or an unsourced figure.

        Searched only in the report the surrounding prose actually cites. A
        nearest-value scan over every report finds a match within 0.0001 for
        almost any figure by coincidence (0.1223 'matched' a per-entity-type
        table it has nothing to do with), which reads as provenance and isn't.
        """
        cands = [c for c in CITED.findall(context) if c in per_report]
        if not cands:
            return "  (no report cited nearby)"
        x = float(n)
        best = min(
            ((abs(v - x), s, i, c) for c in cands for v, s, i in per_report[c]),
            default=None,
        )
        if best is None or best[0] > 0.005:
            return f"  (not in cited {cands[0].rsplit('/', 1)[-1]})"
        return f"  ~ {best[1]} in {best[3].rsplit('/', 1)[-1]}:{best[2]}"

    allow = {(e["doc"], str(e["number"])) for e in _allowlist("numbers")}
    total = untraceable = exempt_sup = exempt_dated = exempt_allow = 0
    residue: list[tuple[str, int, list[str], str]] = []

    for doc in DOCS:
        p = REPO / doc
        lines = p.read_text(encoding="utf-8").splitlines()
        blocks = _blocks(lines)
        for bi, blk in enumerate(blocks):
            btxt = "\n".join(l for _, l in blk)
            around = "\n".join(
                l for j in (bi - 1, bi + 1) if 0 <= j < len(blocks) for _, l in blocks[j]
            )
            # Separate, wider window for "which report does this cite?". A table
            # is typically three blocks below the paragraph naming its report
            # (prose, then a bold column caption, then the table), so the ±1
            # window that bounds the exemptions is too tight to find the
            # citation -- while widening the *exemption* window would start
            # excusing live numbers because a retired table sits nearby.
            cite_ctx = "\n".join(
                l for j in range(bi - 3, bi + 4) if 0 <= j < len(blocks) for _, l in blocks[j]
            )
            for lineno, ln in blk:
                nums = NUM.findall(ln)
                total += len(nums)
                miss = [n for n in nums if n not in known]
                if not miss:
                    continue
                untraceable += len(miss)
                # Supersession markers may trail the table they retire as easily
                # as precede it (the 6-embedder chunker table carries its caveat
                # in the *next* paragraph), so look either side.
                if SUPERSEDED.search(ln) or SUPERSEDED.search(around):
                    exempt_sup += len(miss)
                    continue
                if DATED.search(btxt) or DATED.search(around):
                    exempt_dated += len(miss)
                    continue
                keep = [n for n in miss if (str(doc).replace("\\", "/"), n) not in allow]
                exempt_allow += len(miss) - len(keep)
                if keep:
                    residue.append(
                        (str(doc).replace("\\", "/"), lineno, keep, ln.strip()[:160], cite_ctx)
                    )

    n_res = sum(len(r[2]) for r in residue)
    record(
        "D2 every figure traces to a report", not residue,
        f"{n_res} untraceable of {total} figures across {len(DOCS)} docs "
        f"({untraceable} not found in {len(artifacts)} reports; "
        f"{exempt_sup} cited as superseded, {exempt_dated} in a dated snapshot, "
        f"{exempt_allow} allowlisted)",
    )
    for doc, lineno, keep, ctx, blk in (residue if show_all else residue[:12]):
        print(f"        {doc}:{lineno}  {', '.join(keep)}{nearest(keep[0], blk)}")
        print(f"          {ctx}")
    if not show_all and len(residue) > 12:
        print(f"        ... {len(residue) - 12} more (--list)")


# ------------------------------------------------------------------- D3
def audit_significance_wording() -> None:
    bad: list[str] = []
    for doc in DOCS:
        text = (REPO / doc).read_text(encoding="utf-8")
        # Markdown emphasis splits the verdict phrase: "**not** significant"
        # doesn't match /not significant/, so the sentence reads as a bare
        # "significant" and the check reports the exact opposite of the truth.
        flat = re.sub(r"[*`_]+", "", text.replace("\n", " "))
        # Scope to the parenthetical holding the p-value, not a character
        # window. These docs pack contrasting verdicts a few words apart
        # ("+0.0350 ... survives LOO (Holm-adj 0.0252) -- MRR is ns"), so any
        # window wide enough to catch the verdict also catches its opposite:
        # a +-90 char window produced 27 hits, every one spurious.
        for par in re.finditer(r"\(([^()]{0,300})\)", flat):
            inner = par.group(1)
            ps = [float(x) for x in P_VALUE.findall(inner)]
            # More than one p in the parenthetical means it is a list of arms
            # ("hybrid ... bm25 ... all Holm 0.0000, m2v +0.0217 ns") whose
            # verdict words belong to different arms; not decidable here.
            if len(ps) != 1:
                continue
            p = ps[0]
            loc = f"{str(doc).replace(chr(92), '/')}:{text[:par.start()].count(chr(10)) + 1}"
            if p < 0.05 and NOT_SIG.search(inner) and not IS_SIG.search(inner):
                bad.append(f"{loc}  p={p} called not-significant: ({inner.strip()[:120]})")
            elif p >= 0.05 and IS_SIG.search(inner) and not NOT_SIG.search(inner):
                bad.append(f"{loc}  p={p} called significant: ({inner.strip()[:120]})")
    # WARN, not FAIL. Natural prose has an irreducible false-positive rate here:
    # a parenthetical can attribute its p-value to one arm and its verdict word
    # to another ("all Holm 0.0000, m2v +0.0217 ns"), or quote a retired verdict
    # ("was significant pre-rebuild"). Tuning until it reaches zero would only
    # make it vacuous; it is a reading aid that surfaces the sentences worth
    # re-reading, and the ones it surfaces are cheap to dismiss.
    record(
        "D3 p-value agrees with its verdict word", not bad,
        f"{len(bad)} parentheticals to re-read (known false-positive-prone, see source)",
        warn=True,
    )
    for b in bad[:12]:
        print(f"        {b}")


# ------------------------------------------------------------------- D4
def audit_eval_inputs() -> None:
    cleared = {(e["src"], e["report"]) for e in _allowlist("inputs")}
    stale, missing, exempt = [], [], 0
    for src, reports in EVAL_INPUTS.items():
        p = REPO / src
        if not p.exists():
            missing.append(src)
            continue
        changed = _mtime(p)
        for rname in reports:
            r = RESULTS / rname
            if not r.exists():
                missing.append(f"{rname} (declared under {src})")
                continue
            if changed > _mtime(r):
                if (src, rname) in cleared:
                    exempt += 1
                else:
                    stale.append(f"{rname} predates a change to {src} ({changed:%Y-%m-%d %H:%M})")
    record(
        "D4 reports newer than the inputs they read", not stale,
        f"{len(stale)} of {sum(len(v) for v in EVAL_INPUTS.values())} (input, report) pairs"
        + (f"; {exempt} cleared by allowlist" if exempt else ""),
    )
    for s in stale[:15]:
        print(f"        {s}")
    if missing:
        record("D4b declared paths exist", False, f"{len(missing)} missing: {', '.join(missing[:5])}", warn=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print every D2 hit, not the first 12")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    started = datetime.now(timezone.utc)
    audit_reports()
    audit_numbers(args.list)
    audit_significance_wording()
    audit_eval_inputs()

    fails = [f for f in findings if f[1] == "FAIL"]
    warns = [f for f in findings if f[1] == "WARN"]
    print(f"\n{len(findings)} checks: {len(findings) - len(fails) - len(warns)} pass, "
          f"{len(warns)} warn, {len(fails)} fail")

    if args.report:
        lines = [
            "# Doc-claims audit",
            "",
            f"Run {started:%Y-%m-%d %H:%M} UTC. "
            f"{len(findings) - len(fails) - len(warns)} pass / {len(warns)} warn / {len(fails)} fail.",
            "",
            "Generated by `tools/eval/audit_doc_claims.py`.",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
        lines += [f"| {c} | {s} | {d} |" for c, s, d in findings]
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
