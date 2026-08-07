"""Diff two significance-test markdown reports by *verdict*, not by line position.

Written because this diff has now been needed twice (the 2026-08-06 rebuild-#3
refresh and the 2026-08-07 thematic refresh) and thrown away once. Both times the
question was the same: after re-running an eval, did any conclusion actually
change, or only the digits?

Two things make `diff` the wrong tool here:

  * **Rows move.** Several report generators sort pairwise rows by effect size or
    p-value, so a run where nothing changed still produces a large positional
    diff the moment two adjacent rows swap.
  * **Labels repeat across sections.** `bge_m3` appears under
    "recall@10: hybrid_E vs BM25-alone" *and* under
    "recall@10: hybrid_E vs dense-alone_E". Collapsing every table into one
    label->verdict dict silently overwrites one with the other.

So the key is `(section heading, leading label cells)` and the compared value is
the `significant` column, with numeric columns reported as movement. A row's
identity is its leading non-numeric cells: `| bge_m3 | recall | +0.0800 | ...`
keys on `("bge_m3", "recall")`.

Cells that are neither numeric nor a verdict get their own report and also gate
the exit. Added 2026-08-07 after this script silently passed a real regression:
`rq4_score.md`'s phantom-citation column is formatted `count/total`, so `0/370`
-> `4/359` (fabricated citations appearing where there had been none) matched
neither the numeric branch nor the verdict branch and was skipped. A column this
script cannot classify is exactly the one worth showing, not dropping.

Exit code is 1 if any verdict flipped, any row appeared/disappeared, or any
non-numeric cell changed, so this can gate a refresh; numeric drift alone
exits 0.

Run:
    .venv/Scripts/python.exe tools/eval/diff_significance_reports.py OLD.md NEW.md
    ... --threshold 0.02      # only report numeric moves above this
    ... --quiet               # verdict changes only, no movement table
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")
_VERDICT = re.compile(r"^\*{0,2}(yes|no)\*{0,2}$", re.IGNORECASE)
# A confidence interval is a bracketed pair of numbers -- numeric in substance, so
# it is excluded from the non-numeric report (its endpoints move on every re-run
# and would bury the columns that report actually exists to surface).
_CI = re.compile(r"^\[[+-]?[\d.]+,\s*[+-]?[\d.]+\]$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s|:-]+\|?", line.strip()))


def _is_label(cell: str) -> bool:
    """Leading label cells identify a row; numbers, CIs and verdicts do not."""
    if not cell or _NUMERIC.match(cell) or _VERDICT.match(cell):
        return False
    return not cell.startswith("[")


def parse(path: Path) -> dict[tuple, dict[str, str]]:
    """(section, label-cells) -> {column header: cell}."""
    section = ""
    headers: list[str] = []
    in_table = False
    rows: dict[tuple, dict[str, str]] = {}

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            in_table = False
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        if _is_separator(line):
            in_table = bool(headers)
            continue
        cells = _cells(line)
        if not in_table:
            headers = cells  # header row; the separator that follows opens the table
            continue

        labels = tuple(c for c in _takewhile_label(cells))
        if not labels:
            continue
        cols = list(headers) + [f"col{i}" for i in range(len(headers), len(cells))]
        key = (section, labels)
        if key in rows:
            # Same label twice in one section is a report bug, not a diff problem
            print(f"  !! duplicate key {key} in {path.name}", file=sys.stderr)
        rows[key] = dict(zip(cols, cells))
    return rows


def _takewhile_label(cells: list[str]):
    for c in cells:
        if not _is_label(c):
            return
        yield c


def _verdict(row: dict[str, str]) -> str | None:
    for header, cell in row.items():
        if "significant" in header.lower() and _VERDICT.match(cell):
            return _VERDICT.match(cell).group(1).lower()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="report numeric moves at least this large (default 0.02)")
    ap.add_argument("--quiet", action="store_true", help="verdict changes only")
    args = ap.parse_args()

    old, new = parse(args.old), parse(args.new)
    print(f"{args.old.name}: {len(old)} keyed rows")
    print(f"{args.new.name}: {len(new)} keyed rows\n")

    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    flips, moves, other = [], [], []
    for key in sorted(set(old) & set(new)):
        vo, vn = _verdict(old[key]), _verdict(new[key])
        if vo is not None and vo != vn:
            flips.append((key, vo, vn))
        for header, cell in new[key].items():
            prev = old[key].get(header)
            if prev is None:
                continue
            if _NUMERIC.match(cell) and _NUMERIC.match(prev):
                delta = float(cell) - float(prev)
                if abs(delta) >= args.threshold:
                    moves.append((key, header, float(prev), float(cell), delta))
            elif (cell != prev
                    and not (_VERDICT.match(cell) and _VERDICT.match(prev))
                    and not (_CI.match(cell) and _CI.match(prev))):
                # Neither a number nor a verdict -- e.g. the `count/total`
                # phantom-citation column. Never drop these (see module docstring).
                other.append((key, header, prev, cell))

    n_verdicts = sum(1 for k in set(old) & set(new) if _verdict(old[k]) is not None)
    print(f"VERDICT FLIPS: {len(flips)} of {n_verdicts} rows carrying a verdict")
    for (section, labels), vo, vn in flips:
        print(f"  {section} | {' / '.join(labels)}: {vo} -> {vn}")

    print(f"\nNON-NUMERIC CELL CHANGES: {len(other)}")
    for (section, labels), header, a, b in other:
        print(f"  {section} | {' / '.join(labels)} | {header}: {a!r} -> {b!r}")

    if only_old or only_new:
        print(f"\nROWS ONLY IN OLD: {len(only_old)}")
        for section, labels in only_old:
            print(f"  {section} | {' / '.join(labels)}")
        print(f"ROWS ONLY IN NEW: {len(only_new)}")
        for section, labels in only_new:
            print(f"  {section} | {' / '.join(labels)}")

    if not args.quiet:
        print(f"\nNUMERIC MOVES >= {args.threshold}: {len(moves)}")
        for (section, labels), header, a, b, d in sorted(
                moves, key=lambda m: -abs(m[4]))[:60]:
            print(f"  {section} | {' / '.join(labels)} | {header}: "
                  f"{a:+.4f} -> {b:+.4f} ({d:+.4f})")

    return 1 if (flips or only_old or only_new or other) else 0


if __name__ == "__main__":
    raise SystemExit(main())
