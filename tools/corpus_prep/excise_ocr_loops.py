# -*- coding: utf-8 -*-
"""Mechanically remove OCR hallucination-loop garbage found by
`scan_ocr_repetition.py`, without re-OCRing anything.

Rationale: a hallucination loop is, by construction, junk *appended after*
content the model already transcribed once correctly (see the four loop
shapes documented in scan_ocr_repetition.py). Collapsing each detected run
down to its first occurrence therefore never destroys information that
existed anywhere in the run -- it only removes verbatim repeats. This was
verified by hand against source-PDF page images for two files before writing
this script (2564/12 คณะครุศาสตร์ฯ and 2564/5 คณะครุศาสตร์อุตสาหกรรม...) and
confirmed to hold, plus a third case in pure prose (2564/10 ปริญญาในสาขาวิชา
"B.F.A." x374).

The one exception is a numeric-enumerator flood ("๒. ๓. ๔. ... ๓๔๒."): the
numbers themselves carry no content (a lone "๒." means nothing once
separated from whatever it used to enumerate), so that span is deleted
outright rather than collapsed to one instance -- this is an honest gap, not
a silent repair, where the real cell content is unrecoverable without
re-OCRing that page.

Four edit kinds, one per loop shape in scan_ocr_repetition.py:

- table-name-loop: keep the first occurrence's whole `<tr>`; delete every
  later `<tr>` whose normalized signature (digits blanked) matches one
  already seen for that name.
- numeric-flood: delete the whole incrementing-number span.
- cyclic-flood: keep the first repeat cycle; delete the rest.
- generic identical-token run (find_runs' base case): keep the first token;
  delete the rest of the run.

Dry-run by default (prints a summary + before/after snippet per edit).
Pass --apply to write changes. Each modified file is backed up first as
`<name>.md.bak` (written via Python, not shell copy, since some corpus
filenames are long enough to hit git-bash's MAX_PATH on Windows).

Run from the repo root:

    python tools/corpus_prep/excise_ocr_loops.py            # dry run
    python tools/corpus_prep/excise_ocr_loops.py --apply
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_ocr_repetition as scan  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = scan.ROOT
Edit = tuple[int, int, str, str]  # start, end, replacement, description
TABLE_OPEN = re.compile(r"<table>", re.I)
TABLE_CLOSE = re.compile(r"</table>", re.I)


def find_table_regions(text: str) -> list[tuple[int, int, bool]]:
    """Like scan.TABLE but safe against a missing `</table>`: a table whose
    close tag never got generated (OCR cut off mid-loop) must not be allowed
    to swallow the *next* table's real `</table>` as its own -- that's the
    exact "table blanking swallows real content" bug found in
    split_curriculum_bundles.py. Each region ends at its own `</table>` if
    one exists before the next `<table>`; otherwise it ends right before the
    next `<table>` (or EOF), and `is_closed` is False.

    Returns (open_start, region_end, is_closed) sorted by open_start. For a
    closed region, region_end is the end of its own `</table>` tag.
    """
    opens = [m.start() for m in TABLE_OPEN.finditer(text)]
    regions: list[tuple[int, int, bool]] = []
    for idx, open_start in enumerate(opens):
        next_open = opens[idx + 1] if idx + 1 < len(opens) else len(text)
        close_m = TABLE_CLOSE.search(text, open_start, next_open)
        if close_m:
            regions.append((open_start, close_m.end(), True))
        else:
            regions.append((open_start, next_open, False))
    return regions


PAGE_MARKER = re.compile(r"\n\n---\n\n## Page \d+\n\n")
UNCLOSED_TR_OPEN = re.compile(r"<tr>", re.I)


def close_unclosed_table_edits(text: str, touched_regions: set[tuple[int, int]]) -> list[Edit]:
    """For malformed table regions where we already removed a loop, append
    the `</table>` the region was missing. If generation was cut off mid-row
    (a dangling `<tr>` with no closing `</tr>`), that fragment is deleted too
    -- but never anything past it (a page-break marker or a legitimate
    heading that happens to follow is always preserved)."""
    edits: list[Edit] = []
    for open_start, region_end, is_closed in find_table_regions(text):
        if is_closed or (open_start, region_end) not in touched_regions:
            continue
        span_text = text[open_start:region_end]
        rows = list(scan.TR.finditer(span_text))
        last_good_pos = open_start + (rows[-1].end() if rows else len("<table>"))

        dangling_m = UNCLOSED_TR_OPEN.search(text, last_good_pos, region_end)
        if dangling_m:
            marker_m = PAGE_MARKER.search(text, dangling_m.start(), region_end)
            deletion_end = marker_m.start() if marker_m else region_end
            edits.append((dangling_m.start(), deletion_end, "</table>", "close unclosed table + drop dangling row fragment"))
        else:
            edits.append((last_good_pos, last_good_pos, "</table>", "close unclosed table"))
    return edits


def generic_run_edits(text: str) -> list[Edit]:
    table_spans = [(m.start(), m.end()) for m in scan.TABLE.finditer(text)]
    scan_text = scan.blank_spans(text, table_spans + scan.curriculum_map_spans(text))
    tokens = list(re.finditer(r"\S+", scan_text))
    edits: list[Edit] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = i + 1
        while j < n and tokens[j].group() == tokens[i].group():
            j += 1
        run_len = j - i
        tok = tokens[i].group()
        if run_len >= scan.MIN_REPEAT and not scan.BENIGN_TOKEN.match(tok):
            start, end = tokens[i].end(), tokens[j - 1].end()
            if end > start:
                edits.append((start, end, "", f"generic-run {tok!r} x{run_len}"))
        i = j
    return edits


def numeric_flood_edits(text: str, touched: set[tuple[int, int]]) -> list[Edit]:
    edits: list[Edit] = []
    for open_start, region_end, _is_closed in find_table_regions(text):
        span_text = text[open_start:region_end]
        tokens = list(re.finditer(r"\S+", span_text))
        n = len(tokens)
        i = 0
        while i < n:
            m = scan.NUMERIC_TOKEN.match(tokens[i].group())
            if not m:
                i += 1
                continue
            expected = int(m.group(1).translate(scan.THAI_DIGITS)) + 1
            j = i + 1
            while j < n:
                m2 = scan.NUMERIC_TOKEN.match(tokens[j].group())
                if m2 and int(m2.group(1).translate(scan.THAI_DIGITS)) == expected:
                    expected += 1
                    j += 1
                else:
                    break
            run_len = j - i
            if run_len >= scan.NUMERIC_FLOOD_MIN:
                start = open_start + tokens[i].start()
                end = open_start + tokens[j - 1].end()
                edits.append((start, end, "", f"numeric-flood {tokens[i].group()}..{tokens[j-1].group()} x{run_len}"))
                touched.add((open_start, region_end))
                i = j
            else:
                i += 1
    return edits


def cyclic_flood_edits(text: str) -> list[Edit]:
    cmap_spans = scan.curriculum_map_spans(text)
    tokens = list(re.finditer(r"\S+", text))
    n = len(tokens)
    edits: list[Edit] = []
    i = 0
    while i < n:
        found = False
        for period in range(scan.CYCLE_MIN_PERIOD, scan.CYCLE_MAX_PERIOD + 1):
            if i + period >= n or tokens[i].group() != tokens[i + period].group():
                continue
            base = [tokens[k].group() for k in range(i, i + period)]
            if all(scan.BENIGN_TOKEN.match(t) for t in base):
                continue
            reps = 1
            j = i + period
            while j + period <= n and [tokens[k].group() for k in range(j, j + period)] == base:
                reps += 1
                j += period
            if reps >= scan.CYCLE_MIN_REPEATS:
                # Generation can get cut off mid-cycle, leaving a trailing
                # partial repeat (a prefix of `base`) that wouldn't count as
                # a full rep and would otherwise survive untouched -- fold
                # it into the deleted span too.
                k = j
                extra = 0
                while extra < period and k < n and tokens[k].group() == base[extra]:
                    extra += 1
                    k += 1
                last_kept_end = j - 1 if extra == 0 else k - 1
                offset = tokens[i].start()
                if not scan._in_any_span(offset, cmap_spans):
                    start = tokens[i + period - 1].end()
                    end = tokens[last_kept_end].end()
                    if end > start:
                        edits.append((start, end, "", f"cyclic-flood {' '.join(base)!r} x{reps}"))
                i = k
                found = True
                break
        if not found:
            i += 1
    return edits


def table_name_loop_edits(text: str, touched: set[tuple[int, int]]) -> list[Edit]:
    edits: list[Edit] = []
    for open_start, region_end, _is_closed in find_table_regions(text):
        span_text = text[open_start:region_end]
        rows = [(m.start(), m.end(), m.group()) for m in scan.TR.finditer(span_text)]
        name_rows: dict[str, list[tuple[int, int, str]]] = {}
        for row_start, row_end, row_text in rows:
            seen_in_row: set[str] = set()
            for name_m in scan.PERSON_TITLE.finditer(row_text):
                name = re.sub(r"\s+", " ", name_m.group()).strip()
                if name in seen_in_row:
                    continue
                seen_in_row.add(name)
                name_rows.setdefault(name, []).append((row_start, row_end, row_text))
        for name, occurrences in name_rows.items():
            if len(occurrences) < scan.NAME_REPEAT_MIN:
                continue
            signatures = {scan._row_signature(r[2]) for r in occurrences}
            if len(signatures) / len(occurrences) > scan.DISTINCT_ROW_RATIO_MAX:
                continue
            # Keep the first occurrence; delete every later row whose
            # signature has already been seen (i.e. is a near-duplicate),
            # so a genuinely distinct row sandwiched in the middle survives.
            seen_sigs = {scan._row_signature(occurrences[0][2])}
            for row_start, row_end, row_text in occurrences[1:]:
                sig = scan._row_signature(row_text)
                if sig in seen_sigs:
                    start = open_start + row_start
                    end = open_start + row_end
                    edits.append((start, end, "", f"table-name-loop row for {name!r}"))
                    touched.add((open_start, region_end))
                else:
                    seen_sigs.add(sig)
    return edits


def collect_edits(text: str) -> list[Edit]:
    touched_regions: set[tuple[int, int]] = set()
    edits = (
        generic_run_edits(text)
        + numeric_flood_edits(text, touched_regions)
        + cyclic_flood_edits(text)
        + table_name_loop_edits(text, touched_regions)
    )
    edits += close_unclosed_table_edits(text, touched_regions)
    # Later edits must not overlap earlier ones -- sort and drop overlaps
    # defensively (shouldn't happen given the four detectors are disjoint by
    # construction, but corpus text is messy; better safe than corrupting).
    edits.sort(key=lambda e: e[0])
    clean: list[Edit] = []
    last_end = -1
    for start, end, repl, desc in edits:
        if start < last_end:
            continue
        clean.append((start, end, repl, desc))
        last_end = end
    return clean


def apply_edits(text: str, edits: list[Edit]) -> str:
    out = text
    for start, end, repl, _desc in sorted(edits, key=lambda e: e[0], reverse=True):
        out = out[:start] + repl + out[end:]
    return out


def main() -> None:
    apply = "--apply" in sys.argv
    files = sorted(ROOT.rglob("*.md"))
    total_edits = 0
    touched_files = 0

    for f in files:
        if f.suffix != ".md" or f.name.endswith(".dup"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")

        edits = collect_edits(text)
        if not edits:
            continue

        touched_files += 1
        total_edits += len(edits)
        rel = f.relative_to(ROOT)
        print(f"[{rel}]  {len(edits)} edit(s)")
        for start, end, _repl, desc in edits:
            removed_len = end - start
            print(f"    {desc}  (removing {removed_len} chars at offset {start})")

        if apply:
            new_text = apply_edits(text, edits)
            backup = f.with_suffix(f.suffix + ".bak")
            if not backup.exists():
                shutil.copy(f, backup)
            f.write_text(new_text, encoding="utf-8")

    print(f"\n{touched_files} file(s), {total_edits} edit(s) total.")
    print("(dry run -- pass --apply to write changes)" if not apply else "Applied.")


if __name__ == "__main__":
    main()
