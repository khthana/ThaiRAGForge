# -*- coding: utf-8 -*-
"""Scan the corpus for OCR hallucination-loop artifacts (read-only report).

The OCR model occasionally gets stuck and emits the same short token over and
over -- e.g. a misread URL/port string repeated 668x ("8080 8080 8080 ..."),
or a garbled year clause repeated many times ("๒๔.๖๒ ๒๔.๖๒ ..."). Both known
instances were found by eye while doing unrelated work; this script scans
every live `.md` file for the same *shape* (one short token, repeated
consecutively many times) so the corpus-wide extent of the problem is known
before deciding whether/how to clean it up. It only reports -- it does not
delete or edit anything.

Detection: split on whitespace, walk the token stream, flag any run of the
same token repeated >= MIN_REPEAT times. Table/list separators made purely of
punctuation (`-`, `_`, `=`, `|`, `.`, `*`) are ignored -- those repeat
legitimately in markdown (table rules, dotted leaders) and are not OCR
hallucinations. Two classes of legitimate-but-lexically-identical content are
also excluded, per user confirmation that they are false positives by
design, not artifacts:

- `<table>...</table>` spans (blanked before scanning, offsets kept aligned)
  -- course/faculty tables, signature lists etc. legitimately repeat short
  cell values many times.
- "Curriculum Mapping" tables (แผนที่แสดงการกระจายความรับผิดชอบ...): a PLO
  (program learning outcome) x-subject grid that is *supposed to* repeat "0"
  or "1" across dozens of columns per row. Detected by heading text and
  skipped through the end of the `<table>` span(s) that immediately follow it
  -- chaining consecutive tables within CURRICULUM_MAP_CHAIN_GAP chars of each
  other, so a grid split across a page break by "---"/"## Page N" markers
  still counts as one continuous table. Only falls back to a flat
  CURRICULUM_MAP_WINDOW-char skip when no table follows at all (a malformed/
  never-closing tag). A flat window for every heading was tried first and
  wrongly swallowed an unrelated OCR-loop table that happened to start within
  8000 chars of a Curriculum Mapping heading elsewhere on the same page.

Even with both exclusions this is a heuristic, not a precise classifier --
other repetitive-by-design tables (citation lists, etc.) can still surface
as candidates. Read the actual file before trusting any single hit.

Each hit is also tagged "table" or "prose" by checking the ~300 chars around
it for table markup (`<td`/`<tr`/`<table` or 3+ `|` pipes). This is a
priority signal, not a filter: per user feedback, a garbled cell inside a
data table (e.g. a checkbox/mapping placeholder OCR couldn't transcribe) is
low-value to chase because nobody searches/cites that cell and it doesn't
affect retrieval -- whereas a loop inside narrative prose corrupts text that
could actually be searched or cited. "table"-tagged hits are listed but
deprioritized; "prose"-tagged hits are the ones worth reading first.

A third pass looks *inside* `<table>` spans specifically for a narrower defect
that the table-blanking above would otherwise hide completely: the OCR model
looping on a person's name (title + Thai name, e.g. "ผศ.ดร.ศรัณย์ อินทโกสุม")
instead of transcribing the actual row-by-row table content. A name appearing
twice in a change-of-faculty table (old column, new column) is normal; the
same exact name string appearing many times in one table is not -- it means
the model got stuck and the real names further down that table were never
transcribed. Tagged "table-name-loop" and treated with the same priority as
"prose" hits (unlike ordinary "table" hits) since a professor's name is
exactly the kind of thing someone would search/cite.

Besides the console report, writes `academic_resolutions/ocr_repetition_review.md`:
a deduplicated list of source documents with >=1 "prose" hit, one line per
document (curriculum-split siblings collapse back to their shared original,
since a re-OCR has to happen against the one source PDF, not each piece).
This is the list meant to be handed to whoever re-runs OCR -- read-only, it
does not touch any corpus file itself.

Run from the repo root:

    python tools/corpus_prep/scan_ocr_repetition.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "academic_resolutions"
REVIEW_FILE = ROOT / "ocr_repetition_review.md"
SPLIT_PIECE = re.compile(r"__\d+$")

MIN_REPEAT = 8  # same token this many times in a row is not normal prose
BENIGN_TOKEN = re.compile(r"^[\-_=|.*#\s]*$")
TABLE = re.compile(r"<table.*?</table>", re.S | re.I)
CURRICULUM_MAP_HEADING = re.compile(
    r"Curriculum\s*Mapping|แผนที่แสดงการกระจายความรับผิดชอบ", re.I
)
CURRICULUM_MAP_WINDOW = 8000  # fallback when no table follows (malformed/unclosed tag)
CURRICULUM_MAP_CHAIN_GAP = 500  # max gap between chained tables (covers a PLO grid
# split across a page break by "---"/"## Page N" markers) before treating the next
# <table> as unrelated content, not a continuation of the mapping grid
CONTEXT_WINDOW = 300
TABLE_MARKUP = re.compile(r"<t[dr]|<table", re.I)

# Title + Thai name, e.g. "ผศ.ดร.ศรัณย์ อินทโกสุม" or "รศ.ดร.นวลสวาท หิรัญสกลวงศ์".
# Thai-letter class deliberately excludes the digit block (๐-๙ = U+0E50-0E59)
# which otherwise falls inside a naive "[ก-๙]" range and turns "พ.ศ. ๒๕๖๐" (a
# Buddhist-era year) into a false "name" match. Bare "ศ." / "อ." prefixes are
# also excluded -- both collide too often with other abbreviations ("พ.ศ.",
# "อ." for อำเภอ) to be a reliable title marker on their own; "ผศ."/"รศ." are
# unambiguous.
THAI_WORD = r"[ก-ฮะ-๎]+"
PERSON_TITLE = re.compile(
    rf"(?:ผศ|รศ)\.(?:ดร\.)?[ \t]*{THAI_WORD}(?:[ \t]+{THAI_WORD}){{0,3}}"
)
NAME_REPEAT_MIN = 4  # same exact name string this many times in one table is a loop, not a real roster
TR = re.compile(r"<tr>.*?</tr>", re.S | re.I)
ROW_DIGITS = re.compile(r"[0-9๐-๙]+")
DISTINCT_ROW_RATIO_MAX = 0.5  # loops repeat near-identical rows; real rosters vary row to row

# Three more loop shapes found by hand while fixing table-name-loop hits:
# (1) a numbered-list enumerator flooding upward ("๒. ๓. ๔. ... ๓๔๒.") with no
#     real content between the numbers, drowning out whatever cell it replaced;
# (2) a short phrase cycling verbatim many times ("๑๔ ๖-๘ มนาคม ๒๕๖๒" x90) --
#     four *different* tokens repeating as a unit, invisible to find_runs()
#     which only catches one *identical* token repeating.
# (3) a whole garbled sentence repeated line-by-line, found by a user reading
#     a course-description block ("Basic knowledge of food components, ...
#     protein, hina" x78) -- a 26-token unit, past CYCLE_MAX_PERIOD (8) and
#     therefore invisible to find_cyclic_floods() too. See find_line_repeats().
NUMERIC_TOKEN = re.compile(r"^([0-9๐-๙]+)\.$")
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
NUMERIC_FLOOD_MIN = 8  # consecutive N., N+1., N+2., ... this long is not a real list
CYCLE_MIN_PERIOD = 2
CYCLE_MAX_PERIOD = 8
CYCLE_MIN_REPEATS = 6  # the same short token-sequence recurring this many times in a row


def classify_context(text: str, offset: int) -> str:
    window = text[max(0, offset - CONTEXT_WINDOW): offset + CONTEXT_WINDOW]
    if TABLE_MARKUP.search(window) or window.count("|") >= 3:
        return "table"
    return "prose"


def blank_spans(text: str, spans: list[tuple[int, int]]) -> str:
    out = list(text)
    for start, end in spans:
        for i in range(start, end):
            out[i] = " "
    return "".join(out)


def curriculum_map_spans(text: str) -> list[tuple[int, int]]:
    """Span(s) to treat as PLO-grid body for each Curriculum Mapping heading:
    bounded by the actual table(s) that follow, not a blanket char count --
    see the CURRICULUM_MAP_CHAIN_GAP docstring above for why."""
    tables = list(TABLE.finditer(text))
    spans = []
    for m in CURRICULUM_MAP_HEADING.finditer(text):
        end = m.end()
        cursor = m.end()
        found_any = False
        for t in tables:
            if t.start() < cursor:
                continue
            if t.start() - cursor > CURRICULUM_MAP_CHAIN_GAP:
                break
            end = t.end()
            cursor = t.end()
            found_any = True
        if found_any:
            spans.append((m.start(), end))
        else:
            spans.append((m.start(), min(len(text), m.start() + CURRICULUM_MAP_WINDOW)))
    return spans


def _row_signature(row_text: str) -> str:
    """Normalize a <tr> for near-duplicate comparison: collapse whitespace and
    blank out digits, so only the leading enumerator ("๑.", "๒.", ...) differs
    between an otherwise-identical looped row."""
    row_text = ROW_DIGITS.sub("0", row_text)
    return re.sub(r"\s+", " ", row_text).strip()


def find_table_name_loops(text: str) -> list[tuple[str, int, int, str]]:
    """Find a name repeated many times within a single <table> span, but only
    when the rows carrying it are themselves near-duplicates of each other.

    A name legitimately repeating across many *different* rows (e.g. one
    course-responsible instructor listed against a dozen different guest
    lecturers/courses in a roster table) is not a loop -- each row has
    substantial distinct content. A genuine OCR loop repeats the whole row
    almost verbatim (only the leading enumerator changes), which is what
    DISTINCT_ROW_RATIO_MAX below is checking for.
    """
    hits: list[tuple[str, int, int, str]] = []
    for table_m in TABLE.finditer(text):
        span_text = table_m.group()
        rows = [(m.start(), m.group()) for m in TR.finditer(span_text)]
        name_rows: dict[str, list[tuple[int, str]]] = {}
        for row_start, row_text in rows:
            seen_in_row: set[str] = set()
            for name_m in PERSON_TITLE.finditer(row_text):
                name = re.sub(r"\s+", " ", name_m.group()).strip()
                if name in seen_in_row:
                    continue
                seen_in_row.add(name)
                name_rows.setdefault(name, []).append(
                    (table_m.start() + row_start + name_m.start(), row_text)
                )
        for name, occurrences in name_rows.items():
            if len(occurrences) < NAME_REPEAT_MIN:
                continue
            signatures = {_row_signature(row_text) for _, row_text in occurrences}
            if len(signatures) / len(occurrences) <= DISTINCT_ROW_RATIO_MAX:
                hits.append((name, len(occurrences), occurrences[0][0], "table-name-loop"))
    return hits


def _in_any_span(offset: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= offset < end for start, end in spans)


def find_numeric_floods(text: str) -> list[tuple[str, int, int, str]]:
    """Find a run of incrementing 'N.' tokens inside a <table> span -- an OCR
    loop that counts upward instead of transcribing real list/table content.
    Tagged "table"/"prose" via classify_context like find_runs(), so a flood
    inside e.g. a Curriculum Mapping PLO grid (legitimately full of small
    numbers) gets the same low priority as any other table-context hit."""
    cmap_spans = curriculum_map_spans(text)
    hits: list[tuple[str, int, int, str]] = []
    for table_m in TABLE.finditer(text):
        span_text = table_m.group()
        tokens = list(re.finditer(r"\S+", span_text))
        n = len(tokens)
        i = 0
        while i < n:
            m = NUMERIC_TOKEN.match(tokens[i].group())
            if not m:
                i += 1
                continue
            expected = int(m.group(1).translate(THAI_DIGITS)) + 1
            j = i + 1
            while j < n:
                m2 = NUMERIC_TOKEN.match(tokens[j].group())
                if m2 and int(m2.group(1).translate(THAI_DIGITS)) == expected:
                    expected += 1
                    j += 1
                else:
                    break
            run_len = j - i
            if run_len >= NUMERIC_FLOOD_MIN:
                offset = table_m.start() + tokens[i].start()
                if not _in_any_span(offset, cmap_spans):
                    label = f"{tokens[i].group()}..{tokens[j - 1].group()}"
                    hits.append((label, run_len, offset, classify_context(text, offset)))
                i = j
            else:
                i += 1
    return hits


def find_cyclic_floods(text: str) -> list[tuple[str, int, int, str]]:
    """Find a short token sequence (2-8 tokens) that repeats verbatim as a
    unit many times in a row -- e.g. a garbled date cycling "๑๔ ๖-๘ มนาคม
    ๒๕๖๒" over and over. Each individual token differs from its neighbor, so
    find_runs()'s same-token-repeated check can't see this shape. Tagged
    "table"/"prose" like find_runs() -- most hits of this shape turn out to
    be legitimate table content (citation lists, PLO grids) cycling through a
    small fixed vocabulary, not an OCR loop, so they need the same
    deprioritization as ordinary "table" hits."""
    cmap_spans = curriculum_map_spans(text)
    tokens = list(re.finditer(r"\S+", text))
    n = len(tokens)
    hits: list[tuple[str, int, int, str]] = []
    i = 0
    while i < n:
        found = False
        for period in range(CYCLE_MIN_PERIOD, CYCLE_MAX_PERIOD + 1):
            if i + period >= n or tokens[i].group() != tokens[i + period].group():
                continue
            base = [tokens[k].group() for k in range(i, i + period)]
            if all(BENIGN_TOKEN.match(t) for t in base):
                continue
            reps = 1
            j = i + period
            while (
                j + period <= n
                and [tokens[k].group() for k in range(j, j + period)] == base
            ):
                reps += 1
                j += period
            if reps >= CYCLE_MIN_REPEATS:
                offset = tokens[i].start()
                if not _in_any_span(offset, cmap_spans):
                    label = " ".join(base)
                    hits.append((label, reps, offset, classify_context(text, offset)))
                i = j
                found = True
                break
        if not found:
            i += 1
    return hits


LINE_REPEAT_MIN = 4  # same non-trivial line, repeated this many times in a row, is a loop
LINE_REPEAT_MIN_LEN = 20  # ignore short lines (table rules, blank markers, single words)


def find_line_repeats(text: str) -> list[tuple[str, int, int, str]]:
    """Find a whole line (a garbled sentence, not just a short token or a 2-8
    token cycle) repeated verbatim many times in a row -- e.g. a hallucinated
    course description that reads "Basic knowledge of food components,
    ... protein, hina" and then repeats that exact sentence, one per line, 78
    times instead of moving on to the next course. find_cyclic_floods() only
    catches repeating units up to CYCLE_MAX_PERIOD (8) tokens; a 26-word
    sentence like this is invisible to it. Line-based instead of token-based
    because the OCR output puts one repetition per line here, which is a
    cheaper and more reliable signal than re-deriving token-period cycles at
    arbitrary length."""
    cmap_spans = curriculum_map_spans(text)
    hits: list[tuple[str, int, int, str]] = []
    offset = 0
    lines = text.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if len(stripped) < LINE_REPEAT_MIN_LEN:
            offset += len(lines[i]) + 1
            i += 1
            continue
        j = i + 1
        run_offset = offset
        cursor = offset + len(lines[i]) + 1
        while j < n and lines[j].strip() == stripped:
            cursor += len(lines[j]) + 1
            j += 1
        run_len = j - i
        if run_len >= LINE_REPEAT_MIN and not _in_any_span(run_offset, cmap_spans):
            hits.append((stripped, run_len, run_offset, classify_context(text, run_offset)))
        offset = cursor
        i = j
    return hits


def find_runs(text: str) -> list[tuple[str, int, int, str]]:
    """Return (token, run_length, char_offset, context) for each offending run."""
    table_spans = [(m.start(), m.end()) for m in TABLE.finditer(text)]
    scan_text = blank_spans(text, table_spans + curriculum_map_spans(text))

    tokens = list(re.finditer(r"\S+", scan_text))
    hits: list[tuple[str, int, int, str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = i + 1
        while j < n and tokens[j].group() == tokens[i].group():
            j += 1
        run_len = j - i
        tok = tokens[i].group()
        if run_len >= MIN_REPEAT and not BENIGN_TOKEN.match(tok):
            offset = tokens[i].start()
            hits.append((tok, run_len, offset, classify_context(text, offset)))
        i = j
    hits.extend(find_table_name_loops(text))
    hits.extend(find_numeric_floods(text))
    hits.extend(find_cyclic_floods(text))
    hits.extend(find_line_repeats(text))
    return hits


def main() -> None:
    files = sorted(ROOT.rglob("*.md"))
    offenders: list[tuple[Path, list[tuple[str, int, int, str]]]] = []

    for f in files:
        if f.suffix != ".md" or f.name.endswith(".dup"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")
        hits = find_runs(text)
        if hits:
            offenders.append((f, hits))

    priority_files = sum(
        1 for _, hits in offenders if any(h[3] in ("prose", "table-name-loop") for h in hits)
    )
    table_only_files = len(offenders) - priority_files

    print(f"Scanned {len(files)} .md files under {ROOT}")
    print(f"Found {len(offenders)} file(s) with a repeated-token run (>= {MIN_REPEAT}x)")
    print(f"  -> {priority_files} have at least one 'prose' or 'table-name-loop' hit (worth reading first)")
    print(f"  -> {table_only_files} are 'table'-only hits (deprioritized)\n")

    # prose/name-loop-tagged files first -- these are the ones worth a human reading them
    offenders.sort(key=lambda pair: not any(h[3] in ("prose", "table-name-loop") for h in pair[1]))

    for f, hits in offenders:
        rel = f.relative_to(ROOT)
        print(f"[{rel}]")
        for tok, run_len, offset, ctx in hits:
            preview = tok if len(tok) <= 30 else tok[:30] + "…"
            print(f"    [{ctx}] token={preview!r} x{run_len}  (char offset {offset})")
        print()

    write_review_file(offenders)


def write_review_file(offenders: list[tuple[Path, list[tuple[str, int, int, str]]]]) -> None:
    # collapse curriculum-split siblings back to their shared source document
    sources: dict[tuple[Path, str], bool] = {}
    for f, hits in offenders:
        if not any(h[3] in ("prose", "table-name-loop") for h in hits):
            continue
        base_stem = SPLIT_PIECE.sub("", f.stem)
        key = (f.parent, base_stem)
        was_split = bool(SPLIT_PIECE.search(f.stem))
        sources[key] = sources.get(key, False) or was_split

    lines = [
        "# OCR repetition-loop review queue",
        "",
        "Auto-generated by `tools/corpus_prep/scan_ocr_repetition.py`. Read-only "
        "report -- nothing here has been edited or deleted. Each entry is a "
        "source document with at least one 'prose'-context repeated-token run "
        "(a short token repeated >=8x in narrative text, e.g. \"8080 8080 8080 "
        "...\") or a 'table-name-loop' hit (a person's name, e.g. \"ผศ.ดร.ศรัณย์ "
        "อินทโกสุม\", repeated >=4x inside one table instead of the real "
        "row-by-row roster) -- the OCR model got stuck and looped instead of "
        "transcribing the real content. This corrupts searchable/citable text, "
        "unlike the same defect inside a data-table cell (checkbox/mapping "
        "placeholder), which was excluded because it doesn't affect retrieval.",
        "",
        "These need a fresh OCR pass, not a text-splice fix -- the original "
        "content is gone, not just misformatted.",
        "",
        f"{len(sources)} source document(s):",
        "",
    ]
    for (parent, base_stem), was_split in sorted(sources.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        rel_dir = parent.relative_to(ROOT)
        tag = "  _(was split into multiple curriculum pieces)_" if was_split else ""
        lines.append(f"- `{rel_dir}\\{base_stem}.md`{tag}")

    REVIEW_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(sources)} source document(s) to {REVIEW_FILE}")


if __name__ == "__main__":
    main()
