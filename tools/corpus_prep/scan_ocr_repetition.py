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
  skipped for a fixed window after it, regardless of whether the table is
  `<table>`-wrapped (some are malformed/never close their tag, which is what
  let one slip through the `<table>` blank in an earlier pass).

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
CURRICULUM_MAP_WINDOW = 8000  # chars after the heading treated as table body
CONTEXT_WINDOW = 300
TABLE_MARKUP = re.compile(r"<t[dr]|<table", re.I)


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
    spans = []
    for m in CURRICULUM_MAP_HEADING.finditer(text):
        spans.append((m.start(), min(len(text), m.start() + CURRICULUM_MAP_WINDOW)))
    return spans


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

    prose_files = sum(1 for _, hits in offenders if any(h[3] == "prose" for h in hits))
    table_only_files = len(offenders) - prose_files

    print(f"Scanned {len(files)} .md files under {ROOT}")
    print(f"Found {len(offenders)} file(s) with a repeated-token run (>= {MIN_REPEAT}x)")
    print(f"  -> {prose_files} have at least one 'prose' hit (worth reading first)")
    print(f"  -> {table_only_files} are 'table'-only hits (deprioritized)\n")

    # prose-tagged files first -- these are the ones worth a human reading them
    offenders.sort(key=lambda pair: not any(h[3] == "prose" for h in pair[1]))

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
        if not any(h[3] == "prose" for h in hits):
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
        "...\") -- the OCR model got stuck and looped instead of transcribing "
        "the real content. This corrupts searchable/citable text, unlike the "
        "same defect inside a data-table cell (checkbox/mapping placeholder), "
        "which was excluded because it doesn't affect retrieval.",
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
