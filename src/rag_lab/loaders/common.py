"""Shared loader helpers: read the file and derive fields from the corpus path.

Corpus layout: <year พ.ศ.>/ครั้งที่ N/<เรื่อง>.md — special sessions (วาระพิเศษ)
use an `s` suffix on the session number: ครั้งที่ Ns.

Each meeting folder may carry a `meeting_manifest.json`: the metadata source of
truth mapping each .md file to its full resolution title and source URL.
Filenames are truncated to ~100 chars by the download tooling, so the manifest
title (recovered from the agenda capture) wins over the filename; files absent
from the manifest fall back to filename-derived metadata.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from pythainlp.util import thai_digit_to_arabic_digit

_DOCUMENT_HEADER = re.compile(r"^\s*#\s*Document:.*$", re.MULTILINE)
_MANIFEST_NAME = "meeting_manifest.json"

_MAPPING_HEADING = re.compile(
    r"Curriculum\s*Mapping|SKILL\s*MAPPING|แผนที่แสดงการกระจายความรับผิดชอบ", re.I
)
_TABLE = re.compile(r"<table.*?</table>", re.S | re.I)
_MAPPING_CHAIN_GAP = 500  # max gap between chained tables (covers a grid split
# across a PDF page break by "---"/"## Page N" markers) before treating the
# next <table> as unrelated content, not a continuation of the mapping grid
_MAPPING_FALLBACK_WINDOW = 8000  # used only when no table follows the heading
# at all (a malformed/never-closing tag) -- same bounding heuristic as
# tools/corpus_prep/scan_ocr_repetition.py's curriculum_map_spans(), kept as a
# separate implementation because that tool is a read-only diagnostic outside
# this package's boundary, not something src/rag_lab imports from

_COURSE_TABLE_MARKER = re.compile(
    r"รหัส\s*/\s*หน่วยกิต|Title\s+and\s+Course\s+description|เปลี่ยนเป็น", re.I
)
_TABLE_CODE = re.compile(r"\d{8}")
_HTML_TAG = re.compile(r"<[^>]*>?")
_WS = re.compile(r"\s+")
_TD_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_CREDIT_TUPLE = re.compile(r"\d\s*\(\s*\d+\s*-\s*\d+\s*-\s*\d+\s*\)")


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8-sig")


@lru_cache(maxsize=None)
def _meeting_manifest(dir_path: str) -> dict[str, dict]:
    """filename -> manifest entry for the folder's meeting_manifest.json.

    Cached per directory for the lifetime of the process; a missing or invalid
    manifest degrades to an empty mapping (filename-derived metadata)."""
    try:
        entries = json.loads((Path(dir_path) / _MANIFEST_NAME).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {e["file"]: e for e in entries if isinstance(e, dict) and e.get("file")}


def parse_path(path: str) -> tuple[str | None, str | None, str]:
    """Return (year, session, title) derived from the path/filename.

    Session keeps the special-meeting suffix (e.g. "3s") so a วาระพิเศษ never
    collides with the regular session of the same number. Title comes from the
    meeting manifest when the file is listed there."""
    p = Path(path)
    entry = _meeting_manifest(str(p.parent)).get(p.name)
    title = (entry or {}).get("title") or p.stem
    session_match = re.search(r"(\d+s?)", p.parent.name)
    session = session_match.group(1) if session_match else None
    grandparent = p.parent.parent.name
    year = grandparent if re.fullmatch(r"\d{4}", grandparent) else None
    return year, session, title


def make_resolution_id(path: str, year: str | None, session: str | None, title: str) -> str:
    if year and session:
        return f"{year}/{session}/{title}"
    return str(Path(path).as_posix())


def is_real_resolution_path(path: Path) -> bool:
    """True if `path` is a genuine resolution file, not one of the
    gitignored non-corpus report files that can share the same root (see
    `iter_corpus_files`). Shared by every `rglob("*.md")` walker that can't
    use `iter_corpus_files`'s relative-to-corpus-root gate directly (its
    `parts[0]` check breaks when `input_dir` already points inside a year
    folder, e.g. `dev_smoke.yaml`) -- this checks the same year+session
    signal via `parse_path`/`make_resolution_id` instead, gate-independent
    of where `path` sits relative to any particular root."""
    year, session, _ = parse_path(str(path))
    return year is not None and session is not None


def iter_corpus_files(corpus_root: Path) -> Iterator[Path]:
    """Yield every live (non-`.dup`) resolution file under `<year>/<session>/*.md`.

    `academic_resolutions/` also holds gitignored non-corpus working/report
    files (entity_tags/, llm_ocr_scan/, top-level *_review.md, ...) that live
    at or near the same root -- a bare `corpus_root.rglob("*.md")` sweeps
    those up too and, once matched against a real corpus resolution_id
    accidentally quoted inside one of those reports, misattributes it as a
    genuine mention. Gate on the same real-year/session structure
    `make_resolution_id` uses to mint an id (vs. its path-fallback) so only
    actual resolution files are ever walked, regardless of what else gets
    dropped into the corpus root later."""
    for f in sorted(corpus_root.rglob("*.md")):
        if f.name.endswith(".dup"):
            continue
        parts = f.relative_to(corpus_root).parts
        if len(parts) < 3 or not re.fullmatch(r"\d{4}", parts[0]):
            continue
        yield f


def strip_document_header(text: str) -> str:
    """Remove the OCR `# Document: <name>.pdf` header line(s); keep everything
    else (including `## Page N` markers the chunkers rely on)."""
    return _DOCUMENT_HEADER.sub("", text, count=1).lstrip("\n")


def strip_mapping_tables(text: str) -> str:
    """Remove Curriculum/SKILL Mapping tables (a PLO/skill x-subject checkbox
    grid) before chunking/embedding: nobody searches or cites a checkbox grid,
    and it's routinely the single largest structural block in a
    curriculum-revision document (a major share of this corpus) -- keeping it
    inflates chunk counts/embedding cost for zero retrieval value.

    Bounded to the actual `<table>` span(s) that follow the heading, chaining
    tables within _MAPPING_CHAIN_GAP chars of each other so a grid split
    across a PDF page break by "---"/"## Page N" markers still counts as one
    continuous table. Falls back to a flat _MAPPING_FALLBACK_WINDOW-char
    removal only when no table follows at all (a malformed/never-closing
    tag). A flat window for every heading was tried first (in the OCR
    repetition scanner) and wrongly swallowed unrelated content that happened
    to start within the window of an unrelated Mapping heading on the same
    page -- this bounding avoids repeating that mistake here."""
    tables = list(_TABLE.finditer(text))
    spans: list[tuple[int, int]] = []
    for m in _MAPPING_HEADING.finditer(text):
        end = m.end()
        cursor = m.end()
        found_any = False
        for t in tables:
            if t.start() < cursor:
                continue
            if t.start() - cursor > _MAPPING_CHAIN_GAP:
                break
            end = t.end()
            cursor = t.end()
            found_any = True
        if found_any:
            spans.append((m.start(), end))
        else:
            spans.append((m.start(), min(len(text), m.start() + _MAPPING_FALLBACK_WINDOW)))
    if not spans:
        return text
    out = []
    cursor = 0
    for start, end in spans:
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _compact_course_table(table_html: str) -> str:
    """Collapse one course-comparison `<table>` to one `CODE title` line per
    unique course code, dropping everything else -- notably the
    paragraph-long English course-description prose that makes these tables
    the corpus's single largest chunks (17,077 chars in one real document,
    see docs/chunker-embedder-comparison-log.md). The credit-tuple (e.g.
    `4 (2-4-6)`) is deliberately dropped from the output, not just the
    description -- it's not searchable content, and keeping code+title with
    nothing but whitespace between them means match_courses's existing
    "followed by a letter" plausibility check tags these codes for free, no
    regex change needed (it previously never matched here: a credit-tuple
    sat between code and title in both the raw HTML and an earlier version
    of this compaction that kept it).

    Anchors on `<td>` cell boundaries, not a flat char-window: in every
    sampled table, a code+credit-tuple cell (e.g. `20626214<br/>4 (2-4-6)`)
    is immediately followed by a cell holding *only* the short title (e.g.
    `Respiratory and Excretory System`) -- the long description lives in a
    *separate* `<tr>` (a `rowspan`'d continuation with no code in its own
    row), so "next cell" reliably lands on the title, never mid-description,
    unlike a char-count window.

    Deduped to first occurrence per code: `rowspan`-driven table
    reconstruction re-emits the same code+title header cells on every
    spanned row (confirmed against raw HTML -- not an OCR misread, since
    the repeated description text differs slightly each time; a table-
    extraction artifact instead), so a naive per-occurrence line would
    repeat the same course 5-20x."""
    normalized = thai_digit_to_arabic_digit(table_html)
    cells = [m.group(1) for m in _TD_CELL.finditer(normalized)]
    seen: set[str] = set()
    lines = []
    for i, cell in enumerate(cells):
        codes = _TABLE_CODE.findall(cell)
        if not codes or codes[0] in seen:
            continue
        code = codes[0]
        seen.add(code)
        title = ""
        if i + 1 < len(cells):
            title = _WS.sub(" ", _HTML_TAG.sub(" ", cells[i + 1])).strip(" -,:")
        lines.append(f"{code} {title}" if title else code)
    return "\n".join(lines)


def strip_course_comparison_tables(text: str) -> str:
    """Simplify (not remove) old/new course-comparison tables -- a different
    table type than strip_mapping_tables targets (that one strips a
    checkbox/PLO grid entirely; this one keeps the course code + a short
    label per course, only dropping the long description prose).

    Detected by header markers unique to this table type ("รหัส/หน่วยกิต",
    "Title and Course description", "เปลี่ยนเป็น") -- but the loosest of
    those three ("เปลี่ยนเป็น", "changed to") is ordinary Thai prose, not
    table-specific, and produced one confirmed false positive: an MoA/joint-
    degree fee table with no course codes at all matched on that phrase
    alone. A second, structural gate fixes it -- also require the table to
    contain at least one course-code-shaped or credit-tuple-shaped run
    (`\\d{8}` / `N (x-y-z)`), which every real course-comparison table has
    and the MoA table didn't -- without narrowing to the single strictest
    header marker, which would also drop real course tables that only use
    the other two headers. Every detected code is preserved verbatim, only
    the text following it is shortened -- but the ORDER matters for
    CourseLoader.match_courses: running this BEFORE match_courses actively
    *improves* tagging coverage (compacted `CODE title` text satisfies
    match_courses's "followed by a letter" check; the raw HTML's
    `CODE<br/>credit-tuple` never did), so a loader that wants both should
    call this first, not after."""
    out = []
    cursor = 0
    touched = False
    for m in _TABLE.finditer(text):
        table = m.group(0)
        if not _COURSE_TABLE_MARKER.search(table):
            continue
        if not (_TABLE_CODE.search(table) or _CREDIT_TUPLE.search(table)):
            continue
        touched = True
        out.append(text[cursor : m.start()])
        out.append(_compact_course_table(table))
        cursor = m.end()
    if not touched:
        return text
    out.append(text[cursor:])
    return "".join(out)


def read_source_url(path: str) -> str | None:
    """Return the resolution's source URL (provenance): the meeting-manifest
    entry when present, else the sibling `<stem>_LINK.txt`."""
    p = Path(path)
    entry = _meeting_manifest(str(p.parent)).get(p.name)
    if entry and entry.get("url"):
        return entry["url"]
    link = p.with_name(f"{p.stem}_LINK.txt")
    if link.exists():
        return link.read_text(encoding="utf-8-sig").strip()
    return None
