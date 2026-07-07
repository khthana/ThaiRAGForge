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
