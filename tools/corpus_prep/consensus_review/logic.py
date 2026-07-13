"""Pure, Streamlit-free logic for the consensus review app.

Parses `consensus_priority.md` (built from `full_review_<year>.md` by the
scan in `llm_ocr_scan.py`), extracts a flagged page's full Markdown live from
the real corpus file, and detects split-document siblings. See
tools/corpus_prep/consensus_review/SPEC.md and tickets.md (ticket 1).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# tools/corpus_prep/ has no __init__.py -- add it to sys.path directly and
# import the bare module name, same convention app/pages/1_build_run.py uses
# for src/ (insert then plain import).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_ocr_scan import _source_key, split_pages  # noqa: E402


@dataclass(frozen=True)
class ModelFlag:
    reason: str
    span: str = ""


@dataclass
class PageEntry:
    page: str
    models: dict[str, ModelFlag] = field(default_factory=dict)


@dataclass
class FileEntry:
    year: str
    file: str  # path relative to academic_resolutions/, corpus-native separators
    pages: list[PageEntry] = field(default_factory=list)


_FILE_HEADER = re.compile(r"^## \[(?P<year>\d{4})\] (?P<file>.+?)\s+\(\d+ consensus page\(s\)\)\s*$")
_PAGE_HEADER = re.compile(r"^### (?P<page>.+)$")
_MODEL_LINE = re.compile(r"^- \*\*\[(?P<model>.*?)\]\*\* (?P<reason>.*)$")
_SPAN_LINE = re.compile(r"^\s*> (?P<span>.*)$")


def parse_consensus_priority(path: Path) -> list[FileEntry]:
    """Parse consensus_priority.md, preserving its existing file order
    (already sorted by descending consensus-page count -- this function does
    not re-sort)."""
    entries: list[FileEntry] = []
    current_file: FileEntry | None = None
    current_page: PageEntry | None = None
    pending_model: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        m = _FILE_HEADER.match(line)
        if m:
            current_file = FileEntry(year=m["year"], file=m["file"])
            entries.append(current_file)
            current_page = None
            pending_model = None
            continue

        m = _PAGE_HEADER.match(line)
        if m and current_file is not None:
            current_page = PageEntry(page=m["page"])
            current_file.pages.append(current_page)
            pending_model = None
            continue

        m = _MODEL_LINE.match(line)
        if m and current_page is not None:
            pending_model = m["model"]
            current_page.models[pending_model] = ModelFlag(reason=m["reason"])
            continue

        m = _SPAN_LINE.match(line)
        if m and current_page is not None and pending_model is not None:
            current_page.models[pending_model] = ModelFlag(
                reason=current_page.models[pending_model].reason, span=m["span"]
            )
            pending_model = None
            continue

    return entries


def load_page_markdown(corpus_root: Path, relpath: str, page_label: str) -> str | None:
    """Read the actual corpus file and return the full Markdown body of the
    page matching `page_label` (e.g. "Page 1"), splitting the same way
    `llm_ocr_scan.split_pages` does. None if the file or page is missing."""
    file_path = Path(corpus_root) / relpath
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = file_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None

    for label, body in split_pages(text):
        if label == page_label:
            return body
    return None


def split_siblings(entries: list[FileEntry], year: str, relpath: str) -> list[str]:
    """Other files in `entries` (the consensus-flagged set, not the whole
    corpus) that are pieces of the same split document as `relpath`, using
    `llm_ocr_scan._source_key` to recognise the `__N` naming convention.
    Empty if `relpath` isn't a split piece, or has no sibling among
    `entries`."""
    parent, stem, is_split = _source_key(Path(relpath))
    if not is_split:
        return []

    siblings = []
    for e in entries:
        if e.year != year or e.file == relpath:
            continue
        other_parent, other_stem, other_is_split = _source_key(Path(e.file))
        if other_is_split and other_parent == parent and other_stem == stem:
            siblings.append(e.file)
    return siblings
