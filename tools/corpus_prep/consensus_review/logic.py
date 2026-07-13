"""Pure, Streamlit-free logic for the consensus review app.

Parses `consensus_priority.md` (built from `full_review_<year>.md` by the
scan in `llm_ocr_scan.py`), extracts a flagged page's full Markdown live from
the real corpus file, detects split-document siblings, and persists reviewer
verdicts to an append-only decision log. See
tools/corpus_prep/consensus_review/SPEC.md and tickets.md (tickets 1-2).
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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


def is_split_piece(relpath: str) -> bool:
    """True if `relpath`'s filename matches the split-document `__N` naming
    convention (`llm_ocr_scan._source_key` / `SPLIT_PIECE`), regardless of
    whether any sibling piece is present anywhere. This is the badge trigger:
    the risk category is "this file is a split piece," not "another piece
    happens to also be consensus-flagged.\""""
    return _source_key(Path(relpath))[2]


def consensus_siblings(entries: list[FileEntry], target: FileEntry) -> list[str]:
    """Other files in `entries` (the consensus-flagged set, not the whole
    corpus) that are pieces of the same split document as `target`, using
    `llm_ocr_scan._source_key` to recognise the `__N` naming convention.
    Empty if `target` isn't a split piece, or no sibling piece also happens
    to be consensus-flagged -- use `is_split_piece` for the badge itself,
    this is only the supplementary "which siblings are already on your list"
    detail."""
    parent, stem, is_split = _source_key(Path(target.file))
    if not is_split:
        return []

    siblings = []
    for e in entries:
        if e.year != target.year or e.file == target.file:
            continue
        other_parent, other_stem, other_is_split = _source_key(Path(e.file))
        if other_is_split and other_parent == parent and other_stem == stem:
            siblings.append(e.file)
    return siblings


VERDICT_REOCR = "ควร re-OCR"
VERDICT_FALSE_POSITIVE = "false positive"
VERDICT_UNSURE = "ไม่แน่ใจ"
VERDICTS = (VERDICT_REOCR, VERDICT_FALSE_POSITIVE, VERDICT_UNSURE)


@dataclass(frozen=True)
class Decision:
    year: str
    file: str
    verdict: str
    note: str = ""
    timestamp: str = ""


def append_decision(
    log_path: Path,
    year: str,
    file: str,
    verdict: str,
    note: str = "",
    timestamp: str | None = None,
) -> None:
    """Append one verdict record to the append-only decision log. Never
    rewrites or removes any earlier line -- `resolve_decisions` is what
    decides which record for a file currently applies."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    decision = Decision(
        year=year,
        file=file,
        verdict=verdict,
        note=note,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")


def load_decisions(log_path: Path) -> list[Decision]:
    """Every record in the decision log, in append order. Empty list if the
    log doesn't exist yet (nothing decided so far)."""
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    decisions = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        decisions.append(Decision(**json.loads(line)))
    return decisions


def resolve_decisions(decisions: list[Decision]) -> dict[str, Decision]:
    """Current per-file state: the latest record for each file (by append
    order) wins over any earlier one for the same file, per the log's
    append-only, no-in-place-rewrite design."""
    resolved: dict[str, Decision] = {}
    for d in decisions:
        resolved[d.file] = d
    return resolved


def generate_worklist(resolved: dict[str, Decision]) -> str:
    """Markdown content for the re-OCR worklist: every file whose current
    verdict is VERDICT_REOCR, sorted for a deterministic diff between
    regenerations."""
    files = sorted(d.file for d in resolved.values() if d.verdict == VERDICT_REOCR)
    lines = [
        "# Re-OCR worklist",
        "",
        "Auto-generated from `review_decisions.jsonl` -- regenerated in full "
        "each time, never appended.",
        "",
        f"{len(files)} file(s) marked \"{VERDICT_REOCR}\":",
        "",
    ]
    lines.extend(files)
    return "\n".join(lines) + "\n"


def write_worklist(worklist_path: Path, resolved: dict[str, Decision]) -> None:
    """Write the re-OCR worklist, replacing any prior contents in full."""
    worklist_path = Path(worklist_path)
    worklist_path.parent.mkdir(parents=True, exist_ok=True)
    worklist_path.write_text(generate_worklist(resolved), encoding="utf-8")
