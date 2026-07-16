"""Pure, Streamlit-free logic for the consensus review app.

Parses `consensus_priority.md` (built from `full_review_<year>.md` by the
scan in `llm_ocr_scan.py`), extracts a flagged page's full Markdown live from
the real corpus file, detects split-document siblings, persists reviewer
verdicts to an append-only decision log, and generates the re-OCR worklist
from that log. See tools/corpus_prep/consensus_review/SPEC.md and
tickets.md (tickets 1-3).
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
from reocr_adjudicate import load_full_page_text  # noqa: E402


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


def highlight_spans(body: str, spans: list[str]) -> str:
    """Wrap each span that appears verbatim in `body` with <mark></mark> so
    it renders highlighted (caller must render with unsafe_allow_html=True).
    A model's quoted span sometimes doesn't match the corpus text exactly
    (paraphrased or trimmed) -- those are silently skipped, not an error.
    Processes longer spans first so a short span that's a substring of a
    longer one doesn't get double-wrapped."""
    highlighted = body
    for span in sorted((s.strip() for s in spans), key=len, reverse=True):
        if span and span in highlighted:
            highlighted = highlighted.replace(span, f"<mark>{span}</mark>")
    return highlighted


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
    """A plain list of full relative Resolution file paths whose current
    verdict is VERDICT_REOCR, one per line, sorted for a deterministic diff
    between regenerations. Empty string if none. No header/decoration --
    this is meant to be directly consumable by the next (manual, scripted-or-
    not) re-OCR-diff step, not just read by eye."""
    files = sorted(d.file for d in resolved.values() if d.verdict == VERDICT_REOCR)
    if not files:
        return ""
    return "\n".join(files) + "\n"


def write_worklist(worklist_path: Path, resolved: dict[str, Decision]) -> int:
    """Write the re-OCR worklist, replacing any prior contents in full.
    Returns the number of files written, so callers don't have to re-derive
    the same VERDICT_REOCR filter just to report a count."""
    worklist_path = Path(worklist_path)
    worklist_path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_worklist(resolved)
    worklist_path.write_text(content, encoding="utf-8")
    return len(content.splitlines())


# --- Old-vs-new re-OCR adjudication review (reocr_consensus_pages.py /
# reocr_adjudicate.py, Phase 1+2) -----------------------------------------
#
# A different grain from the rest of this module: verdicts above are one per
# *file*, but Phase 2 adjudicates one physical (source PDF, page number) at a
# time -- a page can be shared by several split-document sibling files. The
# decision log below is keyed the same way.


def load_jsonl(path: Path) -> list[dict]:
    """Every JSON record in an append-only JSONL log, in file order. Empty
    list if the file doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def needs_reocr_review(record: dict) -> bool:
    """False only when both models independently verdict "new" -- the one
    pairing strong enough to auto-apply without a human decision. Every
    other pairing (disagreement, both prefer old, both say no real
    difference, both still bad) needs a human to look."""
    verdicts = record.get("verdicts", {})
    return not (verdicts and all(v.get("verdict") == "new" for v in verdicts.values()))


def staged_text_by_key(staging_records: list[dict]) -> dict[tuple[str, int], str]:
    """(pdf, page) -> the Phase 1 fresh re-OCR text, from staged JSONL
    records."""
    return {(r["pdf"], r["page"]): r["new_text"] for r in staging_records}


def load_old_text(corpus_root: Path, record: dict) -> str | None:
    """The corpus's current text for an adjudication record's page, read from
    `old_text_source` (the first sibling file Phase 2 resolved it from).
    Reuses `reocr_adjudicate.load_full_page_text` so oversized pages that
    `split_pages` sub-chunks ("Page N.1", "Page N.2", ...) are reassembled
    the same way Phase 2 itself read them -- an exact "Page N" lookup would
    silently return None for those."""
    return load_full_page_text(Path(corpus_root), record["old_text_source"], record["page"])


def meeting_info(corpus_root: Path, relpath: str) -> dict | None:
    """Year, session, title, and source URL for the meeting a corpus-relative
    path (e.g. "2564/ครั้งที่ 11/xxx.md") belongs to -- year/session come
    from the path itself (corpus layout, ADR-0003), title/url come from that
    meeting's `meeting_manifest.json` (the metadata source of truth, never
    the filename). Reviewers need this to go find the actual source PDF/Drive
    link to check against, since the filename alone doesn't say which meeting
    a flagged page came from. None if `relpath` has no year/session prefix or
    the manifest doesn't exist; title/url within the result are None if no
    manifest entry matches this exact filename."""
    parts = Path(relpath).parts
    if len(parts) < 3:
        return None
    year, session, filename = parts[0], parts[1], parts[-1]
    manifest_path = Path(corpus_root) / year / session / "meeting_manifest.json"
    if not manifest_path.exists():
        return None
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((e for e in entries if e.get("file") == filename), None)
    return {
        "year": year,
        "session": session,
        "title": entry.get("title") if entry else None,
        "url": entry.get("url") if entry else None,
    }


REOCR_VERDICT_APPLY_NEW = "ใช้ข้อความใหม่ (new)"
REOCR_VERDICT_KEEP_OLD = "เก็บของเดิม (old)"
REOCR_VERDICT_DEFER = "รอไว้ก่อน (ทั้งคู่ยังพัง/ไม่แน่ใจ)"
REOCR_REVIEW_VERDICTS = (REOCR_VERDICT_APPLY_NEW, REOCR_VERDICT_KEEP_OLD, REOCR_VERDICT_DEFER)


@dataclass(frozen=True)
class ReocrReviewDecision:
    pdf: str
    page: int
    verdict: str
    note: str = ""
    timestamp: str = ""


def append_reocr_review_decision(
    log_path: Path,
    pdf: str,
    page: int,
    verdict: str,
    note: str = "",
    timestamp: str | None = None,
) -> None:
    """Append one verdict record for a (pdf, page) to the append-only
    decision log -- never rewrites or removes an earlier line, same
    convention as `append_decision`."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    decision = ReocrReviewDecision(
        pdf=pdf,
        page=page,
        verdict=verdict,
        note=note,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")


def load_reocr_review_decisions(log_path: Path) -> list[ReocrReviewDecision]:
    """Every record in the re-OCR review decision log, in append order.
    Empty list if the log doesn't exist yet."""
    return [ReocrReviewDecision(**d) for d in load_jsonl(log_path)]


def resolve_reocr_review_decisions(
    decisions: list[ReocrReviewDecision],
) -> dict[tuple[str, int], ReocrReviewDecision]:
    """Current per-(pdf, page) state: the latest record wins over any
    earlier one for the same page, per the log's append-only design."""
    resolved: dict[tuple[str, int], ReocrReviewDecision] = {}
    for d in decisions:
        resolved[(d.pdf, d.page)] = d
    return resolved
