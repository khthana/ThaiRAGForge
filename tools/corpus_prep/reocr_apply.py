"""Phase 3 of the consensus-flagged re-OCR pipeline: write adjudicated re-OCR
text back into the real corpus.

Combines three sources of truth per (pdf, page):
- `reocr_adjudication.jsonl` (Phase 2, `reocr_adjudicate.py`) -- auto-apply
  when both models independently verdict "new" (729/760 pages).
- `reocr_review_decisions.jsonl` (human review UI,
  `consensus_review/pages/1_reocr_diff_review.py`) -- apply-new / keep-old /
  defer for every other page. A page with no decision yet is left untouched
  and reported as still pending -- this script never guesses.
- `reocr_pages_staging.jsonl` (Phase 1, `reocr_consensus_pages.py`) -- the
  actual fresh-OCR replacement text for an apply decision.

A page can be shared by several split-document sibling files (ADR-0004); the
same replacement text is written to every file in the adjudication record's
`files` list, not just the one Phase 2 read as `old_text_source`.

Each corpus file is backed up at most once, to `<name>.md.pre_reocr.bak`,
before its first write -- same backup convention as the 2026-07 re-OCR round
(see docs/llm-ocr-scan-log.md, [[project_index_rebuild_pending]]) and the same
"only copy if the backup doesn't already exist" idempotency as
`excise_ocr_loops.py`.

Dry-run by default -- prints/writes a report but touches no corpus file. Pass
--apply to actually write. Safe to re-run: a page already carrying its
adjudicated text is a no-op (no duplicate backup, no rewrite).

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/reocr_apply.py           # dry run
    .venv/Scripts/python.exe tools/corpus_prep/reocr_apply.py --apply   # writes
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "consensus_review"))
import logic as review_logic  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
STAGING_FILE = CORPUS_ROOT / "llm_ocr_scan" / "reocr_pages_staging.jsonl"
ADJUDICATION_FILE = CORPUS_ROOT / "llm_ocr_scan" / "reocr_adjudication.jsonl"
DECISIONS_FILE = CORPUS_ROOT / "llm_ocr_scan" / "reocr_review_decisions.jsonl"
REPORT_FILE = CORPUS_ROOT / "llm_ocr_scan" / "reocr_apply_report.md"
PAGE_OCCURRENCE_OVERRIDES_FILE = CORPUS_ROOT / "llm_ocr_scan" / "reocr_page_occurrence_overrides.json"

PAGE_HEADER = re.compile(r"^## Page (\d+)\s*$", re.M)

ACTION_APPLY = "apply"
ACTION_SKIP = "skip"


@dataclass(frozen=True)
class ApplyDecision:
    action: str  # ACTION_APPLY | ACTION_SKIP
    reason: str


def decide_action(record: dict, decisions: dict) -> ApplyDecision:
    """What to do with one adjudication record. Auto-apply requires both
    models to agree "new" (`needs_reocr_review` false); every other pairing
    needs an explicit human decision, and "no decision yet" is its own skip
    reason distinct from an explicit keep-old/defer -- callers use this to
    tell "pending review" apart from "reviewed, staying old"."""
    if not review_logic.needs_reocr_review(record):
        return ApplyDecision(ACTION_APPLY, "both models verdict new")

    decision = decisions.get((record["pdf"], record["page"]))
    if decision is None:
        return ApplyDecision(ACTION_SKIP, "awaiting human review")
    if decision.verdict == review_logic.REOCR_VERDICT_APPLY_NEW:
        reason = "human decision: apply-new"
        return ApplyDecision(ACTION_APPLY, reason + (f" ({decision.note})" if decision.note else ""))
    if decision.verdict == review_logic.REOCR_VERDICT_KEEP_OLD:
        return ApplyDecision(ACTION_SKIP, "human decision: keep-old")
    return ApplyDecision(ACTION_SKIP, "human decision: defer")


def load_page_occurrence_overrides(path: Path = PAGE_OCCURRENCE_OVERRIDES_FILE) -> dict[str, dict[str, int]]:
    """Corpus files where a page number is genuinely ambiguous -- more than
    one real `## Page N` header for the same N -- because the original OCR
    ingestion duplicated a page number instead of incrementing it (confirmed
    2026-07-16 across all 10 header-ambiguity files: the first occurrence is
    always generic meeting-item boilerplate that recurs verbatim across every
    curriculum item, the second is the actual page-specific content; verified
    by confirming each flagged page's own quoted model `span` text appears
    only in the second occurrence, never the first). Maps corpus-relative
    path -> {str(page number): 1-based occurrence to replace}. Empty dict if
    the file doesn't exist yet (no file has an override)."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def replace_page_text(text: str, page_num: int, new_body: str, occurrence: int | None = None) -> str | None:
    """Substitute the body of a '## Page N' section with `new_body`. Real
    corpus files normally carry exactly one physical header per page
    (`llm_ocr_scan.split_pages`'s "N.1"/"N.2" sub-chunking is an in-memory
    LLM-budget device, never a real file header) -- so by default this
    refuses to guess when that invariant doesn't hold: None if the header is
    missing, or if it's ambiguous (more than one). Pass `occurrence` (1-based)
    to pick a specific one among several matches instead -- for the small,
    hand-verified set of files where duplication is a known, confirmed defect
    (see `load_page_occurrence_overrides`), not a guess made here."""
    headers = list(PAGE_HEADER.finditer(text))
    matches = [i for i, m in enumerate(headers) if int(m.group(1)) == page_num]
    if occurrence is not None:
        if not (1 <= occurrence <= len(matches)):
            return None
        i = matches[occurrence - 1]
    elif len(matches) == 1:
        i = matches[0]
    else:
        return None
    start = headers[i].end()
    end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
    # `PAGE_HEADER`'s `\s*$` always backtracks to consume exactly the header
    # line's own trailing newline (never further -- `$` needs to land right
    # before the *next* "\n" to match), so text[:start] already ends in one
    # "\n"; only one more is needed to open a blank line before the body.
    return text[:start] + "\n" + new_body.strip() + "\n\n" + text[end:]


@dataclass(frozen=True)
class FileWriteResult:
    status: str  # "written" | "unchanged" | "missing_header"


def apply_to_file(
    path: Path, page_num: int, new_body: str, apply: bool, occurrence: int | None = None,
) -> FileWriteResult:
    """Replace one page's text in one corpus file. `apply=False` computes and
    reports the outcome without touching disk (dry run). Backing up happens
    only on an actual change, and only once per file (mirrors
    `excise_ocr_loops.py`'s `if not backup.exists()`)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")

    replaced = replace_page_text(text, page_num, new_body, occurrence=occurrence)
    if replaced is None:
        return FileWriteResult("missing_header")
    if replaced == text:
        return FileWriteResult("unchanged")

    if apply:
        backup = path.with_suffix(path.suffix + ".pre_reocr.bak")
        if not backup.exists():
            shutil.copy(path, backup)
        path.write_text(replaced, encoding="utf-8")
    return FileWriteResult("written")


def main() -> None:
    apply = "--apply" in sys.argv

    adjudication_records = review_logic.load_jsonl(ADJUDICATION_FILE)
    staged_text = review_logic.staged_text_by_key(review_logic.load_jsonl(STAGING_FILE))
    decisions = review_logic.resolve_reocr_review_decisions(
        review_logic.load_reocr_review_decisions(DECISIONS_FILE)
    )
    occurrence_overrides = load_page_occurrence_overrides()

    skip_reasons: dict[str, int] = {}
    file_results: dict[str, int] = {}
    problems: list[str] = []
    applied_pages = 0

    for record in adjudication_records:
        decision = decide_action(record, decisions)
        if decision.action == ACTION_SKIP:
            skip_reasons[decision.reason] = skip_reasons.get(decision.reason, 0) + 1
            continue

        key = (record["pdf"], record["page"])
        new_text = staged_text.get(key)
        if new_text is None:
            problems.append(f"{record['pdf']} page {record['page']}: no staged re-OCR text found")
            continue

        applied_pages += 1
        for relpath in record["files"]:
            occurrence = occurrence_overrides.get(relpath, {}).get(str(record["page"]))
            result = apply_to_file(
                CORPUS_ROOT / relpath, record["page"], new_text, apply, occurrence=occurrence,
            )
            file_results[result.status] = file_results.get(result.status, 0) + 1
            if result.status == "missing_header":
                problems.append(f"{relpath}: no '## Page {record['page']}' header found")

    lines = [
        "# Phase 3 apply report" + ("" if apply else " (DRY RUN -- pass --apply to write)"),
        "",
        f"- {len(adjudication_records)} adjudicated pages total",
        f"- {applied_pages} pages to apply -> "
        + ", ".join(f"{n} {status}" for status, n in sorted(file_results.items())),
        f"- {sum(skip_reasons.values())} pages skipped:",
    ]
    for reason, n in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"    - {n}: {reason}")
    if problems:
        lines.append(f"- {len(problems)} problem(s):")
        lines.extend(f"    - {p}" for p in problems)

    report = "\n".join(lines) + "\n"
    print(report)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
