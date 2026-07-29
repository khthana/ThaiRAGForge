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


def replace_page_text(text: str, page_num: int, new_body: str) -> str | None:
    """Substitute the body of a '## Page N' section with `new_body`. A file
    can carry more than one physical '## Page N' header for the same N -- a
    confirmed OCR-ingestion defect where one physical page's content got
    split across two consecutive headers instead of staying under one
    (`llm_ocr_scan.split_pages`'s "N.1"/"N.2" sub-chunking is a different,
    in-memory-only LLM-budget device, never a real file header). `new_body`
    is always a fresh re-OCR of the *whole* physical page, and Phase 2's own
    adjudication text (`reocr_adjudicate.load_full_page_text`) already
    concatenated every same-numbered chunk into one blob before judging it --
    so when N repeats, every matching header from the first through the last
    is collapsed into a single one, replaced by `new_body` whole; never
    picked apart into "correct" vs. "boilerplate" halves, since both halves
    are real content that belongs together. None if the header is missing
    entirely, or if the repeats aren't contiguous (some other issue is going
    on -- don't guess)."""
    headers = list(PAGE_HEADER.finditer(text))
    matches = [i for i, m in enumerate(headers) if int(m.group(1)) == page_num]
    if not matches:
        return None
    if matches != list(range(matches[0], matches[0] + len(matches))):
        return None
    start = headers[matches[0]].end()
    last = matches[-1]
    end = headers[last + 1].start() if last + 1 < len(headers) else len(text)
    # `PAGE_HEADER`'s `\s*$` always backtracks to consume exactly the header
    # line's own trailing newline (never further -- `$` needs to land right
    # before the *next* "\n" to match), so text[:start] already ends in one
    # "\n"; only one more is needed to open a blank line before the body.
    return text[:start] + "\n" + new_body.strip() + "\n\n" + text[end:]


@dataclass(frozen=True)
class FileWriteResult:
    status: str  # "written" | "unchanged" | "missing_header"


def apply_to_file(
    path: Path, page_num: int, new_body: str, apply: bool,
) -> FileWriteResult:
    """Replace one page's text in one corpus file. `apply=False` computes and
    reports the outcome without touching disk (dry run). Backing up happens
    only on an actual change, and only once per file (mirrors
    `excise_ocr_loops.py`'s `if not backup.exists()`)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")

    replaced = replace_page_text(text, page_num, new_body)
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
            result = apply_to_file(CORPUS_ROOT / relpath, record["page"], new_text, apply)
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
