"""Batch-run CourseLoader's matcher over the whole corpus and write a
frequency + spot-check report, so coverage and (since there is no
pre-built ground-truth dictionary to check false positives against, unlike
tag_programs.py) plausibility can be eyeballed before this becomes a
retrieval-time filter. Same overall shape as tag_programs.py.

Reuses rag_lab.loaders.course_loader.match_courses rather than
reimplementing matching here -- that logic already has unit tests
(tests/test_course_loader.py); this script is only the corpus-wide
walk/aggregate/report glue around it.

Unlike programs.json, course codes have no pre-built canonical list --
the matched 8-digit strings ARE the canonical values. That means the usual
"matched vs. dictionary size" coverage check doesn't apply here; instead
this report samples MATCHED codes with surrounding context (not just
unmatched files) so a human can spot-check for false positives (e.g. a
stray 8-digit number this pattern let through).

Read-only: never writes into the corpus itself. Output goes to
academic_resolutions/entity_tags/ (gitignored, same as tag_programs.py's
output and the rest of the corpus).

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/tag_courses.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
OUT_DIR = CORPUS_ROOT / "entity_tags"

# tools/corpus_prep is run directly (not via pytest's configured pythonpath),
# so src/ needs to be on sys.path explicitly to reuse the already-tested
# matcher instead of duplicating it here.
sys.path.insert(0, str(REPO / "src"))
from rag_lab.loaders.common import (  # noqa: E402
    iter_corpus_files,
    strip_document_header,
    strip_mapping_tables,
)
from rag_lab.loaders.course_loader import match_courses  # noqa: E402

_SAMPLE_UNMATCHED = 20  # how many zero-match files to list for a quick spot-check
_SAMPLE_MATCHED = 30  # how many matched (code, context) pairs to list for a false-positive check
_CONTEXT_CHARS = 40  # chars of surrounding text to show per matched sample


def tag_corpus(corpus_root: Path) -> dict[str, list[str]]:
    """relpath -> sorted course-code list for every live (non-`.dup`)
    corpus file."""
    tags: dict[str, list[str]] = {}
    for f in iter_corpus_files(corpus_root):
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")
        text = strip_mapping_tables(strip_document_header(text))
        tags[str(f.relative_to(corpus_root).as_posix())] = match_courses(text)
    return tags


def sample_matches(corpus_root: Path, limit: int) -> list[tuple[str, str, str]]:
    """(relpath, code, context) for the first `limit` matches encountered,
    walked in the same deterministic order as tag_corpus -- a human
    false-positive spot-check, since there's no dictionary to validate
    against automatically."""
    samples: list[tuple[str, str, str]] = []
    for f in iter_corpus_files(corpus_root):
        if len(samples) >= limit:
            break
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")
        text = strip_mapping_tables(strip_document_header(text))
        for code in match_courses(text):
            if len(samples) >= limit:
                break
            pos = text.find(code)
            if pos == -1:
                continue  # digits were Thai-numeral in source; skip context lookup
            start = max(0, pos - _CONTEXT_CHARS)
            end = min(len(text), pos + len(code) + _CONTEXT_CHARS)
            context = text[start:end].replace("\n", " ")
            samples.append((str(f.relative_to(corpus_root).as_posix()), code, context))
    return samples


def summarize(tags: dict[str, list[str]]) -> dict:
    """Pure aggregation over a precomputed relpath->courses mapping --
    testable without touching the corpus."""
    counts: Counter = Counter()
    unmatched: list[str] = []
    for relpath, courses in tags.items():
        if not courses:
            unmatched.append(relpath)
        counts.update(courses)
    return {"total": len(tags), "counts": counts, "unmatched": sorted(unmatched)}


def render_report(summary: dict, matched_samples: list[tuple[str, str, str]]) -> str:
    total = summary["total"]
    counts: Counter = summary["counts"]
    unmatched = summary["unmatched"]
    matched = total - len(unmatched)
    match_rate = f"{matched / total:.0%}" if total else "n/a"
    lines = [
        "# Course tagging coverage report",
        "",
        f"- Total live files: {total}",
        f"- Files matched to >=1 course: {matched} ({match_rate})",
        f"- Files matched to 0 courses: {len(unmatched)}",
        f"- Distinct course codes matched: {len(counts)}",
        "",
        "No pre-built course dictionary exists (unlike programs.json) -- the",
        "matched samples below are a false-positive spot-check, not a",
        "coverage-vs-known-list check.",
        "",
        "## Frequency (most to least mentioned)",
        "",
    ]
    for name, n in counts.most_common():
        lines.append(f"{n}\t{name}")
    lines += ["", f"## Sample of matched codes with context (first {_SAMPLE_MATCHED})", ""]
    for relpath, code, context in matched_samples:
        lines.append(f"- `{code}` in {relpath}: ...{context}...")
    lines += ["", f"## Sample of unmatched files (first {_SAMPLE_UNMATCHED})", ""]
    for relpath in unmatched[:_SAMPLE_UNMATCHED]:
        lines.append(f"- {relpath}")
    return "\n".join(lines) + "\n"


def main() -> None:
    tags = tag_corpus(CORPUS_ROOT)
    summary = summarize(tags)
    matched_samples = sample_matches(CORPUS_ROOT, _SAMPLE_MATCHED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "courses_by_file.json").write_text(
        json.dumps(tags, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT_DIR / "courses_report.md").write_text(render_report(summary, matched_samples), encoding="utf-8")

    print(f"[INFO] tagged {summary['total']} files")
    print(f"[INFO] {len(summary['unmatched'])} files matched 0 courses")
    print(f"[INFO] {len(summary['counts'])} distinct course codes matched")
    print(f"[INFO] wrote {OUT_DIR / 'courses_by_file.json'}")
    print(f"[INFO] wrote {OUT_DIR / 'courses_report.md'}")


if __name__ == "__main__":
    main()
