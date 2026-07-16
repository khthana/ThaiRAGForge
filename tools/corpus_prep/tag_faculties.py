"""Batch-run FacultyLoader's matcher over the whole corpus and write a
frequency report, so coverage (how many documents got tagged, how evenly
across the 20 confirmed faculties, how many got nothing) can be eyeballed
before this becomes a retrieval-time filter.

Reuses rag_lab.loaders.faculty_loader.match_faculties rather than
reimplementing matching here -- that logic already has unit tests
(tests/test_faculty_loader.py); this script is only the corpus-wide
walk/aggregate/report glue around it.

Read-only: never writes into the corpus itself. Output goes to
academic_resolutions/entity_tags/ (gitignored, same as the rest of the
corpus and the llm_ocr_scan/ output).

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/tag_faculties.py
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
from rag_lab.loaders.common import strip_document_header, strip_mapping_tables  # noqa: E402
from rag_lab.loaders.faculty_loader import match_faculties  # noqa: E402

_SAMPLE_UNMATCHED = 20  # how many zero-match files to list for a quick spot-check


def tag_corpus(corpus_root: Path) -> dict[str, list[str]]:
    """relpath -> sorted faculty list for every live (non-`.dup`) corpus file."""
    tags: dict[str, list[str]] = {}
    for f in sorted(corpus_root.rglob("*.md")):
        if f.name.endswith(".dup"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")
        text = strip_mapping_tables(strip_document_header(text))
        tags[str(f.relative_to(corpus_root).as_posix())] = match_faculties(text)
    return tags


def summarize(tags: dict[str, list[str]]) -> dict:
    """Pure aggregation over a precomputed relpath->faculties mapping --
    testable without touching the corpus."""
    counts: Counter = Counter()
    unmatched: list[str] = []
    for relpath, faculties in tags.items():
        if not faculties:
            unmatched.append(relpath)
        counts.update(faculties)
    return {"total": len(tags), "counts": counts, "unmatched": sorted(unmatched)}


def render_report(summary: dict) -> str:
    total = summary["total"]
    counts: Counter = summary["counts"]
    unmatched = summary["unmatched"]
    matched = total - len(unmatched)
    match_rate = f"{matched / total:.0%}" if total else "n/a"
    lines = [
        "# Faculty tagging coverage report",
        "",
        f"- Total live files: {total}",
        f"- Files matched to >=1 faculty: {matched} ({match_rate})",
        f"- Files matched to 0 faculties: {len(unmatched)}",
        f"- Distinct faculties matched: {len(counts)} / 20",
        "",
        "## Frequency (most to least mentioned)",
        "",
    ]
    for name, n in counts.most_common():
        lines.append(f"{n}\t{name}")
    lines += ["", f"## Sample of unmatched files (first {_SAMPLE_UNMATCHED})", ""]
    for relpath in unmatched[:_SAMPLE_UNMATCHED]:
        lines.append(f"- {relpath}")
    return "\n".join(lines) + "\n"


def main() -> None:
    tags = tag_corpus(CORPUS_ROOT)
    summary = summarize(tags)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "faculties_by_file.json").write_text(
        json.dumps(tags, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT_DIR / "faculties_report.md").write_text(render_report(summary), encoding="utf-8")

    print(f"[INFO] tagged {summary['total']} files")
    print(f"[INFO] {len(summary['unmatched'])} files matched 0 faculties")
    print(f"[INFO] {len(summary['counts'])}/20 distinct faculties matched")
    print(f"[INFO] wrote {OUT_DIR / 'faculties_by_file.json'}")
    print(f"[INFO] wrote {OUT_DIR / 'faculties_report.md'}")


if __name__ == "__main__":
    main()
