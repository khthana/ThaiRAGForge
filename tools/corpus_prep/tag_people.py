"""Batch-run PersonLoader's matcher over the whole corpus and write a
frequency report, so coverage and precision can be eyeballed before this
becomes a retrieval-time filter. Same shape as tag_faculties.py /
tag_programs.py, adapted for entries keyed by full_name instead of a flat
canonical string (people don't have a closed dictionary -- see
person_loader.py's module docstring).

Reuses rag_lab.loaders.person_loader.match_people rather than
reimplementing matching here -- that logic already has unit tests
(tests/test_person_loader.py); this script is only the corpus-wide
walk/aggregate/report glue around it.

Read-only: never writes into the corpus itself. Output goes to
academic_resolutions/entity_tags/ (gitignored, same as the rest of the
corpus and the llm_ocr_scan/ output).

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/tag_people.py
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
from rag_lab.loaders.person_loader import match_people  # noqa: E402

_TOP_N = 200  # how many of the most-mentioned people to print in the report


def tag_corpus(corpus_root: Path) -> dict[str, list[dict[str, str]]]:
    """relpath -> list of {title, given_name, surname, full_name} for every
    live (non-`.dup`) corpus file."""
    tags: dict[str, list[dict[str, str]]] = {}
    for f in sorted(corpus_root.rglob("*.md")):
        if f.name.endswith(".dup"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")
        text = strip_mapping_tables(strip_document_header(text))
        tags[str(f.relative_to(corpus_root).as_posix())] = match_people(text)
    return tags


def summarize(tags: dict[str, list[dict[str, str]]]) -> dict:
    """Pure aggregation over a precomputed relpath->people mapping --
    testable without touching the corpus."""
    counts: Counter = Counter()
    unmatched: list[str] = []
    for relpath, people in tags.items():
        if not people:
            unmatched.append(relpath)
        counts.update(p["full_name"] for p in people)
    return {"total": len(tags), "counts": counts, "unmatched": sorted(unmatched)}


def render_report(summary: dict) -> str:
    total = summary["total"]
    counts: Counter = summary["counts"]
    unmatched = summary["unmatched"]
    matched = total - len(unmatched)
    match_rate = f"{matched / total:.0%}" if total else "n/a"
    lines = [
        "# Person tagging coverage report",
        "",
        f"- Total live files: {total}",
        f"- Files matched to >=1 titled person: {matched} ({match_rate})",
        f"- Files matched to 0 people: {len(unmatched)}",
        f"- Distinct people matched: {len(counts)}",
        "",
        f"## Frequency (top {_TOP_N} most-mentioned)",
        "",
    ]
    for name, n in counts.most_common(_TOP_N):
        lines.append(f"{n}\t{name}")
    return "\n".join(lines) + "\n"


def main() -> None:
    tags = tag_corpus(CORPUS_ROOT)
    summary = summarize(tags)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "people_by_file.json").write_text(
        json.dumps(tags, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT_DIR / "people_report.md").write_text(render_report(summary), encoding="utf-8")

    print(f"[INFO] tagged {summary['total']} files")
    print(f"[INFO] {len(summary['unmatched'])} files matched 0 people")
    print(f"[INFO] {len(summary['counts'])} distinct people matched")
    print(f"[INFO] wrote {OUT_DIR / 'people_by_file.json'}")
    print(f"[INFO] wrote {OUT_DIR / 'people_report.md'}")


if __name__ == "__main__":
    main()
