"""Builds data/entity_dictionaries/programs.json from meeting_manifest.json
titles, not by scanning raw OCR body text.

Curriculum-bundle documents were split into one file per program during
ADR-0004, and each split file's manifest title carries a structured tail:
"<original bundle title> — <degree>[ สาขาวิชา<field>] (<qualifiers>)" -- e.g.
"... คณะวิศวกรรมศาสตร์ — หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมเคมี
(หลักสูตรนานาชาติ) (การปรับปรุงแก้ไขหลักสูตร ฉบับปี พ.ศ. ๒๕๖๐)". These
titles were hand-curated during that split (recovered from the agenda
capture, not retyped OCR), so scanning them gives clean degree/field pairs
directly -- no OCR typos, no professor-specialization false positives (a
"(สาขาวิชาวิศวกรรมโยธา)" next to a person's name), no HTML-table junk. All
three of those showed up when the same regex was tried against raw body
text instead, which is why body text is not the source here: it would need
the same hand-review/fuzzy-clustering machinery faculties.json needed, at
~10x the entry count. Body text is still useful later as a *gap-finder* --
which degree/field pairs appear in body text but never in a title -- but
that is a separate, smaller follow-up, not this script's job.

Read-only with respect to the corpus: only writes programs.json.

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/build_program_dictionary.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
OUT_PATH = REPO / "data" / "entity_dictionaries" / "programs.json"
_MANIFEST_NAME = "meeting_manifest.json"

# Degree word directly glued onto "หลักสูตร" (no space -- real degree-program
# mentions in this corpus never have one; a stray "หลักสูตร <course name>"
# with a space is a different, much messier open-ended category -- elective
# course titles -- deliberately out of scope here). \S+?บัณฑิต requires at
# least one character before "บัณฑิต" so a bare garbled "หลักสูตรบัณฑิต"
# (seen once, mid-OCR-corruption) doesn't get captured as a degree named
# "บัณฑิต". อนุปริญญา (associate degree) is a separate alternative because
# it doesn't end in "บัณฑิต" at all.
_TITLE_PROGRAM = re.compile(
    r"หลักสูตร(อนุปริญญา\S*|\S+?บัณฑิต)"
    r"(?:\s+สาขาวิชา\s*([^()\n]+?))?"
    r"(?=\s*[(\n]|$)"
)

# A field capture with no "(" ahead of it runs to end-of-string by default,
# which occasionally swallows a trailing explanatory clause the title tacks
# on after the real field name ends (enrollment counts, closure notices --
# these are titles hand-recovered from agenda text, not machine-generated,
# so they're not perfectly uniform). These discourse-marker words reliably
# start such a clause in this corpus; cut there rather than trust "(" alone.
_CLAUSE_BREAK = re.compile(
    r"\s+(?:โดย|ทั้งนี้|จำนวน|เนื่องจาก|หากไม่มี|ให้คณะ|ผ่านความเห็นชอบ|ระดับ)"
)


def extract_program_from_title(tail: str) -> tuple[str, str | None] | None:
    """(degree, field) from the part of a manifest title after the "— "
    split. Returns None if `tail` doesn't match this structure at all --
    most manifest titles don't (only curriculum-revision resolutions do)."""
    m = _TITLE_PROGRAM.search(tail)
    if not m:
        return None
    degree = m.group(1).strip()
    # one source title literally doubles the word ("หลักสูตรหลักสูตรบริหาร
    # ธุรกิจบัณฑิต...") -- a title-recovery typo, not a real degree name
    if degree.startswith("หลักสูตร"):
        degree = degree[len("หลักสูตร") :]
    field = None
    if m.group(2):
        field = m.group(2)
        clause = _CLAUSE_BREAK.search(field)
        if clause:
            field = field[: clause.start()]
        field = field.strip()
    return degree, field


def canonical_name(degree: str, field: str | None) -> str:
    if field:
        return f"หลักสูตร{degree} สาขาวิชา{field}"
    return f"หลักสูตร{degree}"


def collect_program_counts(corpus_root: Path) -> Counter:
    """(degree, field) -> occurrence count, across every meeting_manifest.json
    title in the corpus."""
    counts: Counter = Counter()
    for manifest_path in corpus_root.rglob(_MANIFEST_NAME):
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        for entry in entries:
            title = entry.get("title") or ""
            if "—" not in title:
                continue
            tail = title.split("—", 1)[1].strip()
            result = extract_program_from_title(tail)
            if result:
                counts[result] += 1
    return counts


def _dedup_key(degree: str, field: str | None) -> tuple[str, str | None]:
    """Merge near-duplicate entries that differ only in internal whitespace
    placement -- titles are hand-recovered text, not machine-generated, so
    the same field name occasionally gets a stray/missing space between the
    same two words across different resolutions (e.g. "...ก้าวหน้าของ
    มนุษยชาติ" vs "...ก้าวหน้าของมนุษยชาติ" for what is the same program)."""
    norm_field = re.sub(r"\s+", "", field) if field else None
    return degree, norm_field


def build_dictionary(counts: Counter) -> list[dict]:
    merged: dict[tuple[str, str | None], Counter] = {}
    for (degree, field), n in counts.items():
        key = _dedup_key(degree, field)
        merged.setdefault(key, Counter())[(degree, field)] += n

    entries = []
    for variants in merged.values():
        # the most-frequently-spelled variant becomes the canonical spelling
        (degree, field), _ = variants.most_common(1)[0]
        entries.append(
            {
                "canonical": canonical_name(degree, field),
                "prefix_type": "หลักสูตร",
                "degree": degree,
                "field": field,
                "count": sum(variants.values()),
            }
        )
    entries.sort(key=lambda e: (-e["count"], e["canonical"]))
    return entries


def main() -> None:
    counts = collect_program_counts(CORPUS_ROOT)
    dictionary = build_dictionary(counts)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[INFO] {len(dictionary)} distinct (degree, field) programs")
    print(f"[INFO] {sum(counts.values())} total title matches")
    print(f"[INFO] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
