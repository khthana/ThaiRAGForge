"""Audit `resolution_id` uniqueness across the corpus.

`resolution_id` is `<year>/<session>/<title>` (ADR-0002 makes it the relevance
and citation unit) and `title` comes from `meeting_manifest.json` (ADR-0003).
Nothing in that chain guarantees uniqueness: one meeting can list two agenda
items under the same title, a split bundle's pieces can be patched with the
same curriculum title, and a wrong title can be copied onto the wrong file.
Two files on one id merges unrelated documents into a single "resolution", so a
retrieval hit on either counts for both.

`make_resolution_id` now appends a `#N` rank to break folder-local title
clashes, which keeps ids unique but says nothing about *why* they clashed --
a duplicated title can equally mean "two real agenda items share a name"
(benign, disambiguation is the right answer) or "this file's manifest title
belongs to a different document" (a data error to fix at the source). This
script surfaces every clash with the evidence needed to tell those apart:
whether the manifest title matches the filename, and whether the two files
point at the same source PDF.

Run (read-only):
    PYTHONPATH=src python tools/corpus_prep/audit_resolution_ids.py
    PYTHONPATH=src python tools/corpus_prep/audit_resolution_ids.py --report out.md

Exit code is 1 when any clash is found, so it can gate a corpus change the way
ADR-0004's title audit does.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from rag_lab.loaders.common import (  # noqa: E402
    _meeting_manifest,
    iter_corpus_files,
    make_resolution_id,
    parse_path,
)

CORPUS = Path("academic_resolutions")
_HEADING = re.compile(r"มติคณะกรรมการสภาวิชาการ[^\n]*?(เรื่อง\s*.{0,160})", re.S)

# Reviewed 2026-07-30 and accepted: two distinct agenda items that one meeting's
# agenda genuinely listed under one identical title, so no correct different
# title exists to recover and the `#N` rank is the right answer. Listed here so
# the exit code stays a signal -- red means "a clash nobody has looked at",
# which a permanently-red gate could never tell you. Anything not in this list
# is unreviewed: decide it, then either fix the title at the source
# (fix_manifest_title_collisions.py) or add it here with the reason.
ACCEPTED = {
    "2567/11/เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีกระทบกระเทือนโครงสร้าง) คณะวิศวกรรมศาสตร์":
        "a re-submission of the item withdrawn at มติ ๙/๒๕๖๗ plus a new "
        "ดุษฎีบัณฑิต วิศวกรรมไฟฟ้า request, both under the agenda's one title",
    "2568/4/เรื่อง ขออนุมัติขยายระยะเวลาการศึกษาของนักศึกษาระดับปริญญาตรีและบัณฑิตศึกษา":
        "two ขยายระยะเวลาการศึกษา items covering different student cohorts "
        "(0.29 text similarity -- not duplicate scans of one document)",
}


def body_heading(path: Path) -> str:
    """The `เรื่อง ...` line from the document's own first-page heading -- the
    one title source that is independent of both the manifest and the filename,
    and so the tie-breaker when those two disagree."""
    text = path.read_text(encoding="utf-8-sig")[:4000]
    m = _HEADING.search(re.sub(r"\s+", " ", text))
    return m.group(1).strip() if m else ""


def collisions() -> dict[str, list[Path]]:
    """Undisambiguated id -> files sharing it (2+ only)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in iter_corpus_files(CORPUS):
        year, session, title = parse_path(str(p))
        groups[f"{year}/{session}/{title}"].append(p)
    return {k: v for k, v in sorted(groups.items()) if len(v) > 1}


def render(groups: dict[str, list[Path]]) -> str:
    lines = [
        "# resolution_id collision audit",
        "",
        f"Colliding titles: **{len(groups)}** "
        f"({sum(len(v) for v in groups.values())} files)",
        "",
    ]
    for key, paths in groups.items():
        status = f"accepted -- {ACCEPTED[key]}" if key in ACCEPTED else "**UNREVIEWED**"
        lines += [f"## `{key}`", "", f"Status: {status}", ""]
        urls = set()
        for p in paths:
            entry = _meeting_manifest(str(p.parent)).get(p.name) or {}
            url = entry.get("url") or ""
            urls.add(url)
            title_matches_name = (entry.get("title") or p.stem) == p.stem
            lines += [
                f"- `{p.name}` ({p.stat().st_size:,}B)",
                f"  - id after disambiguation: `{make_resolution_id(str(p), *parse_path(str(p)))}`",
                f"  - manifest title {'==' if title_matches_name else '!='} filename",
                f"  - body heading: {body_heading(p) or '(not found)'}",
                f"  - url: {url or '(none)'}",
            ]
        verdict = (
            "same source PDF -- expected for split-bundle pieces (ADR-0004), "
            "but their titles should still differ per curriculum"
            if len(urls) == 1
            else "different source PDFs -- two distinct documents sharing one title"
        )
        lines += ["", f"**{verdict}**", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, help="write the Markdown report here")
    args = ap.parse_args()

    groups = collisions()
    report = render(groups)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
        print(f"wrote {args.report}")
    print(report)

    unreviewed = [k for k in groups if k not in ACCEPTED]
    stale = [k for k in ACCEPTED if k not in groups]
    if stale:
        print(f"note: {len(stale)} accepted entr(ies) no longer clash -- drop them:")
        for k in stale:
            print(f"  {k}")
    if unreviewed:
        print(f"FAIL: {len(unreviewed)} unreviewed title clash(es)")
        return 1
    print(
        f"OK: {len(groups)} clash(es), all reviewed and accepted; "
        "every resolution_id is unique after the #N rank"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
