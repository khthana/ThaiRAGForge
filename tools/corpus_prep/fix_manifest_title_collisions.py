"""Repair the `meeting_manifest.json` titles behind real `resolution_id` clashes.

Found by `audit_resolution_ids.py` (2026-07-30). Of the 6 clashing titles it
reported, 4 are data errors with a recoverable correct title and are fixed here;
the other 2 are two genuinely distinct agenda items that the agenda itself
listed under one identical title, where there is no correct different title to
recover -- those keep their shared title and are separated by
`make_resolution_id`'s `#N` rank instead of by invented metadata.

Every replacement is verified against the title currently on disk before it is
written, so a re-run after the fix is a no-op rather than a second edit, and a
manifest that has drifted since this table was written fails loudly instead of
being overwritten.

Run:
    python tools/corpus_prep/fix_manifest_title_collisions.py           # dry run
    python tools/corpus_prep/fix_manifest_title_collisions.py --apply
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

CORPUS = Path("academic_resolutions")

_ENG = "เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีกระทบกระเทือนโครงสร้าง) คณะวิศวกรรมศาสตร์"
_ENG_NOSTRUCT = (
    "เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) คณะวิศวกรรมศาสตร์"
)
_BIOMED = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมชีวการแพทย์ (หลักสูตรนานาชาติ) (พหุวิทยาการ)"


@dataclass(frozen=True)
class Fix:
    folder: str
    file: str
    expected: str
    title: str
    why: str


FIXES = [
    Fix(
        folder="2564/ครั้งที่ 11",
        file="เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีกระทบกระเทือนโครงสร้าง) วิทยาลัยนวัตกรรมการผลิตขั้นสูง.md",
        expected=_ENG,
        title="เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีกระทบกระเทือนโครงสร้าง) วิทยาลัยนวัตกรรมการผลิตขั้นสูง",
        why="another file's title was copied onto this one; its filename and its "
        "own first-page heading agree on วิทยาลัยนวัตกรรมการผลิตขั้นสูง",
    ),
    Fix(
        folder="2567/ครั้งที่ 9",
        file="เรื่อง ขออนุมัติแต่งตั้งอาจารย์บัณฑิตประจำ และอาจารย์บัณฑิตพิเศษ คณะวิทยาศาสตร์.md",
        expected="เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) คณะวิทยาศาสตร์",
        title="เรื่อง ขออนุมัติแต่งตั้งอาจารย์บัณฑิตประจำ และอาจารย์บัณฑิตพิเศษ คณะวิทยาศาสตร์",
        why="a curriculum-revision title on an อาจารย์บัณฑิตพิเศษ appointment "
        "document -- unrelated subject matter, and the one clash a gold query "
        "actually cites (2 queries in gold_query_set_73det.yaml)",
    ),
    Fix(
        folder="2567/ครั้งที่ 10",
        file="เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีกระทบกระเทือนโครงสร้าง) คณะวิศวกรรมศาสตร์ (2).md",
        expected=_ENG,
        title="เรื่อง ขอความเห็นชอบการปรับปรุงแก้ไขหลักสูตร (กรณีกระทบกระเทือนโครงสร้าง) คณะวิศวกรรมศาสตร์",
        why="the two items differ in the agenda too (ปรับปรุงแก้ไขหลักสูตร -- a "
        "follow-up to มติ ๘/๒๕๖๗ -- vs. ปรับปรุงหลักสูตร); taken from the body "
        "heading because the filename carries a ' (2)' download-dedup artifact",
    ),
    Fix(
        folder="2567/ครั้งที่ 1",
        file="เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) คณะวิศวกรรมศาสตร์__1.md",
        expected=f"{_ENG_NOSTRUCT} — หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมชีวการแพทย์ (หลักสูตรนานาชาติ)",
        title=f"{_ENG_NOSTRUCT} — {_BIOMED} (การปรับปรุงแก้ไขหลักสูตร ฉบับปี พ.ศ. ๒๕๖๓)",
        why="split-bundle pieces (ADR-0004) whose per-curriculum titles came out "
        "identical: the extractor dropped the ฉบับปี qualifier that is the only "
        "thing separating these two revisions of one curriculum. Restores the "
        "convention the same bundle's __3/__4 already follow",
    ),
    Fix(
        folder="2567/ครั้งที่ 1",
        file="เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) คณะวิศวกรรมศาสตร์__2.md",
        expected=f"{_ENG_NOSTRUCT} — หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมชีวการแพทย์ (หลักสูตรนานาชาติ)",
        title=f"{_ENG_NOSTRUCT} — {_BIOMED} (การปรับปรุงแก้ไขหลักสูตร ฉบับปี พ.ศ. ๒๕๖๔)",
        why="see __1 above (this is the ฉบับปี ๒๕๖๔ revision)",
    ),
]

# Left alone deliberately -- two distinct documents the agenda itself listed
# under one identical title, separated by make_resolution_id's `#N` rank.
KEPT_AS_IS = [
    "2567/ครั้งที่ 11: two คณะวิศวกรรมศาสตร์ curriculum items (one a re-submission "
    "of the item withdrawn at มติ ๙/๒๕๖๗, one a new ดุษฎีบัณฑิต วิศวกรรมไฟฟ้า request)",
    "2568/ครั้งที่ 4: two ขยายระยะเวลาการศึกษา items covering different student "
    "cohorts (0.29 text similarity -- not duplicate scans of one document)",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the changes")
    args = ap.parse_args()

    by_folder: dict[Path, list[Fix]] = {}
    for fix in FIXES:
        by_folder.setdefault(CORPUS / fix.folder, []).append(fix)

    changed = 0
    for folder, fixes in by_folder.items():
        manifest_path = folder / "meeting_manifest.json"
        entries = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        index = {e.get("file"): e for e in entries if isinstance(e, dict)}
        dirty = False
        for fix in fixes:
            entry = index.get(fix.file)
            if entry is None:
                raise SystemExit(f"not in manifest: {folder / fix.file}")
            current = entry.get("title")
            if current == fix.title:
                print(f"already fixed: {fix.folder}/{fix.file[:50]}...")
                continue
            if current != fix.expected:
                raise SystemExit(
                    f"unexpected title on {folder / fix.file}\n"
                    f"  expected: {fix.expected}\n  found   : {current}"
                )
            print(f"\n{fix.folder}/{fix.file}")
            print(f"  - {current}")
            print(f"  + {fix.title}")
            print(f"  why: {fix.why}")
            entry["title"] = fix.title
            dirty = True
            changed += 1
        if dirty and args.apply:
            # same call as rebuild_manifests.py:286 -- anything else (indent,
            # trailing newline) reformats all ~50 entries and buries a
            # one-line title change in a whole-file diff
            manifest_path.write_text(
                json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(f"  -> wrote {manifest_path}")

    print(f"\n{changed} title(s) {'written' if args.apply else 'to write (dry run)'}")
    print("\nleft as-is (no correct different title exists; `#N` rank separates them):")
    for note in KEPT_AS_IS:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
