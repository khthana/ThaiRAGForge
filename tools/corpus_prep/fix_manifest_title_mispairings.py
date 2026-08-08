"""Repair the 4 `meeting_manifest.json` titles that name the wrong document.

Found by `audit_title_body_agreement.py` (2026-07-30): 7 files whose manifest
title disagrees with their own page-1 `เรื่อง` subject line, all 7 reviewed and
genuine. They split three ways, and only one group is fixable here:

  * **4 mispairings** -- the right document is present in the meeting and the
    manifest points the wrong title at it. Metadata-only, fixed by this script.
  * **2 never-fetched documents** (`2564/ครั้งที่ 12`) -- the CHECO shape. There
    is no correct title to move; the document itself is absent. Needs a
    re-download + re-OCR, tracked separately.
  * **1 generic-title case** (`2568/ครั้งที่ 6`, `ฝ่ายเลขานุการแจ้งให้ที่ประชุมทราบ`)
    -- a container agenda item whose body legitimately shows one sub-item.
    Left alone; see `KEPT_AS_IS`.

Sibling of `fix_manifest_title_collisions.py` and deliberately identical in
shape: a table of `Fix`es, each verified against the title currently on disk
before it is written, so a re-run is a no-op and a manifest that has drifted
fails loudly instead of being silently overwritten.

**Why the title moves and the file does not.** In two of the four the manifest
title was copied from a neighbouring entry while the *filename* already agrees
with the body (coverage 0.90 and 1.00), so the repair is to put the title back
in agreement with the file it names. In the other two -- a mutual A<->B swap in
`2565/ครั้งที่ 8` -- title, file and url are each internally consistent but the
*downloaded content* landed in the other entry's file, so swapping the two
titles is what re-aligns title, body **and** url at once. Swapping `url`
instead would fix the source and leave the text wrong; swapping `file` instead
would fix the text and leave the url wrong. Only the title swap fixes both.

**Replacement titles are the filename stem, whitespace-collapsed** -- never a
title reconstructed from the body. Two reasons: the stem is already what a
correct entry in this corpus holds (the sibling `2565/4` item 44 and `2568/2`
item 26 both carry their own stem verbatim, truncation included), and inventing
a completed title would be encoding metadata that exists in no source, which is
what ADR-0003 exists to prevent. The stems here end mid-word (`... เป`,
`... บูรณ`) because the download truncates long filenames; that is cosmetic and
`audit_title_body_agreement.py` scores it 1.00 by design -- its coverage metric
is asymmetric precisely so a truncated title is not punished for the words it
is missing.

`resolution_id` is `<year>/<session>/<title>` (ADR-0002), so all four ids move.
None of the four is cited by the 106-query gold set -- verified directly, and
the check is not vacuous: 57 gold ids come from these same three meetings. Run
`relabel_index_resolution_ids.py`-style relabelling afterwards; chunk *text* is
untouched, so this is a rename, not a rebuild.

Note the `2565/8` pair is a genuine **swap**: both ids survive, they change
which file they name. Anything applying this downstream must do it atomically,
not as two sequential renames.

Run:
    python tools/corpus_prep/fix_manifest_title_mispairings.py           # dry run
    python tools/corpus_prep/fix_manifest_title_mispairings.py --apply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS = Path("academic_resolutions")

_ONLINE = "เรื่อง ขอความเห็นชอบการเรียนการสอนออนไลน์สำหรับรายวิชาภายในสถาบัน ภาคการศึกษาที่ 1/2565 (เพิ่มเติม)"
_TRANSFER = "เรื่อง ขอความเห็นชอบรายวิชาที่ขอเทียบโอนหลักสูตรฝึกอบรมเพื่อสะสมหน่วยกิต ของ คณะวิทยาศาสตร์"


@dataclass(frozen=True)
class Fix:
    folder: str
    file: str
    expected: str
    title: str
    why: str


FIXES = [
    Fix(
        folder="2565/ครั้งที่ 4",
        file="เรื่อง ขออนุมัติให้นักศึกษาถอนรายวิชาและลงทะเบียนเรียนน้อยกว่า 9 หน่วยกิต ในภาคการศึกษาที่ 2_2564 เป.md",
        expected="เรื่อง ขอเสนอแนวทางการจัดสอบปลายภาค ปีการศึกษา 2564 แบบ on-site ตามมาตรการควบคุมการแพร่ระบาดของโรค COVID-19",
        title="เรื่อง ขออนุมัติให้นักศึกษาถอนรายวิชาและลงทะเบียนเรียนน้อยกว่า 9 หน่วยกิต ในภาคการศึกษาที่ 2_2564 เป",
        why="the neighbouring item 44's title was copied onto this entry; that "
        "item exists separately with its own url and its own on-site-exams "
        "document. This file's name and page-1 heading agree on ถอนรายวิชาฯ "
        "(coverage 0.90 vs 0.07 for the title it carried)",
    ),
    Fix(
        folder="2565/ครั้งที่ 8",
        file="เรื่อง  ขอความเห็นชอบรายวิชาที่ขอเทียบโอนหลักสูตรฝึกอบรมเพื่อสะสมหน่วยกิต ของ คณะวิทยาศาสตร์.md",
        expected=_TRANSFER,
        title=_ONLINE,
        why="one half of a mutual A<->B swap: this file holds the ONLINE-teaching "
        "document (coverage 1.00 against that title, 0.33 against its own)",
    ),
    Fix(
        folder="2565/ครั้งที่ 8",
        file="เรื่อง ขอความเห็นชอบการเรียนการสอนออนไลน์สำหรับรายวิชาภายในสถาบัน ภาคการศึกษาที่ 1_2565 (เพิ่มเติม).md",
        expected=_ONLINE,
        title=_TRANSFER,
        why="the other half of that swap: this file holds the credit-transfer "
        "document (coverage 0.91 against that title, 0.18 against its own)",
    ),
    Fix(
        folder="2568/ครั้งที่ 2",
        file="เรื่อง  ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) คณะเทคโนโลยีนวัตกรรมและบูรณ.md",
        expected="เรื่อง ขออนุมัติให้นักศึกษาลงทะเบียนเรียนต่ำกว่า 9 หน่วยกิต ในภาคการศึกษาที่ 2/2567",
        title="เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) คณะเทคโนโลยีนวัตกรรมและบูรณ",
        why="item 26's title was copied onto this entry; item 26 exists "
        "separately with its own url and the actual ลงทะเบียนต่ำกว่า 9 หน่วยกิต "
        "document. This file's name matches its body exactly (coverage 1.00 "
        "vs 0.00 for the title it carried)",
    ),
]

# Flagged by the same audit, deliberately not fixed here.
KEPT_AS_IS = [
    "2564/ครั้งที่ 12 (x2): the titled documents were never fetched -- the CHECO "
    "shape. No correct title exists to move; needs re-download + re-OCR.",
    "2568/ครั้งที่ 6: 'ฝ่ายเลขานุการแจ้งให้ที่ประชุมทราบ' is a generic container "
    "agenda item, so its body legitimately shows one sub-item. Not a defect.",
]


def collapse(s: str) -> str:
    """Whitespace-collapsed, matching how a correct entry's title relates to its
    filename. These names mix NBSP with ordinary spaces, so an exact copy of a
    stem is unreproducible by hand."""
    return re.sub(r"\s+", " ", s).strip()


def find_entry(entries: list[dict], want: str) -> dict:
    """The manifest entry for `want`, exact match first.

    Collapsing whitespace to find an entry is necessary (see `collapse`) but on
    its own it is unsafe: this corpus really does contain pairs of files whose
    names differ *only* by a double space -- `2569/ครั้งที่ 2` has two such pairs
    -- and a collapsed lookup silently returns whichever it meets first, which
    would then have another document's title written onto it. That is the exact
    defect this script exists to repair, so it must not be able to cause one.
    An ambiguous collapsed match is therefore an error, not a coin flip."""
    for e in entries:
        if e.get("file") == want:
            return e
    hits = [e for e in entries if collapse(e.get("file", "")) == collapse(want)]
    if len(hits) > 1:
        raise SystemExit(
            f"ambiguous: {len(hits)} manifest entries collapse to {want!r}:\n"
            + "\n".join(f"  {e['file']!r}" for e in hits)
        )
    if not hits:
        raise SystemExit(f"not in manifest: {want!r}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the changes")
    args = ap.parse_args()

    by_folder: dict[Path, list[Fix]] = {}
    for fix in FIXES:
        by_folder.setdefault(CORPUS / fix.folder, []).append(fix)

    changed = skipped = 0
    for folder, fixes in by_folder.items():
        manifest_path = folder / "meeting_manifest.json"
        entries = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        dirty = False
        for fix in fixes:
            entry = find_entry([e for e in entries if isinstance(e, dict)], fix.file)
            if not (folder / entry["file"]).exists():
                raise SystemExit(f"manifest names a missing file: {folder / entry['file']}")
            current = entry.get("title")
            if current == fix.title:
                print(f"already fixed: {fix.folder}/{fix.file[:55]}...")
                skipped += 1
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
            # trailing newline) reformats every entry and buries a one-line
            # title change in a whole-file diff
            manifest_path.write_text(
                json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8"
            )

    print(f"\n{changed} title(s) to change, {skipped} already correct")
    for note in KEPT_AS_IS:
        print(f"  kept as-is: {note}")
    if changed and not args.apply:
        print("\ndry run -- re-run with --apply to write")
    elif changed:
        print("\nwritten. resolution_ids have moved: relabel built indices and "
              "persisted results next, then re-run audit_resolution_ids.py and "
              "audit_title_body_agreement.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
