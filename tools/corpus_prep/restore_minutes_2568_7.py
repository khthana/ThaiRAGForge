"""Restore the รับรองรายงานการประชุม document that 2568/ครั้งที่ 7 lost to a download bug.

One bug produced two defects in that meeting. The CHECO agenda item was fetched
from the *minutes* item's Drive id, so two byte-identical PDFs (same SHA-256)
landed under two names; the OCR of one was then archived as a duplicate -- which it
literally was -- leaving the CHECO-titled file holding minutes text and the minutes
item with no file of its own. Re-OCRing the correct CHECO PDF fixed the first half
and, by removing the minutes text from the corpus, exposed the second.

That absence is a real gap, not a policy: 11 of 2568's 12 meetings carry a
รับรองรายงานการประชุม item (57 across the corpus), and ครั้งที่ 7 -- exactly the
meeting hit by the bug -- is the only one that does not.

This script restores it from the archive and records it in both metadata sources:

  * `academic_resolutions/.../เรื่อง รับรองรายงานการประชุม.md` (+ its `_LINK.txt`),
    un-suffixed from the `.dup` archive copies. The OCR text is untouched -- it is
    the same text that was being served under the CHECO title until today, so it
    needs no re-OCR.
  * a `meeting_manifest.json` entry (ADR-0003 makes the manifest the source of
    truth for titles/URLs), matching how the other 11 meetings spell this item.
  * a `master_list.csv` row, since the reconciled inventory is what "2853/2853
    ครบ" was counted from -- restoring the file without it would just move the
    discrepancy.

Writing style matters here: manifests are CRLF, `indent=1`, no trailing newline
(the same `json.dumps` call as rebuild_manifests.py:286), so an ordinary dump
would reformat all 61 entries into an unreviewable diff.

**Consequence, deliberately accepted**: the corpus goes 2,853 -> 2,854 files, so
every built index is one document short until the next rebuild (audit I3b coverage
2853/2854, I6). No cited number depends on it -- no gold query in either set
references this document -- and the same rebuild is already owed for the CHECO text
fix, so both changes ride one rebuild rather than forcing one now.

Idempotent: re-running after a successful run finds the file already live and the
entries already present, and changes nothing.

Run:
    python tools/corpus_prep/restore_minutes_2568_7.py            # dry run
    python tools/corpus_prep/restore_minutes_2568_7.py --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

CORPUS = Path("academic_resolutions")
FOLDER = CORPUS / "2568" / "ครั้งที่ 7"
ARCHIVE = Path(
    r"D:/academic_resolutions (ข้อมูลดิบ + OCR)/_superseded_from_repo"
) / "academic_resolutions" / "2568" / "ครั้งที่ 7"

STEM = "เรื่อง รับรองรายงานการประชุม"
TITLE = STEM
URL = "https://drive.google.com/file/d/1MtrNEXaPw5PG3a_dOXbTwHkbUDy7eOEy/view"
NOTE = "กู้คืนจาก archive 2026-07-30: ดาวน์โหลดเดิมหยิบ Drive id ผิดให้วาระ CHECO ทำให้ไฟล์นี้ถูกมองว่าซ้ำและถูกเก็บเข้า archive"


def planned_moves() -> list[tuple[Path, Path]]:
    return [
        (ARCHIVE / f"{STEM}.md.dup", FOLDER / f"{STEM}.md"),
        (ARCHIVE / f"{STEM}_LINK.txt.dup", FOLDER / f"{STEM}_LINK.txt"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    moves = planned_moves()
    for src, dst in moves:
        state = "already live" if dst.exists() else ("ready" if src.exists() else "MISSING SOURCE")
        print(f"  [{state}] {dst.name}")
        if state == "MISSING SOURCE":
            raise SystemExit(f"cannot restore: {src} not found")

    mpath = FOLDER / "meeting_manifest.json"
    entries = json.loads(mpath.read_bytes().decode("utf-8-sig"))
    have_manifest = any(e.get("file") == f"{STEM}.md" for e in entries)
    print(f"  manifest: {len(entries)} entries, this item present: {have_manifest}")

    master = CORPUS / "master_list.csv"
    with master.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    have_row = any(
        r["ปี"] == "2568" and r["การประชุม"] == "7" and STEM in r["ชื่อเรื่อง"] for r in rows
    )
    print(f"  master_list: {len(rows)} rows, this item present: {have_row}")

    if not args.apply:
        print("\ndry run -- pass --apply to restore")
        return 0

    for src, dst in moves:
        if not dst.exists():
            shutil.move(str(src), str(dst))
            print(f"  restored {dst.name}")

    if not have_manifest:
        entries.append(
            {"file": f"{STEM}.md", "title": TITLE, "url": URL, "title_source": "manifest"}
        )
        # byte-for-byte the writer rebuild_manifests.py uses, so the diff is the
        # appended entry and nothing else
        mpath.write_text(
            json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8", newline="\r\n"
        )
        print(f"  manifest entry appended ({len(entries)} entries)")

    if not have_row:
        rows.append({
            "ปี": "2568", "การประชุม": "7", "ลำดับใน docx": "",
            "ชื่อเรื่อง": TITLE, "URL": URL,
            "ไฟล์": f"2568/ครั้งที่ 7/{STEM}.md",
            "สถานะ": "ครบ", "หมายเหตุ": NOTE,
        })
        with master.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  master_list row appended ({len(rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
