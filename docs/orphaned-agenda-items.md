# Agenda items with no document of their own

**2026-08-09.** Task #21 was recorded as *"re-download + re-OCR the 3 never-fetched
documents (CHECO `2568/ครั้งที่ 7` + 2× `2564/ครั้งที่ 12`)"*. Every part of that
sentence turned out to be wrong: CHECO was already repaired, the class is larger than
3, and only **1 of the 10** flagged items is repairable by re-download. That one is
now repaired. This is the record of how each verdict was reached, because the
verdicts differ per item and the evidence is what separates them.

Scanner: `tools/corpus_prep/scan_duplicate_bodies.py` (report-only, writes nothing).
Repair tool: `tools/corpus_prep/refetch_mispaired_document.py` (dry-run by default).

## The defect

Two agenda items in one meeting cannot both be described by one document. When two
files in a meeting folder carry the same body, at most one of the titles is right and
the other item has no document — its title is a claim the corpus cannot support.

`audit_title_body_agreement.py` is structurally unable to see this. It scores each
title against **its own** body, so two items sharing a body both score against that
one subject line; where the titles differ only in a faculty name, the shared
boilerplate carries both well above its 0.34 threshold (`2564/ครั้งที่ 5` items 20
and 21 score 0.692 and 0.583 against a subject line naming a third faculty
entirely). Comparing items **to each other** is what makes the defect visible, and it
is why 3 of the 5 exact-duplicate pairs below had never been seen by any check.

## Two signals, because the obvious one undercounts

**A — identical OCR text.** Exact, and proves the two files came from one PDF. But it
only fires when the *same* PDF was fetched twice. Where the source holds two separate
scans or exports of one document, the OCR differs in a few characters and the hash
misses it: `2564/ครั้งที่ 5` items 18–21 are four files, four distinct hashes, one
subject line.

**B — a shared page-1 `เรื่อง` subject line within a meeting.** Segmentation-independent,
no threshold, no tokenizer. Catches the separate-export case A cannot.

**The subject is compared at full length, and that was calibrated rather than
guessed.** A 60-character prefix reports 229 groups / **1,255 orphans — 44% of the
corpus** — because curriculum items share a long boilerplate opening and the faculty
that distinguishes them falls past the cut. The count collapses 229 → 28 between 60
and 80 and is then flat from 100 to full length (13 → 11 groups), with both known
separate-export cases still grouped at every window. Full length is the stable end of
that plateau and needs no constant.

Filename decorations are stripped before asking whether two files are different
items: `__N` (the `split_curriculum_bundles.py` piece index, ADR-0004) and ` (N)`
(the browser-style suffix the download stage adds when it fetches one document
twice). Both may legitimately repeat. **Adding the ` (N)` half moved the headline
from 11 groups / 16 items to 9 / 11** — `2564/ครั้งที่ 1` (5 members) and
`2568/ครั้งที่ 7` (2) are one item under decorated names, not missing documents.

## What the scan finds

```
2,876 files scanned  (54 with no locatable subject line)

A. identical OCR text -- 11 groups        B. one subject line, several agenda items
     same-meeting       5                      8 groups, 10 items flagged
     cross-meeting      0                      (9 genuine; see 2565/7 below)
     same-item-variant  6
```

## Per-item verdicts

The deciding question is not what the *corpus* holds — that only shows a duplicate —
but what the **recorded Drive id serves now**. Page 1 of every member of every group
was fetched and rasterised (`pdftoppm`) and compared. All 21 ids are distinct and
correctly recorded; the manifest, `_LINK.txt` and `master_list.csv` agree throughout,
so there is no alternative id to try anywhere.

| meeting | items | what the ids serve | verdict |
|---|---|---|---|
| `2564/ครั้งที่ 12` | 2 | one document, byte-identical from 3 endpoints | not repairable |
| `2564/ครั้งที่ 12` | 2 | one document, byte-identical | not repairable |
| `2564/ครั้งที่ 3` | 2 | one document, two exports | not repairable |
| `2564/ครั้งที่ 5` | 4 | one document under **four** ids | not repairable |
| `2565/ครั้งที่ 2` | 2 | one document, two exports | not repairable |
| `2565/ครั้งที่ 6` | 2 | one document, two exports | not repairable |
| `2565/ครั้งที่ 7` | 2 | **two real documents** | **signal-B false positive** |
| `2566/ครั้งที่ 3` | 2 | **two real documents**, file held the wrong one | **repaired 2026-08-09** |
| `2568/ครั้งที่ 11` | 2 | second id is **404** | not repairable — dead at source |

**9 agenda items have no document of their own**, and 8 of those are unfixable.

### Not repairable: the source lists one document under several items

For the byte-identical pairs, three download endpoints (`docs.google.com/uc`,
`drive.google.com/uc`, `drive.usercontent.google.com/download`) serve the same bytes.
For the byte-*differing* ones the PDFs are the same document exported twice, which is
a distinction the byte hash cannot make and the thumbnail alone should not be trusted
for:

* `2564/ครั้งที่ 3` — 2,274,112 vs 2,274,111 b, both 4 pages, CreationDate
  `Wed Mar 31 16:55:27 2021` to the second, producer `DocuCentre-V 4070`, identical
  extracted text.
* `2565/ครั้งที่ 2` — 349,940 vs 209,820 b, both 2 pages, CreationDate
  `Fri Feb 25 07:59:20 2022`, producer `Adobe PDF Library 11.0`, identical 2,833 chars.
* `2565/ครั้งที่ 6` — 446,361 vs 446,183 b, both 2 pages, same CreationDate to the
  second, producer `Microsoft: Print To PDF`; page-1 renders read identically by eye.

This is a **source** defect. No fetch can produce the missing document, and the
manifest title for the orphaned item is simply unsupported by anything that exists.

### Repaired: `2566/ครั้งที่ 3`

The one item matching the CHECO mechanism — the *download stage* attached the wrong
blob while the manifest kept the right id. Both ids serve genuinely different letters
(คณะวิทยาศาสตร์, 4 ราย vs วิทยาลัยเทคโนโลยีและนวัตกรรมวัสดุ, 1 ราย), yet both corpus
files carried the คณะวิทยาศาสตร์ body. The tell is in the file itself: its
`# Document:` header names `กลุ่ม 14 ... ว.เทคโนโลยีและนวัตกรรมวัสดุ.pdf` while its
body is the other faculty's letter.

Re-downloaded from the id the manifest already held
(`1jrRMmJ6O7K-2SAbXl_sbgrMfAugqgNW0`) and re-OCR'd with the corpus's own pipeline
(`ocr_pdf_to_md.process_pdf`, `scb10x/typhoon-ocr1.5-3b`). 2,182 → 1,621 chars; the
new text names ดร.ศศิธร เอื้อวิริยะวิทย์ and carries the appointment table, and no
longer mentions คณะวิทยาศาสตร์. Original kept at `<name>.md.pre_refetch.bak`.

**This is a text change, not a relabel** — unlike the 2026-08-08 title repair, chunk
text moves, so every built index holds a stale vector for this file. It is free
anyway: **0 gold entries in either gold set cite any resolution from
`2566/ครั้งที่ 3`**, so no metric can move, and the `resolution_id` is unchanged
(titles were not touched). No rebuild is owed for eval purposes.

### Not repairable: dead at source

`2568/ครั้งที่ 11`'s `เรื่อง ขอความเห็นชอบเปลี่ยนแปลงแก้ไขรายชื่ออาจารย์พิเศษ…` holds
the คณะบริหารธุรกิจ document (its header names
`11-2568 4.7-9_อาจารย์พิเศษสอนเกินกว่าร้อยละ 50_คณะบริหารธุรกิจ.pdf`), and its own id
`1S594k62ICODQK5Ypp58vj5EJBNdi9mTP` returns **HTTP 404** from every endpoint. The
document existed once; it does not now.

### False positive: `2565/ครั้งที่ 7`

Both ids serve real, different documents (2 pages vs 1 page, different renders, both
matching their own corpus file). They collide on signal B because the resolution's
page-1 heading is a **combined** one — *"ขอพระราชทานทูลเกล้าฯ ถวายปริญญา … **และ**
ขอความเห็นชอบ (ร่าง) ประกาศราชสดุดี…"* — printed on both. Not a defect; both titles
are correct. Signal B's expected false-positive shape, and the reason the table above
is per-item rather than a count.

## Two traps worth keeping

**A 404 is not a different document.** Drive answers a missing id with a 1,652-byte
HTML error page. An early version of the probe compared page counts without checking
the `%PDF` magic and reported that as "two distinct documents" — the exact opposite of
the truth. `refetch_mispaired_document.py` checks the magic, not the status code.

**Pixel-hash equality proves sameness; inequality proves nothing.** Two exports of one
Word print differ by a handful of bytes and render to visibly identical but
non-identical PNGs. The automated comparison called `2565/ครั้งที่ 6` "distinct
documents" on that basis; reading the two renders showed one document. Every
"distinct" verdict in the table was confirmed by eye, and only the "identical"
verdicts rest on the hash.

## Status

* Repairable and repaired: **1** (`2566/ครั้งที่ 3`).
* Unrepairable, recorded: **8** — 7 where the source itself lists one document under
  several items, 1 where the id is dead.
* The `2568/ครั้งที่ 7` CHECO defect named in the original ticket was **already
  repaired** before this task began (`restore_minutes_2568_7.py`); the CHECO-titled
  file holds genuine CHECO content and is distinct from the restored minutes file.
  The "3 URLs owed" claim in `CLAUDE.md` and `docs/title-body-agreement.md` was stale
  and is corrected.
* `audit_resolution_ids.py` after the repair: **OK, 2 accepted clashes, exit 0**.

Re-run `tools/corpus_prep/scan_duplicate_bodies.py` after any corpus or manifest
change.
