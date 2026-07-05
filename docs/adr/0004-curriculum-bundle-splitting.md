# Curriculum-bundle resolutions are split into per-curriculum files

---
Status: accepted
---

## Context

Some resolutions (มติ) bundle several academic curricula into one agenda item,
e.g. "ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง)
สถาบันโคเซ็นฯ" covers 3 separate curricula in one file (the largest such file
was 1 MB / 170 OCR pages), each with its own rationale and course-description
changes. Because `resolution_id` is the citation/retrieval unit (ADR-0002) and
each curriculum has its own title a user would search for individually, one
`resolution_id` per bundle is too coarse — this was the "unit to be confirmed"
ambiguity noted in `CONTEXT.md`'s original Resolution definition.

Two other bundling categories were surveyed (อาจารย์พิเศษ course-load filings,
ผลการปฏิบัติงาน/มาตรฐาน 50 filings) and deliberately **not** split — the user
judged those per-item entries too fine-grained to be worth citing separately.
Only curriculum bundles (ปรับปรุงหลักสูตร, หลักสูตรใหม่, and structurally
identical variants like ปิดหลักสูตร / หลักสูตรวิชาโท) are in scope.

### Physical vs. logical split

Two ways to give each curriculum its own `resolution_id` were considered:

- **Logical**: keep the bundle file intact; extend `meeting_manifest.json` so
  one file maps to N manifest entries (title + page/char range per
  curriculum), and change the Loader stage to expand one path into N
  `Resolution`s at load time.
- **Physical**: cut the bundle into N real `.md` files on disk, one per
  curriculum, and let each flow through the pipeline exactly like any other
  resolution file.

Logical was rejected: `BaseLoader.load(path) -> Resolution` is a strict 1:1
contract used by every Loader (`PlainLoader`, `MetadataLoader`, `NerLoader`)
and both call sites (`cli.py`, `runner.py`). Supporting N Resolutions per file
means changing that contract across the tested core (ADR-0001 territory) —
and it still has to hand the Chunker *sliced* text per sub-Resolution to avoid
duplicating every chunk N times, so it never actually preserves an "intact
file" advantage over physical splitting. Physical splitting confines all of
the complexity to corpus-prep tooling that already owns this kind of
transformation.

## Decision

`tools/corpus_prep/split_curriculum_bundles.py` detects curriculum bundles by
content (a "จำนวน N หลักสูตร ... ดังนี้" declaration followed by N sequential
top-level enumerators, N ≥ 2) and splits each into N physical `.md` files:

- New files are named `<stem>__1.md` … `<stem>__N.md`. Filenames stay opaque
  pointers (ADR-0003) — no curriculum title is encoded in the name.
- Each piece carries the shared meeting/committee preamble plus its own body,
  so it reads as a self-contained resolution.
- `meeting_manifest.json` is patched with one entry per piece
  (`title = "<original มติ title> — <curriculum title>"`), so titles and
  `resolution_id`s stay distinct per curriculum without any loader changes.
- The original file and its `_LINK.txt` are archived as `*.dup` (existing
  convention: recoverable, excluded from `rglob("*.md")`) rather than deleted.
- `_LINK.txt` is duplicated to every piece unchanged — one source PDF
  legitimately covers every curriculum in the bundle, so N resolutions
  sharing one URL is expected here (this required a companion fix in
  `rebuild_manifests.py`'s duplicate-link detector, which previously treated
  any same-URL-same-folder group as a suspected accidental duplicate).

Splitting is validated, not guessed: a split only happens when the enumerator
count matches the declared N exactly, every resulting piece clears a
minimum-length and balance-ratio bar, and no two pieces extract to the same
title or identical body text. Files that don't clear these bars are logged
to `academic_resolutions/curriculum_split_review.md` for manual handling
instead of being split incorrectly. Every one of these guards was added
*after* a real failure mode was observed on the corpus, not speculatively —
see "Lesson: validate the already-applied output too" below for why that
matters.

### Lesson: validate the already-applied output too

The guards above were added incrementally across several rounds as new
failure modes surfaced (subsection headings reusing the same "๑) ๒)" numbering
as the top-level curriculum list; a closing-recap line echoing curriculum 1's
name; two pieces resolving to byte-identical content). Each round's *new*
files were checked against the *current* rules — but files applied in an
*earlier* round, under the *older, less strict* rules, were never
retroactively re-checked. A dedicated audit was needed to confirm the earlier
output was still sound: scan every `meeting_manifest.json` entry produced by
a split for titles that don't contain "หลักสูตร" (a direct sign the boundary
landed on a subsection heading like "สาระในการปรับปรุงแก้ไข" instead of a
curriculum name). That audit caught 28 files split by the first round, before
the "enumerator must be followed by หลักสูตร" guard existed — they were
restored from their `.md.dup` archive and re-split under the current rules.
**Whenever a new validation guard is added to this script, re-run the title
audit across the whole corpus, not just new candidates** — it is the cheap,
mechanical check that catches this class of regression. It was needed a
second time in the same rollout, after adding degree-level-reset handling
(below) — the audit stayed clean both times, which is what let the rollout
keep going rather than freezing at the first corruption find.

### Lesson: a keyword match is not a semantic match

Two more failure modes surfaced after the corruption-audit round, both from
the same root cause: requiring the enumerator be followed by "หลักสูตร" stops
it landing on a subsection *heading* (which uses different words), but it
does **not** stop it landing on a *sentence* that happens to start with the
word "หลักสูตร" as its grammatical subject — e.g. a numbered list of
committee recommendations ("๑. หลักสูตรเป็นหลักสูตรระดับบัณฑิตศึกษา ควรมี...")
or of program objectives ("๑) หลักสูตรเน้นการวิจัยเป็นหลัก เพื่อผลิต...").
Fixed with a negative lookahead rejecting a small set of verbs that never
follow "หลักสูตร" in a real curriculum name (เป็น, เน้น, ควร, ต้อง, จะ, ให้,
มี, อยู่, คือ) — a real curriculum name is always `หลักสูตร<degree-type>
สาขาวิชา<field>`, never a clause. Related: a curriculum list is often split
into degree-level sections (ระดับปริญญาตรี, ระดับบัณฑิตศึกษา, ...) that each
restart their own "๑) ๒)..." numbering from 1; the boundary matcher now
treats a level-header line as a valid reason for a "๑)" to reappear
mid-sequence (continuing the running count rather than resetting it) — but
the header regex must be anchored to the *start of a line*, not matched
anywhere in the text, or it fires on an incidental mention like "...เกณฑ์
มาตรฐานหลักสูตรระดับบัณฑิตศึกษา พ.ศ. ๒๕๕๘" inside an unrelated sentence.

## Consequences

- Each curriculum bundled in one มติ gets its own `resolution_id` and title,
  enabling a per-curriculum Silver query and independent citation.
- No changes to `src/rag_lab/` — the 1:1 Loader contract holds; new split
  files are indistinguishable from any other corpus file to the pipeline.
- Automated coverage is partial by design: after several rounds of extending
  the detector (period-style "๑." enumerators in addition to "๑)", a
  subtotal-report exclusion, a piece-balance ratio check in place of a bare
  length floor, duplicate-title/duplicate-body rejection, degree-level-reset
  handling, a verb-lookahead against sentence-not-name matches) and one
  corruption audit + repair pass, 168 files were split cleanly (512
  resulting pieces). ~159 files remain in the manual-review queue,
  categorized by why the detector declined them (see the queue file's
  header) — many of these are *correctly* left unsplit (e.g. two
  closely-related programs sharing one joint rationale, which would be
  duplicate-guessed or thinned out by forcing a split) rather than being an
  open backlog of bugs.
- Deliberately out of scope: curricula listed as table rows instead of
  "๑)/๑." text, and English-only program-name lists with no "หลักสูตร"
  prefix. Both appeared in the review queue in small numbers (~15-20 files
  combined); the schema variability of table parsing made it a poor
  effort/risk trade for that yield, especially given how many rounds of
  hidden bugs the text-based detector alone produced.
- User-confirmed policy call on "bulk name-only lists" (a "unbalanced"-flagged
  shape recurring dozens of times in the review queue: N curricula correctly
  counted, each piece just a short curriculum-name line, with no
  per-curriculum rationale — the substance is one shared paragraph covering
  all N): **kept as one resolution, not split.** Splitting would produce N
  content-free stub resolutions differing only by name; the ratio guard's
  rejection of this shape is correct behavior, not a detector gap, and is not
  being loosened.
- Re-running `rebuild_manifests.py --apply` after a split run is required to
  regenerate `master_list.csv`; the split script itself never touches it.
