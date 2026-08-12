# programs_by_file.json regeneration (2026-08-12)

`tools/corpus_prep/audit_program_tag_regeneration.py`. The cached program
tags dated **2026-07-25**; `match_programs` was repaired **2026-08-11**
(degree filter + cross-subject guard, `docs/program-matcher-absorption.md`)
and the corpus moved three times in between. This report regenerates the
artifact, splits the delta into the two causes, and verifies for real the
blast radius that could previously only be stated as a counterfactual --
`build_gold_candidates.py` was reading the *cached* file, so nothing
published had moved yet.

## 1-2. What changed, and why

| arm | files | files changed | tags gained | tags lost | stripped bare | total tags | files tagged |
|---|---|---|---|---|---|---|---|
| **overall** (07-25 artifact -> 08-12 artifact) | 2853 | 500 | +212 | -617 | 117 | 4743 -> 4338 | 1711 -> 1595 |
| (B) corpus drift (old matcher, 07-25 -> 08-12 corpus) | 2853 | 109 | +95 | -46 | 2 | 4743 -> 4792 | 1711 -> 1710 |
| (A) matcher repair (08-12 corpus, old -> new matcher) | 2854 | 446 | +140 | -594 | 115 | 4792 -> 4338 | 1710 -> 1595 |

The middle row is the corpus changes alone (07-28 OCR remediation, 08-09
re-OCR); the 08-08 title repair cannot appear here at all, because
`tag_programs.py` keys on the file path and matches over body text, and a
title repair touches neither. The bottom row is the two guards, and it
reproduces `docs/program-matcher-absorption.md`'s own per-file figures
(S3) -- which is what licenses reading the middle row as drift rather
than as a residue.

## 3. Does it reach the program qrels?

**No, and the reason is stronger than a measurement.** `program_candidates()`
iterates the mapping's **keys**, and `tag_programs.py` writes a key for every
live corpus file -- including the ones that match zero programs. The tag
*values* are never read: the function derives a `resolution_id` per file and
gates on `canonical in resolution_id`, an exact substring of the manifest
title. So the matcher cannot move the program qrels **structurally**, not
merely in this sample; S2 executes that claim by blanking every value and
requiring identical output. (CLAUDE.md's wording -- "seeds from tagged
files" -- overstated the coupling; only corpus *membership* matters, which
is what S4 watches.)

Both arms were recomputed anyway:

- program candidates from the 07-25 mapping: **147** (662 pairs)
- program candidates from the 08-12 mapping: **147** (662 pairs)
- candidates appearing in only one: **0 / 0**
- candidates whose resolution_id list moved: **0**

Against the published `gold_candidates.json` (built 2026-07-25, before the
08-08 title repair moved 4 resolution_ids):

- published program candidates: **147**; recomputed: **147**
- only published / only recomputed: **0 / 0**
- candidates whose list moved: **1**

Read that last figure against the right artifact. `gold_candidates.json`
dates 2026-07-25 and is a **superseded intermediate**: the scored set was
regenerated 2026-07-30 for the `resolution_id` uniqueness fix, so a
candidate can differ from it while the gold set is correct. The one that
does is `...สาขาวิชาวิศวกรรมชีวการแพทย์`, whose single truncated
`2567/1/...(หลักสูตรนานาชาติ)` id became the two full ids that end
`...ฉบับปี พ.ศ. ๒๕๖๓` / `๒๕๖๔` -- and `gold_query_set_73det.yaml` already
holds both. What must hold is that no difference reaches a *scored* pair,
which is S6.

The corpus also gained one file since 07-25 (S4):
`2568/ครั้งที่ 7/เรื่อง รับรองรายงานการประชุม.md`, from the 2026-08-09 CHECO
restoration. It matches zero programs and its `resolution_id` contains no
program canonical, so it adds no hit to any candidate -- which is why
membership moved and the qrels did not.

And against the scored gold set itself (`gold_query_set_73det.yaml`):

- program gold queries: **30**, gold pairs: **247**
- gold pairs no longer produced by the recomputed candidate set: **0**

## 4. Does it change a route?

`classify_query` asks only whether *any* program matched, never which one.

| gold entity_type | route | queries |
|---|---|---|
| course | course | 33 |
| faculty_adjunct_aggregate | faculty | 13 |
| person | person | 30 |
| program | program | 30 |

## 5. Self-checks

| check | result | detail |
|---|---|---|
| S1 cache reconstruction == regenerated artifact | PASS | 2854 of 2854 files agree |
| S2 program_candidates reads only the key set | PASS | 147 candidates from an all-empty mapping, identical: True |
| S3 repair figures reproduce docs/program-matcher-absorption.md | PASS | got {'gained': 140, 'lost': 594, 'stripped_bare': 115, 'changed': 446} vs published {'gained': 140, 'lost': 594, 'stripped_bare': 115, 'changed': 446} |
| S4 corpus membership change moves no program candidate | PASS | 2853 -> 2854 files (removed 0, added 1: 2568/ครั้งที่ 7/เรื่อง รับรองรายงานการประชุม.md); 0 program candidates moved |
| S6 no scored gold pair is lost by the recomputed candidate set | PASS | 0 of 247 program gold pairs lost; 1 published candidate(s) differ from the 07-25 artifact (superseded by the 07-30 gold-set regeneration) |
| S5 router distribution unchanged | PASS | got {('course', 'course'): 33, ('faculty_adjunct_aggregate', 'faculty'): 13, ('person', 'person'): 30, ('program', 'program'): 30} |
