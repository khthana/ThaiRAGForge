# Corpus Preparation Tooling

Scripts that produced and maintain the `academic_resolutions/` corpus (2,854
Resolutions) from the KMITL academic-office website. These are **data-prep
utilities**, not part of the RAG framework (`src/rag_lab/`) — the framework never
imports them, and ADR-0001's importability rule does not apply here.

Run them from the **repository root** (some paths are relative to it). Several
scripts have a hardcoded `BASE_DIR` / `TARGET_SUBFOLDER` at the top — edit those to
target a specific year/session before running.

The directory holds far more than the original acquisition pipeline: the OCR
remediation campaign, the entity taggers, the Gold-set builders, and the audits all
live here. They are grouped below by what they are *for*, since only §1 is a
sequence — everything after it is invoked on demand.

**Two things to know before running anything that writes.** (a) Most write-capable
scripts here are dry-run by default and need `--apply`; that is deliberate and the
default should not be flipped. (b) Anything that changes a title, a filename, or a
manifest entry can change a `resolution_id`, which **makes every built index stale
for the affected files** (ADR-0002). Re-run `audit_resolution_ids.py` after such a
change, and expect to rebuild.

## 1. Acquisition pipeline (in order)

1. **`scrape_kmitl.py`** — Scrapes the KMITL academic-office page for a given ปี พ.ศ.,
   downloads each resolution's PDF, and writes a sibling `<name>_LINK.txt` containing
   the source Google Drive URL. Output → `academic_resolutions/<year>/ครั้งที่ N/`.

2. **`ocr_pdf_to_md.py`** — Main OCR. Converts each PDF to Markdown using a local
   Ollama model (`scb10x/typhoon-ocr1.5-3b`), with image preprocessing, retries, and
   bad-output detection. Writes `<name>.md` next to each PDF. Requires Poppler
   (`POPPLER_PATH`) and the Ollama model built from `Modelfile`.

3. **`delete_bad_ocr.py`** — Cleanup. Scans a session folder for `.md` files containing
   the OCR error marker and deletes them (has a `DRY_RUN` guard — defaults matter, read
   before running).

4. **`check_ocr_coverage.py`** — Diagnostic. Lists PDFs in a session folder and whether
   a matching `.md` already exists. Useful to see what still needs OCR.

4a. **`scan_ocr_repetition.py`** — Diagnostic, read-only. Scans every corpus `.md` for
    an OCR hallucination-loop artifact (the model gets stuck and repeats the same short
    token many times instead of transcribing real content, e.g. a garbled URL repeated
    600+ times). Tags each hit "table" (inside a data cell — doesn't affect retrieval,
    deprioritized) or "prose" (corrupts searchable/citable text — worth fixing) and
    writes the deduplicated list of affected source documents to
    `academic_resolutions/ocr_repetition_review.md`. Fixing requires a fresh `ocr_pdf_to_md.py`
    pass on the *source PDF* (the corrupted text is gone, not just misformatted) —
    this script only detects and reports.

5. **`split_curriculum_bundles.py`** — Curriculum splitting (ADR-0004). Some
   resolutions bundle several curricula into one มติ (e.g. one "ปรับปรุงหลักสูตร"
   file covering 3 curricula); this splits each into one physical `.md` file per
   curriculum, patches `meeting_manifest.json`, and archives the original as
   `*.md.dup`. Detection + boundary validation is content-based with a hard
   length/count guard — anything it can't split cleanly goes to
   `academic_resolutions/curriculum_split_review.md` for manual handling rather
   than being guessed at. Dry-run by default; pass `--apply` to write. Run
   **before** `rebuild_manifests.py` so the new files get picked up as ordinary
   corpus entries.

6. **`rebuild_manifests.py`** — Reconciliation (ADR-0003). Scans the corpus tree and
   writes per-meeting `meeting_manifest.json` (full titles + URLs — the metadata
   source of truth) and `academic_resolutions/master_list.csv`. If the original agenda
   capture `1.docx` is still at the repo root it is reconciled in (join on Google Drive
   file IDs); once retired, the tool runs corpus-only and keeps titles from the existing
   manifests. Dry-run by default; pass `--apply` to write. Re-run after adding/renaming
   corpus files (including after a curriculum split); see
   `docs/corpus-reconciliation-log.md` for the reconciliation history.

## 2. OCR-corruption remediation

The corpus's OCR is imperfect in ways that a spell-check cannot see, so corruption is
found by **LLM consensus** and repaired by targeted re-OCR. Full narrative, including
what each batch found and why several detection heuristics were rejected:
`docs/llm-ocr-scan-log.md`. Status: **complete** — both the original 872-page
consensus batch and the much larger "kernel A" batch (1,982 pages / 393 files) are
written back, and the human-review queue reached 0.

- **`llm_ocr_scan.py`** — Detection. Splits every corpus `.md` on its `## Page N`
  headers and asks two local models whether each page is garbled. The AND-gate
  (both models agree) is a **cost filter, not a quality filter** — its consensus
  threshold is exactly what left 1,329 files untouched, which is what the kernel-A
  batch later had to cover. Also exposes `split_pages`, reused by the review app.
- **`reocr_consensus_pages.py`** / **`reocr_kernel_a_stage.py`** — Re-OCR the flagged
  pages from the *source PDF* (corrupted text is gone, not merely misformatted).
- **`reocr_adjudicate.py`** — Dual-model old-vs-new comparison per page. Whole-page
  comparison **overstates** the defect rate (~83% raw vs ~56% span-confirmed); read
  §8 of the log before trusting a verdict rate.
- **`reocr_apply.py`** — Write-back. `replace_page_text` is a contiguous-run union
  merge, *not* "replace the first `## Page N` block" — the earlier version treated a
  duplicated header as boilerplate and silently dropped real content (fixed
  2026-07-28 across 109 files). Backs a file up **once**, so a second re-OCR would
  archive the already-repaired text, not the original.
- **`reocr_tiebreak_round.py`** / **`reocr_tiebreak_analyze.py`** — Tie-break passes.
  These perturb temperature/DPI on purpose: re-OCR at `temperature=0.0` in this
  pipeline is fully deterministic, so a literal repeat reproduces its input
  byte-for-byte and settles nothing.
- **`excise_ocr_loops.py`** — Surgical removal of hallucination-loop runs (the
  blind spot the garbled-prose check misses: one short token repeated hundreds of
  times).
- **`sample_kernel_a_check.py`** — Stratified sample for auditing kernel-A output.
- **`consensus_review/`** — Streamlit triage UI
  (`streamlit run tools/corpus_prep/consensus_review/review_app.py`). Historical: the
  queue it drained is at 0. Build record and one **known defect** (a calibration
  section that renders reference documents on the same page as the candidate, which
  invalidated a manual judging pass in the pooling-bias study): `docs/tickets.md`.
- **`restore_minutes_2568_7.py`** — One-off repair for a specific meeting.

## 3. Metadata, splitting, and audits

- **`fix_manifest_title_collisions.py`** — Repairs manifest titles that collide.
- **`audit_resolution_ids.py`** — **Run after any corpus or manifest change.**
  Reports every `resolution_id` clash with the evidence needed to tell a data error
  from a genuinely shared agenda title, and exits 1 on any clash (ADR-0002
  amendment).
- **`relabel_index_resolution_ids.py`** — Rewrites ids inside an already-built index
  so a gold set and an index agree again without a rebuild. Written for the 2026-07-30
  fix and, as it turned out, **never needed**: rebuild #3 minted the corrected ids
  from scratch. Kept for the next time an id changes without a rebuild.
- **`audit_title_body_agreement.py`** — Flags manifest titles that disagree with the
  document's own page-1 `เรื่อง` line. Note the design: it compares by **asymmetric
  token containment**, not string similarity, because the similarity version was
  rejected on measurement (median 0.660, 544 false alarms). This one scores median
  1.000 with 7 flags, 7/7 genuine. Report: `docs/title-body-agreement.md`.
  **Not applied** — 4 of the 7 would change `resolution_id`s.
- **`patch_gold_ids_for_split_titles.py`** — Migrates gold-set ids after a curriculum
  split renames their target.

## 4. Entity dictionaries and tagging

Feeds `metadata['people'|'programs'|'courses'|'faculties']`, consumed by the
`entity_tags` loader and the `entity_lookup` / `entity_boost` retrieval modes. All
rule-based (regex on Thai academic rank, or a curated dictionary) — **deliberately
not NER**, which fragmented organisation names; see
`docs/entity-extraction-and-gold-eval-log.md`.

- **`build_program_dictionary.py`** / **`build_course_dictionary.py`** — Build
  `programs.json` (253 entries, from manifest titles rather than a body-text scan)
  and `courses.json` (1,547 name→code entries, Latin/English only — Thai course-name
  extraction was tried and dropped as unreliable).
- **`canonicalize_people.py`** — Deduplicates person entities across spelling and
  rank variants.
- **`tag_people.py`** / **`tag_programs.py`** / **`tag_courses.py`** /
  **`tag_faculties.py`** — Batch tagging + per-run coverage reports.
- **`scan_entity_candidates.py`** — Read-only full-corpus scan. Use this to **size
  exposure before writing a broad regex change**; two real tagger gaps (the bare "อ."
  rank, and names split across adjacent `<td>` cells) were both scoped this way
  before any code was written.
- **`evaluate_ner.py`** — Evaluation of the GPU NER taggers, kept as the record of
  why the rule-based path won.

## 5. Query-set construction

- **`build_gold_candidates.py`** — Generates Gold query→Resolution candidates by a
  per-entity-type matching rule. That rule being **deterministic and re-derivable** is
  what makes a single annotator defensible (`docs/eval-validity-threats.md`) — and
  also the source of the pooling-bias threat, since containment matching is close to
  what BM25 does. Output: `config/eval/gold_query_set_73det.yaml`.
- **`llm_thematic_bootstrap.py`** / **`add_thematic_to_gold_set.py`** — Build the 179
  thematic queries. Keep them **separate** from the entity-anchored set: they carry
  signal pointing the opposite way on the chunker axis, so pooling cancels two real
  effects.

## Supporting files

- **`Modelfile`** — Ollama model definition for the OCR model (system prompt +
  parameters). Build with `ollama create ... -f Modelfile`.
- **`ocr_legacy.py`** — Earlier OCR script, superseded by `ocr_pdf_to_md.py` (no image
  preprocessing / retry logic). Kept for reference only.
- Superseded `*.dup` / `*.bak` backups were moved **off-repo** in 2026-07-30 (2,389
  files, path structure preserved) to
  `D:\academic_resolutions (ข้อมูลดิบ + OCR)\_superseded_from_repo\`. Anything that
  reads `.pre_reocr.bak` / `.corrupted_ocr.bak` must be pointed there.

## Dependencies

Covered by the base `pyproject.toml` dependencies (`uv sync`): `ollama`, `pdf2image`,
`pillow`, `typhoon-ocr`, `requests`, `beautifulsoup4`. Also needs a local **Ollama**
install and **Poppler** on the system PATH.
