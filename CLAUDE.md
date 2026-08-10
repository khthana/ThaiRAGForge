# CLAUDE.md

Guidance for Claude Code when working in this repository. For a project overview see
`README.md`; for domain vocabulary see `CONTEXT.md`; for past architectural decisions
see `docs/adr/`.

## Development

- **Install (framework)**: `uv sync --extra lab` — installs the `lab` extra + the
  `dev` group (pytest). Corpus-prep only needs `uv sync`.
- **Run tests**: `.venv/Scripts/python.exe -m pytest` (pytest reads `src/` via
  `pythonpath` in `pyproject.toml`). The heavy bge-m3 smoke test is skipped unless
  `RAG_LAB_SMOKE=1` is set.
- **CLI** (needs `PYTHONPATH=src`): `python -m rag_lab.cli run --config
  config/experiments/dev_smoke.yaml` (batch build), or the low-level `build` /
  `retrieve` commands.
- **UI**: `.venv/Scripts/streamlit.exe run app/streamlit_app.py --server.fileWatcherType
  none` — Mode B (Query & Compare, the main script), Mode A (Build/Run,
  `app/pages/1_build_run.py`), and the Chunk Inspector (`app/pages/2_chunk_inspector.py`
  — visually compare how each chunker splits a document, and triage abnormally large
  chunks; reads `chunks.parquet` only, never `embeddings.npy`) in the sidebar nav. The
  `--server.fileWatcherType none` flag suppresses a harmless but noisy
  `ModuleNotFoundError: torchvision` warning: Streamlit's auto-reload watcher walks
  every loaded module's `__path__`, which triggers `transformers`' lazy-import
  machinery on unrelated submodules (e.g. `zoedepth`) that need optional deps we don't
  install. All three pages are thin shells over `rag_lab.query_service` /
  `rag_lab.runner` + `rag_lab.config` (the tested core); the widgets themselves are
  smoke-tested via `streamlit.testing.v1.AppTest`, not unit-tested individually.

## Conventions

- The core package `src/rag_lab/` must not import Streamlit (ADR-0001): keep it
  importable and unit-testable; UI/CLI are thin layers on top.
- Add a strategy by creating a file + registering it (`src/rag_lab/registry.py`);
  never edit the runner (Open/Closed).
- `Chunk.resolution_id` is load-bearing — relevance is judged at the Resolution level
  (ADR-0002). It must be unique per file, and that is now enforced rather than
  assumed (ADR-0002 amendment, 2026-07-30): `make_resolution_id` appends a
  folder-local ` #N` rank when a meeting lists two items under one title,
  `pipeline.build_index` refuses to build on a collision, and
  `tools/corpus_prep/audit_resolution_ids.py` reports every clash (exit 1) with
  the evidence to tell a data error from a genuine shared title. Run it after any
  corpus/manifest change. An id change makes built indices stale for the affected
  files — they store the ids they were built with.
- **Run `tools/eval/audit_pipeline_invariants.py` before trusting any eval refresh.**
  Three silent-corruption bugs have been found by accident rather than by looking
  (corpus-discovery contamination, stale BM25/hybrid result cache, `resolution_id`
  collisions); they share a shape — a mismatch between two artifacts produced at
  different times by different scripts, which never crashes, it just makes a number
  wrong. This script checks 25 such invariants across corpus/index/eval layers
  (id uniqueness, row alignment of chunks↔vectors↔lexical, index-vs-corpus
  membership, embedding sanity, gold-id resolution, results-vs-index freshness)
  and exits 1 on any FAIL. Latest report: `docs/pipeline-invariant-audit.md`.
  **The caveat it surfaced is CLOSED (2026-08-09, check `E0`), and the way it was
  closed is the reusable part.** `BuildCombo.id` hashes loader+chunker+embedder but
  **not the corpus**, so a smoke-subset combo and a full-corpus combo share an id
  and a persisted result could not be attributed to one index — this is *why* the
  stale-cache incident was invisible (12 of 43 combo ids really do exist under >1
  index root). **Hashing the corpus into the id was the obvious fix and is
  disqualified**: the id *is* the on-disk directory name and the prefix of every
  result filename, so it would rename 55 index dirs on every corpus edit, orphan
  ~24k results, and break the combo names hardcoded in eval scripts
  (`plain__fixed_size__local__ceea7536`). **Attributability, not renaming**: the
  disambiguating data already existed but never left the index —
  `build_manifest` writes `docset_hash`/`n_resolutions`, and `ArtifactStore.load`
  simply never read it. It now stamps `Index.provenance`, `pipeline.retrieve`
  copies it onto `RetrievalResult.index_dir`/`docset_hash`, and E0 attributes every
  result by three rules, strongest first: **recorded** (the result names its index),
  **unique name** (the combo id exists under one root), **elimination** (exactly one
  candidate index holds every `resolution_id` the result cites — sound because the
  result *did* come from one of them). Only >1 survivor at rule 3 is a FAIL; "no
  built index" (the 8 deleted superseded combos) and "no candidate fits" (drift,
  which is E3a's finding) are classified, not failed — the
  [[feedback_cleanup_can_break_an_audit]] lesson. **Elimination alone resolves
  100% of the live ambiguity**, which is why no backfill was needed: 0 of 23,156
  unattributable today (15,038 by unique name, 7,268 by elimination, 850 no built
  index). Two things it *sharpens* rather than merely turning green: E3a now checks
  ids against the **attributed** index instead of the union over every root sharing
  the name (a union is a superset, so it would accept an id only the smoke fixture
  holds), and E4 keys staleness on the attributed dir, so a rebuilt smoke fixture
  can no longer make full-corpus results look stale. `provenance` is deliberately
  kept **out of `meta`** (which is what `save` writes back, so a load-time field
  must not round-trip) and the new result fields are optional (the ~24k legacy
  results must keep validating); `select()` carries it because a filtered view is
  still the same build, unlike `lexical_scorer`. Rule 1 currently fires on **zero**
  files on disk — nothing has been re-run since the fields landed — so
  `tests/tools/test_audit_pipeline_invariants.py` pins all six outcomes, or it
  would be exactly the vacuous PASS the next bullet warns about.
  Two lessons about the audit itself, both learned by breaking it the same day it was
  written: (a) **a check whose subject matter moves becomes a vacuous PASS** — C4
  (orphaned `.md.dup`) went 24→0 the moment those archives were moved off-repo, so it
  now follows them to `ARCHIVE_ROOT` and says "0 of 240" rather than "0"; (b) **don't
  let a known-retired artifact keep the gate red** — deleting the 8 superseded combos
  removed the only indices still holding the pre-contamination-fix ids, which made
  E3a jump 7→3,106 for result sets nothing reads. Those are now classified separately
  (E3c contamination ids, E3d pre-repair titles, `RETIRED_RESULT_DIRS`) so a FAIL
  still means a *live* result set has drifted. Because 0 is ambiguous between
  "examined and clean" and "nothing left to examine", the E3 checks now print their
  denominator — `E3a 0 of 23,156 live result files` is a real pass, `E3d 0 of 0` says
  so out loud. **`I6` was sharpened 2026-08-08 after it was caught unable to see a
  whole class of corpus change**: it derived "the corpus's last edit" from `*.md`
  mtimes alone, but a `resolution_id` is built from the manifest title (ADR-0003),
  so the title repair that day moved 4 ids without touching a single `.md` — I6
  would have called all 41 affected indices current while they still held
  pre-repair ids. It now reads `meeting_manifest.json` mtimes too, and counts a
  recorded relabel (`relabeled_mispairings.at` in an index manifest) as bringing
  an index current without a rebuild — without that second half it would sit
  permanently red after any title repair, and an always-red check is one nobody
  reads. Current state (**re-run 2026-08-09 after E0 landed**): **24 pass /
  1 warn / 0 fail** — the gate is green for the first time. **The warn is real and
  is not E0's doing**: it is that same `I6`, 41 indexes built 2026-08-08 19:33
  against a corpus last edited 2026-08-09 09:53, i.e. the `2566/ครั้งที่ 3`
  re-download + re-OCR earlier that day. **Unlike the title repair, that one is a
  text change, so a relabel cannot discharge it** — those indices genuinely hold
  the old OCR of that file. It is left standing rather than waived because a
  16.4h rebuild is not worth it here: 0 gold entries in either gold set cite any
  resolution from that meeting, so no published metric can have moved. The
  previous state was **24 pass / 0 warn / 1 fail** (that lone FAIL being the
  `BuildCombo.id` caveat, now closed above). That headline was written here before it was true —
  the report on disk at the time said **21 pass / 3 warn / 1 fail**, the 3 warns
  being index-staleness ones nobody had chased (`I3b` coverage 2853/2854, `I5` 41
  manifests drifted, `I6` 41 indexes built before the corpus's last edit). Rebuild
  #3 cleared all three to 0/0/2854-of-2854, so the claim is now verified rather than
  asserted. **`E4` (results newer than their index) passing at 0 across 23,156 result
  files is the mechanical confirmation that the whole 08-06/08-07 refresh chain is
  complete** — that is the check to look at after a rebuild, not the headline count.
  Both former warns were chased to root cause rather
  than waived, and each turned out to be a symptom of something bigger than the
  warning said (the 5 duplicate thematic queries → the whole 179-entry subset was
  unanswerable; see above). C4's 24 orphan archives were reviewed one by one and
  closed (nothing was lost: 21 tail fragments of a wrapped title, 1 rename, 2
  misfiled-but-live); the verdicts are encoded as rules, and the same-document
  test compares page-1 `เรื่อง` headings because whole-file similarity decays
  across the re-OCR boundary. That review surfaced what was then the corpus's one
  known title↔content defect: `2568/ครั้งที่ 7`'s CHECO-titled file held
  รับรองรายงานการประชุม instead. Cause was the download stage fetching the wrong
  Drive id (two byte-identical PDFs, same SHA-256) while the manifest, `_LINK.txt`
  and `master_list.csv` all already held the correct one
  (`1d4iz1dpnPweAn7pxBfxlvJf9IJZwIJFJ`) — so the fix was a re-download + re-OCR of
  that one URL, no metadata change (0 gold queries in the 73det set cite it).
  **Done (`restore_minutes_2568_7.py`); verified 2026-08-09 by reading both files —
  the CHECO file holds CHECO text, the minutes file holds the minutes.** Its
  mechanism recurs: see the orphaned-agenda-items bullet below.
  A general title-vs-body check was prototyped and **rejected on
  measurement**: median agreement is 0.660 over 2,820 files with 544 below 0.5,
  nearly all false alarms from agenda-number prefixes.
- **Run `tools/eval/audit_doc_claims.py` after editing `CLAUDE.md` or
  `docs/paper-results-summary.md`, and after any eval refresh.** It is the docs
  layer the sweep above was missing: `audit_pipeline_invariants.py` gates
  corpus/index/eval and `diff_significance_reports.py` gates report-vs-report,
  but **nothing read the prose**, which is where this project's avoidable errors
  actually live — a number typed by hand, correct that day, that no later
  refresh touches because a refresh re-runs scripts and diffs reports. Four
  checks: D1 report older than its generator (+ reports that don't declare one),
  **D2 every 4-decimal figure in the prose must appear in some report** (the main
  one), D3 a p-value quoted against a contradicting verdict word, D4 an eval
  *input* changed after a report that reads it (the "editing `ROUTE_COMBO`
  silently re-scores `soft_vs_hard_routing.md`" failure). Report:
  `docs/doc-claims-audit.md`; triaged exemptions with written reasons in
  `tools/eval/doc_claims_allowlist.yaml`. **First run found three real stale
  tables, and the way it found them is the point**: all three had drifted in the
  2026-08-06 refresh *without a single verdict cell changing*, so
  `diff_significance_reports.py` correctly reported 0 flips and nobody re-copied
  the numbers. (1) the per-chunker BM25-vs-embedder table — every one of 36
  cells off by ~0.001-0.003; (2) the MAP/precision@1 summary — same, plus it
  still said hybrid/aggregate precision@1 beat **4 of 8** where the report says
  **5 of 8**, a figure CLAUDE.md had already been updated with, so the two docs
  openly disagreed; (3) the structural-ceiling table, which was never extended
  when the 33 `course` queries landed — it was a ceiling for two-thirds of the
  set (now `all 106` 0.8856, was `all 73` 0.8922; `program` 0.9000 → 0.8979,
  `course` 0.8729 added). Two design notes worth keeping. **D2's haystack is
  deliberately `data/results/**/*.md` only** — including the per-query JSON
  makes it *vacuous rather than thorough*, since 225 MB of scores contains
  almost any 4-decimal value by coincidence (untraceable count 122 → 27, and not
  one of those 95 was genuinely sourced). **D2 clears a figure two ways** — cited
  as superseded, or inside a dated snapshot — because `paper-results-summary.md`
  keeps its own supersession history on purpose; `tests/tools/test_audit_doc_claims.py`
  pins those exemptions in both directions so the next one added can't quietly
  make the check vacuous. D3 is a **WARN by design** (irreducible false
  positives: a parenthetical can attribute its p to one arm and its verdict word
  to another), and D1b warns on reports with no identifiable generator so
  D1a's denominator stays honest. Uses filesystem mtimes, not git dates — reports
  are gitignored and a script's commit lands *after* the run, which flagged all
  10 pairs as false positives on the first run.
  **D1b closed 2026-08-09 (18 → 0), and the naive way to close it would have
  traded one benign WARN for 8 FAILs.** The line belongs in the *generator*, not
  the report — a hand-added line is erased by the next run — so 9 live reports
  got it from their emitting script (`embedder_matrix_9way`, `run_gold_bm25_eval`,
  `run_gold_hybrid_eval`, `run_gold_entity_{boost,lookup}_eval`,
  `residual_relevance_sample`, `rq4_score`, `hybrid_significance_test_9way`).
  **That edit moves the script's mtime, which is exactly what D1a watches**, so
  the 4 seconds-level ones were **re-run** rather than hand-patched (every
  published figure reproduced identically: rq4 +0.1181/+0.1005/+0.0734 and the
  guarded +0.0706, residual 0.191/0.224/0.224, thematic −0.0449/−0.0516) and the
  6 hours-level ones got the byte-identical string the script now emits, so the
  next real run is a no-op. The other **8 are superseded snapshots** whose
  generators are live scripts that have moved on (four `run_gold_chunker_eval.py`
  rollups, the Silver one, the ConGen/SCT truncation fix, the 2026-07-30
  `pipeline_invariant_audit.md`) — they declare their generator *and* say they are
  superseded, and `RETIRED_REPORTS` classifies them so D1a is not permanently red,
  the same rule `RETIRED_RESULT_DIRS`/`RETIRED_RESULTS` already apply one layer
  down. `person_cross_cell_fix_review.md` is the 9th entry and the only report
  that can name **no** generator honestly (a one-off diff from `e1523b3`; the
  throwaway script was not kept). New **D1c** warns if a `RETIRED_REPORTS` entry
  names a missing file, because an exemption list is the easiest way to make a
  check vacuous — the [[feedback_cleanup_can_break_an_audit]] shape again — and
  the tests pin that no current report (`routing_eval`, `rq4_score`,
  `oracle_union_ceiling`, `power_analysis`, the three 9-way tables) is exempt.
  Was **5 pass / 1 warn / 0 fail**, the warn being D3's 3 known false positives.
  **Now 4 pass / 1 warn / 1 fail (2026-08-10), and the FAIL is a TRUE POSITIVE —
  do not allowlist it.** D4 reports that `rq4_score.md` and
  `rq4_score_guarded.md` predate a change to `tools/eval/rq4_generate.py`, which
  is exactly right: that change is the truncation repair (see the RQ4 bullet),
  so those two reports really do describe answers produced by a generator that
  no longer exists. It clears when the 80 truncated cells are regenerated, and
  until then a red D4 is the correct reading of the world. This is the one case
  where the "don't let a known-retired artifact keep the gate red" rule does
  **not** apply — nothing here is retired, the work is simply outstanding.
- The corpus (`academic_resolutions/`) is gitignored and lives at the repo root;
  corpus-prep tooling in `tools/corpus_prep/` needs Poppler + Ollama.
- **Superseded backups live off-repo** (2026-07-30): 2,389 `*.dup` / `*.bak` files
  moved to `D:\academic_resolutions (ข้อมูลดิบ + OCR)\_superseded_from_repo\`,
  path structure preserved (moved, not deleted — ADR-0004 recoverability intact).
  `tools/archive_unused.py` is the mover: it records a per-category verdict
  (SAFE = no code path reads these; GATED = something does, needs a flag) and
  dry-runs by default. Two consequences to know before touching `.bak` again:
  `llm_ocr_scan.py`'s floor check reads `.pre_reocr.bak`/`.corrupted_ocr.bak`, so
  point it at the archive; and `reocr_apply.py` backs a file up only once, so a
  future re-OCR would archive *current* text — the true pre-re-OCR original now
  exists only on D:. See `docs/llm-ocr-scan-log.md` (last section).
- Corpus layout is `<ปี>/ครั้งที่ N/` (special sessions: `ครั้งที่ Ns`); per-meeting
  `meeting_manifest.json` is the metadata source of truth for titles/URLs — never
  encode metadata in filenames (ADR-0003). The reconciled inventory is
  `academic_resolutions/master_list.csv`.
- OCR-corruption remediation (LLM consensus scan + re-OCR + dual-model
  old-vs-new adjudication + write-back) lives in `tools/corpus_prep/`
  (`llm_ocr_scan.py`, `reocr_consensus_pages.py`, `reocr_adjudicate.py`,
  `reocr_apply.py`), with a review UI at `tools/corpus_prep/consensus_review/`
  (`streamlit run tools/corpus_prep/consensus_review/review_app.py`).
  Status/handoff: `docs/llm-ocr-scan-log.md`. Original 872-page
  consensus-AND-gate batch complete and written back (commit `b692480`,
  2026-07-16): 753/768 pages live, 18 kept old. **2026-07-25**: found one
  file the original scan had missed entirely (a coverage gap) — sized and
  found to actually be 2 separate mechanisms, not a simple gap: Mechanism A
  (consensus-threshold exclusion, 1,329 files/~47% of the corpus never
  touched by the AND-gate) and Mechanism B (a detection blind spot for
  massive single-character repetition, 2 severe files). A 100-page sample
  found only ~56% of phi4-only flags are real span-confirmed defects (not
  the ~83% raw new/new verdict rate — see `docs/llm-ocr-scan-log.md` §8 for
  why whole-page comparison overstates it). **2026-07-27**: user approved
  and ran the full Mechanism-A (kernel-A) remediation anyway, with explicit
  disclosure that the pipeline has no ground-truth-vs-source-image check —
  **complete**: 1,982 pages written across 393 files. The combined human-
  review queue (both batches) was then reduced 308→203→137→99→88 over 4
  rounds of evidence-based bulk decisions (length-delta, span-confirmation,
  independent-reOCR-agreement, table-repetition-signature) — see
  `docs/llm-ocr-scan-log.md` §9-§10 for the full methodology, including a
  key finding that re-OCR at temperature=0.0 is fully deterministic in this
  pipeline (a literal-repeat tie-break round reproduced its input
  byte-for-byte, so a later tie-break round was redesigned to genuinely
  perturb temperature/DPI instead). The 88-page review queue reported above
  was a 27 ก.ค. snapshot — **the user finished reviewing the rest by hand;
  the queue is actually at 0** (re-verified twice 2026-07-28: 343/343
  (pdf,page) pairs have a logged decision, and a direct `decide_action`
  sweep found zero "awaiting human review" records). Mechanism B's 2 severe
  files (the LLM-detection blind spot — massive single-character repetition
  that neither model's garbled-prose check catches) are now **also fixed**
  (2026-07-27): re-OCR'd, adjudicated (unanimous new/new), applied, and
  verified clean by direct read. **2026-07-28**: a dry-run surfaced a real
  bug in the 2026-07-16 duplicate-`## Page N`-header fix (it treated the
  first occurrence as disposable boilerplate; both occurrences actually held
  different real content) — rewrote `replace_page_text` as a contiguous-run
  union-merge, fixed 94+10+5 files (incl. 5 keep-old-verdict files the old
  mechanism never reached). Sized the blast radius before deciding whether
  to rebuild: 41/111 changed files (37%) and 43/106 gold queries (41%) are
  gold-eval-relevant — `chunker_compare_full` was rebuilt the same day as a
  result (see the entity-tagging paragraph above for how that rebuild
  incidentally reached the entity-loader fixes too).
- Entity tagging (person/program/course/faculty, all rule-based — regex
  anchored on a Thai academic rank or a curated dictionary, not NER; see
  `src/rag_lab/loaders/{person,program,course,faculty}_loader.py`) writes
  `metadata['people'|'programs'|'courses'|'faculties']`, consumed by the
  `entity_tags` loader + `entity_lookup`/`entity_boost` retrieval modes.
  Full narrative: `docs/entity-extraction-and-gold-eval-log.md`.
  **2026-07-25**: fixed a real gap in `match_people` — the bare "อ."
  academic rank (below ผศ./รศ./ศ., what most special/part-time instructors
  are cited with) had no title pattern at all, so plain-Thai instructor
  names in several common table types were invisible to tagging regardless
  of language (commit `a4e250e`; +10% people-tags corpus-wide, +37% in the
  document type that surfaced it). **2026-07-26**: fixed a second gap,
  scoped via `/grill-me` against a full-corpus scan before writing any code
  — a name split across *adjacent* `<td>` cells (title+given name in one
  cell, surname in the next, joined by literal `</td><td>` markup with no
  `<br/>` or whitespace) was invisible too (commit `e1523b3`; 8 genuine
  cases found corpus-wide, all in "รับรองรายงานการประชุม" rank-correction
  tables — 7 new tags across 4 files after guarding against 2 false
  positives found in a different, OCR-corrupted table type). English-titled
  foreign-faculty names (Mr./Assoc.Prof.Dr.) are still unmatched — deferred,
  user judged low priority given how few foreign-faculty mentions exist.
  The one index built with the `entity_tags` loader
  (`data/index/entity_tags_full`) needs rebuilding after any
  `person`/`program`/`course` loader change for a fix to reach
  `entity_lookup`/`entity_boost` in the UI — both fixes above are reflected
  as of the 2026-07-28 rebuild (refreshed again 2026-08-05 to pick up the
  `resolution_id` fix below, since that index predated it).
- **Simple relation graph (`tools/corpus_prep/build_relation_graph.py` →
  `data/graph/relations.json` + `docs/relation-graph.md`, 2026-08-10)** — the two
  edges that need no new model, no GPU and no new dictionary: **A**
  `program —belongs_to→ faculty` and **A′** `person —affiliated_with→ faculty`
  (the inline `(สังกัดคณะX)` shape). Edges B (`person→responsible_for→program`)
  and C (`person→replaces→person`) are deliberately out of scope. **No gold query
  in either set is multi-hop, so this is a capability the current eval is
  structurally unable to score — no retrieval gain may be claimed**, and every
  denominator is an entity *dictionary*, itself a curated subset, so read the
  coverage as "of what was found". Tags are recomputed from the tested matchers
  rather than read from `academic_resolutions/entity_tags/*_by_file.json`, which
  are dated 2026-07-17..25 and predate the person-loader fixes, the 07-28 OCR
  remediation, the 08-08 title repair and the 08-09 re-OCR — building on them
  would be this project's signature two-artifacts-from-different-days failure.
  One corpus walk (~22 min, 2,854 files) caches its evidence to
  `data/results/relation_graph_raw.json` so `--render` re-derives graph and report
  free. **Edge A: 170 of 253 programs resolved, 60 ambiguous, 23 no_evidence**
  — and the `ambiguous` bucket is **not one thing**: only **9** have two faculties
  genuinely pointing at each other, the other **51** have a single faculty with
  fewer witnesses than `min_votes=2`. Two distinct causes underneath, and the
  first is the important one: **`program → faculty` is not a function** —
  `วิศวกรรมเครื่องกล` really is offered by both `คณะวิศวกรรมศาสตร์` and
  `วิทยาเขตชุมพรเขตรอุดมศักดิ์` (23 vs 20 votes), so any graph forcing one faculty
  per program is wrong for that group by construction. The second is a matcher
  finding worth its own ticket: **`match_programs` has no "matches nothing" exit
  for a near-miss**, so a corpus name absent from the dictionary is absorbed by
  its nearest neighbour — `หลักสูตรทันตแพทยศาสตรบัณฑิต` *and*
  `หลักสูตรพยาบาลศาสตรบัณฑิต` both match `หลักสูตรแพทยศาสตรบัณฑิต`, which accounts
  for that whole ambiguous row. Scope was **measured, not guessed**: 0 of 253
  dictionary names collide with each other, so the collision is only with names
  outside it. **Not fixed here** — `match_programs` is also read by
  `build_gold_candidates.py` and `router`, so moving the threshold would move
  published numbers. **That blast radius is now MEASURED and it is zero on both
  call sites (2026-08-10, `tools/eval/audit_program_matcher_absorption.py` →
  `docs/program-matcher-absorption.md`, ~23 min walk cached so `--render` is
  free).** Corpus-wide the defect is large — 9,141 accepted matches over 1,710
  files, **23.1% (2,114) absorb a genuinely different name**, 210 of 249 matched
  canonicals absorb at least one — and its dominant shape is the one worth
  naming: **35.7% of absorptions swap the degree level** (บัณฑิต ↔ มหาบัณฑิต ↔
  ดุษฎีบัณฑิต), one token apart so the ratio stays far above 0.82 while a
  master's programme is tagged as the bachelor's of the same subject. **But it
  reaches neither published path, and both were verified rather than assumed**:
  `program_candidates()` never calls `match_programs` for membership — it seeds
  from tagged files then gates on `canonical in resolution_id`, an exact
  substring of the manifest title, and **0 of 30** program queries' gold pairs
  have a `resolution_id` failing to contain the program; and `classify_query`
  asks only whether *any* program matched, never which one, so a name-for-name
  swap **cannot** change a route (33/13/30/30 exact, 0 program queries routed
  elsewhere). So the ticket closes as *documented, not repaired* — and if it is
  ever repaired, the degree-level swap is the shape to fix first. **Its first
  run was wrong in a way worth remembering**: it reported "99.3% absorb a
  foreign name" with the inserted-character distribution's mode at exactly
  **4** — which is `_WINDOW_SLACK`, i.e. it was counting the matcher's own
  read-ahead window as absorbed text
  ([[feedback_a_mode_on_a_constant_is_your_instrument]]). `S5` now pins that a
  pure window tail scores 0 (6,333 such spans) while a longer tail still
  registers its excess. **Edge A′ is ~7x smaller than the scan note claimed and the
  note is corrected**: `สังกัดคณะ` was recorded as appearing in "1,465 files
  (51%)"; direct counting over the 2,854 live files gives `สังกัด` in **209**
  files and `สังกัดคณะ` in **73**, and no alternative anchor supplies a larger
  person→faculty source (`จากคณะ` 1,083 is `คณะกรรมการ` boilerplate). Anchored
  extraction accepts 206 of 435 marker occurrences, attaches a person to 71, and
  yields **66 people with an edge across 86 files (3.0%)**. **Read its
  `resolved` 100% as "nothing contradicts it", not as quality** — 62 of 66 rest
  on a single witness. **The finding that matters more than the count is what the
  marker *means*, and it was measured rather than assumed**: the parenthetical is
  written **100% of the time (64 of 64 determinable) for a person from a
  *different* faculty than the document's own**, i.e. it is a cross-appointment
  disambiguator. So A′ is a **biased** sample of people, and deriving
  person→faculty from plain document co-occurrence the way edge A does would be
  wrong *with a direction*, not merely noisy. Two independent cross-checks on A,
  reported and never gated: manifest title vs OCR'd body agree on 158 of 169
  programs both can name (two independent text sources — typed vs scanned), and a
  split-half over disjoint document sets agrees on 103 of 112. Four self-checks
  (S1 every faculty node is canonical, S2 `no_evidence` stays *undefined* rather
  than a low score, S3 a window-extracted faculty must also appear in the
  document's own tags, S4 every A′ name must be one `find_people` found in that
  same file) all pass; **S1 first reported a false FAIL from operator precedence**
  (`{a} | {b} - dict` parses as `{a} | ({b} - dict)`), and **S2's first version was
  vacuous** because the graph is built by iterating the dictionary, so "the buckets
  add up" was true by construction — it now gates on the buckets staying
  *distinguishable*. `docs/relation-graph.md` is in `audit_doc_claims.py`'s
  `ARTIFACT_FILES` so its figures can be cited in prose.
- **Query routing** (`src/rag_lab/router.py`'s `classify_query` + `ROUTE_COMBO`,
  driven by `query_service.route_query`): classify a query by shape and retrieve
  against that route's index only. Validated offline by `tools/eval/routing_eval.py`
  against the 106-query 73det set on persisted results (no retrieval) →
  `data/results/routing_eval.md`. **Add a route whenever the Gold set gains an
  entity_type** — the failure mode here is silent: the router shipped 2026-07-17
  with 3 routes (person/program/unmatched), the set gained 33 `course` queries eight
  days later, and nothing broke, they just fell to the `unmatched` default; together
  with the 13 never-covered `faculty` queries that was **46/106 = 43% of the set
  silently unrouted for three weeks**. `tests/test_router.py` now pins the structural
  invariant (every per-retriever map's key set == the set of routes `classify_query`
  can return), so a route with no target is a test failure rather than a query-time
  `KeyError`.
  Closed 2026-08-08: 5 routes, 0/106 unrouted, classification exact per type with no
  cross-firing. **Cite the two results separately**: the 5-route router significantly
  beats the 3-route one (**+0.0958** dense recall@10, Holm-adj p=0.0000, m=18 —
  and this margin is *invariant to the target refresh below*, since both arms hold
  the same person/program targets so the only difference between them is coverage),
  but **no deployable** routed arm significantly beats just using the best single
  combo for everything (shipped +0.0481, p=0.1548; LOO +0.0349, p=0.3568 — both ns).
  The one arm that *does* clear the bar is `routed (oracle)`, significant on all
  three dense metrics (+0.0586 recall@10 p=0.0462, +0.0642 MRR p=0.0160, +0.0701
  nDCG@10 p=0.0036) — **but an oracle is not a system**: read it as the headroom a
  perfect per-route map would have, real but small. So the claim is *matches a
  well-chosen single index without knowing which one, and closes a 43% coverage
  hole*, not *beats it*.
  Under hybrid the gain shrinks to ns (+0.0408, p=0.1152) because BM25 partly rescues
  the misrouted queries. Ordering inside `classify_query` is load-bearing: course is
  checked **ahead of both program branches** because the program route's ConGen
  embedder scores **0.0000** recall@10 on course queries. **Both structural facts
  that were open here are now closed in code (2026-08-08).** (1) The best target per
  route is **retriever-dependent** (person = semantic+qwen3 dense, sentence+bge_m3
  hybrid), which one flat dict can't express — so `ROUTE_COMBO` became
  `ROUTE_COMBO_BY_RETRIEVER` (`dense`/`hybrid` maps) behind a
  `route_targets(retriever_type)` accessor; `route_query` resolves it from
  `retriever_spec.type`, and unmeasured retrievers (bm25/entity_lookup/qdrant) fall
  back to the hybrid map — an extrapolation, labelled as one in the source, not a
  measurement. `ROUTE_COMBO` still exists as that fallback alias. (2) The stale
  `person`/`program` targets were **refreshed** under a stated adoption rule rather
  than an argmax: adopt only when the LOO selector picks the same target in ≥29/30
  folds. person → semantic+qwen3 (dense) / sentence+bge_m3 (hybrid), program →
  fixed_size+qwen3_0.6b / semantic+qwen3_0.6b; `course` was already the argmax and
  `faculty` **deliberately stays** at the unmatched default (3 distinct dense LOO
  targets over 13 folds; hybrid gap only +0.0305; n=13 is inside the embedder
  family's own MDE). **The refresh made the `shipped` arm less honest, not more**:
  4 of 5 targets are now chosen on the 106 queries it is scored on, so it sits near
  `routed (oracle)` by construction (dense 0.6189 vs 0.6293) — **cite `routed (loo)`
  as the generalisation estimate**, and note it is *unchanged* by the refresh
  (+0.0349 dense / +0.0499 hybrid, both ns) because it never read the constants.
  That is the cleanest statement of what the refresh bought: it raises the shipped
  router to what LOO already predicted, rather than creating new gain. The
  `unmatched_strategy="rrf"` branch is now unexercised by any eval (0/106 unrouted)
  and its old numbers (`t=0.59`, "+15% MRR") came from the retired 252-query/3-route
  eval — withdrawn, don't cite them.
- **Hybrid fusion weight (`alpha`)** — swept 2026-08-08 at last
  (`tools/eval/hybrid_alpha_sweep.py` → `data/results/hybrid_alpha_sweep.md`;
  21-point grid × 106 queries × 3 combos). Every hybrid number before this date was
  at an implicit, unswept 50:50. **`alpha` is applied to the `rrf` branch**
  (`Σ wᵢ/(k+rankᵢ)`, weighted RRF), *not* to the separate `weighted` score-fusion
  branch — sweeping that one would confound the weight with a switch from rank
  fusion to score fusion. The payoff of that choice: a uniform 0.5× factor cannot
  reorder, so **alpha=0.50 is rank-order-identical to plain RRF** and every
  published number is reproduced exactly at the grid's midpoint
  (`tests/retrievers/test_hybrid_retriever.py` pins this as a regression guard).
  Findings: (a) **a single global alpha is worth nothing** where 0.50 was already
  sane (+0.0016 / +0.0189 recall@10, both ns, and both are *oracle* values fitted on
  the test set); (b) **a per-`entity_type` alpha is worth +0.0350 recall@10 /
  +0.0360 nDCG@10 and survives leave-one-out** (Holm-adj 0.0252 / 0.0210, m=9, on
  `sentence+qwen3_0.6b`) — **MRR is ns, don't include it**; (c) the per-type optima
  are so far apart that `person` (best 0.15, plateau 0.00-0.35) and `program` (best
  0.75, plateau 0.40-1.00) have **disjoint** non-degrading ranges and the shipped
  0.50 sits *outside* `person`'s; (d) **the gain is conditional** — it needs the two
  arms' relative strength to *invert* across query types, so `semantic+bge_m3` gains
  nothing (ns everywhere; it is the `person` specialist, its dense arm has no
  per-type weak spot) and `fixed_size+m2v` wants alpha=0.00 outright (drop the
  broken arm; per-type adds only +0.0105 over global). Report **ranges, not a single
  best value** — tuning alpha on the 106 queries it is reported on is overfitting,
  which is what the LOO arm exists to bound. Nothing is changed in shipped defaults;
  `HybridRetriever` still ships 0.5/0.5. **Decided 2026-08-08 not to wire a
  per-`entity_type` alpha into `query_service` at all**, and the reason is a
  wrong-pair trap worth remembering: the motivating +0.0350 is measured against *no
  routing*, which stopped being the shipped configuration the same day. Against the
  hard router that now ships it shows **no gain on any metric** (recall@10 −0.0202,
  MRR +0.0182, nDCG@10 +0.0066, all ns, m=12) and the entire remaining headroom is
  the oracle gap **+0.0071**. Mechanism, so this isn't read as a power problem:
  per-type alpha repairs a per-type weak dense arm, and hard routing already hands
  each route a specialist index that doesn't have one (`person` alpha* moves
  0.15 → 0.30, *toward* neutral, once routed). **The one branch that flips it** is
  deployment cost: if 5 indices is too many, the move is to *replace* hard routing
  with soft (arm B, one index, 0.6631, ns vs hard) — a cost decision, not an
  accuracy one. Never both. Two things worth reusing: the sweep caches
  each arm's rank vector once and re-fuses in numpy (21 alphas for the cost of 1
  retrieval pass, since the ~1.9s/query `BM25Okapi` rebuild dominates), and its
  self-check pins the vectorised fusion against the **real** retrievers at all three
  anchored grid points (0.00=BM25-alone, 0.50=hybrid, 1.00=dense-alone) — that check
  caught a genuine tie-break bug (`HybridRetriever` settles exact score ties by dense
  rank, via stable `sorted` over a dense-first dict; a naive `argsort` settles them
  by chunk index).
- **Soft vs hard routing (2026-08-08, `tools/eval/soft_vs_hard_routing.py` →
  `data/results/soft_vs_hard_routing.md`)** — the two routing results above were
  measured on different axes and are now compared directly: 4 arms, each retrieving
  k=10 from **exactly one index per query** (equal budget), index choice held at its
  shipped value, alpha the only fitted quantity (LOO within route), routing by
  `classify_query`. **READ THE DATE ON ANY NUMBER FROM THIS SCRIPT.** It was first
  run against `ROUTE_COMBO`'s 2026-07-17 targets and reported soft ≥ hard; those
  targets were refreshed the same day (see the routing bullet above), the script
  re-run, and **the verdict flipped** — hard routing had been judged on a `program`
  target that actively hurt. The soft arm never moved. Post-refresh: **hard**
  (per-route index, 5 indices) 0.6831 recall@10 > **soft** (per-route fusion weight,
  1 index) 0.6631 > **neither** 0.6281. **Two significant cells, one per mechanism,
  on different metrics**: `hard vs none` recall@10 +0.0549 (Holm-adj 0.0242, m=12)
  and `soft vs none` nDCG@10 +0.0360 (Holm-adj 0.0216). **Soft vs hard is ns on all
  three** — CI rules out soft beating hard by more than 0.0156 recall@10, and hard
  beating soft by more than 0.0575. So: hard leads numerically everywhere and owns
  the only significant recall@10 result, but it is **not shown to beat soft**, and it
  costs 5 indices to soft's 1. The cost-per-point argument for soft survives the
  flip; "soft is at least as good" does not. Arm C reproduces `routing_eval.md`'s
  hybrid `routed (shipped)` to 4 decimals (0.6831) from an independent code path.
  Note arm C's targets are now fitted on this same set, so cite `routing_eval.md`'s
  `routed (loo)` (0.6780) as the hard arm's generalisation estimate — still above
  soft. **Still substitutes, not complements, but for a sharper reason**: doing both
  (0.6629) is *below* hard alone and `D vs C` is negative (−0.0202, CI excludes zero,
  ns after Holm) — yet at the oracle bound D′ (0.6901) is the best arm in the table.
  There *is* a sliver of headroom for alpha on top of routing, and LOO fitting costs
  more than the sliver is worth (the pre-refresh version had D worse than B even at
  the oracle, i.e. no headroom at all). Per-route, **hard now wins every route**
  (person +0.1044, program +0.0440, course +0.0316, faculty +0.0253) where before it
  won only course and faculty and *lost* `program` by −0.0784 — that one route was
  most of the old verdict. The `person` row still gives the mechanism: optimal alpha
  is **0.15** on the generic index (hand it to BM25, 0.8147 there) but **0.30** on the
  routed index, whose target *is* the person dense specialist; both mechanisms repair
  the same per-type weak dense arm. **Family-size
  trap, worth reading before citing:** this script's arms A/B reproduce
  `hybrid_alpha_sweep.py` to 4 decimals from an independent code path, yet the
  `recall@10` **verdict** differs (Holm-adj 0.0252 at m=9 there, 0.0580 at m=12 here).
  Cite the sweep's m=9 for "is a per-route alpha worth anything"; cite this table's
  m=12 only for its own four comparisons.
- `strip_course_comparison_tables` (`src/rag_lab/loaders/common.py`, commit
  `71764a8`) compacts old/new course-comparison tables (code + credit-tuple
  + English description, the corpus's single largest chunks — 17,077 chars
  in one document) to `CODE Title` lines, dropping the description. The old
  blocker was "needs a thematic-inclusive eval first", since course-name gold
  queries never touch description text. That eval closed the question the
  other way instead: only **13 files corpus-wide** contain such a table
  (0.46%, 39.8% of their text removed), cited by **0 of 106** 73det queries
  and **3 of 179** thematic ones — no gold set can detect a
  description-stripping regression, so waiting for an eval was waiting for
  evidence that cannot arrive. User resolved it on domain grounds instead:
  **course descriptions are not what people ask about**, so the unmeasurable
  regression is also the unimportant one. **Wired into `PlainLoader.load()`
  2026-08-03** (ahead of `strip_mapping_tables`, before `match_courses` per
  the ordering constraint below) and rode rebuild #3 (completed
  2026-08-05T07:56) — live in `chunker_compare_full` now, not a pending item.
  Note the ordering constraint: run it *before* `match_courses`, which it
  improves. Narrative: `docs/chunker-embedder-comparison-log.md` (course-table
  compaction section).
- Chunker/embedder/BM25/hybrid comparison eval lives in `tools/eval/`. Current (9-embedder)
  scripts: `run_gold_chunker_eval.py`, `run_gold_bm25_eval.py`, `run_gold_hybrid_eval.py` +
  `run_gold_hybrid_eval_9way_new.py`, `embedder_matrix_9way.py` (retrieval + breakdown +
  aggregate significance in one script — defines the `(type, model_name)`-keyed embedder
  labels and superseded-combo exclusions every other 9-way script imports),
  `embedder_significance_test_by_entity_type_9way.py`, `bm25_vs_embedder_significance_test_9way.py`,
  `hybrid_significance_test_9way.py`. (Originals without the `_9way` suffix cover only the
  first 6 embedders and are superseded but kept for reference.) Scores against the Gold query
  set `config/eval/gold_query_set_73det.yaml` (73 deterministic queries — use this one, not the
  252-entry `gold_query_set.yaml`, which dilutes results with low-discrimination thematic
  queries — and the reason they don't discriminate is now known: all 179 were
  meeting-scoped but never named the meeting ("ในการประชุมครั้งนี้"), so they were
  unanswerable as posed. `tools/eval/qualify_thematic_queries.py` rewrote all 179 to name
  their meeting 2026-07-30 and `run_thematic_eval.py` re-evaluated them, which changed the
  reason to keep them apart: they carry signal that points **the opposite way** on the
  chunker axis (fixed_size − semantic is **+0.0258** thematic vs **−0.0363** entity-anchored;
  `semantic` is the *worst* chunker on thematic and the best on entity-anchored), so pooling
  the sets cancels two real effects instead of diluting one. Still low-powered per pair
  (2/27 significant, 62% ties) — cite them as a separate query shape, never averaged in.
  Side-by-side: `tools/eval/thematic_vs_deterministic.py`. **The BM25/hybrid arms
  (`hybrid_significance_test_9way.py --thematic`) reverse this project's most
  robust finding and are the bigger result**: BM25 is weak on thematic (0.2990 vs 0.4930
  entity-anchored — no name to match exactly), so "hybrid beats dense-alone for every
  embedder" is **entity-anchored-specific**. On thematic recall@10 it is 3 significant for
  hybrid / 4 ties / **2 significant against** (`e5` −0.0449, `qwen3` −0.0516), and the
  hybrid−dense delta is monotone in dense strength (**r = −0.921**). What sits at BM25's own
  score is the **significance boundary** (every embedder below it is significantly helped, none
  above it is), *not* the sign flip — the point estimate crosses zero near 0.40, between
  `bge_m3` and `jina_v5`. (An earlier "flipping sign right at BM25's own score" phrasing here
  and in `paper-results-summary.md` was never supported by its own table; corrected
  2026-08-07.) General rule, now measured in both directions and subsuming the old
  m2v/sct "RRF failure case": **RRF helps the weaker arm and taxes the stronger one — fuse
  only when the two arms are comparable, whichever one is weak.** Report:
  `data/results/thematic_hybrid_significance_test.md`. **Whole thematic arm re-run
  2026-08-07 against rebuild #3** (dense+BM25+hybrid, 5h12m, exit=0): **0 verdict flips**
  across all 81 significance cells, every effect size moved <0.02 — the numbers above are
  current, and the old "indices predate the 2026-07-30 corpus fixes" caveat is discharged). Full process narrative: `docs/chunker-embedder-comparison-log.md`; clean
  citation-ready numbers for paper-writing: `docs/paper-results-summary.md` (update this one
  whenever a headline number changes — the log stays append-only). **Refreshed 2026-07-25**
  for the corpus-discovery contamination fix (0% contamination, see
  [[project_corpus_discovery_contamination_bug]] / `docs/chunker-embedder-comparison-log.md`,
  fix `8c86b63`/`b36f96f`/`dd0c0ae`, rebuild commit `2d36663`) — every conclusion held through
  that refresh. **Refreshed again 2026-07-29**, this time for a full second index rebuild
  (2026-07-28, unrelated OCR-remediation fix, [[project_reocr_remediation_pipeline]]) — and
  this refresh genuinely changed several conclusions below, not just numbers. A stale-cache
  incident is worth recording as its own lesson: `embedder_matrix_9way.py` recomputes
  dense-alone retrieval fresh every run, so dense-alone numbers refreshed automatically and
  correctly — but the BM25/hybrid persisted-results scripts
  (`run_gold_bm25_eval.py`/`run_gold_hybrid_eval.py`) were not re-invoked for 3 days, so
  every BM25/hybrid significance test kept silently comparing fresh dense numbers against
  stale (pre-rebuild) BM25/hybrid numbers until an implausible-looking batch of "flipped"
  findings prompted an mtime check. **Lesson: after any index rebuild, refresh every
  retrieval path with persisted results (dense, BM25, hybrid) — not only whichever one an
  eval script happens to recompute automatically.** Full incident:
  `docs/chunker-embedder-comparison-log.md`, "Re-eval หลัง OCR-remediation rebuild" entry.
  Current bottom line (2026-07-29, bootstrap + Holm-corrected, all 9 embedders,
  OCR-remediation-rebuilt indices):
  **the "`semantic` chunking wins" headline this project has repeated since the first
  comparison round does not survive being significance-tested, and should be retired.**
  `semantic × qwen3_0.6b` recall@10 dropped from a stale 0.7048 to a fresh **0.6152**, no longer
  even the top chunker for that embedder numerically (`sentence` 0.6265, `fixed_size` 0.6154) —
  which prompted building the missing test
  (`tools/eval/hybrid_chunker_significance_test.py`, chunker-vs-chunker at a fixed
  embedder+retriever, one family per embedder + an aggregate family across all 9). **Result:
  `semantic` never significantly beats any other chunker, anywhere** — not for `qwen3_0.6b`
  (all 4 chunkers fully tied, Holm-adj p≥0.44) nor in the aggregate (`recursive` is now
  numerically highest at 0.5291 recall@10 vs. `semantic`'s 0.5206, not significant either). The
  *only* significant chunker-pairwise result in the whole test is `fixed_size` losing to
  `recursive` (aggregate nDCG@10 + several individual embedders). **Revised framing:
  `recursive`/`semantic`/`sentence` are a statistically tied top cluster with no provable
  winner; `fixed_size` is the one demonstrated laggard.** `semantic` is still a perfectly
  reasonable default (never proven worse than anything, and still the one chunker where a
  strong dense embedder demonstrably beats BM25, see below) — just no longer citable as "the
  best chunker." Cross-chunker-averaged
  hybrid recall@10 (`qwen3_0.6b` 0.6167, `qwen3` 0.5945, `jina_v5` 0.5831, `e5` 0.5753,
  `bge-m3` 0.5730, `e5_small` 0.5658, `congen` 0.4692, `sct` 0.3939, `m2v` 0.3028). **The
  dedicated semantic-only top-5 pairwise tie test
  (`tools/eval/hybrid_significance_test_semantic_top5.py`) was re-run 2026-07-29** — the tie
  **partially broke**: `bge-m3` now loses significantly to `qwen3_0.6b`/`qwen3`/`jina_v5` on
  recall@10, and to `qwen3_0.6b`/`qwen3` on nDCG@10 (still ties all three on MRR, and after the
  2026-07-30 `resolution_id` relabel it also ties `jina_v5` on nDCG@10 — Holm-adj p=0.0928, the
  one verdict that relabel narrowed), so it drops out of the cluster. The
  remaining four (`qwen3_0.6b` 0.6152, `qwen3` 0.6051, `jina_v5` 0.5995, `e5_small` 0.5877
  recall@10, semantic-only) are still fully, mutually tied on every metric. Don't cite a single
  "best combo" among those four — the tie there is confirmed, not provisional. Hybrid still
  significantly beats dense-alone for essentially every one of the 9 embedders on every metric
  (still 26/27 tests significant — same count, but **the sole exception moved from `qwen3` on
  MRR to `qwen3_0.6b` on MRR**, Holm-adj p=0.30; `qwen3` is now significant on MRR too) — still
  the most robust finding of the comparison. Hybrid vs. BM25-alone shifted more: `qwen3_0.6b`,
  `qwen3`, `jina_v5`, `e5`, `bge-m3`, `e5_small` all significantly beat BM25 on recall@10 now
  (`jina_v5` newly clearly significant, was borderline before); `congen` dropped **out** of that
  group (BM25 itself got stronger post-OCR-fix, closing the gap); `sct`/`m2v` remain the
  cautionary cases where hybrid is significantly worse than BM25-alone, and **`sct`'s recall@10
  deficit is significant again** (reversing the 2026-07-25 "no longer significant, p=0.08"
  finding — that finding was itself measured against the since-superseded index). The
  cross-chunker **dense-alone 3-way tie at the top is broken**: `qwen3_0.6b` now significantly
  beats both `bge-m3` (+0.1173 recall@10) and `Qwen3-Embedding-4B` (+0.0486) in aggregate — this
  was always computed fresh (unaffected by the BM25/hybrid staleness bug), so it's a genuine
  finding, not an artifact. The per-entity_type specialist/weak-spot pattern underneath the old
  aggregate tie is **unaffected** by this (separate, already-fresh dense-alone test): `bge-m3` =
  person-query specialist, `Qwen3-4B` = only embedder with no provable weak spot across both
  main categories (ties bge-m3 on person AND ties ConGen/qwen3_0.6b on program), `Qwen3-0.6B` =
  now aggregate-leading but still has a real person-query weak spot `Qwen3-4B` doesn't,
  `ConGen-PhayaThaiBERT` = program-query specialist. BM25 alone (`retrievers/bm25.py`) no
  longer ties the top dense tier the way it used to — it now **significantly beats `bge-m3`**
  (aggregate recall@10 rose 0.3908→0.4930 post-OCR-fix, more than any embedder's own gain,
  because lexical matching is far more sensitive to OCR token corruption than dense embeddings)
  and still significantly beats every weaker embedder; it statistically ties only `qwen3` and
  `qwen3_0.6b` (and borderline-ties `jina_v5`, Holm-adj p=0.053). The per-chunker breakdown
  (`tools/eval/bm25_vs_embedder_significance_test_per_chunker.py`, re-run 2026-07-29) still
  shows `bge-m3` losing to BM25 significantly under `sentence` chunking specifically, and now
  additionally shows `qwen3_0.6b`'s numerically-negative BM25 margin under `semantic` chunking
  has become **statistically significant** (Holm-adj p=0.006) — the first cell in this whole
  comparison where an embedder significantly beats BM25 outright, reinforcing `semantic` as the
  one chunker where a strong dense embedder demonstrably earns its cost over free BM25. Don't
  naively RRF a weak embedder with BM25: `m2v` and `sct` both still significantly *hurt* vs. BM25
  alone on recall@10/MRR/nDCG@10 (all three, both models, post-refresh) — a real RRF failure
  mode whenever the fused dense signal is weak enough. Cost/latency:
  `tools/eval/cost_latency_pareto.py` (vector dim, index size, query latency p50/p95) found
  `HybridRetriever.retrieve()` and `BM25Retriever.retrieve()`'s current implementation
  (full-corpus `k=n` fetch before fusing, `BM25Okapi` rebuilt from scratch every query) adds a
  roughly **fixed ~1.9-2.0s of overhead to every hybrid query, nearly independent of embedder**
  (it scales with corpus size, not embedding dim) — the ~2.1-2.7s measured figure is mostly this
  avoidable per-query overhead on top of a ~116-668ms intrinsic cost, not RRF fusion itself.
  **Half of that overhead is now gone, and the measurement that removed it split the two causes
  the sentence above had bundled (2026-08-09).** `BM25Retriever` memoises its `BM25Okapi` on the
  `Index` (`Index.lexical_scorer`) instead of rebuilding it per query. **Quote that saving in
  seconds, not as a multiple of `get_scores`** — `rank_bm25` loops over query *terms* in Python,
  so scoring is linear in query length (~12 ms/token over 74,816 chunks) while the build is not.
  The **26.2x / 1.073s vs 0.041s** first published here was measured on a **3-token synthetic**
  query and is withdrawn as a headline; re-measured 2026-08-09 on the **real 20-token-median
  Gold queries** the project evaluates (n=106, min 13 max 30): build **1035.89 ms** vs
  `get_scores` **253.50 ms** = **4.1x**, and **BM25-alone `retrieve()` p50 is 234.45 ms**
  (p95 332.78). The ~**1.0s** removed from every query is the part that does *not* depend on
  query shape, and that is the number that transfers. **Hybrid goes 2.269s → 1.361s
  (1.7x, −0.907s)** — measured paired, both arms in one process against one loaded index, because
  [[feedback_check_benchmark_position_drift]]. That gap is the finding: the **remaining ~1.36s is
  the `k=n` over-fetch**, i.e. materialising ~75k `RankedChunk` objects per arm and fusing them in
  Python, which is a *separate* and still-open cost. Do not read "the rebuild is fixed" as "the
  hybrid overhead is fixed". The over-fetch is **not** free to remove either: `HybridRetriever`
  fetches k=n so RRF sees complete rankings, so truncating it would change results, unlike this
  change which cannot. **The stale-latency consequence is DISCHARGED (2026-08-09): the whole
  script was re-run on an idle machine** and `cost_latency_pareto.md`'s BM25/hybrid columns are
  current (dense p50 120-840 ms, hybrid p50 1.21-1.86s). `docs/paper-results-summary.md` was
  updated with it, so its old split provenance (07-29 latency / 08-07 quality) no longer applies
  to the latency half.
  **The over-fetch is now MEASURED, and it is a quality/latency trade rather than an
  optimisation (2026-08-09, `tools/eval/hybrid_fetch_depth_sweep.py` →
  `data/results/hybrid_fetch_depth_sweep.md`).** `HybridRetriever` gained a `fetch_depth`
  knob whose default `None` computes `depth = len(index.chunks)` — literally the old k=n
  expression, so every published hybrid number is reproduced *by construction* and
  `tests/retrievers/test_hybrid_retriever.py` pins it; the sweep runs F ∈ {10 … 10,000, n}
  over 36 combos × 106 queries = 3,816 pairs. **The two questions have opposite answers, and
  that is the finding.** *Is the ranking the same?* — only at F=n: **F=10,000 reproduces just
  88.00%** of top-10s in order (96.67% as a set) and F=1,000 only 70.02%. The pre-registered
  guess "F=1000 will be identical" was **wrong**, recorded as such. *Does it matter?* —
  barely: macro recall@10 across the 36 combos is 0.5204 at k=n, **0.5167 at F=100
  (−0.0037)** and **0.5171 at F=200 (−0.0033)**, and it is **non-monotonic** (F=500's −0.0018
  is better than F=1,000's −0.0026) because truncation lifts different chunks' scores at
  different rates as F grows. Mechanism worth keeping: a chunk inside dense's top-F but past
  BM25's cut loses its BM25 term **outright**, not by a little — that is why this is not an
  approximation that merely loses precision. Damage concentrates exactly where this project's
  RRF rule predicts (worst combo at F=50 `semantic × e5_small` −0.0595, at F=200
  `recursive × bge_m3` −0.0224, at F=1,000 `sentence × sct` −0.0145), and **`person` queries
  *gain* at F=50 (+0.0212)** — the only entity_type that does, consistent with BM25 carrying
  `person` (0.8147) while the cut deletes a weak dense arm's tail. **Timing (paired, one
  process, one loaded index, arms alternated per query, BM25 scorer pre-warmed so its one-off
  build lands in neither arm, `plain__sentence__qwen3__ff8f6c49`): k=n p50 1089.5 ms → F=200
  **417.9 ms** (−0.672 s, 2.6x), F=1,000 421.0 ms.** So the over-fetch is **~62% of hybrid
  query time**, and the ~0.42 s left is real scoring work (dense encode + gemv + `get_scores`)
  that no depth cut can touch — **do not read the earlier "the remaining ~1.36s is the k=n
  over-fetch" as all removable**; that sentence bundled the residual in. The trade on the
  table was ~0.67 s/query for −0.0033 macro
  recall@10 at F=200 — a *cost* decision of the same shape as soft-vs-hard routing, and it
  needed re-measuring against the hard router (which now ships) before adoption, since
  that macro figure is an average over a whole combo family, not a system result.
  **That re-measurement is DONE and the decision is made — see the next bullet.** Two method
  notes: the sweep replicates the **truncated** tie-break (the fusion dict is filled
  `dense[:F]` first, then the BM25-only remainder in BM25 rank order, so equal RRF scores stay
  dense-first — the same trap `miss_depth_profile.py` documents at full depth), and **S5
  checks the numpy fusion against a real `HybridRetriever(fetch_depth=F)`**, added because the
  first version anchored only F=n, where the mechanism under test is inert and the check would
  have passed identically had `fuse_at_depth` ignored F. The `weighted` branch truncates too
  (a cut chunk's normalised score reads 0, a harsher approximation) and is **unmeasured**.
  Everything the report renders is cached in `hybrid_fetch_depth_raw.json`, so `--render`
  reproduces it without a GPU.
- **`fetch_depth` against the shipped router, and the ship decision (2026-08-09,
  `tools/eval/routed_fetch_depth_test.py` → `data/results/routed_fetch_depth_test.md`,
  ~2.5 min quality + ~3 min latency).** The sweep above left one blocker: its
  −0.0033 is a macro over 36 combos retrieving with **no router**, and hard routing has
  shipped since 2026-08-08 — the exact wrong-pair trap that killed per-`entity_type` alpha
  and rrf4. Re-measured on the 106 queries routed by `classify_query` to their 4 shipped
  indices, **the trade gets better on both sides**: pre-registered F=200 vs k=n (3 metrics,
  Holm m=3) is recall@10 **+0.0005**, MRR −0.0024, nDCG@10 −0.0022, **all Holm-adj 1.0000**,
  and latency **1193.9 → 475.6 ms p50 (−0.718 s, 2.51x)**, paired on the index each query is
  actually routed to. **Note the null points the other way here than in those two cases** —
  they had to *win* and a null killed them; a depth cut only has to not *lose*, so the null
  is what licenses shipping — which is exactly why it must be **cited as a bound**: the CI
  rules out a loss worse than **0.0078** on the worst of the three metrics. **The
  pre-registered prediction was confirmed and it is the part that carries the mechanism**:
  unrouted, `person` is the one entity type that *gains* from a shallow cut (+0.0202 at F=50)
  because BM25 carries it while the cut deletes a weak dense arm's tail; routing already
  hands `person` its dense specialist, so the gain should shrink — it **reverses** to
  −0.0207. Also worth knowing: routed damage at *shallow* F is **worse** than unrouted
  (F=10 −0.0705 vs −0.0480) because routing raised the baseline there is more to lose from,
  yet routed rankings are *more* stable (84.0% of top-10s identical at F=200 vs 66.0%) —
  don't assume the unrouted damage curve transfers in either direction.
  **DECISION: wired at the query-time layer, NOT as the class default.**
  `app/streamlit_app.py` sets `fetch_depth` per query through `StrategySpec` params
  (default 200, with 1000 and "whole corpus" selectable); `HybridRetriever.__init__` keeps
  `fetch_depth=None`, pinned by `tests/retrievers/test_hybrid_retriever.py`. The split is
  load-bearing, not a hedge: F=200 changes the top-10 on **17 of 106** Gold queries, so as a
  constructor default it would silently re-rank every future eval run while ~24k persisted
  results and every published table still said k=n — this project's signature
  silent-corruption shape. The UI is where 0.72 s is felt; the eval harness is where
  reproducibility is. Containment is checked, not assumed: `audit_pipeline_invariants.py`
  already classifies `mode_b`/`mode_b_routed` as write-only UI dirs, so nothing an eval
  reads can pick up an F=200 result. Anchors: S2 reproduces `routing_eval.md`'s
  `routed (shipped)` **0.6831** and S3 the unrouted **0.6281**, both exactly, from an
  independent code path; S4 is the live-mechanism check against a real
  `HybridRetriever(fetch_depth=F)` on a *routed* index, since S1-S3 exercise only F=n where
  truncation is inert ([[feedback_anchor_a_check_where_the_mechanism_is_live]]). The fusion
  itself is **imported** from `hybrid_fetch_depth_sweep.py` rather than reimplemented — two
  copies of that tie-break would eventually disagree.
  **Refreshed 2026-07-29** against the OCR-remediation-rebuilt indices: latency/cost mechanics
  came back essentially unchanged (confirms these measure model/index/corpus-size mechanics, not
  corpus content), but the recall@10 columns in the report dropped substantially like every other
  quality number in this section (e.g. `qwen3 × semantic` dense 0.6581→0.5382,
  `qwen3_0.6b × semantic` dense 0.6364→0.5688) — per Open item #13 above, semantic is not a
  provable "best chunker", so don't cite this report's recall numbers as a chunker-supremacy
  claim, only as one representative combo's cost/quality profile; report at
  `data/results/cost_latency_pareto.md`. **Re-run 2026-08-07 against rebuild #3, and
  the run split in two: quality adopted, latency rejected.** Quality barely moved
  (max |Δ| recall@10 **0.0034** over 18 cells, ordering identical), so those columns
  are now current. The latency columns were thrown out on evidence: `search p50` at
  dim=1024 is the same numpy op on the same-shaped array for 6 of the 9 embedders, and
  where those 6 agreed to within **1.9%** on 07-29 they spread **74.2%** here — split
  exactly at run position 6, everything timed before the 4B `qwen3` at 301-317ms and
  everything after it at 434-525ms (its memory isn't released before the rest of the
  loop is timed). Underneath that, a uniform ~1.25x floor shift, confirmed by
  re-running a standalone numpy benchmark on an idle machine afterwards (129ms,
  matching this run, not 07-29's 97ms). The tell was `m2v` appearing to cost more per
  hybrid query than `bge_m3` despite a 4ms encode. `docs/paper-results-summary.md`
  carried **deliberately split provenance** there — 07-29 latency, 08-07 quality —
  which was sound because latency measures corpus-*size* mechanics a rebuild doesn't
  change; **that split is now retired by the 08-09 re-run below.** One thing from the
  rejected run survives, since both terms of the ratio saw the same conditions: the
  BM25 build-vs-scoring ratio (22x there, 24x on 07-29) — but read it with the token
  count above, because those were 3-token queries and the honest figure on real
  queries is 4.1x. **The claim that "the k=n over-fetch tax is 66% of dense k=n cost
  in both runs" is WITHDRAWN**: on 08-09 it is **54%** (dense k=10 262.46 ms vs k=n
  575.58 ms, so 313 ms of over-fetch). It is not a constant of the implementation —
  quote it from the current run.
  **Re-run 2026-08-09 on an idle machine (task #28), and this run is the citable one.**
  Every embedder is timed in **its own subprocess** now, which removes the 74.2%
  position effect at its root (the 4B model's memory can't leak into the next
  embedder's timings if the process is gone). Three controls ship *in the report*,
  and the reason there are three is that the first one alone was not enough: (1) a
  **reference probe** — an identical numpy workload run in every child, which catches
  the CPU floor moving (median 156.6 ms, spread 13.5%); (2) a **repeat control** —
  the first embedder re-measured last, which caught what the probe could not, namely
  `bge_m3`'s own `search p50` rising **245.5 → 257.9 ms (+5.1%)** across a 45-minute
  run while the probe moved **−0.4 ms (0.3%)**; (3) **same-dim consistency** — the 7
  dim-1024 embedders do the same numpy op on the same-shaped array, so their spread
  (**10.3%**) *is* the noise floor, not a difference between models. Treat ~5-10% as
  this rig's resolution and don't read a smaller gap as real. The intrinsic-cost phase
  is now **cached** in `cost_latency_raw.json` alongside the per-embedder parts,
  because it wobbled ~15% between renders and a published figure has to be
  reproducible from the artifact that published it (`audit_doc_claims.py` D2 checks
  exactly that); two consecutive renders were verified byte-identical.
  **When re-running this script: idle machine, and check same-dim embedders at
  different loop positions before trusting any timing.** **Refreshed 2026-08-06** against rebuild #3
  (2026-08-05T07:56): `run_gold_bm25_eval.py`/`run_gold_hybrid_eval.py` turned out to
  have *already* been re-run the day before (2026-08-05, retrieval results in
  `data/results/gold_bm25_73det/`/`gold_hybrid_73det/` dated 08-05, discovered by
  mtime — not run by this session, cause unconfirmed but harmless), so this pass
  regenerated only the seconds-level downstream significance tests
  (`bm25_vs_embedder_significance_test_9way.md`, `hybrid_significance_test_9way.md`,
  `hybrid_chunker_significance_test.md`, `hybrid_significance_test_semantic_top5.md`,
  `bm25_vs_embedder_significance_test_per_chunker.md`,
  `bm25_hybrid_entity_type_breakdown.md`, `map_precision_significance_test.md`) against
  already-fresh data and diffed every verdict cell against the pre-refresh (2026-07-29/
  07-30) baseline rather than eyeballing. **Every aggregate/headline claim in this
  bullet and the next two survives untouched** — 0 verdict flips across
  `bm25_vs_embedder_significance_test_9way` (9 pairs), `hybrid_significance_test_9way`
  (54 pairs), and `bm25_vs_embedder_significance_test_per_chunker` (108 cells,
  including the `qwen3_0.6b`-beats-BM25-under-`semantic` cell). Four small, non-headline
  movements, all in the direction of *more* separation, not less: (1)
  `hybrid_chunker_significance_test`'s one citable pairwise result
  (`fixed_size` loses to `recursive`, aggregate nDCG@10) holds (Holm-adj p=0.0396, was
  0.0264); a few individual-embedder cells for that same pair flipped in both
  directions (`congen`/`m2v` lost significance, `bge_m3` gained a different
  significant pair) — doesn't change "no chunker beats another except this one cell";
  (2) the semantic-top-5 tie **sharpens**: `bge_m3` now also loses significantly to
  `qwen3`/`qwen3_0.6b` on MRR (previously its last tied metric), leaving it clearly
  outside the 4-way tied cluster on every metric, not just recall@10/nDCG@10; (3)
  `map_precision_significance_test`'s aggregate-scope precision@1 sharpens from
  `qwen3_0.6b` beating 4/8 to 5/8 (`e5_small` newly loses); MAP stays 4/8, so "8/8
  dense → 4/8 hybrid" below is unaffected; (4) `bm25_hybrid_entity_type_breakdown`
  numbers held within noise (see next bullet). **Thematic-query arm closed
  2026-08-07** — it was a separate gap because `data/results/thematic_{dense,bm25,hybrid}/`
  is populated by `run_thematic_eval.py`, not `run_gold_*_eval.py`, so nothing on 08-05
  touched it. Now re-run in full (all 3 retrieval paths, 5h12m, exit=0) with **0 verdict
  flips**; see the thematic paragraph in the bullet above. Use
  `tools/eval/diff_significance_reports.py` for this kind of before/after check — it keys
  rows on `(section heading, leading label cells)`, because these reports sort rows by
  effect size and reuse the same label across sections, so a positional diff or a naive
  label→verdict dict both give wrong answers.
- **Per-entity_type breakdown of BM25/hybrid (2026-07-29, refreshed 2026-08-06 —
  held flat, see caveat above —
  `tools/eval/bm25_hybrid_entity_type_breakdown.py`) gives the mechanism behind the
  hybrid win, and one caveat to it.** BM25 alone scores **0.8147** on `person` queries —
  beating every embedder's dense-alone person score (best `bge_m3` 0.5735) outright —
  while collapsing to **0.3497** on `program`, where dense nearly doubles it
  (`qwen3_0.6b` 0.6066). **BM25 carries person (exact name match), dense carries program**;
  that is direct evidence for the complementarity the Open item #2 proxies never
  established. Caveat: **"hybrid never hurts" is an aggregate claim, not a per-category
  one** — on `person` specifically hybrid sits *below* BM25-alone for most embedders
  (`qwen3_0.6b` 0.7264, `qwen3` 0.7342, `jina_v5` 0.7382), only `bge_m3` (0.8220) exceeding
  it. Measured against the structural ceiling, hybrid reaches 84.2% on `person`, 72.5%
  `faculty_adjunct_aggregate`, 68.7% `program`, 65.6% `course` — **this reverses the old
  "person has the most addressable headroom" reading, which was dense-alone-specific;
  `course` now has the most.** Also settled the same day: MAP and precision@1 are
  significance-tested at last (`tools/eval/map_precision_significance_test.py`, run at both
  the cross-chunker-aggregate and `semantic`-only scopes) — `qwen3_0.6b` beats all 8 other
  embedders on both, dense-alone (stronger than its recall@10 result), the semantic-scope
  tied cluster **holds on both new metrics** (precision@1 has no significant pair at all),
  and "`qwen3_0.6b` leads every metric" is an **aggregate-scope-only** claim (`qwen3` is
  numerically highest at semantic scope). Hybrid compresses embedder differences: the same
  9-embedder family goes 8/8 significant dense → 4/8 hybrid.
- **Two settled negative/null results — don't re-propose these without new evidence.**
  Both were built and run for real, and both were refreshed 2026-07-29 against the
  OCR-remediation-rebuilt indices (they had gone stale like everything else, see the
  staleness lesson above).
  1. **Cross-encoder reranking hurts hybrid — but the finding belongs to
     *truncate-and-replace*, not to the reranker.** Read the last paragraph of this item
     before citing this as a negative result: fusing the same model's scores in as a fourth
     RRF signal (2026-08-09) **beats the shipped hybrid on recall@10**, so what is settled
     is "don't let a cross-encoder replace the ranking", not "a cross-encoder is useless
     here". `CrossEncoderReranker`
     (`BAAI/bge-reranker-v2-m3`, `rerank_pool_size=50` → truncate to k=10) is wired as a
     query-time stage; `tools/eval/reranker_significance_test.py` re-retrieves live against
     `chunker_compare_full/plain__fixed_size__local__ceea7536` (so it goes stale on every
     index rebuild — it is **not** in the persisted-results refresh chain, and must be
     re-run by hand). **Refreshed 2026-08-05** against rebuild #3: result unchanged —
     **significantly hurts hybrid MRR** (0.7814→0.6778, Holm-adj p=0.0012, was
     0.7775→0.6775 p=0.0048 pre-rebuild), **no significant effect on dense-alone** (all
     three dense metrics Holm-adj p≥0.28), **no significant effect on hybrid recall@10 or
     nDCG@10** either (p=0.797 / p=0.284) — MRR-only is still the correct framing, and it
     costs ~1.22s/query mean (p50 1.17s, p95 1.42s). The nDCG@10 harm reported 2026-07-23
     (p=0.030) still does not replicate (now p=0.284, was p=0.5676 at the 2026-07-29
     refresh) and stays retired as a separate claim — it actually sharpens the
     literature's "phantom hits" mechanism (early-rank disruption without evicting relevant
     docs from the top-10). Literature grounding — including a paper naming
     `bge-reranker-v2-m3` by name — is in `docs/reranker-hybrid-interaction-research.md`.
     **The "wrong pool" escape hatch is now CLOSED (2026-08-09,
     `tools/eval/reranker_pool_source_test.py` → `data/results/reranker_pool_source_test.md`)**,
     on the *best* combo (`sentence × qwen3_0.6b`, not the original test's weaker
     `fixed_size × bge-m3` — say so when citing, the two tables are not comparable):
     pool source ∈ {dense, hybrid} × P ∈ {10,20,50,100,200}, equal 10-doc budget, with an
     **oracle rerank of the same pool beside every real arm**. `miss_depth_profile.md`'s
     "dense is closest on 70 of 84" motivated it; **the hypothesis is rejected in the
     opposite direction** — a dense pool is significantly *worse* (recall@10 **−0.1085**,
     Holm-adj 0.0000, m=3) and loses to shipped hybrid on all three metrics. **The reasoning
     error is the reusable part: "closest on the pairs everyone misses" is about 84 pairs,
     but a pool serves all 1,046** — dense's 0.5034 baseline starts too far behind hybrid's
     0.6281 for the hard pairs to repay. Two things the original test could not show. (1)
     **The evidence is in the pool and the reranker does not find it**: at P=50 the hybrid
     pool holds **0.8869** of the gold and a perfect rerank of it delivers **0.8249**, but
     the real reranker delivers **0.6162** — *below its own baseline*. Without the oracle
     column a null cannot be told apart from "the evidence was never reachable"; it was.
     (2) **Depth and harm point opposite ways, which is what closes the axis**: the misses
     sit at ranks 11-50 but captured headroom goes **−6% / −22% / −33%** at P=50/100/200, so
     it cannot reach them without destroying more than it recovers. Its per-type table shows
     damage concentrated on `person` (**−0.2668** vs `course` −0.0205) — **but do not read
     that as a truncate-and-replace effect; it is a POOL-SOURCE effect, corrected the same
     day** (see the next paragraph). It is the *dense*-pool arm, i.e. what happens when the
     candidates come from the retriever that scores 0.5735 on `person` rather than the one
     BM25 carries to 0.8147. On the hybrid pool, truncate-and-replace *improves* `person`
     (+0.1195) and collapses `program` (−0.1688). The one improving cell
     (hybrid P=20, 0.6535 vs 0.6281) was **not pre-registered** — cite it as a hypothesis for
     a fresh query set, never as a result. Cost is real too: P=50 adds ~1.2 s/query on a
     1.21-1.86 s base. Method worth reusing: a cross-encoder score depends on neither P nor
     the pool's source, so **score each (query, chunk) pair once and derive all 10 arms from
     the cache** (~1.3 arms' cost, and two arms can't disagree about one pair); it is
     persisted so a GPU-free re-render reproduces the report line for line (784 s → 53 s).
     Its S6 rebuilds `miss_depth_profile.md` §2's five delivered figures to 4 decimals from
     an independent path, and S4 pins the structural anchor that at P=k reranking may change
     the *order* but never the *set*, so recall@10 must equal baseline exactly.
     **The 4th-RRF-signal follow-up is now MEASURED, and it is this item's one positive
     result (2026-08-09, `tools/eval/reranker_rrf_signal_test.py` →
     `data/results/reranker_rrf_signal_test.md`, 68 s, no GPU — it re-fuses the *same*
     29,743 cached scores).** Keep shipped hybrid fusion and add the reranker as a third
     ranked system: `fused = (1-w)·[0.5/(60+dense_rank) + 0.5/(60+bm25_rank)] +
     w·[1/(60+rerank_rank)]`, **a document outside the pool contributing 0 from the third
     term** — not penalised, just unvoted-on, with its hybrid rank intact. That asymmetry is
     the whole idea: it is what lets an exact-name BM25 hit survive a reranker that dislikes
     it. **Both ends of the w grid are already-published arms**, so it interpolates between
     known points rather than introducing a scale: w=0.00 is shipped hybrid and **w=1.00 is
     truncate-and-replace exactly** (hybrid term gone; every pool doc scores >0, every
     non-pool doc 0, so the top-10 is the pool's top-10 by cross-encoder score) — S3/S4 pin
     both, S4 reproducing all six published figures to 4 decimals from an independent code
     path. Same discipline as `hybrid_alpha_sweep.py`'s alpha=0.50 anchor. Pre-registered at
     pool=hybrid, P=50, w chosen **leave-one-out** on recall@10 (Family 1, m=6):
     **`rrf4 (loo)` 0.6660 recall@10 vs shipped hybrid 0.6281, +0.0379, Holm-adj 0.0216 —
     significant**, and it beats truncate-and-replace on all three metrics (+0.0497 /
     +0.1171 / +0.0776, all 0.0000). **Cite MRR as REPAIRED, not improved**: the published
     harm reproduces at w=1.00 (−0.1197) and vanishes under fusion but does not become a
     gain (−0.0026, ns; CI rules out a loss worse than 0.0420 or a gain better than 0.0368),
     and nDCG@10 +0.0272 is ns too — **recall@10 is the only claim that clears
     significance.** No fitting premium: all 106 folds pick the same w, so LOO equals the
     oracle to 4 decimals, and the peak is broad — **report the range 0.40–0.55, not a
     point.** **The mechanism, corrected**: the prediction (fuse > replace) survived, the
     stated reason did not. The cross-encoder is *not* uniformly destructive — on the right
     pool it is a `person` specialist that wrecks `program`, and what RRF buys is **keeping
     both sides**, recovering `program` +0.1275 over truncate-and-replace while giving back
     only −0.0165 of the person gain. Also: **once the reranker is only a vote, pool depth
     stops mattering** (P=20 peaks 0.6662 vs P=50's 0.6660, at 487 ms/query instead of
     1,218) — a cost observation from a descriptive column, not a pre-registered result.
     **That +0.0379 was measured without routing, and it does NOT survive the hard router
     (2026-08-09, `tools/eval/reranker_rrf_routed_test.py` →
     `data/results/reranker_rrf_routed_test.md`, 878 s, 10,600 pairs over the 4 routed
     indices).** Measured as a 2×2 because "does it still help" and "substitutes or
     complements" are one experiment: **A** no routing/no rrf4 **0.6281**, **B** rrf4 only
     **0.6660**, **C** routing only **0.6831**, **D** both **0.6847**; every arm sends k=10,
     B and D additionally *fetch* 50. **All six pre-registered tests (m=6) are ns**: `D vs C`
     (the reranker on top of routing) **+0.0017 recall@10, Holm-adj 1.0000** (MRR +0.0116,
     nDCG −0.0005, both 1.0000); `D vs B` +0.0188/+0.0398/+0.0274, all 0.8244. **State it as
     a bound**: the CI rules out the reranker adding more than **+0.0212** on top of the
     router, for ~1.2 s/query and 50 extra fetches. **This is the second intervention to die
     against the router in exactly this way** (per-`entity_type` alpha was the first) and the
     mechanism is identical both times — both repair a per-type weak dense arm, and hard
     routing already hands each route a specialist index that hasn't got one. The per-route
     table shows the near-cancellation: `course` **+0.0496** and `person` +0.0140 against
     `program` **−0.0633** (the same cross-encoder personality as above), because routing had
     already collected the person gain that made the unrouted number large (person 0.7487
     unrouted → 0.8531 routed *before* any reranking). **Substitutes, not complements** — the
     same verdict soft-vs-hard routing reached. Two supporting details: there is no fitted
     signal left either (the P=50 w grid wanders 0.6784-0.6895 with no shape, a jagged plateau
     not a peak, and LOO 0.6847 vs oracle 0.6895 is a real fitting premium where the unrouted
     sweep had none), and truncate-and-replace on a *routed* pool is worse still (0.6000 at
     P=50, 0.6637 at P=20). Descriptively (not pre-registered): **B 0.6660 < C 0.6831**, i.e.
     routing alone beats the reranker path while costing no extra fetch and no query-time GPU.
     Three of the four cells are already-published numbers and the script **checks all three
     rather than assuming them** — S4 reproduces `routing_eval.md`'s 0.6831 from a *third*
     independent code path, S5 reproduces 0.6281/0.6660, S1/S2 reproduce 106/106 persisted
     top-10s. **Neither rrf4 nor per-type alpha is wired into `query_service`, and this is
     why.** **But the axis is NOT dead, and the oracle column is what says so**: a null alone
     cannot separate "this reranker is weak" from "nothing is left to win", so the same
     oracle was computed over the *routed* pool. At P=50 the routed pool **holds** 0.9054 of
     the gold and a perfect selection of 10 from it **delivers 0.8331** — **+0.1500 over arm
     C, against the real reranker's +0.0017, i.e. about 1% of its own ceiling.** So the
     verdict is *this cross-encoder is weak*, not *the headroom is gone*, and **routing
     enlarges the headroom rather than shrinking it** (routed 0.9054 holds / 0.8331
     delivered vs unrouted 0.8869 / 0.8249 — the specialist indices supply *better*
     candidates and the model still cannot select among them, the same shape as the
     unrouted diagnosis). Cite it as a **bound on the axis, not a plan**: an oracle is not a
     system, and closing any of +0.1500 needs a reranker qualitatively better than
     `bge-reranker-v2-m3` here, not a re-tuned fusion. Follow-up (a), a reranker trained on
     hybrid-fused candidates, keeps its motivation and remains untouched. **One trap, found
     by a failing self-check rather than by reasoning**: the delivered oracle is
     `min(#relevant resolutions with a chunk in the pool, K) / #relevant`, so chunks sharing
     a `resolution_id` **must be deduplicated first** — a perfect reranker never spends one
     of its 10 slots on a document it already returned. Sorting the pool relevant-first
     *without* dedup understates the ceiling (0.7790 instead of 0.8249 unrouted at P=50);
     S9, which reproduces `reranker_pool_source_test.md`'s published `delivered/holds` pair
     from an independent code path, is what caught it.
     **"This cross-encoder is weak" is now CONFIRMED by a second route (2026-08-09,
     `tools/eval/reranker_model_comparison.py` → `data/results/reranker_model_comparison.md`,
     112 s from cached scores): swap the model, change nothing else.** Same routed hybrid
     P=50 pool for every arm, same k=10 sent, same LOO-fitted `w`. **The model is a real
     variable and the anchor is a bad one**: over 4 qualified models the spread is
     **0.0355** recall@10 (mmarco-mMiniLM 0.6671 → bge-reranker-**v1**-large 0.7027) against
     the anchor's entire effect of +0.0017, i.e. ~20x. So the null belongs to
     `bge-reranker-v2-m3`, not to cross-encoder reranking on this corpus — the same verdict
     the oracle column reached, from independent evidence. **Cite the recall@10 family as
     inconclusive, not as a win**: 0 of 3 clear the bar (best `bge-v1-large` +0.0196, raw
     0.0282, **Holm 0.0612**, m=3), and the one significant cell is nDCG@10 **+0.0275**
     (Holm 0.0228, m=6, family 2). **The counter-intuitive part is the strongest part**: the
     best model is the *older* v1 lineage that v2-m3 supersedes, so reranker choice here does
     not track general benchmark strength and has to be measured on this corpus;
     `mmarco-mMiniLM` actively **hurts** (−0.0159), which is the project's RRF rule again —
     fuse only when the arms are comparable. **Selection caveat**: the winner is an argmax
     over 4 models on the same 106 queries (`w` is LOO, the *model* is not), so the citable
     claim is *at least one qualified model does materially better*, never *use bge-v1-large*
     — that needs a fresh query set. **The bound is unchanged**: the best model captures 13%
     of +0.1500, so 87% is still untouched and follow-up (a) keeps its motivation. Nothing is
     wired. Confounds measured rather than assumed: `ctx` is the one thing not equal across
     arms (anchor 8192, the rest 512 — each at its own max, since forcing 512 on the anchor
     would stop it reproducing its published number), but only **1.9%** of pairs exceed 512
     and the longest is **2,755** tokens; and S6 pins that the models genuinely disagree
     (Kendall τ +0.344 to +0.546, same top-1 on 17–44 of 106), because two models that rank
     the pool identically would give identical arms and a null would then only be saying the
     swap never happened. **Before measuring a reranker, qualify it —
     `tools/eval/qualify_reranker_model.py` → `data/results/reranker_model_qualification.md`,
     and this is the reusable part.** 2 of 6 candidates are broken under `transformers` 5.x,
     which materialises non-persistent buffers from the meta device as *uninitialised memory*
     rather than re-running `__init__`. `jina-reranker-v2` dies at import (safe).
     `gte-multilingual-reranker-base` **loads, runs, and ranks a hand-written Thai example
     correctly while being completely position-blind** — its RoPE `cos_cached`/`sin_cached`
     came back all zeros, and a sentence and its reversal score **bit-identically**. A
     plausible number from a bag-of-words model is the danger, not a crash: measured
     unchecked it would have scored low and "a second cross-encoder also fails" would have
     been published as a family-level claim. Five gates (G1 buffers, **G2 position
     sensitivity** — the load-bearing one, G3 relevance direction, G4 determinism, G5 padding
     independence), one model per **subprocess** (a CUDA device-side assert poisons the whole
     process, so a single-process loop rejects healthy models by ordering), and the gate is
     **exercised in both directions** — the anchor must PASS and the 2 known-broken must
     FAIL, checked in the report and in the exit code, because a PASS-only gate is not
     evidence. Two rules learned by getting them wrong: G1's integer test is *index out of
     range*, **not** *equals `arange`* (the first version rejected the anchor — XLM-R's
     `token_type_ids` is legitimately all zeros), and every gate but G5 scores its pair
     **alone**, so batch composition can't be mistaken for the effect under test.
     `tests/tools/test_qualify_reranker_model.py` pins both directions of every rule.
  2. **RQ3 preprocessing ablations: normalization and word-aware segmentation do nothing;
     only chunk size matters, and only at 1024.** Configs `config/experiments/rq3_*.yaml`,
     scripts `tools/eval/rq3_*`. Thai normalization (Thai digits + `pythainlp.util.normalize()`)
     and word-aware `newmm`-boundary chunking are both **not significant on any metric**
     (Holm-adj p≥0.335 / ≥0.264). Chunk size **is** significant but the citable claim stays
     narrow: **1024 loses significantly to 512** on dense recall@10/nDCG@10 and hybrid
     recall@10/nDCG@10, and to **256** on dense recall@10 and hybrid recall@10/nDCG@10 (the
     dense nDCG@10 256-vs-1024 cell is a near-miss, Holm-adj p=0.0828, **not** significant —
     don't fold it into "256 beats 1024 on every metric"). **256 vs 512 is a flat tie on
     every dense metric** (recall@10 0.4117 vs 0.4129, Holm-adj p=0.9676) **and on hybrid
     MRR — 256 only wins on hybrid recall@10** (+0.0481, Holm-adj p=0.0154). **Do not cite
     "smaller is monotonically better" or "256 is best"**; the project's 512 default is not
     shown to be suboptimal, only 1024 is shown to be wrong. These ablations' treatment
     indices reuse `chunker_compare_full` combos as their *baseline* arm, so an index rebuild
     silently turns them into a clean-baseline-vs-dirty-treatment confound — they need real
     GPU rebuilds after a corpus change, not just a re-eval, and this happened twice: once
     for the 2026-07-28 OCR-remediation rebuild (fixed 2026-07-29) and again for
     `chunker_compare_full` rebuild #3 (2026-08-05T07:56). **Both times, fixing it meant a
     real GPU rebuild of all 3 RQ3 treatment indices, not just a re-eval** — most recently
     done 2026-08-05 (`data/logs/run_rq3_rebuild_2026_08_05.sh`, ~2.5h, exit=0). **No known
     confound remains as of 2026-08-05; the numbers above are current and citable.** If
     `chunker_compare_full` is rebuilt again, treat RQ3 as stale again until re-run.
- **RQ4 (end-to-end answer quality) is FULLY COMPLETE (2026-08-03)** — generation,
  the prompt ablation, and the `cite_all` extension all done and committed
  (`f3c04f1`/`f107469`/`f7add7d`/`44817bf`). 5 arms (hybrid/dense/bm25/m2v/closed-book,
  all `phi4` local-only, no external API) × 106 queries × **2 prompts** (original
  `sentence_cap` rule 4 + the ablation's `cite_all`) = the full table, scored by
  `tools/eval/rq4_score.py` (paired bootstrap + Holm, same machinery as every other
  significance test here). Report: `data/results/rq4_score.md`; narrative + both
  build phases: `docs/rq4-design.md`. **Refreshed against `chunker_compare_full`
  rebuild #3 on 2026-08-07** — see the currency paragraph at the end of this bullet
  before citing anything here; two findings below are corrected there.
  **READ THIS BEFORE CITING ANY RQ4 NUMBER: every answer on disk was generated at
  `num_ctx=8192`, and 80 of the 1,590 published (query, arm, variant) cells had
  their prompt silently TRUNCATED (2026-08-10, `docs/rq4-prompt-truncation.md`).**
  Found by the mandatory pre-run check before adding the entity arms, not by any
  symptom — there is no symptom. **The rule was measured, not read from the docs**
  (ollama 0.32.6 / phi4): a prompt that *fits* `num_ctx` is fed whole; one that
  *exceeds* it is cut to **`num_ctx // 2 + 2` tokens, keeping the tail**. So the
  threshold is 8,192 tokens, not 4,098 — the tempting "never more than num_ctx/2"
  reading is refuted by its own control (5,651 / 6,885 / 7,508-token prompts are
  fed whole at 8192), and `prompt_eval_count == num_ctx//2 + 2` is an exact
  truncation signature. **The direction is what makes it bad**: `build_prompt`
  lays documents out best-first and puts the rules last, so a cut deletes the
  *highest-ranked evidence* and always spares the instructions — the answer comes
  back fluent, correctly formatted, correctly citing, and evidence-poor. The
  2026-08-03 "instructions after context" fix
  ([[feedback_llm_prompt_truncates_from_front]]) is precisely what made this
  invisible rather than harmless, which is the reusable lesson
  ([[feedback_an_asserted_invariant_is_not_a_check]]: the `--num-ctx` help string
  already asserted "MUST exceed the longest prompt" in capitals, and nothing
  measured it). **Blast radius, exact token counts:** `hybrid` **0/106** in all
  three variants (worst 7,999 — 193 tokens short of the line, by luck, and
  `cite_all_guarded` came within 2.4% of losing it), `closed_book` 0/106 by
  construction, `dense` **16/106**, `bm25` 5/106, `m2v` 7/106; the entity arms
  would have been ~45-50%. **So the confound pushes in the same direction as the
  published `hybrid > {dense, bm25}` ordering — that finding is neither confirmed
  nor refuted by this, it is measured under a confound that flatters it.** The
  prompt ablation (the headline) is a within-arm comparison and survives.
  **Repaired at source**: `rq4_generate.py` gained a `preflight()` that builds
  every prompt, sends the longest with `num_predict=1` and refuses to start on the
  signature; a per-answer guard that names each truncated prompt and exits
  non-zero; `num_ctx`/`prompt_eval_count` recorded in every answer JSON (the
  8192-era answers carry no such field, which is exactly why the damage had to be
  reconstructed prompt by prompt); and default `--num-ctx` 8192 → **16384**.
  Pinned by `tests/tools/test_rq4_prompt_truncation.py`. The 80 cells are **not
  yet regenerated** — and note that a re-run must be *scored*, not eyeballed,
  because the generator's own noise floor is 14/24 identical citation sets at
  temperature 0 ([[feedback_temperature_zero_is_not_reproducible]]). Method note
  for any future prompt-size work here: **chars/token is unusable as an
  estimator** on this corpus — 1.046 (Thai prose) to 3.151 (English course tables)
  within the same prompt family, so screen on the *minimum* ratio and measure
  everything above the line exactly. Findings, in
  the order they were established:
  (a) **retrieval quality survives the generation stage — but state it as
  `hybrid > {dense, bm25} > m2v`, not as a strict 4-way ordering.** Post-refresh:
  under `cite_all` hybrid 0.7268 > dense 0.6629 > bm25 0.5968 > m2v 0.5203, but
  **dense vs bm25 is not significant in either prompt variant** (Holm-adj 1.0000
  under `sentence_cap`, 0.2136 under `cite_all`) and under `sentence_cap` bm25
  (0.6463) numerically *edges* dense (0.6413). The old wording "citation precision
  orders exactly as recall@10 did (hybrid 0.742 > dense 0.670 > bm25 0.625 > m2v
  0.562)" over-read a tie and is **corrected 2026-08-07**;
  (b) **the original run's flat ~0.41 citation recall across every arm was a PROMPT
  artifact, not a generator ceiling — confirmed, not just suspected.** Prompt rule 4
  said `ตอบสั้น ๆ ไม่เกิน 3 ประโยค` against a gold set dominated by aggregation queries
  (mean 9.87 relevant docs); re-running hybrid+bm25 under `cite_all`
  ("cite every relevant document") raised recall significantly for both (hybrid
  0.2862→0.3865, bm25 0.2127→0.3034, Holm-adj p<0.0001) with **no significant
  precision cost** — the model cites more correctly, not more sloppily. **Do not cite
  "the generator is the bottleneck"; the recommendation is "fix the instruction."**
  (Note: `rq4_score.py`'s recall denominator is the full qrels, stricter than the
  original inline script's "present-in-context" denominator — the ~0.41 figure and
  the 0.21-0.29 figures are not the same metric, don't cross-cite them.) The extension
  then covered the remaining 3 arms under `cite_all` too and found **the gain isn't
  universal — m2v doesn't improve (+0.026, Holm p=0.657)**, consistent with retrieval
  quality (the RRF-failure arm likely lacks enough correct evidence in context to cite
  regardless of instruction), and **arm ordering (4c) sharpens under `cite_all`**
  (post-refresh **9/12** pairwise tests significant vs **2/12** under the original
  prompt — direction unchanged and stronger than the 8/12-vs-6/12 first measured;
  m2v significantly worst on both precision and recall). One real cost surfaced:
  **closed-book abstention dropped 106/106 → 104/106 under `cite_all`** (2
  hallucinations, 5 phantom citations) — `cite_all` has no zero-document guard, worth
  a tightened wording before adopting it as the paper's final prompt; (c) **0
  fabricated citations across all 954 citations under the original prompt** — RAG's
  most-feared failure mode is absent here, the payoff for exactly-checkable numeric
  labels — but **this is prompt-specific, corrected 2026-08-07**: under `cite_all`
  the dense arm now shows **4/359 phantoms** (previously 0/370), all from one query
  citing labels `[6]`–`[9]` when only 5 documents were supplied. So `cite_all` shows
  fabrication in two arms, not closed-book alone. Caveat: citation precision is judged against the
  same qrels, so it inherits the pooling-bias threat — direction is conservative (see
  validity bullet below). **De-prioritized, not cancelled**: the `gemma4:e4b`
  robustness check (does a second model agree on arm ordering) — no longer needed to
  answer "is the ceiling real" (that's closed), but if run later for other reasons it
  must use `cite_all`, not the original prompt, or it will just reproduce the retired
  artifact. Two build-phase gotchas worth keeping in mind for any future generation
  work: (a) **Ollama truncates an over-long prompt from the front**, so a default
  `num_ctx=4096` silently deleted the instructions on long prompts and produced
  fluent, plausible, citation-free answers — always set `num_ctx` and put instructions
  *after* the context; (b) the design doc's original "recall@10 ~0.6 so the context
  often lacks the answer" was wrong (recall ≠ presence: 96% of contexts hold ≥1 gold
  doc), so 4b's power lives in the weak arms and closed-book, not the strong ones.
  **Currency: refreshed 2026-08-07 against `chunker_compare_full` rebuild #3, and
  this is the one refresh in the project that must NOT be read as "0 flips".**
  Contexts rebuilt for all 4 retrieval arms, then only the **362 of 530**
  (query, arm) cells whose context actually changed were regenerated — the other
  168 frozen, so the comparison stays paired (4h05m, exit 0, 0 errors;
  `data/logs/rq4_regen_2026_08_07.log`). Result: **5 verdict flips of 33**, and
  they are weak evidence, because `rq4_generate.py`'s "temperature 0 ⇒ no
  sampling variance" docstring **was false** — re-running byte-identical prompts
  reproduces the citation set only 21/24 (`sentence_cap`) / **14/24** (`cite_all`),
  measured by `tools/eval/rq4_determinism_check.py`; see
  [[feedback_temperature_zero_is_not_reproducible]]. All four *lost* verdicts are
  in family 1a and were already borderline (Holm-adj 0.014-0.081), nothing at
  p<0.001 moved, and the largest single driver is one arm's mean precision
  (`phi4 / hybrid_m2v` 0.4945→0.5575) narrowing three m2v comparisons at once —
  **report those four as inconclusive, not reversed.** Everything cited above
  survived: the whole prompt ablation (hybrid +0.1181 / dense +0.1005 / bm25
  +0.0734 all Holm 0.0000, m2v +0.0217 ns), m2v-worst, and 106→104 abstention.
  One further lesson: the phantom-citation regression in (c) was **silently
  skipped by `diff_significance_reports.py`**, because that column is formatted
  `count/total` and matched neither its numeric nor its verdict branch — the differ
  now reports and gates on every non-numeric cell change (CIs excluded).
  **`cite_all`'s two measured costs are now REPAIRED — use `cite_all_guarded`
  (2026-08-07).** `cite_all` is left untouched (the 530 answers on disk are keyed to
  that variant name; editing its wording in place would silently decouple them from
  the prompt that produced them), so the fix is a third `_RULE4` entry in
  `rq4_generate.py` writing to its own `answers/phi4_cite_all_guarded/`. The
  diagnosis is the point: rule 3 already forbade the failure and is *identical*
  between variants, so it was never a missing rule — it was **position**. Rule 4 is
  the last line before the question, and "cite every relevant document" outranked
  rule 3 by recency, the same mechanism `build_prompt` exploits deliberately
  (context first, instructions last), here working against us. So the guard is
  placed *after* rule 4 and says outright that it outranks it: **rule 5** (no
  documents supplied at all ⇒ abstain, cite `-`, cite no number, and this beats
  rule 4) and **rule 6** (cite only labels that literally appear above). Results,
  each confirmed on the failure it was written for: rule 5 → closed-book abstention
  **104/106 → 106/106**, phantom **5/5 → 0/0**; rule 6 → dense phantom
  **4/359 → 0/353** (no other arm produced a phantom under any variant, so dense
  was the only arm that could test it). **All 4 retrieval arms regenerated under
  the guard 2026-08-08** (bm25+m2v, 212 answers, 4678s, exit 0), so the variant
  now carries the full 530 answers and every family is rerunnable under it.
  **The benefit survives**: guarded beats the `sentence_cap` baseline by
  **+0.1123 on dense** (Holm 0.0000 in every family) and **+0.0706 on hybrid**
  (Holm 0.0144 in family 2), so the ablation's headline doesn't depend on the
  unguarded wording; bm25's guarded gain (+0.0539) misses significance where the
  unguarded +0.0734 made it, and m2v moves under neither. **The apparent cost vs
  unguarded `cite_all` is not a finding** — no arm significant and the point
  estimates **don't agree on a direction** (dense +0.0117 vs hybrid −0.0475,
  bm25 −0.0195, m2v −0.0067), which is what the measured noise floor predicts
  (14/24 identical citation sets at temperature 0,
  [[feedback_temperature_zero_is_not_reproducible]]); as bounds, hybrid rules out
  the guard being *better* than `cite_all`, dense rules out a loss > ~0.017.
  **Two things the 4-arm run added that the 2-arm run could not show.** (1) The
  guard is **not free**: rule 5 applies to every arm, not just closed-book, and
  it pushes the weak arms toward abstention — m2v correct-abstain 13→19 and
  hallucination 16→10 but *missed* (gold present, abstained) 11→18; bm25
  hallucination 12→10. Report the trade. (2) **The 4c "sharpening" claim belongs
  to `cite_all`, not to the guard**: family 1's 12 pairwise tests separate
  **2/12** under `sentence_cap`, **9/12** under `cite_all`, but only **3/12**
  under `cite_all_guarded` — the guard pulls the strong arms down (hybrid
  0.3962→0.3487) while dense rises (0.3206→0.3323), compressing the spread.
  Direction is unchanged everywhere and several guarded cells miss narrowly
  (0.0576, 0.0896), so it is smaller separation, not lost separation. **Recommendation:
  report `cite_all_guarded` as the paper's prompt but cite the ordering result from
  `cite_all` with the 3/12 stated alongside.**
  `rq4_score.py` gained `--treatment-variant` and `--out` so a variant is scored
  against the same baseline without clobbering the published `rq4_score.md`
  (guarded report: `data/results/rq4_score_guarded.md`). **Always quote the Holm
  family size** — with 4 arms family 3 holds **24** tests and family 2 holds **9**,
  and on 2026-08-08 they stopped agreeing: `hybrid: guarded vs baseline`, identical
  data, +0.0706 either way, reads **0.0144 (significant) in family 2** and **0.0600
  (not significant) in family 3**. Neither is wrong; family 2 is the one built to
  answer "does this prompt beat the baseline", so cite that one, *as family 2*.
  Family 3 was added 2026-08-08 because the variant-vs-variant pairs
  (`guarded` vs `cite_all`) exist in no other family, so they had been computed ad
  hoc and the doc quoted numbers no script could reproduce — the
  [[feedback_recompute_derived_stats_from_the_table]] failure mode, caught by
  re-reading the report against the prose.
- **Evaluation validity — read `docs/eval-validity-threats.md` before defending any
  number in this project.** Written 2026-07-30 against the question "is 106 queries too
  few for a reviewer". It is not (BEIR peers run 50-300 topics, and this set is unusually
  *deep* at 1,046 judgments / 9.87 relevant docs per query vs MS MARCO's ~1.1) — but three
  other things are real threats, and two are now measured:
  1. **Statistical power: closed.** `tools/eval/power_analysis.py` →
     `data/results/power_analysis.md`. **138/180 comparisons significant, and all 42 ties
     have observed |diff| below their MDE** — no tie here is a power artifact, so every one
     can be cited as a *bound* rather than a null. Chunker ties are the tightest in the
     study (`fixed_size` vs `sentence` rules out >0.031 recall@10; family 0.031-0.052),
     which is the right way to state the retired "semantic wins" headline; embedder ties
     are looser (0.05-0.10); **`e5_small` vs `jina_v5` on MRR (bound 0.1045) is the one
     pair that must be called inconclusive, not tied.** The closed-form MDE is
     simulation-verified against the real bootstrap (achieved power 0.78-0.86 vs nominal
     0.80), so cite it directly. Recomputed from persisted results → **re-run after any
     index rebuild** like everything else.
  2. **Pooling bias: CLOSED 2026-08-03 — not directional, qrels are a modest undercount.**
     The qrels were derived by *string containment*, which is what BM25 does — so a
     relevant document phrased differently (or with an OCR-truncated title, seen in the
     first sampled item) is a false negative that penalises dense retrieval for being
     right. This points straight at "BM25 significantly beats `bge_m3`" and "BM25 carries
     `person` (0.8147)". `tools/eval/residual_relevance_sample.py` builds a **blinded**
     126-item review sheet (29 stratified queries, unjudged top-10 hits, arm held in a
     separate key file); `--score` gives per-arm residual-relevance rates with Wilson
     intervals. **Decision rule pre-registered in the doc** (intervals overlap →
     incomplete, comparisons stand; disjoint → biased, restate every BM25-vs-dense claim).
     **First-pass manual judging (browser Ctrl+F per item) came back ~98-100% relevant for
     every arm and was retracted the same day**: the review app's calibration-reference
     section shows guaranteed-relevant reference docs in full text on the same page as the
     candidate, so a page-wide search found the entity there instead of in the candidate
     100/100 times it was checked — a review-UI defect, not an annotator error. **Corrected
     via `tools/eval/residual_relevance_decompose.py`**, which reapplies
     `build_gold_candidates.py`'s own per-entity-type matching rule directly against each
     candidate's full text — licensed by the corpus owner's domain confirmation that, for
     this query shape (specific named person/course/faculty/programme), relevance requires
     the entity to literally appear; this is a deterministic rule reuse, **not** LLM
     judging, so it doesn't reintroduce the automated-relevance risk the Gold set was built
     to avoid. **Result: residual rate ~19-22% across all three arms** (dense 0.191, BM25
     0.224, hybrid 0.224), Wilson CIs overlap → confirms incomplete-not-biased, but the
     magnitude is a modest ~8-11% undercount (~0.8-1.1 missed relevant docs/query vs. the
     qrels' own mean of 9.87), not a severe one. BM25-vs-dense comparisons stand as relative
     rankings; absolute recall/precision numbers need only a slight-undercount caveat.
     Original manual (retracted) verdicts kept at
     `data/results/residual_relevance/review_sheet.manual_backup_2026_08_03.yaml` for the
     record.
  3. **Circularity** in `entity_lookup`/`entity_boost`: **paragraph DRAFTED
     2026-08-07**, in citable form in `docs/paper-results-summary.md` §"Circularity in
     the entity arms". Their qrels come from the same `programs.json`/`people.json` the
     retrieval mode uses, so the score is an upper bound, not a measurement. Confined to
     those arms — chunker/embedder/BM25/hybrid never touch the dictionaries, and
     `entity_tags_full` is a separate index nothing else is built on. **Cite the number
     as recall = 0.9422, and never as `recall@10`**: the long-quoted `0.9291` predates
     the 2026-08-05 `entity_tags_full` rebuild, and the metric is recall@**1000**
     (`entity_lookup` is exhaustive and unranked, deliberately scored at k=1000 so
     recall/precision reduce to plain set recall/precision) — calling it recall@10
     invites exactly the comparison against the dense/lexical recall@10 columns that the
     paragraph exists to forbid. Two sharpenings worth keeping: the circularity lives in
     the **candidate set**, so `entity_boost`'s rank metrics are contaminated only
     indirectly (hybrid ordering never reads the dictionaries) while `entity_lookup` has
     no ordering to rescue it; and **the pooling-bias verdict above does NOT transfer
     here** — a name the dictionary lacks is absent from the qrels *and* invisible to the
     retriever at once, so the undercount is correlated with the system and reads
     optimistic rather than neutral. That is why this threat cannot be closed by
     measurement the way the other two were.
  Also covered there: single-annotator labelling (defended by the labels being
  *rule-derived and re-derivable*, not judged), query provenance, external validity.
- **Oracle-union ceiling (`tools/eval/oracle_union_ceiling.py` →
  `data/results/oracle_union_ceiling.md`, 2026-08-08)** — union the persisted
  top-10 of all 36 live combos: a pair no system finds is unreachable by any
  reranker/ensemble/fine-tune while the index family and k are fixed. Rewritten
  from an outside analysis in `road-to-wow-demo/`. Its numbers are superseded —
  best single **0.6281** (was 0.6935), hybrid ceiling **0.8948** (was 0.9201),
  pairs **1,046** (was 644) — but **the cause is the query set, not a bug in it,
  and getting that attribution right took a second look**: it ran on a checkout
  whose gold set still had **73** entries (`REPO` is hardcoded to another user's
  OneDrive path), and the 33 `course` queries are the harder ones. Holding this
  script's combo set fixed and scoring only the 73 non-course queries reproduces
  its shape — best 0.6728, union 0.9125. The first draft of this bullet accused
  it of unioning 44 combos; **its own header says 36**, so that was wrong. What
  *is* true is a portability defect: it selects combos by bare `glob`, so re-run
  here it takes 44 (the 8 in `_EXCLUDED_COMBO_DIRS`, indices deleted, results
  pre-rebuild-#3) and the ceiling reads 0.9046 instead of 0.8948 — measured in
  §1, not asserted. Note it moves the *ceiling* only; the retired combos are the
  weak `sct`/`congen` and never the argmax. Derive the combo set from which
  index dirs *exist* and cross-check against the exclusion list — either half
  alone goes stale. See [[feedback_external_analysis_reads_a_stale_slice]]:
  verify what an outside analysis actually ran on before critiquing it. Findings: (a) **at a fixed 10-doc budget, diversity is negative**
  — 2 systems × 5 = **0.5913** vs 1 system × 10 = **0.6281** (**−0.0368**),
  while doubling the budget is **+0.1158**; the original's "ensemble wins" read
  20 docs against 10, and both arms here are greedy-fitted on the test set so
  the bias favours diversity and it still loses; (b) **69.3% of the misses are
  ranking, not absence** (best single 512 pairs, union 882 of 1,046) — but that
  headroom's own ceiling *at 10 docs sent* is 0.7771 (oracle picks the combo) to
  0.8355 (perfect rerank over all 360), **never 0.8948**; (c) unioning the dense
  and BM25 result sets too lifts it to **0.9443 macro / 0.9197 micro**, so
  **80 of the 164 "nothing found it" pairs were a retriever-choice artifact** and
  the floor is **84 pairs (8.0%)** — **cite 84, not the 76 this bullet used to
  publish**: the subtracted 8 were called a labelling artifact, and that premise
  was measured and refuted 2026-08-09 (see the anchor-ambiguity bullet below), so
  the subtraction is withdrawn in the script and the report. **Do not call that floor *structural* — the word
  was withdrawn 2026-08-08 by `tools/eval/miss_depth_profile.py` (next bullet).** The §1d
  router was **dropped, not recomputed** — `routing_eval.py` is its tested
  descendant and reports `routed (loo)` 0.6780 (+0.0499, **ns**), so recomputing
  an untested +0.0465 here would have manufactured a conflict. **S5 pins this
  report against the other ceiling the project publishes**: every row that sends
  only 10 docs must stay under `paper-results-summary.md`'s structural
  0.8856 = `mean(min(1, 10/n_relevant))` (it recomputes that constant and gates
  on it); the union rows exceed it legitimately because they send 360, and that
  is the one way to misread these two tables together.
  **A qrels defect fell out of it, and the rejected hypothesis is worth as much
  as the accepted one.** The single 0.000 query (`รายวิชา CONTROL SYSTEMS`): the
  Gold set also holds `รายวิชา CONTROL SYSTEM`, one character apart, and
  exact-token matching gave them **disjoint** qrels — the union pulls 103 docs
  for the plural query, **0** of its own gold and **9** of the singular's. This
  bullet used to call it **unanswerable *by construction***; **that is withdrawn
  2026-08-09** — see the anchor-ambiguity bullet below, which measured it. Detect
  the shape (one course name a token-prefix of another), don't hard-code the
  pair. The hypothesis that **failed** was the
  attractive one: 38 of 401 `course` pairs (9.5%) are relevant only via another
  course's `PREREQUISITE:` line, which reads like an unretrievable needle — but
  `SIGNALS AND SYSTEMS` is 9/10 prerequisite-only at union recall **1.000** and
  `ELECTRONICS ENGINEERING 1` is 10/10 at 0.900. Cite 9.5% as a category that
  exists, never as a cause.
- **Gold anchor ambiguity (`tools/eval/audit_gold_anchor_ambiguity.py` →
  `data/results/gold_anchor_ambiguity.md`, 2026-08-09)** — the CONTROL SYSTEM(S)
  pair above, chased to its shape instead of its instance. **The reported defect
  does not exist: the qrels are not self-contradicting.** S3 rebuilds every
  course query's qrels from the code tags and reproduces them **33 of 33**; the
  two courses (`01046707`, `01306023`) are genuinely different. **What is real is
  a key mismatch**: `course` is the *only* entity type whose qrels key (the
  8-digit **code**, via `courses_by_file.json`) differs from what its query
  supplies (the **name**) — `program`/`person`/`faculty_adjunct_aggregate` judge
  relevance on exactly the string the query gives, so **73 of 106 queries are
  unexposed by construction**, a denominator rather than an estimate. **Two
  claims are withdrawn by measurement.** (1) "unanswerable *by construction*" is
  **false** — all **8 of 8** of the plural query's gold documents literally
  contain the phrase `CONTROL SYSTEMS` (`gold ที่ไม่เอ่ยชื่อ` = 0); it is
  answerable in principle, just not *by name matching alone*, which is weaker and
  true. (2) Consequently the 8 pairs are **not** a labelling artifact and the
  oracle-union floor is **84 (8.0%), not 76** — corrected in
  `oracle_union_ceiling.py` and its report the same day. **Keep the two mechanisms
  apart** — *ambiguous name* (other courses' documents show the query's anchor
  text: the evidence is all present but competes, `CONTROL SYSTEMS` 8 gold among
  **65** naming documents) versus *silent name* (gold tagged by code that never
  spells the name, 4 of 33 queries) — because dropping queries fixes neither, and
  only the first is what the 0.000 row is. Metric: **anchor precision** =
  `gold ∩ naming / naming`, flagged below 0.5 as a *statement* ("most documents
  showing the query's own anchor text are judged irrelevant") with the full
  distribution printed so the cut isn't load-bearing → **3 of 33 flagged**
  (0.123 / 0.200 / 0.421). **Read `เพดานเห็นแต่ชื่อ` = `min(1, k/naming)` and
  `เพดาน qrels` = `min(1, k/gold)` as a pair, never alone** — the `Δ เพดาน`
  column is what separates the defect from the benign "more than 10 relevant
  documents" case (−0.846 for `CONTROL SYSTEMS`, +0.000 for `CALCULUS 2`). Two
  further findings: **5 of 33** gold course names are sub-phrases of another
  course's name, but sub-phrase ≠ broken (`SIGNALS AND SYSTEMS` union 1.000,
  `INDUSTRIAL AUTOMATION` anchor precision 0.824); and §3b finds the *opposite*
  direction — a shorter dictionary name inside the query's own name adds extra
  codes for **3 of 33** (`ENGLISH FOR`, `INVESTMENT PROJECT ANALYSIS`), which does
  **not** touch `classify_query` (it only asks whether *any* course matched, so
  the route stays 33/33 per `tests/test_router.py`) but does touch
  `detect_entities`/`entity_lookup`. **`courses.json` is deliberately not
  shrunk** — `router._default_course_matcher` reads it, so the gate belongs in
  `build_gold_candidates.py`, which now annotates each course candidate with
  `anchor_status` ∈ `ok`/`ambiguous`/`no_name_evidence` (**414 / 66 / 198** of 678).
  **That third bucket is why the classification is three-way, not a number**: with
  zero naming documents the ratio is *undefined*, not zero, and collapsing them
  reported 264 flags of which 198 were OCR-garbled dictionary names, burying the
  66 real ones. **Whitespace must be collapsed before matching** (OCR'd minutes
  wrap long course names across lines) and that is the conservative direction — an
  inflated `gold_not_naming` would invent a second mechanism that isn't there; the
  contract is that the caller collapses each document **once** (33 names × 2,853
  documents), and `tests/tools/test_gold_anchor_ambiguity.py` pins it in the
  negative along with the boundary rule both scripts share. **The repair by
  deletion was priced before being declined (§4): dropping the 0.000 query moves
  macro recall@10 +0.0050 and dropping all 3 flagged +0.0113 — *upward*, because
  the dropped queries are the low-scoring ones**, so the cost isn't the Δ, it is
  re-running and re-copying every table and every "106 queries" claim. Nothing was
  dropped; the outcome is documentation + the build-time gate + the two
  corrections above.
- **Miss-depth profile (`tools/eval/miss_depth_profile.py` →
  `data/results/miss_depth_profile.md`, 2026-08-08)** — the split the ceiling
  bullet above left open. **The ticket said "run k=50"; that is the wrong
  experiment** — `DenseRetriever`/`BM25Retriever` score the whole corpus then
  `argsort(-scores)[:k]`, so k=50 is free but erases the only distinction being
  asked about (rank 51 vs rank 40,000). It computes **untruncated** ranks for 36
  combos × 3 arms instead, ~14.5 min. **Result: the floor is depth, not absence** —
  of the 84 all-arm misses, **64 (76.2%) sit at ranks 11-50**, 83 of 84 are inside
  the top 1,000, exactly **1** is deep (`รายวิชา CALCULUS 2`, rank **2,984**), and
  **0 are missing from the index**. So drop "structural"; a reranker fetching 50
  candidates can reach three quarters of them. Two consequences: **`person` has 0
  misses** (the 84 are course 33 / faculty 28 / program 23, and course is almost
  purely near-miss at 32-of-33 while faculty splits 14 near / 14 deep — a reranker
  helps course, not half of faculty); and **the candidate pool should come from
  `dense`, not the shipped hybrid** — on these hard pairs dense has median best
  rank **26** and is closest on **70 of 84**, vs hybrid 43 (9) and BM25 210 (6).
  Read against the **already-measured** cross-encoder result (hurts hybrid MRR
  0.7814→0.6778, p=0.0012): the evidence is in reach at P=50, the tried reranker
  does not reach it. **Two replication traps are pinned in the docstring because
  both were hit while writing it.** (1) Batching all 106 queries into one
  `(N,1024)@(1024,106)` matmul reproduces only 98/106 top-10s — the *scores* are
  identical, but BLAS accumulates a batched product differently and
  `np.argsort`'s default quicksort is unstable, so exact ties reorder; replicate
  `DenseRetriever.retrieve`'s per-query gemv exactly (and leave `emb` float32).
  (2) `HybridRetriever` settles equal RRF scores **dense-first**, so fusion must
  be `dorder[argsort(-fused[dorder], kind="stable")]`. S2/S3/S4 gate all three
  arms against the persisted results (3,816 / 848 / 3,816 reproduce, 0 differ),
  S3b *verifies* rather than assumes the chunk-row sharing that licenses the
  per-chunker BM25 cache (36 combos → 4 `BM25Okapi` builds, the 2.4x speedup),
  and **S5/S6 reproduce the ceiling report's 84 and 164 from an independent code
  path**. **S7 exists because the first version of §2 was wrong**: it reported
  "perfect rerank from a pool of 50 = 0.8869", which is *above* the qrels ceiling
  0.8856 and therefore impossible for a reranker that still sends 10 documents —
  the table was measuring what is *in the pool*. §2 now prints both columns and
  S7 gates the deliverable one against the ceiling — **cite the delivered one**:
  a perfect rerank over P=50 is 0.6281 → **0.8249**, and P=1000 buys only 0.8738,
  so the 10-document budget binds, not the pool. Same family as
  [[feedback_state_the_retrieval_budget_in_every_comparison]].
- **Candidate next axis, written up but not started**:
  `docs/colbert-late-interaction-notes.md` (ColBERT: motivated by *our own*
  results — the cross-encoder reranker hurt hybrid MRR, and BM25/dense split
  person vs program — with a pre-registered prediction so an aggregate win can't
  be mistaken for resolving that split). Not committed to — RQ4 (the item that
  used to block starting this) is now complete, see above.
- **Corpus data-quality audit** (`tools/corpus_prep/audit_title_body_agreement.py`,
  2026-07-30): flags manifest titles that disagree with the document's own page-1
  `เรื่อง` subject line. A first version was rejected on measurement (median 0.660,
  544 files below 0.5, nearly all artifacts); this one strips agenda numbering,
  compares by **token containment rather than string similarity**, and scores
  **asymmetrically** (what fraction of the *title's* words the subject line
  supports, so a truncated title scores 1.0). Result: median **1.000**, **7 flagged,
  7/7 genuine**. Report + per-case verdicts: `docs/title-body-agreement.md`. Two
  causes: 4 mispairings (metadata-only, incl. one A↔B swap) and 2 items in
  `2564/ครั้งที่ 12` with no document of their own (recorded here as "the CHECO
  shape" until 2026-08-09 disproved it — see the orphaned-agenda-items bullet
  below). **The 4 mispairings are APPLIED (2026-08-08); flag count 7 → 3**, the
  remaining 3 being those 2 and the 1 generic-title judgement call, all
  deliberately kept.
  `fix_manifest_title_mispairings.py --apply` (titles only — for the `2565/8`
  A↔B swap every field is internally consistent and it is the *content* that
  landed in the other entry's file, so swapping titles is the one edit that
  re-aligns title, body **and** url) then
  `relabel_renamed_resolution_ids.py` (41/55 indices, 2,021 rows; 302/24,217
  result files, 323 rows). **The premise this waited a week on was wrong: it is
  a relabel, not a rebuild.** A title change moves an id but not chunk *text*,
  and embeddings are a function of text alone — minutes, no GPU, and rewriting a
  persisted result is exactly equivalent to re-running retrieval. No metric can
  have moved: 0 of 358 gold `resolution_id` entries reference any old or new id.
  Three things worth reusing. (1) The mapping was **derived, not typed** —
  loader-computed id snapshots either side of the manifest edit, diffed — because
  a typo in a 90-character Thai title would silently mint a third id. (2) **A
  swap is not idempotent**: re-running reverts it, so the relabeller refuses when
  an incoming id already sits on rows it isn't relabelling (this fired on the
  verification re-run, as designed). (3) **`I6` was blind to this whole class of
  change** — it derived "the corpus's last edit" from `*.md` mtimes, but a
  `resolution_id` comes from `meeting_manifest.json` (ADR-0003), so a title
  repair would have left 41 stale indices passing the staleness check. I6 now
  reads manifest mtimes too, and treats a recorded relabel
  (`relabeled_mispairings.at`) as bringing an index current without a rebuild,
  so it doesn't go permanently red after every title repair. Verified by text,
  not by counting: a wrong-way swap gives the right ids in the right row counts,
  so all 4 new ids were checked against the full text of the file that now
  carries them (9/9, 12/12, 16/16, 12/12 chunks). Post-repair gates:
  `audit_resolution_ids.py` unchanged, invariants **24/0/1** (the then-known
  `BuildCombo.id` FAIL, closed 2026-08-09 by `E0`), `E3a` 0 of 23,156,
  doc-claims 3/2/0 (**5/1/0** since `D1b` closed and `D1c` landed 2026-08-09 —
  the family grew, nothing here moved). This section used to
  end "the CHECO re-download now owes **3 URLs, not 1**"; **that was wrong in both
  directions and is withdrawn** — see the next bullet.
- **Agenda items with no document of their own
  (`tools/corpus_prep/scan_duplicate_bodies.py` →
  `docs/orphaned-agenda-items.md`, 2026-08-09)** — the class the title-body audit
  is *structurally* unable to see: it scores each title against **its own** body,
  so when two items in a meeting share one document both score against that one
  subject line and both pass (`2564/ครั้งที่ 5` items 20 and 21 score 0.692/0.583
  against a subject line naming a third faculty). Compare items **to each other**
  instead. Two signals, because the obvious one undercounts: (A) identical OCR
  text — exact, but only fires when the *same* PDF was fetched twice; (B) a shared
  page-1 `เรื่อง` subject line within a meeting, which catches the case where the
  source holds two separate exports of one document and the OCR differs by a few
  characters. **Compare the subject at full length, and that was calibrated, not
  guessed**: a 60-char prefix reports 1,255 orphans (44% of the corpus) because
  the faculty that distinguishes two curriculum items falls past the cut; the
  count collapses 229 → 28 groups between 60 and 80 and is flat from 100 to full
  length. Strip `__N` (ADR-0004 piece index) *and* ` (N)` (the download stage's
  re-fetch suffix) before asking whether two files are different items — adding
  the second one alone moved the headline from 11 groups/16 items to 9/11.
  **The ticket's premise ("re-download the 3 never-fetched documents") was wrong
  on every count.** Result: 8 groups, 10 items flagged, **9 genuine orphans**, of
  which **exactly 1 was repairable** — and CHECO, the one it named, was already
  fixed. **The deciding question is not what the corpus holds (that only shows a
  duplicate) but what the recorded Drive id serves *now*** — so page 1 of all 21
  ids was fetched and rasterised. Three mechanisms, only the second fixable:
  (a) *the source lists one document under N ids* (7 orphans across 6 groups;
  `2564/ครั้งที่ 5` does it under **four**) — no fetch can produce the missing
  document, and the manifest title is simply unsupported; (b) *wrong blob
  attached* — the CHECO mechanism, **`2566/ครั้งที่ 3`, REPAIRED 2026-08-09** via
  `refetch_mispaired_document.py` (re-download from the id the manifest already
  held + re-OCR through `ocr_pdf_to_md.process_pdf`); (c) *dead id* —
  `2568/ครั้งที่ 11`'s recorded id 404s at every endpoint. `2565/ครั้งที่ 7` is a
  signal-B **false positive**: one *combined* page-1 heading printed on two
  genuinely distinct resolutions. All 21 ids are distinct and the manifest,
  `_LINK.txt` and `master_list.csv` agree throughout — the defect is always
  downstream of the metadata, so there is no alternative id to try anywhere.
  **Unlike the title repair above this is a text change, so it stales every index
  holding that file** — free here only because 0 gold entries in either gold set
  cite any resolution from `2566/ครั้งที่ 3` (checked by exact meeting match; a
  loose substring test gave 2/5 false hits from that college's *curriculum*
  documents in other meetings). Two traps, both hit while doing this: **a 404 is
  not a different document** (Drive answers a missing id with a 1,652-byte HTML
  page, and an early probe reported that as "two distinct documents" — check the
  `%PDF` magic, not the status code); and **pixel-hash equality proves sameness,
  inequality proves nothing** (two exports of one Word print differ by ~178 bytes
  and render to visibly identical but non-identical PNGs, which called
  `2565/ครั้งที่ 6` "distinct" — every "distinct" verdict was confirmed by eye).
- Narrative overview of the whole project (what was done in what order, what each step
  found, which conclusions have since been retracted): `docs/project-journey.html` — the
  tracked source; render to PDF with headless Chrome (`--headless --no-pdf-header-footer
  --print-to-pdf=docs/project-journey.pdf`). The PDF is gitignored (generated).
  `docs/project-journey.md` is only a pointer — **don't** duplicate the narrative into it.

## Agent skills

### Issue tracker

Issues & PRDs live in **GitHub Issues** (`khthana/ThaiRAGForge`), via the `gh` CLI.
See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical state roles using their **default names** (`needs-triage`, `needs-info`,
`ready-for-agent`, `ready-for-human`, `wontfix`), plus GitHub's default `bug` /
`enhancement` category labels. See `docs/agents/triage-labels.md`.

### Domain docs

**Single-context**: `CONTEXT.md` + `docs/adr/` at the repo root.
See `docs/agents/domain.md`.
