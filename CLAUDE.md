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

- **What belongs in this file, and what does not.** This file is the guard that
  stops a closed axis being re-proposed and a settled convention being broken, so
  it keeps **verdicts, bounds, rules and traps**. It is not the research record.
  Three things belong elsewhere and must not be re-added here: (a) the derivation
  behind a closed axis → its `docs/*.md` (which must be in
  `audit_doc_claims.DOCS`, or moving prose there silently drops D2/D5 coverage);
  (b) why a *check* is written the way it is → the **docstring of the script that
  runs it**, where the next person to edit that check will read it; (c) **any count
  a script prints** — a pass/fail tally, a "which reports are stale" list, a
  combos-rebuilt figure — → run the script. It was measured at ~79k tokens per
  session on 2026-08-23, of which only 4% was conventions, and (a)-(c) were folded
  out; the rules are in [[project_claude_md_size_reduction]] and
  [[feedback_derive_the_enumeration_keep_the_judgement]]. Rewriting a line here as
  a pointer is right; deleting a *do not* is not.
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
- **Run `tools/eval/audit_pipeline_invariants.py` before trusting any eval refresh,
  and read its module docstring before editing a check.** Three silent-corruption
  bugs have been found by accident rather than by looking (corpus-discovery
  contamination, stale BM25/hybrid result cache, `resolution_id` collisions); they
  share a shape — **a mismatch between two artifacts produced at different times by
  different scripts, which never crashes, it just makes a number wrong**. The script
  checks 29 such invariants across corpus/index/eval/answer layers and exits 1 on any
  FAIL. Report: `docs/pipeline-invariant-audit.md`. **Do not quote a pass/fail count
  from this file — run it**; the docstring's `Lessons` section carries the reusable
  half (why `BuildCombo.id` is deliberately not corpus-hashed and `E0` identifies
  results instead of renaming indices; why `C4` follows its subject matter to
  `ARCHIVE_ROOT` and the `E3` checks print denominators; why `G1` reads the answers
  rather than a staleness proxy and `G1c` reports *unmeasured* rather than clean; why
  `I6` had to learn to read manifest mtimes; why an unsealed index directory is a
  reported gap). **The two checks to look at after a rebuild** are `I6` (indices
  older than the corpus) and `E4` (results newer than their index) — `E4` at 0 across
  every result file is the mechanical confirmation that a whole refresh chain is
  complete, which the headline count does not tell you.
  Two standing operational facts, neither derivable by running it. (1) **A rebuild is
  not always owed**: `I6` sat red on 40 indices holding a pre-re-OCR file for days and
  that was correct to leave — 0 gold entries in either gold set cite any resolution
  from that meeting, so no published metric could have moved; the user elected to
  clear it anyway (rebuild #4, complete 2026-08-17). Distinguish a **text** change
  (needs a rebuild) from a **title** change (a relabel — `resolution_id` moves,
  chunk text does not; see [[feedback_a_title_change_is_a_relabel_not_a_rebuild]]).
  (2) **All four Qdrant collections are copies of an `Index`'s rows, so any rebuild
  stales them** — re-ingest all four and re-run `qdrant_routed_check.py` **once**, not
  per combo. State and protocol: [[project_index_rebuild_pending]].
  A general title-vs-body check was prototyped and **rejected on measurement**
  (median agreement 0.660 over 2,820 files, 544 below 0.5, nearly all false alarms
  from agenda-number prefixes) — the audit that did work is the asymmetric
  token-containment one in the corpus-data-quality bullet below.
- **Run `tools/eval/audit_doc_claims.py` after editing `CLAUDE.md` or any watched
  doc, and after any eval refresh; read its module docstring before editing a
  check.** It is the docs layer the invariant sweep was missing:
  `audit_pipeline_invariants.py` gates corpus/index/eval and
  `diff_significance_reports.py` gates report-vs-report, but **nothing read the
  prose**, which is where this project's avoidable errors actually live — a number
  typed by hand, correct that day, that no later refresh touches because a refresh
  re-runs scripts and diffs reports. Eight checks (**D1** report older than its
  generator, **D2** every 4-decimal figure in the prose must appear in some report,
  **D3** a p-value quoted against a contradicting verdict word, **D4** an eval *input*
  changed after a report that reads it, **D5** the count/total shape D2 is
  structurally blind to, **D6** every allowlist entry still exempts something,
  **D7** the unit-suffixed shape — `2,058.9 ms`, `9.81 q/s` — which is the *other*
  class D2 cannot see, **D8** a *named quantity* quoted at a value it no longer
  has, over 15 watched quantities) over **13 docs**, plus — for D7 only — **every Python docstring** in
  `src/`, `tools/`, `app/` and `tests/`.
  **D7 landed 2026-08-23 and its unit set is evidence, not taste.** Every candidate
  was scored against its own perturbations, and per unit at ≥3 significant digits
  only `ms` (70% real / 8% at n+1) and `q/s` (67% / 0%) are checks: `MB`, `%` and
  `x` clear a wrong number 30–53% of the time, while `s` (49% real) and `GB` (0%,
  because prose rounds to GB and reports state MB) would go red on *correct*
  writing — equally disqualifying. A rounding tolerance was built and **rejected**
  on the same measurement (real 70→76%, n+1 8→23%). **Neither of D2's exemptions is
  inherited**, and that too was measured: `SUPERSEDED` clears 44% of D7's residue
  *including the one true positive it was built on* — `paper-results-summary.md`
  had quoted a reranker latency from a 73-query run for weeks while the report has
  said 106 since, invisible to every earlier check because a latency is neither
  4-decimal nor a count/total. The residue is allowlisted by exact figure with a
  written reason per class, and **26 of those entries are a real gap rather than a
  false positive**: serving measurements taken while building the caches, the
  warm-up and the seal that no generator persists. Report: `docs/doc-claims-audit.md`; triaged exemptions with written
  reasons in `tools/eval/doc_claims_allowlist.yaml`. **Do not quote its counts here —
  run it**; D3's known false positives and D5's standing residue (about a tenth of
  its figures, *below* its own documented ~36% base rate) are its designed state, and
  both move whenever this file is edited.
  **D8 landed 2026-08-23 and it is the first check here that can see rot D2's own
  haystack shares.** D2 asks *does this figure appear in some report*, so when
  rebuild #4 moved `routed (shipped)` 0.6831 → 0.6811 the prose kept saying 0.6831
  and D2 kept passing, because ten other reports still carried it. D8 asks a
  different question — **is the figure quoted beside this named quantity a value
  the quantity USED TO have?** — and the superseded values are **derived, not
  typed**: the `_*/` snapshot directories hold the previous run of each report, so
  `superseded = union(snapshots) − current`. A block flags only when it holds a
  superseded value and **no** current one, which is what lets a deliberate
  supersession trail ("it was 0.6831, now 0.6811") pass. **It is not any of the
  five refuted currency checks, and specifically not (e)** — that one *excluded*
  snapshot copies from D2's haystack and cost 103 residue; this one *uses* them,
  as a positive signal, in the opposite direction.
  **Its first run found 12 stale claims across four documents that D2 passed**,
  including three present-tense "S2 reproduces X" anchor sentences and a whole
  superseded arm table in `paper-results-summary.md` (the oracle-union rows, the
  soft/hard arms, the rrf4 2×2). All 12 are repaired. **The block is the unit and
  the split is what makes it non-vacuous**: a supersession trail routinely puts the
  old and new values 300+ chars apart so a character window flags correct writing,
  and blank lines alone are the wrong split for *this* file, which writes its
  bullets with no blank line between them — the whole Conventions list becomes one
  block, every figure in the file lands in one bag, and the check passes on
  everything. That is D2's own haystack-too-big rule, one level down, and it is
  pinned by a test. The one class D8 must **not** flag is a **frozen
  pre-registration**, where a figure that no longer matches the outcome is the
  point of having written it down; those are allowlisted by (doc, quantity) and
  D6 audits them like every other exemption.
  **Widened from 5 watched quantities to 13 the same day, and the widening found
  13 more stale blocks** — the reranker-vs-hybrid MRR pair `0.7814 → 0.6778,
  p=0.0012` still sitting in three separate documents (it is **0.7730 → 0.6940 at
  Holm 0.0240**), the whole soft-vs-hard significance table in
  `paper-results-summary.md`, the withdrawn alpha-sweep `+0.0350 recall@10` in two
  places (the `per-type (loo)` arm reads **+0.0281** there now, ns at Holm 0.0870,
  and survives only on nDCG@10 at **+0.0333**), and the routed-oracle `+0.1500 / +0.0017 / 1%` sentence (now
  **+0.1520 / −0.0098 / −6%**). **Two rules came out of that widening, both after
  a false positive rather than from taste.** (1) **A CI is not the quantity**: its
  endpoints are arbitrary 4-decimals that collide with unrelated effect sizes, and
  the first widened run flagged `weighted × fetch_depth`'s F=200 loss against a
  *retired CI bound of the alpha sweep* — two experiments with nothing to do with
  each other. Brackets are stripped, and an optional column cap keeps the p-value
  column out of a significance row. (2) **A label must be one that appears where
  the numbers are**: `per-`entity_type` alpha` is how this file *refers* to that
  axis ("the wrong-pair trap that killed per-`entity_type` alpha"), so it matched
  blocks whose figures belong elsewhere. Conversely **broad is safe and
  English-only is not** — adding the bare word `reranker` is what surfaced the
  three Thai blocks above.
  **Widened again to 15 on 2026-08-23, and these two entries are the first added
  BECAUSE a snapshot exists rather than in spite of one.** Refreshing a report is
  double-edged for this family: a `_pre_*` snapshot is an **exemption** under D2
  ("in a dated snapshot") and the **evidence** under D8, so a report that is
  snapshotted and *not* watched here is the worst of both — its old figures gain an
  alibi and nothing gains the means to catch them. `multi_k_report.md` is the
  counter-example and the rule: it has **no** snapshot, so an entry naming it would
  derive an empty superseded set and sit in the registry looking like coverage.
  **Check for a snapshot before adding a quantity.** Adding the two fetch-depth
  reports immediately found **10 stale blocks a targeted grep had missed**, four of
  them in `README.md` and `docs/code-explained.html` — two files that hold published
  figures and that the refresh had not prompted anyone to open. Two mechanism rules
  came out of it, both after a FAIL on correct writing. (1) **`fetch_depth` is not a
  label**: it names three different experiments here (the unrouted sweep, the routed
  test, the qdrant request depth), so it matched the *routed* test's own current
  `+0.0005` against the sweep's retired `−0.0005` — the same failure the alpha
  sweep's label had, one report along. (2) **A row set that misses the quantity's
  headline column is worse than no entry**: the first `weighted` entry watched the
  zeroed-term and comparison tables but not the main depth table, so its own
  headline was unwatched and a correct block read as having no current value at all.
  **The measurement that motivated D8 is worth more than D8, and it is about D2:
  scored against its own perturbations, D2 clears a wrong 4-decimal number 77% of
  the time** (over its 2,970 figures; D7, for contrast, clears 7%). D2 was never
  calibrated that way — the perturbation method arrived with D5 and nobody went
  back. **Do not read "D2 passes" as "the prose traces"**; read it as a weak
  filter, with D8 as the sharp one over the quantities it watches.
  **D7 now reads Python docstrings too (2026-08-23), and D2 deliberately does
  not.** A docstring is prose and was outside every check here: re-quoting
  `warm_serving_caches`'s figures from a report found the *same* superseded pair
  still sitting in the docstring one layer down and in the test that pins it.
  Extending only D7 was measured, not scoped by taste — over these docstrings the
  ms/q-s rule scores **61% real / 15% at n+1** (a check) while the 4-decimal rule
  scores **96% / 71%**, i.e. exactly as weak as D2 already is. The sweep took the
  untraceable docstring figures **24 → 6**, and the 6 that remain are two
  deliberate classes: the pre-fix `657 ms` engine-probe reading (three copies, its
  post-fix 197.5 ms quoted beside each) and the discarded 2026-08-07
  position-effect run plus the 3-token synthetic that produced the withdrawn "26x".
  **A docstring edit moves the generator's mtime, so D1a and D4 go red on a
  comment-only change** — the discharge is to `--render` the affected reports and
  check they come back **byte-identical**, which all three did; that identity is
  the proof the edit changed nothing, where an allowlist entry would only have been
  an assertion.
  **Its first run found three real stale tables, and how it found them is the point**:
  all three had drifted in the 2026-08-06 refresh *without a single verdict cell
  changing*, so `diff_significance_reports.py` correctly reported 0 flips and nobody
  re-copied the numbers — including a summary that openly disagreed with CLAUDE.md
  (4 of 8 vs 5 of 8) and a structural-ceiling table never extended when the 33
  `course` queries landed. **The widening from 2 docs to 12 (2026-08-20) was decided
  by measuring the cost first**: 8 of 10 candidates added zero D2 residue. Three files
  stay **out**, each because including them makes the check *vacuous rather than
  thorough* — `chunker-embedder-comparison-log.md` is append-only (a stale number in a
  log **is** the record), `reranker-hybrid-interaction-research.md` quotes 211 figures
  from the *literature*, and pre-registration sections state **predictions**, where a
  figure that no longer matches the outcome is the point of having written it down.
  **What the widening found is the class this file keeps getting hurt by — one layer
  disagreeing with another**: the RQ4 entity arms were stale in three layers at once
  and carried two unrecorded verdict flips, while CLAUDE.md contradicted *itself*
  between two of its own bullets. Two triage rules from that work:
  **reproduce an artifact's own pipeline before calling it wrong** (a raw-text recount
  "refuted" a figure that reproduces exactly once the generator's own stripping is
  applied), and **match the whole composite key, never one component** (a
  `resolution_id` is `<year>/<session>/<title>` and a title-only match reports a false
  hit on a different meeting carrying the same title text).
- **REPORT CURRENCY — which published numbers describe the current indices.
  RUN `tools/eval/report_currency.py`; do not read a list typed into this file.**
  It derives the answer from each index manifest's own `timestamp` against each
  report's mtime → `data/results/report_currency.md`, and excludes what a date test
  would otherwise keep permanently red: `_`-prefixed snapshot directories, and
  `RETIRED_REPORTS` **imported** from `audit_doc_claims.py` rather than re-listed.
  **The comparison is per index ROOT, not against the newest build anywhere, and
  the first version got that wrong**: `entity_tags_full` was rebuilt after the
  corpus last changed, so it is current — but rebuild #4 finished five days later
  on a *different* root, and the global maximum called both `gold_entity_*` reports
  stale when the index they were scored on had not moved. A report naming a root in
  its own text is judged against that root (*attributed*); one naming none is
  *screened* against the global newest, which is conservative and marked as such,
  because screened and attributed are different claims.
  **This bullet used to carry the list by hand and it was wrong in both directions
  inside four days**, contradicting itself in two places at once by 2026-08-23
  (it called the reranker family both current and stale, and called the two
  `rq4_score_gemma4*.md` stale four days after they were re-scored). *A to-do list
  written into living guidance is a claim that needs re-verifying like any other* —
  and the derived version immediately found reports the hand list had missed (five
  pre-9-way tables from 2026-07-21, `multi_k_report.md`) and one it wrongly
  accused (`gold_anchor_ambiguity.md`, current since the 08-21 re-run).
  **What the script deliberately does NOT decide: whether a stale report is worth
  refreshing.** "Stale" is a fact about a timestamp; "worth refreshing" is a
  judgement about whether a verdict could flip or a published bound could sharpen,
  and lumping the two together is how a list nobody can ever clear becomes a list
  nobody reads. Judgements already made live in the script's two exemption tables
  **with their reasons** — `NOT_WORTH_REFRESHING` (the whole HyDE family:
  directional losses an order of magnitude larger than anything rebuild #4 moved,
  so no verdict can flip and there is no bound to sharpen) and `CORPUS_INDEPENDENT`
  (the two model-qualification reports and the pylate cross-check, which gate on
  hand-written probes). An entry naming a missing file is reported as **BROKEN**,
  because an exemption list is the easiest way to make a check vacuous.
  **The reusable finding, and it is a real hole in the D family.**
  `audit_doc_claims.py`'s **D2** asks whether a figure in the prose appears in
  *some* report under `data/results/**/*.md` — a union **regardless of currency**.
  So when rebuild #4 moved `routed (shipped)` from 0.6831 to 0.6811, the prose kept
  saying 0.6831 and D2 kept passing, because ten *other* reports still carried
  0.6831 — **and they carried it for exactly the same reason the prose was wrong**.
  Sharper instance: `qwen3_0.6b` `program` dense recall@10 moved 0.6066 → **0.6034**
  and the prose was traceable to precisely the stale ColBERT reports that were wrong
  the same way. **A traceability check cannot detect rot that its own haystack
  shares** ([[feedback_a_traceability_check_shares_its_haystacks_rot]]); D1a cannot
  see it either, because no generator changed — the *indices* did. Note the
  direction this cuts: **refreshing a stale report is what makes stale prose
  detectable**, so a refresh is not only a currency fix, it removes a figure's false
  alibi. Honest fixes are (a) re-run the stale generator, (b) let D2 prefer a report
  newer than the last index build, or (c) **date the claim in the prose** — and
  until (b) exists, **(c) is the standing convention in this file**, which is why
  every refreshed figure above carries the date of the run it came from. Four other
  candidate currency checks were built and **all four refuted** — see the doc-claims
  bullet; the root cause is structural, D2 is a bag of numbers, not a
  quantity→value map.
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
  and C (`person→replaces→person`) were out of scope pending a decisive measurement;
  **that measurement is done and they are NOT being built (2026-08-10 — see the RQ4
  entity-arms paragraph below for the numbers and the bound).** **No gold query
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
  free. **Edge A: 180 of 250 programs resolved, 56 ambiguous, 14 no_evidence**
  — and the `ambiguous` bucket is **not one thing**: only **8** have two faculties
  genuinely pointing at each other, the other **48** have a single faculty with
  fewer witnesses than `min_votes=2`.
  **The denominator is 250 and not 253 as of 2026-08-21, and the change is a
  DEFINITION not a measurement — never read the two side by side as drift.**
  `programs.json` holds 253 canonical names for 250 programmes: KOSEN renamed
  three associate degrees in 2568 and both names must stay in the dictionary so
  `match_programs` still matches documents written under either. Keyed on
  *entries*, the graph counted those three twice **and split their evidence**,
  which is not cosmetic — one half of the แมคคาทรอนิกส์ pair had a single witness
  and sat in `ambiguous` while its twin resolved on four votes at the *same*
  faculty. `build_graph` now iterates `programme_groups()`, pools each group's
  votes onto its first entry in dictionary order and keeps the rest as `aka`.
  The whole delta is those 6 entries (5 resolved + 1 ambiguous → 3 resolved), on
  **byte-identical evidence** (the 08-11 cache), so this is not a fourth matcher
  repair. **The check had to follow its subject**: `S2` gated
  `len(records) == len(programs)` and would have FAILED on a correct graph, so it
  now compares against the group count and prints both numbers
  ([[feedback_cleanup_can_break_an_audit]]); new **`S5`** gates that every one of
  the 253 names is still reachable as a node or an `aka`, because grouping's real
  risk is not a wrong merge (`programme_groups` is tested for that) but an entry
  going **missing**, which would shrink every denominator here while every other
  check still passed. This is also what made `programme_groups` a live function
  rather than one only its own tests called. The §3 cross-checks are deliberately
  **left per-name**: they ask whether two sources agree about one written name,
  which is a different question from counting programmes.
  **The pre-grouping figures are kept because the three walks below are stated
  on them, and are superseded: 182 / 57 / 14 of 253 (split 8 / 49). They are
  the SECOND 2026-08-11 re-walk's, and both re-walks were forced rather than
  routine**: this script calls
  `match_programs`, so each half of the matcher repair below moved the graph
  without touching its own generator — the two-artifacts-from-different-days shape
  again, now guarded by a `program_loader.py → relation-graph.md` edge in
  `audit_doc_claims.py`'s `EVAL_INPUTS`. Both moves went in the predicted
  direction: **170 / 60 / 23** (split 9 / 51) before either repair → **177 / 62 /
  14** (12 / 50) after the degree half → **182 / 57 / 14** (8 / 49) after the
  cross-subject half — all three on the 253-entry denominator, so compare them
  with each other and not with the 250-programme figures above. A rescued tag returns evidence to the program that actually
  owns it, which is why `no_evidence` fell by 9 on the first walk; a *dropped*
  absorption stops lending a foreign programme's votes to its neighbour, which is
  why `ambiguous` fell by 5 on the second. **The motivating case of the whole
  audit visibly moved**: `หลักสูตรแพทยศาสตรบัณฑิต` — the row that held both
  `ทันตแพทยศาสตรบัณฑิต` and `พยาบาลศาสตรบัณฑิต` — left the `ambiguous` table
  altogether and is now `resolved → คณะแพทยศาสตร์` on 8 votes. The two
  cross-checks moved too (see below) and all **five** self-checks still PASS. Two
  distinct causes underneath, and the
  first is the important one: **`program → faculty` is not a function** —
  `วิศวกรรมเครื่องกล` really is offered by both `คณะวิศวกรรมศาสตร์` and
  `วิทยาเขตชุมพรเขตรอุดมศักดิ์` (23 vs 18 votes), so any graph forcing one faculty
  per program is wrong for that group by construction. The second is a matcher
  finding worth its own ticket: **`match_programs` has no "matches nothing" exit
  for a near-miss**, so a corpus name absent from the dictionary is absorbed by
  its nearest neighbour — `หลักสูตรทันตแพทยศาสตรบัณฑิต` *and*
  `หลักสูตรพยาบาลศาสตรบัณฑิต` both match `หลักสูตรแพทยศาสตรบัณฑิต`, which accounts
  for that whole ambiguous row. Scope was **measured, not guessed**: 0 of 253
  dictionary names collide with each other, so the collision is only with names
  outside it. **Both halves are now REPAIRED (2026-08-11) — the degree half in
  the paragraph after next, the same-degree cross-subject half (exactly the
  dental/nursing row above) in the one after that.** `match_programs` is also read by
  `build_gold_candidates.py` and `router`, so moving the threshold would move
  published numbers. **That blast radius was MEASURED and it is zero on both
  call sites (2026-08-10, `tools/eval/audit_program_matcher_absorption.py` →
  `docs/program-matcher-absorption.md`, ~23 min walk cached so `--render` is
  free).** Corpus-wide the defect is large — 9,141 accepted matches over 1,710
  files, **23.1% (2,114) absorb a genuinely different name**, 210 of 249 matched
  canonicals absorb at least one — and its dominant shape is the one worth
  naming: **35.7% of absorptions swap the degree level** (บัณฑิต ↔ มหาบัณฑิต ↔
  ดุษฎีบัณฑิต), one token apart so the ratio stays far above 0.82 while a
  master's programme is tagged as the bachelor's of the same subject. **But it
  reaches neither published path, and both were verified rather than assumed**:
  `program_candidates()` never calls `match_programs` for membership — and it
  never reads the *tags* either, which is stronger than this file used to say
  (the old "seeds from tagged files" wording overstated the coupling, corrected
  2026-08-12). It iterates `programs_by_file.json`'s **keys**, and
  `tag_programs.py` writes a key for every live corpus file including the
  zero-match ones, so the pool *is* the corpus and the only gate is `canonical
  in resolution_id`, an exact substring of the manifest title. The matcher
  therefore cannot move the program qrels **structurally**; only corpus
  *membership* can. That is executed, not argued — `S2` in
  `tools/corpus_prep/audit_program_tag_regeneration.py` blanks every value and
  requires identical output — and the independent second route still holds:
  **0 of 30** program queries' gold pairs
  have a `resolution_id` failing to contain the program; and `classify_query`
  asks only whether *any* program matched, never which one, so a name-for-name
  swap **cannot** change a route (33/13/30/30 exact, 0 program queries routed
  elsewhere).
  **The degree-level swap was then REPAIRED (2026-08-11), and the useful part is
  how the rule was chosen: three candidate rules were each walked over the whole
  corpus, and both losers were rejected by their own numbers.** The naive
  *reject* (drop the tag whenever the text's degree contradicts the winner's)
  and *fallthrough* (take the next candidate) both assume the guard's job is to
  decide keep-or-drop. It is not: of the **752** mentions where a winner's degree
  is contradicted, **354 (47%) had a same-degree candidate whose subject the text
  also supports already sitting in the dictionary** — the matcher had not run out
  of options, it had **ranked** them wrong. So what ships is *select*: the degree
  **filters the candidate set** and the best surviving candidate is re-selected,
  which strictly dominates reject (**134 tags gained / 340 lost / 44 files
  stripped bare**, against reject's 0 / 340 / 71). **The subject test inside the
  rescue is load-bearing, not belt-and-braces**: 213 of the 752 have a
  same-degree candidate that disagrees on subject, so filtering on degree alone
  would hand `...ดุษฎีบัณฑิต ...ไฟฟ้า` to `...โยธา` — trading a wrong degree for a
  wrong subject. The remaining 129 are **undecidable** (one side has no
  `สาขาวิชา`, so no evidence means no rescue, [[feedback_undefined_is_not_zero]])
  and 56 have nothing at the text's own degree, where *matches nothing* is now
  finally an available answer. **Blast radius re-measured after the repair, not
  inherited**: 0 of 247 published program gold pairs move and the router stays
  33/13/30/30 — which was a counterfactual when it was written, because
  `build_gold_candidates.py` reads the **cached**
  `academic_resolutions/entity_tags/programs_by_file.json` (2026-07-25), so
  nothing published moved until that artifact was regenerated. **It was
  regenerated 2026-08-12 and the counterfactual is now closed as a real
  measurement** — see the regeneration bullet below; the 0 and the 33/13/30/30
  both reproduce against the new artifact.
  **The same-degree cross-subject half — the dental/nursing shape the degree
  filter is structurally unable to see — was then CLOSED the same day, and the
  mechanism is the reusable part: dilution by concatenation
  ([[feedback_a_similarity_over_a_concatenation_dilutes]]).** The ratio runs
  over head noun + subject *joined*, so a disagreement confined to one half is
  averaged away by the other half agreeing — `ทันตแพทยศาสตรบัณฑิต` against
  `แพทยศาสตรบัณฑิต` scores **0.88**, comfortably over the 0.82 threshold, purely
  because the shared `แพทยศาสตรบัณฑิต` outweighs the `ทันต` prefix. Testing the
  two halves separately (`_head_contradicted` / `_subject_contradicted`) removes
  the dilution. **The two halves need different tests, and the asymmetry is
  structural, not a fudge**: `_bounded_span_for` sizes the window from the match
  position, so the head noun is always covered in full while the subject sits at
  the tail and is routinely truncated — truncation is therefore *forgiven* in
  the subject (longest-common-substring coverage, so a cut name still scores
  1.00) and **not** forgiven in the head (extra leading material is a different
  formal degree name). Result: the guard fires on **606** of 9,141 accepted
  matches and is mostly **drops** (12 rescued, 594 dropped: 159 head, 435
  subject) — which is itself the finding, because where a degree swap usually
  has the right answer one level away in the dictionary, a cross-subject
  absorption usually does not; the absorbed programme is simply **not in
  `programs.json`**, and *matches nothing* is the right answer. Per file, both
  guards together: **140 gained / 594 lost / 115 stripped bare / 446 changed**.
  Blast radius unchanged and re-verified rather than inherited (§4 **0 of 30**
  program gold queries, §5 router **33/13/30/30**), and the degree half's own
  figures reproduce **exactly** (752 → 354/213/129/56), which is why the two are
  reported as separate rows and never merged. **Two rules were added after the
  first cross-subject walk, both because a self-check went red — the corpus was
  fine and the guard was wrong, twice.** (1) **A drop needs *both* subject tests
  against it.** Coverage is blind to a difference spread through the string
  rather than sitting at one end, so a one-character OCR/dictionary variant
  (`วิศวกรรมเล็กทรอนิกส์` vs `วิศวกรรมอิเล็กทรอนิกส์`) covers 0.55 while
  `_field_agrees` — this project's own settled test for the same relation — puts
  it at 0.95. **One pair must not be agreeing and contradicting at once**, so a
  drop now needs coverage **and** ratio against it; **133 of 569** subject drops
  sat in exactly that band ([[feedback_agreement_and_contradiction_are_one_relation]]). (2) **A cross-subject rescue may not cross the
  degree**: that branch is reached precisely when the *winner's* degree is
  uncontradicted, which says nothing about the *runner-up's*, and 6 rescues in
  the first walk moved a mention between บัณฑิต and มหาบัณฑิต — a cross-subject
  rescue silently undoing the settled degree rule (now `S9`). **The unit test for
  (2) had to be taken from the corpus, not invented**: two brute-force searches
  over degree/subject/tail grids returned **0** synthetic fixtures, because a
  candidate that agrees on the subject is normally textually closer than the
  contradicted winner and simply wins outright — the real mention
  (`2564/ครั้งที่ 11`, where `<br/>` markup inflates each candidate's window
  differently) is what reaches the branch. **What is still NOT fixed**: nothing
  in either guard gives `match_programs` a *lexical* notion of subject identity,
  so a subject the window truncates past recognition is still undecidable rather
  than wrong. **The measurement that decided all this was
  itself wrong once and the correction is the reusable part**
  ([[feedback_a_guards_precondition_biases_its_own_test]]): the first A/B asked
  "does the text support the **subject** of the tag the guard removed?" and
  answered 573 loss / 55 fix — an instrument that cannot return the other answer,
  because every firing has a contradicted *degree* by construction, so the
  subject is always the half that agreed. **Its first
  run was wrong in a way worth remembering**: it reported "99.3% absorb a
  foreign name" with the inserted-character distribution's mode at exactly
  **4** — which is `_WINDOW_SLACK`, i.e. it was counting the matcher's own
  read-ahead window as absorbed text
  ([[feedback_a_mode_on_a_constant_is_your_instrument]]). `S5` now pins that a
  pure window tail scores 0 (6,333 such spans) while a longer tail still
  registers its excess. **And the same trap caught the repair's own check**:
  `S7` (every rescue agrees with the span's subject) FAILED on **4 of 354**
  because it judged the rescue against the span sized to the **winner's** length,
  while `_bounded_span_for` sizes the window to *each candidate*, so a rescue
  onto a longer sibling name (`...การจัดการโลจิสติกส์` →
  `...การจัดการโลจิสติกส์และซัพพลายเชน`) read further into the text than the
  check let it. The record now carries `selected_span`, and the 4 are reported
  as a fact about the dictionary (`S8`) instead of a violation that never
  happened — **diagnose a red self-check from the cache before editing either
  the rule or the check**; both times here the instrument was wrong and the
  corpus was fine. **`entity_tags_full` was deliberately NOT rebuilt at first,
  and was then rebuilt 2026-08-12 together with the cached artifact — never
  alone, and that coupling is the rule to keep.** `entity_loader.py` is a third
  call site, so leaving the index at its 2026-08-05 build held pre-repair program
  tags in front of `entity_lookup`/`entity_boost` and the published RQ4 entity
  arms, while rebuilding it *alone* would have decoupled its tags from the qrels'
  own 2026-07-25 cached tags in an unmeasured way. So both moved together:
  `programs_by_file.json` regenerated (see the bullet below) and the index
  rebuilt from it (71,073 chunks, `docset_hash 7a274096d8609f61`), then both
  entity result sets re-scored. **Only the program-bearing rows moved, which is
  the built-in control** — the person/course/faculty loaders were untouched, and
  under `entity_lookup` (pure set membership) their scores are identical to 4
  decimals while `program` goes 0.8918 → **0.9013** and overall 0.9422 →
  **0.9449**; `entity_boost` `program` recall@10 0.5765 → **0.5834**, with
  person/course moving ±0.007 in *both* directions because a tag line is part of
  chunk *text*, so changed program tags perturb the embeddings and BM25 of the
  same documents even for a person query. **This refuted a pre-registered
  prediction**: recall was predicted to fall, since the cross-subject guard cuts
  far more tags than it adds (594 vs 140) — but the degree guard *re-selects* a
  same-degree candidate instead of merely dropping, so a rescued tag lands on the
  programme that actually owns it. The gating verdict in
  [[project_rq4_entity_arms_gating]] was **re-measured, not inherited** (RQ4
  entity arms re-run 2026-08-12 — only the cells whose context actually changed
  were regenerated, the rest frozen byte-for-byte, per
  `docs/rq4-design.md`): it is
  unchanged, resting on **−0.2523** at Holm 0.0000, a *ranking* failure over
  already gold-dense contexts (**0.6501**), which better program tags were not
  expected to close — and did not. Rebuild it only together with a
  regeneration of `programs_by_file.json`, and re-run the RQ4 entity arms if you
  do. **Edge A′ is ~7x smaller than the scan note claimed and the
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
  reported and never gated: manifest title vs OCR'd body agree on **170 of 181**
  programs both can name (two independent text sources — typed vs scanned), and a
  split-half over disjoint document sets agrees on **105 of 115**. Read them
  across all three walks rather than as a level, because the two repairs pull the
  denominator in opposite directions and neither is a quality signal on its own:
  manifest-vs-body 158/169 → 165/180 → 170/181 (93.5% → 91.7% → **93.9%**),
  split-half 103/112 → 110/118 → 105/115 (92.0% → 93.2% → **91.3%**). The degree
  half *rescued* tags so both denominators grew; the cross-subject half is mostly
  *drops*, so the split-half denominator fell 118 → 115 — a programme no longer
  named in both halves is one the guard stopped inventing, which is the intended
  effect and not a loss of coverage. Agreement stayed inside ~2 points throughout.
  Four self-checks
  (S1 every faculty node is canonical, S2 `no_evidence` stays *undefined* rather
  than a low score, S3 a window-extracted faculty must also appear in the
  document's own tags, S4 every A′ name must be one `find_people` found in that
  same file) all pass; **S1 first reported a false FAIL from operator precedence**
  (`{a} | {b} - dict` parses as `{a} | ({b} - dict)`), and **S2's first version was
  vacuous** because the graph is built by iterating the dictionary, so "the buckets
  add up" was true by construction — it now gates on the buckets staying
  *distinguishable*. `docs/relation-graph.md` is in `audit_doc_claims.py`'s
  `ARTIFACT_FILES` so its figures can be cited in prose.
- **`programs_by_file.json` regeneration, and the counterfactual it closed
  (2026-08-12, `tools/corpus_prep/audit_program_tag_regeneration.py` →
  `docs/program-tag-regeneration.md`).** The cached tags dated **2026-07-25**;
  `match_programs` was repaired 08-11 and the corpus moved three times in
  between, so the bullet above could only ever say "nothing published moves
  *until that artifact is regenerated*". It is regenerated, and the useful part
  is that **drift and repair were separated instead of reported as one delta**.
  Overall 07-25 → 08-12 is 500 files changed, **+212 / −617** tags, 4,743 →
  4,338; but that splits into **(B) corpus drift** (old matcher, new corpus:
  109 files, +95/−46) and **(A) the matcher repair** (new corpus, old → new
  matcher: 446 files, +140/−594), and the (A) row **reproduces
  `docs/program-matcher-absorption.md`'s own per-file figures exactly** (S3) —
  which is what licenses reading (B) as drift rather than as a residue of a
  mis-modelled repair. The 08-08 title repair cannot appear in either row and
  that is a fact about the pipeline, not an omission: `tag_programs.py` keys on
  the **file path** and matches over **body text**, so a title change touches
  neither. **The blast-radius claim got stronger, not merely confirmed**:
  `program_candidates()` iterates the mapping's **keys** and never reads a tag
  *value*, so the matcher cannot move the program qrels **structurally** — and
  that is *executed*, not argued (S2 blanks every value and requires identical
  output: 147 candidates either way). Measured anyway, both arms agree
  147/662 with 0 moved; **0 of 247** scored program gold pairs lost; router
  **33/13/30/30**. Two triage rules worth keeping. (1) **Diff against the
  artifact that is actually published**: 1 candidate differs from
  `gold_candidates.json`, which is a **superseded intermediate** (07-25,
  regenerated 07-30 for the `resolution_id` fix) — the truncated
  `2567/1/…(หลักสูตรนานาชาติ)` id became the two full `…๒๕๖๓`/`๒๕๖๔` ids and
  `gold_query_set_73det.yaml` already holds both, so S6 gates on *scored pairs*
  rather than on that file. (2) **Corpus membership is the only channel that
  can move a candidate**, so S4 watches it: the corpus gained exactly one file
  (`2568/ครั้งที่ 7/เรื่อง รับรองรายงานการประชุม.md`, the 08-09 CHECO
  restoration), it matches zero programs and its `resolution_id` holds no
  program canonical, which is why membership moved and the qrels did not.
  `docs/program-tag-regeneration.md` is in `audit_doc_claims.py`'s
  `ARTIFACT_FILES`, and `program_loader.py` now names it as a third `EVAL_INPUTS`
  consumer — the whole report is a function of the matcher, so a future repair
  turns it into a record of a matcher that no longer exists with nothing else on
  disk saying so.
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
  and **the deployable claim changed sign at rebuild #4 (re-run 2026-08-18) — read
  the date on any routing number.** Until then no deployable arm beat the best
  single combo on either retriever and this bullet said so; that still holds on
  **dense** (shipped +0.0500 Holm 0.1608, LOO +0.0317 Holm 0.4424, both ns, m=18)
  but under **hybrid** both now clear the bar: `routed (shipped)` **+0.0581**
  (Holm **0.0480**) and — the one that matters — `routed (loo)` **+0.0825**
  (Holm **0.0000**), so the *generalisation* estimate is the stronger of the two
  rather than a discount on a fitted one. The claim is now **beats a well-chosen
  single index under hybrid, matches it under dense, and closes a 43% coverage hole
  either way.** Levels: hybrid best-single **0.6229** / shipped **0.6811** / loo
  **0.6794** / oracle **0.6863**; dense 0.5673 / 0.6173 / 0.5989 / 0.6277.
  **Two things moved with it and both belong in the sentence.** (1) **Most of the
  widening is the baseline falling, not the router rising** — the hybrid
  best-single combo is `sentence × qwen3_0.6b` before *and* after (its identity did
  not change), but it fell 0.6281 → **0.6229** while the routed arm fell only
  0.6831 → **0.6811**, taking the margin 0.0549 → 0.0581 across a bar it had been
  sitting just under (Holm 0.0672 → **0.0480**). A weaker baseline as much as a
  stronger router; state both, and note the pre-rebuild margin was already +0.0549
  — the "+0.0408, p=0.1152" this file used to quote was older still. (2) The **dense
  oracle lost two of its three significant metrics**: it was significant on all
  three (+0.0586 / +0.0642 / +0.0701) and is now nDCG@10 only (**+0.0744**, Holm
  0.0126; recall@10 +0.0605 Holm 0.0640, MRR +0.0681 Holm 0.0980). An oracle is
  still not a system, so nothing deployable rests on it — but on two metrics of
  three that headroom is now a bound, not a result.
  The old "under hybrid the gain shrinks to ns (+0.0408, p=0.1152) because BM25
  partly rescues the misrouted queries" is **superseded** by the paragraph above;
  the mechanism it named is real, it is simply no longer enough to erase the gain. Ordering inside `classify_query` is load-bearing: course is
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
  `routed (oracle)` by construction (dense 0.6173 vs 0.6277) — **cite `routed (loo)`
  as the generalisation estimate**, and note it was *unchanged* by that refresh
  (+0.0349 dense / +0.0499 hybrid, both ns at the time) because it never read the
  constants. **Rebuild #4 then moved it, and only on one retriever: dense +0.0317
  still ns, hybrid +0.0825 at Holm 0.0000.**
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
  **Re-run 2026-08-20 against rebuild #4, and the headline lost half of itself.**
  Findings, at current values: (a) **a single global alpha is worth nothing** where
  0.50 was already sane (+0.0066 / +0.0217 recall@10, both ns, and both are *oracle*
  values fitted on the test set); (b) **a per-`entity_type` alpha now survives
  leave-one-out on nDCG@10 ONLY — +0.0333, Holm-adj 0.0392 (m=9, on
  `sentence+qwen3_0.6b`) — while recall@10 went ns at +0.0281, Holm 0.0870.** It was
  +0.0350 at 0.0252 before the rebuild, so **the "+0.0350 recall@10" this bullet
  published is withdrawn**; MRR is still ns (+0.0369, 0.5016) — don't include it.
  Read the surviving claim as *a per-type alpha reorders the top-10 better without
  putting more gold into it*, which is what an nDCG-only win means. The oracle
  `per-type best` arm is significant on all three (+0.0456 / +0.0547 / +0.0560),
  MRR newly so — **but an oracle is not a system**, so that is a ceiling; (c) the
  per-type optima are so far apart that `person` (best 0.15, plateau 0.00-0.35) and
  `program` (best **0.70**, plateau **0.45**-1.00) have **disjoint** non-degrading
  ranges and the shipped
  0.50 sits *outside* `person`'s — **this survived and the gap widened** (program's
  plateau used to start at 0.40); (d) **the gain is conditional** — it needs the two
  arms' relative strength to *invert* across query types, so `semantic+bge_m3` gains
  nothing (ns everywhere, and its per-type LOO arm is now *negative* on MRR and
  nDCG@10; it is the `person` specialist, its dense arm has no
  per-type weak spot) and `fixed_size+m2v` wants alpha=0.00 outright (drop the
  broken arm; per-type adds only **+0.0110** over global). Report **ranges, not a single
  best value** — tuning alpha on the 106 queries it is reported on is overfitting,
  which is what the LOO arm exists to bound. Nothing is changed in shipped defaults;
  `HybridRetriever` still ships 0.5/0.5. **Decided 2026-08-08 not to wire a
  per-`entity_type` alpha into `query_service` at all**, and the reason is a
  wrong-pair trap worth remembering: the motivating gain (then +0.0350 recall@10,
  now the nDCG-only +0.0333) is measured against *no
  routing*, which stopped being the shipped configuration the same day. Against the
  hard router that now ships it shows **no gain on any metric** (recall@10 −0.0202,
  MRR +0.0182, nDCG@10 +0.0066, all ns, m=12 — pre-rebuild-#4 figures; the decision
  only got safer when the motivating recall gain went ns) and the entire remaining
  headroom is the oracle gap **+0.0071**. Mechanism, so this isn't read as a power problem:
  per-type alpha repairs a per-type weak dense arm, and hard routing already hands
  each route a specialist index that doesn't have one (`person` alpha* moves
  0.15 → 0.30, *toward* neutral, once routed). **The one branch that flips it** is
  deployment cost: if 5 indices is too many, the move is to *replace* hard routing
  with soft (arm B, one index, **0.6510**, ns vs hard) — a cost decision, not an
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
  target that actively hurt. The soft arm never moved. **Numbers below are the
  2026-08-18 re-run against rebuild #4** (the pre-rebuild ones were 0.6831 / 0.6631 /
  0.6281 with two significant cells): **hard** (per-route index, 5 indices)
  **0.6811** recall@10 > **D both** 0.6648 > **soft** (per-route fusion weight,
  1 index) **0.6510** > **neither** **0.6229**. **Now ONE significant cell, not two**
  — `hard vs none` recall@10 **+0.0581** (Holm-adj **0.0264**, m=12) survives, and
  `soft vs none` nDCG@10 **lost it** (+0.0333, Holm 0.0528, was +0.0360 at 0.0216).
  **So soft routing no longer owns a significant result anywhere in this table**, and
  it is now numerically below doing both, where before it was above. **Soft vs hard
  is still ns on all three** — CI rules out soft beating hard by more than **0.0060**
  recall@10 (was 0.0156, i.e. the bound tightened a lot), and hard beating soft by
  more than **0.0687**. So: hard leads numerically everywhere and owns the only
  significant cell, and it is *still* **not shown to beat soft**, at a cost of 5
  indices to soft's 1. The cost-per-point argument for soft is weaker than it was;
  "soft is at least as good" is still not refuted. Arm C reproduces `routing_eval.md`'s
  hybrid `routed (shipped)` to 4 decimals (**0.6811**) from an independent code path.
  Note arm C's targets are fitted on this same set, so cite `routing_eval.md`'s
  `routed (loo)` (**0.6794**) as the hard arm's generalisation estimate — still above
  soft. **Still substitutes, not complements**: doing both
  (0.6648) is *below* hard alone and `D vs C` is negative (**−0.0163**, CI
  [−0.0327, −0.0021] excludes zero, ns after Holm) — yet at the oracle bound D′
  (**0.6909**) is the best arm in the table.
  There *is* a sliver of headroom for alpha on top of routing, and LOO fitting costs
  more than the sliver is worth (the pre-refresh version had D worse than B even at
  the oracle, i.e. no headroom at all). Per-route, **hard now wins every route**
  (person +0.1091, program +0.0504, course +0.0299, faculty +0.0300) where before it
  won only course and faculty and *lost* `program` by −0.0784 — that one route was
  most of the old verdict. The `person` row still gives the mechanism: optimal alpha
  is **0.15** on the generic index (hand it to BM25, 0.8147 there) but **0.30** on the
  routed index, whose target *is* the person dense specialist; both mechanisms repair
  the same per-type weak dense arm. **Family-size
  trap, worth reading before citing:** this script's arms A/B reproduce
  `hybrid_alpha_sweep.py` to 4 decimals from an independent code path, yet the
  `recall@10` **verdict** differed (Holm-adj 0.0252 at m=9 there, **0.1960** at m=12
  here) — **and as of 2026-08-20 it no longer does, so this paragraph has lost the
  example it was built on**: the sweep re-run puts that cell at **0.0870**, ns at
  both family sizes. The rule is unchanged and still worth quoting — a Holm p is a
  property of its *family*, not of the pair, so state m with any p from either table
  — it simply has no live illustration here now, and the report says so in those
  words rather than pretending otherwise. **Both jobs behind that are done, and the
  reason they were needed is the reusable part.** `hybrid_alpha_sweep.md` had dated
  from 2026-08-08 and was not re-run, so arm B's effect sizes silently stopped
  matching it (**+0.0281** here against **+0.0350** there) while
  `soft_vs_hard_routing.py` went on *printing* that they "reproduce it to 4 decimal
  places" — **an assertion, not a check** — with the `0.0252` frozen as a literal:
  the **fourth** cross-artifact anchor of the kind `561102e` replaced in three other
  scripts, and the one that sweep missed. Now `parse_alpha_sweep_loo` reads the
  report's `per-type (loo)` rows, the paragraph **compares and reports disagreement**
  instead of claiming agreement, and a missing or renamed report prints "the
  cross-check could not be made" rather than passing silently
  ([[feedback_a_traceability_check_shares_its_haystacks_rot]]). Pinned in both
  directions by `tests/tools/test_report_anchor_parsers.py`, whose fixture repeats
  the same table under two combo headings — verified to discriminate by running two
  plausible wrong parsers against it, both of which return the *other* combo's row.
  Re-rendering changed **only** that paragraph: all 12 significance rows and every
  arm mean reproduced exactly.
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
  (all 4 chunkers fully tied, Holm-adj p≥0.44) nor in the aggregate. **Post-rebuild-#4
  (2026-08-18) the aggregate order is `recursive` 0.5318 > `sentence` 0.5212 >
  `semantic` 0.5186 > `fixed_size` 0.5073 recall@10** — so `semantic` has now slipped
  to *third* numerically (it was second at 0.5206 behind `recursive`'s 0.5291), which
  only sharpens the retirement. The
  *only* significant chunker-pairwise result in the whole aggregate is still
  `fixed_size` losing to `recursive` on **nDCG@10** (−0.0298, Holm **0.0216**);
  aggregate recall@10 has **no** significant pair at all (that same cell is −0.0204 at
  Holm 0.8256), so do not state the laggard finding on recall@10 — several individual
  embedders do carry significant recall@10 cells, the aggregate does not. **Revised framing:
  `recursive`/`semantic`/`sentence` are a statistically tied top cluster with no provable
  winner; `fixed_size` is the one demonstrated laggard.** `semantic` is still a perfectly
  reasonable default (never proven worse than anything, and still the one chunker where a
  strong dense embedder demonstrably beats BM25, see below) — just no longer citable as "the
  best chunker." Cross-chunker-averaged
  hybrid recall@10, **2026-08-18** (`qwen3_0.6b` 0.5999, `qwen3` 0.5792,
  `jina_v5` 0.5696, `bge-m3` 0.5664, `e5` 0.5644, `e5_small` 0.5598, `congen` 0.4804,
  `sct` 0.4065, `m2v` 0.3140 — ordering unchanged except `bge-m3` and `e5` swapping
  two adjacent places by 0.0020). **The
  dedicated semantic-only top-5 pairwise tie test
  (`tools/eval/hybrid_significance_test_semantic_top5.py`) — the tie
  **partially broke** in 2026-07-29 and **rebuild #4 partially UN-broke it, which is
  the movement to know before citing `bge-m3`.** By 2026-08-06 `bge-m3` was outside
  the cluster on *every* metric; as of the 2026-08-18 re-run it is outside on
  **recall@10 and nDCG@10 only, and only against two rivals** — it loses to
  `qwen3_0.6b` (0.0020 / 0.0060) and `qwen3` (0.0342 / 0.0060), while `jina_v5`
  (0.1216 / 0.2992) and `e5_small` (0.1904 / ns) have gone back to ties, and **on MRR
  it now ties everything again** (its closest cell, vs `qwen3`, is Holm **0.0940**).
  So: **`bge-m3` is separated from the two `qwen3` models on 2 metrics of 3 and tied
  with the rest** — not "clearly outside the cluster on every metric", which is
  withdrawn. The
  remaining four (`qwen3_0.6b` 0.6153, `qwen3` 0.6014, `jina_v5` 0.5941, `e5_small`
  0.5854 recall@10, semantic-only; `bge-m3` 0.5436) are still fully, mutually tied on
  every metric. Don't cite a single
  "best combo" among those four — the tie there is confirmed, not provisional. Hybrid still
  significantly beats dense-alone for essentially every one of the 9 embedders on every metric
  (**24/27** as of 2026-08-18, down from 26/27 — `qwen3_0.6b` on MRR stays the standing
  exception (Holm 0.3674) and **`congen` newly joins it on recall@10 (0.3424) and
  nDCG@10 (0.2062)**, which is the RRF rule doing exactly what it says: BM25 got
  relatively stronger, `congen` did not, and fusing an arm that is no longer comparable
  stops paying) — still the most robust finding of the comparison. Hybrid vs. BM25-alone shifted more: `qwen3_0.6b`,
  `qwen3`, `jina_v5`, `e5`, `bge-m3`, `e5_small` all significantly beat BM25 on recall@10 now
  (`jina_v5` newly clearly significant, was borderline before); `congen` dropped **out** of that
  group (BM25 itself got stronger post-OCR-fix, closing the gap); `sct`/`m2v` remain the
  cautionary cases where hybrid is significantly worse than BM25-alone, and **`sct`'s recall@10
  deficit is significant again** (reversing the 2026-07-25 "no longer significant, p=0.08"
  finding — that finding was itself measured against the since-superseded index). The
  cross-chunker **dense-alone 3-way tie at the top is broken, but only half of it stayed
  broken.** `qwen3_0.6b` still significantly beats `bge-m3` on every metric
  (**+0.1161** recall@10, Holm 0.0000) — but **against `Qwen3-Embedding-4B` its
  recall@10 margin went ns at rebuild #4** — Holm **0.0592** on a +0.0475 margin,
  where the pre-rebuild +0.0486 had cleared the bar — while MRR (+0.0915) and
  nDCG@10 (+0.0694) both still clear it. **Cite it
  as: `qwen3_0.6b` beats the 4B model on the ranking metrics and ties it on
  recall@10** — a 0.0592 is a near-miss, not a reversal
  ([[feedback_a_replication_disagrees_by_sign_not_verdict]]). The per-entity_type specialist/weak-spot pattern underneath the old
  aggregate tie is **unaffected** by this (separate, already-fresh dense-alone test): `bge-m3` =
  person-query specialist, `Qwen3-4B` = only embedder with no provable weak spot across both
  main categories (ties bge-m3 on person AND ties ConGen/qwen3_0.6b on program), `Qwen3-0.6B` =
  now aggregate-leading but still has a real person-query weak spot `Qwen3-4B` doesn't,
  `ConGen-PhayaThaiBERT` = program-query specialist. BM25 alone (`retrievers/bm25.py`) no
  longer ties the top dense tier the way it used to — it still **significantly beats
  `bge-m3`** on recall@10 (**+0.0829**, Holm 0.0216; aggregate BM25 recall@10 rose
  0.3908→0.4930 post-OCR-fix and reads **0.4863** after rebuild #4, more than any
  embedder's own gain, because lexical matching is far more sensitive to OCR token
  corruption than dense embeddings)
  and still significantly beats every weaker embedder; it statistically ties `qwen3` and
  `qwen3_0.6b` on recall@10, and `jina_v5` has slipped **from borderline-significant to a
  clear tie** (+0.0784, Holm 0.0192 raw → **0.0576** adjusted, was 0.053).
  **The bigger 2026-08-18 movement is on MRR, and it is the first AGGREGATE cell in this
  whole comparison where a dense embedder significantly beats BM25 outright**:
  `bm25 − qwen3_0.6b` MRR **−0.1249**, Holm **0.0072**. Until rebuild #4 that had only
  ever happened in one per-chunker cell (below); it is now true of the headline
  cross-chunker table. On the same table `bge-m3` and `e5` **lost** their significant
  MRR margins over BM25 (0.0664 each), so BM25's MRR standing polarised rather than
  moved. The per-chunker breakdown
  (`tools/eval/bm25_vs_embedder_significance_test_per_chunker.py`) still
  shows `bge-m3` losing to BM25 significantly under `sentence` chunking specifically, and
  the `qwen3_0.6b`-under-`semantic` cell **strengthened from one metric to all three**
  (recall@10 −0.1044 Holm 0.0070, MRR −0.1809 and nDCG@10 −0.1604 both 0.0000; it was
  recall@10 alone at 0.006) — reinforcing `semantic` as the
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
  87.84%** of top-10s in order (96.59% as a set) and F=1,000 only 70.13%. The pre-registered
  guess "F=1000 will be identical" was **wrong**, recorded as such. *Does it matter?* —
  barely: macro recall@10 across the 36 combos is 0.5197 at k=n, **0.5162 at F=100
  (−0.0035)** and **0.5170 at F=200 (−0.0027)**, and it is **non-monotonic** (F=500's −0.0015
  is better than F=1,000's −0.0025) because truncation lifts different chunks' scores at
  different rates as F grows. Mechanism worth keeping: a chunk inside dense's top-F but past
  BM25's cut loses its BM25 term **outright**, not by a little — that is why this is not an
  approximation that merely loses precision. Damage concentrates exactly where this project's
  RRF rule predicts (worst combo at F=50 `semantic × e5_small` −0.0579, at F=200
  `recursive × bge_m3` −0.0224, at F=1,000 `sentence × sct` −0.0145 — **the last two
  identical to the pre-rebuild run, combo and value both**), and **`person` queries
  *gain* at F=50 (+0.0217)** — the only entity_type that does, consistent with BM25 carrying
  `person` (0.8147) while the cut deletes a weak dense arm's tail. **Timing (paired, one
  process, one loaded index, arms alternated per query, BM25 scorer pre-warmed so its one-off
  build lands in neither arm, `plain__sentence__qwen3__ff8f6c49`): k=n p50 1089.5 ms → F=200
  **417.9 ms** (−0.672 s, 2.6x), F=1,000 421.0 ms.** So the over-fetch is **~62% of hybrid
  query time**, and the ~0.42 s left is real scoring work (dense encode + gemv + `get_scores`)
  that no depth cut can touch — **do not read the earlier "the remaining ~1.36s is the k=n
  over-fetch" as all removable**; that sentence bundled the residual in. **Re-run 2026-08-23 against rebuild #4 (the figures above are that run's): every
  finding survived and only the levels moved** — the two questions still answer
  oppositely, non-monotonicity still holds, `person` is still the only type that
  gains, and the F=200/F=1,000 worst combos are unchanged in both combo and value.
  All 6 self-checks pass at full scale (S2 and S4 both 3,816 reproduce / 0 differ
  against the *current* persisted results, which is what confirms the refresh is
  aligned with the rebuild).
  The trade on the
  table was ~0.67 s/query for −0.0027 macro
  recall@10 at F=200 — a *cost* decision of the same shape as soft-vs-hard routing, and it
  needed re-measuring against the hard router (which now ships) before adoption, since
  that macro figure is an average over a whole combo family, not a system result.
  **That re-measurement is DONE and the decision is made — see the next bullet.** Two method
  notes: the sweep replicates the **truncated** tie-break (the fusion dict is filled
  `dense[:F]` first, then the BM25-only remainder in BM25 rank order, so equal RRF scores stay
  dense-first — the same trap `miss_depth_profile.py` documents at full depth), and **S5
  checks the numpy fusion against a real `HybridRetriever(fetch_depth=F)`**, added because the
  first version anchored only F=n, where the mechanism under test is inert and the check would
  have passed identically had `fuse_at_depth` ignored F. Everything the report renders is
  cached in `hybrid_fetch_depth_raw.json`, so `--render` reproduces it without a GPU.
- **`weighted` × `fetch_depth`: MEASURED 2026-08-12 and the guard is LIFTED
  (`tools/eval/hybrid_weighted_fetch_depth.py` → `data/results/hybrid_weighted_fetch_depth.md`,
  36 combos × 106 queries = 3,816 pairs, 16 min).** From 08-11 to 08-12 that pair *raised*
  in `HybridRetriever.__init__`, and the entry here said so — **containment, not a verdict**,
  with its own exit condition written into it ("measure it and lift the guard"). The
  measurement was run and the pre-registered rule (frozen in the script as `DECISION_RULE`,
  committed before the run) came out **LIFT**: the raise and its `allow_unmeasured_truncation`
  hatch are gone, and `tests/retrievers/test_hybrid_retriever.py` now pins that permitting the
  pair did not quietly make it a **no-op** — truncation under `weighted` must still really
  truncate, since that is the whole cost. **LIFT is not a recommendation and the number is the
  point**: at F=200 `weighted` loses **−0.0605** macro recall@10 against its own F=n, about
  **22x** `rrf`'s −0.0027 at the same depth (it read 18x against −0.0033 before the
  2026-08-23 refresh — **the multiple grew because the DENOMINATOR shrank**, not because
  `weighted` got worse; state both terms, never the ratio alone), and it does **not** recover
  with depth the way
  `rrf` does — at F=10,000 of ~75,000 chunks it is still −0.0112 against `rrf`'s −0.0004, so
  for `weighted` "deep enough" is essentially n and the knob buys nothing. What licenses
  permitting it anyway is that this codebase bans an **unmeasured** configuration from passing
  as measured, not a measured-but-worse one (nothing bans `m2v`); the docstring now carries
  the cost. Four things worth more than the verdict. (1) **The smoke slice reversed the sign
  of the headline** — on 2 combos × 8 queries `weighted` *gained* from truncation, peaking
  0.7708 at F=100 against 0.5938 at F=n, which is the KEEP branch; on the full set every Δ is
  negative. A smoke run checks that the code runs, it is **not a small version of the answer**
  ([[feedback_a_smoke_slice_is_not_a_small_answer]]). (2) **P3 refuted, and the plausible
  reasoning behind it is the trap**: `BM25Okapi` floors negative IDF, so BM25 scores are ≥ 0
  and the *last-ranked* chunk really does score 0 — but only **0.1%** of the terms a cut
  zeroes were already 0, because a chunk scores exactly 0 only when it matches **no** query
  term and a ~20-token Thai query has common tokens reaching nearly every chunk. BM25 carries
  73% of dense's zeroed mass at F=50 (88,313 vs 121,449). The promotion half is real and
  negligible (2 of 157,731 dense terms at F=50), so the perturbation is one-sided after all —
  for the opposite reason to `rrf`'s. (3) **The mechanism, corrected by the same data**: the
  hypothesis was that truncation *creates* the intersection signal `weighted` structurally
  lacks (at F=n "also in the other arm's list" is true of every chunk). Truncation does not
  add that signal mildly — it makes intersection membership **nearly decisive**, because a cut
  arm's normalized term is worth 0.5 × 0.27–0.95 (max-normalized cosine is *flat*: 0.9491 at
  rank 10, 0.2699 at rank n) where `rrf` at rank 1,000 forfeits only 0.5/1060 ≈ 0.0005. So
  `weighted`'s top-10 goes 8.25/10 in-both-arms at F=200 and 9.99/10 at F=1,000 (`rrf` 7.41 /
  8.30): it becomes an intersection-only ranker and evicts what one arm alone found. That
  lands exactly where a single arm carries a type — **`person` −0.1957** at F=200 (BM25 carries
  person at 0.8147) against `program` **+0.0216**. (4) **P4 refuted in the interesting
  direction and it is a hypothesis, never a result**: at F=n `weighted` scores **above** `rrf`
  (0.5439 vs 0.5197, **+0.0241** macro recall@10). Descriptive only — no significance test,
  macro over 36 combos, **unrouted**, and nothing ships `weighted`; the wrong-pair trap that
  killed per-`entity_type` alpha and rrf4 applies here too, so it would need re-measuring
  against the hard router before it means anything. **Re-run 2026-08-23 against rebuild #4, as a PAIR with the sweep above and never alone:
  the LIFT verdict, every refutation and every mechanism survived, only levels moved.** The
  pairing is forced by S7 — this run's `rrf` columns must reproduce the *published* sweep, so
  refreshing either report on its own breaks the anchor rather than merely dating it; the
  sweep was re-run first and S7 then reproduced it at all 11 depths. The fusion is
  **imported** from
  `hybrid_fetch_depth_sweep.py` rather than reimplemented, which makes this run's `rrf` columns
  a cross-artifact anchor (S7 reproduces that sweep at all 11 depths); S5/S6 check against the
  real `HybridRetriever` at F=n **and** at F ∈ {5, 50, 200, 1000}, since S5 alone would pass
  unchanged if the fusion ignored F ([[feedback_anchor_a_check_where_the_mechanism_is_live]]).
  The F-invariance of the two normalizers (`_normalize` runs over the already-truncated,
  descending-sorted arm list, so `max(top-F) == max(all n)` for any F ≥ 1) is reported as a
  lemma and deliberately **not** a self-check — it is true by construction, and a check that
  cannot fail is a vacuous PASS dressed up as evidence. Raw cache
  `hybrid_weighted_fetch_depth_raw.json` is written before `render()`, so `--render` is free
  and a render crash after a GPU run loses nothing.
- **`fetch_depth` against the shipped router, and the ship decision (2026-08-09,
  `tools/eval/routed_fetch_depth_test.py` → `data/results/routed_fetch_depth_test.md`,
  ~2.5 min quality + ~3 min latency).** The sweep above left one blocker: its
  −0.0027 is a macro over 36 combos retrieving with **no router**, and hard routing has
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
  `routed (shipped)` **0.6811** and S3 the unrouted **0.6229**, both exactly, from an
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
  (`qwen3_0.6b` 0.6034). **BM25 carries person (exact name match), dense carries program**;
  that is direct evidence for the complementarity the Open item #2 proxies never
  established. Caveat: **"hybrid never hurts" is an aggregate claim, not a per-category
  one** — on `person` specifically hybrid sits *below* BM25-alone for most embedders
  (`qwen3_0.6b` 0.7264, `qwen3` 0.7342, `jina_v5` 0.7382), only `bge_m3` (0.8220) exceeding
  it. Measured against the structural ceiling (2026-08-18 figures), hybrid reaches 84.2%
  on `person`, 72.5% `faculty_adjunct_aggregate`, 67.9% `program`, 65.6% `course` — **this reverses the old
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
  1. **Cross-encoder reranking — the axis is NARROWED, not closed. Derivation:
     `docs/reranker-axis-notes.md`** (folded out 2026-08-23; it holds every
     figure, mechanism and self-check story behind the verdicts here, and is in
     `audit_doc_claims.DOCS` so those figures stay under D2/D5/D7). Reports:
     `data/results/reranker_*.md`. Pre-registration for (a):
     `docs/reranker-trained-on-hybrid-design.md`.
     **The null belongs to the WIRING and the MODEL, never to the axis** — three
     independent routes say so, and that is the sentence to re-read before
     re-proposing anything here.
     **Verdicts.** (i) *Truncate-and-replace* with off-the-shelf
     `bge-reranker-v2-m3` **significantly hurts hybrid MRR** (0.7730 → 0.6940,
     Holm 0.0240 at the 2026-08-18 re-run; the margin has shrunk at every rebuild
     while staying significant), is ns on dense and on hybrid recall@10/nDCG@10,
     and costs ~1.2 s/query. (ii) *Fusing it as a 4th RRF signal* (`rrf4`) beats
     **unrouted** hybrid on recall@10 (+0.0392, Holm 0.0108) — MRR is **repaired,
     not improved** — but **dies against the hard router** (−0.0098, Holm 0.9768).
     (iii) *Fine-tuning it on hybrid-fused candidates* (follow-up (a)) is the
     **first intervention in this line to survive the router**: +0.0730 recall@10
     over it, Holm 0.0000.
     **Bounds, which are the citable part.** The routed P=50 pool **holds** 0.9054
     of the gold and a perfect selection delivers **0.8331** — **+0.1520** over the
     router, against the off-the-shelf model's **−0.0098**. Trained captures
     **48%** of that; the best of 4 qualified off-the-shelf models captures
     **11%**. rrf4 on top of routing is bounded at **+0.0037**. **An oracle is not
     a system**, so this is a bound on the axis, not a plan.
     **Do not re-propose.** (a) Confirming `bge-v1-large` on a fresh query set —
     **closed as dominated 2026-08-12**: no decision hangs on it, the claim it
     served already holds from two independent routes, and no clean disjoint
     query set exists. (b) Re-deriving "a dense pool is better because dense is
     closest on the hard pairs" — measured and **rejected in the opposite
     direction** (−0.1143); *selecting on the pairs everyone misses is the wrong
     denominator for all 1,046* ([[feedback_selecting_on_hard_cases_misjudges_a_pool]]).
     **The reusable rule, now seen twice.** rrf4 is the **second** intervention to
     die against the hard router after per-`entity_type` alpha, with the identical
     mechanism: both repair a per-type weak dense arm, and hard routing already
     hands each route a specialist index that has not got one. **Measure the 2×2
     against what SHIPS** ([[feedback_per_type_repair_substitutes_for_routing]]).
     **What is wired, and what defaults to nothing.** `lexical_containment`
     (arm L′, 2026-08-20) — a free, GPU-less containment partition of the hybrid
     top-50 — **beats the shipped router significantly on all three metrics**
     (+0.0489 recall@10, Holm 0.0000) at ~+20% latency and **no GPU**, the layer
     that actually saturates. **Nothing defaults to it**; `dense`/`hybrid` ship
     unchanged, opt-in by name like `qdrant_hybrid`. The trained checkpoint is
     **not** wired. **Cite it with the circularity**: the person/program/faculty
     qrels were themselves derived by string containment, so this arm sits closer
     to the labelling generator than to "relevance" — never citable as *lexical
     beats learned ranking*.
     **The control that must appear beside the headline.** Arm **L** is fed the
     entity from the **gold YAML**, an input no other arm gets and no deployment
     has; it reaches 0.7438 against trained's 0.7541. `T vs L` is **ns on all
     three metrics** after rebuild #4 — so *"the fine-tune's contribution is
     ordering"* is **WITHDRAWN**; read it as a bound (T beats L by at most 0.0298
     recall@10). Against the **deployable** L′ it does separate everywhere.
     **Never cite `T vs C` without arm L's number beside it**; `T vs D` is the
     clean comparison ([[feedback_a_control_arm_can_be_fed_an_oracle_input]]).
     **Bare-field matching (2026-08-20)** — a user types the field, not the
     60-character canonical, so `classify_query` returned `unmatched` and L′
     silently degraded to plain hybrid. `match_programs_by_field` resolves a bare
     field to **every** programme offering it. Wired **OFF by default** in
     `detect_entities`, **ON** in `LexicalContainmentRetriever`, and as the
     **last** branch in `classify_query` — that position is load-bearing and was
     placed on evidence — several faculty queries carry a programme field inside
     their own faculty name, so anywhere above `match_faculties` it would steal
     them, and `tests/test_program_field_matching.py` IS the source of those
     counts. **The rejected rule must stay rejected**: collapsing
     dictionary entries when one field *extends* another merged 28 real
     programmes — **a longer field name is normally a DIFFERENT programme**
     ([[feedback_a_longer_name_is_a_different_thing]]). The Gold set is
     structurally unable to score any of it, so **no retrieval number may be
     claimed in either direction**; `tests/test_program_field_matching.py` IS the
     source of its counts.
     **Four traps, each found by a failure rather than by reasoning.**
     `--reuse-scores` is **not GPU-free** — it skips the cross-encoder, not
     retrieval ([[feedback_reuse_scores_is_not_gpu_free]]). **Qualify a model
     before measuring with it**: candidates arrive broken under transformers 5.x —
     and one *ran, ranked a Thai example correctly and was completely
     position-blind*
     ([[feedback_qualify_a_model_before_measuring_with_it]]). An oracle **must
     dedupe to the judged unit** or it understates the ceiling
     ([[feedback_oracle_must_dedupe_to_the_judged_unit]]). And **exactness is a
     claim about scores, not tie order** — C2 failed on a gap of exactly
     0.000e+00 ([[feedback_exactness_is_a_claim_about_scores_not_tie_order]]).
  2. **RQ3 preprocessing ablations: normalization and word-aware segmentation do nothing;
     only chunk size matters, and only at 1024.** Configs `config/experiments/rq3_*.yaml`,
     scripts `tools/eval/rq3_*`. Thai normalization (Thai digits + `pythainlp.util.normalize()`)
     and word-aware `newmm`-boundary chunking are both **not significant on any metric**
     (Holm-adj p≥0.335 / ≥0.264). Chunk size **is** significant but the citable claim stays
     narrow: **1024 loses significantly to 512** on dense recall@10/nDCG@10 and hybrid
     recall@10/nDCG@10, and to **256** on dense recall@10 and hybrid recall@10/nDCG@10 (the
     dense nDCG@10 256-vs-1024 cell is a near-miss, Holm-adj p=**0.0948**, **not**
     significant —
     don't fold it into "256 beats 1024 on every metric"). **256 vs 512 is a flat tie on
     every dense metric** (recall@10 0.4117 vs **0.4139**, Holm-adj p=**0.9338**) **and on
     hybrid MRR — 256 only wins on hybrid recall@10** (+**0.0533**, Holm-adj p=**0.0076**). **Do not cite
     "smaller is monotonically better" or "256 is best"**; the project's 512 default is not
     shown to be suboptimal, only 1024 is shown to be wrong. These ablations' treatment
     indices reuse `chunker_compare_full` combos as their *baseline* arm, so an index rebuild
     silently turns them into a clean-baseline-vs-dirty-treatment confound — they need real
     GPU rebuilds after a corpus change, not just a re-eval, and this happened twice: once
     for the 2026-07-28 OCR-remediation rebuild (fixed 2026-07-29) and again for
     `chunker_compare_full` rebuild #3 (2026-08-05T07:56). **Both times, fixing it meant a
     real GPU rebuild of all 3 RQ3 treatment indices, not just a re-eval** —
     `data/logs/run_rq3_rebuild_2026_08_05.sh`, ~2.5h, exit=0. **Rebuild #4 was the
     third time, and it was already handled: the 4 RQ3 treatment indices are part of
     its 40** (36 chunker x embedder + 4 RQ3), rebuilt 2026-08-17T12:01-13:50, and
     they carry the same `docset_hash 091b7a0ad8a5cfbe` as both baselines — so no
     confound. **Only the eval was owed, and it ran 2026-08-20** (14 min, no GPU
     rebuild): **0 verdict flips across all three reports**, so every claim in this
     bullet holds with the point estimates above refreshed. Two things worth keeping.
     **Read a treatment index's currency off `manifest["timestamp"]`, never off the
     directory mtime** — those folders still read 2026-08-08 while their contents are
     from 08-17, and taking the folder at face value nearly bought a 2.5-hour rebuild
     that was already done (the same read-the-artifact's-own-provenance lesson as
     `E0`, and the same shape as the Qdrant re-ingest that had also already happened).
     And **`diff_significance_reports.py` keys 0 rows on
     `rq3_chunksize_sweep_report.md`** — it has no verdict-bearing key that differ
     recognises, so "0 flips" there came from a hand diff of all 18 significance
     cells, not from the tool ([[feedback_verdict_diffing_misses_number_drift]]).
     If `chunker_compare_full` is rebuilt again *without* RQ3 in the batch, treat RQ3
     as stale until re-run.
- **RQ4 (end-to-end answer quality) — COMPLETE, and refreshed against rebuild #4
  on 2026-08-20 (all five model×variant jobs 424/424, every report re-scored).**
  5 arms × 106 queries × 3 prompt variants, `phi4` local-only, plus a `gemma4:e4b`
  second-generator check and two entity arms. Design, pre-registrations, build
  phases and every superseded figure: `docs/rq4-design.md`,
  `docs/rq4-second-generator-check.md`, `docs/rq4-prompt-truncation.md`; scripts
  `tools/eval/rq4_{generate,score,supervisor.sh,status}`; reports
  `data/results/rq4_score{,_guarded,_entity,_gemma4,_gemma4_guarded}.md`.
  **What to cite.** (a) Retrieval quality survives the generation stage:
  **`{hybrid, dense} > bm25 > m2v` is generator-independent** (precision ordering
  holds across both models), while **`hybrid > dense` is a phi4 result, not a
  system result** — it is significant for phi4 (recall −0.0678, Holm 0.0410) and ns
  pointing the other way for gemma. Levels do not transfer between generators at
  all. (b) **The prompt ablation is the headline and it is an instruction problem,
  not a generator ceiling**: the original `ตอบสั้น ๆ ไม่เกิน 3 ประโยค` rule
  suppressed citation recall against a gold set averaging 9.87 relevant documents;
  `cite_all` raises it significantly on **2 of 3** answering arms (`hybrid`
  **+0.0871**, `bm25` +0.0789, both Holm 0.0000) — the dense arm went to a **bound**
  at rebuild #4 (+0.0407, Holm 0.6610) because better contexts shrink the marginal
  value of the instruction. (c) **Report `cite_all_guarded` as the paper's prompt
  but cite the arm ordering from `cite_all`** — only the unguarded variant
  separates the two strong arms from each other; the guard's own 6-of-12 separations
  are all m2v pairs. **Always quote the Holm family size**: identical data reads
  significant in family 2 (m=9, "does this prompt beat the baseline" — cite this
  one) and ns in family 3 (m=24).
  **Two guards that generalise past RQ4.** `cite_all`'s missing zero-document rule
  cost phi4 2 hallucinations and `gemma4:e4b` **24**, with 37/37 of its closed-book
  citations phantom; rule 5 (abstain when no documents are supplied, and it
  outranks rule 4) takes that to **24 → 1**. The fix was **position, not a missing
  rule** — rule 3 already forbade it and is identical between variants, but rule 4
  is the last line before the question and won by recency
  ([[feedback_prompt_rule_recency_beats_earlier_rules]]). 100% of closed-book
  hallucination in both models is `course` queries.
  **Operational rules, all now enforced in `rq4_generate.py` rather than written
  down.** `--num-ctx` below **16,384** is refused (ollama truncates an over-long
  prompt to `num_ctx//2 + 2` keeping the **tail**, so the evidence dies and the
  instructions survive — see [[project_rq4_prompt_truncation]];
  `prompt_eval_count == num_ctx//2 + 2` is an exact signature and `G1a`/`G1b` in
  the invariant audit read it off the artifacts). `--variant sentence_cap` is
  refused for any model but `phi4`. `think` is **read** from the model's
  capabilities and disabled, never assumed — it changes the answer text, not just
  the cost ([[feedback_a_generation_default_is_part_of_the_measurement]]).
  `num_predict` is capped at **4,096**, which is part of the measurement rather
  than a timeout: **3 of 1,272** phi4 cells hit it, all `รายวิชา` course queries,
  and a capped answer never reaches its `อ้างอิง:` line, so it scores as *cited
  nothing* — a generator failure that reads like a retrieval result. **Quote that
  count with any number from this run.** `--out` is mandatory when scoring another
  model, and passing a non-default `--arms` refuses to write `rq4_score.md`
  (family 1's Holm size *is* the number of arm pairs).
  **Two things that make an isolated RQ4 verdict flip uninformative.**
  Temperature 0 is **not** reproducible here — byte-identical prompts reproduce the
  citation set only **14 of 24** times under `cite_all`
  ([[feedback_temperature_zero_is_not_reproducible]]) — so every refresh
  regenerates **only** the cells whose context actually changed and freezes the
  rest byte-for-byte, which is what separates repair from drift
  ([[feedback_repair_a_subset_paired_with_a_control]]). And `closed_book` is the
  built-in control: its context is empty, so it must come back byte-identical.
  **The entity arms answered a gating question and the answer is: do NOT build
  relation-graph edges B/C.** `entity_lookup` is decisively worse than hybrid on
  recall (**−0.2384**, Holm 0.0000) — and the *stated reason* did not survive:
  its contexts hold a **higher** gold density than hybrid's (0.6501 vs 0.5352) and
  it still abstained on 40 gold-bearing queries, so it is a **ranking** failure, not
  an evidence failure ([[feedback_exhaustive_retrieval_dies_at_the_context_budget]]).
  `entity_boost` is the arm that answers the gate and its recall margin is **ns**:
  ranked dictionary use buys at most **+0.1114** citation recall, a bound that is
  **optimistic** because the qrels and the retriever read the same dictionaries.
  Details and the precision cell that moved: [[project_rq4_entity_arms_gating]].
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
     are looser (0.05-0.10); **`e5_small` vs `jina_v5` on MRR (bound **0.1029**) is the one
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
     as recall = 0.9449, and never as `recall@10`** — and **read it off
     `data/results/gold_entity_lookup_73det_report.md`, which stamps the build and
     `docset_hash` it was scored against, rather than off any figure quoted in prose**:
     it has moved on every rebuild of that index (`0.9291` pre-2026-08-05, `0.9422`
     after, `0.9449` after the 2026-08-12 `match_programs` repair reached it), and the
     metric is recall@**1000**
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
  best single **0.6229** (was 0.6935), hybrid ceiling **0.8916** (was 0.9201),
  pairs **1,046** (was 644) — but **the cause is the query set, not a bug in it,
  and getting that attribution right took a second look**: it ran on a checkout
  whose gold set still had **73** entries (`REPO` is hardcoded to another user's
  OneDrive path), and the 33 `course` queries are the harder ones. Holding this
  script's combo set fixed and scoring only the 73 non-course queries reproduces
  its shape — best 0.6746, union 0.9125. The first draft of this bullet accused
  it of unioning 44 combos; **its own header says 36**, so that was wrong. What
  *is* true is a portability defect: it selects combos by bare `glob`, so re-run
  here it takes 44 (the 8 in `_EXCLUDED_COMBO_DIRS`, indices deleted, results
  pre-rebuild-#3) and the ceiling reads 0.9025 instead of 0.8916 (+0.0110) — in
  §1, not asserted. Note it moves the *ceiling* only; the retired combos are the
  weak `sct`/`congen` and never the argmax. Derive the combo set from which
  index dirs *exist* and cross-check against the exclusion list — either half
  alone goes stale. See [[feedback_external_analysis_reads_a_stale_slice]]:
  verify what an outside analysis actually ran on before critiquing it. Findings (levels are the 2026-08-18 re-run against rebuild #4;
  pre-rebuild figures given where the shape matters): (a) **at a fixed
  10-doc budget, diversity is negative** — 2 systems × 5 = **0.5936** vs 1 system
  × 10 = **0.6229** (**−0.0294**, was −0.0368), while doubling the budget is
  **+0.1196**; the original's "ensemble wins" read
  20 docs against 10, and both arms here are greedy-fitted on the test set so
  the bias favours diversity and it still loses; (b) **67.8% of the misses are
  ranking, not absence** (best single **509** pairs, union **873** of 1,046) — but
  that headroom's own ceiling *at 10 docs sent* is **0.7775** (oracle picks the
  combo) to **0.8342** (perfect rerank over all 360), **never** the hybrid union's
  **0.8916**; (c) unioning
  the dense and BM25 result sets too lifts it to **0.9418 macro / 0.9130 micro**, so
  **82 of the 173 "nothing found it" pairs were a retriever-choice artifact** and
  the floor is **91 pairs (8.7%)** — **cite 91**, re-derived from the 2026-08-18
  re-run against rebuild #4; it was **84** before that (and **76** before
  2026-08-09, when the 8 pairs subtracted as a labelling artifact were measured and
  the premise refuted — see the anchor-ambiguity bullet below, and note the
  subtraction stays withdrawn). **The floor moved UP while retrieval got better,
  which is not a contradiction**: rebuild #4 re-OCR'd a meeting, so the union of
  36 combos covers a slightly different corpus, and a pair no arm reaches is a
  property of the qrels *and* the text. **Do not call that floor *structural* — the word
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
  `anchor_status` ∈ `ok`/`ambiguous`/`no_name_evidence` (**414 / 66 / 198** of 678). **VERIFIED 2026-08-21 — it reproduces exactly.** It had been flagged UNVERIFIED on 08-20 because the on-disk `gold_candidates.json` carries no `anchor_status` key at all; that turned out to be provenance, not disagreement — the artifact is dated 2026-07-25 and the annotation landed 08-09, so it *could not* contain the field. Re-running the generator to a scratch directory (`--output-dir`, ~4 min, so the 07-25 artifacts that the published gold set was curated from are **left untouched** — they are its provenance) returns `ok 414 / ambiguous 66 / no_name_evidence 198` of 678. **Re-deriving it also found something the figure itself never would have**, which is why re-running beat hand-checking: the person pool moved **1,139 → 1,119**, and exactly **1 of the 30 published person gold entities is no longer nominable as a candidate** — `ดร.กลกรณ์ วงศ์ภาคิกะเสรี` had `hit_count 2` in July and has 1 today, because its two documents now spell the surname differently (`วงศ์ภา**คิ**กะเสรี` and `วงศ์ภา**ติ**กะเสรี`, one character) after a re-OCR, so neither spelling clears `min_person_hits=2`. **The qrels are still correct and nothing is being changed**: both resolutions genuinely concern that person, and the gold string is the one the corpus carried when the entry was curated. What it *is* is a concrete instance of the pooling-bias threat already written up in `docs/eval-validity-threats.md` §2 — a relevant document phrased differently is a hard retrieval case, not a mislabelled one — and a reminder that **a candidate pool is dated evidence, not a standing property of the corpus**: re-deriving one after a re-OCR will not reproduce it. Editing the entity string would move every published person figure for one query and is deliberately not done.
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
  of the **91** all-arm misses (2026-08-18 re-run; **84** before rebuild #4),
  **71 (78.0%) sit at ranks 11-50**, 90 of 91 are inside the top 1,000, exactly
  **1** is deep (`รายวิชา CALCULUS 2`, rank **2,988**), and **0 are missing from
  the index**. So drop "structural"; a reranker fetching 50 candidates can reach
  three quarters of them. **Every one of those figures held its shape across the
  rebuild**, which is the useful part — the profile is a property of the query set,
  not of one build. Two consequences: **`person` has 0 misses** (the 91 are course
  41 / faculty 28 / program 22, and course is almost purely near-miss at 40-of-41
  while faculty still splits **14 near / 14 deep** — a reranker helps course, not
  half of faculty); and **the candidate pool should come from `dense`, not the
  shipped hybrid** — on these hard pairs dense has median best rank **22** and is
  closest on **74 of 91**, vs hybrid 39 (13) and BM25 200 (6).
  Read against the **already-measured** cross-encoder result (hurts hybrid MRR
  0.7730→0.6940, Holm 0.0240 as of the 2026-08-18 re-run): the evidence is in reach at
  P=50, the tried reranker
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
  and **S5/S6 reproduce the ceiling report's all-arm and hybrid-only miss counts
  from an independent code path** (91 and 173 at the 2026-08-18 re-run, 84 and 164
  before it — stated as a rule rather than a pair, since both move with the corpus). **S7 exists because the first version of §2 was wrong**: it reported
  "perfect rerank from a pool of 50 = 0.8869", which is *above* the qrels ceiling
  0.8856 and therefore impossible for a reranker that still sends 10 documents —
  the table was measuring what is *in the pool*. §2 now prints both columns and
  S7 gates the deliverable one against the ceiling — **cite the delivered one**:
  a perfect rerank over P=50 is 0.6229 → **0.8268**, and P=1000 buys only 0.8738,
  so the 10-document budget binds, not the pool. Same family as
  [[feedback_state_the_retrieval_budget_in_every_comparison]].
- **ColBERT / late interaction: CLOSED 2026-08-13, verdict STOP; re-run against
  rebuild #4 on 2026-08-20 with the verdict unchanged. Do not adopt, and do not
  reopen the failed prediction as a continuation.** Full narrative, build log and
  the ship-decision reasoning: `docs/colbert-late-interaction-notes.md`; code
  `src/rag_lab/colbert/`; reports `data/results/colbert_pilot.md`,
  `colbert_model_qualification.md`, `colbert_pylate_crosscheck.md`.
  The pre-registered prediction was **conjunctive** — tie-or-beat BM25 on `person`
  **and** the best dense embedder on `program`, in the same run — and it FAILED:
  `person` tied (**+0.0308**, Holm 0.3974), `program` lost by **−0.3337** (Holm
  0.0000), 6.7x the STOP margin. **The conjunction is the reusable part**: ColBERT
  carries the highest overall figure in the table (**0.5555** vs BM25 0.5088 /
  dense 0.5264), so written as an aggregate this run would have been published as
  a success ([[feedback_a_conjunction_refuses_an_aggregate_win]]). Mechanism: it
  **inherits** one side of the person/program split rather than covering it
  (`person` 0.8360 ≈ BM25 0.8053; `program` 0.2749 ≈ BM25 0.3278 vs dense 0.6086),
  though not purely lexical — it beats both arms on `course` (0.6176). The
  512/48 length rider was **executed and does not fire**: granting truncation the
  most damage arithmetically possible explains at most **0.0837** of a 0.3337 gap.
  **Ship decision (separate from the axis verdict,
  [[feedback_an_axis_verdict_is_not_a_ship_decision]]): no** — the failed cell is
  the route the shipped router depends on; it was never measured against hybrid or
  the router at all; it costs 1,578.9 ms p50 against 475.6 and 1.89 GB fp16 per
  chunker; and the `course` win is a per-`entity_type` repair, a shape that has
  died against the hard router twice.
  **Two things to carry off this axis even if ColBERT never returns.** (1) The
  checkpoint arrives **broken** under transformers 5.x — all 24 layers'
  `RotaryEmbedding.inv_freq` are uninitialised memory, so the model is
  position-blind, and **it scored the relevance example BETTER that way**
  ([[feedback_qualify_a_model_before_measuring_with_it]]); `_repair_rotary`
  restores it and self-retires. The corruption differs per load, so a one-off
  probe of a buffer proves nothing — only the check that recomputes `inv_freq`
  from the checkpoint's own code fires every time. (2) An N-check gate is a
  battery of **self**-consistency tests, and a convention uniformly wrong on both
  sides of every internal comparison is invisible to all of them — the pylate
  cross-check found `mask_punctuation` masking whitespace and no punctuation
  ([[feedback_a_self_consistency_gate_cannot_see_a_shared_convention]]).
  **Not established** (new predictions, not a continuation): ColBERT against the
  shipped hard router, fused with BM25, or on a second checkpoint.
  The artifact at `data/index/colbert/<chunker>__doc300_q32` is deliberately not
  an `Index` (packed `vecs`+`lengths`, no row-per-chunk), so
  `audit_pipeline_invariants.py`'s `I1` does not see it — that is scope, not a
  coverage gap; it carries its own L1a-L6 alignment check instead.
- **HyDE: CLOSED 2026-08-13 on both query sets — a significant LOSS, not a null.
  Do NOT re-run it after an index rebuild** (no verdict can flip at these margins
  and the bullet states no bound a refresh could sharpen — the distinction between
  *stale* and *worth refreshing*). Narrative and the frozen predictions:
  `docs/hyde-axis-notes.md`; scripts `tools/eval/hyde_{generate,retrieval_test}.py`;
  reports `data/results/hyde_retrieval_{73det,thematic}.md`, `hyde_generation.md`,
  `hyde_generation_cost.md`.
  **P1** held in the harder half of its own wording: dense recall@10
  **0.5034 → 0.3135, −0.1898**, Holm **0.0000**, all six family-1 cells worse.
  **P2 was REFUTED** — thematic, the set these notes called HyDE's only real
  chance, loses too (**−0.0736**, Holm 0.0008, all 9 embedders down). **But P2's
  reasoning survives its own prediction and that is the transferable part**:
  damage comes from diluting an exact-token signal, so it is smaller where BM25 is
  weak (2.6x smaller) — cite that as *less harmful where the lexical signal is
  weak*, **never** as *HyDE helps thematic*
  ([[feedback_less_to_lose_is_not_something_to_gain]]).
  Three rules for anyone re-proposing it. (1) The `person` mechanism is **dilution,
  not deletion** — 29 of 30 generated documents still literally contain the queried
  name, and `person` is still the worst type (**−0.2798**). (2) **Keep the split**:
  feeding the same document to BM25 as well costs a further **−0.2735**, more than
  the entire dense-arm effect (P3, this design's one untested premise). (3) The
  four formulations order by how much of the raw query survives, and only `concat`
  reaches ns anywhere — as a bound, dense recall@10 loses no more than **0.0576**
  and gains no more than **0.0061**, for **7.85 s/query** against a 475.6 ms routed
  hybrid query. The 100%-cap-hit objection is bounded for free (greedy decoding is
  a prefix process, so `hyde_half` costs no extra generation): no consistent sign,
  every gap under 0.03. **No re-measurement against the hard router is owed** —
  that follow-up was made conditional on a *positive* unrouted result precisely so
  a negative one could not be kept alive by an untested "but maybe with routing".
  Two method notes worth keeping: the inherited 15.6 s/query cost figure never
  transferred (an RQ4 prompt carries ~8k tokens of context, a HyDE prompt ~300 —
  [[feedback_state_the_input_size_with_any_timing]]), and a length instruction
  written in natural language enforces **nothing** (the model wrote 564-843 tokens
  for a prompt saying "ไม่เกิน 5 ประโยค").
- **Qdrant serving pilot (2026-08-13, `tools/eval/qdrant_pilot_{ingest,test}.py` →
  `data/results/qdrant_pilot.md`; narrative `docs/qdrant-serving-pilot.md`)** — the first
  work here aimed at **deployment rather than the paper**
  ([[project_real_deployment_intent]]). **The pilot had to change shape before it could
  start, and that is the first thing to know: embedded Qdrant is exact brute force, not
  ANN** — `LocalCollection.search` scores every vector and `argsort`s it, and the
  `HnswConfig` it reports back is a fabricated default. So the 2026-07-16 vertical slice
  could not have answered "does ANN change the answer" however it was read, and its
  20k-point warning was about a code path that does no approximate search. This pilot
  therefore runs a real `qdrant/qdrant` **server container** (v1.18.0, `rag-qdrant`,
  named volume), client and server pinned to the **same** version — the first ingest ran
  1.18.0 against 1.15.1 and the warning was closed by matching the image and re-ingesting
  onto a clean volume, not by silencing the check. One collection
  (`plain__sentence__local__bf8b7ebb`, the `person` route's shipped target), 57,174
  chunks, 106 Gold queries, K=10, `fetch_depth=200`. **Q1 is measured as THREE dense arms,
  not two**, because numpy-vs-ANN bundles two causes: `numpy_exact` vs `qdrant_exact`
  (`SearchParams(exact=True)`) isolates storage/arithmetic, `qdrant_exact` vs
  `qdrant_ann_ef*` isolates HNSW traversal alone. **Storage and arithmetic are free**
  (0.3954 → 0.3957, **+0.0003**, agree@10 0.9858, residual is tie-break convention), and
  **HNSW costs accuracy while buying almost nothing here**: monotone in `ef` but still
  **-0.0028** at ef=1024, against `qdrant_exact`'s **17.8** ms p50 vs ANN's 10-13.
  **RECOMMENDATION: serve dense with `exact=True`** — fused end to end it reproduces the
  reference (0.5834 → **0.5851**, **+0.0017**, inside tie-break noise) where ANN at ef=512
  loses **-0.0199**, and it needs no `ef` retuning when `fetch_depth` moves. **The `ef` <
  `limit` trap is worth more than the table and it was MY confound, caught after a full
  run had been read**: the beam holds `ef` candidates, so asking for 200 results at ef=128
  is beam-starved *by construction* and the **-0.0421** there says the request was
  malformed, not that the graph is inaccurate; the first grid topped out at 256 and I read
  that number as an ANN cost. The grid was widened past `FETCH_DEPTH` and the run
  repeated. **Those rows stay in the published table on purpose** — ef=128 is Qdrant's own
  default, i.e. exactly what an operator gets by configuring nothing, so deleting it would
  hide the most likely real misconfiguration behind a clean curve. **Any deployment
  raising `fetch_depth` must raise `hnsw_ef` with it**, and only one of those two lives in
  this repo. **Q2 is exact BY CONSTRUCTION and is therefore a check, not a result**:
  ingestion precomputes each chunk's BM25 weights from `BM25Okapi`'s *own floored IDF
  table*, the query sends term counts, the engine takes a plain sparse dot product, and
  `Modifier.IDF` is deliberately unused (Qdrant's IDF is not this project's IDF); the
  vocabulary is an explicit sorted enumeration sidecar (78,333 terms), not a hash. Result:
  identical recall (0.5034 both), score sequence agreeing to **2.00e-07** relative at
  every rank, **zero** id disagreements wherever scores actually differ. **The check had
  to be corrected and that is the lesson**: its first version demanded rank-for-rank
  identical `chunk_id`s and failed 3/3 — every differing id sat inside an exact BM25 tie
  group (four chunks at 50.677741 on one query; tie groups are large on this corpus), and
  numpy's `argsort` and Qdrant's scan settle a tie differently with **neither more
  correct**. So "exact by construction" is a claim about the **score sequence**, never
  about tie order. The check was wrong; the ingestion was right. Latency, **within-process
  only** (the numpy arms pay no network hop, the Qdrant arms pay REST serialization, so
  the deployable figure is the served total): dense exact **195.0 → 17.8** ms p50, lexical
  **219.7 → 8.8**, i.e. ~0.4 s of per-query Python scoring replaced by ~27 ms of engine
  work — which is the resource a faculty VM at 5-50 concurrent users actually runs out of.
  `S2` anchors the whole pilot from an independent path (reference fusion 0.5834 at F=200
  vs persisted `gold_hybrid_73det` **0.5850** at k=n, the gap being the already-measured
  `fetch_depth` truncation effect). One observation **recorded rather than glossed** and
  since **EXPLAINED**: `indexed_vectors_count` reads **110,422** against 57,174 points
  over 6 segments (~1.93x) — a reported counter, not duplicated data (`points_count`
  equals the row count exactly, payload/vector alignment passes on every sampled row,
  searches return distinct ids). It was left unexplained here and is resolved by `C6` of
  the four-collection check below: **a point carries two vectors** (dense + sparse), so
  the counter is `2N − (dense rows in segments still under `indexing_threshold`=20,000)`,
  bounded `N ≤ indexed ≤ 2N` — the 1.93x is 3,926 dense rows (6.87%) being plain-scanned
  rather than HNSW-traversed, which is free **only because the recommendation is
  `exact=True`**. **What this pilot does NOT establish**: no significance test (every Δ is
  descriptive); its "one collection / one combo / one route" and "no concurrency
  measurement at all" gaps are both **CLOSED — see the next two bullets.** **Nothing is
  wired**: `query_service`/`registry` still route to the in-process retrievers.
- **Qdrant under concurrent load: the engine is NOT the layer that saturates
  (2026-08-13, `tools/eval/qdrant_concurrency_test.py` → `data/results/qdrant_concurrency.md`,
  9/9 self-checks PASS, decision rule frozen in the module before the run).** Run before
  ingesting the other 3 collections, because it was the one open item that could still
  invalidate the serving design. **Five arms, since "numpy vs Qdrant" is not the question a
  deployment asks**: `null` (harness alone), `qdrant` (engine only — vector and term counts
  precomputed, GPU out of the loop), `encode` (embedder alone), `glue` (tokenize + this
  repo's RRF over cached rankings, pure Python), `end_to_end`. Closed loop, C ∈ {1,2,5,10,25,50}
  worker threads replaying the 106 Gold queries. Plateaus: `qdrant` **82.40** q/s (at C=10),
  `encode` **68.46** (at **C=1**), `glue` 3,693, `end_to_end` **29.51** (at C=2).
  **Verdict ENCODE-BOUND — and the plateau row understates it, which is the part to cite.**
  The rule compares each arm's own best level (the conservative reading); *at matched C* the
  GPU delivers ~41 q/s from C=5 on against the engine's 72–82, i.e. **the engine has ~2x the
  headroom of the embedder at every level a deployment sits at**. **The mechanism is that the
  two layers scale in opposite directions**: Qdrant *gains* 2.1x from C=1→10 (more cores,
  more in-flight segments) while the embedder *loses* 41% (68.46 → 40.59) — a GPU is one
  device, so concurrent requests queue rather than overlap. **Corollary: batching, not
  concurrency, is the only lever on the GPU side**, and nothing in this repo batches at query
  time. **Composition holds** — `predicted(C) = 1/(1/encode + 1/qdrant + 1/glue)` at matched
  C (a worker runs the stages serially, so residence times add) — ratio **0.973** at C\*=2 and
  0.855–1.036 across the grid, so no hidden app-layer cost; `glue` is **0.83%** of the
  harmonic sum, i.e. this repo's Python fusion is not a layer. Target answered by
  **inversion**, since the arrival rate is unknown: 50 users need T ≥ **1.7 s** between
  queries at this plateau; 50 at T=10 s is 5 q/s against 29.51, so **capacity is not the
  constraint at this scale, latency is** (p50 46 ms at C=1 → 1,961 ms at C=50, which is
  queueing on the GPU — the `qdrant` arm's own p50 there is 654 ms). **The `encode` curve
  does NOT transfer** (RTX 3060 in-process vs a separate faculty GPU server) and is measured
  alone precisely so another GPU's plateau can be substituted without re-running anything;
  in-process encoding also bundles GIL contention with request handling that a network hop
  would separate, so this rig **understates** the app layer's headroom.
  **A correction is owed to `cost_latency_pareto.md` and it is measured, not argued**: its
  published `bge_m3` encode p50 **82.94 ms** disagrees ~6x with this run's 13.8 ms on the
  same model/box/queries, because **encode cost on this card is a function of how long the
  GPU sat idle beforehand** — control 4 inserts a fixed sleep and p50 goes
  **13.62 → 76.08 → 80.77 → 181.00 ms (13.3x)** at gaps 0/0.5/1.0/1.8 s, and the pareto loop
  leaves ~0.26 s and ~1.46 s between its encodes. So 82.94 ms is **encode-after-an-idle-GPU,
  the low-load regime**; cite the pareto figure for a lightly loaded system and this one for
  a busy one, neither supersedes the other. **Four harness defects were caught by the smoke
  slice and every one was the instrument, not the system** — each fixed at the mechanism,
  never by moving a threshold, which is what licenses the numbers: (1) dispatch behind a
  `threading.Lock` reported the harness's own dispatch cost as concurrency that never
  happened (Little's law 1.30 at C=4) → `itertools.count()`, atomic under the GIL; (2) S4
  then failed on `glue` alone and **that is a resolution limit, not a defect** — 0.3 ms of
  residence is an order of magnitude below CPython's 5 ms switch interval, so S4's domain now
  comes from `sys.getswitchinterval()` (not from whatever clears the failing arm) and **S9
  makes the exemption safe by measurement** rather than assertion; (3) the repeat-C=1 control
  had no warm-up while every sweep level did, so it compared warm against cold and called it
  drift (35.5% → 4.3%); (4) the idle-gap rows were not warmed *at their own gap*, so the
  zero-gap row measured a transition rather than steady state and missed S7's pre-chosen 0.35
  line at 38.5% → **1.3%, threshold untouched**. Anchors: S2 reproduces the pilot's cached
  top-10s 106/106, S1 pins that concurrent encoding returns bit-identical vectors
  (max |Δ| 0.000e+00), S5 that the harness is 12,989x faster than the system. **Still not
  established**: no network hop between app/embedder/engine, no bursty arrival process (the
  loop is closed); "one collection" is closed by the next bullet.
- **All four routed collections ingested and served end to end (2026-08-13,
  `tools/eval/qdrant_routed_check.py` → `data/results/qdrant_routed_check.md`, 106 Gold
  73det queries, ~118 s, 8/8 self-checks PASS — 7/7 when first run, `C8` landing with the
  wiring below).** With the concurrency question answered,
  the other 3 collections were ingested with the same `qdrant_pilot_ingest.py`. **This is a
  completion check on the ingestion, not an experiment** — no pre-registration, no
  significance test, no verdict; the only question is whether the served stack returns the
  published answer. It does: published `routed_fetch_depth_test.md` F=200 **0.6835**,
  reference (numpy dense + `BM25Okapi` through this code path) **0.6835**, **served (Qdrant
  `exact=True` + sparse) 0.6827, −0.0008** — per route `course` +0.0000, `person` +0.0000,
  `program` −0.0014, `faculty` −0.0033, worst relative score error over every rank of every
  query dense **3.63e-07** / sparse **2.27e-07**. Route → index is resolved through the
  shipped path (`discover_indices` + `route_targets("hybrid")` + `resolve_index`), so the 5
  routes give **4** distinct collections (`faculty` and `unmatched` share
  `fixed_size × bge-m3`) and a change to `ROUTE_COMBO_BY_RETRIEVER` moves the check with it.
  Two things worth more than the numbers. (1) **`C2` anchors per query, not on the macro**:
  `routed_fetch_depth_raw.json` holds all 106 per-query recall@10 at F=200, so the reference
  arm is gated at exact equality on **106/106** — a served arm agreeing with a subtly wrong
  reference would otherwise read as a pass. (2) **The dense check had to be rewritten, and
  it is the sparse arm's own correction arriving on the dense side**: C4 was first written
  as *set identity* of the top-10 and FAILED at `agree@10` 0.7500 on a smoke combo — yet
  every differing id carried an **identical** numpy score, and one `course` query's whole
  **top-12** sat at a single score because a course table repeated verbatim across
  curriculum revisions embeds identically. **Inside a tie group the returned set is not
  defined by either engine**, so a set test asserts an order nobody promised
  ([[feedback_exactness_is_a_claim_about_scores_not_tie_order]]). The rule is now **C4** (the
  *score sequence* agrees at every rank, < 1e-5 relative) plus **C4b** (every moved top-10 id
  carries the tied reference score: **160 of 1,060 moved, 160 in-tie, 0 out-of-tie, 0
  unresolved**), with `agree@10` demoted to a descriptive column beside `largest tie group`
  (recursive **22**, semantic 11, fixed_size 7, sentence 7). Fixed at the mechanism; **no
  tolerance was widened**. `C6` is what explains the pilot's ~1.93x `indexed_vectors_count`
  (see that bullet); dense rows still unindexed: fixed_size 0.00%, recursive 0.00%,
  semantic 5.22%, sentence 6.87%. **Not established**: one query set, one fetch depth, one
  fusion, **no network hop**, nothing about ANN (deliberately — the recommendation is
  `exact=True`). **A collection is a copy of an `Index`'s rows, so any index rebuild stales
  it**: re-ingest and re-run this. **Rebuild #4 staled 3 of the 4
  (`person`, `program`, `course`) on 2026-08-16 and all four were re-ingested on
  2026-08-18** — **once, after all 40 combos were done, never per rebuilt combo**, which
  is the protocol: re-ingest is cheap, `qdrant_routed_check.py` is not free, and a
  collection rebuilt against a half-rebuilt index family would have to be redone anyway.
  Never write the route→combo mapping down: resolve it the way the check does, through
  `route_targets("hybrid")` + `resolve_index`, so a `ROUTE_COMBO_BY_RETRIEVER` change
  moves it with them.
  **Re-verified 2026-08-20 by re-ingesting all four again and re-running: 8/8 pass,
  reference 0.6815 reproducing the published per-query F=200 figures on 106/106, served
  0.6829 (+0.0014), worst relative score error dense 3.24e-07 / sparse 2.15e-07.** Two
  things worth keeping from that. (a) **The re-ingest was redundant and the report is what
  says so** — `points_count` already matched the rebuilt indices on 08-18 and `C4`/`C5`
  agreed at 1e-7, which a stale vector set cannot do; the pending item in this file had
  simply outlived its own discharge. (b) **Served recall moved 0.6810 → 0.6829 across two
  ingests of identical data**, and that is the tie-order point one layer up: re-ingesting
  changes Qdrant's segment layout, so which member of a tie group is returned changes
  (`C4b`: 167 of 1,060 top-10 positions moved, **167 inside a tie group, 0 outside**).
  **Do not read a sub-0.002 movement in this table as a data change**
  ([[feedback_exactness_is_a_claim_about_scores_not_tie_order]]).
  **IT IS NOW WIRED (2026-08-13, `src/rag_lab/retrievers/qdrant_hybrid.py`,
  `docs/qdrant-serving-pilot.md` §8d) — the served path is the shipped `route_query`, not a
  script.** `qdrant_hybrid` is a registered retriever taking an all-scalar
  `StrategySpec(params={"url", "fetch_depth", "exact", ...})`, and **one spec serves all four
  collections** because the collection name is resolved *at query time* from
  `Index.provenance["index_dir"]` — so a `ROUTE_COMBO_BY_RETRIEVER` change moves the served
  path with it and no per-route config exists to drift. It is a **sibling of
  `HybridRetriever`, not a flag on it**: an engine-backed arm shares none of its internals but
  must share its ranking, so `fuse_rrf` was lifted to module level in `hybrid.py` as **the
  project's one copy of RRF** (tie-break included) and is imported by both — plus by
  `qdrant_pilot_test.py`, whose two reports re-render byte-identically after the change.
  **`BaseRetriever.reads_index_rows` (default `True`) is one flag with two consequences**:
  `query_indices` skips the ~234 MB `embeddings.npy` load for an engine retriever, and it
  **refuses** a row-level year filter or entity boost with a `ValueError` naming the reason —
  narrowing the in-process `Index` cannot narrow what the engine returns, so the quiet
  behaviour would be a silently ignored filter. The UI carries it as a Retriever option with a
  URL box, and disables/**resets** those two widgets there (a disabled Streamlit widget keeps
  its session value) and refuses `k=n`. **Nothing defaults to it** — `dense`/`hybrid` still
  ship in-process, this is opt-in. **`C8` is the gate**: it drives the *shipped* `route_query`
  on a per-route subset (8 of 106) and requires the same top-10 **and** the same resolved
  collection as the hand-assembled arm (8/8, 8/8), asserting
  `route_targets("qdrant_hybrid") == route_targets("hybrid")` rather than assuming the
  fallback, and gating on `bool(compared)` so an empty subset cannot pass vacuously. **Why a
  subset**: `build_embedder` has no cache and `query_indices` never releases, so N
  `route_query` calls are N model loads — a **pre-existing** serving gap (the Streamlit UI has
  it too), and per §8b the embedder is the layer that saturates anyway.
- **Serving: the caches, the warm-up, the seal, and what saturates under load.
  Derivation: `docs/serving-architecture.md`** (§1-§10 — the request path, where
  the time goes, the three caches, warm-up, footprint, the rebuild-underneath-a-
  server race, topology under load, the three defects, what is NOT established,
  and how to operate it). Reports: `data/results/serving_{cost_profile,
  cache_memory,warmup_profile,concurrency}.md`, `qdrant_{pilot,concurrency,
  routed_check}.md`. **This is the deployment-facing half of the project
  ([[project_real_deployment_intent]]), so the rules below are operational, not
  findings.**
  **Levels.** A served query goes **12,465 ms cold → 463 ms fully warm (26.9x)**,
  steady state **446 ms**. Startup warm-up takes the first four routed queries
  **31,719.7 → 1,613.0 ms**. Under concurrent load the shipped `route_query`
  plateaus at **9.81 q/s** on the engine topology against **2.53** in-process
  (3.87x), i.e. 50 users need one query every ~5.1 s.
  **Three rules that are not negotiable.** (1) **The caches are on the SERVING
  path only** — `build_embedder_cached` / `load_index_cached` /
  `build_retriever_cached`; `build_embedder` and `ArtifactStore.load` stay
  uncached for every eval script, because a global cache holds
  Qwen3-Embedding-4B resident beside its neighbours during a 9-embedder sweep,
  which is the OOM this project already lost five runs to. **No published number
  can move**, and tests pin that exclusion in both directions. (2) **The warm-up
  is OFF by default** (`RAG_LAB_WARM_ON_START=1`): it holds ~3.2 GB RAM + 3.3 GB
  VRAM on a card the eval scripts share, so an automatic grab at UI start is how
  a GPU run dies. (3) **Size for the PEAK, not the steady state** — process peak
  working set is **4,126 MB** against a 3,135 MB host-RAM + 3,310 MB VRAM
  resident set. Cache bounds: embedder 2 (`RAG_LAB_EMBEDDER_CACHE`), index 4
  (`RAG_LAB_INDEX_CACHE`), retriever 4 (`RAG_LAB_RETRIEVER_CACHE`) — each equals
  the number of *distinct* things the five routes resolve to, not a round number.
  **The verdict that reversed.** `qdrant_concurrency.md` (08-13) called the system
  **encode-bound**; that was a hand-assembled pipeline with the embedder and Index
  built once outside the loop. Measured through the *shipped* path the winning arm
  reaches only **32.3%** of the `encode` ceiling it contains — **not
  encode-dominated** — and the engine arm is already 86% of its plateau at C=1, so
  **size it by making one query cheaper, not by adding users**. Cite the 08-13
  figures only for the hand-assembled topology.
  **A writer must seal.** `ArtifactStore.save` writes `_complete.json` **last**
  and `index_cache._settle` refuses a directory that disagrees with it; without
  that, a reader landing between the writer's own two files pairs one build's
  chunks with another's vectors — undetectable downstream, and measured at
  **36,865 of 43,505** reads before the seal, **0 of 6,326** after. **An in-place
  writer owes a re-seal** (`relabel_index_resolution_ids.py` is the repo's one
  such writer); **unsealed is a reported gap, never a pass** (`I7` watches the
  fleet, `tools/seal_index_dirs.py --apply` seals). **Never downgrade a mismatch
  to "probably an out-of-band edit, read it anyway"** — during the inter-file
  window the directory is stable too, so stability cannot tell the two apart.
  **Four traps, each of which reversed a conclusion before it was found.**
  `localhost` cost **2,058.9 ms** per request against **15.1 ms** for
  `127.0.0.1` (136.3x) because Docker publishes IPv4-only and `getaddrinfo`
  returns `::1` first — **a cost identical across every knob is the transport,
  not the work** ([[feedback_a_hostname_is_not_free]]); every eval script had
  passed the fast spelling while the shipped default took the slow one.
  `LocalSTEmbedder._load()` is **lazy**, so `build_embedder` timed 0.0 ms and the
  cost landed on *encode* — **separate first use from warm use**
  ([[feedback_a_lazy_constructor_hides_the_cost_you_are_pricing]]).
  **A control arm that clears the treatment's cache is not a control** — that bug
  measured both caches at ~1.0x. And **stamping a cached read before AND after is
  still not enough** on its own; the seal above is what closes the remaining case
  ([[feedback_stamp_a_cached_read_before_and_after]]).
  **NOT established** (do not cite past these): no network hop anywhere — app,
  embedder and engine are one process on one box, which makes this box look
  *worse* at the app layer than a real deployment; no bursty arrival process; and
  **the rebuild-underneath-a-query-fleet race has never been run under real
  concurrent load** — the tests drive it deterministically.
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
  have moved: 0 of 1,014 distinct gold `resolution_id`s (both gold sets) reference
  any of the 4 repaired titles — **re-verified 2026-08-12 against the `FIXES`
  table itself**, because the "358" this line used to quote reproduces from
  nothing on disk today (the sets hold 2,265 entries / 1,014 distinct ids). The
  one near-hit is instructive: `2565/10/…เทียบโอนหลักสูตรฝึกอบรม…` carries the
  *same title text* as one half of the `2565/8` swap, so a title-only match
  says "hit" and the id is in a different meeting the repair never touched —
  match the whole `<year>/<session>/<title>` triple, never the title alone.
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
