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
  Known caveat it surfaced: `BuildCombo.id` hashes loader+chunker+embedder but
  **not the corpus**, so a smoke-subset combo and a full-corpus combo share an id
  and a persisted result cannot be attributed to one index — this is *why* the
  stale-cache incident was invisible. Check the index root by hand when it matters.
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
  denominator — `E3a 0 of 9,552 live result files` is a real pass, `E3d 0 of 0` says
  so out loud. Current state: **24 pass / 0 warn / 1 fail**, the single FAIL being the
  `BuildCombo.id` caveat above. Both former warns were chased to root cause rather
  than waived, and each turned out to be a symptom of something bigger than the
  warning said (the 5 duplicate thematic queries → the whole 179-entry subset was
  unanswerable; see above). C4's 24 orphan archives were reviewed one by one and
  closed (nothing was lost: 21 tail fragments of a wrapped title, 1 rename, 2
  misfiled-but-live); the verdicts are encoded as rules, and the same-document
  test compares page-1 `เรื่อง` headings because whole-file similarity decays
  across the re-OCR boundary. That review surfaced the corpus's one known
  title↔content defect: `2568/ครั้งที่ 7`'s CHECO-titled file actually holds
  รับรองรายงานการประชุม and the CHECO text is absent. Cause is the download stage
  fetching the wrong Drive id (two byte-identical PDFs, same SHA-256); the manifest,
  `_LINK.txt` and `master_list.csv` all already hold the correct id
  (`1d4iz1dpnPweAn7pxBfxlvJf9IJZwIJFJ`), which has never been fetched — so the fix
  is a re-download + re-OCR of that one URL, no metadata change (0 gold queries in
  the 73det set cite it). A general title-vs-body check was prototyped and **rejected on
  measurement**: median agreement is 0.660 over 2,820 files with 544 below 0.5,
  nearly all false alarms from agenda-number prefixes.
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
  perturb temperature/DPI instead). Remaining 88 pages need real human
  review. Mechanism B's 2 severe files (the LLM-detection blind spot —
  massive single-character repetition that neither model's garbled-prose
  check catches) are now **also fixed** (2026-07-27): re-OCR'd, adjudicated
  (unanimous new/new), applied, and verified clean by direct read.
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
  document type that surfaced it). English-titled foreign-faculty names
  (Mr./Assoc.Prof.Dr.) and a name split across adjacent `<td>` cells are
  still unmatched — both deferred, user judged low priority/rare. The one
  index built with the `entity_tags` loader
  (`data/index/entity_tags_full`) needs rebuilding after any
  `person`/`program`/`course` loader change for the fix to reach
  `entity_lookup`/`entity_boost` in the UI.
- `strip_course_comparison_tables` (`src/rag_lab/loaders/common.py`, commit
  `71764a8`) compacts old/new course-comparison tables (code + credit-tuple
  + English description, the corpus's single largest chunks — 17,077 chars
  in one document) to `CODE Title` lines, dropping the description. **Not
  yet wired into any loader/config**, but the blocker is now **resolved and the
  decision is to wire it** (2026-07-30). The old blocker was "needs a
  thematic-inclusive eval first", since course-name gold queries never touch
  description text. That eval now exists — and measuring it **closed the question
  the other way**: only **13 files corpus-wide** contain such a table (0.46%,
  39.8% of their text removed), and they are cited by **0 of 106** 73det queries
  and **3 of 179** thematic ones. No gold set can detect a description-stripping
  regression, so waiting for an eval was waiting for evidence that cannot arrive.
  User resolved it on domain grounds instead: **course descriptions are not what
  people ask about**, so the unmeasurable regression is also the unimportant one.
  Wire it into the loader; it must ride the pending rebuild (it changes chunk text,
  so it needs re-embedding). Note the ordering constraint already documented: run
  it *before* `match_courses`, which it improves. Narrative:
  `docs/chunker-embedder-comparison-log.md` (course-table compaction section).
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
  chunker axis (fixed_size − semantic is **+0.0256** thematic vs **−0.0359** entity-anchored;
  `semantic` is the *worst* chunker on thematic and the best on entity-anchored), so pooling
  the sets cancels two real effects instead of diluting one. Still low-powered per pair
  (2/27 significant, 62% ties) — cite them as a separate query shape, never averaged in.
  Side-by-side: `tools/eval/thematic_vs_deterministic.py`. **The BM25/hybrid arms
  (2026-07-30, `hybrid_significance_test_9way.py --thematic`) reverse this project's most
  robust finding and are the bigger result**: BM25 is weak on thematic (0.2988 vs 0.4930
  entity-anchored — no name to match exactly), so "hybrid beats dense-alone for every
  embedder" is **entity-anchored-specific**. On thematic recall@10 it is 3 significant for
  hybrid / 4 ties / **2 significant against** (`e5` −0.0445, `qwen3` −0.0526), and the
  hybrid−dense delta is monotone in dense strength (**r = −0.925**), flipping sign right at
  BM25's own score. General rule, now measured in both directions and subsuming the old
  m2v/sct "RRF failure case": **RRF helps the weaker arm and taxes the stronger one — fuse
  only when the two arms are comparable, whichever one is weak.** Report:
  `data/results/thematic_hybrid_significance_test.md`). Full process narrative: `docs/chunker-embedder-comparison-log.md`; clean
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
  **Refreshed 2026-07-29** against the OCR-remediation-rebuilt indices: latency/cost mechanics
  came back essentially unchanged (confirms these measure model/index/corpus-size mechanics, not
  corpus content), but the recall@10 columns in the report dropped substantially like every other
  quality number in this section (e.g. `qwen3 × semantic` dense 0.6581→0.5382,
  `qwen3_0.6b × semantic` dense 0.6364→0.5688) — per Open item #13 above, semantic is not a
  provable "best chunker", so don't cite this report's recall numbers as a chunker-supremacy
  claim, only as one representative combo's cost/quality profile; report at
  `data/results/cost_latency_pareto.md`.
- **Per-entity_type breakdown of BM25/hybrid (2026-07-29,
  `tools/eval/bm25_hybrid_entity_type_breakdown.py`) gives the mechanism behind the
  hybrid win, and one caveat to it.** BM25 alone scores **0.8147** on `person` queries —
  beating every embedder's dense-alone person score (best `bge_m3` 0.5735) outright —
  while collapsing to **0.3484** on `program`, where dense nearly doubles it
  (`qwen3_0.6b` 0.6023). **BM25 carries person (exact name match), dense carries program**;
  that is direct evidence for the complementarity the Open item #2 proxies never
  established. Caveat: **"hybrid never hurts" is an aggregate claim, not a per-category
  one** — on `person` specifically hybrid sits *below* BM25-alone for most embedders
  (`qwen3_0.6b` 0.7220, `qwen3` 0.7340, `jina_v5` 0.7382), only `bge_m3` (0.8211) exceeding
  it. Measured against the structural ceiling, hybrid reaches 84.1% on `person`, 72.3%
  `faculty_adjunct_aggregate`, 68.7% `program`, 65.1% `course` — **this reverses the old
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
  1. **Cross-encoder reranking hurts hybrid.** `CrossEncoderReranker`
     (`BAAI/bge-reranker-v2-m3`, `rerank_pool_size=50` → truncate to k=10) is wired as a
     query-time stage; `tools/eval/reranker_significance_test.py` re-retrieves live against
     `chunker_compare_full/plain__fixed_size__local__ceea7536` (so it goes stale on every
     index rebuild — it is **not** in the persisted-results refresh chain). Result:
     **significantly hurts hybrid MRR** (0.7775→0.6775, Holm-adj p=0.0048), **no significant
     effect on dense-alone**, and it costs ~1.17s/query. The nDCG@10 harm reported
     2026-07-23 (p=0.030) **did not replicate** post-refresh (p=0.5676) and is retired as a
     separate claim — MRR-only is the current framing, and it actually sharpens the
     literature's "phantom hits" mechanism (early-rank disruption without evicting relevant
     docs from the top-10). Literature grounding — including a paper naming
     `bge-reranker-v2-m3` by name — is in `docs/reranker-hybrid-interaction-research.md`.
     Untested follow-ups (a reranker trained on hybrid-fused candidates; blending its score
     into RRF as a 4th signal instead of truncate-and-replace) are hypotheses, not results.
  2. **RQ3 preprocessing ablations: normalization and word-aware segmentation do nothing;
     only chunk size matters, and only at 1024.** Configs `config/experiments/rq3_*.yaml`,
     scripts `tools/eval/rq3_*`. Thai normalization (Thai digits + `pythainlp.util.normalize()`)
     and word-aware `newmm`-boundary chunking are both **not significant on any metric**
     (Holm-adj p≥0.42 / ≥0.4524). Chunk size **is** significant but the citable claim is
     narrower than it used to be: **1024 loses significantly to both 512 and 256** (dense +
     hybrid recall@10, hybrid nDCG@10), while **256 vs 512 is a flat tie on dense retrieval
     with 512 numerically ahead** (0.4146 vs 0.4103, p=0.8802) — 256 only wins on hybrid
     recall@10. **Do not cite "smaller is monotonically better" or "256 is best"** (both were
     true of the 2026-07-23 numbers and did not replicate); the project's 512 default is not
     shown to be suboptimal, only 1024 is shown to be wrong. Note these ablations' treatment
     indices reuse `chunker_compare_full` combos as their *baseline* arm, so **an index
     rebuild silently turns them into a clean-baseline-vs-dirty-treatment confound** — they
     need real GPU rebuilds after a corpus change, not just a re-eval.
     **That confound is now LIVE, not hypothetical (verified 2026-07-30 by mtime):** the 4
     RQ3 treatment indices were built 2026-07-23, `chunker_compare_full` was rebuilt
     2026-07-28 for OCR remediation. So **every RQ3 number above is currently void** and
     must be treated as "unconfirmed, pending rebuild" rather than cited — in *both*
     directions: 1024's significant loss is inflated (1024 is the dirty arm), and the
     normalize/segmentation nulls are unsafe too (a handicapped treatment arm that still
     ties could be genuinely positive on equal footing). Rebuilding the 4 RQ3 indices
     alone would not fix it and would merely flip the confound's sign, because
     `chunker_compare_full` itself predates the 2026-07-30 corpus fixes: the only correct
     order is rebuild `chunker_compare_full` first (~20-24 h), then the 4 RQ3 indices
     (bge-m3, full corpus, ~3-6 h), then re-run the three tests.
- **RQ4 generation is COMPLETE (2026-07-30): 530/530, 0 errors, ~96 min.** Provisional
  numbers (inline script, **no bootstrap/Holm yet** — `rq4_score.py` is still the
  deliverable) in `docs/rq4-design.md`. Three findings worth knowing before writing any
  RQ4 text: (a) **citation precision orders exactly as recall@10 did** (hybrid 0.742 >
  dense 0.670 > bm25 0.625 > m2v 0.562) — retrieval quality survives the generation
  stage, and this one is unaffected by the correction below; (b) **citation *recall* is
  flat at ~0.41 across every arm — but the cause is probably OUR PROMPT, not the model,
  and this was first written up the wrong way.** Citations per answer sit at mean 2.65 /
  median 2 no matter how much gold is present, and recall *falls* as availability rises
  (0.778 at 2 gold docs → 0.381 at 5+) — the signature of a fixed budget, not of a model
  that cannot use evidence. Prompt rule 4 says `ตอบสั้น ๆ ไม่เกิน 3 ประโยค` while the gold
  set is dominated by aggregation queries (mean 9.87 relevant docs). **So do not cite
  "the generator is the bottleneck" — the recommendation would flip from "use a stronger
  model" to "fix the instruction".** Pending ablation: re-run hybrid + bm25 arms with
  rule 4 replaced by "cite every relevant document" (~212 gens, ~45 min); recall rises →
  prompt artifact, stays ~0.41 → real ceiling. **Run that BEFORE the deferred
  `gemma4:e4b` check** — a second model under the same cap would reproduce the flat line
  and look like confirmation while only re-measuring the instruction;
  (c) **0 fabricated citations out of 978** — RAG's most-feared failure mode is absent
  here, the payoff for exactly-checkable numeric labels. 4b's claims belong to the weak
  arms + closed-book only (context lacks the answer in 4/6 cases for hybrid/dense vs
  23/27/106 for bm25/m2v/closed-book), and closed-book abstaining 106/106 is the run's
  validity check. Caveat: citation precision is judged against the same qrels, so it
  inherits the pooling-bias threat — direction is conservative (see validity bullet).
- **RQ4 (end-to-end answer quality) design** —
  `docs/rq4-design.md` + its build log. Steps 1-2 built (`tools/eval/rq4_build_contexts.py`,
  `rq4_generate.py`); `rq4_score.py` is what remains. Local-only generation (`phi4`,
  no external API), objective citation/abstention metrics rather than LLM-as-judge.
  **RQ4 does not depend on the pending index rebuild** — contexts come from persisted
  retrieval results — but it *will* need re-running after one, like every other
  persisted-result consumer. Two hard-won gotchas: (a) **Ollama truncates an
  over-long prompt from the front**, so a default `num_ctx=4096` silently deleted
  the instructions on long prompts and produced fluent, plausible, citation-free
  answers — always set `num_ctx` and put instructions *after* the context; (b) the
  design doc's "recall@10 ~0.6 so the context often lacks the answer" was wrong
  (recall ≠ presence: 96% of contexts hold ≥1 gold doc), so 4b's power lives in the
  weak arms and closed-book, not the strong ones.
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
  3. **Circularity** in `entity_lookup`/`entity_boost`: their qrels come from the same
     `programs.json`/`people.json` the retrieval mode uses, so 0.9291 is an upper bound,
     not a measurement. Confined to those arms — chunker/embedder/BM25/hybrid never touch
     the dictionaries — but it needs an explicit paragraph in the paper, not a footnote.
  Also covered there: single-annotator labelling (defended by the labels being
  *rule-derived and re-derivable*, not judged), query provenance, external validity.
- **Candidate next axis, written up but not started**:
  `docs/colbert-late-interaction-notes.md` (ColBERT: motivated by *our own*
  results — the cross-encoder reranker hurt hybrid MRR, and BM25/dense split
  person vs program — with a pre-registered prediction so an aggregate win can't
  be mistaken for resolving that split). Not committed to; RQ4 is the one
  that blocks the paper.
- **Corpus data-quality audit** (`tools/corpus_prep/audit_title_body_agreement.py`,
  2026-07-30): flags manifest titles that disagree with the document's own page-1
  `เรื่อง` subject line. A first version was rejected on measurement (median 0.660,
  544 files below 0.5, nearly all artifacts); this one strips agenda numbering,
  compares by **token containment rather than string similarity**, and scores
  **asymmetrically** (what fraction of the *title's* words the subject line
  supports, so a truncated title scores 1.0). Result: median **1.000**, **7 flagged,
  7/7 genuine**. Report + per-case verdicts: `docs/title-body-agreement.md`. Two
  causes: 4 mispairings (metadata-only, incl. one A↔B swap) and 2 never-fetched
  documents (the CHECO shape). **Not applied** — mispairings change `resolution_id`s,
  so it is a decision; both should ride the rebuild already owed.
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
