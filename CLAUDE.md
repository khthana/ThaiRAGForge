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
- The corpus (`academic_resolutions/`) is gitignored and lives at the repo root;
  corpus-prep tooling in `tools/corpus_prep/` needs Poppler + Ollama.
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
  yet wired into any loader/config** — needs a thematic-inclusive eval
  before integration (course-name gold queries can't detect a
  description-stripping regression, since they never touch description
  text). Narrative: `docs/chunker-embedder-comparison-log.md` (course-table
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
  queries). Full process narrative: `docs/chunker-embedder-comparison-log.md`; clean
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
  recall@10 and nDCG@10 (still ties all three on MRR) and drops out of the cluster. The
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
