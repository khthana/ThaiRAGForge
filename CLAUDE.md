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
  none` — Mode B (Query & Compare, the main script) plus Mode A (Build/Run,
  `app/pages/1_build_run.py`) in the sidebar nav. The `--server.fileWatcherType none`
  flag suppresses a harmless but noisy `ModuleNotFoundError: torchvision` warning:
  Streamlit's auto-reload watcher walks every loaded module's `__path__`, which
  triggers `transformers`' lazy-import machinery on unrelated submodules (e.g.
  `zoedepth`) that need optional deps we don't install. Both modes are thin shells
  over `rag_lab.query_service` / `rag_lab.runner` + `rag_lab.config`
  (the tested core); the widgets themselves are smoke-tested via
  `streamlit.testing.v1.AppTest`, not unit-tested individually.

## Conventions

- The core package `src/rag_lab/` must not import Streamlit (ADR-0001): keep it
  importable and unit-testable; UI/CLI are thin layers on top.
- Add a strategy by creating a file + registering it (`src/rag_lab/registry.py`);
  never edit the runner (Open/Closed).
- `Chunk.resolution_id` is load-bearing — relevance is judged at the Resolution level
  (ADR-0002).
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
  Status/handoff: `docs/llm-ocr-scan-log.md`. Complete and written back into
  the real corpus (commit `b692480`, 2026-07-16): 753/768 consensus-flagged
  pages live, 18 kept old on human review, no outstanding blockers.
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
  whenever a headline number changes — the log stays append-only). **Refreshed 2026-07-25**:
  the bottom line below was recomputed against the clean, rebuilt `chunker_compare_full`
  indices (0% corpus-discovery contamination, see
  [[project_corpus_discovery_contamination_bug]] / `docs/chunker-embedder-comparison-log.md`,
  fix `8c86b63`/`b36f96f`/`dd0c0ae`, rebuild commit `2d36663`) — every conclusion below held
  through the refresh (none flipped), with numbers now slightly stronger in most cases. Current
  bottom line (2026-07-25, bootstrap + Holm-corrected, all 9 embedders — clean-index numbers):
  the best system overall is
  **`semantic` chunking + `hybrid` retrieval (BM25 + dense via RRF)** — the single best combo is
  now `semantic × qwen3_0.6b` at recall@10=0.7048 (up from 0.6935 pre-rebuild), beating every
  other chunker's best combo (`recursive` 0.6800, `sentence` 0.6529, `fixed_size` 0.6322).
  Cross-chunker-averaged, the top hybrid embedders (`qwen3_0.6b` 0.6571, `bge-m3` 0.6563,
  `congen` 0.6467, `qwen3` 0.6291, `e5_small` 0.6289, `jina_v5` 0.6270) are close together, and
  **the dedicated semantic-only top-5 pairwise tie test
  (`tools/eval/hybrid_significance_test_semantic_top5.py`) was re-run 2026-07-25 against the
  clean indices** — still no pair significant on any metric, confirmed genuine tied cluster
  (`qwen3_0.6b` 0.7048, `bge-m3` 0.6893, `e5_small` 0.6871, `qwen3` 0.6832, `jina_v5` 0.6703
  recall@10 semantic-only). Don't cite a single "best combo" — the tie is confirmed, not
  provisional. Hybrid significantly beats dense-alone for essentially every one of the 9
  embedders on every metric (26/27 tests significant; the one exception is `qwen3` on MRR,
  Holm-adj p=0.09) — still the most robust finding of the comparison — and beats BM25-alone on
  recall for 6/9 embedders (`jina_v5`/`sct` tie BM25 instead of beating it, `m2v` significantly
  loses to it; not reliably ahead on MRR/nDCG either). Dense-alone, `bge-m3`, `Qwen3-Embedding-4B`, and `Qwen3-Embedding-0.6B` are still a
  3-way statistical tie at the top — pick by profile if not hybridizing (`bge-m3` = person-query
  specialist, `Qwen3-4B` = strongest generalist with no provable weak spot, `Qwen3-0.6B` = ties
  `Qwen3-4B` in aggregate but has a real person-query weak spot `Qwen3-4B` doesn't,
  `ConGen-PhayaThaiBERT` = program-query specialist). BM25 alone (`retrievers/bm25.py`)
  statistically **ties** that 3-way top tier and significantly beats every weaker embedder — but
  the per-chunker breakdown (`tools/eval/bm25_vs_embedder_significance_test_per_chunker.py`,
  re-run 2026-07-25) shows this "tie" is chunker-dependent: `bge-m3` actually loses to BM25
  significantly under `sentence` chunking specifically, and `qwen3`/`qwen3_0.6b` are the only
  embedders where BM25's margin goes numerically negative anywhere (`semantic` only) — the
  aggregate "ties" framing is most true for `semantic` chunking, the one this project already
  recommends. Don't naively RRF a weak embedder with BM25: `m2v` significantly *hurts* vs. BM25 alone on all
  3 metrics; `sct` (at its corrected 510-token context) now hurts significantly on MRR/nDCG@10
  but recall@10 is no longer significant post-refresh (Holm-adj p=0.08, was significant
  pre-rebuild) — a real RRF failure mode whenever the fused dense signal is weak enough, though
  `sct`'s exact severity shifted slightly with the clean indices. Cost/latency:
  `tools/eval/cost_latency_pareto.py` (vector dim, index size, query latency p50/p95) found
  `HybridRetriever.retrieve()` and `BM25Retriever.retrieve()`'s current implementation
  (full-corpus `k=n` fetch before fusing, `BM25Okapi` rebuilt from scratch every query) adds a
  roughly **fixed ~2.1-2.3s of overhead to every hybrid query, nearly independent of embedder**
  (it scales with corpus size, not embedding dim) — the ~2.3-2.9s measured figure is mostly this
  avoidable per-query overhead on top of a ~130-730ms intrinsic cost, not RRF fusion itself;
  report at `data/results/cost_latency_pareto.md`.

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
