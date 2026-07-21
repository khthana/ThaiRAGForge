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
  whenever a headline number changes — the log stays append-only). Current bottom line
  (2026-07-21, bootstrap + Holm-corrected, all 9 embedders): the best system overall is
  **`semantic` chunking + `hybrid` retrieval (BM25 + dense via RRF)** — which embedder to pair
  it with is an open, untested horse race among the top five hybrid combos (`qwen3_0.6b`
  0.6935, `bge-m3` 0.6845, `e5_small` 0.6821, `qwen3` 0.6797, `jina_v5` 0.6796); none of them
  is significance-tested against the others yet, so don't cite any single one (bge-m3 included)
  as "the best combo" — the system-level claim (semantic + hybrid) is the robust part.
  Hybrid significantly beats dense-alone for every one of the 9 embedders on every metric
  (most robust finding of the comparison), and beats BM25-alone on recall for 7/9 embedders
  (not reliably on MRR/nDCG). Dense-alone, `bge-m3`, `Qwen3-Embedding-4B`, and
  `Qwen3-Embedding-0.6B` are a 3-way statistical tie at the top — pick by profile if not
  hybridizing (`bge-m3` = person-query specialist, `Qwen3-4B` = strongest generalist with no
  provable weak spot, `Qwen3-0.6B` = ties `Qwen3-4B` in aggregate but has a real person-query
  weak spot `Qwen3-4B` doesn't, `ConGen-PhayaThaiBERT` = program-query specialist). BM25 alone
  (`retrievers/bm25.py`) statistically **ties** that 3-way top tier and significantly beats
  every weaker embedder — but don't naively RRF a weak embedder with BM25: **both `m2v` and
  `sct`** (the latter even at its corrected 510-token context, ties m2v's near-random
  dense-alone performance) significantly *hurt* vs. BM25 alone — a real RRF failure mode
  whenever the fused dense signal is weak enough, regardless of why it's weak. Cost/latency:
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
