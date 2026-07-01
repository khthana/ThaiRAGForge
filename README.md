# RAG Lab — Indexing + Retrieval Experimentation Framework

A modular bench for experimenting with the **indexing** and **retrieval** stages of a
RAG system over Thai academic-council resolutions (มติสภาวิชาการ KMITL). Swap any stage
— **Loader × Chunker × Embedder × Retriever** — run combinations, and compare which
setup retrieves the most relevant chunks for a query.

## Status

Design is complete; framework implementation has not started yet. The corpus already
exists (produced by the corpus-prep tooling below).

## Repository layout

```
├── CONTEXT.md                # domain glossary (start here)
├── docs/
│   ├── PRD-indexing-retrieval-framework.md   # what we're building (Thai)
│   ├── adr/                  # architecture decisions (why)
│   ├── grill-session-2026-07-01.md           # design interview log
│   └── req0-original-spec.md # original spec (superseded by the PRD)
├── tools/corpus_prep/        # scrape → OCR → clean pipeline that built the corpus
├── src/rag_lab/              # the framework (to be implemented per the PRD)
├── config/experiments/       # YAML experiment definitions
├── tests/                    # framework unit tests
└── academic_resolutions/     # the corpus (gitignored; ~1,215 .md resolutions)
```

## The corpus

`academic_resolutions/` holds OCR'd resolutions organised as
`year(พ.ศ.)/ครั้งที่ N/<เรื่อง>.md`. Each `.md` is one **Resolution**; sibling
`_LINK.txt` files hold the Google Drive URL of the original PDF (provenance, not
content). See `tools/corpus_prep/README.md` for how it was produced.

## Design documents

- **`CONTEXT.md`** — shared glossary (Resolution, Chunk, Index artifact, Retriever,
  Dense/BM25/Hybrid, Silver/Gold query sets, …).
- **`docs/PRD-indexing-retrieval-framework.md`** — problem, solution, user stories,
  implementation & testing decisions.
- **`docs/adr/0001`** — scope extends to retrieval; Index-build vs Retrieval phase split.
- **`docs/adr/0002`** — relevance is judged at the Resolution level, not the chunk level.

## Environment

- Python 3.13, managed with `uv`.
- Corpus-prep only: `uv sync`
- Framework: `uv sync --extra lab`
- Local embedding runs on GPU (developed on an RTX 3060 12GB). OCR uses a local Ollama
  model (see `tools/corpus_prep/`).
