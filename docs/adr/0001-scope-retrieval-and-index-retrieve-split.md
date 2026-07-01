# Scope extends to retrieval; pipeline splits into Index-build and Retrieval phases

---
Status: accepted
---

## Context

`req0.md` (the original written spec) scoped the framework to the *indexing* phase
only — Loader → Chunker → Embedder — and explicitly listed **retrieval / similarity
search**, **hybrid/lexical search**, and a **UI/Dashboard** as *out of scope* (§10),
with the pipeline output ending at "embeddings serialized to a file."

During grilling, the user's actual goal proved larger: the success criterion is
"enter a query → find the most relevant chunks," compared across module combinations,
through a **Streamlit UI**. The user further chose **hybrid/filter retrieval**, which
promotes Retrieval to a first-class swappable axis and makes the Loader's extracted
metadata (year, faculty, เรื่อง/title, NER entities) functionally relevant.

## Decision

Extend scope beyond `req0.md` to include **query-time retrieval** as a fourth
swappable module (Dense / BM25 / Hybrid + orthogonal metadata filter), plus a
Streamlit UI. Architecturally, split the pipeline into two phases:

1. **Index-build** (offline, expensive, cached) — each (Loader × Chunker × Embedder)
   triple produces a persisted **Index artifact**: chunks + embeddings + a BM25
   lexical index + metadata. This *is* `req0.md`'s serialized-index output.
2. **Retrieval** (query-time, cheap) — a Retriever ranks chunks from a chosen Index
   artifact for a query. Retrievers never re-embed.

Consequently the expensive build count is Loader × Chunker × Embedder (≈24), *not*
the full 4-axis cartesian (≈72); the Retriever axis is explored for free at query
time. The core is a plain, Streamlit-free, importable/testable Python package; the
UI and a Typer CLI are thin entry points over it.

## Consequences

- Deviates deliberately from `req0.md` §10 — a future reader comparing code to that
  spec will otherwise wonder why retrieval, hybrid search, and a UI exist.
- Cross-embedder similarity scores are not directly comparable (cosine 0.8 in e5 ≠
  0.8 in another space); the metric layer must rank/normalize within a combination,
  not compare raw scores across embedders.
- The split is hard to undo later: caching keys, artifact layout, and the UI's two
  modes all assume it.
