# Relevance is judged at the Resolution level, not the Chunk level

---
Status: accepted
---

## Context

The framework retrieves and displays **Chunks**, and the naïve instinct is to label
ground truth at the same granularity ("chunk #2 answers this query"). But the whole
point of the framework is to swap **Chunkers**, and chunk boundaries — and therefore
chunk identities — change with every Chunker. A chunk-level label produced under a
fixed-size chunker is meaningless under a semantic chunker that split the same
Resolution differently.

The corpus has a stable natural unit above the chunk: the **Resolution** (one
academic-council มติ, one source `.md` file), which persists regardless of Chunker.

## Decision

Anchor relevance at the **Resolution** level. Retrieval still returns and displays
top-k **Chunks**, but a result is judged "correct" by mapping each retrieved chunk
back to its `resolution_id`: a Query's ground truth is "Resolution R is relevant,"
and a **Hit** is a top-k chunk whose source Resolution is labelled relevant. Metrics
(recall@k, MRR, nDCG) are computed over Hits at the Resolution level. Every Chunk
therefore must carry a stable `resolution_id`.

This makes labelled evaluation reusable across every Chunker without re-annotation,
and lets the **Silver query set** (each Resolution's เรื่อง/title as a query relevant
to itself) work uniformly across combinations.

## Consequences

- The Chunk schema's `resolution_id` (≈ `req0.md`'s `Chunk.doc_id`) is load-bearing
  for evaluation, not just provenance.
- Evaluation cannot reward "which chunk was best" — only "did a relevant Resolution
  surface in top-k." Chunk-granularity quality is assessed qualitatively (eyeball),
  not by the metric layer.
