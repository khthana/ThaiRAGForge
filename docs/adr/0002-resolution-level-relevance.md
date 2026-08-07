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

### Amendment (2026-07-30): one file, one id — enforced, not assumed

`resolution_id` is `<year>/<session>/<title>`, and `title` comes from
`meeting_manifest.json` (ADR-0003). Nothing in that chain guarantees uniqueness,
and "every Chunk carries a stable `resolution_id`" quietly assumed it: an audit
found **6 ids shared by 12 files** (of 2,853). Because relevance is judged here,
at the Resolution level, a shared id does not merely blur provenance — it merges
two documents into one relevance unit, so a top-k hit on either counts as a hit
on both, and for a gold query citing that id the metric is inflated by a
document that has nothing to do with it. Four were data errors (a title copied
onto the wrong file; a split bundle's two pieces patched with one curriculum
title) and were fixed at the source. The other two are two genuinely distinct
agenda items that one meeting's agenda really did list under one identical
title, where no correct different title exists to recover.

So uniqueness is now produced and checked rather than hoped for:

- `make_resolution_id` appends a ` #N` rank when a title is shared inside its
  meeting folder. The rank is folder-local, so an id does not depend on which
  subset of the corpus a run loads; rank 1 keeps the bare id, so a newly
  discovered clash mints an extra id instead of renaming one a gold query set
  or a built index already references. The suffix lands on the id only — the
  human-facing `title` stays as the agenda wrote it, and nothing is encoded in
  a filename (ADR-0003).
- `pipeline.build_index` refuses to build when two source files still reach it
  on one id, rather than producing an artifact whose every metric is wrong.
- `tools/corpus_prep/audit_resolution_ids.py` reports each clash with the
  evidence needed to tell a data error from a genuine shared title (does the
  manifest title match the filename, does the document's own first-page heading
  agree, do the two files point at the same source PDF), and exits non-zero —
  so disambiguation never silently papers over a fixable error.

**Closed out 2026-08-05.** The open question this amendment left was what to do
with the indices already built under the old ids — a relabelling tool was written
for it. In the event it was never needed: `chunker_compare_full` rebuild #3 ran
after the fix and so minted the corrected ids from scratch, and the one index
outside that chain (`entity_tags_full`) was rebuilt separately the same day for
exactly this reason. Re-scoring `entity_boost`/`entity_lookup` afterwards moved
one number materially (program MRR 0.6238 → 0.6544) and left the rest flat, which
is the expected shape: relabelling 6 files out of 2,853 cannot change recall, but
it can un-merge two documents that were ranked as one.

## Consequences

- The Chunk schema's `resolution_id` (≈ `req0.md`'s `Chunk.doc_id`) is load-bearing
  for evaluation, not just provenance.
- A corpus change that adds or retitles a file can change an id. Re-run the audit
  after one, and remember that a built index stores the ids it was built with:
  relabelling those artifacts (or rebuilding them) is what makes a gold set and
  an index agree again.
- Evaluation cannot reward "which chunk was best" — only "did a relevant Resolution
  surface in top-k." Chunk-granularity quality is assessed qualitatively (eyeball),
  not by the metric layer.
