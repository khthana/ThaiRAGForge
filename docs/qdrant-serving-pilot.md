# Qdrant serving pilot (2026-08-13)

Artifacts: `tools/eval/qdrant_pilot_ingest.py`, `tools/eval/qdrant_pilot_test.py` →
`data/results/qdrant_pilot.md` + `qdrant_pilot_raw.json`.

This is the first piece of work in this project aimed at **deployment rather than the
paper** (`[[project_real_deployment_intent]]`, stated 2026-08-13). It asks whether the
retrieval numbers this project publishes survive being served out of a vector database,
and it deliberately answers that in two separable halves rather than one.

## 1. Why the pilot had to change shape before it started

The 2026-07-16 Qdrant vertical slice ran **embedded** (`QdrantClient(path=...)`), and
embedded Qdrant is **exact brute force, not ANN**: `LocalCollection.search` scores every
vector and `np.argsort`s them. The `HnswConfig` it reports back is a fabricated default —
nothing traverses a graph. So the slice could not have answered "does ANN change the
answer" no matter how it was read, and the 20k-point warning it recorded was about a code
path that does no approximate search at all.

That is why this pilot runs a real `qdrant/qdrant` **server container**
(`v1.18.0`, `rag-qdrant`, `http://127.0.0.1:6333`, named volume `rag_qdrant_storage`).
Client and server are held at the **same** version on purpose: the first ingest ran
client 1.18.0 against server 1.15.1 and raised a compatibility warning, which was closed
by pulling the matching image and re-ingesting onto a clean volume rather than by
silencing the check.

## 2. The two questions

**Q1 (dense).** Does HNSW change the answer relative to the exact numpy search every
published number in this project was measured with?

**Q2 (lexical).** Does Qdrant's sparse arm reproduce `BM25Okapi`?

Q1 is measured as **three** arms, not two, because a straight numpy-vs-ANN comparison
bundles two independent causes:

| comparison | isolates |
|---|---|
| `numpy_exact` vs `qdrant_exact` (`SearchParams(exact=True)`) | storage + arithmetic (f32 in the engine vs float64 in numpy, cosine normalised on write, tie-break convention) |
| `qdrant_exact` vs `qdrant_ann_ef*` | HNSW traversal, and nothing else |

Q2 is **exact by construction and is therefore a check, not a result**. Ingestion
precomputes each chunk's BM25 document weights using `BM25Okapi`'s *own floored IDF
table*; the query sends term counts; the engine takes a plain sparse dot product.
`Modifier.IDF` is deliberately **not** used — Qdrant's IDF is not this project's IDF, and
using it would make the arm a different scorer wearing the same name. The vocabulary is an
explicit sorted enumeration sidecar (78,333 terms, 2.1 MB JSON), not a hash, so a term id
is auditable.

Fusion is this repo's own RRF (k=60, 0.5/0.5, dense-first tie-break) for **both** arms, so
the engine's `FusionQuery` is not a variable in any comparison here.

**Honesty note on registration.** Q1 and Q2, the three-arm decomposition and the
"exact by construction" design were all fixed before the measurement ran, but they were
fixed *in conversation*; this file is written after the numbers exist. Read it as a
**record**, not as a frozen pre-registration of the kind `docs/rq4-design.md` or
`docs/colbert-late-interaction-notes.md` carry. Nothing here is a hypothesis test —
one combo, no significance testing, everything descriptive.

## 3. Setup

- Collection `plain__sentence__local__bf8b7ebb` (the `person` route's shipped target:
  sentence chunker × BAAI/bge-m3), **57,174 chunks**, dim 1024.
- 106 Gold `73det` queries, K=10 sent, `fetch_depth=200` (the shipped query-time value).
- Ingest self-checks all PASS: vocabulary bijection over 78,333 terms; sparse dot vs
  `get_scores` worst relative error **5.502e-08**; point count 57,174 == index rows;
  payload/vector at id *i* == row *i* over 200 sampled; HNSW built, status green.

## 4. Q1 — the answer is that ANN is a trade this collection does not have to make

| arm | recall@10 | Δ vs numpy_exact | agree@10 vs qdrant_exact | p50 ms |
|---|---|---|---|---|
| numpy_exact | 0.3954 | +0.0000 | 0.9858 | 195.0 |
| qdrant_exact | 0.3957 | +0.0003 | 1.0000 | 17.8 |
| qdrant_ann_ef16 | 0.3347 | −0.0606 | 0.8745 | 10.4 |
| qdrant_ann_ef64 | 0.3398 | −0.0555 | 0.8764 | 9.9 |
| qdrant_ann_ef128 | 0.3533 | −0.0421 | 0.9075 | 10.7 |
| qdrant_ann_ef256 | 0.3777 | −0.0177 | 0.9528 | 10.8 |
| qdrant_ann_ef512 | 0.3872 | −0.0082 | 0.9717 | 11.9 |
| qdrant_ann_ef1024 | 0.3926 | −0.0028 | 0.9783 | 13.3 |

**Storage and arithmetic are free**: `qdrant_exact` lands within **+0.0003** of the numpy
arm the whole project is measured on, with agree@10 0.9858 and the residual sitting in
tie-break convention. Whatever else moves, it is not f32 or the cosine normalisation.

**HNSW costs accuracy and buys almost nothing here.** The loss shrinks monotonically with
`ef` but is still −0.0028 at `ef=1024`, while `qdrant_exact` — which has no loss at all —
costs **17.8 ms p50** against ANN's 10-13 ms. Six milliseconds is not a reason to accept a
worse ranking on a 57k-vector collection.

### The `ef` < `limit` trap, which is worth more than the table

**Every row with `ef` below the 200-result request is a malformed request, not a
measurement of HNSW.** The beam holds `ef` candidates; asking it for 200 results from a
beam of 128 is beam-starved by construction, and its loss says the request was wrong, not
that the graph is inaccurate.

This was **my own confound**, caught after a full run had already been read. The first
grid topped out at 256 and I read the −0.0421 at `ef=128` as an ANN cost; it is not. The
grid was widened to reach past `FETCH_DEPTH` and the run repeated.

Those rows are nonetheless **kept in the published table**, because `ef=128` is Qdrant's
own default: it is exactly what a deployment gets by configuring nothing, so it is the
trap an operator falls into, and deleting it would hide the most likely real-world
misconfiguration behind a clean-looking curve. **Any deployment that raises `fetch_depth`
must raise `hnsw_ef` with it** — the two are coupled and only one of them is in this
repo's code.

## 5. Q2 — the sparse arm reproduces `BM25Okapi`, and the residual is tie order

| arm | recall@10 | agree@10 | worst rel. score gap | id disagreements at non-tied ranks | p50 ms |
|---|---|---|---|---|---|
| numpy_bm25 | 0.5034 | 1.0000 | 0.00e+00 | 0 | 219.7 |
| qdrant_sparse | 0.5034 | 0.9774 | 2.00e-07 | 0 | 8.8 |

Identical recall, score sequence agreeing to **2.00e-07 relative at every rank**, and
**zero** id disagreements at any position where the scores actually differ.

**The check had to be corrected, and that is the finding to keep.** Its first version
demanded rank-for-rank identical `chunk_id`s and failed 3/3 on the smoke run. Diagnosed
before changing any code: the scores agreed to ~1e-6 and the `resolution_id`s matched, and
every differing id sat inside an **exact BM25 tie group** (four chunks at 50.677741 on one
query — tie groups are large on this corpus). numpy's `argsort` and Qdrant's scan settle a
tie differently and **neither is more correct**. So "exact by construction" is a claim
about the **score sequence**, never about tie order, and the check now tests that. The
check was wrong; the ingestion was right.

The same convention explains the one set-level miss at depth 200 (200/200, 200/200,
199/200 across the smoke queries): the F=200 cut passes through a tie group, so the two
engines keep different members of it.

## 6. End to end, and what to deploy

| arm | recall@10 | MRR | nDCG@10 | agree@10 vs reference |
|---|---|---|---|---|
| reference (numpy dense + `BM25Okapi`) | 0.5834 | 0.7471 | 0.6360 | — |
| **served, exact** (`exact=True` + qdrant sparse) | **0.5851** | 0.7512 | 0.6383 | 0.9547 |
| served, ANN (`hnsw_ef=512` + qdrant sparse) | 0.5634 | 0.7632 | 0.6215 | 0.9321 |

**Recommendation: serve dense with `exact=True` and sparse as ingested.** It reproduces
the reference ranking (**+0.0017**, inside tie-break noise) where ANN loses **−0.0199**,
and it needs no `ef` tuning to stay correct as `fetch_depth` changes.

The latency case is the strong half. Per-arm p50, one process, one loaded index:

| | numpy | qdrant |
|---|---|---|
| dense, exact | 195.0 ms | **17.8 ms** |
| lexical (BM25) | 219.7 ms | **8.8 ms** |

Read these as **within-process** comparisons only — the numpy arms pay no network hop and
the Qdrant arms pay REST serialization, so the deployable figure is the served total, not
an arm-by-arm subtraction. Even discounted, the direction is not marginal: the engine
replaces ~0.4 s of per-query Python scoring with ~27 ms of engine work, on the workload
(≤ 50 concurrent users on a faculty VM) where per-query CPU is what the box actually runs
out of.

`S2` anchors the whole pilot against published numbers from an independent code path: the
reference fusion scores **0.5834** at F=200 against the persisted `gold_hybrid_73det`
**0.5850** at k=n, and the −0.0016 is the already-measured `fetch_depth` truncation
effect, not a serving artifact.

## 7. Observations recorded rather than glossed

- **`indexed_vectors_count` reads 110,422 against 57,174 points** (~1.93×, 6 segments).
  It is a reported counter, not duplicated data: `points_count` equals the index row count
  exactly, the payload/vector alignment check passes 200/200, and every search returns
  distinct ids. Most likely cross-segment accounting during optimization. **Unexplained,
  and reported as observed** — it has no effect on any number above, and this note exists
  so the next reader does not rediscover it as a scare.
- **Recall@10 of the dense arm alone (0.3954) is far below the fused 0.5834.** That is the
  published behaviour of this combo, not a pilot defect — `bge-m3` is the person
  specialist and BM25 carries `person` at 0.8147.
- The `hnsw_ef` grid is **not inert**: 293 distinct top-10s across 6 `ef` values × 106
  queries, so HNSW really is being traversed and the ANN arms are not silently exact.

## 8. What this pilot does NOT establish

- **One collection, one combo, one route.** The other three routed collections are not
  ingested; nothing here says their numbers transfer.
- **No significance test.** Every Δ above is descriptive.
- **No concurrency measurement.** The stated deployment target is 5-50 concurrent users
  on a faculty VM with a separate GPU server; this pilot ran one query at a time in one
  process, so it says nothing about throughput, contention, or what happens when the
  embedder is on a different machine from the engine.
- **Nothing is wired.** `query_service`/`registry` still route to the in-process
  retrievers; adopting Qdrant in the serving path is a separate decision that has not been
  taken.
