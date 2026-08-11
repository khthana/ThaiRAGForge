# ColBERT / late interaction — notes and a pre-registered test

Status: **not started, not committed to.** Written 2026-07-30 so the option is
recorded with its reasoning while the findings that motivate it are fresh. This
is a *new research axis*, not a small experiment; the intended order was RQ4
first.

**2026-08-07 — RQ4 is complete, so the stated blocker is gone; this is still not
the next thing to do.** Three cheaper items now rank above it, all justified by
this project's own measurements rather than by the literature: (1) caching
`BM25Okapi` instead of rebuilding it per query, which returns a fixed ~1.9–2.0 s
of every hybrid query for no research risk; (2) sweeping the fusion weight —
`HybridRetriever` has a `method="weighted"` path with `dense_weight`/`bm25_weight`
that **has never been used**, so every hybrid number in the project sits at an
implicit 50:50, and the "RRF helps the weaker arm and taxes the stronger one"
rule (r = −0.921) predicts where a sweep pays; (3) the two cheap graph edges.
ColBERT stays ranked where it is because it is the only remaining item whose
index-side cost is comparable to a full rebuild. The pre-registered prediction
below is unchanged and still the point of the exercise.

## What it is

Three retrieval architectures, of which this project currently uses the outer two:

| architecture | encodes | scoring | in this project |
|---|---|---|---|
| bi-encoder (dense) | whole chunk → **one** vector | dot product, precomputed | all 9 embedders |
| **ColBERT (late interaction)** | **each token → its own vector** | MaxSim: each query token takes its best-matching document token, summed | — |
| cross-encoder | nothing precomputed; query+doc go through the model together | full attention | `bge-reranker-v2-m3` (tested, **harmful**) |

The name is the mechanism: interaction between query and document is *deferred*
to query time, but the document side is still precomputed, so it remains indexable.
A bi-encoder destroys token-level detail at index time by pooling a whole chunk
into one vector; a cross-encoder keeps everything but cannot precompute anything.
ColBERT keeps per-token vectors and pays for it in storage instead of latency.

Primary references (verify details before citing):
- Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via
  Contextualized Late Interaction over BERT*, SIGIR 2020 — the architecture.
- Santhanam et al., *ColBERTv2: Effective and Efficient Retrieval via Lightweight
  Late Interaction*, NAACL 2022 — residual compression, which is what makes the
  storage argument below work.
- `jina-colbert-v2` (2024) — multilingual late-interaction checkpoint; the
  practical option here, since most ColBERT checkpoints are English-only.

## Why it is a good fit for *this* project specifically

Not a generic "newer is better" argument — three of this project's own measured
results point at it.

1. **Our cross-encoder reranking failed, and late interaction is the principled
   alternative to it.** `bge-reranker-v2-m3` significantly *hurt* hybrid MRR
   (0.7775 → 0.6775, Holm-adj p=0.0048), with the literature's "phantom hits"
   mechanism: a reranker trained on first-stage candidates disrupts early ranks
   when applied to fused ones. ColBERT delivers the token-level matching a
   cross-encoder provides, but as the **retriever itself** rather than as a
   truncate-and-replace stage on top of someone else's ranking — so the specific
   failure mode we measured does not apply.

2. **It targets the exact complementarity we quantified.** The per-entity_type
   breakdown found BM25 carries `person` outright (0.8147, beating every dense
   embedder's best of 0.5735) while collapsing on `program` (0.3484), where dense
   nearly doubles it (0.6023). That is the classic exact-match-vs-semantics split,
   and MaxSim is exact-ish matching *in embedding space* — the one mechanism that
   could plausibly do both without RRF. Worth noting we also measured that RRF is
   not free: fusing a weak dense signal with BM25 (`m2v`, `sct`) is significantly
   *worse* than BM25 alone.

3. **It sidesteps Thai word segmentation on the lexical side.** BM25 here depends
   on `newmm` segmentation; ColBERT scores over subword tokens with no word
   boundaries required. RQ3 found word-aware segmentation made no difference to
   *chunking*, but that says nothing about lexical matching, where segmentation is
   load-bearing.

## Storage: not the blocker it is assumed to be

Sized against a real index (`plain__semantic__local__834c4336`):

| | value |
|---|---|
| chunks | 74,819 |
| text | 23.4M characters (mean 313/chunk) |
| estimated tokens | ~8-12M (Thai at roughly 2-3 chars/token on multilingual subword vocabularies — **measure with the actual tokenizer before relying on this**) |
| current single-vector index | 1024-dim float32 = **306 MB** |
| ColBERT raw, 128-dim fp16 | ~2.5-3 GB |
| ColBERT with ColBERTv2 residual compression (~32 B/token) | **~300-400 MB** |

So a compressed late-interaction index lands in the same order of magnitude as
the single-vector index we already ship. On a 12 GB card this is workable.

## The real blockers

1. **It breaks an index invariant we now enforce.** `chunks.parquet` and
   `embeddings.npy` are strictly row-aligned, one row per chunk — audit check I1
   verifies exactly that, and that alignment is what makes `resolution_id`
   attribution safe. ColBERT is many vectors per chunk, so it needs a different
   artifact shape plus an I1 variant that checks *chunk→token-block* alignment
   instead. This is the actual work; the retriever itself is a registry entry
   (ADR-0001 open/closed — new file + register, no runner edit).
2. **Checkpoint choice.** `jina-colbert-v2` is the realistic multilingual option
   and conveniently the same family as the `jina_v5` embedder already in the
   matrix. Whether it handles Thai council prose well is an empirical question, not
   an assumption — a pilot on a subset should precede any full index build
   (`feedback_scan_before_broad_preprocessing_fix`).
3. **Scope.** ColBERT is a retrieval-side axis that sits alongside RQ1-RQ2. This
   used to read "RQ4 is unstarted and is the gap that blocks the paper";
   **RQ4 completed 2026-08-03** (and has been refreshed twice since), so that is
   no longer the reason to defer. What replaces it is a sharper one, measured
   after these notes were written: the routed oracle over a P=50 pool delivers
   **0.8331** against arm C's 0.6831 (`data/results/reranker_rrf_routed_test.md`),
   i.e. **+0.1500 of headroom that a better ranker could reach and
   `bge-reranker-v2-m3` reaches 1% of** — and swapping the model moves recall@10
   by 20x the anchor's own effect (`reranker_model_comparison.md`). So the axis is
   motivated by a *measured* ceiling now, not by a gap in the study.

## A pre-registered prediction, so this is falsifiable

If the motivation in §2 is right, ColBERT-alone should show a **specific**
signature on the per-entity_type breakdown, not merely a better aggregate:

> ColBERT-alone ties or beats **BM25** on `person` **and** ties or beats the best
> **dense** embedder on `program`, in the same run.

That is the thing neither family can currently do alone. Writing it down now
matters: an aggregate improvement could equally come from simply being a bigger,
better-trained model, which would be a much weaker claim. If ColBERT wins overall
but keeps the same person/program split, the honest conclusion is "a stronger
retriever", not "late interaction resolves the complementarity".

Evaluation cost is low — it reuses `gold_query_set_73det.yaml`, the existing
paired-bootstrap + Holm machinery, and `bm25_hybrid_entity_type_breakdown.py`
unchanged. Only the index build and a `ColbertRetriever` are new.

## Where this belongs in the paper regardless

Even unbuilt, late interaction should appear in related work. An IR reviewer will
ask why a project that tested a cross-encoder reranker did not test ColBERT, and
the answer is a strength rather than an excuse: **we measured the cross-encoder
reranker making hybrid worse** when it replaces the ranking, and — after these
notes were written — that it adds **+0.0017 (ns)** on top of the shipped router
even when fused as a fourth RRF signal, against an oracle **+0.1500** over the
same pool. State it that way rather than as the bare "it hurt": the evidence that
motivates late interaction is that the *evidence is reachable and this model does
not reach it*, which is exactly what a scoring mechanism change is for.
