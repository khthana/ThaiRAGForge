# RQ4 design: does better retrieval produce better answers?

Status: **design, not yet built** (drafted 2026-07-30). RQ1-RQ3 are complete and
measure retrieval only; RQ4 is the last unstarted research question.

## The constraint that shapes the whole design

**No external API.** Generation must run on the local RTX 3060 (12 GB), so the
generator is one of the Ollama models already present: `phi4` (9.1 GB),
`gemma4:e4b` (9.6 GB), `phi4-mini` (2.5 GB). Two consequences, both design-level
rather than cosmetic:

1. **Generation and retrieval cannot run concurrently.** A retrieval sweep already
   occupies ~9.8 GB of the 12 GB card, and either full-size chat model needs
   ~9-10 GB alone. RQ4 runs must be sequenced after any retrieval job, never
   alongside — the harness should retrieve once, persist contexts to disk, then
   generate from those files. **One model resident at a time, always**: each arm
   loads its generator, runs, and unloads it (`keep_alive=0`, the same call
   `ocr_pdf_to_md.reset_model` uses) before the next arm loads anything. Two
   9-10 GB models on a 12 GB card will either fail or silently spill to CPU and
   make the timing numbers meaningless.
2. **LLM-as-judge is off the table**, and that is a feature. The only judges
   available are the same models doing the generating, so any judge score would
   carry self-preference bias that cannot be controlled for. RQ4 therefore rests
   on **objective metrics computed from the existing ground truth**, which this
   project happens to be unusually well set up for: relevance is already defined
   at resolution level (ADR-0002), so "did the answer cite the right documents"
   is checkable without a judge.

Honest limitation to state in the paper: these local models are not strong Thai
generators. RQ4 measures *whether retrieval quality propagates to answer quality
under a fixed, modest generator* — not the ceiling of what a frontier model could
do with this corpus. That is still the question RQ4 needs to answer, but the
absolute numbers are a floor, not an estimate of best achievable quality.

## Question, decomposed

> Does the retrieval ranking established in RQ1-RQ3 translate into better
> generated answers, or does it wash out once a generator is in the loop?

Three sub-questions, each with an objective measurement:

### 4a. Citation grounding — does the answer point at the right resolutions?

Prompt the model to answer **with explicit citations** to the resolutions it used
(the context blocks are labelled with their `resolution_id`). Then compare the
cited set against the query's `relevant_resolution_ids`:

- **citation precision** = cited ∩ gold / cited — how much of what it cited was
  actually relevant;
- **citation recall** = cited ∩ gold / gold — how much of the answer it should
  have given it actually found;
- **unsupported-citation rate** = citations naming a `resolution_id` that was not
  in the provided context at all (a fabrication mode this setup can detect
  exactly, because the context is known).

This is the core metric: fully objective, reuses the 106-query deterministic gold
set unchanged, and measures the thing a user of this system would care about
(can I trust the citation and go read the document).

### 4b. Abstention correctness — does it say "not found" when the answer is absent?

Retrieval recall@10 for the best configuration is ~0.6, so **the context often
does not contain the answer**. That makes the interesting case measurable: for
each query we know from the retrieval results whether *any* gold document made
the context.

|                              | context has ≥1 gold doc | context has none |
|------------------------------|-------------------------|------------------|
| model answers substantively  | expected                | **hallucination**|
| model abstains               | **missed**              | expected         |

Report the two error rates separately. The bottom-right cell is where a RAG system
earns trust, and no existing number in this project speaks to it.

### 4c. Does the retrieval ranking survive? — the actual RQ4 comparison

Hold the generator and prompt fixed; vary only retrieval, using configurations
whose retrieval-level differences are already established and significance-tested:

| arm | rationale |
|---|---|
| hybrid × `qwen3_0.6b` × semantic | best-measured configuration |
| dense-alone, same embedder/chunker | hybrid-vs-dense is the most robust RQ2 finding (26/27 significant) |
| BM25-alone | free baseline; carries `person` queries outright (0.8147) |
| `m2v` hybrid | known RRF failure case — does a bad retriever visibly damage answers? |
| no retrieval (closed-book) | floor: how much of the apparent quality is the corpus at all |

If 4a/4b metrics order these arms the same way recall@10 does, retrieval quality
propagates. If the middle arms collapse together, the honest finding is that
answer quality saturates once retrieval is "good enough" — which is itself a
publishable result and directly relevant to whether the cost of a 4B embedder is
justified end-to-end.

## Scope and cost

- Query set: `config/eval/gold_query_set_73det.yaml` (106 queries) — the same set
  every cited number uses, so RQ4 is comparable to RQ1-RQ3 by construction. The
  179 thematic queries are excluded until their own re-eval lands.
- 106 queries × 5 arms = **530 generations**, one pass, no sampling variance to
  average over (temperature 0).
- Rough local cost: with ~4-6k input tokens of context and a short answer,
  `phi4` on this card runs on the order of a few seconds per generation → **under
  an hour per arm**, so the whole matrix is a few hours of GPU with no retrieval
  running.
- Retrieval contexts come from the **already-persisted** result files for the
  arms above, so no new retrieval sweep is needed except for the `m2v` and
  closed-book arms.

## What must be built

1. `tools/eval/rq4_build_contexts.py` — read persisted retrieval results for each
   arm, assemble the prompt context (top-k chunks, each labelled with its
   `resolution_id`), and persist one JSON per (query, arm). Deterministic, no GPU.
2. `tools/eval/rq4_generate.py` — Ollama chat over those context files, **one arm
   and one resident model at a time**, unloading between arms, resumable per query
   (the thematic bootstrap's pattern: write each result immediately so an
   interrupted run loses nothing). Must refuse to start if `ollama ps` shows
   another model already resident, rather than trusting the card to cope.
3. `tools/eval/rq4_score.py` — parse citations, compute 4a/4b metrics, and run the
   same paired-bootstrap + Holm machinery the retrieval tests use so arm
   comparisons are significance-tested, not eyeballed.

## Decisions still open

- **Generator**: `phi4` vs `gemma4:e4b`. Worth a 20-query pilot on both before
  committing — Thai instruction-following and citation-format compliance differ,
  and a model that will not emit parseable citations makes 4a unmeasurable.
- **Citation format**: a fenced list of ids is easiest to parse reliably; free-text
  citation extraction would add its own error term to every 4a number.
- **k**: 10 matches every retrieval number, but 10 chunks of Thai council text may
  overflow a small context window. If k must drop for the generator, report
  retrieval metrics at the same reduced k so the arms stay comparable.

Per the project's own lesson (`feedback_scan_before_broad_preprocessing_fix`), the
pilot in the first open decision should happen before the full 530-generation run.

---

## Build log (2026-07-30): steps 1-2 done, and two corrections to the design above

**Status**: `rq4_build_contexts.py` and `rq4_generate.py` are built and committed;
the full 530-generation matrix is running on `phi4`. `rq4_score.py` is the
remaining piece. Contexts live in `data/rq4/contexts/`, answers in
`data/rq4/answers/<model>/<arm>/` (both gitignored — regenerable).

### Correction 1: §4b's premise was wrong

> "Retrieval recall@10 for the best configuration is ~0.6, so **the context often
> does not contain the answer**."

That confuses *recall* (what fraction of the gold documents were retrieved) with
*presence* (did **any** gold document reach the context). Measured:

| arm | context has ≥1 gold doc |
|---|---|
| hybrid × qwen3_0.6b × semantic | 102/106 (96%) |
| dense × qwen3_0.6b × semantic | 100/106 (94%) |
| BM25-alone | 83/106 (78%) |
| hybrid × m2v | 79/106 (75%) |
| closed-book | 0/106 |

So on the best arm only **4** queries are ones where abstention is correct — far
too few to measure a hallucination rate. 4b survives, but its statistical power
lives in the *weak* arms (BM25 23, m2v 27) and above all in **closed-book, where
all 106 are no-gold by construction** and hallucination is measured cleanly.
Report 4b per arm; do not pool.

### Correction 2: the generator was not the problem — the context window was

The design doc's first open decision ("a model that will not emit parseable
citations makes 4a unmeasurable") turned out to be a real risk arriving by an
unexpected route. The first pilot produced **0/4 citations** on prompts carrying
context, while the short closed-book prompts came back **4/4 correctly
formatted** — same model, same instructions.

Cause: **Ollama truncates an over-long prompt from the front**, `num_ctx`
defaulted to 4096, and prompts ran to ~7k tokens, so the instructions at the top
were cut away before the model saw them. Nothing errors; the answers look
fluent and plausible and simply carry no citations.

Fixes, all three kept: set `num_ctx` explicitly (8192); **put the instructions
after the documents** so recency helps and front-truncation cannot reach them;
cap each context block at 900 chars. Re-piloted: **5/5 citations parsed**,
answers 200-400 chars instead of 600-1400, closed-book still abstains 5/5, and
throughput improved to ~16s/query.

### Decisions taken

* **Citation format**: numeric labels `[1]`..`[k]`, not `resolution_id`s. An id
  here is a full Thai document title; asking a local model to reproduce one
  verbatim would measure copying accuracy rather than grounding. The fabrication
  mode 4a needs is still detectable exactly — a citation to `[11]` when 10 blocks
  were supplied.
* **Context blocks are documents, not chunks** — grouped by `resolution_id`
  (ADR-0002), or one document filling 4 of 10 slots would corrupt the
  citation-precision denominator.
* **k stays 10**; the per-block char cap absorbed the context-window pressure
  instead, so every existing recall@10 number remains directly comparable.
* **Generator: `phi4`**, at `num_ctx=8192` → 10 GB, 100% GPU, no CPU spill.
  `gemma4:e4b` was **not** piloted. It is deferred, not rejected: the planned
  check is 30 queries × 5 arms, asking only whether the **arm ordering** agrees.
  RQ4's claim is about the ordering of arms, so a generator that reorders them
  would be a threat to validity worth reporting; one that preserves the ordering
  strengthens the result.

### Scoring caveat for `rq4_score.py`

Models cite the same label twice (once inline, once in the `อ้างอิง:` line).
**Dedupe before computing citation precision**, or the denominator inflates.

## Generation run complete (2026-07-30)

530 generations (5 arms × 106 queries), `phi4`, `num_ctx=8192`, temperature 0.
**0 errors**, 5,749 s wall clock (~13 s/query with context, 1.8 s closed-book).
Raw answers: `data/rq4/answers/phi4/` (gitignored — regenerable).

### Provisional numbers, before `rq4_score.py`

Computed inline from the raw answers with a throwaway script. **No bootstrap,
no Holm correction** — these say which effects are worth testing, not which are
significant. `rq4_score.py` remains the deliverable.

| arm | citation precision | citation recall | phantom citations | answered when gold present |
|---|---|---|---|---|
| hybrid × qwen3_0.6b | 0.742 | 0.421 | 0 / 275 | 87/102 |
| dense × qwen3_0.6b | 0.670 | 0.410 | 0 / 285 | 84/100 |
| bm25 | 0.625 | 0.407 | 0 / 224 | 69/83 |
| hybrid × m2v | 0.562 | 0.419 | 0 / 194 | 58/79 |
| closed_book | — | — | 0 / 0 | — |

### What RQ4 adds that RQ1–RQ3 could not

1. **Retrieval quality survives the generation stage.** Citation precision
   orders exactly as recall@10 did (0.742 → 0.670 → 0.625 → 0.562). RQ1–RQ2
   could only show that the right documents arrive; this shows the model then
   uses them correctly at a proportionally higher rate. It is the first
   end-to-end confirmation that the retrieval work pays out in answers.

2. **Citation *recall* is flat across every arm (~0.41) — the new result, and
   the most consequential one.** Better retrieval does not make the model cite
   *more* of the gold documents in front of it; it makes a larger share of what
   it cites correct. **The bottleneck is the generator, not the retriever**:
   `phi4` uses roughly 40% of the available gold regardless of context quality.
   That caps the return on further retrieval investment in a way no recall@10
   number can reveal, and it is a recommendation the paper could not otherwise
   make. Worth re-testing with a second generator before leaning on it (see the
   deferred `gemma4:e4b` check) — a flat line across arms is exactly the shape a
   *model-specific* ceiling would take.

3. **Zero fabricated citations in 978 citations.** Not one reference to a label
   outside the supplied context. RAG's most-feared failure mode does not appear
   in this setup — a clean negative result, and the payoff for choosing exactly
   checkable numeric labels over free-text ids.

4. **4b behaves as the corrected design predicted.** Abstention is measurable
   only where the context genuinely lacks the answer, and the strong arms are
   too good to supply such cases: 4 for hybrid, 6 for dense, against 23 for
   BM25, 27 for m2v, and 106 for closed-book. So **4b's claims belong to the
   weak arms and closed-book** — "abstains correctly ~half the time when the
   context lacks the answer (m2v 59%, BM25 52%), and 100% with no context at
   all" is supportable; "hybrid abstains better than dense" is not, on n=4 vs
   n=6. Closed-book abstaining 106/106 is also the run's validity check: the
   prompt controls the behaviour, and the model is not answering from parametric
   knowledge.

### Threat inherited from the retrieval evaluation

Citation precision is judged against the *same* qrels, so it inherits the
pooling-bias threat in `docs/eval-validity-threats.md` §3: a cited document that
is genuinely relevant but unjudged counts as a false positive. The direction
favours the conclusion — semantically-retrieving arms should absorb more of that
penalty than BM25 — so the 0.742 vs 0.625 gap is **conservative**, likely wider
in truth rather than narrower. State it rather than rely on it until the
residual-relevance study reports.

### Correction (same day): the flat citation recall is probably the prompt, not the model

Finding 2 above was written as "the bottleneck is the generator, not the
retriever". The *observation* stands — more retrieval does not raise citation
recall — but the **stated cause is probably wrong**, and the correction changes
what the paper would recommend.

Two measurements point at a fixed citation budget rather than a comprehension
limit:

* citations per answer: **mean 2.65, median 2** (BM25 arm: 2.41 / 2), roughly
  constant regardless of how much gold the context holds;
* citation recall **declines as more gold becomes available** — 0.778 with 2
  gold documents in context, 0.492 with 3, 0.472 with 4, **0.381 with 5+**
  (hybrid arm; BM25 shows the same shape).

A model that could not use the retrieved evidence would score badly at every
level of availability. This model cites ~2 documents and stops.

**That budget is self-inflicted.** Rule 4 of the prompt reads
`ตอบสั้น ๆ ไม่เกิน 3 ประโยค`, while the gold set is dominated by aggregation
queries (mean 9.87 relevant documents: "how many revisions, and what was in
each"). Mean answer length came out at 348 characters — the model obeyed.

Consequences:

* **Finding 1 is unaffected.** Every arm ran under the same budget, so the
  citation-precision ordering remains a fair comparison.
* **Finding 2's recommendation flips** if this is confirmed: from "a stronger
  generator is needed" to "the instruction caps the metric" — a prompt fix, not
  a model upgrade.
* **The `gemma4:e4b` robustness check should not run first.** It would answer
  the wrong question: a second model under the same 3-sentence cap would very
  likely reproduce the same flat line, which would look like confirmation of a
  general ceiling while merely re-measuring the instruction.

**Pending ablation** (not yet run): re-generate the `hybrid_qwen3_0.6b_semantic`
and `bm25_semantic` arms with rule 4 replaced by an instruction to cite every
relevant document, ~212 generations / ~45 min. Recall rises → prompt artifact,
and the RQ4 write-up changes. Recall stays ~0.41 → a real generator ceiling,
and *then* `gemma4:e4b` is the right next test. Keep the current run: the
comparison between the two prompts is itself a reportable result about
instruction sensitivity in citation-grounded generation.
