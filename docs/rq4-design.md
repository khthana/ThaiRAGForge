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
