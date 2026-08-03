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
in truth rather than narrower.

**Residual-relevance study reported 2026-08-03**: closed at a modest ~19-22%
residual rate across arms (not directionally biased — see
`docs/eval-validity-threats.md` §3), i.e. qrels undercount true relevant
documents by ~8-11%, not the ~43-49% an initial (retracted, buggy) judging
pass suggested. The conservative-direction argument above still holds, just
at a smaller, more believable magnitude: the 0.742 vs 0.625 citation-precision
gap is genuinely conservative, but the correction owed to it is modest, not
large.

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

## Ablation run complete (2026-08-03): the ceiling was a prompt artifact

**`tools/eval/rq4_score.py` built** — the deliverable this doc left open.
Parses `[n]` citations (deduped per the scoring caveat above), computes 4a/4b
per (prompt variant, arm), and runs the same paired-bootstrap + Holm machinery
every other significance test in this project uses, in two independent
families: arm ordering (does citation grounding order like recall@10 did?) and
the prompt ablation. Report: `data/results/rq4_score.md`.

**Ablation result — the prompt-artifact hypothesis is confirmed, not the
generator-ceiling one.** `--variant cite_all` (rule 4 replaced with "cite
every relevant document found, unlimited length") was generated for
`hybrid_qwen3_0.6b_semantic` and `bm25_semantic` (212 generations, phi4,
`num_ctx=8192`, temperature 0, 0 errors) and scored against the original
`sentence_cap` prompt, paired per query:

| arm | metric | sentence_cap | cite_all | diff | Holm-adj p | significant |
|---|---|---|---|---|---|---|
| hybrid × qwen3_0.6b | citation recall | 0.2862 | 0.3865 | +0.1003 | <0.0001 | **yes** |
| bm25 | citation recall | 0.2127 | 0.3034 | +0.0907 | <0.0001 | **yes** |
| hybrid × qwen3_0.6b | citation precision | 0.7088 | 0.7139 | −0.0002 | 1.0000 | no |
| bm25 | citation precision | 0.6114 | 0.5962 | +0.0015 | 1.0000 | no |

Both arms gain a significant ~0.09–0.10 absolute jump in citation recall with
**no cost to precision** — the model doesn't just cite more sloppily, it
correctly cites more of the gold set. This settles the question the design
doc's same-day correction raised: **the flat recall was self-inflicted by rule
4, not a phi4 comprehension limit.** Per the pre-registered decision rule, this
means the recommendation is "fix the instruction," not "swap the generator" —
**the deferred `gemma4:e4b` check is correspondingly de-prioritized**: there is
no longer an open "is this a real ceiling" question for it to answer. If
`gemma4:e4b` is tested later for other reasons, it should run under `cite_all`
(or whatever prompt the paper settles on), not `sentence_cap` — testing it
under the prompt now known to suppress recall would just reproduce the
artifact and risk being misread as confirming a ceiling that isn't there.

**Methodology correction, worth flagging explicitly:** the rigorous score
(recall ≈ 0.21–0.29 under `sentence_cap`) is **lower** than this doc's earlier
provisional "~0.41" figure, and the two aren't the same metric. §4a defines
citation recall as `cited ∩ gold / gold` — against the *full* qrels set (mean
9.87 relevant docs/query) — which is what `rq4_score.py` computes, macro-averaged
over all 106 queries including the ones where the model abstained (scored as
recall 0, not excluded). The original throwaway script's bucket breakdown
("0.778 at 2 gold docs → 0.381 at 5+") was denominated by *gold documents
actually present in the k=10 context*, a smaller and more forgiving
denominator that inflates the headline number. The **direction and
significance of every comparison are unaffected** by this correction (both
prompt variants and all five arms were scored the same way), but the ~0.41
figure should not be re-cited — use the table above instead.

**Arm-ordering family (`sentence_cap`, all significance-tested for the first
time) confirms 4c**: citation precision and recall both order
hybrid > dense > bm25 > m2v, with hybrid-vs-m2v and dense/bm25-vs-m2v
significant on precision, and hybrid-vs-m2v and hybrid-vs-bm25 significant on
recall (Holm-adjusted). The top-3 pairwise gaps (hybrid vs dense, dense vs
bm25) don't clear Holm correction — consistent with this project's general
pattern of a tied top cluster and one clear laggard.

**Not yet done, and not required by the ablation's own scope** (docs said
"~212 generations," which is what ran): `dense_qwen3_0.6b_semantic` and
`hybrid_m2v_semantic` have not been regenerated under `cite_all`, so there is
no single-prompt, all-five-arm table yet. That would need ~318 more
generations (~2h) and is only worth doing if the paper wants `cite_all` as the
final reported prompt rather than just as this ablation's proof.

## Extension run complete (2026-08-03): all 5 arms now scored under both prompts

Regenerated `dense_qwen3_0.6b_semantic`, `hybrid_m2v_semantic`, and
`closed_book` under `cite_all` (318 generations, 0 errors), completing the
5-arm × 2-prompt table. `rq4_score.py` was extended (`arm_ordering_family`
helper) to run the 4c arm-ordering significance family under **both** prompt
variants, not just `sentence_cap` — because the interesting question is
whether ordering holds under the prompt that actually raised recall, not just
the one that suppressed it.

**Full recall picture, both prompts:**

| arm | recall: sentence_cap → cite_all | diff | Holm-adj p | significant |
|---|---|---|---|---|
| hybrid × qwen3_0.6b | 0.2862 → 0.3865 | +0.1003 | <0.0001 | **yes** |
| dense × qwen3_0.6b | 0.2354 → 0.3279 | +0.0924 | <0.0001 | **yes** |
| bm25 | 0.2127 → 0.3034 | +0.0907 | <0.0001 | **yes** |
| hybrid × m2v | 0.1603 → 0.1862 | +0.0258 | 0.6570 | no |

**New finding: the prompt fix's recall gain is not universal — it doesn't
reach significance on the weakest arm.** Three of four arms replicate the
significant, precision-neutral recall gain found in the original ablation;
`hybrid_m2v_semantic` does not (95% CI [-0.009, +0.057] crosses zero). This is
consistent with the retrieval-quality story rather than contradicting it: m2v
is the known RRF-failure-case arm (weak dense signal), and it is plausible the
model's citations there are capped less by *instruction* than by *not enough
correct evidence being available in the context to cite in the first place*.
The prompt fix works where the context has more to give.

**Arm ordering under `cite_all` (family 1b) — replicates and sharpens 4c.**
Under `sentence_cap` (family 1a), 6/12 arm-pair tests were significant, all
involving m2v except the hybrid-vs-bm25 pair. Under `cite_all`, **8/12** are
significant: m2v is now significantly worse than *all three* other arms on
*both* precision and recall (dense-vs-m2v recall and bm25-vs-m2v recall newly
clear Holm correction), while hybrid/dense/bm25 remain a mostly-tied top
cluster (hybrid-vs-bm25 still separates on both metrics; hybrid-vs-dense and
dense-vs-bm25 still don't). **Conclusion: retrieval quality's effect on
citation grounding survives the better prompt, and is if anything more
cleanly separable under it** — the earlier prompt's tight citation budget was
partly masking how bad m2v's retrieval really is.

**Closed-book side effect, worth flagging rather than burying:** under
`cite_all`, closed-book picked up 2 hallucinations (0 under `sentence_cap`)
and 5 phantom citations out of 5 total (0 under `sentence_cap`, where it never
cited anything). The instruction to "cite every relevant document" has no
guard for the zero-document case, and in 2/106 queries the model cited a label
anyway despite being told explicitly (rule 3, unchanged between variants) to
abstain when no answer is available. This is a small, real cost of the
`cite_all` wording, not a null: **abstention correctness (4b) degrades
slightly (106/106 → 104/106) as citation recall improves elsewhere.** Worth
naming in the paper if `cite_all` is adopted as the reported prompt — a
tightened version ("cite every relevant document among those you are given; if
none are given or none are relevant, abstain per rule 3") would be worth a
quick follow-up pilot before relying on this wording for a final table.

Updated report: `data/results/rq4_score.md`.
