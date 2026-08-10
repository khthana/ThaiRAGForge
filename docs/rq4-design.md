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

### The tightened wording, built and piloted (2026-08-07): `cite_all_guarded`

The follow-up pilot suggested above was run. `cite_all` is left **untouched** —
the 530 answers on disk are keyed to that variant name, and editing its wording
in place would silently decouple them from the prompt that produced them — so
the repair is a third variant, `cite_all_guarded` (`tools/eval/rq4_generate.py`,
`_RULE4`), writing to its own `answers/phi4_cite_all_guarded/` directory.

**The diagnosis that shaped it.** Both hallucinations are the same shape: the
context carried `label_map == {}` (closed-book supplies no documents at all) and
the model answered anyway, citing `[1]`/`[2]`/`[5]` — labels that cannot exist.
Rule 3 already forbids exactly this and is *identical* between variants, so the
failure is not a missing rule. It is **position**: rule 4 is the last thing
before the question, and "cite every relevant document" read as an instruction
to produce citations regardless. This is the same recency mechanism that
`build_prompt` already exploits deliberately (context first, instructions last)
— here it worked against us. So the guard does not merely add a constraint, it
adds one *after* rule 4 and says outright that it outranks it:

- **rule 5** — if no documents are supplied at all, answer `ไม่พบข้อมูล`, cite
  `-`, cite no number whatsoever, and *this rule is more important than rule 4*.
- **rule 6** — cite only labels that literally appear in the documents above.
  This targets the second, separate failure: the dense arm emitted 4/359 phantom
  labels (`[6]`–`[9]` against 5 supplied documents), the same over-production
  bounded by a real context rather than an empty one.

**Result on the arm that showed the regression** (closed-book, all 106 queries
regenerated, 489 s). Counted with `rq4_score.py`'s own `parse_citations` /
`is_abstained`, not a separate regex:

| variant | abstained | phantom / total citations |
|---|---|---|
| `sentence_cap` | 106/106 | 0 / 0 |
| `cite_all` | **104/106** | **5 / 5** |
| `cite_all_guarded` | **106/106** | **0 / 0** |

The cost of `cite_all` is fully repaired: closed-book behaves exactly as it did
under the original prompt.

**What that does and does not establish.** It shows the guard removes the
*cost*. It does not show the guard preserves the *benefit* — rules 5-6 are in
the prompt for every arm, not just closed-book, and it is entirely possible for
extra constraints to dampen the "cite everything" push that produced the recall
gain in the first place. That is the question the ablation actually turns on, so
every retrieval arm was regenerated under `cite_all_guarded` too — hybrid and
dense first, bm25 and m2v on 2026-08-08; see the next subsection. That suspicion
was justified: rules 5-6 do reach the other arms, shifting the weak ones toward
abstention (point 4 below). **Do not adopt `cite_all_guarded` as the paper's
prompt on the strength of the abstention table alone.**

`rq4_score.py` takes `--treatment-variant` (default `cite_all`, so the published
run reproduces unchanged) and `--out` (so a non-default variant writes its own
report instead of clobbering `rq4_score.md` — the same principle as giving each
prompt variant its own answers dir). The descriptive/abstention table needs no
flag; it enumerates whatever variants are on disk.

### Does the guard keep the benefit? Yes — and the apparent cost is noise

**All four retrieval arms are now on disk under `cite_all_guarded`** — dense and
hybrid on 2026-08-07, then `bm25_semantic` and `hybrid_m2v_semantic` on
2026-08-08 (212 queries, 4678 s, exit 0, `data/logs/rq4_guarded_2arms_2026_08_08.log`),
so the variant carries the full 5 × 106 = 530 answers the other two do. Report:
`data/results/rq4_score_guarded.md`, regenerated with:

```
PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_score.py \
    --treatment-variant cite_all_guarded --out data/results/rq4_score_guarded.md
```

The three-way table below is that report's **significance family 3** (every
prompt-variant pair × both metrics × every arm carrying all three variants).
With two arms that family held 12 tests; with four it holds **24**. It was first
computed ad hoc in a session, which left these figures unreproducible; family 3
was added to `rq4_score.py` on 2026-08-08 so the table has a generating command,
and the numbers below are now its output.

**Quote the family size with any Holm p from here — on 2026-08-08 this stopped
being a caution and became a live disagreement.** The guarded-vs-baseline pairs
appear in family 2 as well (now 9 tests, up from 5), and for `hybrid` the two
families no longer agree on the verdict: the identical test on identical data,
point estimate **+0.0706** either way, reads **Holm 0.0192 (significant) in
family 2** and **Holm 0.0720 (not significant) in family 3**. Neither is wrong.
Family 2 is the family built to answer "does this prompt beat the baseline", so
that is the one to cite for the ablation claim — but cite it *as* family 2, of
9 tests. An unqualified "Holm p = 0.02" is now demonstrably ambiguous. The 95%
CIs also jitter in the fourth decimal between families because the bootstrap RNG
is consumed in a different order.

**All numbers in this section are the 2026-08-10 re-score, after the 81
truncated cells were regenerated** (`docs/rq4-prompt-truncation.md` §4b). `hybrid`
had no truncated cell and is byte-identical to the pre-repair table; `dense`,
`bm25` and `m2v` all moved, and three verdicts in family 1b flipped `no → yes`.

| arm | variant | recall | precision | phantom / total |
|---|---|---|---|---|
| dense | `sentence_cap` | 0.2261 | 0.6549 | 0 / 274 |
| dense | `cite_all` | 0.3356 | 0.6798 | **4 / 391** |
| dense | `cite_all_guarded` | **0.3460** | 0.6746 | **0 / 375** |
| hybrid | `sentence_cap` | 0.2781 | 0.7016 | 0 / 269 |
| hybrid | `cite_all` | 0.3962 | 0.7268 | 0 / 421 |
| hybrid | `cite_all_guarded` | 0.3487 | 0.6900 | 0 / 369 |
| bm25 | `sentence_cap` | 0.2265 | 0.6607 | 0 / 229 |
| bm25 | `cite_all` | 0.2961 | 0.6104 | 0 / 330 |
| bm25 | `cite_all_guarded` | 0.2798 | 0.6217 | 0 / 303 |
| m2v | `sentence_cap` | 0.1820 | 0.5508 | 0 / 209 |
| m2v | `cite_all` | 0.2038 | 0.5278 | 0 / 287 |
| m2v | `cite_all_guarded` | 0.1943 | 0.4817 | 0 / 280 |

Recall comparisons, family 3 (24 tests):

| comparison | n | diff (recall) | 95% CI | Holm p | sig |
|---|---|---|---|---|---|
| dense: `cite_all` vs base | 106 | +0.1095 | [+0.0643, +0.1586] | 0.0000 | **yes** |
| dense: `guarded` vs base | 106 | **+0.1198** | [+0.0673, +0.1769] | 0.0000 | **yes** |
| dense: `guarded` vs `cite_all` | 106 | +0.0104 | [−0.0179, +0.0418] | 1.0000 | no |
| hybrid: `cite_all` vs base | 106 | +0.1181 | [+0.0734, +0.1643] | 0.0000 | **yes** |
| hybrid: `guarded` vs base | 106 | **+0.0706** | [+0.0232, +0.1192] | 0.0720 | no *(0.0192 in family 2)* |
| hybrid: `guarded` vs `cite_all` | 106 | −0.0475 | [−0.0884, −0.0096] | 0.2376 | no |
| bm25: `cite_all` vs base | 106 | +0.0696 | [+0.0368, +0.1064] | 0.0000 | **yes** |
| bm25: `guarded` vs base | 106 | +0.0533 | [+0.0116, +0.0951] | 0.1786 | no *(0.0938 in family 2)* |
| bm25: `guarded` vs `cite_all` | 106 | −0.0163 | [−0.0539, +0.0183] | 1.0000 | no |
| m2v: `cite_all` vs base | 106 | +0.0217 | [−0.0081, +0.0484] | 1.0000 | no |
| m2v: `guarded` vs base | 106 | +0.0122 | [−0.0120, +0.0370] | 1.0000 | no |
| m2v: `guarded` vs `cite_all` | 106 | −0.0095 | [−0.0427, +0.0251] | 1.0000 | no |

No precision comparison is significant — all 12 sit at Holm p = 1.0000, the
*smallest* raw p among them being 0.1126 (bm25, `sentence_cap` vs
`cite_all_guarded`).

**Four things this establishes.**

1. **Both guards work, and each is confirmed on the failure it was written for.**
   Rule 5: closed-book abstention 104/106 → **106/106**, phantom 5/5 → **0/0**.
   Rule 6: dense phantom **4/391 → 0/375** — and note no other arm had phantoms
   under any variant, so dense was the only arm that could test rule 6 at all.
   Adding bm25 and m2v introduced no new phantoms (0/303, 0/280).
2. **The benefit survives, and the arm it survives on is dense.** `cite_all_guarded`
   beats the `sentence_cap` baseline by **+0.1198 on dense** (Holm 0.0000 in both
   families — the one result here that no family-size choice can touch) and by
   **+0.0706 on hybrid** (family 2's 0.0192; see the family-size note above).
   The ablation's headline — the flat ~0.41 recall was a prompt artifact, not a
   generator ceiling — does not depend on the unguarded wording. On bm25 the
   guarded gain (+0.0533) does not reach significance in either family, where
   the unguarded `cite_all` gain (+0.0696) did; on m2v neither variant moves,
   the same null the unguarded run found.
3. **The apparent cost relative to unguarded `cite_all` is not a finding.**
   No arm is significant, and more tellingly **the point estimates do not agree
   on a direction** (dense +0.0104 against hybrid −0.0475, bm25 −0.0163, m2v
   −0.0095). A real constraint-induced dampening would push the same way on all
   four. This is what the measured generator noise floor predicts: at temperature
   0 this pipeline reproduces the citation set only 14/24 under `cite_all`
   (`tools/eval/rq4_determinism_check.py`), so differences of this size are
   inside the noise. Stated as bounds rather than nulls: on hybrid the interval
   rules out the guard being *better* than `cite_all` and is consistent with a
   loss of ~0.01-0.09; on dense it rules out a loss greater than ~0.018; on bm25
   and m2v the intervals straddle zero.
4. **The guard makes the model more cautious on every arm, not only closed-book
   — and that has a price.** Rule 5 was written for the zero-document case, but
   it is in the prompt for all of them, and the 4b table shows it changing the
   weak arms' behaviour: m2v correctly-abstained 13 → **16** and hallucinations
   16 → **13**, but "missed" (gold present, abstained anyway) rose 10 → **18**;
   bm25 hallucinations 11 → **10** with missed 14 → **11**. On the strong arms
   the shift is small (hybrid missed 8 → 9, dense 14 → 9). This is a genuine
   trade the paper should state, not a free repair: the guard buys grounding
   discipline partly with recall on arms whose context is poor. (The m2v figures
   were milder after the truncation repair — they read 13 → 19 / 16 → 10 before
   it — so quote them from this table, not from a pre-2026-08-10 copy.)

**Arm ordering under the guard (family 1b) — the deliverable that needed all
four arms.** Direction is unchanged, `hybrid (0.3487) > dense (0.3460) >
bm25 (0.2798) > m2v (0.1943)` on recall, and m2v is still significantly worst.
But the *number of separated pairs* depends on the prompt:

| prompt variant | significant pairs, family 1 (12 tests) |
|---|---|
| `sentence_cap` | 2 / 12 |
| `cite_all` | **8 / 12** |
| `cite_all_guarded` | **6 / 12** |

**This table was the section's biggest casualty of the truncation repair
(2026-08-10) and the conclusion it used to carry is withdrawn.** It read
2/12 → 9/12 → 3/12, and the guarded column was explained as the guard
"compressing the spread". With the truncated cells regenerated it is 2 → 8 → 6,
and **the difference is not compression, it is a different set of separated
pairs**: under `cite_all_guarded` all **six** m2v pairs separate (they did not
before — the repair pushed dense and bm25 up while m2v's precision fell 0.5278 →
0.4817), while `hybrid vs bm25` separates under `cite_all` (recall Holm 0.0032,
precision 0.0180) and not under the guard (0.1056 / 0.2430). So the honest
reading is: **`cite_all` is the only variant that separates the two strong arms
from each other; the guard separates the weak arm from everything.** Read all of
it against the noise floor before treating 8 vs 6 as a precise quantity.

**Recommendation, unchanged in substance: adopt `cite_all_guarded` as the paper's
reported prompt.** The guard keeps a significant recall gain over the baseline
(unambiguously on dense, and on hybrid in the family built for that question) and
removes both of `cite_all`'s measured costs outright — including the 106/106
closed-book abstention, which is the whole experiment's validity check and not
merely two lost points. What changed on 2026-08-10 is the caveat attached to it:
the "the guard costs you the 4c sharpening" clause is **no longer supportable**
at 6/12, and what should be reported alongside instead is *which* pairs each
wording separates. Both variants' answers are on disk, so both tables are
reproducible.

## Refresh against `chunker_compare_full` rebuild #3 (2026-08-07)

Everything above was built and scored 2026-08-03, against retrieval results that
predate the rebuild that finished 2026-08-05T07:56. Per
`feedback_refresh_all_retrieval_paths_after_rebuild`, RQ4 was the last retrieval
path still un-refreshed after that rebuild (the main BM25/hybrid chain closed
08-06, the thematic arm 08-07). It is now closed too — and unlike those two, it
did **not** come back "0 flips".

### What was re-run

Contexts were rebuilt for all 4 retrieval arms, then only the queries whose
context actually changed were regenerated; the rest were left frozen so the
before/after comparison stays paired. **362 of 530 (query, arm) cells changed
context and were regenerated; 168 were frozen** (`closed_book` has no context, so
all 106 froze by construction). Both prompt variants, `phi4` local-only as before.
4h05m wall clock, exit 0, 0 generation errors. Log:
`data/logs/rq4_regen_2026_08_07.log`. Pre-refresh report preserved at
`data/results/_pre_2026_08_07_rq4_refresh/rq4_score.md.2026-08-03`.

### The reason this refresh cannot be read like the others

`rq4_generate.py`'s docstring asserted "**Temperature 0.** One pass, no sampling
variance to average over." **That is false for `phi4` via Ollama**, and it was
believed partly on this project's own precedent — re-OCR at temperature 0.0
reproduced its input byte-for-byte (`docs/llm-ocr-scan-log.md`). Greedy decoding
is deterministic in *exact* arithmetic, but GPU reductions are not associative, so
two near-tied logits can swap between runs and the continuation diverges. Measured
before reading any diff (`tools/eval/rq4_determinism_check.py`, 24 byte-identical
prompts per variant):

| variant | identical text | identical citation set | identical abstention |
|---|---|---|---|
| `sentence_cap` | 19/24 (79%) | 21/24 (88%) | 24/24 (100%) |
| `cite_all` | 6/24 (25%) | **14/24 (58%)** | 23/24 (96%) |

The scorer reads only the `[n]` labels and the abstention token, and those are far
more stable than the prose (58-88% vs 25-79%) — so a strict text diff overstates
the problem. But under `cite_all`, **42% of queries move their citation set with
zero data change.** Any movement in this refresh smaller than that floor is not
attributable to the rebuild. The docstring has been corrected.

### Result: 5 verdict flips of 33, and they are all borderline

`tools/eval/diff_significance_reports.py` against the 08-03 baseline: 53 keyed
rows both sides, no rows added or dropped, **5 verdict flips of the 33 rows
carrying a verdict**, 99 numeric moves ≥ 0.02.

| family | comparison | was | now |
|---|---|---|---|
| 1a `sentence_cap` | `bm25`[precision] vs `m2v`[precision] | yes | no |
| 1a `sentence_cap` | `dense`[precision] vs `m2v`[precision] | yes | no |
| 1a `sentence_cap` | `hybrid`[precision] vs `bm25`[precision] | yes | no |
| 1a `sentence_cap` | `hybrid`[recall] vs `bm25`[recall] | yes | no |
| 1b `cite_all` | `hybrid`[recall] vs `dense`[recall] | no | **yes** |

Three things make these weak evidence of a real change. **All four losses are in
family 1a** and every one was already in the modest-evidence band (Holm-adj
0.0140 / 0.0216 / 0.0476 / 0.0808) — nothing at p < 0.001 moved in either
direction. The **single biggest numeric driver is one arm's mean precision**:
`phi4 / hybrid_m2v` went 0.4945 → 0.5575, which narrows three m2v comparisons at
once, and m2v is also the arm with the most context churn. And the flips sit
inside the noise floor above. **Report these four as inconclusive, not as
reversed findings** — the honest statement is that family 1a's arm ordering is
not robustly separable under the original prompt, which is a weaker claim than
either the old table or the new one makes on its own.

Family 1a is now 2/12 significant and family 1b 9/12 (was 6/12 and 8/12). The
*direction* of finding 4c — arm ordering sharpens under `cite_all` — therefore
holds and in fact strengthens; the counts change.

> **Superseded 2026-08-10.** Every figure in this dated section is the state
> *after* the 08-07 refresh and *before* the prompt-truncation repair. The 81
> truncated cells were regenerated on 08-10 (`docs/rq4-prompt-truncation.md`
> §4b) and this family-1b count moved 9/12 → **8/12**, family 1a staying 2/12.
> The section is kept as the record of what that refresh found; cite the current
> numbers from the guarded-section tables above or from
> `data/results/rq4_score.md`.

### What survives untouched

- **The prompt ablation, entirely.** Recall rises significantly under `cite_all`
  for hybrid (+0.1181), dense (+0.1005) and bm25 (+0.0734), all Holm-adj 0.0000,
  and **not** for m2v (+0.0217, Holm-adj 0.8052). No significant precision cost
  anywhere (every precision cell Holm-adj ≥ 0.8052). This is RQ4's headline and
  it did not move at all.
- **m2v significantly worst on both precision and recall under `cite_all`** — all
  six comparisons against it still significant.
- **Abstention 106/106 → 104/106** under `cite_all`, exactly as reported.
- **0 fabricated citations under the original prompt**: all four `sentence_cap`
  arms are still 0 phantoms across 954 citations.

### Correction 1: the citation-precision ordering is prompt-dependent

Finding 4a was stated as "citation precision orders exactly as recall@10 did".
Under `cite_all` it still does: hybrid 0.7268 > dense 0.6629 > bm25 0.5968 > m2v
0.5203. **Under `sentence_cap` it no longer does** — bm25 0.6463 now edges dense
0.6413. The inversion is 0.005 and Holm-adj p = 1.0000, i.e. a flat tie, and
the same pair is not significant under `cite_all` either (Holm-adj 0.2136). So
the defensible claim is narrower than the old one in a way that was always true
and merely happened to be masked: **hybrid > {dense, bm25} > m2v, with dense and
bm25 tied and never separated in either variant.** Don't restate 4a as a strict
4-way ordering.

### Correction 2: fabricated citations are not at zero under `cite_all`

`phi4_cite_all / dense` went **0/370 → 4/359 phantom citations**. All four are one
query (`q045`) citing `[6][7][8][9]` when only 5 documents were supplied. So the
mechanism is benign — out-of-range labels in a single answer, not content
attributed to a real-but-wrong document — and "0 fabricated out of 978" remains
true of the original prompt. But `cite_all` now shows fabrication in **two** arms
(dense and closed-book), not closed-book alone, which strengthens the existing
recommendation to tighten that wording before adopting it for the paper's final
table.

**This one was nearly missed, and that is its own finding.**
`diff_significance_reports.py` compared only cells it could parse as numbers, and
the phantom column is formatted `count/total` — so `0/370 → 4/359` matched neither
the numeric branch nor the verdict branch and was **silently skipped**. It was
found by comparing that column by hand. The differ now reports every non-numeric
cell change and exits 1 on one (confidence intervals excluded, being numeric in
substance). Same shape as every silent-corruption bug this project has hit: not a
crash, just a number nobody looked at.

## Pre-registration (2026-08-10): the two entity arms as a decisive upper bound

Written **before** the run, so the reading rule is not chosen after seeing the number.
This is task #1, and it exists to gate the 5-7 days of work behind relation-graph
edges B (`person → responsible_for → program`) and C (`person → replaces → person`).

**The argument being tested.** `entity_lookup` reads `programs.json` / `people.json` /
`courses.json` and returns *every* document tagged with the query's entity — exhaustive
and unranked. It is structurally advantaged **twice**: (a) exhaustive, so it has no
ranking to lose evidence to, and (b) **circular**, because its qrels were derived from
the same dictionaries it retrieves with (see `docs/eval-validity-threats.md` §3). A
richer graph built on those same dictionaries can, at best, route evidence the
dictionaries already contain. **So if the dictionary-driven arm cannot lift end-to-end
citation recall above the shipped hybrid arm, edges B and C cannot either**, and the
graph axis closes on measurement rather than on taste.

**Arms.** `entity_lookup_semantic` and `entity_boost_semantic`, contexts already built
(106 each, `data/rq4/contexts/`). Baseline for comparison: the existing `hybrid` arm.

**Prompt variant: `cite_all`, pre-registered.** Three reasons, and the choice is
deliberately the one most favourable to the entity arms, because this is an *upper*
bound: (1) `cite_all` is the variant under which the arm-ordering result separates most
(family 1 splits 9/12, vs 3/12 under `cite_all_guarded` — the counts as they stood when
this was pre-registered; after the 08-10 truncation repair they read 8/12 vs 6/12, which
weakens the margin but not the choice); (2) it instructs the model to
cite every relevant document, which is exactly what an exhaustive retriever needs to
convert its recall into citation recall; (3) the `hybrid` baseline under `cite_all`
is **uncontaminated by the truncation defect** — see below. `cite_all_guarded` remains
the paper's recommended prompt; it is not what this bound is measured under.

**`--num-ctx 16384`, and why the comparison is clean anyway.** Every published RQ4
answer was generated at 8192 (`docs/rq4-prompt-truncation.md`). The entity arms would
have been ~45-50% truncated there, so they *must* run at 16384. The comparison is still
paired and valid because **`hybrid` is 0/106 truncated in all three variants** (longest
prompt 7,999 tokens, 193 short of the line) and the measured rule is that a prompt which
*fits* is fed whole — so the `hybrid` answers on disk are byte-for-byte the experiment a
16384 run would have produced. **`dense`, `bm25` and `m2v` are NOT clean** (16 / 5 / 7
of 106 truncated) and any entity-vs-those comparison must be reported as confounded
until those 80 cells are regenerated.

**Primary comparison and decision rule, fixed now.**
- Primary metric: **citation recall**, `entity_lookup` vs `hybrid`, under `cite_all`.
- Secondary: citation precision, and the same pair for `entity_boost`.
- Holm within the family `rq4_score.py` already defines; **quote the family size**.
- **If `entity_lookup` does not significantly beat `hybrid` on citation recall** → the
  upper bound fails to clear, and edges B/C are **not built**. Report it as a bound
  ("rules out a graph gain larger than X"), never as "no difference"
  ([[feedback_report_ties_as_bounds]]).
- **If it does** → the axis has measured headroom; the size of the margin is the budget
  edges B/C have to earn, and it is still an *upper* bound, not a forecast, because of
  the circularity.
- Either way the `entity_lookup` number is **recall at k=1000, not recall@10** — the arm
  is exhaustive and unranked, and that distinction is why it must never be set beside the
  dense/lexical recall@10 columns.

**Known noise floor.** phi4 at temperature 0 reproduces only **14/24** identical citation
sets on byte-identical prompts under `cite_all`
([[feedback_temperature_zero_is_not_reproducible]]). A margin near that floor is not a
result; this is why the run is *scored*, not read.

## Result (2026-08-10): the entity arms, and the decision on edges B/C

Run as pre-registered above: `cite_all`, `--num-ctx 16384`, 212 answers (14,434 s),
0 generation errors, 0 empty answers, **0 truncated**. Report:
`data/results/rq4_score_entity.md` (family 1b, **m=6**). The run was necessary rather
than merely tidy: the recorded `prompt_eval_count` peaks at **13,636** (`entity_lookup`)
and **14,515** (`entity_boost`), so at the old 8192 default roughly half of these answers
would have been generated from evidence-stripped prompts.

| arm (all `cite_all`) | precision | n w/ citations | recall | gold density in ctx | missed |
|---|---|---|---|---|---|
| `hybrid_qwen3_0.6b_semantic` | 0.7268 | 101 | 0.3962 | 0.5352 | 8 |
| `entity_lookup_semantic` | 0.5918 | **65** | **0.1431** | 0.6448 | **40** |
| `entity_boost_semantic` | **0.8048** | 104 | **0.4379** | 0.7814 | 5 |

(`gold density` and `missed` from `tools/eval/rq4_entity_arm_diagnosis.py` →
`data/results/rq4_entity_arm_diagnosis.md`; `missed` = gold was in the context and
the model abstained anyway.)

**The pre-registered primary comparison failed, and it failed in the opposite
direction.** `entity_lookup` vs `hybrid` on citation recall is **−0.2531**, CI
[−0.3223, −0.1815], **Holm-adj 0.0000** — not "fails to beat", but decisively worse;
precision too (−0.1583, Holm 0.0150). By the letter of the pre-registration that
settles it: **edges B and C are not built.**

**But the stated reason for the inference does not survive, and that matters more than
the verdict.** The argument was that `entity_lookup` is structurally advantaged, so its
failure bounds anything built on the same dictionaries. What actually happened is a
**ranking failure, not a knowledge failure**, and the 4b table says so exactly:
`entity_lookup`'s contexts are **not** evidence-poor — they hold a *higher* share of
gold than hybrid's (**0.6448 vs 0.5352**) — yet **40 of its 97 gold-bearing contexts
produced an abstention** (hybrid 8, `entity_boost` 5).

**That high density is the circularity itself, and the 40 abstentions are what it
costs.** These qrels call a document relevant when it *contains the entity*, and
`entity_lookup` retrieves exactly the documents containing the entity — so an
almost-pure-gold context is true by construction rather than a sign of a good context.
The generator was handed ~8 documents that all name the entity and, on 40 queries,
judged that none of them answered the question. **That is an independent judge saying
string containment over-counts relevance for this query shape** — the threat
`docs/eval-validity-threats.md` §3 argues for the entity arms, here visible in a
measurement instead.

**And what separates the two entity arms is ranking, which is worth more here than the
dictionaries are.** Both draw on the same dictionaries and both fill the budget with
entity-bearing documents (density 0.6448 vs 0.7814); `entity_boost` orders them by
hybrid relevance, so what it supplies also answers the question — citation recall
**0.4379 vs 0.1431**, missed 5 vs 40. That gap dwarfs `entity_boost`'s entire
non-significant margin over shipped hybrid. **An exhaustive retriever's advantage does
not survive a fixed context budget it cannot rank into.** So `entity_lookup` bounds
*unranked dictionary retrieval under a k≈10 budget*; it does not bound what the
dictionaries know.

**So cite `entity_boost` as the arm that actually answers the gating question**: same
dictionaries, but ranked (hybrid ordering with an entity boost). It is the **numerically
best arm in the whole RQ4 table on both metrics** — recall 0.4379 vs hybrid's 0.3962,
precision 0.8048 vs 0.7268 — and **neither clears Holm** (0.1652 and 0.1192, m=6).
**State it as a bound**: with the dictionaries ranked properly, and scored on qrels
partly derived from those same dictionaries, the end-to-end gain over shipped hybrid is
**at most +0.1001 citation recall / +0.1337 precision**, and the point estimate
(+0.0417) sits inside the generator's own measured noise floor
([[feedback_temperature_zero_is_not_reproducible]]: 14/24 identical citation sets at
temperature 0 under this variant).

**Decision: edges B and C are not built.** Both arms point the same way, the honest one
(`entity_boost`) bounds the payoff below the noise floor, and the bound is **optimistic**
because of the circularity — a name the dictionary lacks is invisible to the retriever
*and* absent from the qrels at once (`docs/eval-validity-threats.md` §3). A graph over
those dictionaries would be paying 5-7 days for a quantity measured at "no more than
+0.10, probably ~0.04, on a metric that flatters it".

**One descriptive finding worth keeping, not pre-registered**: `entity_boost` has the
highest citation precision of any arm ever measured here (0.8048, vs the previous best
0.7268), and the lowest miss count (5). If the entity signal is ever used for anything,
use it as a **boost on a ranked retriever**, never as a retriever.
