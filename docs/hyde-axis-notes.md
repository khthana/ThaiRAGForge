# HyDE — notes, a pre-registered prediction, and a measured price

Status: **not started, not committed to.** Written 2026-08-07 when the axis was
first assessed; **costed by measurement 2026-08-12**
(`tools/eval/probe_hyde_generation_cost.py` → `data/results/hyde_generation_cost.md`).
Same shape as `docs/colbert-late-interaction-notes.md`: the option is recorded
with its reasoning while the findings that motivate it are fresh, so a later
decision is made against evidence rather than against a fading memory of why it
looked attractive.

## What it is

HyDE (Hypothetical Document Embeddings): instead of embedding the *question*, ask
an LLM to write the passage that **would** answer it, embed that, and retrieve
with it. The claim is that a hypothetical answer lives closer in embedding space
to a real answer than a question does — questions and answers are different
registers, and a bi-encoder has to bridge that gap.

Here the hypothetical document would be a short paragraph in the register of the
council minutes. The prompt used for pricing (deliberately unengineered — it
prices the axis, it does not tune it) is in the probe script.

## Why it is predicted to FAIL on the main query set

Not a generic scepticism — three of this project's own measurements point the
same way.

1. **73det is the opposite regime to the one HyDE repairs.** HyDE's mechanism is
   *semantic elaboration* for underspecified queries. On this set **BM25 alone
   scores 0.8147 on `person`**, beating every dense embedder (best `bge_m3`
   0.5735): the discriminative signal is an exact token, not a semantic
   neighbourhood. The query already contains the string that matters, and HyDE
   can only dilute it in generated filler.

2. **The generator does not know this corpus, and here that is the sharpest
   argument.** `phi4` has never seen these people, programmes, or meeting
   numbers. It will emit fluent Thai administrative prose containing
   **fabricated names, programmes and numbers**, and that is what gets embedded.
   Contrast RQ4's **0 fabricated citations** under the original prompt: that zero
   holds because generation is *constrained by supplied context*. **HyDE
   generates before retrieval, with nothing to ground it**, so it has no such
   protection. In a system where entity matching is the whole game, this is the
   worst possible noise to inject.

3. **This project is 0/1 on the family.** The cross-encoder reranker was also a
   literature-backed query-time model insertion, also plausible, and
   significantly **hurt** hybrid MRR (0.7814 → 0.6778, Holm-adj p=0.0012) when it
   replaced the ranking. It took a fine-tune on hybrid-fused candidates before
   anything in that family beat the shipped router — and even then a **free**
   lexical control landed 0.0043 behind it
   (`data/results/reranker_trained_test.md`).

**Where it genuinely might work**: thematic queries, where BM25 collapses to
**0.2990** (vs 0.4930 entity-anchored) because there is no name to match. Same
shape as the project's RRF rule — help arrives when the arm is weak.

## Design consequence, if it is ever built

**Feed HyDE to the dense arm only; give BM25 the raw query.** Hallucinated tokens
entering a lexical matcher should be actively harmful, and mixing the two would
confound "HyDE helps dense" with "HyDE poisons BM25". Seam is
`src/rag_lab/query_service.py`, where `retrieve(query, index, embedder, ...)`
takes the raw string and embeds it internally — a registry entry, no runner edit
(ADR-0001, Open/Closed).

## A pre-registered prediction, so this is falsifiable

1. On **73det**, HyDE ties or degrades, and is **worst on `person`**.
2. On **thematic**, it may improve, most for the weak embedders.
3. An improvement on **73det** would falsify the reasoning above and is the more
   interesting outcome — record it as such if it happens.

## The price, measured rather than quoted

**The figure this doc used to carry was wrong, and how it was wrong is worth
keeping.** It said generation costs **15.6 s/query**, taken from the RQ4 run's
own log. An RQ4 prompt carries ~8,000 tokens of retrieved context; a HyDE prompt
carries only the question (~300 tokens). A timing measured on one prompt shape
does not transfer to another, however identical the model and the machine
(`feedback_state_the_input_size_with_any_timing`). So it was measured:

| | uncapped | `num_predict=256` |
|---|---|---|
| per query (warm) | **17.57 s** | **7.85 s** |
| 73det (106 queries) | 31 min | 14 min |
| thematic (179 queries) | 52 min | 23 min |

throughput ≈ **33 tok/s**, flat across every cell.

**The cost is output-bound, not prompt-bound**, and that is the finding that
changes the price. The prompt is ~300 tokens but the model writes 564–843 in
reply, despite the prompt saying "ไม่เกิน 5 ประโยค" — **a length instruction
written in natural language enforces nothing**. Capping at 256 tokens roughly
halves the cost and is *closer* to HyDE's intent, not a compromise of it: the
hypothetical document is meant to be a short passage, not an essay.

**One generation serves every embedder.** HyDE is a pure query transform with no
index dependency, so the paragraphs cache to JSON once and any number of combos
sweep against them. **No index rebuild at all.** What remains is a normal
retrieval pass (~20–40 min per query set) plus the implementation, which is the
larger block — the GPU time is not the expensive part of this axis.

**`temperature=0` is not reproducible here either.** The probe's first three runs
reproduced output-token counts exactly on all four queries; the fourth disagreed
on all four. Read the table as one run, not a constant. The reproducibility
sentence in the report is *derived* from `data/results/hyde_generation_cost_runs.json`
rather than typed, which is the only reason the report did not end up asserting
determinism on the strength of those three runs.

## The objection that has grown, not shrunk

**Serving latency is the real problem, and it got worse for HyDE as the system
got faster.** When these notes were written a hybrid query cost ~2.1–2.7 s, of
which ~1.9–2.0 s was a fixable `BM25Okapi`-rebuild-per-query defect, so adding
15.6 s read as ~7x. Both halves have since moved: the scorer is memoised on the
`Index` and `fetch_depth=200` ships at the query-time layer, so a **routed hybrid
query is now 475.6 ms p50** — while the honest HyDE figure is 7.85 s capped, not
15.6 s. That is **~17x** the entire query, on the same GPU that serves the
embedder. As an *offline eval* axis the price is fine (14 min a set); as a
*shipped* feature it is not, and the two should be argued separately.

## Priority

Every item this axis was originally ranked below has since been done — relation
graph edges A/A′ built, the `BM25Okapi` cache fixed, the fusion weight swept,
`weighted` × `fetch_depth` measured. So the ranking now rests on its own merits,
and they are mixed: HyDE is **cheap** (no rebuild, 14 min a set) and
**falsifiable** (§ above), which is exactly what a paper wants from a negative
result — but the prediction is negative, and the measured headroom on this corpus
sits elsewhere (the routed P=50 oracle delivers **0.8331** against the shipped
router's 0.6831, and a trained cross-encoder captures 44% of that gap). Run it
for the pre-registered prediction and for the thematic arm, not in the
expectation of a win.
