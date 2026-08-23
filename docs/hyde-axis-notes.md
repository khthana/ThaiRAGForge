# HyDE — notes, a pre-registered prediction, and a measured price

Status: **BUILT AND RUN 2026-08-13 on both query sets. P1 held on 73det as a
significant loss rather than a tie; P2 was REFUTED on thematic — HyDE is
significantly worse there too, on the set that was supposed to be its best case.
See "What actually happened" at the end before reading anything above it as
speculative, and in particular do not cite §"Where it genuinely might work"
without it.** Written 2026-08-07 when the
axis was first assessed; **costed by measurement 2026-08-12**
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
   significantly **hurt** hybrid MRR (**0.7730 → 0.6940**, Holm-adj **0.0240** — re-run
   2026-08-18 against rebuild #4; it read 0.7814 → 0.6778 at 0.0012 before) when it
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

## What actually happened (2026-08-13)

Built as two scripts so the expensive half runs once:
`tools/eval/hyde_generate.py` → `data/results/hyde_documents.json` +
`hyde_generation.md` (285 documents, 40.6 min, one setting), and
`tools/eval/hyde_retrieval_test.py` → `data/results/hyde_retrieval_73det.md`
(36 combos × 106 queries, 20.4 min, 8 self-checks PASS) and
`hyde_retrieval_thematic.md` (36 combos × 179 queries, 32.1 min, 9 PASS).
`temperature=0` is not reproducible on this stack, so **every arm reads one
cache** and the comparison is paired by construction. The two sets ran
sequentially, never concurrently — one GPU.

**P1 held, and as the harder half of its own wording.** The prediction allowed
"ties or degrades"; the result is a significant loss — dense recall@10
**0.5034 → 0.3135, −0.1898**, CI [−0.2446, −0.1345], Holm-adj **0.0000**, all
six family-1 cells worse. Nothing to state as a bound: it is directional.

**P1's `person` clause held, and the mechanism is now evidence rather than
reasoning.** `person` is the worst type (**−0.2798**), falling
0.3604 → **0.0807** — 78% of itself — while `faculty_adjunct_aggregate` is the
only type that misses significance (−0.0699, Holm 0.0560). **It is dilution, not
deletion**: `hyde_generation.md` §2 shows **29 of 30** person documents still
literally contain the queried name. The token is not lost, it is averaged into
~250 tokens of invented context. That is precisely the reasoning in §"Why it is
predicted to FAIL" item 1, and it is the first time this project has been able to
show that mechanism instead of arguing it.

**P3 held, and it was this design's one untested premise.** "Feed the dense arm
only; give BM25 the raw query" was an *assertion* until now. Poisoning BM25 with
the same document costs a further **−0.2735** recall@10 on top of HyDE's own loss
(0.5864 → 0.3128, Holm 0.0000 on all three metrics) — **larger than the entire
dense-arm effect**. Anyone re-proposing HyDE here must keep the split.

**P4 was not triggered, and the macro says the primary combo was not unlucky.**
All 9 embedders lose on dense recall@10. The correlation between baseline
strength and HyDE's effect is **r = −0.887**: the damage is *worst on the
strongest* embedders (`qwen3_0.6b` −0.2206 vs `sct` −0.0613). Not a mercy — a
weak embedder simply had less signal to destroy.

**The null belongs to HyDE, not to one wiring, and the four arms order by how
much of the raw query survives**: `concat` −0.0817 (keeps the question in the
embedded text, and is the only formulation reaching ns on hybrid, −0.0209),
`hyde_q` −0.1405, `hyde_half` −0.1769, `hyde` −0.1898. Damage monotone in
distance from the question is the shape of a real effect, not of a bug.

**The 256-token cap objection is bounded and points the wrong way for HyDE.**
Every document hit the cap, so "a longer one would have done better" is live.
Greedy decoding is a prefix process, so a prefix of a generation is *exactly*
what a smaller cap would have produced — which buys the length slope for free,
with no second generation run and no unpairing. Across **both** sets it is four
cells with **no consistent sign**: `hyde_half` − `hyde` is +0.0130 / −0.0282
(73det dense / hybrid) and −0.0080 / 0.0000 (thematic dense / hybrid), largest
magnitude 0.0282 against a treatment effect of −0.1898. Neither contrast was
pre-registered and halving by characters only approximates a token cap, so the
claim is the modest one — nothing in the data suggests length is the constraint,
and the cap is not an escape hatch for the null.

### Thematic: P2 refuted, but its reasoning survives

**P2 predicted "may improve, most for the weak embedders". Neither half
happened.** Dense recall@10 goes **0.4469 → 0.3733, −0.0736**, CI [−0.1139,
−0.0326], Holm-adj **0.0008**; all six family-1 cells are significantly worse,
all 9 embedders lose, and the weakest of them (`m2v`, baseline 0.1923) takes one
of the largest losses (−0.1042). So the one place these notes said HyDE
"genuinely might work" is a loss too, and §"Where it genuinely might work" must
not be quoted without this paragraph.

**The reasoning behind P2 nonetheless survives its own prediction, and that is
the finding.** The argument was that 73det is an exact-token regime HyDE can only
dilute, while thematic has no name to match (BM25 0.2990 vs 0.4930) so there is
room for elaboration. If that is right the damage should be much smaller here —
and it is: **−0.0736 against −0.1898**, 2.6x smaller on the same arm, with the
same pattern in P3 (**−0.0462** against **−0.2735**, ~6x less, because there is
no exact token left to destroy) and in the correlation (**r = −0.282** against
−0.887, i.e. damage far less concentrated on the strong embedders). **The
mechanism was identified correctly and its magnitude was mis-extrapolated: less
to lose is not something to gain.** Cite it as *HyDE is less harmful where the
lexical signal is weak*, never as *HyDE helps thematic*.

**The only ties are the formulations that keep the question, and they are
bounds.** `concat` is ns on both retrievers here (dense −0.0250, Holm 0.2316;
hybrid −0.0195), as are `hyde_q` and `hyde_half` on hybrid. The best case
anywhere in either table is that dense recall@10 loses no more than **0.0576**
and gains no more than **0.0061** — for 7.85 s/query against a 475.6 ms routed
hybrid query. These arms are exploratory by the frozen decision rule and cannot
promote a null in family 1 to anything.

**No follow-up against the shipped hard router is owed, on either set.** The
known-limitation clause made that conditional on a *positive* unrouted result,
exactly so a negative one could not be kept alive by an untested "but maybe with
routing". **The axis closes here.** Anything reviving it needs a new mechanism —
grounding the generation in retrieved context, which is a different treatment,
not a re-tuning of this one.

**The serving objection is now moot but should still be quoted correctly if the
axis is ever revisited**: 7.85 s/query capped against a 475.6 ms routed hybrid
query, i.e. ~17x the entire query — for a treatment measured at −0.1898.
