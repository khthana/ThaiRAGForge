# Threats to validity of the evaluation

Written 2026-07-30, in response to the question of whether the ground truth is
too small to carry the paper's weight. The short answer is that **size is not
the weak point** — but three other properties of the Gold set are, and one of
them touches a headline finding. This document records the assessment, what was
done about each threat, and what is still owed.

The intent is that the paper carries a *threats to validity* section derived
from this file. A reviewer who finds a limitation the authors already named and
bounded gives credit; a reviewer who discovers it unaided does not.

## 1. Is n=106 too few?

No, and the numbers are on our side.

| | queries | relevance judgments | mean relevant/query |
|---|---|---|---|
| entity-anchored (`gold_query_set_73det.yaml`) | 106 | 1,046 | 9.87 |
| thematic (rewritten 2026-07-30) | 179 | ~179 | ~1 |

For comparison, widely-cited BEIR/TREC test collections run at a similar or
smaller scale: TREC-COVID 50 topics, Touché 49, Signal-1M 97, Robust04 249,
SciFact 300. 106 is unremarkable for a domain-specific collection.

Where this set is *unusually strong* is depth: 9.87 relevant documents per
query, against roughly 1.1 for MS MARCO. Recall@10 is a meaningful quantity
here rather than a near-binary hit/miss, and the entity-anchored queries are
genuinely multi-document ("how many times was this programme revised, and
when"), which is the shape the stakeholder actually asked for.

Depth also buys statistical power that query count alone understates, since
each query's score is an average over ~10 judgments rather than a single
Bernoulli draw. That is visible in §2.

## 2. Statistical power — addressed

**The threat.** This project's central claims are largely *null* results: the
top-4 embedders are fully tied, `semantic` never significantly beats any
chunker, normalization and word-aware segmentation do nothing. The first
question about any null result is whether the effect is absent or merely
invisible at n=106. "We found no difference" is a much weaker sentence than
"we can rule out differences larger than X".

**What was done.** `tools/eval/power_analysis.py` →
`data/results/power_analysis.md`. For every comparison the study makes, it
reports the sd of the paired per-query differences, the minimum detectable
effect (MDE) at 80%/90% power both nominally and at the Holm worst case
(`alpha/m`), the n that would be required to detect the observed effect, and
the 95% bootstrap CI bound — the largest difference the data is consistent
with.

The closed-form MDE assumes a normal test statistic while the tests are
percentile paired bootstraps on discrete, zero-inflated differences, so it is
**verified by simulation** rather than assumed: resampling from the observed
mean-centered differences shifted by the computed MDE and re-running the real
bootstrap gives achieved power 0.78–0.86 against a nominal 0.80 (Monte-Carlo
se ≈ 0.02). The closed form is safe to cite, and is mildly conservative.

**Result — the strongest single outcome of this whole assessment.** Across 180
pairwise comparisons on three metrics, **138 are significant and all 42 ties
have an observed difference below the MDE**. Not one tie in the study is an
artifact of insufficient power. Every tie can therefore be restated as a
bounded equivalence claim rather than an absence of evidence:

- **Chunker ties are the tightest in the study.** `fixed_size` vs `sentence`
  rules out recall@10 differences larger than **0.031**; the whole chunker
  family bounds at 0.031–0.052. The claim "the chunker axis barely matters" is
  not a failure to find a difference — it is a genuinely tight bound, and it is
  the correct way to report the retirement of the old "semantic wins" headline.
- **Embedder ties are looser**, 0.05–0.10, so the "top-4 tied cluster" is a
  real but weaker claim than the chunker one.
- **The weakest tie in the paper is `e5_small` vs `jina_v5` on MRR**,
  consistent with a difference as large as **0.1045**. That pair should be
  reported as *inconclusive*, not as equivalent.

The last bullet is the kind of distinction the report exists to force: before
this analysis, all 42 ties were being cited in the same voice.

## 3. Pooling bias — the sharpest threat, measurement in progress

**The threat.** The Gold qrels were derived by **string containment**
(`tools/corpus_prep/build_gold_candidates.py`): a resolution is relevant to a
programme query when the canonical programme string appears in its *title*, and
to a person query when the person is named in its *body*. That is
approximately what BM25 does at query time.

So a document that is genuinely relevant but phrases the entity differently —
or, as observed in the very first sampled item, whose OCR'd title is truncated
before the programme name — is a **false negative in the qrels**. A retriever
that finds it semantically is penalised for being right, while the lexical
retriever is graded against a key built the way it works.

This is textbook pooling bias, and it points at two of the most-quoted
findings: *"BM25 significantly beats `bge_m3`"* and *"BM25 carries `person`
outright (0.8147)"*. If dense retrieval's top-10 contains relevant documents
the qrels never judged, at a **higher rate** than BM25's does, both are
inflated.

**What was done.** `tools/eval/residual_relevance_sample.py` builds a blinded
human-review sheet: for 29 stratified queries it collects every top-10 hit the
qrels do not judge at all, samples up to 2 per arm (dense / BM25 / hybrid), and
writes 126 candidates with the retrieving arm held in a separate key file so a
judgement cannot drift toward the hypothesis. Each item shows up to 3
already-judged-relevant documents as a calibration reference for what
"relevant" means for that query. `--score` reads the filled sheet back and
reports a per-arm residual-relevance rate with Wilson intervals.

**First signal, before any judging.** The size of the unjudged pool is already
measurable and is *not* asymmetric in the feared direction:

| arm | unjudged documents per query, of top-10 |
|---|---|
| dense (`qwen3_0.6b` × semantic) | 4.28 |
| BM25 (semantic) | 4.86 |
| hybrid | 4.00 |

BM25's unjudged pool is the *largest*, not the smallest. That is mildly
reassuring — a gross bias would have shown as a much larger dense pool — but it
settles nothing on its own, because pool size is not relevance. The verdicts
decide it.

**Decision rule, fixed in advance** (the script writes whichever conclusion the
data supports, so it cannot be chosen after seeing the numbers):

- Wilson intervals **overlap** → the qrels are *incomplete*, which depresses
  all arms' absolute scores but leaves the comparison intact. Report
  incompleteness as a limitation on absolute values; BM25-vs-dense stands.
- Intervals **disjoint** → the qrels are *biased*, not merely incomplete.
  Every BM25-vs-dense claim needs restating, with the per-query estimate giving
  the correction's rough size.

**Status: awaiting human judgement of 126 items** (~1–2 hours). This cannot be
delegated to an LLM: automated relevance judgement is precisely the risk the
Gold set was designed to avoid (`docs/entity-extraction-and-gold-eval-log.md` —
the insight that entity dictionaries make relevance *deterministic* is the
reason this ground truth is defensible at all).

## 4. Circularity in the entity-lookup arms

**The threat.** The qrels for programme and person queries are derived from
`programs.json` / `people.json` — the **same dictionaries** the `entity_tags`
loader and the `entity_lookup` / `entity_boost` retrieval modes use. Evaluating
those modes against these qrels is partly self-fulfilling: `entity_lookup`'s
recall@10 of 0.9291 substantially measures the dictionary agreeing with itself.

**Scope, which is the mitigating fact.** The chunker, embedder, BM25 and hybrid
comparisons — the bulk of the paper — do **not** touch the entity dictionaries
at query time. The circularity is confined to the entity-lookup/entity-boost
arms and does not propagate.

**What is owed.** The project notes internally that 0.9291 "is not the
user-facing number". In the paper this must be an explicit validity paragraph,
not a footnote: state that the entity arms share a source with the ground
truth, that their scores are therefore an upper bound rather than a
measurement, and that they are not comparable to the dense/lexical arms.

## 5. Single annotator, no inter-annotator agreement

**The threat.** Relevance was labelled by one person (the corpus owner). IR
reviewers routinely ask for agreement statistics.

**The defence, which is genuine.** Most labels are **not human judgements at
all** — they are rule-derived and re-derivable from committed code, which is a
stronger reproducibility property than a second annotator would provide. The
rules are auditable (`tools/eval/audit_pipeline_invariants.py`, 25 invariants),
and two real bugs in them were found and fixed by running them against the
whole corpus (the faculty-name substring collision, and the secretary-signature
contamination that inflated one person to 24% of the corpus).

**What is owed.** State the derivation rules in the paper so the labels can be
reproduced rather than trusted. The residual-relevance study in §3 does involve
genuine human judgement and inherits this limitation, which its report says out
loud.

## 6. Query provenance and realism

Queries originate from four real stakeholder questions, were generalised into
entity-anchored templates, and had their wording varied by an LLM. **The LLM
never decided relevance** — only phrasing. This chain should be stated plainly
rather than elided; the honest framing is that the queries are realistic in
*shape* and stakeholder-grounded in origin, not that they are a sampled query
log.

The 179 thematic queries are a separate shape with their own history: they were
initially unanswerable as posed (meeting-scoped but never naming the meeting),
were rewritten 2026-07-30, and carry *opposing* signal on the chunker axis.
They must be reported as a separate query shape and never pooled with the
entity-anchored set — pooling cancels two real effects rather than diluting one.

## 7. External validity

One corpus, one institution, one language, one document genre. Not fixable, and
not worth pretending otherwise: declare it as scope. Thai academic-governance
retrieval is under-served, and being the collection rather than a sample of
collections is a contribution as much as a limitation — provided the paper does
not generalise beyond it.

## Summary of what changes in the paper

| threat | status | action |
|---|---|---|
| sample size | not a real weakness | report depth (1,046 judgments, 9.87/query) alongside count |
| statistical power | **addressed** | cite MDE + CI bounds; restate every tie as a bound, flag `e5_small`/`jina_v5` MRR as inconclusive |
| pooling bias | **measurement built, awaiting judgement** | 126-item blinded review; decision rule pre-registered above |
| circularity | known, scoped | explicit validity paragraph; entity-arm scores as upper bounds |
| single annotator | mitigated by construction | publish the derivation rules |
| query provenance | fine, needs stating | describe the chain; keep thematic separate |
| external validity | inherent | declare as scope |
