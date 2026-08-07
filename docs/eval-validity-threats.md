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
  consistent with a difference as large as **0.1029**. That pair should be
  reported as *inconclusive*, not as equivalent.

The last bullet is the kind of distinction the report exists to force: before
this analysis, all 42 ties were being cited in the same voice.

**Currency.** This analysis is recomputed from persisted results, so it goes
stale on every index rebuild like everything else. **Re-run 2026-08-05 against
`chunker_compare_full` rebuild #3** (`data/results/power_analysis.md`): the
headline is unchanged — still **138 significant / 42 ruled out / 0
underpowered** out of 180 — and the individual bounds moved only in the third
decimal (the `e5_small`/`jina_v5` MRR bound above went 0.1045 → 0.1029, which is
why this section quotes the new figure). The conclusion "not one tie in the
study is a power artifact" is therefore verified against the current indices,
not inherited from the pre-rebuild run.

## 3. Pooling bias — CLOSED 2026-08-03: qrels are modestly incomplete, and not directionally biased

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

**First-pass manual verdicts (2026-08-03) were retracted the same day — a
review-app design bug, not a real finding.** All 126 items were first judged
by the corpus owner via the sheet's companion Streamlit app
(`residual_relevance_review_app.py`), reading each candidate and copy-pasting
the entity name into the browser's native find-in-page (Ctrl+F) to check for
a match. That first pass came back at a residual rate of ~0.98-1.000 for all
three arms (125/126 "relevant") — implausibly high, and traced to a genuine
bug: the same page also displays each item's "already judged relevant"
calibration references **in full text** (added earlier that day specifically
so a `person`-type calibration reference wasn't useless title-only text — see
[[project_residual_relevance_review_app]]). A calibration reference is
relevant *by construction* (it comes straight from the qrels), so a
page-wide Ctrl+F almost always finds the entity **somewhere on the page** —
just not necessarily in the candidate being judged. Verified mechanically:
of the 100 items marked "relevant" whose candidate text does *not* contain
the entity, **100/100** have it in a calibration reference on the same page.
Zero counterexamples. The annotator's process was sound; the page design
was not — it let a page-wide search silently answer a different question
than the one being asked.

**Corrected methodology, confirmed by the corpus owner's own domain
judgement**: for these entity-anchored query shapes, a document cannot be
relevant unless the named person/course/faculty/program is literally
present in it — there is no query in this set where the answer is
plausibly present without the name appearing (the annotator's own
assessment, checked against every sampled item). That makes literal
entity-presence a valid, *automatable* relevance criterion here — unlike
the general case, where "not literally present" could still mean "relevant
via different phrasing" (the pooling-bias mechanism this study exists to
catch, and which does NOT reduce to a phrasing question for this query
shape). `tools/eval/residual_relevance_decompose.py` reapplies the *exact
same* per-entity-type rule `build_gold_candidates.py` used to construct the
qrels themselves (title-substring for programme, secretarial-mention-aware
exact given+surname regex for person, filing-title-gated dictionary tag for
faculty, canonical-name substring for course) against each candidate's full
document text (not just the shown chunk), corpus-wide. Original manual
verdicts are preserved at
`data/results/residual_relevance/review_sheet.manual_backup_2026_08_03.yaml`
for the record; the sheet's live verdicts were overwritten with this
automated, rule-based check (100 of 126 flipped from `y` to `n`).

**Corrected verdicts** (`data/results/residual_relevance.md`):

| arm | judged | relevant | not | residual rate | 95% CI (Wilson) | unjudged/query | est. missed relevant/query |
|---|---|---|---|---|---|---|---|
| dense (`qwen3_0.6b` × semantic) | 47 | 9 | 38 | 0.191 | [0.104, 0.325] | 4.28 | 0.82 |
| BM25 (semantic) | 49 | 11 | 38 | 0.224 | [0.130, 0.359] | 4.86 | 1.09 |
| hybrid | 49 | 11 | 38 | 0.224 | [0.130, 0.359] | 4.00 | 0.90 |

The three Wilson intervals still overlap heavily, so the **qualitative
verdict is unchanged**: incompleteness, not directional bias — every
BM25-vs-dense comparison in this project stands as a relative ranking. But
the *magnitude* is now a modest, believable one instead of an implausible
near-total failure: ~0.8-1.1 additional genuinely-relevant documents per
query beyond the qrels' own mean of 9.87/query, i.e. **~8-11% more**, not
43-49%. Absolute recall@10/precision numbers are still a slight
underestimate of true performance, but the correction is small enough that
it does not materially change how any absolute number in this project
should be read — the caveat is now "modest undercount," not "severe."

**Method lesson**: a review UI that shows confirmed-relevant reference
material on the same page as the item being judged creates a live risk that
any page-scoped verification (not just Ctrl+F — a human's eye can drift the
same way) answers "is X relevant *to this query*" instead of "is X relevant
*to this specific candidate*." If this kind of blinded-judgement UI is built
again, keep calibration material either off-page (a separate, explicitly
different view) or scoped so a search tool cannot cross the boundary.

## 4. Circularity in the entity-lookup arms

**The threat.** The qrels for programme and person queries are derived from
`programs.json` / `people.json` — the **same dictionaries** the `entity_tags`
loader and the `entity_lookup` / `entity_boost` retrieval modes use. Evaluating
those modes against these qrels is partly self-fulfilling: `entity_lookup`'s
recall of **0.9422** substantially measures the dictionary agreeing with itself.

> **Corrected 2026-08-07.** This paragraph previously read "recall@10 of 0.9291",
> which was wrong twice over: the figure predates the 2026-08-05 `entity_tags_full`
> rebuild and its 08-06 re-score (current value **0.9422**), and the metric is
> **recall@1000**, not recall@10 — `entity_lookup` is exhaustive and unranked, and is
> deliberately scored at k=1000 so recall/precision reduce to plain set recall and
> precision. Quoting it as recall@10 invites exactly the false comparison against the
> dense/lexical recall@10 columns that the rest of this section argues against.
> Source: `data/results/gold_entity_lookup_73det_report.md`.

**Scope, which is the mitigating fact.** The chunker, embedder, BM25 and hybrid
comparisons — the bulk of the paper — do **not** touch the entity dictionaries
at query time. The circularity is confined to the entity-lookup/entity-boost
arms and does not propagate.

**What is owed — DRAFTED 2026-08-07.** The project notes internally that this "is
not the user-facing number". In the paper this must be an explicit validity
paragraph, not a footnote: state that the entity arms share a source with the
ground truth, that their scores are therefore an upper bound rather than a
measurement, and that they are not comparable to the dense/lexical arms. **That
paragraph is now written in citable form** — see `docs/paper-results-summary.md`,
§"Circularity in the entity arms — the paragraph the paper owes", which also
records three sharpenings worth keeping:

1. the circularity sits in the **candidate set**, so `entity_boost`'s rank-ordering
   metrics are contaminated only indirectly (hybrid ranking never reads the
   dictionaries), while `entity_lookup` has no ordering to rescue it;
2. **§3's "incomplete, not directional" finding does not transfer to these arms** —
   a name absent from the dictionary is missing from the qrels *and* invisible to the
   retriever at the same time, so here the undercount is correlated with the system
   and its effect is optimistic rather than neutral. This is why the threat cannot be
   closed by measurement the way §1 and §2 were;
3. the scope limit is genuine — `entity_tags_full` is a separate index no other arm
   is built on, so nothing propagates into a headline claim.

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

## 8. Generator non-determinism (RQ4 only) — measured 2026-08-07

Added after the RQ4 refresh against rebuild #3, because it invalidates a
reading the other refreshes in this project earned honestly.

**The threat.** Every retrieval number here is deterministic: re-run the same
index against the same queries and you get the same file, byte for byte. That
is what makes "0 verdict flips after a rebuild" a strong statement — any flip
must come from the corpus change. RQ4 is not like that. `tools/eval/
rq4_generate.py` used to document its temperature-0 setting as "one pass, no
sampling variance to average over". **That is false.** Greedy decoding is
deterministic in exact arithmetic, but GPU reductions are not associative, so
two near-tied logits can swap between runs and the continuation diverges from
there.

**Measured, not assumed.** `tools/eval/rq4_determinism_check.py` re-runs
byte-identical prompts (24 per prompt variant) and compares the citation set —
the thing the scorer actually reads:

| prompt variant | identical citation sets |
|---|---|
| `sentence_cap` | 21/24 (88%) |
| `cite_all` | **14/24 (58%)** |

**Consequence for how RQ4 diffs are read.** The 2026-08-07 refresh produced 5
verdict flips out of 33. Four were losses in the `sentence_cap` family, all
already borderline (Holm-adj 0.014–0.081), and nothing at p<0.001 moved. Those
four **cannot be attributed to the rebuild**, because a re-run with no data
change at all moves comparisons of that size. They are reported as
*inconclusive*, not reversed. The prompt-ablation result (all Holm 0.0000) sits
far outside this noise floor and is unaffected.

**General rule this establishes.** Before diffing any before/after of an
LLM-generated eval, measure the generator's own noise floor on the quantity the
scorer consumes. A diff smaller than the noise floor is not a finding in either
direction. This is the mirror image of §2: there, ties were only citable once
the MDE was known; here, *differences* are only citable once the reproducibility
floor is known.

## Summary of what changes in the paper

| threat | status | action |
|---|---|---|
| sample size | not a real weakness | report depth (1,046 judgments, 9.87/query) alongside count |
| statistical power | **addressed** | cite MDE + CI bounds; restate every tie as a bound, flag `e5_small`/`jina_v5` MRR as inconclusive |
| pooling bias | **CLOSED 2026-08-03** | not directional (all arms tied, ~19-22% residual relevance rate after correcting a review-app measurement bug) — relative comparisons stand, absolute recall/precision numbers need only a modest undercount caveat (~8-11% more relevant docs/query than the qrels record) |
| circularity | known, scoped | explicit validity paragraph; entity-arm scores as upper bounds |
| single annotator | mitigated by construction | publish the derivation rules |
| query provenance | fine, needs stating | describe the chain; keep thematic separate |
| external validity | inherent | declare as scope |
| generator non-determinism (RQ4) | **measured 2026-08-07** | report the reproducibility floor (88% / 58% identical citation sets at temperature 0) alongside RQ4's numbers; call the 4 borderline `sentence_cap` flips inconclusive rather than reversed |
