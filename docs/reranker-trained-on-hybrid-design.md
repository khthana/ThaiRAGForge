# Follow-up (a): a reranker trained on hybrid-fused candidates — pre-registration

**Status: RUN 2026-08-12. §1–§5 are the pre-registration, frozen as written before
the treatment existed; §6 is the outcome.** Decision rule 1 fired: `T vs D`
recall@10 **+0.0637**, Holm 0.0000. Read §6.3 before citing the headline — the free
lexical control lands within **0.0043** of the trained model on recall@10, which is
§5's central threat arriving in the form it was written to detect.

Every
number quoted here is an already-published figure the experiment will be anchored
against; the prediction and the decision rule are written down before the treatment
exists, because this is the fourth reranker intervention on this corpus and the
previous three all had a tempting post-hoc reading available.

---

## 1. What (a) is, and why it is the one reranker axis still open

`docs/reranker-hybrid-interaction-research.md` closes with two candidate
interventions. (b) — blending the reranker's score back into RRF instead of
truncate-and-replace — was built and measured
(`reranker_rrf_signal_test.py`, then `reranker_rrf_routed_test.py`) and is settled:
it beats its unrouted baseline **+0.0379** and does **not** survive the hard router
(**+0.0017**, Holm 1.0000).

(a) is quoted verbatim from that doc:

> a reranker trained or at least validated on hybrid-fused candidate distributions
> specifically (HYRR's own approach, §2) rather than assuming an off-the-shelf
> single-retriever-trained cross-encoder transfers

It has survived every result since, and it survived them for a specific reason
rather than by neglect: **the oracle column says the headroom is real and the model
is what is weak.** On the routed hybrid pool at P=50 the pool *holds* 0.9054 of the
gold and a perfect selection of 10 from it *delivers* **0.8331** — **+0.1500** over
arm C — while `bge-reranker-v2-m3` delivers **+0.0017**, about **1%** of its own
ceiling. Swapping the model (`reranker_model_comparison.py`) moved recall@10 by
**0.0355** across four qualified models, ~20x the anchor's entire effect, and the
best of them captured **13%** of the +0.1500. So 87% is untouched, and two
independent routes — the oracle and the model swap — already agree the verdict is
*this cross-encoder is weak*, not *nothing is left to win*.

(a) is the remaining hypothesis about **why** it is weak: it was trained on
single-retriever (MS MARCO-style, dense- or BM25-derived) negatives, and the
candidates it is asked to rank here come from a *fused* distribution over a Thai
minutes corpus. If that mismatch is the cause, training on the actual candidate
distribution fixes it. If it is not the cause, nothing here will move, and that
closes the axis for cross-encoders at this scale.

## 2. Design — one thing varies

| held fixed | value |
|---|---|
| model architecture + starting weights | `BAAI/bge-reranker-v2-m3` (the published anchor) |
| query set | the 106-query `gold_query_set_73det.yaml` |
| routing | `classify_query` → the 4 shipped hybrid indices (0/106 unmatched) |
| pool | routed hybrid, **P=50**, built by the same code path as `reranker_rrf_routed_test.py` |
| fusion | rrf4, `w` fitted **leave-one-out** on recall@10 |
| budget | k=10 sent, 50 fetched — identical to published arm D |
| **varies** | **the cross-encoder's weights: off-the-shelf vs fine-tuned on hybrid-fused candidates** |

Starting from the anchor's own weights rather than a fresh encoder is deliberate:
it makes the headline a **within-model paired before/after**, so a difference cannot
be attributed to model size, tokenizer, or language coverage — the three things
`reranker_model_comparison.py` already showed matter more here than the intervention
under test. It also means the "before" arm needs no new run: it is published arm D.

### 2.1 Training queries — disjoint by construction

Training queries are minted by **the eval qrels' own generator**,
`tools/corpus_prep/build_gold_candidates.py`, over entities **not present in the
106-query eval set**. Its candidate pool holds 147 programs / 1,139 people /
14 faculties / 678 courses; removing the eval entities leaves roughly
117 / 1,109 / 1 / 645.

Using the same generator is a deliberate choice with a cost, stated in §5: it means
training labels and eval qrels share one relevance definition. The alternative —
inventing a second labelling rule for training — would measure a different task and
make a null uninterpretable.

**`faculty` is an out-of-distribution route by construction.** Only one faculty
entity survives the disjointness filter, so 13 of 106 eval queries (12%) come from a
route the model has essentially never trained on. This is a limitation of the corpus,
not a choice, and the per-route table must report it rather than average it away.

### 2.2 Training candidates come from the pool the model will face

For each training query: route it, retrieve the **routed hybrid P=50 pool** with the
same `rank_one_index` / RRF code path the eval uses, and label each candidate by the
generator's rule for that entity type. That is the entire point of (a) — the negatives
are hybrid-fused near-misses from this corpus, not MS MARCO negatives.

Objective: group-wise softmax cross-entropy (1 positive vs N sampled negatives drawn
from the same query's pool), which is the loss `bge-reranker` itself is trained with,
so the fine-tune continues the model's objective rather than switching it.

### 2.3 Fitting in 12 GB

XLM-R-large ≈ 560M params, of which ≈ 256M is the 250,002 × 1024 word-embedding
matrix. Freezing word embeddings leaves ≈ 304M trainable → ≈ 5.9 GB of
params + grads + AdamW state, plus activations under bf16 autocast and gradient
checkpointing. No new dependency: a plain PyTorch loop over
`AutoModelForSequenceClassification` (`accelerate`, `datasets` and `peft` are not
installed, and this experiment is not a reason to install them).

## 3. Pre-registered comparisons, family, and decision rule

Two arms are compared, both against the **shipped** configuration, because this
project has twice published a gain that existed only against a baseline that had
stopped shipping:

- **T vs C** — trained reranker + routing, against routing alone (0.6831).
  *Is it worth deploying?*
- **T vs D** — trained against untrained, everything else identical (0.6847).
  *Is training on hybrid-fused candidates the missing ingredient?* **This is the
  test the intervention is named after.**

**Family: m = 6** (2 comparisons × {recall@10, MRR, nDCG@10}), paired bootstrap
10,000 resamples, seed 42, Holm. Primary metric **recall@10**, matching every
previous arm in this line of work.

**Decision rule, fixed now:**

1. **`T vs D` recall@10 significant and positive** → (a) is confirmed: the
   off-the-shelf model's failure was candidate-distribution mismatch. Report the
   share of the **+0.1500** oracle captured, and only then consider wiring.
2. **`T vs D` ns** → report as a **bound** (per [[feedback_report_ties_as_bounds]]),
   and the axis closes: neither model choice (0.0355 spread) nor training
   distribution moves a cross-encoder far into the +0.1500, so what remains needs a
   qualitatively different reranker, not a re-tuned one.
3. **`T vs C` significant while `T vs D` is not** → do **not** report this as (a)
   working. It would mean the trained and untrained models are indistinguishable
   from each other while both beat the router, which contradicts published arm D and
   is a signal to look for a bug, not a result.

**Nothing is wired into `query_service` on the strength of this experiment**,
whatever it returns. Wiring is a separate decision with a cost side (~1.2 s/query and
50 extra fetches) that a recall@10 gain does not settle on its own.

### Prediction (recorded before the run)

**A small positive `T vs D`, not significant.** Reasoning, so a wrong prediction is
informative: the oracle diagnosis is that the model cannot *select* among candidates
that already contain the gold — a ranking failure over an evidence-rich pool — and
candidate-distribution mismatch is only one of several possible causes of that
(others: the 512-token window against long Thai minutes, and the model's Thai
capability, which §5 of the research doc explicitly could not rule in or out). The
family's MDE at n=106 is ~0.05 on recall@10, and the anchor's own effect is 0.0017,
so a fine-tune would have to buy ~30x the anchor's effect to clear the bar.

If the prediction is wrong in the *positive* direction, that is the finding. If it is
wrong in the *negative* direction — a trained model that is significantly **worse** —
that is also a finding, and the first control in §4 is what would explain it.

## 4. Controls, all pre-registered

1. **Lexical-containment control (no GPU).** Score each (query, chunk) pair by
   whether the chunk text contains the query's anchor string, and fuse it through the
   identical rrf4 path. If the trained model does not beat this trivial scorer, what
   it learned is the labelling function, not relevance. This is the control that
   makes §5's circularity threat measurable rather than merely disclosed.
2. **Held-out training-query metric.** Report the trained model's ranking quality on
   a held-out slice of *training* queries. Without it, a null on the eval set cannot
   be told apart from "the fine-tune never converged" — the same reason
   `reranker_pool_source_test.py` had to carry an oracle column.
3. **Anchor reproduction (S-checks).** The untrained model, run through the new
   script, must reproduce published arm C (**0.6831**) at w=0 and arm D
   (**0.6847**) at its LOO w, and the routed P=50 oracle must reproduce
   **0.8331 delivered / 0.9054 holds**. A new code path that cannot reproduce the
   old numbers is not measuring the same thing.
4. **Truncate-and-replace, reported beside the fusion arm.** Published at
   **0.6000** on the routed pool for the untrained model; if training helps at all it
   should show there most strongly, since that arm has no hybrid ranking to hide
   behind.
5. **Qrels ceiling.** No arm sending 10 documents may exceed **0.8856**
   (`paper-results-summary.md`'s structural ceiling). Gated, not assumed.

## 5. Validity threats

- **Shared labelling rule (the central one).** Training labels and eval qrels are
  produced by the same string-containment generator, so a gain may mean "the model
  learned the labelling function" rather than "the model learned relevance". This is
  the [[project_rq4_entity_arms_gating]] circularity one layer up. It is *disclosed
  and controlled* (§4.1) rather than removed, because removing it would require a
  second, independently-derived relevance judgement that this project does not have.
  Any positive result must be reported with the lexical control's number beside it.
- **Pooling bias, inherited.** The qrels are a ~8-11% undercount
  (`docs/eval-validity-threats.md` §2, closed as *incomplete-not-biased*), and a
  trained model is judged against them exactly as every other arm here is. Direction
  is conservative for a *comparison* between two rerankers on the same qrels.
- **Selection.** `w` is fitted leave-one-out, but the *checkpoint* — how many epochs
  the fine-tune runs for — must be selected on held-out **training** queries, never on
  the 106. Selecting a checkpoint on the eval set would make the whole experiment an
  argmax, the same trap `reranker_model_comparison.py` had to disclose for its model
  choice.
- **`faculty` is untrained** (§2.1). 13 of 106 eval queries sit on a route with one
  disjoint training entity; report that route separately.
- **Single fine-tune.** One training run, one seed. A null is a statement about this
  recipe on this corpus, not about fine-tuned cross-encoders in general.

---

## 6. Outcome (2026-08-12)

Artifacts: `tools/eval/train_hybrid_reranker.py` →
`data/results/reranker_training_run.md` (67.6 min, 3 epochs, checkpoint at
`data/models/reranker_hybrid_trained/`, gitignored) and
`tools/eval/reranker_trained_test.py` → `data/results/reranker_trained_test.md`
(716 s the first time, ~95 s GPU-free with `--reuse-scores`).

### 6.1 The pre-registered family — all six significant

| arm | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| C — router only (ships) | 0.6831 | 0.8686 | 0.7502 |
| D — off-the-shelf `bge-reranker-v2-m3` | 0.6847 | 0.8801 | 0.7497 |
| **T — trained on hybrid-fused candidates** | **0.7485** | 0.9717 | 0.8607 |
| L — lexical containment (control, no GPU) | 0.7442 | 0.9308 | 0.8336 |

`T vs C` **+0.0654** recall@10 [+0.0397, +0.0926], MRR +0.1031, nDCG@10 +0.1105.
`T vs D` **+0.0637** recall@10 [+0.0386, +0.0915], MRR +0.0916, nDCG@10 +0.1110.
All six Holm-adj **0.0000** at m=6. **The prediction in §3 was wrong in the positive
direction**, which §3 recorded in advance as the interesting failure: it forecast a
small positive `T vs D`, not significant, on the reasoning that candidate-distribution
mismatch was only one of several possible causes of the oracle's diagnosis. It was
enough of the cause to clear ~30x the anchor's effect.

**This is the first intervention in the whole reranker line to survive the hard
router.** Per-`entity_type` alpha, rrf4 and the model swap all won against no routing
and died against the shipped configuration; this one is measured against arm C from
the start and beats it.

Share of the ceiling: the routed P=50 oracle delivers **+0.1500** over arm C. Trained
captures **44%**, off-the-shelf **1%**. So the axis is not closed by this either —
56% remains — but "this cross-encoder is weak" is now *explained* rather than merely
observed: it was weak because it had never seen a hybrid-fused pool from this corpus.

### 6.2 Where the gain lives, and where it does not

Per-route recall@10 at P=50:

| route | n | C | D | T | L | T − C |
|---|---|---|---|---|---|---|
| person | 30 | 0.8531 | 0.8672 | 0.8816 | **0.9005** | +0.0285 |
| program | 30 | 0.6545 | 0.5912 | **0.7634** | 0.7260 | **+0.1089** |
| course | 33 | 0.6262 | 0.6758 | 0.7145 | **0.7214** | +0.0883 |
| faculty | 13 | 0.5008 | 0.5024 | 0.4931 | 0.4832 | **−0.0077** |

`program` — the route the off-the-shelf model actively *damaged* (−0.0633 in
`reranker_rrf_routed_test.md`) — is where training pays most (+0.1089, and +0.1723
over D). **`faculty` is the one route that gets worse**, exactly as §5 pre-registered:
one disjoint training entity, so 13 of 106 eval queries sit on a route the model never
learned. Dev confirms it from the other side — `faculty` sits at **0.5000 in every
epoch including epoch 0**, i.e. the fine-tune never moved it at all.

The w grid separates the two models qualitatively, not just numerically: T peaks at
**0.7516 at w=0.65** and stays high to w=1.00, while **D declines monotonically past
w=0.40** down to 0.6000. The untrained model is a signal you must dilute; the trained
one is a signal you can lean on.

### 6.3 The control did most of it — read this before citing 6.1

**Arm L, string containment with no GPU and no training, reaches 0.7438 against T's
0.7541** (2026-08-20, against rebuild #4; the 2026-08-12 original read
0.7442 / 0.7485). §4.1 fixed this control as the thing a positive result must be reported
beside, and §5 named the reason: training labels and eval qrels come from one
string-containment generator, so a model that learned the *generator* scores like a
model that learned relevance. Two point estimates side by side are not a comparison,
so the script gained an **exploratory, not pre-registered, family 2** (T vs L, L vs C
× 3 metrics, own Holm, m=6):

- **`T vs L` recall@10 +0.0043, Holm 0.6426 — not significant.** As a bound: the CI
  rules out the trained model beating free string containment by more than **0.0229**
  recall@10.
- **`T vs L` MRR +0.0409 (Holm 0.0150) and nDCG@10 +0.0271 (Holm 0.0432) — both
  significant.** So the fine-tune's separable contribution is **ordering**, not
  *which* documents come back.
- **`L vs C` +0.0611 / +0.0623 / +0.0834, all Holm 0.0000.** The free control alone
  beats the shipped router on all three metrics.

**§4.1's own prediction was refuted, and that is a second finding.** L was predicted
weakest on `course`, because `course` qrels are keyed on the 8-digit code while the
query supplies the name (`gold_anchor_ambiguity.md`). L instead **beats T on
`course`** (0.7214 vs 0.7145) and on `person`; its actual weak route is `faculty` (0.4832),
which every arm fails.

**How to cite this.** `T vs D` (+0.0637) is clean — both arms are cross-encoders
scored against the same qrels, so the shared labelling rule cancels, and *that* is the
test intervention (a) is named after. `T vs C` (+0.0654) is real but **must carry arm
L's number**: most of it is reachable for free, and the honest reading is that
recall@10 on these qrels is largely a containment test. The claim that survives
without qualification is narrower and more useful: **training on hybrid-fused
candidates fixes an off-the-shelf reranker that was damaging `program`**, and buys a
significant ordering improvement over lexical containment.

### 6.4 A threat the pre-registration missed

`S7` checks disjointness and found 0 shared queries and 0 shared entities — but also
**325 resolutions relevant to both the training and eval sets**. Unavoidable in one
corpus of 2,854 documents, and the model never saw an eval *label*, but it did see
some of those documents as positives for other entities. It is reported in the check's
own output rather than waived, and it is the kind of overlap §2.1's entity-level
disjointness argument is structurally unable to see.

### 6.5 Not wired

Per §3, nothing goes into `query_service` on the strength of this. The cost side is
unchanged (~1.2 s/query, 50 extra fetches) and now has a sharper competitor: **arm L
costs nothing and is 0.0043 behind on recall@10.** A deployment decision here is a
choice between three options, not two.
