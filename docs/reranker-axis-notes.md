# The cross-encoder reranking axis — full derivation

**This is the research record. The verdicts, bounds and traps live in
`CLAUDE.md`; this file is why they hold.** It was folded out of `CLAUDE.md` on
2026-08-23 because that bullet had grown to 43.6 KB — 19% of a file loaded into
every session — and the convention at the top of `CLAUDE.md` is that a closed
axis keeps its verdict there and its derivation here. The text below is the
bullet **verbatim**, so every figure it carried is still under
`audit_doc_claims.py`'s D2/D5/D7 (this file is in `DOCS`); nothing was rewritten
in the move.

Related artifacts: `docs/reranker-trained-on-hybrid-design.md` (the frozen
pre-registration for follow-up (a)), `docs/reranker-hybrid-interaction-research.md`
(literature; deliberately **not** in `DOCS` — it quotes 211 figures from other
people's papers), and the reports under `data/results/reranker_*.md`.

---

## The headline, and what the null actually belongs to

**Cross-encoder reranking hurts hybrid — but the finding belongs to
*truncate-and-replace* and to the *off-the-shelf model*, not to reranking.**
**This item is no longer a null: follow-up (a) — a cross-encoder fine-tuned on
hybrid-fused candidates — beats the shipped hard router (+0.0730 recall@10, Holm
0.0000) and is the FIRST intervention in this whole line to do so. Read its
paragraph at the end of this item before citing anything above it as settled.**
Two earlier corrections still stand: fusing the same model's scores in as a fourth
RRF signal (2026-08-09) **beats the shipped hybrid on recall@10**, so what is settled
is "don't let a cross-encoder replace the ranking", not "a cross-encoder is useless
here". `CrossEncoderReranker`
(`BAAI/bge-reranker-v2-m3`, `rerank_pool_size=50` → truncate to k=10) is wired as a
query-time stage; `tools/eval/reranker_significance_test.py` re-retrieves live against
`chunker_compare_full/plain__fixed_size__local__ceea7536` (so it goes stale on every
index rebuild — it is **not** in the persisted-results refresh chain, and must be
re-run by hand). **Refreshed 2026-08-05** against rebuild #3: result unchanged —
**significantly hurts hybrid MRR**, and that survived rebuild #4 too
(re-run 2026-08-18: **0.7730→0.6940**, diff **−0.0790**, Holm-adj **0.0240**; it was
0.7814→0.6778 p=0.0012 at rebuild #3 and 0.7775→0.6775 p=0.0048 before that — the
margin has shrunk at each rebuild while staying significant). **No significant effect on
dense-alone** (all three dense metrics Holm-adj p≥0.3270), **no significant effect on
hybrid recall@10 or nDCG@10** either (**0.7112 / 0.5442**) — MRR-only is still the
correct framing, and it costs ~1.23s/query mean (p50 1.17s, p95 1.42s). The nDCG@10 harm reported 2026-07-23
(p=0.030) still does not replicate (now p=0.284, was p=0.5676 at the 2026-07-29
refresh) and stays retired as a separate claim — it actually sharpens the
literature's "phantom hits" mechanism (early-rank disruption without evicting relevant
docs from the top-10). Literature grounding — including a paper naming
`bge-reranker-v2-m3` by name — is in `docs/reranker-hybrid-interaction-research.md`.
**The "wrong pool" escape hatch is now CLOSED (2026-08-09,
`tools/eval/reranker_pool_source_test.py` → `data/results/reranker_pool_source_test.md`)**,
on the *best* combo (`sentence × qwen3_0.6b`, not the original test's weaker
`fixed_size × bge-m3` — say so when citing, the two tables are not comparable):
pool source ∈ {dense, hybrid} × P ∈ {10,20,50,100,200}, equal 10-doc budget, with an
**oracle rerank of the same pool beside every real arm**. `miss_depth_profile.md`'s
"dense is closest on 70 of 84" motivated it; **the hypothesis is rejected in the
opposite direction** — a dense pool is significantly *worse* (recall@10 **−0.1143**,
Holm-adj 0.0000, m=3 — re-run 2026-08-20; was −0.1085) and loses to shipped hybrid on all three metrics. **The reasoning
error is the reusable part: "closest on the pairs everyone misses" is about 84 pairs,
but a pool serves all 1,046** — dense's 0.5041 baseline starts too far behind hybrid's
0.6229 for the hard pairs to repay. Two things the original test could not show. (1)
**The evidence is in the pool and the reranker does not find it**: at P=50 the hybrid
pool holds **0.8896** of the gold and a perfect rerank of it delivers **0.8268**, but
the real reranker delivers **0.6182** — *below its own baseline*. Without the oracle
column a null cannot be told apart from "the evidence was never reachable"; it was.
(2) **Depth and harm point opposite ways, which is what closes the axis**: the misses
sit at ranks 11-50 but captured headroom goes **−2% / −17% / −29%** at P=50/100/200, so
it cannot reach them without destroying more than it recovers. Its per-type table shows
damage concentrated on `person` (**−0.2620** vs `course` −0.0192) — **but do not read
that as a truncate-and-replace effect; it is a POOL-SOURCE effect, corrected the same
day** (see the next paragraph). It is the *dense*-pool arm, i.e. what happens when the
candidates come from the retriever that scores 0.5735 on `person` rather than the one
BM25 carries to 0.8147. On the hybrid pool, truncate-and-replace *improves* `person`
(+0.1243) and collapses `program` (−0.1553). The one improving cell
(hybrid P=20, 0.6464 vs 0.6229) was **not pre-registered** — cite it as a hypothesis for
a fresh query set, never as a result. Cost is real too: P=50 adds ~1.2 s/query on a
1.21-1.86 s base. Method worth reusing: a cross-encoder score depends on neither P nor
the pool's source, so **score each (query, chunk) pair once and derive all 10 arms from
the cache** (~1.3 arms' cost, and two arms can't disagree about one pair); it is
persisted so a re-render reproduces the report line for line (784 s → 53 s).
**`--reuse-scores` is NOT GPU-free and this file said it was until 2026-08-20**, in
all three reranker scripts plus the routed report's own footer: it skips the
*cross-encoder*, which is the expensive part, but `rank_one_index` runs
unconditionally and loads an embedder onto the card. The wrong claim is what caused a
re-render to be launched beside a running fine-tune on the one 12 GB card — it
survived on free VRAM, not by design. Corrected at the source in every place that
asserted it.
Its S6 rebuilds `miss_depth_profile.md` §2's five delivered figures to 4 decimals from
an independent path, and S4 pins the structural anchor that at P=k reranking may change
the *order* but never the *set*, so recall@10 must equal baseline exactly.
**The 4th-RRF-signal follow-up is now MEASURED, and it is this item's one positive
result (2026-08-09, `tools/eval/reranker_rrf_signal_test.py` →
`data/results/reranker_rrf_signal_test.md`, 68 s, no GPU — it re-fuses the *same*
29,743 cached scores).** Keep shipped hybrid fusion and add the reranker as a third
ranked system: `fused = (1-w)·[0.5/(60+dense_rank) + 0.5/(60+bm25_rank)] +
w·[1/(60+rerank_rank)]`, **a document outside the pool contributing 0 from the third
term** — not penalised, just unvoted-on, with its hybrid rank intact. That asymmetry is
the whole idea: it is what lets an exact-name BM25 hit survive a reranker that dislikes
it. **Both ends of the w grid are already-published arms**, so it interpolates between
known points rather than introducing a scale: w=0.00 is shipped hybrid and **w=1.00 is
truncate-and-replace exactly** (hybrid term gone; every pool doc scores >0, every
non-pool doc 0, so the top-10 is the pool's top-10 by cross-encoder score) — S3/S4 pin
both, S4 reproducing all six published figures to 4 decimals from an independent code
path. Same discipline as `hybrid_alpha_sweep.py`'s alpha=0.50 anchor. Pre-registered at
pool=hybrid, P=50, w chosen **leave-one-out** on recall@10 (Family 1, m=6):
**`rrf4 (loo)` 0.6622 recall@10 vs shipped hybrid 0.6229, +0.0392, Holm-adj 0.0108 —
significant** (re-run 2026-08-20; it was 0.6660 / 0.6281 / +0.0379 / 0.0216, i.e. both
levels fell and the margin *strengthened*), and it beats truncate-and-replace on all
three metrics (+0.0439 / +0.1024 / +0.0698, Holm 0.0064 / 0.0020 / 0.0000). **Cite MRR as REPAIRED, not improved**: the published
harm reproduces at w=1.00 (−0.1197) and vanishes under fusion but does not become a
gain (**−0.0221**, ns; CI rules out a loss worse than **0.0642** or a gain better than
**0.0181** — the bound loosened at rebuild #4, it read −0.0026 / 0.0420 / 0.0368),
and nDCG@10 **+0.0202** is ns too — **recall@10 is the only claim that clears
significance.** No fitting premium: all 106 folds pick the same w, so LOO equals the
oracle to 4 decimals — **still true after rebuild #4** (1 distinct w over 106 folds,
modal **0.45**) — but the peak **narrowed**: **report the range 0.40–0.45, not a
point** (it was 0.40–0.55; w=0.50 now drops to 0.6522 from the 0.6622 peak). **The mechanism, corrected**: the prediction (fuse > replace) survived, the
stated reason did not. The cross-encoder is *not* uniformly destructive — on the right
pool it is a `person` specialist that wrecks `program`, and what RRF buys is **keeping
both sides**, recovering `program` +0.1155 over truncate-and-replace while giving back
only −0.0190 of the person gain. Also: **once the reranker is only a vote, pool depth
stops mattering** (P=20 peaks 0.6614 vs P=50's 0.6622, at 486 ms/query instead of
1,216) — a cost observation from a descriptive column, not a pre-registered result.
**That +0.0392 was measured without routing, and it does NOT survive the hard router
(2026-08-09, re-run 2026-08-20 against rebuild #4 — 142 s,
`tools/eval/reranker_rrf_routed_test.py` →
`data/results/reranker_rrf_routed_test.md`, 10,600 pairs over the 4 routed
indices; **10/10 self-checks PASS**).** Measured as a 2×2 because "does it still help"
and "substitutes or complements" are one experiment: **A** no routing/no rrf4
**0.6229**, **B** rrf4 only **0.6622**, **C** routing only **0.6811**, **D** both
**0.6713**; every arm sends k=10, B and D additionally *fetch* 50. **All six
pre-registered tests (m=6) are ns, before and after the rebuild — 0 verdict flips**:
`D vs C` (the reranker on top of routing) **−0.0098 recall@10, Holm-adj 0.9768**
(MRR +0.0047, nDCG −0.0071, both 1.0000); `D vs B` +0.0091/+0.0408/+0.0239, Holm
1.0000/0.9768/0.9768. **The verdicts held but the bound moved a long way, and that is
the citable change**: the point estimate flipped sign (**+0.0017 → −0.0098**) and
the CI now rules out the reranker adding more than **+0.0037** on top of the router —
it was **+0.0212**, so the case against wiring rrf4 is ~5.7x tighter than published,
for ~1.2 s/query and 50 extra fetches. **This is the second intervention to die
against the router in exactly this way** (per-`entity_type` alpha was the first) and the
mechanism is identical both times — both repair a per-type weak dense arm, and hard
routing already hands each route a specialist index that hasn't got one. The per-route
table showed a near-cancellation before the rebuild (`course` **+0.0496**, `person`
+0.0140, `program` **−0.0633**) and now shows a small net loss: `course` **+0.0126**,
`faculty` +0.0019, `person` −0.0047, `program` **−0.0445** — i.e. **the one route
the reranker used to help lost three quarters of that gain**, while the `program` damage
(the same cross-encoder personality as above) shrank less. Routing had
already collected the person gain that made the unrouted number large (person 0.7440
unrouted → 0.8531 routed *before* any reranking). **Substitutes, not complements** — the
same verdict soft-vs-hard routing reached. Two supporting details: there is no fitted
signal left either (the P=50 w grid wanders with no shape, a jagged plateau
not a peak, and LOO **0.6713** vs oracle **0.6867** is a real fitting premium where the
unrouted sweep had none — and rebuild #4 **widened** it, 0.0048 → **0.0154**), and
truncate-and-replace on a *routed* pool is worse still (**0.5987** at
P=50, **0.6640** at P=20). Descriptively (not pre-registered): **B 0.6622 < C 0.6811**, i.e.
routing alone beats the reranker path while costing no extra fetch and no query-time GPU.
Three of the four cells are already-published numbers and the script **checks all three
rather than assuming them** — S4 reproduces `routing_eval.md`'s **0.6811** from a *third*
independent code path, S5 reproduces **0.6229/0.6622**, S1/S2 reproduce 106/106 persisted
top-10s. **All three were frozen literals until 2026-08-20** — the 8th, 9th and 10th
cross-artifact anchors of the kind `561102e` replaced elsewhere — and are now parsed
live from their reports (`parse_routing_eval_routed`, which must select the
`-- hybrid` section, since the `-- dense` one reads 0.6173; `parse_rrf_signal_arms`;
`parse_pool_source_oracle`), each printing "UNPARSED — the cross-check could not be
made" rather than passing silently. **Neither rrf4 nor per-type alpha is wired into `query_service`, and this is
why.** **But the axis is NOT dead, and the oracle column is what says so**: a null alone
cannot separate "this reranker is weak" from "nothing is left to win", so the same
oracle was computed over the *routed* pool. At P=50 the routed pool **holds** 0.9054 of
the gold and a perfect selection of 10 from it **delivers 0.8331** — **+0.1520 over arm
C, against the real reranker's −0.0098, i.e. −6% of its own ceiling** (it was "about
1%"; the oracle is unmoved by the rebuild to 4 decimals and the real arm went
negative). So the
verdict is *this cross-encoder is weak*, not *the headroom is gone*, and **routing
enlarges the headroom rather than shrinking it** (routed 0.9054 holds / 0.8331
delivered vs unrouted **0.8896 / 0.8268** — the specialist indices supply *better*
candidates and the model still cannot select among them, the same shape as the
unrouted diagnosis). Cite it as a **bound on the axis, not a plan**: an oracle is not a
system, and closing any of +0.1520 needs a reranker qualitatively better than
`bge-reranker-v2-m3` here, not a re-tuned fusion. Follow-up (a), a reranker trained on
hybrid-fused candidates, keeps its motivation and remains untouched. **One trap, found
by a failing self-check rather than by reasoning**: the delivered oracle is
`min(#relevant resolutions with a chunk in the pool, K) / #relevant`, so chunks sharing
a `resolution_id` **must be deduplicated first** — a perfect reranker never spends one
of its 10 slots on a document it already returned. Sorting the pool relevant-first
*without* dedup understates the ceiling (0.7790 instead of 0.8268 unrouted at P=50);
S9, which reproduces `reranker_pool_source_test.md`'s published `delivered/holds` pair
from an independent code path, is what caught it.
**"This cross-encoder is weak" is now CONFIRMED by a second route (2026-08-09,
re-run 2026-08-20 against rebuild #4 — 7 min for 3 models,
`tools/eval/reranker_model_comparison.py` → `data/results/reranker_model_comparison.md`,
8/8 self-checks PASS): swap the model, change nothing else.** Same routed hybrid
P=50 pool for every arm, same k=10 sent, same LOO-fitted `w`. **The verdict holds and
its strongest form is now a fact about ordering, not a ratio.** Over 4 qualified models
the spread is **0.0262** recall@10 (was 0.0355), and **the anchor is now the WORST of
the four** (0.6713, the only one *below* the router) while the other three all beat it
numerically — `bge-reranker-**v1**-large` **0.6975**, `bge-v1-base` 0.6820,
`mmarco-mMiniLM` 0.6822. **Do not restate the old "the spread is ~20x the anchor's whole
effect"**: that ratio was 0.0355 against +0.0017, and with the anchor's effect now
−0.0098 it reads ~2.7x — a much weaker sentence for the same conclusion, which is why
the ordering is the thing to cite. So the null belongs to
`bge-reranker-v2-m3`, not to cross-encoder reranking on this corpus — the same verdict
the oracle column reached, from independent evidence. **Cite the recall@10 family as
inconclusive, not as a win**: 0 of 3 clear the bar (best `bge-v1-large` +0.0164, raw
0.0512, **Holm 0.1536**, m=3 — it moved *away* from the bar, it was +0.0196 / 0.0282 /
0.0612), and the one significant cell is still nDCG@10 **+0.0257**
(Holm 0.0336, m=6, family 2; was +0.0275 / 0.0228). **The counter-intuitive part is the
strongest part**: the
best model is the *older* v1 lineage that v2-m3 supersedes, so reranker choice here does
not track general benchmark strength and has to be measured on this corpus.
**The `mmarco-mMiniLM` illustration is WITHDRAWN**: it was cited here as the RRF rule
again (actively **hurts**, −0.0159) and after rebuild #4 it reads **+0.0011**, i.e. it
no longer hurts — the rule is unaffected, this table simply stopped illustrating it,
the same shape as the family-size example `soft_vs_hard_routing.md` lost. The generator
printed the verdict word "hurts" beside its own *positive* number until 2026-08-20
([[feedback_a_hardcoded_verdict_word_rots_unseen]]); that word, the "20x", and the
nDCG cell are now **derived from the tables** rather than typed. **Selection caveat**: the winner is an argmax
over 4 models on the same 106 queries (`w` is LOO, the *model* is not), so the citable
claim is *at least one qualified model does materially better*, never *use bge-v1-large*
— that would need a fresh query set, and **that confirmation is CLOSED as dominated
(2026-08-12): do not re-propose it.** Three reasons no measurement would change. (i) The
outcome has no decision attached — follow-up (a) reaches 0.7541 and its **free** lexical
control 0.7438, so `bge-v1-large`'s 0.6975 is dominated by an arm costing no GPU. (ii) The
claim it served (*the model is the weak part, not the axis*) already holds from two
independent routes, the oracle column and (a) capturing 48% of that gap at Holm 0.0000.
(iii) No clean fresh set exists: the 179 thematic queries move query *shape* and retrieval
regime together (BM25 collapses there), so a non-replication could not be attributed —
the wrong-pair trap that killed per-`entity_type` alpha and rrf4 — and the only same-shape
disjoint queries are (a)'s own training set, ~2 h GPU for a result nobody would act on.
**The bound is unchanged**: the best model captures **11%**
of **+0.1520** (was 13% of +0.1500), so **89%** is still untouched and follow-up (a)
keeps its motivation. Nothing is
wired. Confounds measured rather than assumed: `ctx` is the one thing not equal across
arms (anchor 8192, the rest 512 — each at its own max, since forcing 512 on the anchor
would stop it reproducing its published number), but only **1.9%** of pairs exceed 512
and the longest is **2,755** tokens; and S6 pins that the models genuinely disagree
(Kendall τ +0.344 to +0.546, same top-1 on 17–44 of 106), because two models that rank
the pool identically would give identical arms and a null would then only be saying the
swap never happened. **Before measuring a reranker, qualify it —
`tools/eval/qualify_reranker_model.py` → `data/results/reranker_model_qualification.md`,
and this is the reusable part.** 2 of 6 candidates are broken under `transformers` 5.x,
which materialises non-persistent buffers from the meta device as *uninitialised memory*
rather than re-running `__init__`. `jina-reranker-v2` dies at import (safe).
`gte-multilingual-reranker-base` **loads, runs, and ranks a hand-written Thai example
correctly while being completely position-blind** — its RoPE `cos_cached`/`sin_cached`
came back all zeros, and a sentence and its reversal score **bit-identically**. A
plausible number from a bag-of-words model is the danger, not a crash: measured
unchecked it would have scored low and "a second cross-encoder also fails" would have
been published as a family-level claim. Five gates (G1 buffers, **G2 position
sensitivity** — the load-bearing one, G3 relevance direction, G4 determinism, G5 padding
independence), one model per **subprocess** (a CUDA device-side assert poisons the whole
process, so a single-process loop rejects healthy models by ordering), and the gate is
**exercised in both directions** — the anchor must PASS and the 2 known-broken must
FAIL, checked in the report and in the exit code, because a PASS-only gate is not
evidence. Two rules learned by getting them wrong: G1's integer test is *index out of
range*, **not** *equals `arange`* (the first version rejected the anchor — XLM-R's
`token_type_ids` is legitimately all zeros), and every gate but G5 scores its pair
**alone**, so batch composition can't be mistaken for the effect under test.
`tests/tools/test_qualify_reranker_model.py` pins both directions of every rule.
**FOLLOW-UP (a) IS DONE AND POSITIVE (2026-08-12) — a cross-encoder trained on
hybrid-fused candidates is the first intervention in this line to survive the hard
router.** Pre-registration **and** outcome in one file, §1-5 frozen as written before
the treatment existed: `docs/reranker-trained-on-hybrid-design.md`; artifacts
`tools/eval/train_hybrid_reranker.py` → `data/results/reranker_training_run.md`
(67.6 min, checkpoint `data/models/reranker_hybrid_trained/`, gitignored) and
`tools/eval/reranker_trained_test.py` → `data/results/reranker_trained_test.md`
(716 s, then ~95 s with `--reuse-scores` — which still uses the GPU for retrieval, see
above). Only the **weights** vary: pool,
routing, rrf4, the `w` grid, P=50, k=10, metrics and bootstrap all held at published
values, and the fine-tune **starts from the anchor's own weights** so the headline is a
within-model paired before/after — a difference can't be attributed to model size,
tokenizer or language coverage, the three things `reranker_model_comparison.py` showed
matter more here. **REFRESHED AGAINST REBUILD #4 (2026-08-20) — every figure in this
paragraph is the re-run's; the 08-12 original is named wherever a claim changed, and
one of them is withdrawn outright.** **T vs D +0.0828 recall@10 / T vs C +0.0730, all
six pre-registered tests Holm 0.0000 (m=6)**; arm T 0.7541 vs C 0.6811, D 0.6713.
**`T vs D` grew (+0.0637 → +0.0828) mostly because arm D FELL** (0.6847 → 0.6713,
the off-the-shelf model's own rebuild-#4 loss — see the routed bullet above), not
because training got better; state the two separately. **The §3 prediction
(small positive, ns) was wrong in the positive direction**, recorded in advance as the
informative failure. It **explains the earlier nulls rather than contradicting them**:
the oracle column and the 4-model swap had already said *this cross-encoder is weak,
not the headroom is gone*, and (a) names the weakness — `program`, the route the
off-the-shelf model actively **damaged** (−0.0445 at rebuild #4, was −0.0633), is where
training pays most (**+0.1118** over the router, +0.1562 over D), and the w grid
separates them qualitatively (T rises to **w=0.90** and stays above C all the way to
1.00; **D peaks at w=0.10 and declines** to 0.5987). Trained captures **48%** of the
routed P=50 oracle's +0.1520 against off-the-shelf's **−6%** — the off-the-shelf
arm now sits *below* the router, so the gap it leaves is the whole ceiling. **The axis
is narrowed, not closed.** **`faculty` gets worse
(−0.0064)** exactly as pre-registered: one faculty entity survives the disjointness
filter, so 13 of 106 queries sit on a route the fine-tune never learned — dev recall
there is **0.5000 in every epoch including epoch 0**. **THE CONTROL IS THE PART TO
READ BEFORE CITING THE HEADLINE.** A free lexical-containment scorer fused through the
identical rrf4 path (arm L, no GPU, no training) reaches **0.7438** against T's 0.7541.
An **exploratory, not pre-registered** family 2 bounds it — and **rebuild #4 took
away the two cells that used to carry it, which is the one claim here that is
WITHDRAWN rather than restated.** `T vs L` was significant on **MRR** (+0.0409, Holm
0.0150) and **nDCG@10** (+0.0271, 0.0432); it is now **ns on all three** — recall@10
**+0.0103** (Holm 0.2896), MRR **+0.0330** (0.1368), nDCG@10 **+0.0288** (0.0930). So
*the fine-tune's separable contribution over string containment is **ordering, not
which documents come back*** is **withdrawn**: nothing separates the two on any
metric now. **Read it as power, not reversal** — every sign is unchanged and the
nDCG effect even *grew* (+0.0271 → +0.0288) while its Holm p doubled, which is the
bootstrap widening, not a direction changing
([[feedback_a_replication_disagrees_by_sign_not_verdict]]) — and therefore **cite it
as a bound: T beats L by at most 0.0298 recall@10, L beats T by at most 0.0089.**
`L vs C` is still significant on all three (+0.0627/+0.0623/+0.0849), so the free
control genuinely beats the shipped router. And recall@10 on
these qrels is largely a containment test — the shared-labelling circularity §5
pre-registered, arriving in the form it was written to detect. **Cite `T vs D` cleanly**
(both arms are cross-encoders against the same qrels, so the shared rule cancels — and
it is the test (a) is named after); **never cite `T vs C` without arm L's number beside
it.** §4.1's own prediction was **refuted** too and that is a second finding: L was
predicted weakest on `course` (qrels keyed on the 8-digit code, query supplies the
name) and instead **beats T on `person`** (0.9033 vs 0.8816); its real weak route is
`faculty` (0.4773). **The `course` half of that illustration is withdrawn at rebuild
#4**: L is unmoved there (0.7214) while T rose to 0.7328, so L no longer beats it.
The refutation itself stands, because it rests on which route L is *weakest* on, not
on which routes it happens to win — the same distinction the `mmarco` withdrawal
above turns on. Two more things worth keeping. **S7 surfaced a threat the
pre-registration missed**: 0 shared queries and 0 shared entities, but **325 resolutions
are relevant to both sets** — unavoidable in one corpus, never a label the model saw,
and structurally invisible to an entity-level disjointness argument. And **the
checkpoint-loading contract was verified on a CPU fixture *before* the GPU time was
spent** (`scratchpad/probe_ckpt_load.py`, 8/8): sentence-transformers accepts the
trainer's bare `save_pretrained` directory, the base head is **already 1-logit** so
`num_labels=1` keeps the pretrained ranking head rather than reinitialising it (C2's
2e-7 agreement with ST independently confirms it — a random head could not match), and
ST caps the tokenizer at the *config's* `max_position_embeddings`, so the trained arm
scores at the same **8192** as every published arm despite training at max_len 1024
(49 of 25,250 training pairs truncated, 0.19%). **The trained model is not wired**
— per §3, and the cost side has a sharper competitor than it did (~1.2 s/query and
50 extra fetches, against a control at zero cost).
**ARM L′ IS WIRED (2026-08-20, `src/rag_lab/retrievers/lexical_containment.py`,
registered `lexical_containment`), and the reason it is L′ and not the published
arm L is the finding.** `lexical_cache` reads the entity out of the **gold YAML**, so
arm L is handed the very string the qrels were derived from while arms C/D/T see only
the query text — **it is not merely free of GPU, it is fed an input no other arm gets
and no deployment has.** The deployable form recovers the entity with the shipped
`router.detect_entities`; a new **arm L′** measures exactly that, in its own Holm
family (**family 3, m=9, exploratory**) so no published p in families 1–2 moves.
Three results, and the third corrects this file's own wording from earlier the same
day. (1) **Losing the oracle string is cheap**: `L′ vs L` is ns on all three
(−0.0138 / −0.0186 / −0.0135, Holm 0.1950) — cite it as ruling out a loss worse
than **0.0304** recall@10. (2) **The deployable arm still beats the shipped router
significantly on every metric**: arm L′ **0.7300** vs C 0.6811, `L′ vs C`
**+0.0489** recall@10 (Holm 0.0000), +0.0437 MRR (0.0084), +0.0714 nDCG@10 (0.0000),
at **no GPU cost** — the layer that actually saturates under load. (3) **`T vs L′`
IS significant on all three** (+0.0241 / +0.0516 / +0.0423, Holm 0.0290 / 0.0424 /
0.0042) where `T vs L` is significant on none. Not a contradiction (T−L +0.0103,
L−L′ +0.0138, T−L′ +0.0241) but the reading changes: **"the fine-tune is not
separable from string containment" holds only against the ORACLE-FED control; against
the deployable one it separates everywhere**, so part of what the training buys is
*not needing an entity extractor*. `S11` pins that the two arms really are fed
different strings (**63 of 106** — `person` 30/30 title-stripped, which still matches
as a substring; `course` 33/33 returning the **8-digit code** instead of the name,
which is a *different signal*, not a degraded one) so family 3 cannot be a tautology
reported as a measurement. Two implementation notes. **`w` is 1.00 on all 106 LOO
folds** (oracle-on-all 1.00, no fitting premium), and at w=1.00 the hybrid term is
annihilated, so the arm reduces to a **stable partition** of the hybrid top-50 by
containment — the class implements the partition directly and
`tests/retrievers/test_lexical_containment.py` pins it against a *transcription* of
`fuse_grid` (not an import of it, which would let the test agree with itself), plus
the w=0.00 end that must be plain hybrid, so a future re-run picking w<1 cannot
silently decouple the shipped class from the measured arm. And **`contains_phrase`
moved to `src/rag_lab/text_match.py`**: it lived in `tools/eval/`, which the core
package must not import (ADR-0001), and two copies of a matching rule would diverge
the way two copies of RRF would. The move was checked in both directions —
16,084,108 haystack×needle pairs with **0** disagreements (the sole intended change
is an empty needle now returning False instead of True, unreachable from any caller),
and re-running `audit_gold_anchor_ambiguity.py` moved **only** its `union จริง`
column, which is retrieval and belongs to rebuild #4; every containment-derived
column is byte-identical. Cost to state with it: `detect_entities` ~100 ms/query
(the course matcher is ~75 ms of that) plus fetching 50 instead of 10, i.e. ~+20% on
a 475.6 ms routed query and **no GPU**. **Nothing defaults to it** — `dense`/`hybrid`
ship unchanged and this is opt-in by name, the same rule `qdrant_hybrid` follows.
**Read it with the circularity**: the `person`/`program`/`faculty` qrels were
themselves derived by string containment, so this arm is closer to the labelling
generator than to "relevance" — defensible to ship only because the corpus owner's
domain judgement is that for this query shape relevance genuinely requires the entity
to appear, and never citable as *lexical beats learned ranking*.
**BARE-FIELD MATCHING (2026-08-20) — the deployment gap arm L′ opened, and the
rule that was REJECTED is worth more than the one that shipped.** A person types
the field (`วิศวกรรมคอมพิวเตอร์`), not the 60-character canonical
(`หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมคอมพิวเตอร์`), so `match_programs`
found nothing, `classify_query` returned `unmatched`, and arm L′ silently degraded
to plain hybrid on exactly the queries a deployment sees most.
`match_programs_by_field` resolves a bare field to **every** programme offering it
— all four for that field, never a guess at the degree level, which would be the
degree-swap error `match_programs`' own 2026-08-11 guard exists to prevent. Wired in
three places, each deliberately scoped: `detect_entities(include_field_matches=...)`
**OFF by default** (it also feeds `entity_lookup`/`EntityFilter`, whose published
numbers were measured without it), **ON** in `LexicalContainmentRetriever`, and a
**last** branch in `classify_query` before `ROUTE_UNMATCHED`.
**Four things measured rather than argued.** (1) **The branch's last position is
load-bearing and was placed on evidence**: 0 of the 106 Gold queries reach it, so it
cannot move a published routing number by construction — and **5 of the 13 faculty
queries contain a programme field inside their faculty name**
(`คณะบริหารธุรกิจ` → the 3 `บริหารธุรกิจ` programmes), so anywhere above
`match_faculties` it would steal them. (2) **The same 5 queries caught a real defect
one layer down, by running a claim this file had already written down as
measured.** `detect_entities`' fallback was first gated on `not programs`, and the
retriever's own docstring asserted "changes nothing on the 106 Gold queries" —
**false**: it fired on those 5, widening an already-resolved faculty query and
moving arm L′'s published number silently. It now fires only when the query resolved
to **nothing at all**, which is the case the feature exists for and makes the claim
true **by construction** (all 106 detect something), pinned in both directions.
[[feedback_an_asserted_invariant_is_not_a_check]] again, in a docstring I wrote the
same hour. (3) **`programme_groups` collapses 253 dictionary entries → 250**, exactly
the 3 KOSEN associate-degree renames (`วิศวกรรมคอมพิวเตอร์`,
`วิศวกรรมแมคคาทรอนิกส์`, `วิศวกรรมไฟฟ้าและอิเล็กทรอนิกส์`), where the degree title
was renamed in 2568 — `2569/3` amends *"ฉบับปี พ.ศ. ๒๕๖๗"*, i.e. the very curriculum
`2567/2` approved under the older name. The discriminator needs **two** signals, not
one: disjoint years alone flagged 5 pairs of which **3 were spurious** (a master's
and a doctorate in one field simply do not co-occur in a 6-year window), so a rename
also requires one degree name to be the other **extended**. (4) **The rejected rule,
and it must stay rejected**: the symmetric-looking *same degree, one field extends
the other* branch collapsed **28** entries including `วิศวกรรมไฟฟ้า` with
`วิศวกรรมไฟฟ้าสื่อสารและเครือข่าย` and `ภาษาญี่ปุ่น` with `ภาษาญี่ปุ่นธุรกิจ` —
**a longer field name is normally a DIFFERENT programme**, the prefix-group problem
`program_loader`'s own docstring opens with. It was caught by **reading the groups,
not the count**, and the one real case it was written for is a single 2566 manifest
title that dropped `วิศวกรรม` from `สาขาวิชาวิศวกรรมแมคคาทรอนิกส์` — a typo costing
one `count=1` entry, not worth a rule that cannot tell it from a real programme.
**It was left standing as "a manifest typo worth fixing in the data" until
2026-08-21, and that reading was wrong**: `2566/ครั้งที่ 9`'s own minutes print the
agenda line as `หลักสูตรอนุปริญญา สาขาวิชาแมคคาทรอนิกส์` verbatim (checked in the
body; the other **4** titles of that programme across the corpus all carry
`วิศวกรรม`), so the manifest is a **faithful transcription** and the typo is the
source document's. That closes the item rather than queueing it: editing the title
would make our metadata disagree with the document it points at — the *inverse* of
the 2026-08-08 mispairing repairs, which fixed titles that disagreed with their own
file — and would move a `resolution_id`, i.e. a relabel across 40+ indices and ~24k
results, to buy one dictionary entry. **There is therefore no cheap data fix that
removes the rejected rule's motivation**, which is one more reason it stays
rejected. So
the collapse is **7 entries → 4, not the 7 → 3 asked for**, and the difference is
stated rather than quietly delivered. `tests/test_program_field_matching.py` pins
every negative, and was **verified to fail on the rejected rule** before being
trusted (reinstating it fails exactly the 2 tests written to forbid it). **The Gold
set is structurally unable to score any of this** — all 30 program queries name a
full canonical — so it is a deployment fix, and no retrieval number may be claimed
for it in either direction. **No report emits any count in this paragraph, so
`tests/test_program_field_matching.py` IS their source** — 253→250, the 3 KOSEN
pairs, the 0-of-106 and the 5-of-13 are each pinned by a named test and re-derived
on every `pytest` run, which is the same rule the rebuild-#4 combo count follows
(derived from the artifact, never copied into a snapshot report that would then
rot). The `program_loader.py`/`router.py` edits also tripped `D4` on four reports;
all four are cleared **by content, not by pair** — `doc_claims_allowlist.yaml`'s
`inputs` section now takes an optional `src_sha`, and the exemption holds only
while the source still hashes to it, so a future matcher repair re-flags rather
than inheriting today's clearance. That mattered here: the
`program_loader → relation-graph.md` edge exists *because* a matcher repair moved
that report twice without touching its generator, and a permanent pair-keyed
exemption would have disarmed it. `D6` now audits the section, and the mechanism
was exercised in the failing direction before being trusted (a one-byte change to
the loader re-flags all three pairs and marks all three entries dead).
**That happened for real on 2026-08-21, on a COMMENT-ONLY edit, and the right
discharge turned out to be deleting all three.** `D4` went red on the three reports
and `D6` reported the three entries dead — the mechanism working — and the fix was
to re-render each report (absorption and tag-regeneration **byte-identical**,
relation-graph identical but for its timestamp line), after which the pairs pass on
their own merit and need no exemption at all. **An exemption's strongest state is
not existing**: re-stamping `src_sha` would have pre-armed the *next* edit with a
clearance nobody had earned. `tests/tools/test_audit_doc_claims.py` now pins that
these three pairs are **not** exempted, plus the standing rule that any future
`program_loader` entry must be content-keyed.
**REFRESHED AGAINST REBUILD #4 (2026-08-20) — pools re-minted, model retrained, and
two defects in the harness came out of it that matter more than the numbers.**
(1) **The training pools had no way of naming the indices they came from.** The
builder's `input_fingerprint()` covers dicts/tags/manifests — the **label** side, which
is all a minted candidate's *labels* depend on — but a pool is a **retrieval result**,
so rebuild #4 left the 2026-08-12 `train_pools.json` stale with every fingerprint field
unmoved and nothing on disk saying so. It now records `docset_hash` per routed index
(read from the index's own `manifest.json`, **not** from `IndexRef.provenance`, which
`discover_indices` leaves `{}` — recording that would have written four rows of `None`
and looked like provenance), and `train_hybrid_reranker.py` gates on it as **C6**, with
*no provenance recorded* classified as **unmeasured**, never as current
([[feedback_undefined_is_not_zero]], [[feedback_identify_the_artifact_not_rename_it]]).
(2) **C2 was asserting tie order and had to be rewritten** — the third instance of that
shape here, after BM25 and dense ([[feedback_exactness_is_a_claim_about_scores_not_tie_order]]).
Written as *the same top-K ids* it FAILED on 1 of 3 probe pools at
`max |sigmoid(logit) − ST score| = 1.23e-06`; diagnosed from the artifact first, this
path scored the two divergent candidates **2.0239624977 and 2.0239624977, a gap of
exactly 0.000e+00**, while sentence-transformers separated them by **2.98e-07** of its
own float noise and `argsort` broke the tie by index. **The rule is now: score both
delivered sets under both paths and require the sorted score vectors to agree** —
0.000e+00 here and 2.980e-07 under ST, i.e. the sets are *interchangeable*, which is
sharper than "the scores are close"; `pools ordering identically` is demoted to a
descriptive column, as `agree@10` was. `_TIE_TOL = 1e-5` is **not** a number chosen to
clear the failure: this file had already measured that fp32 at batch 16 vs batch 8
moves values ~6e-6 through BLAS reduction order alone. The rewrite was **exercised in
both directions before the retrain was launched** (passes the real pools, FAILS a
monkeypatched non-tied disagreement at 3.27) — a check relaxed to make today's failure
pass, and never shown to still fail anything, is not a check. **Five more frozen
literals** in `reranker_trained_test.py` (`PUBLISHED`, incl. truncate-and-replace 0.6000
→ 0.5987) are now parsed from `reranker_rrf_routed_test.md`.
