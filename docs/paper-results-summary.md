# Paper results summary (data reference)

Clean, citation-ready numbers for RQ1/RQ2 of the paper (see
`docs/research-framework-gap-analysis.md` for how these research questions
were scoped against the user's draft notes, `Embedding โมเดล.docx`). This
file is a **consolidated reference**, not a process log — for the full
narrative of how each result was reached (dead ends, bugs found, decisions
made), see `docs/chunker-embedder-comparison-log.md`. Update this file
whenever a new result changes a headline number; keep the underlying log
as the append-only record of how we got there.

All numbers below are from the **73-deterministic Gold query set**
(`config/eval/gold_query_set_73det.yaml` — 30 program + 30 person + 13
faculty_adjunct_aggregate queries, entity-anchored, hand-rephrased away from
document title wording). Do **not** cite numbers from the 252-entry set
(`gold_query_set.yaml`) — it's diluted with 179 thematic queries that have
near-zero discriminative power (see
[[project_thematic_query_bootstrap]] / `docs/chunker-embedder-comparison-log.md`)
and materially changes rank order. **Why they had no discriminative power is now
known, and it was not a property of thematic retrieval** (2026-07-30): all 179
were meeting-scoped — each entry's gold ids come from exactly one meeting — yet
every one asked about "ในการประชุม**ครั้งนี้**" without ever naming the meeting, so
no retriever could tell which of ~120 meetings was meant, and every other meeting
that discussed the same theme scored as a miss. The queries were unanswerable as
posed, so their scores were noise rather than a measurement of chunking.
`tools/eval/qualify_thematic_queries.py` rewrote all 179 to name their meeting
(the identity was already in each entry's own gold ids, so nothing was
re-judged), and **the re-eval has now been run**
(`tools/eval/run_thematic_eval.py`, 179 queries × 36 combos, dense; report
`data/results/thematic_eval.md`). It changes the reason to keep them separate:

**the thematic queries do not carry *no* signal — they carry signal that points
the opposite way on the chunker axis.** Same 36 combos, same metric, only the
query shape differs (`tools/eval/thematic_vs_deterministic.py`):

| | thematic (179) | entity-anchored (106) |
|---|---|---|
| mean diff, fixed_size − semantic | **+0.0258** | **−0.0363** |
| exact ties on that pair | 997/1611 = 62% | 430/954 = 45% |
| chunker order (dense recall@10) | recursive 0.412 > sentence 0.387 > fixed_size 0.377 > **semantic 0.351 (worst)** | **semantic 0.366 (best)** > recursive 0.346 > fixed_size 0.330 > sentence 0.324 |
| top embedders | qwen3 0.495 > **e5 0.488** > congen 0.454 > qwen3_0.6b 0.436 | **qwen3_0.6b 0.527** > qwen3 0.479 > jina_v5 0.414 > bge_m3 0.409 |

So the old description — "dilutes with near-zero-signal queries, and materially
changes rank order" — was right about the symptom and wrong about the mechanism:
folding the sets together **cancels two opposing real effects** rather than adding
noise to one. The retired t=0.02 / mean-diff +0.0004 figure was measuring
unanswerable questions; the rewritten queries give a pooled mean diff 64× larger
and consistently signed. Per-pair discrimination is still weak (**2 of 27**
fixed_size-vs-semantic tests significant after Holm, both `bge_m3`, ties still
62%), so these queries remain individually low-powered — but they are no longer
evidence-free, and `semantic` being *worst* here is independent support for
retiring the "semantic chunking wins" headline (#13).

`e5` rising from 5th to 2nd and `congen` from 7th to 3rd also says embedder choice
is **query-shape dependent**, which no cited number currently reflects.

### The BM25/hybrid arms of the thematic set reverse this project's most robust finding

BM25-alone and hybrid were then run over the same 179 queries and put through the
**same** significance machinery — literally the same script, `--thematic` pointing
it at the thematic result dirs (`hybrid_significance_test_9way.py`; the default
73-det run was verified byte-identical afterwards, so the reversal below is not a
methodology artifact). Report: `data/results/thematic_hybrid_significance_test.md`.

**BM25 is much weaker on this query shape**: 0.2990 recall@10 aggregate, against
0.4930 on the entity-anchored set where it ties the top dense tier and beats
`bge_m3` outright. That is the person/program mechanism again — thematic queries
contain no name to match exactly, which is the only thing lexical retrieval is
better at.

**Consequence: "hybrid beats dense-alone for every embedder" (26/27 significant)
is entity-anchored-specific and does not generalize.** On thematic recall@10 the
family splits three ways — 3 significant *for* hybrid, 4 ties, **2 significant
against**:

| dense-alone recall@10 | | hybrid − dense | verdict |
|---|---|---|---|
| `m2v` 0.1923 | below BM25 | **+0.0681** | hybrid significantly better |
| `sct` 0.2733 | below BM25 | **+0.0948** | hybrid significantly better |
| `e5_small` 0.2861 | below BM25 | **+0.0407** | hybrid significantly better |
| `bge_m3` 0.3959 | above BM25 | +0.0280 | tie |
| `jina_v5` 0.4193 | above BM25 | −0.0248 | tie |
| `qwen3_0.6b` 0.4356 | above BM25 | −0.0122 | tie |
| `congen` 0.4535 | above BM25 | −0.0221 | tie |
| `e5` 0.4879 | above BM25 | **−0.0449** | hybrid significantly **worse** |
| `qwen3` 0.4946 | above BM25 | **−0.0516** | hybrid significantly **worse** |

The benefit of fusion is **monotone in how far the dense arm sits from the lexical
arm** (r = **−0.921** between dense score and hybrid−dense delta). What lines up
with BM25's own 0.2990 is the **significance boundary, not the sign**: every
embedder scoring *below* BM25 is significantly helped by fusion and no embedder
above it is. The point-estimate sign flips higher — between `bge_m3` (0.3959, still
**+**0.0280) and `jina_v5` (0.4193, −0.0248), OLS zero-crossing 0.401 — so there is
a band above BM25 where fusion is measurably neither helping nor hurting. (Versions
of this section before 2026-08-07 said the sign flipped "almost exactly at BM25's
own 0.2988"; that was never supported by the table beneath it — on the 2026-07-30
numbers the crossing was 0.399 — and is corrected here.) MRR and nDCG@10 give the
same ordering, with `congen`/`qwen3_0.6b` also crossing into significantly-worse.

**This subsumes the m2v/sct "RRF failure case" as a special case of one rule rather
than a quirk of two bad models.** Stated generally, and now measured in both
directions:

> RRF fusion helps the weaker arm and taxes the stronger one. It is worth doing
> when the two arms are comparable in strength, and it damages the better arm
> whenever they are not — regardless of *which* arm is the weak one.

On the entity-anchored set BM25 was the strong arm, so fusion lifted every dense
embedder and hurt only the two dense models weaker than BM25. On the thematic set
BM25 is the weak arm, so the same rule fires in reverse. Hybrid still beats
BM25-alone for 8 of 9 embedders here (`m2v` significantly worse, `e5_small` a tie
on recall) — the asymmetry is real, not an artifact of one direction being easier.

**Practical reading**: hybrid is the right default only where lexical retrieval is
competitive. A router that already classifies queries (`query_service`) could in
principle skip fusion for entity-free queries; that is a hypothesis this result
motivates, not something measured.

**Operational advice is unchanged but better founded**: keep the two shapes
reported separately; never average them. The thematic numbers are citable *as a
separate query shape* now, not as part of the headline comparison.

**Currency**: every number in these two sections was re-run 2026-08-07 against
`chunker_compare_full` rebuild #3 (2026-08-05T07:56) — full dense + BM25 + hybrid
retrieval over all 179 queries, not just a re-score. **0 verdict flips** across the
54 significance cells of `thematic_hybrid_significance_test.md` and the 27 of
`thematic_eval.md`; every effect size moved by less than 0.02 and only p-values
shifted materially. The old caveat that these indices predated the 2026-07-30
corpus fixes is therefore discharged, and it was discharged in the predicted
direction: all 550 distinct gold `resolution_id`s still resolve post-rebuild, so
nothing changed about which documents count as relevant.

**Status**: gap-analysis Tier 1 and Tier 2 (`docs/research-framework-gap-analysis.md`
§8) are both fully closed as of 2026-07-21 — MAP/Precision@k/multi-k, BM25 baseline,
bootstrap+Holm significance testing, cost/latency Pareto table, and the `sct` /
`qwen3_0.6b` embedder additions. Tier 3's RQ3 ablations (normalization,
word-aware segmentation, chunk-size sweep) ran to completion 2026-07-23 — see
"RQ3 ablation results" section below. The cross-encoder reranker item also ran
to completion 2026-07-23 — see "Cross-encoder reranker results" section below
(a significant *negative* result for hybrid, literature-grounded). RQ4
(end-to-end RAG) ran to completion 2026-08-03 and was refreshed against rebuild
#3 on 2026-08-07 — see "RQ4" section below; **Tier 3 is now closed too**. See the
Open items list at the end of this file for what's still outstanding within the
closed tiers.

**Methodology caveat, added 2026-07-23, resolved 2026-07-25 for the
corpus-discovery bug, then superseded by a second, larger rebuild 2026-07-28
(see below)**: every number in this document was originally built from
indices affected by a corpus-discovery bug — the full-corpus indices behind
these results contained ~7-8% chunks from non-resolution files that should
never have been in the corpus. The bug was fixed 2026-07-23, all 4 chunkers
× 9 embedders in `chunker_compare_full` were rebuilt clean and spot-checked
(0 contaminated chunks), and the numbers in the "Hybrid retrieval" and "BM25
lexical baseline" sections were regenerated 2026-07-25 against those clean
indices.

**Second rebuild + a stale-retrieval-cache incident, 2026-07-28/29**: the
36-combo `chunker_compare_full` index was rebuilt again 2026-07-28 for the
unrelated OCR-remediation fix (`docs/llm-ocr-scan-log.md`) — this changes
chunk *text content* (garbled OCR spans repaired), not corpus membership, so
it invalidates the same numbers a second time. `embedder_matrix_9way.py`
recomputes dense-alone retrieval fresh on every run, so the dense-alone
tables below were correctly refreshed automatically — but `run_gold_bm25_eval.py`
and `run_gold_hybrid_eval.py` were not re-invoked, so their persisted results
under `data/results/gold_bm25_73det` / `gold_hybrid_73det` silently kept
their 2026-07-25 mtimes for three days while every downstream BM25/hybrid
significance script kept reading them — comparing fresh dense numbers
against stale BM25/hybrid numbers without anything flagging the mismatch.
Caught by checking persisted-result mtimes against the rebuild timestamp
before trusting an eval run that looked like it had flipped several
conclusions at once (see `docs/chunker-embedder-comparison-log.md`,
"Re-eval หลัง OCR-remediation rebuild" entry, for the full incident and the
process lesson: **after any index rebuild, refresh every retrieval path with
persisted results — dense, BM25, hybrid — not only the one an eval script
happens to recompute automatically**). Both were re-run 2026-07-29 against
the rebuilt indices, along with all 4 downstream significance scripts
(`bm25_vs_embedder_significance_test_9way.py`,
`bm25_vs_embedder_significance_test_per_chunker.py`,
`hybrid_significance_test_9way.py`, `hybrid_significance_test_semantic_top5.py`).
**This refresh did change real conclusions** (unlike the 2026-07-25 one,
which held every conclusion) — see the "Hybrid retrieval" and "BM25 lexical
baseline" sections below for the updated numbers, and in particular the "Top
single-combo across the entire study" claim, which no longer holds as
previously stated. Nothing in this document still cites pre-2026-07-29
BM25/hybrid numbers. `cost_latency_pareto.py` was long the one exception — its
08-07 re-run against rebuild #3 had its timings rejected as contaminated, so it
carried a split provenance (08-07 quality, 07-29 latency) for two days. **That
split is retired: re-measured 2026-08-09 on an idle machine, one subprocess per
embedder, with three timing controls, so both halves are now one run.** Note
its BM25/hybrid latency columns are *not* comparable with any run dated
2026-07-29 or earlier, which predate `5cc71a1` and are high by ~1s per query.
Both the reason and the evidence are in that section's currency banner.

## Superseded 2026-08-18 by rebuild #4 — read this before citing any retrieval number below

Rebuild #4 (finished 2026-08-17) put the 2026-08-09 re-OCR of `2566/ครั้งที่ 3`
into all 40 `chunker_compare_full` combos, and the persisted BM25/hybrid results
plus every seconds-level significance test were re-run on 2026-08-18. **27 of the
78 reports in `data/results/` are dated 2026-08-18 or later; the other 51 predate
the rebuild.** The sections below carry their original dates and are left as
written — this section says what moved. Anything not listed here held.

**Four verdicts changed, and two of them change what the paper can claim.**

| claim | before | after (2026-08-18) |
|---|---|---|
| hybrid `routed (shipped)` vs best single combo | +0.0549, Holm 0.0672 — ns | **+0.0581, Holm 0.0480 — significant** |
| hybrid `routed (loo)` vs best single (loo) | +0.0499, Holm 0.0780 — ns | **+0.0825, Holm 0.0000 — significant** |
| dense `routed (oracle)` | significant on all 3 metrics | **nDCG@10 only** (+0.0744, 0.0126) |
| soft routing (`B vs A`, nDCG@10) | +0.0360, Holm 0.0216 — significant | **+0.0333, Holm 0.0528 — ns** |

So **the headline "routing matches but does not beat a well-chosen single index"
is now false under hybrid and still true under dense.** The section
"Query routing — … and it ties the best single combo" below is superseded on
exactly that point; its coverage result (5 routes, 0/106 unrouted) is unchanged,
and so is the 5-route-vs-3-route margin (+0.0958 dense recall@10, Holm 0.0000).
Read the new claim as: **beats a well-chosen single index under hybrid, matches
it under dense, and closes a 43% coverage hole either way.**

**Most of that gain is the baseline falling, not the router rising, and that
belongs in the sentence.** The hybrid best-single combo is `sentence ×
qwen3_0.6b` before *and* after — its identity did not change — but it fell 0.6281
→ **0.6229** while the routed arm fell only 0.6831 → **0.6811**, taking the margin
0.0549 → 0.0581 across a bar it had been sitting just under (Holm 0.0672 →
**0.0480**). And **soft routing no longer owns a
significant cell anywhere**, which weakens the cost-per-point argument for it
without refuting "soft is at least as good" (`B vs C` is still ns; the CI now
rules out soft beating hard by more than 0.0060 recall@10, tighter than the 0.0156
quoted below).

**Levels that moved (cite these, not the ones in the sections below).**

| quantity | before | after |
|---|---|---|
| unrouted hybrid, best single combo | 0.6281 | **0.6229** |
| hard routing, `routed (shipped)` | 0.6831 | **0.6811** |
| hard routing, `routed (loo)` | 0.6780 | **0.6794** |
| soft routing (arm B, LOO) | 0.6631 | **0.6510** |
| both (arm D, LOO) | 0.6629 | **0.6648** |
| oracle-union floor — pairs no arm reaches at k=10 | 84 (8.0%) | **91 (8.7%)** |
| hybrid-union ceiling (36 combos, 360 docs) | 0.8948 | **0.8916** |
| BM25 alone, aggregate recall@10 | 0.4930 | **0.4863** |

**RQ3 re-evaluated 2026-08-20: 0 verdict flips.** Its 4 treatment indices turned
out to be part of rebuild #4's 40 (rebuilt 2026-08-17, carrying the same
`docset_hash` as both baselines), so the confound this document warns about never
opened and only the eval was owed. Every RQ3 claim stands; three point estimates
moved and are refreshed in `CLAUDE.md`: the dense nDCG@10 256-vs-1024 near-miss is
now Holm **0.0948** (was 0.0828, still ns), 256-vs-512 dense recall@10 is 0.4117 vs
**0.4139** at Holm **0.9338** (still a flat tie), and 256's one win — hybrid
recall@10 over 512 — strengthened to +**0.0533** at Holm **0.0076**.

**Three smaller movements worth knowing.** (1) `bge-m3` came **back into** the
semantic top-5 tie cluster on MRR (its closest cell, vs `qwen3`, is Holm 0.0940)
and against `jina_v5`/`e5_small` on recall@10 — it is now separated only from the
two `qwen3` models, on 2 metrics of 3, so "clearly outside the cluster on every
metric" is withdrawn. (2) `qwen3_0.6b` beating `Qwen3-Embedding-4B` dense-alone
went **ns on recall@10** (+0.0475, Holm 0.0592) while staying significant on MRR
and nDCG@10 — a near-miss, not a reversal. (3) **The first *aggregate* cell where
a dense embedder significantly beats BM25 outright**: `bm25 − qwen3_0.6b` MRR
−0.1249, Holm 0.0072. Until now that had only ever happened in one per-chunker
cell, and that cell (`qwen3_0.6b` under `semantic`) strengthened from one metric
to all three.

**Alpha sweep re-run 2026-08-20 — the per-`entity_type` alpha result is now
nDCG-only, and the section below is superseded on that point.** Its headline was
"+0.0350 recall@10 / +0.0360 nDCG@10, both surviving leave-one-out (Holm-adj
0.0252 / 0.0210, m=9)". Against rebuild #4, on the same combo
(`sentence × qwen3_0.6b`), the citable `per-type (loo)` arm reads:

| metric | before | after |
|---|---|---|
| recall@10 | +0.0350, Holm 0.0252 — **significant** | +0.0281, Holm **0.0870** — ns |
| nDCG@10 | +0.0360, Holm 0.0210 — significant | +0.0333, Holm **0.0392** — significant |
| MRR | ns | +0.0369, Holm 0.5016 — ns |

**So "+0.0350 recall@10" is withdrawn.** What survives is *a per-type alpha
reorders the top-10 better without putting more gold into it* — which is exactly
what an nDCG-only win means, and a narrower claim than the one this document
made. The oracle `per-type best` arm is significant on all three (+0.0456 /
+0.0547 / +0.0560, MRR newly so), but an oracle is a ceiling, not a system.

Three parts of the section held. A single **global** alpha is still worth nothing
(+0.0066 / +0.0217, both ns, both oracle). The **disjoint plateau** finding
survived and the gap widened — `person` best 0.15 (plateau 0.00–0.35) against
`program` best **0.70** (plateau **0.45**–1.00), so the shipped 0.50 still sits
outside `person`'s range. And `fixed_size × m2v` still wants alpha=0.00 outright,
with per-type adding only **+0.0110** over global.

**Nothing about the ship decision changes** — per-`entity_type` alpha was already
not wired, on the grounds that its motivating gain was measured against *no*
routing and vanishes against the hard router that ships. A motivating gain that
has itself gone ns on recall@10 only strengthens that.

**One casualty worth recording: the family-size trap paragraph lost its
example.** It contrasted Holm-adj 0.0252 (m=9, the sweep) with 0.1960 (m=12,
soft-vs-hard) on identical data — same difference, larger family, opposite
verdict. Post-rebuild that is **0.0870 vs 0.1960, ns at both**. The rule stands
(a Holm p is a property of its family, not of the pair — always quote m), but
there is no live illustration of it in these reports today.

**Not yet re-run against rebuild #4, so every figure in these sections is
pre-rebuild-#4:** both `fetch_depth` sweeps
(the *routed* one **is** current — it is the one the ship decision rests on), the
whole reranker family including the trained cross-encoder, HyDE, ColBERT, the
Qdrant pilot and concurrency runs, `gold_anchor_ambiguity`,
`residual_relevance`, and both `gold_entity_*` reports.

## Resolved 2026-07-23: RQ3 ablation results (normalization / segmentation / chunk-size)

Full-corpus builds + paired bootstrap (n_boot=10,000, seed=42, Holm-Bonferroni
correction), Gold 73-det set. Config/eval scripts: `config/experiments/rq3_*.yaml`,
`tools/eval/rq3_*_significance_test.py` / `rq3_chunksize_sweep_report.py`. Raw
tables: `data/results/rq3_normalize_significance_test.md`,
`data/results/rq3_segmentation_significance_test.md`,
`data/results/rq3_chunksize_sweep_report.md`.

- **Normalization** (Thai digit + `pythainlp.util.normalize()`, on `semantic ×
  bge-m3`): **no significant effect on any metric** (Holm-adj p ≥ 0.414 across
  dense + hybrid × {recall@10, MRR, nDCG@10}). Raw diffs are small and
  inconsistent in sign across metrics — not citable as either a help or a harm.
- **Segmentation** (word-aware `newmm`-boundary chunking vs. raw-character
  slicing, on `fixed_size(512) × bge-m3`): **no significant effect on any
  metric** (Holm-adj p = 1.0 across the board). Chunk count/length are nearly
  identical between arms (61,766 vs 62,018 chunks; mean length 447.4 vs 438.9
  chars) — snapping chunk boundaries to Thai word edges does not measurably
  change retrieval quality on this corpus.
- **Chunk size** (256 / 512 / 1024, `fixed_size × bge-m3`): **the one ablation
  with a real, significant effect** — smaller chunks significantly beat larger
  ones on recall@10: 256 > 1024 (dense Holm-adj p=0.0012, hybrid p=0.0006),
  512 > 1024 (dense p=0.0132), 256 > 512 (hybrid p=0.0056). Recall@10 declines
  monotonically with chunk size (dense: 256=0.510, 512=0.480, 1024=0.395;
  hybrid: 256=0.661, 512=0.607, 1024=0.570). MRR/nDCG@10 differences are mostly
  not significant after Holm correction — the effect is concentrated in
  recall@10, not ranking quality among already-found results.

**Headline for the paper**: of the three RQ3 preprocessing variables tested,
only **chunk size** has a demonstrated, significant effect on this corpus
(smaller is better for recall) — Thai-specific normalization and word-boundary
segmentation do not move retrieval quality significantly at the 512-token
scale already used throughout the rest of the study.

### Refreshed 2026-07-29 — treatment indices rebuilt to remove a clean-vs-dirty confound

The three RQ3 treatment-side indices were built 2026-07-23, **before** the
kernel-A OCR remediation (completed 2026-07-27) and the `chunker_compare_full`
rebuild (2026-07-28). Their baseline arm reuses combos *from*
`chunker_compare_full` (`plain__fixed_size__local__ceea7536`,
`plain__semantic__local__8aae9bcd`), which had since been rebuilt on cleaned
text — so every RQ3 comparison was silently pitting a **clean baseline against
a dirty treatment**. That is a genuine methodological confound, not mere
staleness, and unlike the other 2026-07-29 refreshes it could not be fixed by
re-running an eval script: all three treatment indices had to be rebuilt on
GPU. Done 2026-07-29 (`rq3_segmentation_ablation`, `rq3_chunksize_sweep`,
`rq3_normalize_ablation`), then all three significance scripts re-run.

- **Normalization — conclusion unchanged.** Still no significant effect on any
  metric (Holm-adj p ≥ 0.42; closest is dense nDCG@10, raw p=0.0700 →
  Holm-adj 0.4200). Every diff is negative-leaning on the dense side
  (−0.012 to −0.026) and essentially zero under hybrid — same "small and
  inconsistent" picture as before.
- **Segmentation — conclusion unchanged, but no longer p=1.0 across the
  board.** Still nothing significant (Holm-adj p ≥ 0.4524), but two cells
  moved off the floor: dense MRR now +0.0398 (raw p=0.0754) and hybrid
  recall@10 +0.0183 (raw p=0.1054), both in favour of word-aware boundaries.
  Not citable as an effect — but the honest 2026-07-29 framing is "no
  detectable effect", not the stronger "identical to the third decimal" the
  old all-p=1.0 table implied. Chunk stats stay near-identical between arms
  (58,655 raw-char vs 58,198 word-aware; mean length 437.0 vs 444.0 chars),
  confirming the boundary change isn't secretly a size change.
- **Chunk size — the significant effect survives, but "smaller is
  monotonically better" does NOT.** What replicates robustly is the **1024
  penalty**: 1024 loses significantly to both 256 and 512 on dense recall@10
  (Holm-adj p=0.0020 / 0.0000), hybrid recall@10 (p=0.0000 / 0.0028), and
  hybrid nDCG@10 (p=0.0072 both). What does **not** replicate is the 256-vs-512
  ordering: on **dense** retrieval 512 is now *numerically ahead* of 256
  (recall@10 0.4146 vs 0.4103, diff −0.0043, p=0.8802 — a flat tie), reversing
  the old 0.510 > 0.480 gap. 256 only beats 512 significantly on **hybrid
  recall@10** (+0.0509, Holm-adj p=0.0112), and not on hybrid nDCG@10
  (p=0.4094) or any MRR cell (nothing significant anywhere on MRR, same as
  before).

**Revised RQ3 headline (2026-07-29)**: chunk size remains the only RQ3
variable with a demonstrated effect, but the citable claim is narrower than
the 2026-07-23 version. **Cite: "1024-char chunks are significantly worse than
both 512 and 256 on recall@10 and nDCG@10." Do not cite: "recall declines
monotonically with chunk size" or "256 is the best setting"** — 256 and 512
are statistically tied on dense retrieval (with 512 numerically ahead), and
256's advantage exists only under hybrid recall@10. The project's default of
512 is therefore *not* shown to be suboptimal by this refresh; only 1024 is
shown to be a mistake. Normalization and segmentation conclusions are
unchanged.

### Refreshed 2026-08-05 — fourth rebuild, after `chunker_compare_full` rebuild #3

The same confound reopened: `chunker_compare_full` rebuild #3 completed
2026-08-05T07:56 (see `project_index_rebuild_pending` memory), leaving the
RQ3 treatment indices — last rebuilt 2026-07-29 — pointing at a
since-superseded baseline again. All three treatment indices were rebuilt a
fourth time (`data/logs/run_rq3_rebuild_2026_08_05.sh`, ~2.5h, exit=0, no
restarts) and all three significance scripts re-run. Every 2026-07-29
conclusion held; numbers moved slightly (same gold set, freshly rebuilt
text), not qualitatively:

- **Normalization — unchanged.** Still nothing significant (Holm-adj
  p ≥ 0.335; closest is dense MRR, raw p=0.0796 → Holm-adj 0.398).
- **Segmentation — unchanged.** Still nothing significant (Holm-adj
  p ≥ 0.264; closest is dense MRR +0.0458, raw p=0.0440 → Holm-adj 0.264 —
  the same cell that was closest in the 2026-07-29 refresh, same
  direction, still doesn't survive correction).
- **Chunk size — 1024 penalty replicates again, one new nuance.** 1024
  loses significantly to 512 on dense recall@10 (Holm-adj p=0.0000), dense
  nDCG@10 (p=0.0354), hybrid recall@10 (p=0.0008), hybrid nDCG@10
  (p=0.0006); loses to 256 on dense recall@10 (p=0.0016), hybrid recall@10
  (p=0.0000), hybrid nDCG@10 (p=0.0032). **New nuance**: 256-vs-1024 on
  dense nDCG@10 is now a near-miss, Holm-adj p=0.0828 (raw p=0.0414) — not
  significant, so "256 beats 1024 on every dense metric" is no longer
  accurate; 256's edge over 1024 is recall@10-only on the dense side.
  256-vs-512 replicates as a flat tie on every dense metric (recall@10
  0.4117 vs 0.4129, Holm-adj p=0.9676) and on hybrid MRR, and **256 again
  significantly beats 512 only on hybrid recall@10** (+0.0481, Holm-adj
  p=0.0154 — same single cell as 2026-07-29's +0.0509/p=0.0112).

**RQ3 headline is unchanged from 2026-07-29, and is now current against
rebuild #3**: chunk size is the only RQ3 variable with a demonstrated
effect. **Cite: "1024-char chunks are significantly worse than 512 and 256
on recall@10; also worse than 512 on nDCG@10." Do not cite: "recall
declines monotonically with chunk size," "256 beats 1024 on every metric,"
or "256 is the best setting"** — 256 and 512 remain statistically tied on
every dense metric, and 256's only proven edge over 512 is hybrid
recall@10. If `chunker_compare_full` is rebuilt again, treat these numbers
as stale again until re-run — this is now the third time this exact
confound has reopened and been closed (2026-07-29, 2026-08-05, pattern
per `feedback_refresh_all_retrieval_paths_after_rebuild`).

## Resolved 2026-07-23: Cross-encoder reranker results — a significant negative result for hybrid

Gap-analysis Tier 3, item 8. Built a `CrossEncoderReranker` stage
(`BAAI/bge-reranker-v2-m3`, LoRA-tuned on the `bge-m3` backbone) that
re-scores a widened retriever candidate pool (`rerank_pool_size=50`) and
truncates to the final `k=10`. Because retrieval is deterministic, the
no-rerank baseline is exactly the top-10-by-retriever-score slice of the
treatment's 50-candidate pool, so the paired diff isolates the reranker's
re-ordering effect as the only variable. Evaluated on the semantic×bge-m3
combo (`plain__fixed_size__local__ceea7536`), Gold 73-det set, paired
bootstrap (n_boot=10,000) + Holm-Bonferroni correction. Script:
`tools/eval/reranker_significance_test.py`; raw table:
`data/results/reranker_significance_test.md`.

| Retriever reranked | Metric | No-rerank → Reranked | Holm-adj. p | Direction |
|---|---|---|---|---|
| Hybrid (BM25+dense, RRF) | MRR | 0.848 → 0.760 | 0.006 | **significantly worse** |
| Hybrid (BM25+dense, RRF) | nDCG@10 | 0.675 → 0.617 | 0.030 | **significantly worse** |
| Hybrid (BM25+dense, RRF) | recall@10 | 0.607 → 0.584 | 1.000 | worse, not significant |
| Dense-alone (bge-m3) | recall@10 / MRR / nDCG@10 | — | n.s. both directions | no effect |

Reranker latency (call alone, model load excluded, `rerank_pool_size=50`,
73 queries): **p50 1191ms, p95 1522ms, mean 1259ms per query** — not cheap,
on top of the finding being negative for hybrid.

**Refreshed 2026-07-29** against the OCR-remediation-rebuilt index (found
during a full `data/results/*` staleness sweep — this script re-retrieves
live against `plain__fixed_size__local__ceea7536` rather than reading
cached results, so it was one rebuild behind like several other scripts
that day). New numbers, same 106-query Gold set, same methodology:

| Retriever reranked | Metric | No-rerank → Reranked | Holm-adj. p | Direction |
|---|---|---|---|---|
| Hybrid (BM25+dense, RRF) | MRR | 0.7775 → 0.6775 | 0.0048 | **significantly worse** |
| Hybrid (BM25+dense, RRF) | nDCG@10 | 0.6193 → 0.5908 | **0.5676** | worse, **no longer significant** |
| Hybrid (BM25+dense, RRF) | recall@10 | 0.5570 → 0.5683 | 0.6456 | *better*, not significant |
| Dense-alone (bge-m3) | recall@10 / MRR / nDCG@10 | — | n.s. both directions (0.315–0.568) | no effect |

Reranker latency, refreshed: p50 1170ms, p95 1425ms, mean 1227ms — essentially
unchanged (latency measures the reranker model's own compute, not corpus
content).

**One real finding-level change, not just numbers moving**: the hybrid
**nDCG@10** loss is **no longer statistically significant** (Holm-adj
p=0.030 → 0.5676) — only hybrid **MRR** still is. Hybrid recall@10 even
flips sign (was −0.023, now +0.011), still nowhere near significant either
way. **Revised headline: cross-encoder reranking significantly hurts
hybrid MRR; the nDCG@10 harm reported on 2026-07-23 did not replicate
against the OCR-remediation-rebuilt index and should be retired as a
separate claim.** The MRR-only framing is still consistent with the
"phantom hits" literature mechanism cited below (early-rank disruption
without necessarily evicting relevant docs from the top-10) — if anything
it sharpens that story, since nDCG@10 (which weights the whole top-10, not
just rank-of-first-hit) no longer moves significantly while MRR (purely
rank-of-first-hit) still does.

**Refreshed again 2026-08-05** against `chunker_compare_full` rebuild #3
(this script is not in the persisted-results refresh chain — it re-retrieves
live — so it needed a separate manual re-run, see
[[feedback_refresh_all_retrieval_paths_after_rebuild]]). Same 106-query Gold
set, same methodology:

**Refreshed a third time 2026-08-18** against rebuild #4. Verdicts unchanged;
the numbers below are the current ones, with the rebuild-#3 pair beside them.

| Retriever reranked | Metric | No-rerank → Reranked | Holm-adj. p | rebuild #3 | Direction |
|---|---|---|---|---|---|
| Hybrid (BM25+dense, RRF) | MRR | 0.7730 → 0.6940 | **0.0240** | 0.7814 → 0.6778, p=0.0012 | **significantly worse** |
| Hybrid (BM25+dense, RRF) | nDCG@10 | 0.6195 → 0.5909 | 0.5442 | 0.6257 → 0.5879, p=0.2840 | worse, not significant |
| Hybrid (BM25+dense, RRF) | recall@10 | 0.5558 → 0.5649 | 0.7112 | 0.5598 → 0.5663, p=0.7974 | *better*, not significant |
| Dense-alone (bge-m3) | recall@10 / MRR / nDCG@10 | — | n.s. all three (0.3270–0.5442) | n.s. (0.284–0.419) | no effect |

The MRR harm has now shrunk at every rebuild (p=0.0048 → 0.0012 → 0.0240)
while staying significant — cite the direction, not the magnitude.

Reranker latency, refreshed 2026-08-18 against rebuild #4: p50 1167.4ms,
p95 1423.4ms, mean 1225.6ms over 106 queries — still essentially unchanged run
to run. (This line had stayed at the rebuild-#3 figures until 2026-08-23; it was
found by `D7`, the check that reads unit-suffixed figures, because no earlier
check could see a latency — it is neither 4-decimal nor a count/total.) **No finding-level change from the
2026-07-29 refresh**: hybrid MRR is still the sole significant casualty,
now at even tighter significance (p=0.0012 vs 0.0048), everything else
stays non-significant. This is now the third consecutive refresh to land on
the same MRR-only conclusion — treat it as settled unless the reranker
model, pool size, or query set changes.

**Confirmed not an implementation bug**: the reranker was smoke-tested in
isolation and scores semantically sensibly (a tuition-fee chunk correctly
ranked highest for a tuition-reduction query). The isolation logic (baseline
= deterministic slice of the treatment's own wider pool) rules out a
confound from pool-size alone.

**Literature-grounded explanation** (full citations and per-question detail
in `docs/reranker-hybrid-interaction-research.md` — written by a dedicated
research pass against primary IR sources, not inferred from our data alone):

1. **Reranking gains shrink as the first-stage ranking strengthens, and can
   flip to net harm.** Rosa et al. ("In Defense of Cross-Encoders for
   Zero-Shot Retrieval", arXiv:2212.06121) show monoT5 lifts BM25
   (avg nDCG@10 0.441→0.496, +0.055) and a stronger dense retriever, GTR-335M
   (0.451→0.496, +0.045) to nearly the *same* post-rerank ceiling — smaller
   absolute gain over the stronger baseline, though never negative in their
   data. Jacob, Lindgren, Zaharia, Carbin, Khattab, Drozdov ("Drowning in
   Documents: Consequences of Scaling Reranker Inference", ReNeuIR 2025 @
   SIGIR 2025, arXiv:2411.11767) supply the sign-flip: reranking with
   **`bge-reranker-v2-m3` by name** — our exact model — "frequently
   perform[s] worse than retrievers when both rank the full dataset," and
   names the failure mode **"phantom hits"**: confidently high scores on
   documents with no lexical or semantic overlap with the query at all. Our
   own result's fingerprint (MRR/nDCG hurt, recall@10 spared) is consistent
   with phantom hits disturbing top-of-list order without necessarily
   evicting relevant docs from the top-10 entirely.
2. **RRF structurally protects exact-lexical-match signal that a
   cross-encoder cannot see.** Cormack, Clarke, Büttcher ("Reciprocal Rank
   Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR
   2009) define RRF as combining "ranks without regard to the arbitrary
   scores returned by particular ranking methods" — a document BM25 ranks #1
   on exact term match is protected in the fused list by construction. A
   cross-encoder scoring only the raw `(query, chunk)` text pair has no
   visibility into which retrieval arm surfaced a candidate or why. (Note:
   the 2009 paper predates dense retrieval and never itself discusses
   lexical+dense hybrid fusion or reranker interaction — this connection is
   this project's own architecturally-grounded inference, not the paper's
   claim.)
3. **Off-the-shelf rerankers may not transfer to a hybrid-fused candidate
   distribution.** Lu, Hall, Ma, Ni (Google Research, "HYRR: Hybrid Infused
   Reranking for Passage Retrieval", arXiv:2212.10528) motivate training a
   reranker specifically on hybrid-retriever candidates because off-the-shelf
   rerankers (trained on a single retriever's negative distribution, commonly
   BM25) don't reliably transfer otherwise. `bge-reranker-v2-m3`'s public
   training mixture (bge-m3-data, Quora, FEVER, per its model card) has no
   documented hybrid-candidate component.
4. **No paper tested our exact pipeline** (RRF hybrid → bge-reranker-v2-m3) —
   this is a genuinely new data point, not a replication of a documented
   result. Thai-specific reranker weakness came up thin/inconclusive in the
   literature search and is not part of the explanation until tested directly
   on this corpus.

**Headline for the paper (updated 2026-07-29)**: cross-encoder reranking
should **not** be applied to this project's hybrid (RRF) retrieval path as
currently wired — it significantly hurts **MRR** (the nDCG@10 harm
originally reported did not replicate on refresh and is retired as a
separate claim, see above), with literature support (same reranker model,
independently observed "phantom hits" against strong baselines) rather than
being a one-off artifact. Reranking remains untested-but-not-contra-
indicated for weaker single-retriever paths (its dense-alone effect here was
null, not harmful). Two literature-suggested follow-up interventions — a
reranker trained/validated on hybrid-fused candidates specifically, or
blending the reranker's score into RRF as a fourth ranked signal instead of a
hard truncate-and-replace step — are untested hypotheses, not implemented.

**SUPERSEDED IN PART — both follow-ups have since been tested, and (a) is
positive. Read the next section before citing the paragraph above.** What
survives it is narrow and still correct: *the off-the-shelf* `bge-reranker-v2-m3`
should not truncate-and-replace this project's hybrid ranking.

## Resolved 2026-08-12: The reranker axis, closed in four measurements — the model was the problem, not reranking

The 2026-07-23 negative result above was reported for one model, in one wiring,
against one baseline. Four follow-ups took each of those apart, and the axis ends
somewhere quite different from where it started. Reports:
`data/results/reranker_rrf_signal_test.md`, `reranker_rrf_routed_test.md`,
`reranker_model_comparison.md`, `reranker_trained_test.md`. Pre-registration and
outcome for the last one: `docs/reranker-trained-on-hybrid-design.md`.

| what varied | result |
|---|---|
| **the wiring** — reranker as a 4th RRF signal instead of truncate-and-replace | beats unrouted hybrid **+0.0392** recall@10 (Holm 0.0108), but **not** the hard router (**−0.0098**, Holm 0.9768 — the point estimate is now *negative*, and the CI rules out its adding more than **+0.0037**) |
| **the model** — 4 qualified cross-encoders on one routed pool | spread **0.0262** recall@10, and the anchor is the **worst** of the four (the only one below the router); the *older* `bge-reranker-v1-large` is best |
| **the ceiling** — oracle over the same routed P=50 pool | pool holds **0.9054**, a perfect selection of 10 delivers **0.8331** = **+0.1520** over the router |
| **the weights** — fine-tuned on hybrid-fused candidates from this corpus | **+0.0730** over the router, **+0.0828** over the off-the-shelf model, all six pre-registered tests Holm **0.0000** |

Read row 2 as *the model is a real variable*, never as a model recommendation: the winner is an
argmax over four models on the same 106 queries, its pre-registered recall@10 family separates 0
of 3, and it is dominated outright by the free control described at the end of this section.
Confirming it on a fresh query set is deliberately **not** owed — the reasons are recorded in
`docs/reranker-hybrid-interaction-research.md`.

**The headline result** (re-run 2026-08-20 against index rebuild #4; the 08-12
originals are named wherever a claim changed). A cross-encoder that starts from the
published anchor's own weights and is fine-tuned for 57 minutes on 506 routed-hybrid
P=50 pools drawn from this corpus — entities disjoint from the eval set, checkpoint
selected on held-out *training* queries — reaches **0.7541** recall@10 against the
shipped router's **0.6811** and the off-the-shelf model's **0.6713**. `T vs D` grew
(+0.0637 to **+0.0828**) mostly because the off-the-shelf arm *fell* at the rebuild
(0.6847 to 0.6713), not because training improved — state the two separately. Because only the
weights vary (pool, routing, fusion, the `w` grid, P, k, metrics and bootstrap are
all held at published values), `T vs D` is a within-model paired before/after: the
difference cannot be attributed to model size, tokenizer or language coverage.
**This is the first reranker intervention in the study to survive the hard router**,
and it explains the earlier nulls rather than contradicting them — `program`, the
route the off-the-shelf model actively *damaged*, is where training pays most
(**+0.1118** over the router, **+0.1562** over the untrained model).

**Two limits to state with it.** The trained model captures **48%** of the oracle's
+0.1520 (off-the-shelf: **−6%**, i.e. it now sits *below* the router), so the axis is
narrowed rather than closed. And `faculty` is the one route that gets *worse*
(**−0.0064**): only one faculty entity survives the disjointness filter, so 13 of 106
eval queries sit on a route the fine-tune never learned — its held-out training
recall for that route is **0.5000** in every epoch including epoch 0.

**The caveat a reviewer will raise, measured in advance.** A pre-registered control
that scores candidates purely by *whether the query's entity string appears in the
chunk* — no GPU, no training — fused through the identical path reaches **0.7438**.
**Rebuild #4 took away the two cells that used to bound this, and that is the one
claim in this section withdrawn rather than restated.** At 2026-08-12 `T vs L` was
significant on MRR (+0.0409, Holm 0.0150) and nDCG@10 (+0.0271, Holm 0.0432); it is
now **not significant on any metric** — recall@10 **+0.0103** (Holm 0.2896), MRR
**+0.0330** (0.1368), nDCG@10 **+0.0288** (0.0930). Every sign is unchanged and the
nDCG effect even grew, so this is **power, not reversal** — but it must now be cited
as a bound: **T beats L by at most 0.0298 recall@10, and L beats T by at most
0.0089**. The control alone still beats the router significantly on all three
(+0.0627 / +0.0623 / +0.0849). Training labels and eval qrels come from the same
string-containment generator (`docs/eval-validity-threats.md` §2), so **recall@10 on
these qrels is largely a containment test** — and the earlier claim that the
fine-tune's separable contribution over that is *ordering, not which documents come
back* **no longer has a significant cell to rest on**. Cite `T vs D` without qualification —
both arms are cross-encoders under the same labelling rule, so it cancels — and never
cite `T vs C` without the control's number beside it.

**The control was given an input no other arm had, and the deployable version of
it is now what ships (2026-08-20).** `lexical_cache` reads the entity out of the
gold query set, so arm L is handed the very string the qrels were derived from
while arms C, D and T see only the query text — it is not merely free of GPU, it
is *better informed*. **Arm L′** removes that asymmetry by recovering the entity
with the shipped extractor (`router.detect_entities`), which returns a different
string on **63 of 106** queries (`person` title-stripped, `course` an 8-digit code
rather than a name). Three results, in their own exploratory Holm family (m=9) so
no figure above moves:

| comparison | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| L′ vs L — the cost of losing the oracle string | −0.0138 (ns) | −0.0186 (ns) | −0.0135 (ns) |
| **L′ vs C — the deployable arm vs the shipped router** | **+0.0489** | **+0.0437** | **+0.0714** |
| T vs L′ — trained model vs the *deployable* control | **+0.0241** | **+0.0516** | **+0.0423** |

Losing the oracle is cheap (CI rules out a loss worse than 0.0304 recall@10), the
deployable arm still beats the router significantly on every metric at zero GPU
cost, and — the correction that matters — **`T vs L′` is significant on all three
where `T vs L` is significant on none.** The claim that the fine-tune is not
separable from string containment therefore holds only against the *oracle-fed*
control; against the deployable one it separates everywhere, so part of what the
training buys is not needing an entity extractor.

**What is wired.** Arm L′ ships as the `lexical_containment` retriever
(`src/rag_lab/retrievers/lexical_containment.py`), opt-in by name; `dense` and
`hybrid` are unchanged and nothing defaults to it. Its leave-one-out weight is
1.00 on all 106 folds, at which the hybrid term is annihilated, so the arm reduces
to a stable partition of the hybrid top-50 by containment. Cost: ~100 ms/query for
entity detection plus fetching 50 instead of 10 (~+20% on a 475.6 ms routed query),
no GPU. **The trained cross-encoder is still not wired** — its ~1.2 s/query buys
+0.0241 recall@10 over an arm that costs nothing.

**A deployment gap the eval cannot see (2026-08-20).** Arm L′ recovers the entity
with `router.detect_entities`, and a person searching types the *field*
(`วิศวกรรมคอมพิวเตอร์`), not the 60-character canonical
(`หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมคอมพิวเตอร์`). The dictionary is
keyed on full canonicals, so such a query resolved to nothing and the arm silently
degraded to plain hybrid. `match_programs_by_field` now resolves a bare field to
**every** programme offering it — never a guess at the degree level — and
`programme_groups` collapses the dictionary's 253 entries to 250 so a caller
counting programmes does not see a renamed one twice.

**No number in this document moves, and that is a structural fact rather than a
null result**: all 30 `program` Gold queries name a full canonical, so the branch
is reached by **0 of 106** queries and `classify_query` returns the identical label
for all of them. Cite it as a deployment fix; **no retrieval claim may be made for
it in either direction**, because the query shape it serves is absent from the
evaluation set. The counts above are re-derived by
`tests/test_program_field_matching.py`, which is their source — no report emits
them.

One measured caveat worth carrying, because it nearly moved a published arm: the
fallback was first gated on "no programme matched", and in that form it fired on
**5 of the 13** `faculty` queries, whose faculty names *contain* a programme field
(`คณะบริหารธุรกิจ` holds `บริหารธุรกิจ`). It now fires only when the query resolved
to nothing at all, which makes "this changes nothing on the Gold set" true by
construction rather than by luck.

Read all of it with the circularity: the `person`/`program`/`faculty` qrels were
themselves derived by string containment, so this arm is closer to the labelling
generator than to relevance. It is defensible to ship because the corpus owner's
domain judgement is that for this query shape relevance genuinely requires the
entity to appear — never because lexical matching beats learned ranking.

## Resolved 2026-08-13: HyDE — a pre-registered negative result on both query sets

The prediction was frozen in `docs/hyde-axis-notes.md` on 2026-08-07, six days
before anything was built, and is reproduced verbatim in §0 of both reports so the
outcome cannot be read without what was expected beside it. Reports:
`data/results/hyde_retrieval_73det.md`, `hyde_retrieval_thematic.md`, and
`hyde_generation.md` for what the generator actually wrote. Method: one
hypothetical document per query, generated once by `phi4` and cached, because
`temperature=0` is not reproducible on this stack — every arm reads the same
cache, so the comparison is paired by construction. HyDE feeds the **dense arm
only**; BM25 receives the raw query.

| set | dense recall@10, raw → HyDE | diff | Holm-adj |
|---|---|---|---|
| 73det (106 q, entity-anchored) | 0.5034 → 0.3135 | **−0.1898** | **0.0000** |
| thematic (179 q) | 0.4469 → 0.3733 | **−0.0736** | **0.0008** |

All six pre-registered cells are significantly worse on each set (2 retrievers ×
3 metrics, m=6), and all 9 embedders lose on both. **P1 held in the harder half of
its own wording** — it allowed "ties or degrades" and the result is directional, so
there is no bound to state. **P2 was refuted**: thematic, the one regime the
pre-registration said HyDE might genuinely help, loses too.

**The mechanism is now evidence rather than reasoning, and it is dilution, not
deletion.** `person` is the worst entity type (**−0.2798**, 0.3604 → **0.0807**),
yet 29 of 30 generated documents still literally contain the queried name: the
discriminative token is not lost, it is averaged into ~250 tokens of invented
context. That is exactly the argument the prediction rested on — 73det is an
exact-token regime (BM25 alone scores **0.8147** on `person`) which semantic
elaboration can only dilute.

**P2's reasoning survives its own refuted forecast, and that is the transferable
finding.** If damage comes from diluting a lexical signal, it should be smaller
where that signal is weak (BM25 collapses to **0.2990** on thematic against
**0.4930** entity-anchored) — and it is: 2.6x smaller on the primary arm, ~6x
smaller for BM25 poisoning (**−0.0462** vs **−0.2735**), with the correlation
between an embedder's baseline strength and HyDE's damage falling from r = −0.887
to r = −0.282. **Cite this as *HyDE is less harmful where the lexical signal is
weak*, never as *HyDE helps thematic*: less to lose is not something to gain.**

**The null belongs to HyDE, not to one wiring.** Four formulations were measured
and they order by how much of the raw query survives into the embedded text —
`concat` **−0.0817**, `hyde_q` **−0.1405**, `hyde_half` **−0.1769**, `hyde`
**−0.1898**. Damage monotone in distance from the question is the shape of a real
effect, not of an implementation bug. `concat` is the only arm reaching ns
anywhere (73det hybrid −0.0209; thematic dense **−0.0250**, Holm 0.2316) and is
exploratory by the frozen decision rule, so it is a bound: at best dense recall@10
loses no more than **0.0576** and gains no more than **0.0061**, for 7.85 s of
generation per query against a 475.6 ms routed hybrid query.

**Two premises that were assertions until now.** Feeding the same document to BM25
as well costs a further **−0.2735** recall@10 on top of HyDE's own loss — larger
than the entire dense-arm effect — so the dense-only split is measured, not
assumed. And every document hit the 256-token cap, which greedy decoding lets us
bound for free: a prefix of a generation *is* what a smaller cap would have
produced, so the `hyde_half` arm costs no second generation run and does not
unpair the comparison. Across four cells (2 sets × 2 retrievers) it moves the
result with no consistent sign and by under 0.03 — length is not the constraint.

**No re-measurement against the shipped hard router is owed.** The
pre-registration made that follow-up conditional on a *positive* unrouted result,
precisely so a negative one could not be kept alive by an untested "but maybe with
routing". Anchors, from an independent numpy code path: `hybrid_raw` reproduces
the published unrouted hybrid **0.6281** and `dense_raw` the published **0.5034**,
both exactly.

## Resolved 2026-08-13: ColBERT / late interaction — a pre-registered negative result, and a separate ship decision

The prediction was frozen in `docs/colbert-late-interaction-notes.md` before
anything was built, and it was a **conjunction** on purpose: *ColBERT-alone ties or
beats **BM25** on `person` **and** ties or beats the best dense embedder on
`program`, in the same run.* It is motivated by this project's own results — BM25
carries `person` and dense carries `program` (see the per-`entity_type` breakdown
below) — so the question is whether late interaction *covers* that split, and an
aggregate win must not be allowed to answer it. Report: `data/results/colbert_pilot.md`
(+ `colbert_pilot_baselines.md`, `colbert_pylate_crosscheck.md`,
`colbert_model_qualification.md`, `colbert_length_profile.md`). Pilot: `recursive`
chunker only, doc300/q32, 106 Gold queries, unrouted, k=10, 7/7 self-checks PASS.
**Re-run 2026-08-20 against rebuild #4** (figures below are that run; the 2026-08-13
originals are in `data/results/_pre_2026_08_18_rebuild4_refresh/`). The verdict, the
ship decision and every conclusion below are unchanged — `person`, `course` and
`faculty` ColBERT scores are byte-identical and only `program` moved.

| cell | comparator | ColBERT | bar | diff | Holm-adj | |
|---|---|---|---|---|---|---|
| `person` | BM25 | 0.8360 | 0.8053 | **+0.0308** | 0.3974 | clears (tie) |
| `program` | dense `qwen3_0.6b` | 0.2749 | 0.6086 | **−0.3337** | **0.0000** | fails |

`person` clears as a **tie**, with the CI ruling out ColBERT beating BM25 by more
than **0.1030** or losing by more than **0.0429**; `program` fails by 6.7x the STOP
margin. **The bars are recomputed at `recursive`, never taken from the published
cross-chunker aggregates** — a one-chunker treatment against a nine-chunker bar is
the wrong-pair trap that killed per-`entity_type` alpha and rrf4 — and S1/S2
reproduce the published **0.8147** / **0.6034** exactly from the same code path.
(The `program` anchor read **0.6066** before rebuild #4; the `person` one is unmoved.
The value the prediction was *registered* against is kept separately in
`_REGISTERED` and rendered in `colbert_pilot_baselines.md`, so re-pointing an anchor
after a rebuild cannot silently re-base the pre-registration.)

**The mechanism is worth more than the verdict, and it answers the axis's own
motivation in the negative.** ColBERT is strong exactly where the lexical arm is
strong (`person` 0.8360 ≈ BM25 0.8053 against dense 0.4281) and weak exactly where
the lexical arm is weak (`program` 0.2749 ≈ BM25 0.3278 against dense 0.6086): it
**inherits** one side of the person/program split instead of covering it. It is not
purely lexical either — on `course` it beats both arms (0.6176 vs 0.5759 / 0.4280).

**And it carries the highest overall figure in its own table — 0.5555, against BM25
0.5088 and dense 0.5264.** Written as an aggregate, this run would have been
published as a success. That is precisely what a conjunctive pre-registration exists
to refuse, and it is the clearest example in this project of why the cells the
mechanism lives in must be named in advance.

**The 512/48 length rider was executed and did not fire, answered as a bound rather
than a threshold.** The frozen rule conditions the fallback on the losing cell's
truncation being "materially above" the corpus rate, and choosing what counts as
material *after* seeing −0.3337 is the favourable re-reading a frozen rule exists to
prevent. So truncation was granted the most damage arithmetically possible — a gold
resolution with *any* truncated chunk is destroyed outright. Over `program`'s 221
gold resolutions / 7,659 chunks, **32 chunks are truncated (0.42%, below the corpus
rate of 1.11%)** touching 14 resolutions, and total loss of all 14 explains at most
**0.0837** against a **0.3337** gap. Both readings agree, 4x short; 300/32 stands and
truncation remains a confound pointing *against* the treatment.

**The checkpoint arrives broken, which is a methods finding independent of the
verdict.** `jinaai/jina-colbert-v2` loads remote code written for `transformers`
4.43 under 5.12, and all 24 layers' rotary `inv_freq` come up as uninitialised
memory, making the rotation the identity. It was caught by a qualification gate
failing, not by reasoning, and **the broken model scored the hand-written relevance
example *better* than the repaired one** — a position-blind model returning plausible
numbers is the danger, not a crash. The repair (`_repair_rotary`) is restoration, not
modification: an independent pylate reference pinned to `transformers` 4.53.2
reproduces our query vectors **bitwise** (max|Δ| 0.000e+00). That same cross-check
found a defect all 11 internal gates were structurally unable to see —
`mask_punctuation` was masking whitespace and no punctuation at all, because the two
skiplists were built by different tokenizer calls; after the fix the document tensors
agree to min cosine **0.999936** and MaxSim **20.8212** vs pylate's 20.8213.

**Ship decision: do not adopt** — a *separate* decision from the axis verdict, since
the frozen rule only governs whether to spend more GPU on the question. Four grounds,
heaviest first. (1) **The failed cell is the one the shipped system depends on**:
`program` is where the router hands off to a dense specialist *because* BM25
collapses there (0.3278), so adopting ColBERT trades away a capability the system has
in order to buy one BM25 already supplies free — and `person`, the cell it cleared,
only ties. (2) **It was never shown to beat what ships, and was never measured
against it either** — hybrid at the same chunker was never a bar and neither was the
router; indicatively (**not** like-for-like, different chunker/embedder systems)
unrouted hybrid publishes 0.6229 and routed 0.6811 against 0.5555, and for a ship
decision the burden sits on the candidate anyway. (3) **Cost**: query p50 **1578.9 ms**
against a routed hybrid query's 475.6 ms (~3.3x), 1.89 GB fp16 per chunker (7.3 GB
for four, which will not co-reside on a 12 GB card), plus `_repair_rotary` as a
standing maintenance liability keyed to a `transformers` version. (4) The `course` win
is a **per-`entity_type` repair**, and that shape has died against the hard router
twice here already — it is a hypothesis needing its own pre-registration, never a
result to read off this table.

**What is not closed**: ColBERT against the shipped hard router, fused with BM25, or
on a second checkpoint. Those are *new* predictions, not a continuation of the failed
one, and the axis must not be reopened as one.

## Resolved 2026-07-21: ConGen/SCT max_seq_length — investigated, model-specific answer found

Both `congen` and `sct` (PhayaThaiBERT-backbone, kornwtp) ship
`sentence_bert_config.json` with `max_seq_length=128` in their HF repo —
set by the model author, not this project, well below their shared
backbone's true 510-token ceiling (RoBERTa reserves 2 position slots for
the padding offset; `max_position_embeddings=512` in `config.json`, but
`tokenizer_config.json` on both repos independently confirms
`model_max_length: 510`). Verified empirically against real built chunks
that a meaningful share exceed 128 tokens (fixed_size chunks average 172,
max 515; semantic chunks up to 3116) — raising the question of whether the
128 cap was silently discarding useful content.

**Tested directly**: rebuilt both models with `max_seq_length: 510` and ran
a paired-bootstrap before/after comparison on the Gold 73-det set
(`tools/eval/congen_sct_truncation_fix_eval.py`,
`data/results/congen_sct_truncation_fix_report.md`). Result was **not
uniform across the two models**:

| model | 128-cap recall@10 | 510-cap recall@10 | diff | verdict |
|---|---|---|---|---|
| `sct` | 0.1374 | **0.1519** | +0.0144, p<0.0001 | 510 is genuinely better — **adopted 510 going forward** |
| `congen` | **0.4134** | 0.3836 | -0.0298, p=0.0016 | 510 is significantly *worse* — **keeping the original 128** |

**Interpretation**: the 128 cap was not a uniform bug. For `sct` it really was
discarding useful content and 510 fixes that. For `congen`, feeding longer
input actually degrades quality — plausibly because ConGen is a pure
knowledge-distillation from `paraphrase-multilingual-mpnet-base-v2`, a
teacher conventionally used on short sentence-pair inputs; stretching
ConGen's input to 510 tokens pushes it outside the input-length
distribution it was distilled on, a train/test mismatch rather than a
truncation-loss problem. **Practical conclusion: use `max_seq_length=510`
for `sct`, keep the shipped default (128) for `congen`.**

**Consequence for numbers already in this document**: every `congen` result
above (including the "program-query specialist" framing) was measured at
128 tokens — which this investigation confirms is the *correct* setting for
this model, not a bug. **No correction needed for `congen`.** `sct` numbers
using the 510-cap are new (first real numbers for this model — it was still
building when the cap issue was first noticed) and still need an
entity-type breakdown + significance test against the other 7 embedders,
which is pending. See [[project_embedder_comparison]] /
`docs/chunker-embedder-comparison-log.md` for the full investigation
narrative.

### Structural pattern: Thai-specific (Group A) models have a lower architectural context ceiling than the top-tier winners — but reaching it isn't automatically better

Worth stating as a limitation/discussion point independent of the bug fix
above: **every Group A (Thai-specific) embedder candidate surveyed for this
project — `congen`, `sct`, and also SimCSE-WangchanBERTa and
ConGen-XLMR-Thai (considered, not built) — is built on a RoBERTa/XLM-R
**base** backbone, all of which inherit the original BERT architecture's
512-token position-embedding limit.** This is not a per-model coincidence;
it's the out-of-the-box architecture every community-trained Thai sentence
embedder in this space starts from.

By contrast, the embedders that top this project's comparison —
`bge_m3` (8192 tokens) and `qwen3` (32,768-40,960 tokens natively) — are
**not architecturally immune to this limit**, they simply had it engineered
away: `bge-m3` is itself XLM-R-based (same family as the Thai-specific
models) but BAAI specifically extended its position embeddings and
continued-pretrained it for long context (`max_position_embeddings=8194` vs
XLM-R-base's native 514); Qwen3-Embedding uses a long-context decoder
architecture by design. Long-context support is an **engineering investment
choice**, not an inherent property of "understanding Thai well" vs. not —
the Thai-specific models here come from academic/community teams (kornwtp,
mrpeerat) working from off-the-shelf backbones, not from an organization
that invested in a long-context extension pass the way BAAI or Alibaba did.

**Implication for RQ2** ("does Thai-specific beat multilingual/LLM-based
considering quality+cost?"): part of the quality gap this project measures
between Group A and the top-tier multilingual/LLM-based embedders may
reflect this **resourcing/engineering-maturity gap in the Thai embedding
ecosystem**, not a Thai-language-understanding gap per se — an important
distinction for the paper's discussion section, since the two explanations
carry very different implications (one says "multilingual models are
inherently better for Thai", the other says "Thai-specific models haven't
had their long-context pass yet, and might close the gap if someone did
that pass"). The 510-token architectural ceiling (RoBERTa/XLM-R base) is
structurally lower than the top-tier winners' regardless of what each
model's *effective* input length turns out to be — it caps the *maximum
possible* long-chunk handling for every Group A candidate here, even though
(per the section above) actually reaching that ceiling helped one model
(`sct`) and hurt another (`congen`).

**Caveat sharpened by the before/after result above**: it would be too
simple to say "Thai-specific models lose because they can't see long
context, and would close the gap if they could" — `congen`'s result shows
that even where more context is architecturally available, a model's
*training regime* can make using it counterproductive. The honest framing
for the paper is two separate, stackable limitations: (1) an architectural
ceiling (510 tokens) that no Group A candidate here exceeds, imposed by
starting from an off-the-shelf backbone instead of an extended one like
BGE-M3's; and (2) within whatever ceiling exists, a model's *effective*
usable length is set by its training regime, not just its architecture —
`sct` (trained on parallel-corpus sentence *pairs*, scb-mt-en-th-2020) uses
extra context productively, `congen` (pure distillation from a
short-sentence-oriented teacher) does not. Both are real constraints on the
Thai-specific ecosystem's current models, worth stating separately rather
than collapsing into one "needs more investment" story.

## Resolved 2026-07-30, refreshed 2026-08-05: Statistical power — every tie in this document is a bounded claim, not a null

`tools/eval/power_analysis.py` → `data/results/power_analysis.md`. Full
validity assessment (all seven threats, not just this one):
`docs/eval-validity-threats.md`.

Most of this document's headline claims are **null results** — the top-4
embedders are tied, `semantic` never significantly beats any chunker,
normalization and segmentation do nothing. Until now none of them stated what
size of difference the design could have detected, which is the first thing a
reviewer asks about a null. This closes that gap for every comparison at once.

**Refreshed 2026-08-05** against the fully re-generated post-rebuild-#3
retrieval results (dense, BM25, and hybrid all regenerated fresh against the
rebuilt `chunker_compare_full` index — see
[[project_index_rebuild_pending]]/[[feedback_refresh_all_retrieval_paths_after_rebuild]]
in project memory). The split is unchanged in count — a genuine consistency
check, not a coincidence — but one previously-flagged exception pair is now
resolved. Numbers below are the current (refreshed) ones.

**Headline: across 180 pairwise comparisons on 3 metrics, 138 are significant
and all 42 ties have an observed difference below their MDE — zero
"underpowered" verdicts.** No tie in this project is an artifact of low
power. Every one can be restated as a bound ("rules out differences larger
than X") instead of an absence of evidence.

- **The chunker ties are the tightest results in the study**, which matters
  because the retirement of "semantic chunking wins" (2026-07-29) currently
  reads as a failure to find anything. It is not: `fixed_size` vs `sentence`
  rules out recall@10 differences larger than **0.031**, and the whole
  6-pair chunker family bounds at **0.031–0.052**. The correct claim is
  "the chunker axis is bounded small", not "we could not tell".
- **Embedder ties are looser**, bounding at 0.05–0.10 — so the "top-4 tied
  cluster" is real but a weaker statement than the chunker one, and should
  not be written in the same voice.
- **The weakest tie in the paper is now `sct` vs `m2v` on MRR**, consistent
  with a difference as large as **0.1048**. The pair previously flagged as
  the exception (`e5_small` vs `jina_v5` on MRR, reported *inconclusive* as
  of 2026-07-30) has since resolved cleanly to "ruled out": its CI bound
  moved to 0.1029, just under its MDE(80%) of 0.0856, so it no longer needs
  a special caveat. **There is no remaining inconclusive pair in this
  document — every one of the 42 ties is a clean, citable bound.**
- Median MDE(Holm, 80%) by family, recall@10: embedder pairs **0.106**,
  hybrid-vs-dense **0.075**, hybrid-vs-BM25 **0.083**, chunker pairs (see
  above, tightest). Depth is why these are as good as they are — 9.87
  relevant documents per query means each query's score is an average, not a
  near-Bernoulli draw.
- **The closed form is verified, not assumed.** MDE = `(z + z_power)·sd/√n`
  presumes a normal statistic; these tests are percentile paired bootstraps
  on discrete, zero-inflated differences. Simulating the *actual* bootstrap
  at the computed MDE (6 spot-checked pairs) gives achieved power
  **0.777–0.858** against a nominal 0.80 (MC se ≈ 0.02) — mildly
  conservative, safe to cite.
- Also reported per pair: `n needed` to detect the observed effect. The
  closest tie to resolvable is still `e5_small` vs `bge_m3` on recall@10,
  needing **n≈212** (roughly double the current set); the rest need hundreds
  to tens of thousands (`bge_m3` vs `jina_v5`: n≈29,183). That spread is
  itself the argument for not chasing them — doubling the gold set would
  resolve one pair out of 42.

**Known caveat, same as every other persisted-results consumer**: recomputed
from `data/results/`, so it must be re-run after an index rebuild. Last
re-run: 2026-08-05, against `chunker_compare_full` rebuild #3.

## Resolved 2026-08-03: Pooling bias — qrels are a modest ~8-11% undercount, not directionally biased

`tools/eval/residual_relevance_sample.py --score` → `data/results/residual_relevance.md`.
Full validity assessment: `docs/eval-validity-threats.md` §3.

The Gold qrels are built by string containment (the same mechanism BM25 uses
at query time), so a semantically relevant document phrased differently is a
false negative that could unfairly penalise dense retrieval. A blinded,
126-item human review of unjudged top-10 hits (29 stratified queries, dense/
BM25/hybrid) tested this directly.

**A first judging pass came back at ~98-100% residual relevance for every
arm and was retracted the same day** — not a real finding, a review-app
measurement bug. The app's calibration-reference panel (documents the qrels
already confirm relevant, shown for context) displays full text on the same
page as the item being judged; a page-wide browser search for the query's
entity name found it in that guaranteed-relevant reference material instead
of the candidate for **100 of 100** checked cases. `residual_relevance_decompose.py`
corrected this by reapplying the qrels' own construction rule (per
entity-type: title-substring for programme, secretarial-aware given+surname
regex for person, filing-title-gated tag for faculty, canonical-name
substring for course) directly against each candidate's full text — valid
here because these are specific named-entity queries, where the corpus owner
confirmed no case exists where the answer is relevant without the entity
literally appearing (unlike thematic queries, where semantic-without-exact-
match relevance is real).

**Corrected result**: residual relevance rate **dense 0.191, BM25 0.224,
hybrid 0.224** (all Wilson CIs overlap) — **~0.8-1.1 additional genuinely-
relevant documents per query** beyond the qrels' own mean of 9.87/query, i.e.
**~8-11% more**, not the retracted pass's 43-49%. Qualitative verdict
unchanged from the pre-registered decision rule: incomplete, not
directionally biased — every BM25-vs-dense comparison in this project stands
as a relative ranking, and absolute recall/precision numbers need only a
modest-undercount caveat, not a severe one.

**Method lesson worth citing in the methods section**: a blinded-review UI
must not show confirmed-relevant reference material in a scope a search tool
(or a human's eye) can reach while judging a different item — the reference
material is relevant by construction and will silently answer the wrong
question.

## Resolved 2026-08-03, refreshed 2026-08-07: RQ4 — does better retrieval produce better answers?

Design + full narrative: `docs/rq4-design.md`. Report: `data/results/rq4_score.md`.
Scripts: `tools/eval/rq4_build_contexts.py` → `rq4_generate.py` → `rq4_score.py`
(paired bootstrap n_boot=10,000 seed=42 + Holm, the same machinery as every other
significance test in this document). Generator is **local `phi4` via Ollama**, no
external API. 5 arms × 106 queries (`gold_query_set_73det.yaml`) × 2 prompt
variants: `sentence_cap` (the original, rule 4 = "answer in ≤3 sentences") and
`cite_all` (rule 4 = "cite every relevant document").

### Citation grounding (4a) and arm ordering (4c)

| variant | arm | precision | (n with citations) | recall | phantom / total |
|---|---|---|---|---|---|
| `sentence_cap` | hybrid `qwen3_0.6b` × semantic | **0.7016** | 93 | 0.2781 | 0/269 |
| `sentence_cap` | dense `qwen3_0.6b` × semantic | 0.6549 | 92 | 0.2261 | 0/274 |
| `sentence_cap` | BM25 × semantic | 0.6607 | 81 | 0.2265 | 0/229 |
| `sentence_cap` | hybrid `m2v` × semantic | 0.5508 | 84 | 0.1820 | 0/209 |
| `sentence_cap` | closed-book | — | 0 | 0.0000 | 0/0 |
| `cite_all` | hybrid `qwen3_0.6b` × semantic | **0.7268** | 101 | **0.3962** | 0/421 |
| `cite_all` | dense `qwen3_0.6b` × semantic | 0.6798 | 96 | 0.3356 | **4/391** |
| `cite_all` | BM25 × semantic | 0.6104 | 93 | 0.2961 | 0/330 |
| `cite_all` | hybrid `m2v` × semantic | 0.5278 | 89 | 0.2038 | 0/287 |
| `cite_all` | closed-book | — | 0 | 0.0000 | 5/5 |

**This table is the 2026-08-10 re-score, after the 81 prompt-truncated cells were
regenerated at `num_ctx=16,384`** (`docs/rq4-prompt-truncation.md` §4b). `hybrid`
and closed-book had no truncated cell and are byte-identical to the pre-repair
table; `dense`, `BM25` and `m2v` all moved. The pre-repair figures — dense 0.6629
precision / 0.3206 recall under `cite_all`, BM25 0.5968 / 0.2938 — are superseded
and must not be cited.

**Citable claim: retrieval quality survives the generation stage.** Hybrid is
significantly above `BM25` and `m2v` on citation precision and recall under
`cite_all` (Holm-adj ≤ 0.0180), and `m2v` — the RRF-failure arm — is significantly
worst on both metrics against every other arm.

**State it as `hybrid > {dense, BM25} > m2v`, not as a strict 4-way ordering** —
and note that **`hybrid > dense` is now a bound, not a result.** Dense and BM25
are not significantly separated in either variant (Holm-adj 1.0000 under
`sentence_cap`, 0.1798 under `cite_all`), and under `sentence_cap` BM25 (0.6607)
numerically edges dense (0.6549). Hybrid's margin over dense was significant on
citation recall under `cite_all` until the truncation repair (−0.0756, Holm
0.0132); dense was the most-truncated arm and hybrid the least, so restoring
dense's evidence narrowed it to **−0.0606, CI [−0.1115, −0.0098], Holm 0.0760 —
not significant**. Read it as: the ordering holds numerically, and the data rule
out dense beating hybrid by more than 0.0098. An earlier version of this claim
("citation precision orders exactly as recall@10 did") over-read a tie; corrected
2026-08-07, then narrowed again 2026-08-10.

**And that last bound is `phi4`'s, not the system's — withdraw it as a system
claim (2026-08-12).** The second-generator check below re-ran the same contexts
through `gemma4:e4b`, where the same cell reads **+0.0228, CI [−0.0258, +0.0711],
Holm 0.7212** — the sign is reversed and the interval straddles zero, so the
CI-excludes-zero reading above does not replicate. **`hybrid` vs `dense` is
unresolved and generator-dependent**; state the ordering as
**`{hybrid, dense} > BM25 > m2v`**, which both generators support, and attribute
any hybrid-over-dense statement to `phi4`.

Family 1a (`sentence_cap`) is 2/12 comparisons significant, family 1b (`cite_all`)
8/12 (was 9/12 before the repair) — **arm ordering is more cleanly separable under
the better prompt**, because the original prompt's citation budget partly masked
how bad `m2v`'s retrieval is.

### The prompt ablation — RQ4's headline, and the most robust result in it

The original run's flat citation recall was **a prompt artifact, not a generator
ceiling**. Rule 4's ≤3-sentence cap ran against a gold set dominated by
aggregation queries (mean 9.87 relevant documents per query). Under `cite_all`:

| arm | recall `sentence_cap` → `cite_all` | Δ | Holm-adj p |
|---|---|---|---|
| hybrid `qwen3_0.6b` | 0.2781 → 0.3962 | **+0.1181** | 0.0000 |
| dense `qwen3_0.6b` | 0.2261 → 0.3356 | **+0.1095** | 0.0000 |
| BM25 | 0.2265 → 0.2961 | **+0.0696** | 0.0000 |
| hybrid `m2v` | 0.1820 → 0.2038 | +0.0217 | 0.8052 (ns) |

**No significant precision cost anywhere** (every precision cell in this family
Holm-adj ≥ 0.8052) — the model cites more *correctly*, not more sloppily. The gain
is **not universal**: `m2v` does not improve, consistent with a context that
often lacks correct evidence to cite regardless of instruction. **Recommendation
is "fix the instruction", not "the generator is the bottleneck."**

**This is the one RQ4 result the prompt-truncation repair left completely intact**
— all three significant arms stayed at Holm 0.0000 and `m2v` stayed ns, which is
expected because the ablation is a *within-arm* comparison and truncation affected
both variants of an arm alike. Only the point estimates moved (dense +0.1005 →
+0.1095, BM25 +0.0734 → +0.0696).

### Abstention (4b) and fabrication

Closed-book abstains 106/106 under `sentence_cap` and **104/106 under `cite_all`**
(2 hallucinations, 5 phantom citations out of 5). `cite_all` has no zero-document
guard — a real, small cost of that wording, worth naming if it is adopted as the
paper's reported prompt.

**0 fabricated citations out of 981 under the original prompt**, across all four
retrieval arms — RAG's most-feared failure mode is absent here, which is the
payoff for exactly-checkable numeric labels. Under `cite_all` fabrication is not
zero: the dense arm shows 4/391, all from one query citing labels `[6]`–`[9]` when
only 5 documents were supplied. (Both denominators grew with the 2026-08-10
regeneration — 954 and 359 before it — while the phantom *counts* did not move in
any cell, so restoring the truncated evidence produced no new fabrication.)

### Both of those costs are repaired — report `cite_all_guarded` (2026-08-07, all four arms 2026-08-08)

The two paragraphs above describe `cite_all`, which is kept unedited so the 530
answers on disk stay matched to the prompt that produced them. The prompt actually
recommended for the paper is a third variant, `cite_all_guarded`, which adds two
rules **after** rule 4 and states that they outrank it — because the failure was
never a missing rule (rule 3 already forbids it and is identical across variants)
but rule 4's **position**, last before the question, winning on recency:

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

Closed-book abstention returns to **106/106** with **0/0** citations. Both guards
are confirmed on the failure each was written for — rule 5 (zero documents ⇒
abstain) on closed-book, rule 6 (cite only labels present) on dense, which is the
only arm that ever produced phantoms under any variant.

**The benefit survives**: `cite_all_guarded` beats the `sentence_cap` baseline by
**+0.1198 on dense** (Holm p = 0.0000 in every family — the figure no correction
choice can touch) and **+0.0706 on hybrid** (Holm p = 0.0192 in family 2, the
family built for this question; see the family-size warning below), so the
prompt-ablation headline does not depend on the unguarded wording. On bm25 the
guarded gain (+0.0533) does not reach significance where the unguarded one
(+0.0696) did, and on m2v neither variant moves. **The apparent cost relative to
unguarded `cite_all` is not a finding**: no arm is significant and the four point
estimates **do not agree on a direction** (dense +0.0104 against hybrid −0.0475,
bm25 −0.0163, m2v −0.0095). A real constraint-induced dampening would push the
same way on all four; this is what the measured generator noise floor predicts
(14/24 identical citation sets at temperature 0). As bounds rather than nulls: on
hybrid the interval rules out the guard being *better* than `cite_all` and admits
a loss of ~0.01-0.09; on dense it rules out a loss greater than ~0.02; bm25 and
m2v straddle zero. No precision comparison is significant anywhere (all 12 at
Holm p = 1.0000, smallest raw p 0.1126).

**The guard is not a free repair.** Rule 5 was written for the zero-document case
but applies to every arm, and it shifts the weak ones toward abstention: m2v
correctly-abstained 13 → **16** and hallucinations 16 → **13**, but "missed"
(gold present, abstained anyway) rose 10 → **18**; bm25 hallucinations 11 →
**10**, missed 14 → **11**. The strong arms barely move (dense missed 14 → 9).
Report this trade rather than only the closed-book repair.

**Arm ordering is prompt-dependent, and the two prompts separate *different*
pairs.** Direction holds under every variant (`hybrid > dense > bm25 > m2v`, m2v
significantly worst), but the number of separated pairs in family 1's 12 tests
goes `sentence_cap` **2/12** → `cite_all` **8/12** → `cite_all_guarded`
**6/12**. **The earlier reading — that the guard compresses the spread — is
withdrawn 2026-08-10**: it rested on a 3/12 measured before the
prompt-truncation repair, and restoring the truncated evidence lifted the guarded
count to 6/12 while lowering `cite_all` to 8/12. What the two counts actually
differ on is *which* pairs they resolve. Under `cite_all_guarded` all six
significant cells are m2v pairs — it separates the weak arm from everything —
while `hybrid vs bm25` narrowly misses on both metrics (recall Holm 0.1056,
precision 0.2430) and `hybrid vs dense` is a flat tie (−0.0028, Holm 0.9164).
`cite_all` is the only variant that separates the two strong arms from each
other. **Cite the 4c separation claim as a property of `cite_all`, with the
guarded 6/12 and its differing composition stated alongside.**

**Family sizes, and why they must be quoted.** Family 3 of
`data/results/rq4_score_guarded.md` now holds **24** variant-pair × metric tests
(4 arms, up from 12 with 2); family 2 holds **9** (up from 5). On 2026-08-08 they
stopped agreeing: `hybrid: guarded vs baseline`, identical data, point estimate
+0.0706 either way, reads **Holm 0.0192 (significant) in family 2** and **0.0720
(not significant) in family 3**. Neither is wrong. Family 2 exists to answer
"does this prompt beat the baseline", so quote it there — as family 2, of 9
tests. An unqualified "Holm p = 0.02" is now demonstrably ambiguous in this
report.

### Second-generator robustness check — `gemma4:e4b` (2026-08-12)

Every RQ4 figure above is one generator's. `rq4_generate.py --model gemma4:e4b`
re-ran **the same contexts** through a second model under both live prompts (1,060
answers, 76 min, 0 errors, 0 truncated, `think` read from the model's capabilities
and disabled explicitly, recorded per answer). Reports:
`data/results/rq4_score_gemma4.md`, `data/results/rq4_score_gemma4_guarded.md`.
`sentence_cap` is deliberately not re-run — the generator refuses that variant for
any model but `phi4`, whose 530 answers are keyed to it — so family 1b is the
whole comparison and families 1a/2/3 skip by construction.

**What transfers is the ordering.** Citation precision orders identically in all
four positions under `cite_all` (hybrid 0.7417 > dense 0.7375 > BM25 0.6850 > m2v
0.6279), and `m2v` is again significantly worst on both metrics against every
other arm.

**What does not transfer is the levels — and one bound.** Gemma's citation recall
is higher on every arm (dense 0.5074 against `phi4`'s 0.3356), so no absolute
figure in this section is a property of the retrieval arm alone. And
`hybrid` vs `dense` — the one pair this document told the reader to cite as a
bound — is the **only** pair of the twelve that disagrees on **sign**: `phi4`
−0.0606 with a CI excluding zero, gemma **+0.0228, CI [−0.0258, +0.0711], Holm
0.7212**, and +0.0108 (Holm 1.0000) under the guard. Every other difference
between the two models is a verdict resolving or failing to resolve, never a
reversal — which is why the headline is this one cell and not the flip count.
State the ordering as **`{hybrid, dense} > BM25 > m2v`**.

**The larger finding was not the reason for the run: `cite_all`'s closed-book cost
is model-dependent, and rule 5 is what contains it.** Unguarded closed-book
hallucinations are **2** for `phi4` and **24** for gemma (37/37 phantom
citations); `cite_all_guarded` takes gemma to **1** (1/1). In both models they are
**entirely `course` queries** — no person, programme or faculty query produced
one. So the guard whose benefit was measured on the model that barely needed it is
what makes the prompt safe on a model that does, which strengthens the
recommendation to report `cite_all_guarded` as the paper's prompt. Its published
*cost* does not generalise either: on gemma the `missed` count falls on every arm
(hybrid recall 0.4846 → 0.5155) and the cost lands on the weak arm's precision
instead (BM25 0.6850 → 0.6028).

Full narrative, including the sign-vs-verdict comparison in full:
`docs/rq4-second-generator-check.md`.

### Two caveats a reviewer will raise

1. **Citation precision is judged against the same qrels as retrieval**, so it
   inherits the pooling-bias threat above. Direction is conservative (the qrels
   are a ~8-11% undercount, not directionally biased — see the pooling-bias
   section).
2. **This refresh is not a clean before/after.** `phi4` at temperature 0 is *not*
   reproducible (GPU reductions are not associative): re-running byte-identical
   prompts reproduces the citation set 21/24 under `sentence_cap` but only 14/24
   under `cite_all`. Against rebuild #3, 362 of 530 (query, arm) cells changed
   context and were regenerated (the other 168 frozen, keeping the comparison
   paired), and 5 of 33 verdicts flipped — but all four *lost* verdicts were
   already borderline (Holm-adj 0.014-0.081) and sit inside that noise floor, so
   they are reported as **inconclusive, not reversed**. Nothing at p < 0.001 moved.
   Every claim in this section is drawn from what survived.
3. **Every RQ4 figure above is the 2026-08-10 re-score**, after the 81 cells whose
   prompts had been silently truncated at `num_ctx=8192` were regenerated at
   16,384 (`docs/rq4-prompt-truncation.md` §4b). The regeneration is paired — only
   those 81 answer files were moved aside, so the other 1,509 are byte-identical —
   and `hybrid`/`closed_book`, which had 0 truncated cells, come back unchanged,
   which is the internal control separating repair from generator drift. One
   verdict was lost (`hybrid > dense` citation recall under `cite_all`, now a
   bound) and three were gained under the guard. Figures from before that date
   are superseded, not merely older.

## Resolved 2026-08-07: Circularity in the entity arms — the paragraph the paper owes

This is the third and last of the validity threats in `docs/eval-validity-threats.md`
(power and pooling bias are closed above). It is the one threat that cannot be closed
by measurement, because the thing to be measured is the same object on both sides of
the comparison. What it needs instead is an explicit statement, which is drafted here
in citable form so the paper does not have to re-derive it.

**Current numbers, for reference.** `entity_lookup` (exhaustive, unranked, scored at
k=1000 so recall/precision reduce to plain set recall/precision): overall recall
**0.9449** — `course` 0.9804, `faculty_adjunct_aggregate` 1.0000, `person` 0.9255,
`program` 0.9013. `entity_boost` (dictionary-narrowed candidate pool, then hybrid
ranking, recall@10): `person` 0.8182, `course` 0.7174, `program` 0.5834,
`faculty_adjunct_aggregate` 0.5089. Reports:
`data/results/gold_entity_{lookup,boost}_73det_report.md`, re-scored **2026-08-12**
after the `match_programs` repair reached these arms (`programs_by_file.json`
regenerated + `entity_tags_full` rebuilt; the reports now stamp the build and
`docset_hash` they were scored against rather than quoting a hardcoded chunk count).
**Only the two program-bearing rows moved, which is the point**: `program` recall
0.8918 → 0.9013 and `entity_boost` `program` recall@10 0.5765 → 0.5834, while
`person`/`faculty` are identical to 4 decimals under `entity_lookup` — the untouched
loaders are a built-in control. The direction **refutes a pre-registered prediction**
that recall would fall because the repair cuts more tags than it adds (594 vs 140):
the degree-level guard *re-selects* a same-degree candidate rather than merely
dropping, so a rescued tag lands on the programme that actually owns it.
**Note for anyone citing an older draft: 0.9422 (2026-08-06) and 0.9291 (pre-08-05) are both superseded.**
The figure 0.9291 is additionally withdrawn as mislabelled `recall@10`; the metric
is recall@1000 and the current value is 0.9449.

### The paragraph

> The `entity_lookup` and `entity_boost` arms are reported separately from the
> chunker, embedder, BM25 and hybrid comparisons, and their scores are **not
> comparable to them**. Relevance judgements for programme- and person-anchored
> queries were derived from `programs.json` and `people.json`; the same two
> dictionaries are read by the `entity_tags` loader at index time and by the
> `entity_lookup` / `entity_boost` retrieval modes at query time. Evaluating those
> modes against those judgements is therefore partly self-fulfilling: a document the
> dictionary can name is retrievable *and* judged relevant, and a document it cannot
> name is neither. `entity_lookup`'s recall of 0.9449 should be read as an **upper
> bound on what an exhaustive matcher could deliver given this dictionary**, not as a
> measurement of retrieval quality, and it should never be quoted alongside the
> dense or lexical recall figures as though the four arms were ranked on the same
> scale.

### Three things that sharpen it, and are worth keeping in the paper

1. **The circularity is in the candidate set, not uniformly across every metric.**
   Both arms draw their candidates from the dictionaries, so *recall* is circular in
   both. But `entity_boost` orders that pool with ordinary hybrid retrieval, which
   never reads the dictionaries — so its rank-sensitive metrics (MRR 0.9778 person /
   0.6544 programme) are contaminated only through *which* documents are eligible,
   not through *how they were ordered*. That is a weaker form of the problem and can
   be stated as such. `entity_lookup` has no ordering at all (its MRR/nDCG are
   computed over arbitrary corpus order and are meaningless — the report says so),
   so nothing rescues it.
2. **The pooling-bias result does not transfer here, and assuming it does would be
   the error to avoid.** §Pooling bias establishes that the qrels are a ~8-11%
   undercount that is *incomplete but not directional* — safe for BM25-vs-dense
   comparisons because the missing judgements are not correlated with any one system.
   For the entity arms that independence fails **by construction**: a name the
   dictionary lacks is missing from the qrels and invisible to the retriever
   simultaneously. The undercount is aligned with the system's own blind spot, so its
   effect is optimistic rather than neutral. This is the reason the threat cannot be
   closed the way the other two were.
3. **The scope limit is real and is the mitigating fact.** The chunker, embedder,
   BM25 and hybrid comparisons — the bulk of the results above — do not consult the
   entity dictionaries at query time at any point. `entity_tags` is a separate index
   (`entity_tags_full`) that no other arm is built on. The circularity is confined to
   these two arms and does not propagate into any headline claim.

**What this costs the paper: almost nothing, if stated.** The entity arms were never
load-bearing for a conclusion; they exist to show that an entity-aware mode is
buildable on this corpus and roughly where its ceiling sits. Reported as a bounded,
non-comparable side result, they stay useful. Reported next to the dense/lexical
numbers without this paragraph, they would be the easiest thing in the paper for a
reviewer to attack.

## Resolved 2026-08-08, superseded in part 2026-08-18: Query routing — the router covers the whole Gold set, and (post-rebuild-#4) beats the best single combo under hybrid

Report: `data/results/routing_eval.md` (`tools/eval/routing_eval.py`, rewritten
2026-08-08). Reuses persisted rebuild-#3 results; no new retrieval. **Every arm
fetches k=10 from exactly one index and sends 10** — equal retrieval budget, so no
arm is winning by spending more.

**The coverage gap that motivated it.** `classify_query` shipped with three routes
(person / program / unmatched). The Gold set gained 33 `course` queries eight days
later and 13 `faculty_adjunct_aggregate` queries were never covered either, so
**46/106 = 43% of the set fell to the `unmatched` default** and nothing ever failed
— a route with no branch is silent, not loud. Course was also the entity type
furthest from its structural ceiling (65.6%). Adding the two routes takes unrouted
to **0/106**, with classification exact and no cross-firing in either direction
(33/33 course, 13/13 faculty, 30/30 person, 30/30 program).

**What routing is worth (dense, recall@10, n=106, paired bootstrap + Holm, m=18):**

| arm | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| shipped `unmatched` default alone (fixed_size+bge-m3) | 0.4129 | 0.5487 | 0.4342 |
| routed, 3 routes (person/program only, at today's targets) | 0.5230 | 0.6767 | 0.5660 |
| best single combo over all 106 (semantic+qwen3_0.6b) | 0.5707 | 0.8335 | 0.6574 |
| **routed, 5 routes (shipped, targets refreshed 2026-08-08)** | **0.6189** | 0.8366 | 0.7009 |
| routed, 5 routes (leave-one-out targets) | 0.6057 | 0.8701 | 0.6747 |
| routed, 5 routes (oracle targets — upper bound, not a system) | 0.6293 | 0.8976 | 0.7275 |

Two claims, and they must be kept apart:

1. **The 5-route router significantly beats the 3-route one it replaces**:
   **+0.0958 recall@10** (Holm-adj p=0.0000), +0.1599 MRR, +0.1349 nDCG@10, all
   p=0.0000. That is the delivered improvement. This margin is *invariant to the
   target refresh below* — both arms hold the same person/program targets, so the
   difference between them is exactly the course/faculty coverage and nothing else.
2. **No *deployable* routed arm significantly beats simply using the best single
   combo for everything.** Shipped routing is +0.0481 recall@10 over it (Holm-adj
   p=0.1548) and the honest LOO estimate is +0.0349 (p=0.3568), neither significant.
   The only arm that does clear the bar is `routed (oracle)` — significant on all
   three dense metrics (+0.0586 recall@10 p=0.0462, +0.0642 MRR p=0.0160, +0.0701
   nDCG@10 p=0.0036) — **and an oracle is not a system**: it is told each route's
   best target by the same 106 queries it is then scored on. Read those three as
   *the headroom a perfect per-route map would have*, which is real but small, not
   as evidence routing works. **Do not claim routing beats a well-chosen single index
   on this query set** — claim that it *matches* one without having to know in
   advance which one that is, and that it fixes a 43% coverage hole.

Under **hybrid** the case for routing is weaker still, and for an interpretable
reason: BM25 partially rescues the misrouted course/faculty queries, so the 3-route
baseline starts higher (0.6423 vs 0.5230 dense) and the 5-route gain shrinks to
+0.0408 (Holm-adj p=0.1152, not significant). The LOO arm reaches +0.0499 over its
matched baseline with a CI excluding zero, [+0.0137, +0.0884], but Holm-adj p=0.0780
at m=18 — report it as suggestive, not established. Hybrid MRR moves the *other* way
under LOO (−0.0492, CI [−0.0978, −0.0033]), i.e. per-route target selection is
reliable for recall@10 and overfits on MRR.

**One structural finding worth citing on its own: the best combo per route is
retriever-dependent.** `person` peaks at semantic+qwen3 under dense but
sentence+bge_m3 under hybrid; `program` at fixed_size+qwen3_0.6b vs
semantic+qwen3_0.6b. Any "specialist per route" framing (see the Embedder ×
entity_type section) has to name the retriever it holds for. A single
retriever-agnostic dict cannot express this, which is why `ROUTE_COMBO` was
**restructured 2026-08-08** into `ROUTE_COMBO_BY_RETRIEVER` (`dense` / `hybrid`
maps, plus a `route_targets(retriever_type)` accessor that falls back to the hybrid
map for retrievers the eval never covered — bm25, entity_lookup, qdrant, where the
choice is an extrapolation and is labelled as one in the source).

**The `person` and `program` targets were refreshed at the same time**, under a
stated adoption rule rather than an argmax: take the scan's best combo only when
the leave-one-out selector picks that same target in ≥29/30 folds, so the choice
demonstrably does not hinge on any single query.

| route | dense: was → now | hybrid: was → now | LOO folds |
|---|---|---|---|
| `person` | semantic+bge_m3 → **semantic+qwen3** (+0.0598) | semantic+bge_m3 → **sentence+bge_m3** (+0.0361) | 30/30, 30/30 |
| `program` | sentence+congen → **fixed_size+qwen3_0.6b** (+0.0813) | sentence+congen → **semantic+qwen3_0.6b** (+0.1223) | 30/30, 29/30 |
| `course` | unchanged (already the argmax) | unchanged | 30/33, 32/33 |
| `faculty` | **unchanged** — 3 distinct LOO targets over 13 folds | **unchanged** — gap only +0.0305 | 11/13, 12/13 |

`faculty` staying put is the rule doing its job, not an oversight: n=13 is inside
the embedder family's own MDE (~0.05–0.10), so "checked, nothing stable enough" is
the finding. The retired `sentence+congen` program target was not merely stale but
**actively harmful under hybrid** — routing `program` to it scored 0.5321 where not
routing at all scored 0.6105, i.e. −0.0784 on those 30 queries. After the refresh
that route scores **0.6545**, i.e. +0.0440 for routing instead of −0.0784, and it
is the single largest reason the soft-vs-hard verdict below flipped.

**Method note on honesty of the `shipped` arm — and it got weaker, not stronger,
with the refresh.** After 2026-08-08, four of five shipped targets are chosen from
this same 106-query scan, so `routed (shipped)` is largely fitted on the set it is
scored on and now sits near `routed (oracle)` by construction (dense 0.6173 vs
0.6277; hybrid 0.6811 vs 0.6863 — 2026-08-18, against rebuild #4). **Cite
`routed (loo)` as the generalisation estimate.** That arm was *unchanged* by the
2026-08-08 refresh because it never read the shipped constants, and rebuild #4 then
moved it on one retriever only: dense **+0.0317** (Holm 0.4424, ns), hybrid
**+0.0825** (Holm 0.0000, significant) — both against `best single combo (loo)` — which is the cleanest way to
state what the refresh bought: it raises the shipped router to what LOO already
predicted for a well-chosen per-route map, rather than creating new gain. Every
routed arm is compared against a baseline fitted the *same* way (`oracle` vs
argmax-over-106, `loo` vs a per-held-out-query re-picked single combo); comparing a
LOO arm against an oracle baseline would understate routing, the reverse would
overstate it.

**Not measured:** the `unmatched_strategy="rrf"` fan-out in `query_service.route_query`.
With 0/106 unrouted it is now unexercised by any eval, and its target list is
hardcoded to person/program/unmatched — never extended to course/faculty. The
figures once quoted for it ("indistinguishable on recall@10, t=0.59; RRF +15% MRR")
came from the retired 252-query/3-route eval and no script reproduces them; they
are withdrawn.

## Resolved 2026-08-08, superseded in part 2026-08-20: The hybrid fusion weight — one global alpha is worth nothing, a per-`entity_type` alpha is worth +0.0333 nDCG@10 (recall@10 went ns)

Report: `data/results/hybrid_alpha_sweep.md` (`tools/eval/hybrid_alpha_sweep.py`).
21-point grid (step 0.05) × 106 queries × 3 combos, live re-retrieval against
rebuild #3. `alpha` is the **dense** weight, BM25 gets `1-alpha`.

**Why this was worth doing.** Every hybrid number this project has published was
produced at an implicit, unswept 50:50 — `dense_weight`/`bm25_weight` existed on
`HybridRetriever` but a full-repo scan found them used in no config and no eval
script. That would be a minor omission if the two arms were uniformly matched. They
are not: BM25 alone scores **0.8147** recall@10 on `person` and **0.3497** on
`program`, a 0.465 swing wider than any embedder-to-embedder gap in the study.

**Method — alpha is applied to RRF, not to the separate score-fusion branch.** Each
arm's reciprocal-rank contribution is scaled by its weight (`Σ wᵢ/(k+rankᵢ)`, the
weighted-RRF of the literature, previously not implemented here). This matters for
interpretation: a uniform 0.5× factor cannot reorder anything, so **alpha=0.50 is
rank-order-identical to the plain unweighted RRF behind every published number**.
The sweep therefore isolates the weight instead of confounding it with a switch
from rank fusion to score fusion, and the grid has a true no-op control at its
midpoint. Verified, not assumed: the vectorised fusion is checked against the real
retrievers at all three grid points where an independent ground truth exists —
alpha=0.00 vs `BM25Retriever`, 0.50 vs `HybridRetriever`, 1.00 vs `DenseRetriever`
— identical top-10 in every case, so every alpha in between interpolates between
verified anchors.

**Result 1 — a single global alpha buys nothing, on any combo where 0.50 was
already sane.** For `sentence+qwen3_0.6b` the best global alpha is +0.0016
recall@10 over the shipped 0.50 (Holm-adj p=1.0000); for `semantic+bge_m3`,
+0.0189 (p=0.5530). On both, the *oracle* global alpha — fitted directly on the
test set — is not significant. The shipped 50:50 is a good global compromise.

**Result 2 — the per-type optima diverge so far that no single alpha serves them
(`sentence+qwen3_0.6b`, recall@10):**

| scope | n | best alpha | recall@10 at best | at alpha=0.50 | non-degrading plateau |
|---|---|---|---|---|---|
| person | 30 | **0.15** | 0.8446 | 0.7487 | 0.00–0.35 |
| program | 30 | **0.75** | 0.6377 | 0.6105 | 0.40–1.00 |
| course | 33 | 0.65 | 0.6162 | 0.5946 | 0.25–0.80 |
| faculty_adjunct_aggregate | 13 | 0.45 | 0.4865 | 0.4755 | 0.20–1.00 |
| all | 106 | 0.45 | 0.6297 | 0.6281 | 0.30–0.50 |

The `person` and `program` plateaus are **disjoint** — there is no alpha that is
non-degrading for both — and the shipped 0.50 lies *outside* the `person` plateau.
Per the pre-registered reporting rule, these are ranges, not a recommended value.

**Result 3 — per-type alpha is worth a real, constructible gain.** Same combo,
paired bootstrap, Holm family m=9:

| arm | recall@10 | vs 0.50 | Holm-adj p | nDCG@10 | vs 0.50 | Holm-adj p |
|---|---|---|---|---|---|---|
| `alpha=0.50` (shipped) | 0.6281 | — | — | 0.6951 | — | — |
| global best (oracle) | 0.6297 | +0.0016 | 1.0000 | 0.6951 | +0.0000 | 1.0000 |
| per-type best (oracle) | 0.6710 | +0.0429 | **0.0016** | 0.7510 | +0.0559 | **0.0000** |
| **per-type (LOO — the citable one)** | **0.6631** | **+0.0350** | **0.0252** | **0.7311** | **+0.0360** | **0.0210** |

The gain survives leave-one-out, so it is not an artifact of fitting on the scored
queries. **MRR is not significant** (+0.0275, p=0.9912) — cite this as recall@10
and nDCG@10 only.

**Result 4 — but the gain is conditional, and two of the three combos show why.**
Per-type alpha pays off only where the two arms' relative strength *inverts* across
query types:

- `semantic+bge_m3` — **nothing significant on any metric** (per-type LOO +0.0110,
  p=1.0000). Its per-type optima all cluster in 0.40–0.60. The mechanism is already
  in this document: `bge_m3` is the `person` specialist, so its dense arm has no
  per-type weak spot for BM25 to rescue. No inversion, no gain.
- `fixed_size+m2v` — the known RRF failure case. The optimum is **alpha=0.00**,
  i.e. switch the dense arm off entirely, and the gains are enormous (+0.2264
  recall@10 per-type LOO, p=0.0000). But per-type adds only +0.0105 over a single
  global alpha: the fix here is *drop the broken arm*, not *tune per type*. This
  also quantifies the failure case for the first time — on `person`, RRF at 50:50
  takes BM25's 0.8281 down to **0.1969**, destroying 76% of it by fusing with a
  broken partner.

**What this refines in the fusion rule.** "RRF helps the weaker arm and taxes the
stronger one" was established *across systems*; this measures it *within* one
system, across query types, and adds the payoff condition: **a per-type weight is
worth having exactly when arm strength inverts across types — not when the dense
arm is uniformly competent (`bge_m3`), nor when it is uniformly broken (`m2v`).**

**The aggregate is blind to the effect, demonstrably.** On `sentence+qwen3_0.6b`,
pure BM25 (alpha=0.00) and pure dense (alpha=1.00) score **identically at 0.5034**
aggregate recall@10 — while differing by **+0.4650 on `person`** and **−0.2682 on
`program`**. Two systems that invert completely per type are indistinguishable in
the mean. Any single-number comparison of a lexical and a dense arm on this corpus
can hide a total inversion; this is the strongest argument in the study for
reporting the per-`entity_type` breakdown alongside every aggregate.

**Constructibility.** A per-type alpha needs no new classifier: `classify_query`
already labels `entity_type` at query time and, since the 2026-08-08 route
expansion, covers 106/106 with no cross-firing. It is also the *soft* form of
routing — a misclassification costs a slightly wrong blend rather than the wrong
index.

**Caveats.** (1) `per-type best` is an oracle; only the LOO arm is citable. (2) LOO
holds out the scored query but still picks alpha from same-type peers in this
corpus, so it bounds fitting-to-the-query, not fitting-to-this-corpus. (3)
`faculty_adjunct_aggregate` (n=13) has a 17-of-21-point plateau — underpowered,
no per-type conclusion for it. (4) Classification is exact on this set, so the
real-world cost of misrouting is not measured here.

## Resolved 2026-08-08: Soft vs hard routing — hard wins once its targets are current, and the two are still substitutes

Two results landed the same day pointing opposite ways and had never been put on
one axis. **Hard routing** (§"Query routing") switches the *index* per route;
**soft routing** (§ above) switches only the *fusion weight* on one index. Script:
`tools/eval/soft_vs_hard_routing.py` → `data/results/soft_vs_hard_routing.md`.

**Read the date on this section.** It was first run against `ROUTE_COMBO`'s
2026-07-17 targets and reported soft ≥ hard. Those targets were refreshed the same
day (§"Query routing"), the script re-run, and **the verdict flipped**: hard
routing was being judged on a `program` target that actively hurt. The numbers
below are the post-refresh ones. Nothing about soft routing changed.

Four arms, every one retrieving k=10 from **exactly one index per query** — equal
retrieval budget, so no arm wins by fetching more
([[feedback_state_the_retrieval_budget_in_every_comparison]]). Hybrid throughout.
Index choice is held at its shipped value in every arm; the only fitted quantity
is alpha, fitted leave-one-out within a route. Routing uses `classify_query`, not
the gold label (the two partitions agree 106/106).

| arm | what the classifier moves | indices | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| **A** single @ 0.50 | nothing (no classifier) | 1 | 0.6281 | 0.8430 | 0.6951 |
| **C** routed @ 0.50 (**hard**) | the index | 5 | **0.6831** | 0.8686 | **0.7502** |
| **B** single + per-route alpha (**soft**, LOO) | the fusion weight | 1 | 0.6631 | 0.8705 | 0.7311 |
| **D** routed + per-route alpha (both, LOO) | both | 5 | 0.6629 | **0.8868** | 0.7568 |
| _B′ soft (oracle)_ | _upper bound_ | 1 | _0.6710_ | _0.8899_ | _0.7510_ |
| _D′ both (oracle)_ | _upper bound_ | 5 | _0.6901_ | _0.9112_ | _0.7721_ |

Arm C reproduces `routing_eval.md`'s `routed (shipped)` hybrid recall@10 to four
decimals (0.6811 both, 2026-08-18) from an entirely separate code path — a useful cross-check
that the two routing scripts agree on what the shipped router does.

**ONE significant result, not two — and the one that went is soft's.**
`B vs A` on nDCG@10 lost significance at rebuild #4 (+0.0333, Holm 0.0528, was
+0.0360 at 0.0216), so **soft routing no longer owns a significant result anywhere
in this table**, and it is now numerically below doing both. What survives is
`C vs A` on recall@10:

| comparison | recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| B vs A (soft vs none) | +0.0281 [+0.0039, +0.0547] | +0.0369 [−0.0094, +0.0848] | +0.0333 [+0.0102, +0.0581] |
| C vs A (hard vs none) | **+0.0581** [+0.0201, +0.0964] | +0.0140 [−0.0340, +0.0620] | +0.0512 [+0.0089, +0.0927] |
| **B vs C (soft vs hard)** | −0.0300 [−0.0687, +0.0060] | +0.0229 [−0.0317, +0.0761] | −0.0179 [−0.0582, +0.0220] |
| D vs C (does alpha add to routing) | −0.0163 [−0.0327, −0.0021] | +0.0253 [−0.0182, +0.0719] | +0.0086 [−0.0118, +0.0281] |

2026-08-18, against rebuild #4.

Paired bootstrap, 10,000 resamples, Holm within m=12. **`C vs A` on recall@10**
(Holm-adj 0.0242) and **`B vs A` on nDCG@10** (Holm-adj 0.0216) are the only
significant cells. Everything else is a bound.

**The head-to-head is a tie, stated as a bound in both directions.** `B vs C` is
ns on all three metrics; the CIs rule out soft beating hard by more than **0.0156**
recall@10, and hard beating soft by more than **0.0575**. So: **hard routing leads
numerically on every metric and owns the only significant recall@10 result, but it
has not been shown to beat soft — and it costs 5 indices to soft's 1.** The
cost-per-point argument for soft survives the flip; the "soft is at least as good"
claim does not.

**They remain substitutes, not complements — but for a sharper reason than before.**
D (both) is 0.6648 recall@10, *below* C's 0.6811, and `D vs C` is
**negative** (−0.0163, CI [−0.0327, −0.0021]) with a CI excluding zero, though ns
after Holm. Yet at the oracle bound D′ (0.6909) is the best arm in the table, above
C by +0.0098. (2026-08-18, against rebuild #4; the pre-rebuild reading was
0.6629 / 0.6831 / −0.0202 / 0.6901 / +0.0071.) Put
together: **there is a sliver of real headroom left for a per-route alpha on top of
routing, and LOO fitting costs more than that sliver is worth.** That is a stronger
statement than the pre-refresh version of this section made (it had D worse than B
even at the oracle) — the mechanism is a fitting cost, not an absence of headroom.

**Per-route, and this is where the refresh shows up:**

| route | n | A single@0.50 | B soft | C hard | D both | alpha* on single | alpha* on routed |
|---|---|---|---|---|---|---|---|
| `person` | 30 | 0.7487 | 0.8383 | **0.8531** | 0.8326 | **0.15** | **0.30** |
| `program` | 30 | 0.6105 | 0.6266 | **0.6545** | 0.6447 | 0.75 | 0.65 |
| `course` | 33 | 0.5946 | 0.6162 | **0.6262** | 0.5932 | 0.65 | 0.60 |
| `faculty` | 13 | 0.4755 | 0.4623 | **0.5008** | 0.4901 | 0.45 | 0.40 |

Hard routing now wins **every** route (person +0.1044, program +0.0440, course
+0.0316, faculty +0.0253) where before the refresh it won only course and faculty
and *lost* program by −0.0784. The `person` row still gives the mechanism in one
line: on the generic index the optimal alpha is **0.15** — hand the query to BM25,
which scores 0.8147 on `person` — while on the routed index, whose target *is* a
person specialist, it rises to **0.30**, i.e. toward neutral. **Both forms of
routing repair the same defect, a per-type weak dense arm, by different means**,
which is why applying the second after the first adds nothing.

**Decision taken 2026-08-08: do NOT wire a per-`entity_type` alpha into
`query_service`.** The +0.0350 that motivated the idea is arm **B vs A** — measured
against *no routing at all*, which stopped being the shipped configuration the same
day. The decision-relevant comparison is **D vs C**, per-route alpha on top of the
router that now ships, and it shows no gain on any metric: recall@10 −0.0202,
MRR +0.0182, nDCG@10 +0.0066, none significant. Total remaining headroom is the
oracle gap **+0.0098** (D′ 0.6909 vs C 0.6811, 2026-08-18), which LOO fitting costs
more than.
The mechanism says this is not merely a power problem: a per-type alpha exists to
repair a per-type weak dense arm, and hard routing hands each route a specialist
index that by construction does not have one — hence the `person` alpha\* moving
0.15 → 0.30, toward neutral, once the index is already the person specialist. This
matches the sweep's own precondition (the gain needs the two arms' relative strength
to *invert* across types; `semantic+bge_m3` gains nothing).

**The one branch that flips it:** if 5 indices becomes a deployment constraint, the
question is not "add alpha" but "**replace** hard routing with soft" — arm B reaches
**0.6510** on **one** index and is ns against arm C. **Rebuild #4 weakened this
branch and the sentence is corrected rather than merely re-quoted (2026-08-18):**
arm B's nDCG@10 margin over no routing lost significance (+0.0333, Holm 0.0528, was
+0.0360 at 0.0216), so soft routing no longer owns a significant result anywhere in
that table, and it is now numerically *below* doing both (0.6648) where it used to
be above. That is still a cost decision, not an accuracy one. Never ship both.

**Caveats.** (1) Arm A's index is the argmax over 36 combos on this same test set;
its defence is that `routing_eval`'s LOO selector re-picks it in every fold (best
single = 0.6229, and the LOO selector re-picks it in every fold), not that it was
chosen blind. (2) Arm C's
targets are now *also* fitted on this set (§"Query routing" method note), so C is
closer to an upper bound than it was — read `routing_eval.md`'s `routed (loo)`
(**0.6794** hybrid recall@10) as the generalisation estimate for the hard arm, which
still sits above soft's **0.6510**. (3) `faculty` n=13 — no per-route conclusion from
that row alone. (4) The soft arm's numbers reproduce `hybrid_alpha_sweep.py` to 4
decimal places from an independent code path, but its `recall@10` **verdict**
differs (Holm-adj 0.0252 at m=9 there, 0.0580 at m=12 here) — same data, same
difference, larger family. Cite the sweep's m=9 for "is a per-route alpha worth
anything"; cite this table's m=12 only for the four comparisons it was built to
make.

## Methodology

- **Metrics**: recall@k, precision@k, nDCG@k, MRR, MAP, all resolution-level
  (ADR-0002) — `src/rag_lab/metrics.py`. `evaluate()` accepts either a single
  `k` (original behavior) or a list of cutoffs (e.g. `[1, 3, 5, 10]`) to
  report multiple k in one pass; MAP is computed once per combination at
  `max(k)`, not per-cutoff, since Average Precision already aggregates over
  a ranking. Headline numbers throughout this doc still use k=10 only — the
  eval scripts (`run_gold_*_eval.py`) haven't been re-run with multi-k yet,
  see Open items.
- **Significance testing**: paired bootstrap over queries (resample unit =
  query, n_boot=10,000, seed=42, two-sided percentile p-value), per-system
  score averaged across the 4 chunker strategies first when comparing across
  the embedder/BM25 axis. Holm-Bonferroni correction applied **within each
  natural family of simultaneous comparisons separately** (not pooled
  globally) — e.g. the 36 embedder-vs-embedder pairs (9-embedder matrix) are
  one family, the 9 BM25-vs-embedder pairs are a separate family, and the two
  9-embedder hybrid families (hybrid-vs-dense, hybrid-vs-BM25) are each their
  own family per metric. Scripts: `tools/eval/embedder_matrix_9way.py`
  (current, 9-embedder matrix — retrieval + breakdown + aggregate
  significance in one script), `tools/eval/embedder_significance_test_by_entity_type_9way.py`
  (per-entity_type, 9-embedder), `tools/eval/bm25_vs_embedder_significance_test_9way.py`,
  `tools/eval/hybrid_significance_test_9way.py` — all four import shared
  label/exclusion logic from `embedder_matrix_9way.py`. Originals
  (`embedder_significance_test.py`, `embedder_significance_test_by_entity_type.py`,
  `bm25_vs_embedder_significance_test.py`, `hybrid_significance_test.py`,
  6-embedder versions) are superseded but kept for reference.
- **Corpus**: 2,853 resolution documents (`academic_resolutions/`, gitignored),
  chunked 4 ways (fixed_size, recursive, sentence, semantic) × embedded 9
  ways (see below) = 36 combos, plus BM25 (chunker-only, embedder-agnostic)
  and hybrid (RRF of BM25 + each of the 9 dense embedders, same 36 combos —
  fully extended to all 9 embedders as of 2026-07-21).

## Model selection rationale (why each embedder was chosen)

The paper's draft framework (`Embedding โมเดล.docx`, repo root) groups
candidate embedders into 4 categories (A: Thai-specific, B: multilingual
open, C: LLM-based, D: commercial API). This section records why each
model actually in (or explicitly excluded from) the comparison matrix was
chosen, so the paper's methods section can cite a reason per model rather
than just a list.

### Group A: Thai-specific

| model | status | why |
|---|---|---|
| `kornwtp/ConGen-BGE_M3-model-phayathaibert` | **in matrix** (`congen`) | ConGen distillation of BGE-M3 (dense teacher) onto a PhayaThaiBERT backbone — a community-referenced Thai-specific baseline, symmetric (no query/passage prefix asymmetry), zero-code to add (`type: local`, arbitrary `model_name`). |
| `kornwtp/SCT-KD-BGE-M3-model-phayathaibert` | **building** (2026-07-21) | Same backbone (PhayaThaiBERT) and same distillation teacher (BGE-M3) as the model above — differs only in training method (SCT vs ConGen). Chosen specifically to isolate the *training-method* variable while holding backbone constant, a cleaner ablation than picking an arbitrary new model. Secondary motivation: the original Thai-Sentence-Vector-Benchmark literature reports SCT outperforming both ConGen and SimCSE on STS tasks — this tests whether that ranking replicates on RAG-style entity-anchored retrieval, a different task/domain than the benchmark it was reported on. |
| "SEA-Embedding-ModernBERT-300M" | **retracted** (2026-07-21) | Was queued in an earlier session's memory but does not appear anywhere in the actual research notes and could not be located on Hugging Face — concluded to be an unverified/hallucinated entry from a prior conversation. Replaced by the SCT model above. **Do not cite or re-propose this name.** |
| SimCSE-WangchanBERTa | **considered, deferred** | Listed in the original notes as a Group A candidate. Not added alongside the current build because it would change *two* variables at once versus the existing ConGen-PhayaThaiBERT model (backbone: WangchanBERTa vs PhayaThaiBERT, **and** method: SimCSE vs ConGen) — a confounded comparison, mirroring the exact problem already flagged for bge-m3-vs-Qwen3-4B (architecture and size both differ). If a backbone ablation is wanted later, `kornwtp/ConGen-BGE_M3-model-wagchanberta` (verified to exist on the same `kornwtp` HF account) is the cleaner choice — same ConGen method as the existing model, backbone changed to WangchanBERTa, isolating backbone as the sole variable. |
| ConGen-XLMR-Thai (`kornwtp`/`mrpeerat`) | **considered, deferred** | Also a notes-listed Group A candidate; deprioritized on scope/time grounds — adding it would give Group A more models than Group B (3) despite Group A's narrower scope in the notes, for diminishing incremental research value once the ConGen/SCT training-method ablation above is in place. |

**Why PhayaThaiBERT over WangchanBERTa as the primary Thai backbone (literature
citation, not an in-house ablation)**: an external literature review
(`docs/thai-embedding-compare.md`; sourced from the
`mrpeerat/Thai-Sentence-Vector-Benchmark` project and a Forum for Linguistic
Studies benchmark paper, Dec 2025) already ran this exact backbone ablation
— WangchanBERTa vs PhayaThaiBERT, both trained with SimCSE/SCT/ConGen — and
reports it on a retrieval task (TyDiQA, R@1/MRR@10), the closest published
proxy to this project's RAG setting. Findings: PhayaThaiBERT beats
WangchanBERTa on TyDiQA retrieval under every method (SimCSE +8.78 R@1, SCT
+5.24, SCT-Distil +2.09, **ConGen +0.13**), and on STS-B under SimCSE (+7.33)
and SCT (+2.71), but WangchanBERTa edges ahead under ConGen on STS-B (-0.30
for PhayaThaiBERT). The reported cause: PhayaThaiBERT's larger,
XLM-R-augmented vocabulary (249k vs 25k tokens) better preserves
unassimilated English loanwords/code-switching common in casual Thai text,
which matters more for lexically noisy tasks (STS, classification) than for
retrieval that already benefits from ConGen's pure knowledge-distillation
signal from a multilingual teacher — hence the near-tie specifically under
ConGen, the method this project's `congen` model already uses. The paper's
own headline retrieval number for `ConGen-BGE_M3-model-phayathaibert`
(R@1=83.36, MRR@10=88.29) matches the exact model already in this project's
matrix, corroborating that the benchmark is describing the same model
family. **Decision: rely on this citation instead of building
ConGen-WangchanBERTa in-house** — the published ablation already covers the
one comparison (ConGen, same method as our existing model) most relevant to
this project's RAG task, and the margin under that method is small enough
(+0.13 R@1) that an in-house rebuild is unlikely to change the conclusion.
The out-of-scope backbone difference this project *cannot* verify from the
citation alone: whether the near-tie under ConGen holds on this project's
specific entity-anchored resolution-retrieval task (vs. TyDiQA's open-domain
QA) — flagged as a limitation, not pursued further given cost (a full
4-chunker rebuild) vs. expected marginal value.

### Group B: multilingual open

| model | status | why |
|---|---|---|
| `BAAI/bge-m3` | in matrix (`bge_m3`) | SOTA open multilingual, supports 8192-token context and dense/sparse/multi-vector retrieval — explicitly named in the notes as the flagship Group B model. |
| `multilingual-e5-large` | in matrix (`e5`) | Established open multilingual baseline, asymmetric query/passage prefixing (contrasts with bge-m3/ConGen's symmetric encoding — a real architectural axis, not just a different checkpoint). |
| `jina-embeddings-v5-text-small-retrieval` | in matrix (`jina_v5`) | Newer retrieval-tuned multilingual model, smaller than bge-m3/e5-large — adds a size/recency data point within the group. |

**Why stop at 3, no further additions**: user's explicit judgment (2026-07-21)
that these 3 already give adequate quality-tier and size coverage for Group
B, and further additions would unbalance effort against the other groups —
each new embedder costs multiple hours of GPU build time across the full
4-chunker matrix (semantic chunking especially), so headcount per group is a
real cost decision, not a free one.

### Group C: LLM-based embedding

| model | status | why |
|---|---|---|
| `Qwen/Qwen3-Embedding-4B` | in matrix (`qwen3`) | MTEB v2 top-scoring family; largest Qwen3-Embedding variant that fits an RTX 3060's 12GB VRAM at fp16 without quantization — directly tests whether the newest "LLM-based embedding" trend pays off in quality (RQ2). |
| `Qwen/Qwen3-Embedding-0.6B` | **building** (2026-07-21) | Same architecture/family/training as the 4B model above, only parameter count differs — a clean **intra-family size-scaling** comparison for RQ2 ("is bigger worth it"), unlike the bge-m3-vs-Qwen3-4B comparison already run, which confounds size with architecture and training method simultaneously. Fits comfortably in 12GB VRAM, so no quantization or precision compromise needed. |
| `Qwen/Qwen3-Embedding-8B` | **declined** | Needs ~16GB fp16 weights alone — does not fit the local RTX 3060 (12GB) without quantization, and quantization would introduce yet another confound (precision loss) on top of the size variable. User: "ถ้าไม่พอ ก็ไม่เอาครับ" (if it doesn't fit, skip it). |
| e5-mistral / NV-Embed (7-8B class) | **out of scope** | Notes-listed Group C candidates; same VRAM ceiling problem as Qwen3-8B, not pursued for the same hardware-constraint reason. |

### Group D: commercial API — excluded entirely

OpenAI `text-embedding-3-large`, Cohere `embed-v4`, Google Gemini Embedding
(notes' reference/upper-bound candidates) were **not evaluated**. Reasons:
real per-token monetary cost at corpus scale, dependency on API keys/uptime
outside the reproducible local pipeline, and — most importantly for this
corpus specifically — usage would mean sending institutional academic-council
resolution documents to a third-party API, a data-egress tradeoff the user
chose to avoid. The notes themselves frame Group D as an optional upper-bound
reference, not a required comparison arm, and the research-framework gap
analysis (`docs/research-framework-gap-analysis.md`) independently flagged it
as low-priority/optional for the same reasons.

### Non-embedder retrieval methods (for completion, not a Group A-D model choice)

**BM25** (lexical baseline) and **Hybrid** (RRF fusion of BM25 + dense) were
added because the notes explicitly require "at least 1 lexical baseline ...
to test whether dense embedding is worth it, and open the door to hybrid
search" — see the BM25/Hybrid sections below for what was found. Both use
code (`retrievers/bm25.py`, `retrievers/hybrid.py`) that already existed in
the framework before this evaluation round.

## Chunkers compared

fixed_size (512 chars, 50 overlap), recursive, sentence (crfcut), semantic
(bge-m3 breakpoint detection). **Semantic wins on every metric, averaged
across all 6 embedders**:

| chunker | recall@10 | mrr | ndcg@10 |
|---|---|---|---|
| **semantic** | **0.4939** | **0.7184** | **0.5483** |
| recursive | 0.3922 | 0.6135 | 0.4488 |
| fixed_size | 0.3786 | 0.6251 | 0.4417 |
| sentence | 0.3776 | 0.6243 | 0.4393 |

**Important caveat added 2026-07-29 — this table was never significance-tested
and predates the 9-embedder expansion (6-embedder dense-alone raw means
only).** After the "top single combo" retraction above raised the question
of whether any chunker is actually provably best, built the first-ever
chunker-vs-chunker significance test
(`tools/eval/hybrid_chunker_significance_test.py`, pure recompute from
persisted `gold_hybrid_73det` results, no new retrieval) — one 6-pair family
(fixed_size/recursive/semantic/sentence) per embedder, Holm-corrected per
metric, plus an aggregate family (each chunker's per-query hybrid score
averaged across all 9 embedders first, mirroring the embedder-matrix
convention). Full table: `data/results/hybrid_chunker_significance_test.md`.

**Result: `semantic` does not significantly beat any other chunker,
anywhere** — not in the aggregate test, and not for any single embedder
(including `qwen3_0.6b`, the specific combo the retracted "top single combo"
claim was about — its 4 chunkers, recall@10 0.6097–0.6265, are fully,
mutually tied on every metric, Holm-adj p≥0.44 throughout). The **only**
significant chunker-pairwise result anywhere in this whole test (aggregate
or per-embedder, 9 embedders × 3 metrics × 6 pairs + 1 aggregate family) is
**`fixed_size` losing to `recursive`** — significant on nDCG@10 in the
aggregate (Holm-adj p=0.0228) and for `qwen3`/`congen`/`m2v` individually,
and on recall@10 for `m2v`. Aggregate per-chunker means (hybrid, across 9
embedders): `recursive` 0.5291, `semantic` 0.5206, `sentence` 0.5205,
`fixed_size` 0.5073 recall@10 — `recursive` is now numerically highest, not
`semantic`, though the gap is not significant either.

**Practical conclusion**: the "semantic chunking wins" headline this project
has repeated since the very first comparison round does not survive being
tested as an actual significance claim under hybrid retrieval. The honest
framing going forward is **"fixed_size is the one chunker with a
demonstrated, provable disadvantage (specifically vs. recursive); the other
three (recursive, semantic, sentence) form a statistically tied cluster with
no provable winner."** `semantic` remains a perfectly reasonable choice
(never proven worse than anything), and it's still the one chunker where a
strong dense embedder demonstrably earns its cost over BM25 alone (see
"Per-chunker BM25 vs. embedder" below) — but "semantic is the best chunker"
should no longer be cited as a tested finding.

## Embedders compared (9 total)

| embedder | model | group |
|---|---|---|
| bge_m3 | `BAAI/bge-m3` | B (multilingual) |
| e5 | `multilingual-e5-large` | B (multilingual) |
| e5_small | `multilingual-e5-small` | B (multilingual) — size ablation vs e5 |
| congen | `kornwtp/ConGen-BGE_M3-model-phayathaibert` (max_seq_length=128, confirmed correct) | A (Thai-specific) |
| sct | `kornwtp/SCT-KD-BGE-M3-model-phayathaibert` (max_seq_length=510, fixed) | A (Thai-specific) |
| qwen3 | `Qwen3-Embedding-4B` | C (LLM-based) |
| qwen3_0.6b | `Qwen3-Embedding-0.6B` | C (LLM-based) — size ablation vs qwen3 |
| jina_v5 | `jina-embeddings-v5-text-small-retrieval` | B (multilingual) |
| m2v | `Thaweewat/jina-embedding-v3-m2v-1024` (Model2Vec static) | — |

**Aggregate (averaged across all 4 chunkers) — refreshed 2026-07-25 against
the clean, rebuilt indices**:

| embedder | recall@10 | mrr | ndcg@10 |
|---|---|---|---|
| qwen3_0.6b | **0.5240** | **0.8002** | **0.6129** |
| qwen3 | 0.5164 | 0.7927 | 0.5972 |
| bge_m3 | 0.5106 | 0.7340 | 0.5641 |
| jina_v5 | 0.4567 | 0.7165 | 0.5261 |
| e5_small | 0.4413 | 0.6837 | 0.4963 |
| e5 | 0.4328 | 0.6721 | 0.4881 |
| congen | 0.4159 | 0.6439 | 0.4727 |
| sct | 0.1558 | 0.2810 | 0.1793 |
| m2v | 0.1490 | 0.3203 | 0.1910 |

**Significance (36 pairwise tests, Holm-corrected per metric), re-run
2026-07-25**. Full table: `data/results/embedder_significance_test_9way.md`;
script: `tools/eval/embedder_matrix_9way.py`. **Every pairwise claim below
was re-verified against the refreshed table — none changed.**

- **Top tier is now 3-way: {bge_m3, qwen3, qwen3_0.6b} mutually NOT
  significant on any metric** (all Holm-adj p=1.0) — qwen3_0.6b (0.6B
  params) is statistically indistinguishable from qwen3 (4B, ~7x larger).
  **Size buys nothing measurable within the Qwen3 family on this task.**
- **e5 vs e5_small: NOT significant on any metric** (Holm-adj p=1.0) — same
  pattern, a second independent confirmation that a much smaller model
  (~118M vs ~560M, ~4.7x smaller) ties its larger sibling here.
  **Two separate model families, two independent "smaller ties larger"
  results — a real pattern, not a fluke of one family.**
- **sct vs m2v: NOT significant on any metric** (Holm-adj p=1.0) — even
  after the max_seq_length fix, `sct` is statistically indistinguishable
  from the weakest embedder in the whole matrix (a non-transformer static
  lookup-table model). `sct`'s person recall (0.0571) is nearly identical
  to m2v's (0.0572) — both essentially cannot do named-entity retrieval.
- {bge_m3, qwen3, qwen3_0.6b} significantly beat everything below them on
  most metrics; jina_v5 sits ambiguously between tiers; congen/e5/e5_small
  form a tied middle tier; sct/m2v form a tied bottom tier.
- **Cost-efficiency headline for RQ2**: the two "smaller ties larger" results
  mean the *cheapest* member of each strong family (qwen3_0.6b, e5_small) is
  the better pick over its own larger sibling by cost, with no proven
  quality loss — a sharper, more citable RQ2 finding than "biggest doesn't
  automatically win" (the original 252-set read).

**Best single dense-alone combo in the full 36-combo matrix, refreshed
2026-07-25**: `semantic × qwen3` (recall@10=**0.6612**, MRR=**0.8895**,
nDCG@10=**0.7386**, up slightly from 0.6581 pre-rebuild) — still the clear
leader, but **2nd place changed**: `semantic × qwen3_0.6b` is now clearly
2nd (recall@10=0.6435, MRR=0.9018 — actually the single highest MRR of any
dense combo), ahead of `semantic × jina_v5` (0.5884), which held 2nd place
pre-rebuild. `qwen3_0.6b`'s per-chunker dense-alone number was previously
reported as 0.6364 (below qwen3's 0.6581); post-refresh it's 0.6435, still
behind qwen3(4B) but by a smaller margin and clearly ahead of every other
embedder's semantic-chunker dense score. `qwen3` still keeps its
dense-alone per-chunker lead over `qwen3_0.6b`, unlike the hybrid case
(above) where `qwen3_0.6b` numerically overtakes it. ("Cost / latency
characterization" below used to lag this section by a refresh; it is current
as of 2026-08-09 and agrees — `qwen3` 0.5396 vs `qwen3_0.6b` 0.5707 dense on
the `semantic` combo.)

## Embedder × entity_type profile (the "specialist vs generalist" finding)

Cross-chunker average recall@10, broken out by query entity_type —
**refreshed 2026-07-25 against the clean, rebuilt indices**. Full table:
`data/results/gold_embedder_breakdown_9way.md`.

| embedder | faculty_adjunct (n=13) | person (n=30) | program (n=30) |
|---|---|---|---|
| bge_m3 | 0.4519 | **0.5670** | 0.4795 |
| qwen3 | 0.4826 | 0.4806 | 0.5668 |
| qwen3_0.6b | 0.4611 | 0.4361 | **0.6391** |
| congen | 0.3846 | 0.2677 | 0.5777 |
| sct | 0.2622 | 0.0571 | 0.2084 |
| jina_v5 | 0.4254 | 0.4350 | 0.4920 |
| e5 | 0.4571 | 0.4811 | 0.3739 |
| e5_small | 0.4528 | 0.4693 | 0.4084 |
| m2v | 0.2263 | 0.0572 | 0.2074 |

**Per-entity_type significance, full 9-embedder matrix (Holm-corrected per
entity_type × metric, 36-pair families), re-run 2026-07-25**. Full table:
`data/results/embedder_significance_test_by_entity_type_9way.md`; script:
`tools/eval/embedder_significance_test_by_entity_type_9way.py`. **Every
claim in the bullets below was individually re-verified against the
refreshed significance table and held exactly** (bge-m3 vs qwen3(4B) tie on
person still not significant, Holm-adj p=0.293; bge-m3 vs qwen3_0.6b on
person still significant, Holm-adj p=0.000; the program 3-way tie among
congen/qwen3/qwen3_0.6b still holds, all pairwise Holm-adj p≥0.18) — no
conclusion in this subsection changed.

- **person**: bge-m3 significantly beats ConGen/e5/e5_small/jina_v5/sct/m2v
  (all Holm-adj p<0.02). **bge-m3 vs qwen3(4B): NOT significant**
  (Holm-adj p=0.374 — ties, as before). **bge-m3 vs qwen3_0.6b: IS
  significant** (Holm-adj p<0.0001, bge-m3 wins by +0.14) — the "ties
  bge-m3 on person" property belongs to the 4B model specifically, **the
  0.6B model does not share it.**
- **program**: the top is now a **3-way tie**: `congen`, `qwen3`, and
  `qwen3_0.6b` are mutually NOT significantly different from each other
  (congen-vs-qwen3 raw p=0.881; congen-vs-qwen3_0.6b Holm-adj p=0.298;
  qwen3-vs-qwen3_0.6b Holm-adj p=0.188) despite qwen3_0.6b's numerically
  higher mean (0.6396 vs congen's 0.5732 and qwen3's 0.5682). bge-m3 loses
  to all three significantly here (its proven weak spot).
- **Consequence — qualifies the aggregate "smaller ties larger" headline
  above**: `qwen3_0.6b` joins the program top tier but **not** the person
  top tier, while `qwen3(4B)` holds both. The two models' *aggregate*
  scores tie (see Embedders compared section) because qwen3_0.6b's program
  gain offsets its person loss relative to qwen3-4B — but that's an
  averaging coincidence, not evidence the 0.6B model is a strict free
  lunch. **Only qwen3(4B) is the embedder with no statistically provable
  weak spot across both main query categories; qwen3_0.6b has one
  (person), just like bge-m3 has one (program).** For a person-heavy or
  mixed-uncertain workload, this favors the 4B model over its cheaper
  sibling despite the tied aggregate.
- **sct vs m2v ties in every entity_type, not just on average**: person
  (Holm-adj p=1.0), program (Holm-adj p=1.0), faculty_adjunct_aggregate
  (Holm-adj p=1.0) — confirms sct's bottom-tier status isn't an artifact of
  one category dragging the average down; it's uniformly weak.
- **Headline finding (unchanged)**: **Qwen3-Embedding-4B remains the only
  embedder statistically indistinguishable from BOTH category specialists
  in their own strongest category, simultaneously** — ties bge-m3 on person
  AND ties ConGen (now also qwen3_0.6b) on program.
- **Practical framing for the paper**: with reliable entity-type routing
  ([[project_hybrid_routing]]), specialist-per-route (bge-m3 for person,
  ConGen **or** qwen3_0.6b for program) matches or beats Qwen3-4B at much
  lower inference cost. Without reliable routing, **qwen3(4B) specifically**
  — not its 0.6B sibling — is the safer unrouted choice, since it alone
  lacks a provable weak spot across both main categories.
  **Two qualifications added 2026-08-08** (see the Query routing section):
  classification is now reliable — 0/106 unrouted, exact on all four types —
  but (a) an end-to-end routed system does **not** significantly beat simply
  using the best single combo for everything (+0.0082 recall@10, Holm-adj
  p=1.0000), so this framing is about cost and coverage, not about accuracy;
  and (b) **which specialist wins is retriever-dependent** — person peaks at
  semantic+qwen3 under dense but sentence+bge_m3 under hybrid — so name the
  retriever whenever citing a per-route specialist.

### Structural recall@10 ceiling by entity_type (new, 2026-07-22)

Prompted by a user question ("is recall@10≈0.6-0.7 too low?") — worth
stating explicitly because it changes how every recall@10 number in this
doc should be read. **The Gold 73-det set is not one-relevant-doc-per-query**:
each query has on average **8.8 relevant resolutions** (min 2, max **43**),
because several queries (especially `faculty_adjunct_aggregate`, which asks
for "every resolution appointing adjunct faculty at X") are genuinely
list-type questions with many correct answers. `recall_at_k` divides hits by
*total relevant count*, so **a query with 43 relevant resolutions can score
at most 10/43 ≈ 0.23 recall@10 even for a hypothetically perfect
retriever** — there are only 10 chunk slots to place possibly-dozens of
correct answers into.

Computed the resulting **theoretical ceiling** (mean of `min(1.0, 10/n_relevant)`
across queries) per entity_type:

| entity_type | n queries | avg relevant docs | max relevant docs | **ceiling recall@10** |
|---|---|---|---|---|
| person | 30 | 6.0 | 13 | **0.9760** |
| program | 30 | 8.2 | 24 | **0.8979** |
| course | 33 | 12.2 | 35 | **0.8729** |
| faculty_adjunct_aggregate | 13 | 16.8 | 43 | **0.6810** |
| **all 106 (mean)** | 106 | 9.9 | 43 | **0.8856** |

*(Recomputed 2026-08-08 from `config/eval/gold_query_set_73det.yaml` after
`tools/eval/audit_doc_claims.py` flagged the mean as untraceable. The table had
been written when the set held 73 queries and was never extended when the 33
`course` queries landed — so the old `all 73 (mean)` of 0.8922 was a ceiling for
two-thirds of the set. `program` also moved 0.9000 → 0.8979 with the
`resolution_id` repair. `person` and `faculty_adjunct_aggregate` are unchanged,
and the reading below is unaffected: `course` sits mid-table, so nothing about
`faculty_adjunct_aggregate` being the binding constraint changes.)*

**Reading this**: the low-looking recall@10 numbers for
`faculty_adjunct_aggregate` in every table above (e.g. bge_m3 dense-alone
0.4555, congen 0.3966) are **partly a metric artifact, not purely a
retrieval-quality gap** — no retriever, however good, can exceed ~0.68 on
this category under a k=10 window. By contrast, `person` queries have a
ceiling of 0.976 (near 1.0, since most person queries have few correct
answers) — so the current person-query recall@10 numbers (bge_m3 0.5694,
qwen3 0.4807) sit **much further below their own ceiling** than the
faculty numbers do below theirs. **This means person-query retrieval has
more genuine, addressable headroom than the raw numbers next to
faculty_adjunct_aggregate make it look** — the two categories' distance
from 1.0 is not directly comparable without this ceiling.

**Resolved 2026-07-29**: the ceiling above was originally computed against the
dense-alone breakdown, the only one that existed. BM25 and hybrid have now
been broken out by entity_type too
(`tools/eval/bm25_hybrid_entity_type_breakdown.py`, full table:
`data/results/bm25_hybrid_entity_type_breakdown.md`, pure recompute from
persisted results). Ceilings are recomputed there over the full 106-query
set, which adds the `course` category (ceiling 0.8729) that postdates the
73-query table above.

**Ceiling attainment — the comparable quantity across categories** (best
system per category; `% of ceiling` = recall ÷ ceiling):

| entity_type | ceiling | best hybrid | recall | % of ceiling | best dense | % of ceiling | BM25 alone | % of ceiling |
|---|---|---|---|---|---|---|---|---|
| person | 0.9760 | bge_m3 | 0.8220 | **84.2%** | bge_m3 (0.5735) | 58.8% | 0.8147 | **83.5%** |
| faculty_adjunct_aggregate | 0.6810 | jina_v5 | 0.4939 | **72.5%** | qwen3 (0.4729) | 69.4% | 0.4234 | 62.2% |
| program | 0.8979 | qwen3_0.6b | 0.6098 | **67.9%** | qwen3_0.6b (0.6034) | 67.2% | 0.3497 | 38.9% |
| course | 0.8729 | qwen3_0.6b | 0.5723 | **65.6%** | qwen3_0.6b (0.5514) | 63.2% | 0.3585 | 41.1% |

*(Refreshed 2026-08-06 against `chunker_compare_full` rebuild #3's corrected
`resolution_id`s — same source script, `bm25_hybrid_entity_type_breakdown.py`
re-run against already-current retrieval results; see
[[project_eval_refresh_2026_08_06]]. Movement is small (≤0.6 pp on 3 of 4
rows) and the ranking/story below is unchanged, with one attribution change:
`faculty_adjunct_aggregate`'s best hybrid embedder is now `jina_v5`, not
`qwen3_0.6b` — the two were close enough (0.4922 vs 0.4939 pre-refresh) that
this is noise around a near-tie, not a new finding. `program`'s ceiling also
corrected 0.9000→0.8979, a rounding fix in the source script, not new data.)*

**This reverses the headroom reading in the paragraph above, which was
dense-alone-specific.** Under the actually-recommended system (hybrid),
`person` is the *most* solved category at 84.2% of its ceiling, not the one
with the most addressable headroom — dense-alone's person weakness (58.8%)
is almost entirely repaired by fusing BM25. The category with the most real
headroom left is now **`course`** (65.6%), which did not exist when the
original ceiling analysis was written. Hybrid does also close the
`faculty_adjunct_aggregate` gap (62.2% BM25 → 72.5% hybrid), answering the
second open question.

**Two findings that only this breakdown makes visible:**

1. **Direct evidence for the lexical/dense complementarity mechanism.**
   BM25 alone reaches **0.8147** on `person` — beating *every* dense
   embedder's dense-alone person score (best: bge_m3 0.5735) by a wide
   margin — while collapsing to **0.3497** on `program`, where dense
   nearly doubles it (qwen3_0.6b 0.6034). **BM25 carries person queries
   (exact name match); dense carries program queries.** This is the
   mechanistic explanation for the hybrid-beats-both result, and it is
   *direct* evidence, unlike the indirect proxies (rescue rate, union
   coverage, per-query correlation) used in the Open item #2 investigation
   that came back inconclusive.
2. **"Hybrid never hurts" is an aggregate statement, not a per-category
   one.** On `person` queries specifically, hybrid is *below* BM25-alone
   (0.8147) for most embedders — `qwen3_0.6b` 0.7264, `qwen3` 0.7342,
   `congen` 0.7211, `jina_v5` 0.7382 — with only `bge_m3` (0.8220)
   exceeding it and `e5`/`e5_small` (0.8105/0.8051) roughly matching it.
   Fusing a dense signal that is weak on a category can drag that
   category below the BM25 baseline even when the cross-category
   aggregate improves. This is the same failure shape as the
   `sct`/`m2v` RRF cases, but occurring *within* an otherwise-strong
   embedder, per category — worth stating as a limitation of the headline
   hybrid recommendation.

**Implication for future work (not started, candidate direction beyond the
current Tier 1-3 plan)**: because `faculty_adjunct_aggregate` queries are
inherently "list all X" questions, a **top-k similarity search is the wrong
retrieval paradigm for that category almost by construction** — no amount
of better embedding or chunking can push its ceiling above ~0.68. The
`faculty`/`program`/`person` entity taggers already built for this project
(`people.json`, `faculties.json`, `programs.json` — see
[[project_faculty_tagger]], [[project_program_tagger]],
[[project_person_tagger]]) already provide the entity-indexed lookup
infrastructure a structured/graph-style retrieval path would need: for a
detected "list all" query, filtering directly by an entity tag (exact
lookup) rather than top-k semantic/lexical similarity could raise this
category's ceiling toward 1.0. This has **not been tested or built** — it's
a plausible, well-motivated candidate for a future research direction, not
a validated finding.

### Oracle-union ceiling: how much of the Gold set is reachable at all (new, 2026-08-08)

The ceiling above is imposed by the **qrels** (a system sending 10 documents
cannot recall 43 relevant ones). A second, independent ceiling is imposed by
the **index family**: union the persisted top-10 of all 36 live
chunker × embedder combos and any pair no system finds is out of reach of every
reranker, ensemble or fine-tune while the indices and k stay fixed. Script
`tools/eval/oracle_union_ceiling.py` → `data/results/oracle_union_ceiling.md`;
106 queries, 1,046 (query, resolution) pairs, hybrid retriever, read-only over
persisted results.

**The two ceilings do not conflict, and the way they meet is the check.** Every
row that *sends* 10 documents sits under 0.8856 (highest: 0.8355, a perfect
reranker over the full 360-document pool); the union rows exceed it only because
they send 360. The script recomputes 0.8856 from the qrels and gates on this
(check S5), so the two tables cannot drift apart.

| arm | docs sent | docs fetched | recall@10 macro | micro |
|---|---|---|---|---|
| best single combo (`sentence` × `qwen3_0.6b`) | 10 | 10 | 0.6229 | 0.4866 |
| oracle picks the best combo per query | 10 | 10 | 0.7775 | 0.6243 |
| union of all 36 + a perfect reranker | 10 | 360 | 0.8342 | 0.6912 |
| union of all 36 (no budget cap) | 360 | 360 | 0.8916 | 0.8346 |
| + dense results too (72 systems) | 720 | 720 | 0.9359 | 0.9063 |
| + BM25 as well (76 systems) | 760 | 760 | **0.9418** | 0.9130 |

Levels are the 2026-08-18 re-run against rebuild #4; every conclusion below is
unchanged and the point estimates moved by under 0.005.

Three results worth citing, and one retraction.

1. **Diversity is not free, and at a fixed budget it is negative.** Splitting
   the same 10 document slots across 2 systems (5 each) scores **0.5936** against
   a single system's **0.6229** — **−0.0294**. Doubling the budget instead (2
   systems × 10 = 20 documents) gives **+0.1196**. The earlier reading that an
   ensemble beat a single model compared 20 documents against 10. Both arms here
   are chosen greedily *on the test set*, so the comparison is biased toward
   diversity and it still loses. See [[feedback_state_the_retrieval_budget_in_every_comparison]].
2. **The misses are mostly ranking, not absence.** Of 1,046 pairs the best single
   combo finds 509 (48.7%) and the 36-way union finds 873 (83.5%): **364 pairs
   (67.8% of the misses) are found by *some* system**. That is the headroom a
   per-query router or a reranker over a merged pool is aiming at — but at the
   real 10-document budget its ceiling is 0.7775–0.8342, **not** 0.8916.
3. **The floor is 91 pairs (8.7%)**, not the 173 the hybrid-only union
   suggests. Adding the dense and BM25 result sets recovers 82 further pairs, so
   most of "no system found it" was a *retriever-choice* artifact. **This line
   read 76 (7.3%) until 2026-08-09**, subtracting 8 pairs called unanswerable by
   construction; that premise was measured and refuted (below), so the
   subtraction is withdrawn and those 8 belong in the floor. **These were also
   called a *structural* floor until 2026-08-08; that word is withdrawn too** —
   the depth profile below shows they are ranked, not absent.

**Retracted from the earlier version of this analysis**: its best single of
0.6935 is superseded by the 0.6229 above, and its ceiling of 0.9201 is retired
in favour of the 0.8916 above — do not cite either. The cause is the query set:
that analysis ran against a checkout whose Gold set still held **73** entries,
and the 33 `course` queries added on 2026-07-25 are the harder ones. Scoring
only those 73 non-course queries with the combo set used here reproduces its
shape (best 0.6728, union 0.9125), so the difference is coverage, not method. Its leave-one-out router
(+0.0465) is withdrawn in favour of `routing_eval.md`'s tested `routed (loo)`
= 0.6780 (+0.0499), which does **not** reach significance.

#### How deep are the misses? (`tools/eval/miss_depth_profile.py`, 2026-08-08)

The item above left one question open: of the pairs no system returns at k=10,
how many are ranked just outside the cut and how many are effectively absent?
**It was not answered with a k=50 run.** `DenseRetriever` and `BM25Retriever`
both score the whole corpus and then `argsort(-scores)[:k]`, so k=50 costs
exactly what an exhaustive rank costs while destroying the one distinction the
question is about — rank 51 versus rank 40,000. The script therefore recomputes
**untruncated ranks** for all 36 combos × 3 arms, and pins itself to the
persisted results first: dense, BM25 and hybrid top-10 must each reproduce
byte-for-byte (3,816 / 848 / 3,816 reproduce, 0 differ), and the all-arm and
hybrid-only counts must agree with the ceiling report from an independent code
path. **Figures below are the 2026-08-18 re-run against rebuild #4**; the
pre-rebuild counts were 84 all-arm / 164 hybrid-only, and the *shape* held
across the rebuild while both counts rose — a pair no arm reaches is a property
of the qrels **and** of the text, and rebuild #4 re-OCR'd a meeting.

A resolution's rank is its **best chunk's** rank, because the budget is counted
in chunks (k=10 chunks ≈ 7 distinct resolutions).

| best rank achieved by any arm | of the 91 all-arm misses | of the 173 hybrid-only misses |
|---|---|---|
| 11–50 | **71 (78.0%)** | 138 (79.8%) |
| 51–100 | 11 | 20 |
| 101–1000 | 8 | 14 |
| 1001+ | 1 | 1 |
| not in the index at all | **0** | **0** |

**The floor is ranking depth, not absence.** Every one of the 91 pairs has a
chunk in every index, 90 of 91 sit inside the top 1,000, and 78% sit
at ranks 11–50 — reachable by any reranker willing to fetch 50 candidates. Only
one pair is genuinely deep (`รายวิชา CALCULUS 2` → a 2568 curriculum-revision
resolution, best rank **2,988**). This is why "structural" was withdrawn above.

**How deep a candidate pool a reranker would need — and what it could actually
deliver.** These are two different quantities and the first version of this
table conflated them. *In pool* is the fraction of gold sitting inside a
candidate pool of size P: the reranker's raw material, unbounded by the output
budget. *Delivered* is recall@10 after a **perfect** rerank that still returns
only 10 documents, so it is capped by the qrels ceiling 0.8856 for the same
reason every other 10-document row is (macro; best single = `sentence` ×
`qwen3_0.6b`, hybrid):

| P | single, in pool | single, **delivered** | all arms, in pool | all arms, **delivered** |
|---|---|---|---|---|
| 10 | 0.6229 | **0.6229** | 0.9418 | **0.8605** |
| 20 | 0.7720 | **0.7510** | 0.9678 | **0.8711** |
| 50 | 0.8896 | **0.8268** | 0.9837 | **0.8783** |
| 100 | 0.9169 | **0.8356** | 0.9925 | **0.8814** |
| 1000 | 0.9798 | **0.8738** | 0.9990 | **0.8846** |

*(re-run 2026-08-18 against rebuild #4; the pre-rebuild row at P=50 read
0.8869 / 0.8249 / 0.9837 / 0.8783.)*

**Cite the delivered column.** A perfect reranker over a 50-document pool from
one system is worth **0.6229 → 0.8268**, and going ten times deeper (P=1000)
buys only 0.8738 — the budget, not the pool, is what binds. The "in pool"
column crosses 0.8856 at P=50 and reaches 0.9798, which is why it must never be
quoted as a reranker's ceiling. Check S7 gates every delivered cell against the
0.8856 the script recomputes from the qrels.

Two further facts, both of which change where effort should go:

- **`person` has zero misses.** All 180 `person` pairs are found at k=10 by some
  arm. The 91 are `course` 41, `faculty_adjunct_aggregate` 28, `program` 22 —
  and the three types fail differently: `course` is almost purely a near-miss
  (40 of 41 at ranks 11–50), while `faculty_adjunct_aggregate` splits evenly
  (14 at 11–50, 14 deeper) — **that even split is unmoved by the rebuild**.
  A reranker helps `course`; it will not rescue half of `faculty`.
- **The candidate pool should come from dense, not from the shipped hybrid.**
  On exactly these hard pairs, `dense` has median best rank **22** and is the
  closest arm on **74 of 91**; `hybrid` is 39 (13 pairs) and `bm25` is 200 (6).
  The arm that wins the published recall@10 tables is not the arm that gets
  nearest on what they miss.

**Standing caveat before anyone reads this as free headroom.** This project
already measured a real cross-encoder over hybrid candidates and it
*significantly hurt* MRR (0.7730 → 0.6940, Holm-adj p=0.0240) while doing
nothing for recall@10. The depth profile says the evidence is within reach at
P=50; it does not say the reranker that was tried can reach it.

#### A qrels defect this surfaced: two course names, one a prefix of the other

One query — `รายวิชา CONTROL SYSTEMS` — scores **0.000** under the union of all
36 systems. The cause is not retrieval. The Gold set also contains
`รายวิชา CONTROL SYSTEM` (singular), the two names differ by one character, and
because the qrels were built by exact-token match their relevant sets are
**disjoint — 0 documents in common**. The union retrieves 103 documents for the
plural query: 0 of its own gold, 9 of the *other* query's. No system that cannot
tell the two questions apart can score above zero on one of them.

**Corrected 2026-08-09.** This paragraph used to end "so those 8 pairs are a
labelling artifact and are excluded from the structural floor above", and both
halves of that were wrong. `tools/eval/audit_gold_anchor_ambiguity.py` measured
the premise: the course qrels reproduce **exactly** from the code tags (33 of 33
queries), the two codes are genuinely different courses, and **all 8 of the
plural query's relevant documents literally contain the phrase `CONTROL
SYSTEMS`**. Nothing is mislabelled, and the query is *not* unanswerable by
construction — a system that could tell which `CONTROL SYSTEMS` is course
`01306023` would find all 8. What it is not is answerable **by name matching
alone**, because 65 documents in the corpus show that phrase and only 8 are
judged relevant (anchor precision 0.123). Those 8 pairs are therefore counted in
the floor of 84, not subtracted from it.

The general defect is a **key mismatch**, not a bug: `course` is the only entity
type whose qrels are keyed on something the query never supplies — the 8-digit
code, against a query that gives the name. The other three types
(`program`, `person`, `faculty_adjunct_aggregate`, 73 of 106 queries) judge
relevance on exactly the string the query provides and are unexposed by
construction. Across the 33 exposed queries, 3 have anchor precision below 0.5
and 4 have gold documents that never spell the course name at all — a separate
mechanism that no word-matching system can reach and that dropping queries would
not fix. Full measurement: `data/results/gold_anchor_ambiguity.md`. Nothing was
dropped from the Gold set; the price of doing so was measured first and it
*raises* every published number (+0.0050 for the one query, +0.0113 for all
three), because the queries in question are the low-scoring ones.

A second hypothesis was tested here and **rejected by the data**: 38 of the 401
`course` gold pairs (9.5%) are relevant only because the course name appears in
another course's `PREREQUISITE:` line, which looked like an unretrievable
needle. It is not — `SIGNALS AND SYSTEMS` is 9/10 prerequisite-only and the
union recalls **1.000**, `ELECTRONICS ENGINEERING 1` is 10/10 and scores 0.900.
Report the 9.5% as a category that exists, not as an explanation of anything.

## BM25 lexical baseline

`src/rag_lab/retrievers/bm25.py` (`rank_bm25.BM25Okapi` over PyThaiNLP
`word_tokenize`, engine `newmm` — dictionary-based maximum matching
constrained by Thai Character Cluster boundaries, tokenizes the full chunk
text, not just title/metadata). One run per chunker (embedder-agnostic) —
**refreshed 2026-07-29 against the OCR-remediation-rebuilt indices** (see
methodology caveat above — the 2026-07-25 numbers this table used to show
were themselves stale after the 2026-07-28 rebuild):

| chunker | recall@10 | mrr | ndcg@10 |
|---|---|---|---|
| recursive | 0.5096 | 0.6633 | 0.5333 |
| sentence | 0.5041 | 0.7123 | 0.5527 |
| fixed_size | 0.4965 | 0.6816 | 0.5341 |
| semantic | 0.4620 | 0.6326 | 0.4898 |

**BM25 aggregate (averaged across its 4 chunker runs, same framing as the
embedder table above)**: recall@10=**0.4930** (up from a stale 0.3908 —
see methodology caveat), mrr=0.6725, ndcg@10=0.5275. (BM25 is chunker-only /
embedder-agnostic — this table doesn't change when the embedder matrix
grows; extended below is only the significance comparison, now against all
9 embedders.)

**Significance (9 BM25-vs-embedder tests, Holm-corrected per metric —
`tools/eval/bm25_vs_embedder_significance_test_9way.py`, re-run 2026-07-29
against the OCR-remediation-rebuilt indices)**:

| vs. | recall@10 diff (BM25 − X) | Holm-adj p | significant |
|---|---|---|---|
| sct | +0.3920 | 0.0000 | **yes** |
| m2v | +0.3749 | 0.0000 | **yes** |
| congen | +0.2251 | 0.0000 | **yes** |
| e5_small | +0.1251 | 0.0000 | **yes** |
| e5 | +0.1026 | 0.0010 | **yes** |
| bge_m3 | +0.0840 | 0.0216 | **yes** |
| jina_v5 | +0.0796 | 0.0528 | no |
| qwen3 | +0.0153 | 0.7064 | no |
| qwen3_0.6b | -0.0332 | 0.7064 | no |

**Headline finding changed with this refresh**: BM25 no longer ties
`bge_m3` — it now **significantly beats bge_m3** (+0.0840 recall@10,
Holm-adj p=0.0216), on top of significantly beating every weaker embedder
it already beat before (ConGen, e5, e5_small, m2v, sct). It statistically
ties only the top Qwen3 pair (`qwen3`, `qwen3_0.6b`) and `jina_v5` (p=0.0528,
just short of the bar). The mechanism is the same one flagged in the
methodology caveat: **lexical matching is far more sensitive to OCR token
corruption than dense embeddings, so the OCR-remediation rebuild lifted
BM25's own score (0.3908→0.4930) by more than it lifted any embedder's** —
narrowing what "BM25 ties the top tier" can honestly claim to just
{qwen3, qwen3_0.6b} (and arguably jina_v5). Framed for the paper: *only an
embedder in the Qwen3 tier is still provably worth its inference cost over
plain BM25 on this corpus; bge-m3 has lost that status, and every weaker
embedder was already behind.* `sct`'s dense-alone recall@10 is 0.1010 (down
from a stale 0.1558), still statistically indistinguishable from m2v's
0.1181 (both near-random) — sct remains an embedder BM25 beats by a very
wide margin (+0.39 recall), not just a modest one. Working hypothesis for
why BM25 is this strong unchanged: Gold queries are entity-anchored — even
though phrasing is rephrased away from document titles, the anchor entity's
literal name (person/program/faculty) has to stay verbatim to specify which
resolution is being asked about, which gives exact lexical match a
structural advantage on this specific task — and a corpus-wide OCR cleanup
sharpens that advantage further, since garbled tokens hurt lexical matching
more directly than they hurt dense embeddings.

### Per-chunker BM25 vs. embedder (resolves Open item #1 — the "tie" is chunker-dependent)

**Refreshed 2026-07-29** against the OCR-remediation-rebuilt indices —
re-ran `tools/eval/bm25_vs_embedder_significance_test_per_chunker.py`, pure
recompute from the newly-refreshed persisted results (`gold_bm25_73det`,
`gold_73det_full_embedder_matrix`). The 2026-07-25 numbers below were
themselves stale after the 2026-07-28 rebuild (see methodology caveat) —
this is a real second refresh, not a re-verification.

The aggregate table above averages each system across the 4 chunkers before
testing, which can hide a real interaction if BM25's advantage differs by
chunker. Ran `tools/eval/bm25_vs_embedder_significance_test_per_chunker.py`
— 4 independent 9-test families (one per chunker), each Holm-corrected
separately per metric. Full table:
`data/results/bm25_vs_embedder_significance_test_per_chunker.md`.

**recall@10, Holm-adj p per chunker (bold = BM25 significantly beats that embedder)**:

| embedder | fixed_size | recursive | semantic | sentence |
|---|---|---|---|---|
| e5 | +0.0674 (ns) | +0.1161 (**sig**) | +0.1036 (**sig**) | +0.1233 (**sig**) |
| e5_small | +0.1464 (**sig**) | +0.1306 (**sig**) | +0.0814 (ns) | +0.1371 (**sig**) |
| bge_m3 | +0.0838 (ns) | +0.0940 (ns) | +0.0489 (ns) | +0.1080 (**sig**) |
| congen | +0.2360 (**sig**) | +0.2374 (**sig**) | +0.1666 (**sig**) | +0.2619 (**sig**) |
| jina_v5 | +0.1247 (**sig**) | +0.0824 (ns) | −0.0057 (ns) | +0.1164 (**sig**) |
| qwen3 | +0.0580 (ns) | +0.0219 (ns) | −0.0753 (ns) | +0.0534 (ns) |
| qwen3_0.6b | −0.0116 (ns) | −0.0184 (ns) | **−0.1065 (sig)** | −0.0000 (ns) |
| sct | +0.3969 (**sig**) | +0.4235 (**sig**) | +0.3369 (**sig**) | +0.4109 (**sig**) |
| m2v | +0.4000 (**sig**) | +0.3663 (**sig**) | +0.3324 (**sig**) | +0.4016 (**sig**) |

**Two findings the aggregate table can't show, both confirmed again post-refresh
(with one new wrinkle)**:

1. **`bge_m3` still loses to BM25 significantly under `sentence` chunking
   specifically** (Holm-adj p=0.0060, diff +0.1081), and ties BM25 in
   `fixed_size`/`recursive`/`semantic` — same qualitative pattern as before
   the OCR rebuild, though the aggregate table above no longer frames this
   as "BM25 ties bge_m3 overall" (that tie is gone at the aggregate level
   too now — see the BM25 section above).
2. **`qwen3` and `qwen3_0.6b` remain the only embedders where BM25's margin
   goes *negative* under `semantic` chunking** — but this is no longer just
   a numerical trend: for `qwen3_0.6b` it is now **statistically
   significant** (Holm-adj p=0.0060, BM25 −0.1067 behind) — the first
   chunker/embedder cell in this whole comparison where an embedder
   significantly *beats* BM25 outright. `qwen3` (4B) is still only
   numerically ahead under `semantic` (−0.0761, not significant).

**Practical framing, updated**: "BM25 ties the top embedder tier" is now
even more clearly not chunker-invariant, and the direction has sharpened
post-OCR-fix: BM25's lexical edge grew everywhere the corpus text got
cleaner, which pulled it *ahead* of embedders it used to tie
(`bge_m3` overall, see BM25 section above) while `semantic` chunking remains
the one place a strong dense embedder (`qwen3_0.6b`) can still significantly
out-recall it. This still supports the project's semantic-chunking
recommendation for dense/hybrid retrieval, but for a sharper reason than
before: it's now the *only* chunker where a dense embedder demonstrably
earns its cost over free BM25 on this corpus.

## Hybrid retrieval (RRF: BM25 + Dense) — the overall best system found

`src/rag_lab/retrievers/hybrid.py` (Reciprocal Rank Fusion, `rrf_k=60`
default) fuses BM25 and dense rankings from the **same** index — no rebuild
needed, every combo already carries both `embeddings.npy` and
`lexical.json`. Full 24-combo matrix run on Gold 73-det
(`tools/eval/run_gold_hybrid_eval.py`), extended 2026-07-21 with 12 more
combos for `e5_small`, `qwen3_0.6b`, `sct` (at its corrected
max_seq_length=510) via `tools/eval/run_gold_hybrid_eval_9way_new.py` — 36
combos total, all 9 embedders now covered.

**Aggregate recall@10 (averaged across the 4 chunkers), hybrid vs. its two
components — refreshed 2026-07-29 against the OCR-remediation-rebuilt
indices** (the 2026-07-25 table this replaced was itself stale after the
2026-07-28 rebuild — see methodology caveat above; this is a real second
refresh):

| embedder | hybrid | dense-alone | bm25-alone |
|---|---|---|---|
| **qwen3_0.6b** | **0.6167** | 0.5263 | 0.4930 |
| qwen3 | 0.5945 | 0.4777 | 0.4930 |
| jina_v5 | 0.5831 | 0.4135 | 0.4930 |
| e5 | 0.5753 | 0.3905 | 0.4930 |
| bge_m3 | 0.5730 | 0.4090 | 0.4930 |
| e5_small | 0.5658 | 0.3679 | 0.4930 |
| congen | 0.4692 | 0.2679 | 0.4930 |
| sct | 0.3939 | 0.1010 | 0.4930 |
| m2v | 0.3028 | 0.1181 | 0.4930 |

**Significance** (`tools/eval/hybrid_significance_test_9way.py`, re-run
2026-07-29 against the OCR-remediation-rebuilt indices, two 9-test
families — hybrid vs. dense-alone, hybrid vs. BM25-alone — Holm-corrected
separately per metric):

- **Hybrid significantly beats dense-alone for essentially every one of the
  9 embedders, on every metric** (still 26/27 tests significant — same
  count as before, but **the sole exception moved**: it is now
  `qwen3_0.6b` on MRR (Holm-adj p=0.304), not `qwen3` (which is now
  significant on MRR too, Holm-adj p=0.0008)). This remains the single most
  robust finding in the whole study: **adding BM25 to a dense retriever
  never hurts and almost always helps significantly**, even for the two
  very weak dense-alone models (sct, m2v).
- **Hybrid vs BM25-alone shifted more substantially**: `qwen3_0.6b`, `qwen3`,
  `jina_v5`, `e5`, `bge_m3`, `e5_small` all still significantly beat BM25
  on recall@10 — but `jina_v5` is now clearly significant (+0.0901, Holm-adj
  p=0.0000; it used to sit right at the edge of significance) and `congen`
  has dropped **out** of the significant group (+diff no longer
  distinguishable from BM25 on any metric — BM25 itself got stronger
  post-OCR-fix, closing that gap). `sct` and `m2v` remain the two
  cautionary counter-examples where hybrid is significantly *worse* than
  BM25-alone, and **`sct`'s recall@10 deficit is significant again**
  (Holm-adj p=0.0000 — the 2026-07-25 table's "no longer significant,
  p=0.08" note no longer holds; it was itself computed against the
  since-superseded index). The underlying mechanism is unchanged: RRF
  fusion isn't automatically safe when one fused signal is weak enough (per
  the "Resolved 2026-07-21" section above) — if anything the OCR fix made
  BM25 strong enough that a weak dense signal drags hybrid down more
  visibly, not less.
- **Top single-combo across the entire study — this claim no longer holds
  as previously stated.** The 2026-07-25 table cited `semantic ×
  qwen3_0.6b` at recall@10=**0.7048** as the clear overall leader. That
  number came from the same "semantic + hybrid only" measurement as the
  table just below, and its fresh value is **0.6152** — a drop of ~0.09.
  Worse, it is no longer even the highest number among `qwen3_0.6b`'s own
  four chunkers: `sentence × qwen3_0.6b` now reads **0.6265** and
  `fixed_size × qwen3_0.6b` reads 0.6154, both numerically above
  `semantic`'s 0.6152 (`recursive × qwen3_0.6b` is lower, at 0.6097).
  **Resolved same day**: built the missing test
  (`tools/eval/hybrid_chunker_significance_test.py`, see "Chunkers compared"
  section above for the full writeup) — for `qwen3_0.6b` specifically, all
  4 chunkers are fully, mutually tied on every metric (Holm-adj p≥0.44
  throughout). **`sentence` is not the new winner either** — nothing wins.
  The same test run across all 9 embedders plus an aggregate family found
  `semantic` never significantly beats any other chunker anywhere; the only
  significant chunker-pairwise result in the whole test is `fixed_size`
  losing to `recursive`. **The dedicated per-chunker (semantic-only) top-5
  tie test was also re-run 2026-07-29** against the rebuilt indices
  (`hybrid_significance_test_semantic_top5.py`,
  `data/results/hybrid_significance_test_semantic_top5.md`) — the tie
  **partially broke**: `bge_m3` now loses significantly to `qwen3_0.6b`,
  `qwen3`, and `jina_v5` on recall@10 and nDCG@10 (though it still ties all
  three on MRR, Holm-adj p≥0.058) and drops out of the cluster. The
  remaining four — `qwen3_0.6b`, `qwen3`, `jina_v5`, `e5_small` — are still
  fully, mutually tied on every metric (no pair significant, lowest raw
  p=0.14). Updated semantic-only-hybrid means: `qwen3_0.6b` 0.6152, `qwen3`
  0.6051, `jina_v5` 0.5995, `e5_small` 0.5877, `bge_m3` 0.5451 recall@10.
  Don't crown a single embedder as "the best hybrid combo" — that guidance
  still holds, just for a 4-way cluster now instead of 5.

**Headline system recommendation for the paper, updated**: **semantic
chunking + hybrid retrieval (BM25 + dense, RRF)** is still a strong,
defensible system-level recommendation — lexical and dense signals remain
genuinely complementary (the hybrid-beats-dense-alone finding is unchanged),
and semantic chunking is still the one chunker where a strong dense
embedder demonstrably earns its cost over free BM25 (see "Per-chunker BM25
vs. embedder" above). What's **no longer supportable** is the stronger
claim that semantic chunking's *best combo number* is the single highest
recall@10 anywhere in the study — that specific claim was an artifact of
comparing against a now-stale hybrid table, and the fresh numbers put
`sentence`/`fixed_size` numerically ahead of `semantic` for `qwen3_0.6b`
specifically, untested. Which embedder to pair with the recommended system
is still an open, untested horse race among the top four above (narrower
than before — see Open item #8) — the system-level claim (hybrid retrieval,
BM25+dense) is what's robust; the specific chunker-wins-outright and
embedder-pick claims are not.

## Multi-k metrics (MAP, Precision@k, Recall@k, nDCG@k for k=1,3,5,10)

Every table elsewhere in this doc reports k=10 only. This section adds the
multi-k view (`tools/eval/multi_k_report.py`, full report at
`data/results/multi_k_report.md`) — a pure recompute over already-persisted
retrieval results (every combo was retrieved at `top_k=10`, so k≤10 needs no
new retrieval), closing gap-analysis Tier 1 item #1's last open tail.
**Refreshed 2026-07-29 (evening)**: this report had fallen through the same
crack as the BM25/hybrid caches did earlier that day — `multi_k_report.md`
still carried a **2026-07-22** mtime while `gold_hybrid_73det` had been
rewritten 2026-07-29, so every number below was one rebuild behind (a third
instance of the "not in the 5-script refresh chain" staleness pattern, after
BM25/hybrid and `cost_latency_pareto.py`). Re-ran it (no new retrieval
needed) — and re-ran it again **2026-08-23** against rebuild #4, which is where
the numbers below come from. **Both orderings survived; the levels moved.** Note
how the stale copy hid: BM25 `precision@1` had read 0.5849 since 07-29, and
`audit_doc_claims.py`'s D2 passed it the whole time because 0.5849 is still a
live figure — it is the `fixed_size` and `semantic` rows of the same report's
*per-chunker* table. A traceability check that asks only "does this number
appear somewhere" cannot see a number that has drifted onto a neighbour's cell.

**Dense-alone, top 3 embedders by recall@10 (aggregated across 4 chunkers)**:

| embedder | MAP | recall@1 | recall@3 | recall@5 | recall@10 | precision@1 | precision@5 | ndcg@1 | ndcg@5 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_0.6b | **0.4439** | 0.1208 | 0.2777 | 0.3731 | 0.5261 | **0.7453** | 0.5448 | **0.7453** | 0.6294 |
| qwen3 | 0.3869 | 0.1081 | 0.2547 | 0.3411 | 0.4786 | 0.6321 | 0.4778 | 0.6321 | 0.5537 |
| jina_v5 | 0.3146 | 0.0902 | 0.2045 | 0.2784 | 0.4145 | 0.5212 | 0.3953 | 0.5212 | 0.4575 |

**Hybrid (RRF), top 3 embedders by recall@10 (aggregated across 4 chunkers)**:

| embedder | MAP | recall@1 | recall@3 | recall@5 | recall@10 | precision@1 | precision@5 | ndcg@1 | ndcg@5 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_0.6b | **0.4947** | 0.1238 | 0.3011 | 0.4273 | 0.6162 | **0.7406** | 0.5934 | **0.7406** | 0.6738 |
| qwen3 | 0.4729 | 0.1136 | 0.2871 | 0.4128 | 0.5942 | 0.6958 | 0.5717 | 0.6958 | 0.6450 |
| jina_v5 | 0.4541 | 0.1119 | 0.2847 | 0.3919 | 0.5799 | 0.6698 | 0.5401 | 0.6698 | 0.6165 |

**BM25, aggregated across 4 chunkers**: MAP=0.3845, recall@1=0.1020,
recall@5=0.3383, precision@1=0.5967, ndcg@1=0.5967.

**Reading this**: the "mixed story" this section used to report is **gone**
in the refreshed data. Previously (stale, 2026-07-22 numbers, since
retracted): `bge_m3` had the highest MAP while `qwen3_0.6b` had the highest
precision@1 — a genuine cross-metric disagreement among the top 3. With
the current data, `qwen3_0.6b` now leads **every** metric in both tables
(MAP, precision@1/ndcg@1, and recall@10) among the top-3-by-recall@10
embedders, dense-alone and hybrid alike — the ranking is monotonic across
metrics now, not contradictory. This is consistent with `qwen3_0.6b`
breaking the old dense-alone 3-way tie (see "Embedders compared" above) and
`bge_m3` falling out of the top tier generally after the OCR-remediation
rebuild.

### MAP / precision@1 significance test (built 2026-07-29 — resolves the last untested metrics)

The two metrics above were reported but had **never been significance-tested**
— only recall@10/MRR/nDCG@10 ever were. Built
`tools/eval/map_precision_significance_test.py` (pure recompute from persisted
results; full table: `data/results/map_precision_significance_test.md`). It
runs **both scopes**, because the existing tied-cluster finding is scoped to
the `semantic` chunker only while the multi-k tables above aggregate across all
4 — so previously neither could speak to the other. Holm-corrected within each
(retriever, scope, metric) family.

| retriever / scope | metric | highest | significantly beats | ties |
|---|---|---|---|---|
| dense / aggregate | MAP | `qwen3_0.6b` (0.4457) | **8 of 8** | — |
| dense / aggregate | precision@1 | `qwen3_0.6b` (0.7406) | **8 of 8** | — |
| dense / semantic | MAP | `qwen3_0.6b` (0.4970) | 3 of 4 | `qwen3` |
| dense / semantic | precision@1 | `qwen3_0.6b` (0.7736) | 3 of 4 | `qwen3` |
| hybrid / aggregate | MAP | `qwen3_0.6b` (0.4948) | 4 of 8 | `qwen3`, `bge_m3`, `e5`, `e5_small` |
| hybrid / aggregate | precision@1 | `qwen3_0.6b` (0.7429) | **5 of 8** | `qwen3`, `bge_m3`, `e5` |
| hybrid / semantic | MAP | `qwen3` (0.5047) | 1 of 4 (`bge_m3`) | `qwen3_0.6b`, `jina_v5`, `e5_small` |
| hybrid / semantic | precision@1 | `qwen3` (0.7264) | **0 of 4** | all four |

**Three things this settles:**

1. **`qwen3_0.6b`'s dense-alone lead is stronger on MAP/precision@1 than on
   recall@10.** In the aggregate dense scope it significantly beats **all
   eight** other embedders on both metrics — a cleaner result than recall@10,
   where it beats only `bge_m3` and `qwen3`. The "qwen3_0.6b now leads every
   metric" reading of the multi-k tables above is therefore **confirmed as a
   tested claim for dense-alone**, not just a raw-mean observation.
2. **The tied cluster survives on the two new metrics.** At the scope the tie
   was actually claimed at (hybrid, `semantic`), `bge_m3` again loses on MAP
   (the same drop-out seen on recall@10/nDCG@10), and the remaining four are
   fully mutually tied; on precision@1 **nothing is significant at all**, all
   five tie. **Don't crown a single best hybrid embedder — that guidance now
   holds across all five metrics, not three.**
3. **The scope mismatch was real and matters.** `qwen3_0.6b` is highest at the
   aggregate scope but `qwen3` is numerically highest at the semantic scope on
   both new metrics — so "qwen3_0.6b leads every metric" is **an
   aggregate-scope statement only**, and neither difference is significant at
   the semantic scope anyway. Cite the scope alongside the claim.

Note also that hybrid *compresses* embedder differences relative to dense: the
same 9-embedder family goes from 8-of-8 significant (dense) to 4-of-8 (hybrid),
consistent with BM25 supplying a common floor that narrows the gap between
embedders.

## Cost / latency characterization

> **CURRENT AS OF 2026-08-09 — the whole table was re-measured on an idle
> machine and the mixed provenance below is retired.** Both the latency and
> the quality columns now come from one run. What changed, and why the older
> numbers in this section are still worth reading: `BM25Retriever` memoises
> its `BM25Okapi` on the `Index` instead of rebuilding it per query (commit
> `5cc71a1`), which removes ~1.0s from every BM25 and hybrid query, so **every
> BM25/hybrid latency figure dated 2026-07-29 or earlier is high by roughly
> that much**. Dense-alone columns never touched BM25 and are unaffected.
>
> **The headline speedup figure this project first published for `5cc71a1`
> was measured on the wrong query shape, and is corrected here.** The
> "26.2× / 1.073s vs 0.041s / BM25 `retrieve()` 1.094s → 0.050s p50, ~22×"
> figures came from feeding `get_scores` an 8-word slice of a chunk, which
> tokenizes to **3 terms**. `rank_bm25` loops over query *terms* in Python
> (touching all 74,816 doc-frequency dicts per term), so scoring is linear in
> query length at ~12 ms/token while the build is not — and the real Gold
> queries tokenize to **20 terms at the median** (min 13, max 30, n=106).
> Re-measured on those: build **1035.89 ms** vs `get_scores` **253.50 ms** =
> **4.1×**, and BM25-alone `retrieve()` p50 is **234.45 ms** (p95 332.78).
> **State the token count with any BM25 timing; the absolute saving (~1.0 s)
> transfers between query shapes, the multiplier does not.**
>
> The mechanism paragraph further down is *sharpened, not invalidated*: it
> attributed a ~1.92–2.03s fixed hybrid overhead jointly to "BM25Okapi
> rebuilt per query" **and** "hybrid k=n over-fetch", and that now splits into
> ~1.0s of rebuild (gone) and the over-fetch + Python fusion (still there, and
> *not* free to remove — `HybridRetriever` fetches k=n deliberately so RRF
> sees full rankings). Hybrid totals accordingly land at **1.21–1.86 s** here
> rather than the 2.08–2.68 s below.
>
> **Three timing controls ship in the report, because one was not enough.**
> Each embedder is now measured in **its own subprocess**, which removes the
> 08-07 position effect at its root. (1) A **reference probe** — an identical
> numpy workload in every child — held to **13.5%** spread (median 156.6 ms),
> confirming the machine's floor was steady. (2) A **repeat control** — the
> first embedder measured again last — caught what the probe could not:
> `bge_m3`'s own `search p50` rose **245.5 → 257.9 ms (+5.1%)** across the
> 45-minute run while its probe moved −0.4 ms. (3) **Same-dim consistency** —
> the seven dim-1024 embedders run the identical numpy op on identically
> shaped arrays, so their **10.3%** spread *is* the noise floor. **Treat
> ~5–10% as this rig's resolution**; a smaller cross-embedder gap is not real.
>
> One earlier claim is **withdrawn on this evidence**: "the k=n over-fetch tax
> is 66% of dense k=n cost in both runs" is **54%** here (dense k=10
> 262.46 ms vs k=n 575.58 ms). It is not a constant of the implementation —
> quote it from the current run.
>
> **Superseded history (2026-08-07): quality columns were current against
> rebuild #3; latency columns were deliberately still the 2026-07-29
> measurement.** Kept because it is the case that produced control 3. The
> re-run against rebuild #3 confirmed what this section already predicted — that
> the quality columns barely move (max |Δ| recall@10 = **0.0034** across all 18
> embedder cells, plus +0.0022 for BM25 alone; ordering identical on both dense
> and hybrid, `qwen3_0.6b` still highest on both) — and its
> latency columns were then **rejected as contaminated**, on evidence rather
> than suspicion:
>
> - `search p50` at dim=1024 is the same numpy operation on the same-shaped
>   array for six of the nine embedders. On 07-29 those six agreed to within
>   **1.9%** (241.2–245.7 ms). In the 08-07 run they spread **74.2%**
>   (301.3–525.0 ms), split exactly at run position 6: every embedder measured
>   before `qwen3` (the 4B model) sits at 301–317 ms, every one after it at
>   434–525 ms. The 4B model's memory is not released before the rest of the
>   loop is timed, so late-run embedders are charged for their position.
> - Underneath that, a uniform ~1.25× floor shift: a standalone numpy benchmark
>   on the same 74,816 × 1024 array, re-run afterwards on an idle machine,
>   reproduces the 08-07 figure (129 ms), not the 07-29 one (97 ms). The host is
>   slower now; that is not the rebuild either.
>
> The visible symptom is `m2v` appearing to cost *more* per hybrid query
> (4315 ms) than `bge_m3` (3325 ms) despite encoding in 4 ms — mechanically
> impossible, and the tell that made the run worth checking. **Both of the
> "two things that survive the rejected run" have since failed on re-measure**,
> and that is the lesson of this paragraph: the BM25 rebuild-vs-scoring ratio
> (22×, vs 24× on 07-29) held only because both runs used the same 3-token
> synthetic query, and the k=n over-fetch tax "66% in both runs" (558/847 vs
> 460/699) is 54% on 08-09. A quantity reproducing across two runs of the
> *same script* is weak evidence that it is a property of the system rather
> than of the script's own choices.

**Refreshed 2026-07-29** against the OCR-remediation-rebuilt indices —
`cost_latency_pareto.py` was skipped in both the 2026-07-25 and the
2026-07-29 BM25/hybrid-cache-fix refreshes (it's a cost/latency
measurement, not covered by the `embedder_matrix_9way.py` + siblings
chain), so its recall/nDCG columns had gone stale twice over. Re-run in
full (no `--reuse-latency-cache`) so both the quality columns and the
latency measurements reflect the current corpus. **Latency/cost mechanics
came back essentially unchanged** (e.g. `qwen3` encode p50 264.7ms vs. the
old 264ms, `e5_small` 24.7ms vs. 25ms) — confirming these numbers measure
model/index mechanics, not corpus content, and were safe to treat as valid
in the interim. **The quality columns moved substantially** (all lower —
consistent with every other quality table in this document after the two
rebuilds): `qwen3 × semantic` dense recall@10 0.6581→**0.5382**,
`qwen3_0.6b × semantic` 0.6364→**0.5688** (now numerically *above* `qwen3`
on this specific dense-alone cell — consistent with the cross-chunker
aggregate lead found above), `jina_v5 × semantic` 0.5845→**0.4705**.

Full data + methodology: `tools/eval/cost_latency_pareto.py`, rendered
report at `data/results/cost_latency_pareto.md` (gitignored, regenerate by
rerunning the script). All numbers below are measured on each embedder's
`semantic`-chunker combo — the same combos the quality numbers elsewhere in
this doc that are chunker-specific refer to — so cost and quality columns
in the table below are apples-to-apples with each other, unlike an earlier
internal draft of this table which paired semantic-chunker latency against
cross-chunker-*aggregate* recall (a mismatch caught before being cited
anywhere). **Note**: per the "Chunkers compared" section above, `semantic`
is no longer citable as "the best chunker" (a dedicated significance test
found it never significantly beats any other chunker) — these numbers still
describe a reasonable, representative combo, just not a provably-optimal
one.

**Two current-implementation costs are not floors on what dense/hybrid
retrieval must cost. For hybrid specifically, they add a roughly fixed
~0.9-1.0s of overhead to *every* query, almost independent of which embedder
is in the loop** (it was ~2.1-2.3s before `5cc71a1` removed the BM25 rebuild)
— because the overhead comes from re-touching the whole corpus (full-corpus
fetch), which scales with corpus size, not embedding dimension. Found while
building this table, worth stating explicitly rather than silently working
around. **A third cost listed here until 2026-08-08 is now gone** — the
per-query `BM25Okapi` rebuild — and its removal is what splits the old
bundled figure into the two items below:

1. `DenseRetriever.retrieve()` (`src/rag_lab/retrievers/dense.py`)
   recomputes `np.linalg.norm(embeddings, axis=1)` — the corpus's row norms
   — from scratch on **every query**, even though the corpus (and hence its
   norms) doesn't change between queries. Measured cost (2026-08-09, 74,816
   chunks): 40.90ms (dim=384) / 107.40ms (dim=1024) / 264.17ms (dim=2560),
   against an intrinsic dot-product-and-sort of 62.91 / 158.04 / 386.56ms —
   roughly **40% of dense search time is this one avoidable recomputation**,
   and it is free to remove because caching a row norm cannot change a
   ranking. (Essentially unchanged across every refresh — this is
   corpus-size-dependent mechanics, not corpus-content-dependent.)
2. `HybridRetriever.retrieve()` (`src/rag_lab/retrievers/hybrid.py`) asks
   **both** sub-retrievers for `k=n` (the entire 74,816-chunk corpus, not a
   bounded candidate pool) before RRF-fusing and truncating to the caller's
   actual k=10. Measured 2026-08-09: `DenseRetriever.retrieve(k=n)` =
   575.58ms vs. `retrieve(k=10)` = 262.46ms, the **313ms** gap being
   `RankedChunk` construction (full chunk text included) for tens of
   thousands of chunks nobody will ever look at; `BM25Retriever` pays the
   same tax on its side of the fuse. **Unlike (1) this is not free to
   remove** — `HybridRetriever` requests complete rankings precisely so RRF
   fuses full orderings, so a bounded pool is a *different retrieval method*,
   not the same one made faster. This effect — not RRF fusion itself — is now
   the whole of the gap between the intrinsic estimates and the measured
   1.21-1.86s hybrid totals. That *additive* gap is **0.87-1.03s** across all
   9 embedders (the tightest-clustered number in this whole table),
   confirming it is corpus-scanning cost, constant regardless of embedder.
   As a *ratio* the same fixed overhead looks very different depending on the
   embedder's own baseline cost — ~2.2× for `qwen3` (1863ms measured vs.
   839ms intrinsic, the most expensive embedder) up to ~3.5× for `e5_small`
   (1208ms vs. 341ms, the cheapest) — so lead with the additive number; the
   ratio is an artifact of which embedder you divide by, not a real
   difference in how much overhead hybrid retrieval carries.

**Because of this, the honest cost signal for a quality-vs-cost comparison
is query-*encode* time (the one component that's genuinely embedder-
dependent and not an artifact of these implementation choices), not the
measured search/hybrid totals.** The table below reports both:

| embedder | dim | encode p50 (ms) | intrinsic dense¹ (ms) | measured dense total p50 (ms) | intrinsic hybrid² (ms) | measured hybrid total p50 (ms) | recall@10 dense (semantic) | recall@10 hybrid (semantic) |
|---|---|---|---|---|---|---|---|---|
| qwen3 | 2560 | 198.64 | 585.20 | 839.78 | 838.70 | 1862.64 | 0.5396 | 0.6020 |
| qwen3_0.6b | 1024 | 90.09 | 248.13 | 359.44 | 501.63 | 1495.58 | **0.5707** | **0.6141** |
| jina_v5 | 1024 | 86.40 | 244.44 | 369.76 | 497.94 | 1506.91 | 0.4699 | 0.6003 |
| bge_m3 | 1024 | 82.94 | 240.98 | 334.66 | 494.48 | 1387.23 | 0.4154 | 0.5445 |
| e5_small | 384 | 24.91 | **87.83** | **120.56** | **341.33** | **1208.47** | 0.3829 | 0.5843 |
| congen | 1024 | 65.51 | 223.55 | 325.41 | 477.06 | 1433.87 | 0.2976 | 0.4655 |
| e5 | 1024 | 83.82 | 241.86 | 335.71 | 495.36 | 1391.24 | 0.3606 | 0.5442 |
| sct | 1024 | 66.07 | 224.11 | 322.01 | 477.61 | 1445.72 | 0.1273 | 0.3963 |
| m2v | 1024 | **2.38** | 160.42 | 273.43 | 413.92 | 1440.41 | 0.1318 | 0.3214 |
| bm25 | — | 0 | — | — | — | 234.45 (measured; 253.50 intrinsic) | — | 0.4642 (recall@10 alone) |

**Single provenance, 2026-08-09**: latency and `recall@10` columns now come
from one run against rebuild #3, retiring the deliberate 07-29-latency /
08-07-quality split the banner above describes. Read cross-embedder
differences against this rig's ~5-10% resolution (the three controls in the
banner) — the four dim-1024 rows between 322 and 370ms measured dense total,
for instance, are **not** separable.

**The `bm25` row is the one that changed shape, not just value**: at 234.45ms
measured against 253.50ms intrinsic it now sits *below* its own intrinsic
estimate, where it used to sit ~1s above. That is not a contradiction — the
intrinsic phase runs in the parent process after several `BM25Okapi` builds
and three full `embeddings.npy` loads, while the measured figure comes from a
fresh child; the 8% gap is the same process-state drift control 2 quantifies
at +5.1%, pointing the other way. The honest reading is that with the rebuild
memoised, **BM25-alone retrieval is now essentially its scoring cost**, and
the two figures are not comparable to better than ~10%.

¹ intrinsic dense = encode p50 + dot-product-and-sort at that dim (norms
cached, not recomputed). ² intrinsic hybrid = encode p50 + dot-product-and
-sort + BM25 `get_scores`-only (BM25 index cached, no k=n over-fetch on
either side; bounded-pool RRF fuse is <5ms, not separately measured). Build
cost (`embed_seconds`, `chunks_per_sec`, index size on disk) and the full
p50/p95 breakdowns for every number above: `data/results/cost_latency_pareto.md`.
Latency/cost **and** quality columns: **2026-08-09** against rebuild #3, one
run, idle machine, one subprocess per embedder, with the three timing controls
in the banner at the top of this section.

**Reading this table**: `m2v` (Model2Vec static embedding) is by far the
cheapest to encode (2.4ms) but also the weakest embedder in the whole matrix
(see Embedders compared above) — not a real Pareto contender. Among
genuinely competitive embedders, `e5_small` is the standout: intrinsic
dense cost of 88ms (2-7x cheaper than every other option in the top two
quality tiers) for recall@10=0.3829 dense / 0.5843 hybrid — now noticeably
below the hybrid headline number (a 0.030 gap, wider than the pre-rebuild
0.02) but still the cheapest intrinsic cost by far in that tier. `qwen3`
(4B) is the most expensive per query in every column and does not lead on
hybrid recall despite that — its cost is justified mainly by the
entity-type robustness finding above (no significant weak spot), not by
raw recall@10.

**Note on `qwen3_0.6b`'s numerically-highest recall@10 in this table**:
unlike the pre-rebuild version of this section, this is **not** being
cited as "the top combo in the whole study" — per the chunker-vs-chunker
significance test (see "Chunkers compared" above and Open item #13),
`semantic` never significantly beats any other chunker for any embedder,
so a semantic-chunker-specific combo cannot be crowned a study-wide winner
regardless of which embedder numerically tops this table. The gap between
`qwen3_0.6b` (0.6152) and the next two — `qwen3` (0.6051) and `jina_v5`
(0.5995) — is narrow and untested; `bge_m3` fell the furthest of the top
tier (0.6845→0.5451), consistent with the cross-chunker dense-alone
3-way-tie break noted above (`bge_m3` losing ground to `qwen3`/`qwen3_0.6b`).

**Resolved 2026-07-22**: ran the dedicated per-chunker (semantic-only)
pairwise significance test across all five top hybrid combos
(`tools/eval/hybrid_significance_test_semantic_top5.py`,
`data/results/hybrid_significance_test_semantic_top5.md`) — paired
bootstrap, Holm-corrected, 10 pairwise comparisons × 3 metrics
(recall@10/MRR/nDCG@10), semantic chunker + hybrid retrieval only, no
cross-chunker averaging. **Result: none of the 10 pairs is significant on
any metric** (lowest raw p = 0.065, on qwen3_0.6b vs jina_v5/e5_small for
MRR/nDCG — doesn't survive Holm correction either). The +0.009 recall@10
gap between qwen3_0.6b and bge-m3 is confirmed noise, not a real
difference. **Final stance, now confirmed rather than provisional**: the
top five hybrid combos (qwen3_0.6b 0.6935, bge_m3 0.6845, e5_small 0.6821,
qwen3 0.6797, jina_v5 0.6796 recall@10) are a genuine statistically-tied
cluster. Crown neither `qwen3_0.6b` nor `bge-m3` as "the best hybrid
combo" — the paper's headline is the system-level recommendation (semantic
chunking + hybrid retrieval), not a specific embedder pick.

**Re-confirmed 2026-07-25** against the corpus-discovery-bug-fixed indices —
same conclusion held then (genuine tied cluster, no pair significant),
numbers strengthened across the board (`qwen3_0.6b` read 0.7048 at that
point, up from 0.6935).

**Superseded 2026-07-29**: those 2026-07-25 numbers (0.7048 etc.) went stale
after the separate 2026-07-28 OCR-remediation rebuild — this whole
paragraph, the "New finding ... resolves Open item #8" paragraph above it,
and the "Resolved 2026-07-22" paragraph above that are all describing
results from the same script (`hybrid_significance_test_semantic_top5.py`)
at superseded points in time; **do not cite the 0.6935/0.6845/0.7048 figures
above as current**. The tie itself is still real post-refresh but now a
4-way cluster, not 5 (`bge_m3` dropped out on recall@10/nDCG@10) — see the
"Top single-combo across the entire study" bullet in the "Hybrid retrieval"
section above for the current numbers and the full explanation, including
why the "top single combo" claim this section originally resolved no longer
holds as stated.

### Hybrid fetch depth — how much of the over-fetch is removable, and at what cost

*(2026-08-09, `tools/eval/hybrid_fetch_depth_sweep.py` →
`data/results/hybrid_fetch_depth_sweep.md`; 36 combos × 106 queries = 3,816
(combo, query) pairs, n = 57,174–74,816 chunks, k = 10.)*

`HybridRetriever` asks both arms for a ranking over the whole corpus and keeps
10. With the `BM25Okapi` rebuild memoised away (see the cost/latency section
above), that over-fetch is what remains of the per-query overhead. **It is not
free to cut**: a chunk inside dense's top-F but past BM25's cut loses its BM25
term outright rather than earning a small one, so truncation changes the
ranking rather than approximating it.

| F | top-10 identical (order) | identical as a set | recall@10 | Δ vs k=n |
|---|---|---|---|---|
| 50 | 31.34% | 34.93% | 0.5101 | −0.0097 |
| 100 | 46.91% | 51.76% | 0.5162 | −0.0035 |
| 200 | 56.39% | 62.81% | 0.5170 | −0.0027 |
| 500 | 64.28% | 74.06% | 0.5182 | −0.0015 |
| 1,000 | 70.13% | 82.76% | 0.5172 | −0.0025 |
| 10,000 | 87.84% | 96.59% | 0.5193 | −0.0004 |
| n | 100% | 100% | 0.5197 | — |

2026-08-23 re-run against rebuild #4.

**The two questions have opposite answers.** *Is the ranking preserved?* — only
at F=n; even fetching 10,000 candidates reproduces just 87.84% of top-10s in
order. *Does it cost anything?* — almost nothing from F=100 up, and the loss is
**non-monotonic** (F=500's −0.0015 beats F=1,000's −0.0025), because a larger F
raises different chunks' fused scores at different rates. Note this recall@10 is
a **macro average over all 36 combos** — a damage-size indicator, not a system
result.

Damage lands where this project's RRF rule predicts: the worst-hit combos are
the weak arms (`semantic × e5_small` −0.0579 at F=50, `recursive × bge_m3`
−0.0224 at F=200, `sentence × sct` −0.0145 at F=1,000 — the last two unchanged
by the rebuild in both combo and value). Per entity type,
**`person` is the only one that *gains* from truncation** (+0.0217 at F=50),
consistent with BM25 carrying `person` (0.8147) while the cut removes a weak
dense arm's long tail.

**Latency** (paired, one process, one loaded index, arms alternated per query,
BM25 scorer pre-warmed; `plain__sentence__qwen3__ff8f6c49`, 106 queries):
k=n p50 **1089.5 ms** → F=1,000 **421.0 ms** → F=200 **417.9 ms**. So the
over-fetch is ~62% of hybrid query time, and the ~0.42 s that remains is real
scoring work (dense encode + gemv + BM25 `get_scores`) that no depth cut
touches. The trade on offer is ≈0.67 s/query for −0.0033 macro recall@10 at
F=200. **As of this sweep nothing was wired — the class default is still k=n**,
and the knob's default (`fetch_depth=None`) computes `depth = len(index.chunks)`,
i.e. the old expression exactly, so no published number moves. (It was
subsequently shipped at the *query-time* layer only, once re-measured against the
hard router — see the ship decision two sections down; the constructor default
never changed.)

### The same cut measured against the shipped hard router — and the ship decision

*(2026-08-09, `tools/eval/routed_fetch_depth_test.py` →
`data/results/routed_fetch_depth_test.md`; 106 queries routed by
`classify_query` to their 4 shipped indices, k = 10 sent by every arm.)*

The table above is a macro average over 36 combos retrieving **without a
router**, and hard routing has shipped since 2026-08-08 — the same wrong-pair
trap that per-`entity_type` alpha and the rrf4 reranker both fell into. Re-run
against the shipped configuration, **the trade gets better on both sides**:

| arm | docs sent | docs fetched | recall@10 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|
| routed, k=n (shipped) | 10 | n (57k–75k) | 0.6831 | 0.8686 | 0.7502 | 1193.9 ms |
| routed, F=1,000 | 10 | 1,000 | 0.6804 | 0.8686 | 0.7481 | 483.1 ms |
| routed, F=200 | 10 | 200 | 0.6835 | 0.8662 | 0.7480 | 475.6 ms |

Pre-registered family (F=200 vs k=n, 3 metrics, Holm m=3): recall@10
**+0.0005** (Holm-adj 1.0000), MRR **−0.0024** (1.0000), nDCG@10 **−0.0022**
(1.0000). **State it as a bound, since a null is the outcome that licenses
shipping here**: the CI rules out a loss worse than **0.0078** on the
worst-behaved of the three metrics. Latency **−0.718 s/query (2.51×)**, measured
paired on the index each query is actually routed to.

**The registered prediction was confirmed, and it confirms the mechanism rather
than the headline.** Unrouted, `person` is the one entity type that *gains* from
a shallow cut (+0.0202 at F=50 here, +0.0217 in the sweep) because BM25 carries
it while the cut deletes a weak dense arm's tail. Routing already hands `person`
its dense specialist, so that gain should shrink — it **reverses**, to −0.0207.
Same mechanism as the two interventions that died against the router; here it
costs nothing, because a depth cut only has to avoid losing.

**Decision: shipped at the query-time layer, not as the class default**
(`app/streamlit_app.py` sets `fetch_depth` per query via `StrategySpec` params,
defaulting to 200 with `1000` and "whole corpus" selectable;
`HybridRetriever.__init__` keeps `fetch_depth=None`, pinned by a test). The
split is the point: F=200 changes the top-10 on **17 of 106** Gold queries, so
letting it become the constructor default would silently re-rank every future
eval run while all ~24k persisted results and every published table still said
k=n. The UI is where 0.72 s is felt; the eval harness is where reproducibility
is. Nothing an eval reads is touched — the invariant audit already classifies
`mode_b`/`mode_b_routed` as write-only UI dirs.

### The same knob under `weighted` fusion — measured, and it does not transfer (2026-08-12)

*(`tools/eval/hybrid_weighted_fetch_depth.py` → `data/results/hybrid_weighted_fetch_depth.md`;
36 combos × 106 queries = 3,816 pairs, 16 min. The fusion is **imported** from
`hybrid_fetch_depth_sweep.py` rather than reimplemented, so this run's `rrf` columns are a
cross-artifact anchor and reproduce that sweep at all 11 depths.)*

Everything above is the `rrf` branch. `HybridRetriever` also has a `weighted` score-fusion
branch, and from 2026-08-11 to 08-12 that pair *raised* rather than running — containment
for an unmeasured configuration, with its exit condition written into it. The measurement
was run against a decision rule frozen in the script before the run, and the rule returned
**LIFT**: the raise is gone, and a test pins that permitting the pair did not quietly make
truncation a no-op.

**LIFT is not a recommendation, and the number is the point.** At F=200 `weighted` loses
**−0.0609** macro recall@10 against its own k=n — about **18x** `rrf`'s −0.0033 at the same
depth — and unlike `rrf` it does **not** recover with depth: at F=10,000 of ~75,000 chunks
it is still **−0.0112** against `rrf`'s −0.0005. For `weighted`, "deep enough" is essentially
n, so the knob buys nothing. What licenses permitting it anyway is that this codebase bans an
*unmeasured* configuration from passing as measured, not a measured-but-worse one.

**The mechanism, corrected by the run's own data.** Truncation does not mildly *add* the
intersection signal `weighted` structurally lacks — it makes intersection membership nearly
decisive. Max-normalized cosine is flat (0.9491 at rank 10, **0.2699** at rank n), so a cut
arm still forfeits a large term, where `rrf` at rank 1,000 forfeits only ≈0.0005. So
`weighted`'s top-10 goes **8.25/10** in-both-arms at F=200 and **9.99/10** at F=1,000 (`rrf`
7.41 / 8.30): it becomes an intersection-only ranker and evicts what one arm alone found.
That lands exactly where a single arm carries a type — `person` **−0.1965** at F=200 (BM25
carries person at 0.8147) against `program` **+0.0212**.

Two further results worth carrying. The pre-registered guess that a cut merely zeroes terms
that were already zero was **refuted**: `BM25Okapi` floors negative IDF so the last-ranked
chunk really does score 0, but only **0.1%** of the zeroed terms were already 0, because a
~20-token Thai query has common tokens reaching nearly every chunk (BM25 carries 88,301 of
dense's 121,437 zeroed mass at F=50, and only 2 of 157,717 dense terms are promoted). And
descriptively, at F=n `weighted` scores **above** `rrf` — 0.5442 vs 0.5204, **+0.0239** macro
recall@10 — which is a **hypothesis, never a result**: no significance test, macro over 36
combos, unrouted, and nothing ships `weighted`. The wrong-pair trap that killed
per-`entity_type` alpha and rrf4 applies to it too.

**A method note that generalises past this run**: the smoke slice (2 combos × 8 queries)
*reversed the sign of the headline* — `weighted` appeared to **gain** from truncation, peaking
0.7708 at F=100 against 0.5938 at k=n, which is the opposite branch of the frozen rule. A
smoke run checks that the code runs; it is not a small version of the answer.

## Open items (not yet done, needed before the numbers above are "final")

1. ~~Per-chunker point comparison of BM25 vs. embedder (not averaged across
   chunkers) not yet significance-tested~~ — DONE 2026-07-22, re-run
   2026-07-25 against the corpus-discovery-bug-fixed indices, **re-run
   again 2026-07-29 against the OCR-remediation-rebuilt indices**
   (`tools/eval/bm25_vs_embedder_significance_test_per_chunker.py`, see
   "Per-chunker BM25 vs. embedder" section above — the 2026-07-25 numbers
   had gone stale after the 2026-07-28 rebuild, see item 12). **Core
   pattern confirmed real across both refreshes**: `bge_m3` loses to BM25
   significantly under `sentence` chunking specifically (Holm-adj p=0.0060
   post-2026-07-29) despite tying it in the aggregate and in the other 3
   chunkers. One change from the 2026-07-29 refresh: `qwen3_0.6b`'s
   numerically-negative BM25 margin under `semantic` chunking is now
   **statistically significant** (Holm-adj p=0.0060) — the first
   chunker/embedder cell in this comparison where an embedder significantly
   beats BM25 outright, not just numerically. `qwen3`(4B) remains only
   numerically ahead under `semantic`, not significantly.
2. ~~Why bge-m3 overtakes qwen3 specifically under hybrid despite tying it
   as dense-alone~~ — CLOSED 2026-07-22, **premise was false, no GPU
   investigation needed**. First pass
   (`tools/eval/bge_qwen_bm25_complementarity.py`) tested the "error-pattern
   complementarity with BM25" hypothesis directly from persisted top-10
   results (rescue rate, union coverage, per-query recall correlation with
   BM25) and found the opposite of what it predicted: `qwen3` has a *higher*
   aggregate rescue rate (0.3189 vs bge_m3's 0.2995), *higher* union coverage
   (0.6036 vs 0.5823), and *lower* correlation with BM25 in 3 of 4 chunkers —
   yet bge_m3 still numerically led under hybrid (0.6472 vs 0.6235 aggregate
   recall@10). Rather than escalate straight to full-corpus-rank GPU
   retrieval to explain that gap mechanically, a cheap intermediate check was
   run first: which chunker shows the gap, and is it driven by a small,
   dramatic set of "swing" queries? Per-chunker recall@10 diff (bge_m3 −
   qwen3, hybrid): fixed_size +0.0209, recursive +0.0371, sentence +0.0320,
   semantic +0.0048 (already known tied — see top-5 test above). On
   `recursive` (the largest gap), the swing queries split 37-for-bge_m3 vs
   34-for-qwen3 — a broad, nearly-balanced churn across most of the 73-query
   set, not a small dramatic subset, which is the signature of two tied
   systems trading wins rather than one systematically beating the other.
   That predicted the real test: a paired bootstrap on bge_m3-vs-qwen3
   hybrid recall@10, run per chunker + aggregate (5 tests, Holm-corrected).
   Result — **none significant**: sentence raw p=0.0474 (Holm-adj 0.2370,
   the closest of the five), recursive raw p=0.0572 (Holm-adj 0.2370),
   aggregate raw p=0.1156 (Holm-adj 0.3468), fixed_size raw p=0.3178
   (Holm-adj 0.6356), semantic raw p=0.8238 (Holm-adj 0.8238). The "bge_m3
   overtakes qwen3 under hybrid" premise itself never held up to the same
   significance bar applied everywhere else in this study — it was a raw
   number, not a tested effect, exactly like the top-5 hybrid tie before it
   was tested. **No mechanism to explain, because there's no confirmed
   effect** — closing without touching the GPU.
3. ~~Cost/latency table (vector dim, index size on disk, query latency
   p50/p95)~~ — DONE 2026-07-21: see "Cost / latency characterization"
   section above + `data/results/cost_latency_pareto.md` +
   `tools/eval/cost_latency_pareto.py`. Also surfaced two current-
   implementation overheads (BM25Okapi rebuilt per query, hybrid over-
   fetching the full corpus before fusing) that added a roughly fixed
   ~2.1-2.3s of latency to every hybrid query regardless of embedder —
   reported as implementation characteristics, not silently fixed.
   **The first of the two has since been fixed** (`5cc71a1`, 2026-08-08:
   the scorer is memoised on the `Index`), leaving ~0.87-1.03s of over-fetch
   overhead as measured 2026-08-09. Expressed as a ratio that residue is
   ~2.2x for the most expensive embedder up to ~3.5x for the cheapest,
   purely because the same fixed overhead is divided by very different
   intrinsic baselines — the additive number is the real story.
   **The second is now measured, and DECIDED** (2026-08-09, see "Hybrid
   fetch depth" above): capping the fetch at F=200 removes ≈0.67s/query but
   costs −0.0033 macro recall@10 and changes 43% of top-10 orderings, so it
   is a cost/quality trade for the deployer, not a defect to silently
   repair. Re-measured against the **shipped hard router** (rather than the
   36-combo macro, which is not a system result) the same cut is
   **+0.0005 recall@10, all three metrics ns, for −0.718 s/query**, so it is
   now **wired at the query-time layer** — the UI opts in per query while
   `HybridRetriever`'s default stays k=n, which is what keeps every
   published number and every eval script reproducing unchanged.
4. ~~MAP + Precision@k + multi-k (1/3/5/10)~~ — DONE 2026-07-21:
   `precision_at_k` and `average_precision_at_k` added to
   `src/rag_lab/metrics.py`, `evaluate()` now accepts a list of k's and
   always reports `map` alongside `mrr` (backward-compatible — a plain int
   `k` still works, existing callers unaffected). `run_gold_*_eval.py`
   report tables now render precision@k and map columns too.
   ~~Not yet done: no eval script has actually been re-run with a multi-k
   list~~ — DONE 2026-07-22 (`tools/eval/multi_k_report.py`,
   `data/results/multi_k_report.md`): every combo was retrieved at
   `top_k=10`, so k∈{1,3,5,10} needed no new retrieval/GPU/embedding calls —
   pure recompute over already-persisted JSON, runs in seconds. See "Multi-k
   metrics" section below for the citation-ready table.
5. ~~RQ3 (normalization/segmentation ablation)~~ — DONE 2026-07-23, see
   "RQ3 ablation results" section above: only chunk-size has a significant
   effect (smaller is better for recall); normalization and word-aware
   segmentation do not. RQ4 (end-to-end RAG answer quality) was later brought
   back into scope and **completed 2026-08-03, refreshed 2026-08-07** — see the
   "RQ4" section above.
5b. ~~Cross-encoder reranker (Tier 3 item 8)~~ — DONE 2026-07-23, **refreshed
    2026-07-29**, see "Cross-encoder reranker results" section above:
    significantly hurts hybrid MRR only (nDCG@10 no longer significant post-
    refresh, a real finding-level change), no significant effect on
    dense-alone, literature-grounded explanation in
    `docs/reranker-hybrid-interaction-research.md`. RQ4 — the last Tier 3 item
    — is **also done** (2026-08-03, refreshed 2026-08-07), so Tier 3 is closed.
6. ~~Per-entity_type significance test for the 9-embedder matrix~~ — DONE
   2026-07-21 (`tools/eval/embedder_significance_test_by_entity_type_9way.py`).
   `qwen3_0.6b`'s program-query lead is NOT significant vs congen/qwen3-4B
   (3-way tie) — see "Embedder × entity_type profile" section above.
7. ~~BM25 and hybrid (RRF) sections extended to the 3 new embedders~~ — DONE
   2026-07-21 (`bm25_vs_embedder_significance_test_9way.py`,
   `hybrid_significance_test_9way.py`). Confirmed: `sct` at 510 tokens *is*
   a second RRF failure-mode case alongside m2v (hybrid significantly worse
   than BM25-alone); `qwen3_0.6b` numerically edges out `bge_m3` on the
   aggregate hybrid table (0.6543 vs 0.6472) but this isn't yet verified
   per-chunker (see item #8) so don't cite either embedder as the confirmed
   top hybrid combo yet.
8. ~~Best single (chunker × embedder) combo for `qwen3_0.6b` and `e5_small`
   not yet checked per-chunker~~ — CHECKED 2026-07-21, re-checked
   2026-07-25 (both against since-superseded indices — the 0.6935/0.6845/
   0.6821 figures originally recorded here are **stale as of the
   2026-07-28 rebuild, see item 12**; current numbers and the
   now-narrower/differently-shaped conclusion are in the "Hybrid retrieval"
   section's "Top single-combo across the entire study" bullet — notably,
   the "semantic is the clear top chunker for this combo" part of the
   original finding did **not** survive the 2026-07-29 refresh).
   ~~New Open item: a per-chunker (semantic-only) pairwise significance test
   across the top hybrid combos~~ — DONE 2026-07-22, re-run 2026-07-25,
   **re-run again 2026-07-29 against the OCR-remediation-rebuilt indices**
   (`tools/eval/hybrid_significance_test_semantic_top5.py`): the tied
   cluster **partially broke** in the 2026-07-29 refresh — `bge_m3` now
   loses significantly to `qwen3_0.6b`/`qwen3`/`jina_v5` on recall@10 and
   nDCG@10 (still ties on MRR). The remaining four (`qwen3_0.6b`, `qwen3`,
   `jina_v5`, `e5_small`) are still fully tied on every metric. See "Top
   single-combo across the entire study" in the "Hybrid retrieval" section
   above for the current writeup. Don't cite any one embedder as the
   confirmed best hybrid combo among those four.
9. ~~**Updated 2026-07-29**: MAP/precision@1 contradiction + never
   significance-tested~~ — **FULLY CLOSED 2026-07-29**. Two separate
   problems, both now resolved. (a) The original 2026-07-22 version of this
   item (`bge_m3` leading MAP while `qwen3_0.6b` led precision@1 — opposite
   directions) was computed from a stale `multi_k_report.md`, one rebuild
   behind `gold_hybrid_73det` — a third instance of the "not in the
   5-script refresh chain" staleness bug. Re-ran it: the contradiction is
   **gone**. (b) The remaining gap — that neither metric had ever been
   significance-tested, and that the existing tie test's scope
   (`semantic`-only) didn't match the multi-k tables' scope
   (cross-chunker-aggregate) — is closed by
   `tools/eval/map_precision_significance_test.py`, which runs **both**
   scopes × both retrievers. Results in "MAP / precision@1 significance
   test" above: `qwen3_0.6b` significantly beats all 8 other embedders on
   both metrics dense-alone (stronger than its recall@10 result), the
   semantic-scope tied cluster **holds on both new metrics** (nothing at
   all is significant on precision@1), and the scope mismatch turns out to
   matter — `qwen3` is numerically highest at semantic scope, so
   "`qwen3_0.6b` leads every metric" is an aggregate-scope claim only.

10. ~~New 2026-07-22: BM25 and hybrid have never been broken down by
    entity_type~~ — **DONE 2026-07-29**
    (`tools/eval/bm25_hybrid_entity_type_breakdown.py`,
    `data/results/bm25_hybrid_entity_type_breakdown.md`), see "Structural
    recall@10 ceiling by entity_type" above for the full writeup. Both
    original questions answered: hybrid reaches **84.1%** of the `person`
    ceiling (dense-alone only 58.8%), and it **does** narrow the
    `faculty_adjunct_aggregate` gap (BM25 62.0% → hybrid 72.3%). **The
    headroom reading in that section is reversed by this**: under hybrid,
    `person` is the most-solved category, not the one with the most
    addressable headroom — that title now belongs to `course` (65.1%),
    which postdates the original analysis. Two further findings fell out:
    (i) **direct** evidence for the lexical/dense complementarity
    mechanism (BM25 alone 0.8147 on person vs 0.3484 on program; dense
    the reverse) — the mechanism Open item #2's indirect proxies failed to
    establish; and (ii) **"hybrid never hurts" is an aggregate claim, not
    a per-category one** — on `person` specifically, hybrid sits *below*
    BM25-alone for most embedders, only `bge_m3` exceeding it. The larger
    unstarted idea (an entity-indexed/structured lookup path for "list
    all X" queries, using the already-built taggers) remains a candidate
    future direction, not part of the current plan.

11. New 2026-07-23: found + fixed a corpus-discovery bug affecting **every
    number in this document**. `runner.py::_discover_paths` (and
    `cli.py::build`) did a bare `rglob("*.md")` with no filtering, unlike
    `loaders/common.py::iter_corpus_files` which already guarded against
    this. `academic_resolutions/` also holds ~19 gitignored tooling-report
    files (`llm_ocr_scan/`, `llm_thematic_scan/`, `entity_tags/`,
    `ocr_repetition_review.md`) that don't match the real
    `<year>/<session>/file.md` structure; `make_resolution_id` has a
    silent path-fallback for non-conforming paths (no crash), so these
    were ingested as fake resolutions in every historical full-corpus
    build. Confirmed directly in `chunker_compare_full`'s built indices:
    6.87% (fixed_size), 7.03% (recursive), 8.25% (semantic) of chunks
    trace to these bogus files — one alone (`consensus_priority.md`, a
    637KB OCR-scan report) contributed 1,517 chunks to the semantic index,
    ~50x a typical real resolution's share. Contamination rate is
    comparable across chunkers, so it's probably not the reason semantic
    won the chunker comparison, but it is real noise inside every number
    above. Fixed (commit `8c86b63` + a follow-up `cli.py` fix, same
    session) — verified `_discover_paths` now returns exactly the correct
    2,853 real resolutions with zero contamination, full test suite green.
    **Historical indices were initially not rebuilt** (deferred by user
    decision 2026-07-23, given the contamination rate was small and
    roughly uniform across compared conditions) — **the user reversed
    that decision and requested the full rebuild**, split into 4 batches
    (one per chunker) to make an otherwise multi-day undertaking
    manageable. All 4 batches (`fixed_size`, `recursive`, `sentence`,
    `semantic` — configs in `config/experiments/_history/rebuild_clean_*.yaml`)
    completed 2026-07-24/25, each spot-checked with the same
    contamination grep as the original finding: **0 contaminated chunks**
    across all 36 (chunker × embedder) combos in `chunker_compare_full`.
    Two of the four batches (`sentence`, `semantic`) were interrupted
    mid-run by external process kills (machine sleep) and finished via
    `*_resume.yaml` configs covering only the not-yet-written combos —
    `pipeline.build_index` writes each combo's artifacts atomically only
    on completion, so no partial/corrupt data resulted. **DONE 2026-07-25**:
    `tools/eval/embedder_matrix_9way.py` and siblings
    (`run_gold_bm25_eval.py`, `run_gold_hybrid_eval.py`,
    `run_gold_hybrid_eval_9way_new.py`,
    `embedder_significance_test_by_entity_type_9way.py`,
    `bm25_vs_embedder_significance_test_9way.py`,
    `hybrid_significance_test_9way.py`) were re-run against the clean
    indices and every headline number in this document above (dense-alone
    aggregate + significance, BM25 aggregate + significance, hybrid
    aggregate + significance, entity_type breakdown + significance) was
    updated. **Every qualitative conclusion held — none flipped** — the
    single most notable numeric shift is `sct`'s hybrid-vs-BM25 recall@10
    deficit losing significance (was significant pre-rebuild, Holm-adj
    p=0.031; now p=0.082), and `semantic × qwen3_0.6b` moving up to a clear
    2nd place in the dense-alone per-combo ranking (0.6435, was 0.6364 and
    behind jina_v5's old number). **Two sub-analyses were re-run separately,
    also 2026-07-25**: the per-chunker BM25-vs-embedder breakdown
    (`bm25_vs_embedder_significance_test_per_chunker.py`) and the
    semantic-only top-5 hybrid tie test
    (`hybrid_significance_test_semantic_top5.py`) — both flagged inline
    above where cited. Every conclusion in both held except one: `congen`
    under `recursive` chunking lost significance in the per-chunker
    breakdown (Holm-adj p=0.0620, was significant pre-rebuild). At the time,
    nothing in this document still cited pre-rebuild numbers — **but see
    item 12 below: a second, unrelated rebuild made this refresh stale
    again three days later, and it went undetected for a while.**
12. New 2026-07-28/29: the 36-combo `chunker_compare_full` index was
    rebuilt a second time, for the unrelated OCR-remediation fix
    (`docs/llm-ocr-scan-log.md`, kernel-A + Mechanism-B batches) — this
    changes chunk text content, not corpus membership, so it invalidates
    every number above a second time. `embedder_matrix_9way.py` recomputes
    dense-alone retrieval fresh every run, so the dense-alone tables were
    refreshed automatically and correctly — but `run_gold_bm25_eval.py` and
    `run_gold_hybrid_eval.py` were **not** re-invoked, so `gold_bm25_73det`
    and `gold_hybrid_73det` silently carried 2026-07-25 mtimes for three
    days while the downstream significance scripts kept reading them,
    comparing fresh dense numbers against stale BM25/hybrid numbers with no
    warning. Caught only because the resulting eval run looked like it had
    flipped an implausible number of conclusions at once, prompting an mtime
    check before anything was written up (full incident:
    `docs/chunker-embedder-comparison-log.md`, "Re-eval หลัง
    OCR-remediation rebuild" entry). **Fixed 2026-07-29**: both retrieval
    paths re-run against the rebuilt indices (BM25: 1128s; hybrid: 11573s),
    then all 4 downstream significance scripts re-run. **This refresh did
    change real conclusions**, unlike the 2026-07-25 one: BM25 aggregate
    recall@10 rose 0.3908→0.4930 and now significantly beats `bge_m3`
    (previously a tie); the cross-chunker dense-alone 3-way tie
    (bge_m3/qwen3/qwen3_0.6b) is broken, `qwen3_0.6b` now significantly
    ahead of both others; `sct`'s hybrid-vs-BM25 recall@10 deficit is
    significant again (reversing the 2026-07-25 "no longer significant"
    finding — that finding was itself measured against the since-superseded
    index); `jina_v5` now significantly beats BM25-alone under hybrid
    (previously borderline) while `congen` drops out of that group; the
    semantic-only top-5 hybrid tie narrowed from 5 to 4 embedders (`bge_m3`
    dropped out on recall@10/nDCG@10, still ties on MRR); and most
    consequentially, the previously-cited "top single combo in the whole
    study" (`semantic × qwen3_0.6b`, recall@10=0.7048) dropped to 0.6152 and
    is no longer even the highest number among `qwen3_0.6b`'s own four
    chunkers (`sentence` reads 0.6265, `fixed_size` reads 0.6154) — **new
    open item**: no significance test exists yet for a fixed embedder+
    retriever compared *across* chunkers (existing tests are either
    cross-chunker-aggregate or within one chunker across embedders); if the
    paper wants to name a specific chunker as numerically best for a
    specific combo, that comparison needs a dedicated paired-bootstrap test
    first — the plain reading of the fresh numbers should not be cited as a
    tested finding. See the "Hybrid retrieval" and "BM25 lexical baseline"
    sections above for full updated tables; **process lesson recorded**:
    after any index rebuild, refresh every retrieval path with persisted
    results (dense, BM25, hybrid), not only the one an eval script happens
    to recompute automatically — check `data/results/*` mtimes against the
    rebuild timestamp before trusting any significance test.
13. ~~New open item from #12: no significance test exists for chunker vs.
    chunker at a fixed embedder+retriever~~ — DONE 2026-07-29, same day.
    Built `tools/eval/hybrid_chunker_significance_test.py` (pure recompute
    from persisted `gold_hybrid_73det` results): one 6-pair
    (fixed_size/recursive/semantic/sentence) family per embedder, plus an
    aggregate family (each chunker's per-query score averaged across all 9
    embedders first). Full table: `data/results/hybrid_chunker_significance_test.md`.
    **This resolved the immediate question (no, `sentence` is not the new
    winner for `qwen3_0.6b` — all 4 chunkers are fully tied, Holm-adj
    p≥0.44) and surfaced a much bigger one**: `semantic` does not
    significantly beat any other chunker **anywhere** in this whole test —
    not for any single embedder, not in the aggregate. The project's
    long-standing "semantic chunking wins" headline (originally a raw,
    never-significance-tested 6-embedder mean, see "Chunkers compared"
    section above) does not survive being tested as an actual claim. The
    only significant result anywhere in the test is `fixed_size` losing to
    `recursive` (aggregate nDCG@10, plus 3-4 individual embedders) —
    `recursive`/`semantic`/`sentence` form a tied top cluster with no
    provable winner, `fixed_size` is the one demonstrated laggard. See the
    "Chunkers compared" section above for the full writeup and the revised
    practical framing.

14. New 2026-07-30: **`resolution_id` was never unique — 6 ids were shared by 12
    files**, found while verifying (for an unrelated reason) that
    `data/index/entity_tags_full` was current. `resolution_id` is
    `<year>/<session>/<title>` with `title` coming from `meeting_manifest.json`,
    and nothing enforced uniqueness, so 2,853 corpus files produced only 2,847
    distinct ids. Since relevance is judged per resolution (ADR-0002), a shared
    id merges two documents into one relevance unit and a top-k hit on either
    counts for both. Worst case found: an อาจารย์บัณฑิตพิเศษ appointment document
    carrying a curriculum-revision title (different Drive URLs, unrelated
    subject matter). **Fixed 2026-07-30** in three parts:
    (i) 4 of the 6 were data errors with a recoverable correct title, repaired
    at the source (`tools/corpus_prep/fix_manifest_title_collisions.py` —
    idempotent, verifies the on-disk title before writing, which matters because
    `academic_resolutions/` is gitignored so the script *is* the change record);
    (ii) the remaining 2 are two genuinely distinct agenda items that one
    meeting listed under one identical title, and are separated by
    `make_resolution_id`'s new folder-local ` #N` rank rather than by invented
    metadata; (iii) `pipeline.build_index` now refuses to build on a collision,
    and `tools/corpus_prep/audit_resolution_ids.py` reports every clash (exit 1)
    with the evidence needed to distinguish the two cases. Corpus now verifies
    at 2,853 files → 2,853 unique ids, full test suite green. See the ADR-0002
    amendment for the reasoning.
    **Eval exposure, measured not assumed**: an earlier prefix-matched count
    suggesting the worst case was gold-cited was a false positive of that
    matching — exact matching shows **3 of 106** `gold_query_set_73det.yaml`
    queries (1 of 252 in the large set) cite a colliding id, and all 3 cite the
    *split-bundle* case (2567/1 `__1`/`__2`, two ฉบับปี revisions of one
    วิศวกรรมชีวการแพทย์ curriculum that had been patched with one shared title).
    Both pieces are independently relevant under `build_gold_candidates.py`'s
    rules, so those queries' relevant sets legitimately grow by one (12→13,
    9→10, 9→10); patched with
    `tools/corpus_prep/patch_gold_ids_for_split_titles.py` (YAML round-trip
    verified byte-identical before rewriting, so the diff is 4 lines not 4,000),
    and **all 1,046 + 1,219 gold id references now resolve against the corpus,
    0 dangling**. **Closed 2026-07-30 — relabelled and re-evaluated, no number
    in this document changes.** Every built index stores the ids it was built
    with, so the gold set and the indices had briefly disagreed for those 3
    queries; `tools/corpus_prep/relabel_index_resolution_ids.py` rewrote the
    stored ids in place (a pure relabel — chunk text and embeddings are
    untouched, so no GPU rebuild), covering 49 combos / 20,828 index rows and
    568 result files / 907 rows. Attribution was exact rather than heuristic:
    each row was matched by `(old chunk_id, text)` against the pre-relabel
    backup, and the source file behind each block identified through the combo's
    *own* loader by `chunk_index` restarting at 0, not by text similarity —
    which matters because content matching cannot cross the re-OCR boundary.
    **Measured effect, per combo, on aggregate recall@10: dense −0.00002,
    hybrid +0.00018, BM25 −0.00031** — the 3 queries' relevant sets grow by one
    each, so the mean moves in the 5th decimal. The fresh
    `embedder_matrix_9way.py` run reproduces `qwen3_0.6b` 0.5263 and `bge_m3`
    0.4090 exactly as tabulated above. Every significance test that reads
    persisted results was re-run (`bm25_vs_embedder_*_9way`,
    `hybrid_significance_test_9way`, `hybrid_chunker_significance_test`,
    `hybrid_significance_test_semantic_top5`, `map_precision_*`,
    `bm25_hybrid_entity_type_breakdown`, `bm25_vs_embedder_*_per_chunker`,
    `embedder_significance_test_by_entity_type_9way`); all verdicts hold with
    **one narrowing**, on a pair that was already borderline: `bge_m3` losing to
    `jina_v5` under semantic+hybrid on **nDCG@10** is no longer significant
    (Holm-adj p 0.0928, still numerically −0.0462). `bge_m3` dropping out of the
    top-5 tied cluster therefore now rests on recall@10 (all three of
    `qwen3_0.6b`/`qwen3`/`jina_v5` still significant) and on nDCG@10 versus
    `qwen3_0.6b`/`qwen3` only. The bootstrap is seeded (`--seed 42`), so this is
    a real consequence of the relabel, not run-to-run noise.

15. New 2026-07-30: **swept the whole class of bug that #14 belonged to**, instead
    of waiting to trip over the next one. Three silent-corruption bugs had by then
    been found by accident (#11 corpus-discovery contamination, #12 stale
    BM25/hybrid cache, #14 `resolution_id` collisions), all with the same shape: a
    mismatch between two artifacts produced at different times by different
    scripts, which never crashes — it just makes a reported number wrong.
    `tools/eval/audit_pipeline_invariants.py` now checks 25 such invariants
    mechanically (report: `docs/pipeline-invariant-audit.md`). Results:
    - **Clean, verified**: `resolution_id` uniqueness (2,853/2,853), no empty
      document, manifest hygiene (0 dead entries, 0 duplicate keys, 0 unlisted
      files, no URL claimed by differently-titled documents), `master_list.csv`
      count, **row alignment of chunks↔embeddings↔lexical across all 63 built
      combos** (a misalignment here would silently return the wrong chunk for a
      vector, and it had never been checked), embeddings finite and non-zero-norm
      (sampled), and **all 2,265 gold relevant-id references resolve, 0 dangling**.
    - **One new structural finding**: `BuildCombo.id`
      (`combos.py`) hashes loader+chunker+embedder but **not the corpus**, so 12
      combo ids exist under two index roots at once (`chunker_compare_full` and
      `chunker_compare_smoke`) — a 12-file smoke subset and the 2,853-file corpus
      are indistinguishable by id, and a persisted result records only the id.
      This is the structural reason the #12 stale-cache incident was invisible:
      nothing in a result file says which index produced it. (The *cache* key in
      `pipeline._cache_key` does include the docset, so index reuse itself is
      safe — the ambiguity is in naming and results attribution.)
    - **Stale-artifact map, now explicit** (all 7 dirs since archived off-repo,
      2026-07-30, so E4 and E3c/E3d now report 0 of 0): 7 result directories were older than
      the indices they name, all dated 2026-07-16..21, and every one of the 6
      current result sets (2026-07-29/30) is clean — 0 unknown ids. The 14
      unknown ids in the stale dirs classify entirely as pre-fix artifacts
      (contamination ids like `academic_resolutions/entity_tags/...`, and
      pre-curriculum-split titles), so they are evidence the fixes worked rather
      than a new problem. `gold_full_embedder_matrix` (5,928 files, 2,479 with
      dead ids) is read by **no** script; the others are read only by the
      superseded pre-9-way scripts. **Risk to note: re-running one of those
      scripts would silently report pre-fix numbers.**
    - **The 8 pre-contamination-fix indices (`n_resolutions`=2874, built
      2026-07-21) are exactly `_EXCLUDED_COMBO_DIRS` in `embedder_matrix_9way.py`**
      — checked name by name, and the one 9-way script that does not import that
      set (`run_gold_hybrid_eval_9way_new.py`) uses an explicit 12-combo
      allowlist. So no current number is computed from a contaminated index.
    - **Both WARNs are now closed** (2026-07-30). (a) 5 query strings were
      duplicated in the 252-entry `gold_query_set.yaml`, all `thematic`, each pair
      carrying a *different* relevant set — and because results are persisted keyed
      by `sha256(query)`, both entries shared one result file and were graded
      against two different answer keys. Chasing that warning found the **real,
      larger defect** it was a symptom of: every one of the 179 thematic entries is
      meeting-scoped but asks about "ครั้งนี้" without naming the meeting (the 5
      duplicates are merely where two meetings produced byte-identical text — their
      gold sets are *disjoint*, shared=0 for all 5 pairs, which is the giveaway).
      Fixed by rewriting all 179 to name their meeting, deriving it from each
      entry's own gold ids; special sessions are spelled out as วาระพิเศษ per
      ADR-0003. The rewrite touched exactly 179 `query:` lines and nothing else,
      was guarded by a byte-identical YAML round-trip, and **invalidated no cache**
      — checked first that no live result directory answers a thematic query.
      `gold_query_set_73det.yaml` was unaffected (0 duplicates, no thematic
      entries), so no cited number moves. (b) 24 `*.md.dup` archives had no live
      counterpart —
      **reviewed one by one 2026-07-30 and now closed: no corpus file was lost.**
      21 are tail fragments of a wrapped title (before the manifest rebuild, a
      title that wrapped produced one file per line — e.g. `และมาตรฐานคุณวุฒิสาขา`
      + `วิชาเภสัชศาสตร์ ระดับ` + `ปริญญาตรี พ.ศ. ๒๕๖๗` were three files for one
      agenda item), 1 is live under a repaired name, and the last 2 are live but
      misfiled. Those verdicts are encoded as rules in C4, not as a list of 24
      reviewed paths, so the check keeps working as the corpus changes; the
      same-document test compares **page-1 `เรื่อง` headings**, because whole-file
      similarity decays across the re-OCR boundary (one confirmed pair sits at
      0.638 full-text with byte-identical headings).
    - The three checks that FAIL on index artifacts (duplicate `chunk_id` in 49
      indices, 1 orphan resolution_id / 23 chunks, coverage 2847/2853) are all the
      **same** pre-relabel debt from #14, traced to exactly the 6 collision ids —
      not separate bugs. **All three now PASS** (49 indices clean, 0 orphans,
      coverage 2853/2853) after the relabel in #14 was applied.
    - **One genuine corpus defect fell out of that review**, and it is the only
      one: `2568/ครั้งที่ 7`'s file titled *รายงานการส่งหลักสูตร…ความสอดคล้อง (CHECO)*
      contains **รับรองรายงานการประชุม** (minutes approval) instead, and no file in
      that meeting holds the CHECO text. **Diagnosed precisely 2026-07-30**: the
      download stage fetched the CHECO agenda item from the wrong Drive id
      (`1Mtr…`, which belongs to the minutes item), producing two byte-identical
      PDFs — same SHA-256 — under two names. The repo's `_LINK.txt`, the manifest
      and `master_list.csv` all already record the *correct* id
      (`1d4iz1dpnPweAn7pxBfxlvJf9IJZwIJFJ`), and no `_LINK.txt` anywhere on the raw
      drive points at it, so the real PDF has simply never been fetched. The fix is
      therefore a targeted re-download + re-OCR of one known URL, with no metadata
      change needed — not a hunt for a lost document. Scope checked both ways: of 24 CHECO-titled files corpus-wide, **1
      is mismatched** (the other 23 all contain `ความสอดคล้อง`), and **0 gold
      queries in `gold_query_set_73det.yaml` cite it** (2 refs in the 252-entry
      set point at 2567/8 and 2567/11, different meetings), so **no reported
      number is affected**. A second file (`2567/ครั้งที่ 6` MoA) carries a title
      truncated mid-subject — incomplete rather than wrong, but it becomes a
      truncated `resolution_id`.
    - **A title-vs-body check was prototyped for that class and rejected on
      measurement, not intuition.** Comparing every manifest title against its
      document's own page-1 heading gives median agreement of only 0.660 across
      2,820 files, with 544 below 0.5 — and the worst-scoring cases are mostly
      artifacts of the comparison itself (a `20. ` agenda-number prefix, ปรับปรุง
      vs ปรับปรุงแก้ไข), not defects. Shipping it as a gate would have meant 544
      false alarms. The scan did show real mismatches beyond the CHECO one (e.g.
      `2568/8` titled *ระบบ E-Portfolio* over a body about a different project),
      so the population is worth a proper investigation — but the metric needs a
      formulation that survives the noise first.
    - **The cleanup that followed broke two checks, in opposite directions — and
      that is the most transferable lesson here.** Archiving the superseded
      artifacts off-repo took C4 (orphaned `.md.dup`) from WARN 24 to PASS 0, not
      because the orphans were resolved but because its subject matter had left the
      directory it scans: *a check whose inputs move becomes a vacuous PASS*. In
      the other direction, deleting the 8 superseded combos removed the only
      indices still holding pre-contamination-fix ids, so E3a jumped 7 → 3,106 for
      result sets no script reads — *a known-retired artifact must not be able to
      hold the gate red*, or the gate stops being read. Fixes: C4 follows the
      archives to their new root and reports a denominator; retired sets are
      classified separately (E3c contamination-artifact ids, E3d earlier-corpus
      titles, `RETIRED_RESULT_DIRS`); the write-only Streamlit dirs are excluded
      from E3b, since interactive queries are not gold by design. Every E3 check
      now prints its denominator, because 0 is otherwise ambiguous between
      "examined and clean" and "nothing left to examine". State on the day it was
      written: **25 checks, 22 pass / 2 warn / 1 fail** — the FAIL the documented
      `BuildCombo.id` caveat, both WARNs the human-judgement items above (5
      duplicate thematic queries, 24 orphan archives), and both closed by hand
      the same day.
    - **Re-run 2026-08-07 against rebuild #3 — and the re-run corrected the
      record.** Result: **24 pass / 0 warn / 1 fail**, `E3a: 0 of 23,156 live
      result files`. That 24/0/1 figure is what this document and `CLAUDE.md` had
      both been quoting since 2026-07-30 — but the report actually on disk that
      day said **21 pass / 3 warn / 1 fail**. The three unreported WARNs were all
      index staleness (`I3b` coverage 2853/2854, `I5` 41 drifted manifests, `I6`
      41 indexes built before the corpus's last edit), and what cleared them was
      **rebuild #3 plus the 08-06/08-07 refresh chain**, not the two closures
      above. The headline is now verified rather than asserted, and the lesson is
      the same one this item is about: a number written into a summary is not
      evidence — the artifact it was copied from is. The check worth reading
      after any rebuild is not the headline count but **`E4` (results newer than
      their index): 0 across all 23,156 live result files**, the mechanical
      confirmation that no persisted result set is older than the index it
      claims to describe.
16. ~~Index rebuild #3 and the refresh of every evaluation path that depends on
    it~~ — **DONE, chain closed 2026-08-07.** `chunker_compare_full` was rebuilt
    (completed 2026-08-05T07:56, ~16.4 h, exit 0 — the 4th attempt and the first
    clean run), taking in the `resolution_id` uniqueness fix and
    `strip_course_comparison_tables`; `entity_tags_full` and the 3 RQ3 treatment
    indices were rebuilt alongside it. Everything downstream was then re-run and
    **verdict-diffed cell by cell** rather than eyeballed
    (`tools/eval/diff_significance_reports.py`):

    | path | date | scope | flips |
    |---|---|---|---|
    | main BM25 / hybrid chain (7 reports) | 08-06 | 171 pairs + 108 cells | **0** |
    | thematic arm (dense + BM25 + hybrid, 5 h 12 m) | 08-07 | 81 cells | **0** |
    | `entity_boost` / `entity_lookup` re-score | 08-06 | — | flat (program MRR +0.0306) |
    | reranker significance test | 08-05 | 6 cells | **0** |
    | RQ4 (contexts rebuilt, 362 of 530 cells regenerated, 4 h 05 m) | 08-07 | 33 tests | **5** |
    | pipeline invariant audit | 08-07 | 25 checks | 24 pass / 0 warn / 1 fail |

    Two things this table should not be read as saying. First, **RQ4's 5 flips
    are not a refresh result** — they sit inside a measured generator noise floor
    (byte-identical prompts reproduce the citation set only 14/24 of the time
    under `cite_all`), all four *lost* verdicts were already borderline
    (Holm-adj 0.014–0.081), and they are reported as **inconclusive, not
    reversed**. Second, **"0 flips" is not the same as "nothing changed"**: the
    refresh is what surfaced three claims in this document that had been
    over-read all along and are corrected above — the RRF sign-flip-at-BM25's-score
    phrasing, RQ4's 4-way precision ordering, and "0 fabricated citations". Prose
    *about* a table has no cell to diff, so it survives every refresh
    automatically; those have to be recomputed by hand.

    One report is deliberately outside the chain:
    `tools/eval/reranker_significance_test.py` re-retrieves live rather than
    reading persisted results, so it must be invoked manually after any rebuild —
    it was, on 08-05.

## Source scripts (for reproducibility / methods section)

- `tools/eval/embedder_matrix_9way.py` — current 9-embedder matrix:
  retrieval + entity-type breakdown + aggregate pairwise significance test
  in one script (supersedes `embedder_significance_test.py` /
  `gold_embedder_breakdown_73det.py`, the original 6-embedder versions)
- `tools/eval/embedder_significance_test_by_entity_type_9way.py` — 9-embedder
  per-entity_type significance test (imports label/exclusion logic from
  `embedder_matrix_9way.py`; supersedes `embedder_significance_test_by_entity_type.py`)
- `tools/eval/run_gold_chunker_eval.py` — chunker-axis eval (embedder fixed)
- `tools/eval/run_gold_bm25_eval.py` — BM25 baseline eval (chunker-only, embedder-agnostic — never needed a 9-way version)
- `tools/eval/run_gold_hybrid_eval.py` — hybrid (RRF) eval, original 24-combo matrix (6 embedders)
- `tools/eval/run_gold_hybrid_eval_9way_new.py` — hybrid (RRF) eval for the 12 new combos
  (`e5_small`, `qwen3_0.6b`, `sct` at max_seq_length=510 × 4 chunkers); writes into the
  same results dir as the original run so downstream scripts glob both together
- `tools/eval/bm25_vs_embedder_significance_test_9way.py` — BM25 vs each of the 9 embedders
  (imports label/exclusion logic from `embedder_matrix_9way.py`; supersedes
  `bm25_vs_embedder_significance_test.py`, the original 6-embedder version)
- `tools/eval/hybrid_significance_test_9way.py` — hybrid vs. dense-alone and vs. BM25-alone,
  all 9 embedders (imports label/exclusion logic from `embedder_matrix_9way.py`; supersedes
  `hybrid_significance_test.py`, the original 6-embedder version)
- `tools/eval/hybrid_significance_test_semantic_top5.py` — embedder-vs-embedder pairwise
  significance test among the top 5 hybrid combos, semantic chunker only, no cross-chunker
  averaging (imports `build_combo_to_chunker_embedder`/`bootstrap_pvalue`/`holm_correct` from
  `embedder_matrix_9way.py`); resolved Open item #8's "crown neither" question
- `tools/eval/hybrid_alpha_sweep.py` — sweeps the RRF fusion weight (`alpha` = dense
  weight) globally and per `entity_type`, 21-point grid × 3 combos, live re-retrieval.
  Caches each arm's rank vector once and re-fuses in numpy, so 21 alphas cost one
  retrieval pass. `--self-check` pins the vectorised fusion against the real
  `BM25Retriever`/`HybridRetriever`/`DenseRetriever` at alpha 0.00/0.50/1.00
- `tools/eval/routing_eval.py` — query-shape routing on the 106-query 73det set, dense
  **and** hybrid arms, reusing persisted results (no retrieval). Four routed variants
  (`prev3`/`shipped`/`oracle`/`loo`) each against a *fitting-budget-matched* single-combo
  baseline. Section 2 is the evidence behind `router.ROUTE_COMBO_BY_RETRIEVER`, including
  the leave-one-out target-stability table the ≥29/30-folds adoption rule reads
- `tools/eval/soft_vs_hard_routing.py` — puts per-route *fusion weight* (soft) and
  per-route *index* (hard) on one axis: 4 arms, each retrieving k=10 from exactly one
  index per query so the retrieval budget is equal. Resolves route→index through
  `query_service.resolve_index`, so the hard arm switches indices the way the shipped
  router does; alpha is the only fitted quantity, LOO within a route. **Its hard arm
  reads `router.route_targets("hybrid")`, so its headline verdict is a function of
  whatever the shipped targets are that day** — editing those constants silently
  re-scores this table, and doing so on 2026-08-08 reversed it. Re-run it after any
  `ROUTE_COMBO_BY_RETRIEVER` change, and date every number taken from it
- `tools/eval/map_precision_significance_test.py` — MAP + precision@1 pairwise significance,
  run at **both** scopes (cross-chunker aggregate, and `semantic`-only to match the existing
  tie test) × both retrievers (dense, hybrid); closed Open item #9's untested-metric half
- `tools/eval/bm25_hybrid_entity_type_breakdown.py` — BM25 and hybrid recall@10 by
  entity_type against the structural ceiling, with ceiling-attainment percentages; closed
  Open item #10, and produced the first direct evidence of the BM25/dense per-category
  complementarity that makes hybrid work
- `tools/eval/multi_k_report.py` — MAP/Precision@k/Recall@k/nDCG@k for k=1,3,5,10 across
  dense/hybrid (9 embedders, aggregated across 4 chunkers) and BM25; pure recompute over
  already-persisted top-10 retrieval results, no re-retrieval needed; closed Open item #4
- `tools/eval/bm25_vs_embedder_significance_test_per_chunker.py` — BM25 vs each of the 9
  embedders, 4 independent per-chunker test families (no cross-chunker averaging); closed
  Open item #1, found `bge_m3` loses to BM25 significantly under `sentence` chunking
  specifically despite tying it in the aggregate
- `tools/eval/bge_qwen_bm25_complementarity.py` — rescue-rate/union-coverage/correlation
  proxies for whether bge_m3's or qwen3's dense-alone errors are more "complementary"
  with BM25; refuted the standing complementarity hypothesis for Open item #2, and the
  swing-query check inside it (37 vs 34, `recursive` chunker) pointed at noise rather
  than a real effect — confirmed by the follow-up bootstrap test below, no GPU needed
- Follow-up inline check (paired bootstrap, bge_m3 vs qwen3 hybrid recall@10, per chunker +
  aggregate, Holm-corrected across 5 tests) — closed Open item #2 for real: no significant
  gap anywhere (closest: sentence Holm-adj p=0.2370), so the "overtake" was never a tested
  effect in the first place
- `tools/eval/congen_sct_truncation_fix_eval.py` — before/after eval for the
  ConGen/SCT max_seq_length investigation
- `tools/eval/gold_embedder_breakdown_73det.py` — per-entity_type breakdown, original 6 embedders
- `tools/eval/embedder_significance_test.py` — 15-pair embedder significance, original 6-embedder version
- `tools/eval/embedder_significance_test_by_entity_type.py` — same, split by entity_type, original 6-embedder version
- `tools/eval/power_analysis.py` — MDE / achieved-power / CI-bound for every comparison
  family the study reports, plus a simulation check of the closed form against the real
  bootstrap; turns each of the 42 ties into a citable bound
  (`data/results/power_analysis.md`)
- `tools/eval/residual_relevance_sample.py` — blinded human-review sheet for
  relevant-but-unjudged top-10 hits, per retrieval arm, testing whether the
  containment-derived qrels are biased toward BM25 rather than merely incomplete;
  `--score` reports per-arm rates with Wilson intervals. Resolved 2026-08-03,
  see the "Resolved 2026-08-03: Pooling bias" section above
  (`data/results/residual_relevance.md`)
- `tools/eval/residual_relevance_decompose.py` — corrected the first (retracted)
  judging pass by reapplying `build_gold_candidates.py`'s own per-entity-type
  matching rule directly against each candidate's full text
- `tools/eval/audit_pipeline_invariants.py` — sweep across corpus/index/eval for
  silent-corruption invariants (Open item #15); **28 checks as of 2026-08-13**, a count
  that grows as new classes are found (23 when this line was first written), so read it
  as a dated snapshot and take the live figure from the report; read-only, exits 1 on
  FAIL, report at `docs/pipeline-invariant-audit.md`. Run it before trusting an eval refresh
- `tools/corpus_prep/audit_resolution_ids.py` — `resolution_id` uniqueness audit
  (Open item #14); read-only, exits 1 on any clash, reports manifest-title vs.
  filename vs. body-heading agreement and whether the files share a source PDF
- `tools/corpus_prep/fix_manifest_title_collisions.py` — the 4 title repairs behind
  that audit's findings (idempotent, verifies before writing)
- `tools/corpus_prep/patch_gold_ids_for_split_titles.py` — re-points the 4 affected
  gold-query references at the two repaired 2567/1 split-piece ids
- Raw result files referenced above all live under `data/results/` (gitignored) —
  regenerate by rerunning the scripts above against `data/index/chunker_compare_full/`.
