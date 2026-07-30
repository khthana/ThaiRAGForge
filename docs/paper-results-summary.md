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
and materially changes rank order.

**Status**: gap-analysis Tier 1 and Tier 2 (`docs/research-framework-gap-analysis.md`
§8) are both fully closed as of 2026-07-21 — MAP/Precision@k/multi-k, BM25 baseline,
bootstrap+Holm significance testing, cost/latency Pareto table, and the `sct` /
`qwen3_0.6b` embedder additions. Tier 3's RQ3 ablations (normalization,
word-aware segmentation, chunk-size sweep) ran to completion 2026-07-23 — see
"RQ3 ablation results" section below. The cross-encoder reranker item also ran
to completion 2026-07-23 — see "Cross-encoder reranker results" section below
(a significant *negative* result for hybrid, literature-grounded). Only RQ4
(end-to-end RAG) remains not started in Tier 3. See the Open items list at the
end of this file for what's still outstanding within the closed tiers.

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
BM25/hybrid numbers; `cost_latency_pareto.py` remains the one exception
(separate cost/latency measurement, not recall — still not re-run, flagged
inline where it's cited).

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
(`thai-embedding-compare.md`, repo root; sourced from the
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
(above) where `qwen3_0.6b` numerically overtakes it — the note below in
"Cost / latency characterization" still cites the pre-refresh figures for
this pairing (that section's table wasn't part of the 2026-07-25 refresh).

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
| program | 30 | 8.2 | 24 | **0.9000** |
| faculty_adjunct_aggregate | 13 | 16.8 | 43 | **0.6810** |
| **all 73 (mean)** | 73 | 8.8 | 43 | **0.8922** |

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
| person | 0.9760 | bge_m3 | 0.8211 | **84.1%** | bge_m3 (0.5735) | 58.8% | 0.8147 | **83.5%** |
| faculty_adjunct_aggregate | 0.6810 | qwen3_0.6b | 0.4922 | **72.3%** | qwen3 (0.4698) | 69.0% | 0.4224 | 62.0% |
| program | 0.9000 | qwen3_0.6b | 0.6187 | **68.7%** | qwen3_0.6b (0.6023) | 66.9% | 0.3484 | 38.7% |
| course | 0.8729 | qwen3_0.6b | 0.5683 | **65.1%** | qwen3_0.6b (0.5500) | 63.0% | 0.3600 | 41.2% |

**This reverses the headroom reading in the paragraph above, which was
dense-alone-specific.** Under the actually-recommended system (hybrid),
`person` is the *most* solved category at 84.1% of its ceiling, not the one
with the most addressable headroom — dense-alone's person weakness (58.8%)
is almost entirely repaired by fusing BM25. The category with the most real
headroom left is now **`course`** (65.1%), which did not exist when the
original ceiling analysis was written. Hybrid does also close the
`faculty_adjunct_aggregate` gap (62.0% BM25 → 72.3% hybrid), answering the
second open question.

**Two findings that only this breakdown makes visible:**

1. **Direct evidence for the lexical/dense complementarity mechanism.**
   BM25 alone reaches **0.8147** on `person` — beating *every* dense
   embedder's dense-alone person score (best: bge_m3 0.5735) by a wide
   margin — while collapsing to **0.3484** on `program`, where dense
   nearly doubles it (qwen3_0.6b 0.6023). **BM25 carries person queries
   (exact name match); dense carries program queries.** This is the
   mechanistic explanation for the hybrid-beats-both result, and it is
   *direct* evidence, unlike the indirect proxies (rescue rate, union
   coverage, per-query correlation) used in the Open item #2 investigation
   that came back inconclusive.
2. **"Hybrid never hurts" is an aggregate statement, not a per-category
   one.** On `person` queries specifically, hybrid is *below* BM25-alone
   (0.8147) for most embedders — `qwen3_0.6b` 0.7220, `qwen3` 0.7340,
   `congen` 0.7228, `jina_v5` 0.7382 — with only `bge_m3` (0.8211)
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
| e5 | +0.0675 (ns) | +0.1188 (**sig**) | +0.1017 (**sig**) | +0.1223 (**sig**) |
| e5_small | +0.1458 (**sig**) | +0.1332 (**sig**) | +0.0818 (ns) | +0.1396 (**sig**) |
| bge_m3 | +0.0819 (ns) | +0.0985 (ns) | +0.0476 (ns) | +0.1081 (**sig**) |
| congen | +0.2360 (**sig**) | +0.2386 (**sig**) | +0.1632 (**sig**) | +0.2627 (**sig**) |
| jina_v5 | +0.1279 (**sig**) | +0.0847 (ns) | −0.0084 (ns) | +0.1142 (**sig**) |
| qwen3 | +0.0553 (ns) | +0.0299 (ns) | −0.0761 (ns) | +0.0522 (ns) |
| qwen3_0.6b | −0.0138 (ns) | −0.0097 (ns) | **−0.1067 (sig)** | −0.0028 (ns) |
| sct | +0.3965 (**sig**) | +0.4251 (**sig**) | +0.3357 (**sig**) | +0.4108 (**sig**) |
| m2v | +0.3995 (**sig**) | +0.3687 (**sig**) | +0.3292 (**sig**) | +0.4021 (**sig**) |

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
needed) — the numbers below are current.

**Dense-alone, top 3 embedders by recall@10 (aggregated across 4 chunkers)**:

| embedder | MAP | recall@1 | recall@3 | recall@5 | recall@10 | precision@1 | precision@5 | ndcg@1 | ndcg@5 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_0.6b | **0.4447** | 0.1225 | 0.2784 | 0.3741 | 0.5263 | **0.7429** | 0.5462 | **0.7429** | 0.6299 |
| qwen3 | 0.3862 | 0.1085 | 0.2547 | 0.3398 | 0.4777 | 0.6368 | 0.4726 | 0.6368 | 0.5505 |
| jina_v5 | 0.3149 | 0.0912 | 0.2041 | 0.2788 | 0.4135 | 0.5165 | 0.3939 | 0.5165 | 0.4563 |

**Hybrid (RRF), top 3 embedders by recall@10 (aggregated across 4 chunkers)**:

| embedder | MAP | recall@1 | recall@3 | recall@5 | recall@10 | precision@1 | precision@5 | ndcg@1 | ndcg@5 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_0.6b | **0.4922** | 0.1231 | 0.3019 | 0.4224 | 0.6167 | **0.7382** | 0.5868 | **0.7382** | 0.6681 |
| qwen3 | 0.4757 | 0.1176 | 0.2886 | 0.4128 | 0.5945 | 0.7099 | 0.5684 | 0.7099 | 0.6457 |
| jina_v5 | 0.4560 | 0.1131 | 0.2838 | 0.3938 | 0.5831 | 0.6745 | 0.5382 | 0.6745 | 0.6163 |

**BM25, aggregated across 4 chunkers**: MAP=0.3845, recall@1=0.1016,
recall@5=0.3392, precision@1=0.5849, ndcg@1=0.5849.

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
| dense / aggregate | MAP | `qwen3_0.6b` (0.4447) | **8 of 8** | — |
| dense / aggregate | precision@1 | `qwen3_0.6b` (0.7429) | **8 of 8** | — |
| dense / semantic | MAP | `qwen3_0.6b` (0.4976) | 3 of 4 | `qwen3` |
| dense / semantic | precision@1 | `qwen3_0.6b` (0.7830) | 3 of 4 | `qwen3` |
| hybrid / aggregate | MAP | `qwen3_0.6b` (0.4922) | 4 of 8 | `qwen3`, `bge_m3`, `e5`, `e5_small` |
| hybrid / aggregate | precision@1 | `qwen3_0.6b` (0.7382) | 4 of 8 | `qwen3`, `bge_m3`, `e5`, `e5_small` |
| hybrid / semantic | MAP | `qwen3` (0.5014) | 1 of 4 (`bge_m3`) | `qwen3_0.6b`, `jina_v5`, `e5_small` |
| hybrid / semantic | precision@1 | `qwen3` (0.7170) | **0 of 4** | all four |

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
~2.1-2.3s of overhead to *every* query, almost independent of which
embedder is in the loop** — because the overhead comes from re-touching the
whole corpus (BM25 rebuild, full-corpus fetch), which scales with corpus
size, not embedding dimension. Found while building this table, worth
stating explicitly rather than silently working around:

1. `DenseRetriever.retrieve()` (`src/rag_lab/retrievers/dense.py`)
   recomputes `np.linalg.norm(embeddings, axis=1)` — the corpus's row norms
   — from scratch on **every query**, even though the corpus (and hence its
   norms) doesn't change between queries. Measured cost (2026-07-29, 74,819
   chunks): 35ms (dim=384) / 97ms (dim=1024) / 246ms (dim=2560), out of a
   ~250-670ms total dense search — roughly a third of dense search time is
   this one avoidable recomputation. (Essentially unchanged from the
   pre-rebuild measurement — this is corpus-size-dependent mechanics, not
   corpus-content-dependent.)
2. `HybridRetriever.retrieve()` (`src/rag_lab/retrievers/hybrid.py`) asks
   **both** sub-retrievers for `k=n` (the entire 74,819-chunk corpus, not a
   bounded candidate pool) before RRF-fusing and truncating to the caller's
   actual k=10 — and `BM25Retriever.retrieve()` (`src/rag_lab/retrievers/bm25.py`)
   separately rebuilds a fresh `BM25Okapi` index from the tokenized corpus
   on **every single query** instead of caching it once per loaded index.
   Measured 2026-07-29: BM25 rebuild-from-scratch = ~872ms vs.
   `get_scores`-only on an already-built index = ~36ms (24x);
   `DenseRetriever.retrieve(k=n)` = ~699ms vs. `retrieve(k=10)` = ~239ms,
   with the ~460ms gap being `RankedChunk` construction (full chunk text
   included) for tens of thousands of chunks nobody will ever look at.
   Together these two effects — not RRF fusion itself — explain most of the
   measured ~2.1-2.7s hybrid query latency. Comparing measured hybrid total
   to intrinsic hybrid estimate directly: the *additive* gap is
   ~1.92-2.03s for every one of the 9 embedders (the tightest-clustered
   number in this whole table) — confirming the overhead is corpus-scanning
   cost, constant regardless of embedder, not an embedder-dependent effect.
   As a *ratio* the same fixed overhead looks very different depending on
   the embedder's own baseline cost — ~4.0x for `qwen3` (2685ms measured vs.
   668ms intrinsic, the most expensive embedder) up to ~17.9x for
   `e5_small` (2083ms vs. 116ms, the cheapest) — so lead with the additive
   number; the ratio is an artifact of which embedder you divide by, not a
   real difference in how much overhead hybrid retrieval carries.

**Because of this, the honest cost signal for a quality-vs-cost comparison
is query-*encode* time (the one component that's genuinely embedder-
dependent and not an artifact of these implementation choices), not the
measured search/hybrid totals.** The table below reports both:

| embedder | dim | encode p50 (ms) | intrinsic dense¹ (ms) | measured dense total p50 (ms) | intrinsic hybrid² (ms) | measured hybrid total p50 (ms) | recall@10 dense (semantic) | recall@10 hybrid (semantic) |
|---|---|---|---|---|---|---|---|---|
| qwen3 | 2560 | 264.66 | 631.97 | 874.72 | 668.34 | 2684.71 | 0.5382 | 0.6051 |
| qwen3_0.6b | 1024 | 177.61 | 320.59 | 421.06 | 356.96 | 2325.41 | **0.5688** | **0.6152** |
| jina_v5 | 1024 | 166.39 | 309.38 | 408.69 | 345.74 | 2294.54 | 0.4705 | 0.5995 |
| bge_m3 | 1024 | 181.58 | 324.56 | 425.03 | 360.93 | 2316.29 | 0.4144 | 0.5451 |
| e5_small | 384 | 24.68 | **79.83** | **120.43** | **116.20** | 2083.14 | 0.3802 | 0.5877 |
| congen | 1024 | 62.18 | 205.16 | 307.30 | 241.53 | 2276.49 | 0.2989 | 0.4666 |
| e5 | 1024 | 183.61 | 326.59 | 424.34 | 362.96 | 2286.53 | 0.3603 | 0.5455 |
| sct | 1024 | 61.82 | 204.81 | 304.81 | 241.18 | 2274.26 | 0.1264 | 0.3971 |
| m2v | 1024 | **2.02** | 145.00 | 247.80 | 181.37 | 2170.54 | 0.1328 | 0.3231 |
| bm25 | — | 0 | — | — | — | 1067.63 (measured; 36.37 intrinsic) | — | 0.4620 (recall@10 alone) |

¹ intrinsic dense = encode p50 + dot-product-and-sort at that dim (norms
cached, not recomputed). ² intrinsic hybrid = encode p50 + dot-product-and
-sort + BM25 `get_scores`-only (BM25 index cached, no k=n over-fetch on
either side; bounded-pool RRF fuse is <5ms, not separately measured). Build
cost (`embed_seconds`, `chunks_per_sec`, index size on disk) and the full
p50/p95 breakdowns for every number above: `data/results/cost_latency_pareto.md`.
**Refreshed 2026-07-29** against the OCR-remediation-rebuilt indices (see
caveat paragraph above) — every quality column moved down from the
pre-rebuild numbers, latency/cost columns essentially unchanged.

**Reading this table**: `m2v` (Model2Vec static embedding) is by far the
cheapest to encode (2ms) but also the weakest embedder in the whole matrix
(see Embedders compared above) — not a real Pareto contender. Among
genuinely competitive embedders, `e5_small` is the standout: intrinsic
dense cost of 80ms (2-8x cheaper than every other option in the top two
quality tiers) for recall@10=0.3802 dense / 0.5877 hybrid — now noticeably
below the hybrid headline number (a 0.028 gap, wider than the pre-rebuild
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
   fetching the full corpus before fusing) that add a roughly fixed
   ~2.1-2.3s of latency to every hybrid query regardless of embedder
   (expressed as a ratio this is ~4x for the most expensive embedder up to
   ~18x for the cheapest, purely because the same fixed overhead is divided
   by very different intrinsic baselines — the additive number is the real
   story) — reported as implementation characteristics, not silently fixed.
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
   segmentation do not. RQ4 (end-to-end RAG answer quality) remains
   explicitly out of scope for this first paper per the gap analysis — later
   phase.
5b. ~~Cross-encoder reranker (Tier 3 item 8)~~ — DONE 2026-07-23, **refreshed
    2026-07-29**, see "Cross-encoder reranker results" section above:
    significantly hurts hybrid MRR only (nDCG@10 no longer significant post-
    refresh, a real finding-level change), no significant effect on
    dense-alone, literature-grounded explanation in
    `docs/reranker-hybrid-interaction-research.md`. Only RQ4 remains
    unstarted in Tier 3.
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
    `semantic` — configs in `config/experiments/rebuild_clean_*.yaml`)
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
    `tools/eval/audit_pipeline_invariants.py` now checks 23 such invariants
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
    - **Stale-artifact map, now explicit**: 7 result directories are older than
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
    - **Two WARNs worth a human look**: (a) 5 query strings are duplicated in the
      252-entry `gold_query_set.yaml`, all `thematic`, each pair carrying a
      *different* relevant set — and because results are persisted keyed by
      `sha256(query)`, both entries share one result file and get graded against
      two different answer keys. `gold_query_set_73det.yaml` has 0 duplicates, and
      thematic queries are already excluded from every cited result, so nothing
      above is affected. (b) 24 `*.md.dup` archives have no live counterpart; 20
      reconcile to a renamed/merged live file, 3-4 do not and need eyes (all are
      administrative items — minutes approval, a Joint-Degree subsidy filing —
      not curriculum documents).
    - The three checks that FAIL on index artifacts (duplicate `chunk_id` in 49
      indices, 1 orphan resolution_id / 23 chunks, coverage 2847/2853) are all the
      **same** pre-relabel debt from #14, traced to exactly the 6 collision ids —
      not separate bugs. They clear when the relabel in #14 is done.

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
- `tools/eval/audit_pipeline_invariants.py` — 23-check sweep across corpus/index/eval
  for silent-corruption invariants (Open item #15); read-only, exits 1 on FAIL,
  report at `docs/pipeline-invariant-audit.md`. Run it before trusting an eval refresh
- `tools/corpus_prep/audit_resolution_ids.py` — `resolution_id` uniqueness audit
  (Open item #14); read-only, exits 1 on any clash, reports manifest-title vs.
  filename vs. body-heading agreement and whether the files share a source PDF
- `tools/corpus_prep/fix_manifest_title_collisions.py` — the 4 title repairs behind
  that audit's findings (idempotent, verifies before writing)
- `tools/corpus_prep/patch_gold_ids_for_split_titles.py` — re-points the 4 affected
  gold-query references at the two repaired 2567/1 split-piece ids
- Raw result files referenced above all live under `data/results/` (gitignored) —
  regenerate by rerunning the scripts above against `data/index/chunker_compare_full/`.
