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
`qwen3_0.6b` embedder additions. Tier 3 (RQ3 normalization ablation, cross-encoder
reranker, RQ4 end-to-end RAG) has not been started. See the Open items list at the
end of this file for what's still outstanding within the closed tiers.

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

**Aggregate (averaged across all 4 chunkers)**:

| embedder | recall@10 | mrr | ndcg@10 |
|---|---|---|---|
| qwen3_0.6b | **0.5198** | **0.7918** | **0.6078** |
| qwen3 | 0.5155 | 0.7848 | 0.5912 |
| bge_m3 | 0.5107 | 0.7543 | 0.5717 |
| jina_v5 | 0.4503 | 0.7057 | 0.5167 |
| e5 | 0.4265 | 0.6630 | 0.4802 |
| e5_small | 0.4197 | 0.6370 | 0.4602 |
| congen | 0.4134 | 0.6535 | 0.4726 |
| sct | 0.1519 | 0.2586 | 0.1690 |
| m2v | 0.1472 | 0.3107 | 0.1846 |

**Significance (36 pairwise tests, Holm-corrected per metric)**. Full table:
`data/results/embedder_significance_test_9way.md`; script:
`tools/eval/embedder_matrix_9way.py`.

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

**Best single combo in the whole 24-combo matrix**: `semantic × qwen3`
(recall@10=**0.6581**, MRR=**0.8831**, nDCG@10=**0.7339**) — clear leader by
a wide margin over 2nd place (`semantic × jina_v5` = 0.5845/0.8104/0.6493).
`qwen3_0.6b`'s per-chunker dense-alone number is now known (see "Cost /
latency characterization" below): `qwen3_0.6b × semantic` = 0.6364, below
`qwen3 × semantic`'s 0.6581 — `qwen3` keeps its dense-alone lead
per-chunker, unlike the hybrid case where `qwen3_0.6b` numerically
overtakes it.

## Embedder × entity_type profile (the "specialist vs generalist" finding)

Cross-chunker average recall@10, broken out by query entity_type. Full
table: `data/results/gold_embedder_breakdown_9way.md`.

| embedder | faculty_adjunct (n=13) | person (n=30) | program (n=30) |
|---|---|---|---|
| bge_m3 | 0.4555 | **0.5694** | 0.4760 |
| qwen3 | 0.4741 | 0.4807 | 0.5682 |
| qwen3_0.6b | 0.4546 | 0.4283 | **0.6396** |
| congen | 0.3966 | 0.2608 | 0.5732 |
| sct | 0.2400 | 0.0571 | 0.2084 |
| jina_v5 | 0.4130 | 0.4285 | 0.4881 |
| e5 | 0.4603 | 0.4686 | 0.3697 |
| e5_small | 0.4048 | 0.4485 | 0.3974 |
| m2v | 0.2215 | 0.0572 | 0.2049 |

**Per-entity_type significance, full 9-embedder matrix (Holm-corrected per
entity_type × metric, 36-pair families)**. Full table:
`data/results/embedder_significance_test_by_entity_type_9way.md`; script:
`tools/eval/embedder_significance_test_by_entity_type_9way.py`.

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

## BM25 lexical baseline

`src/rag_lab/retrievers/bm25.py` (`rank_bm25.BM25Okapi` over PyThaiNLP
`word_tokenize`, engine `newmm` — dictionary-based maximum matching
constrained by Thai Character Cluster boundaries, tokenizes the full chunk
text, not just title/metadata). One run per chunker (embedder-agnostic):

| chunker | recall@10 | mrr | ndcg@10 |
|---|---|---|---|
| semantic | 0.5902 | 0.7690 | 0.6174 |
| sentence | 0.5801 | 0.7955 | 0.6379 |
| recursive | 0.5526 | 0.7491 | 0.5889 |
| fixed_size | 0.5476 | 0.8019 | 0.6126 |

**BM25 aggregate (averaged across its 4 chunker runs, same framing as the
embedder table above)**: recall@10=0.5676, mrr=0.7789, ndcg@10=0.6142. (BM25
is chunker-only / embedder-agnostic — this table doesn't change when the
embedder matrix grows; extended below is only the significance comparison,
now against all 9 embedders.)

**Significance (9 BM25-vs-embedder tests, Holm-corrected per metric —
`tools/eval/bm25_vs_embedder_significance_test_9way.py`, updated 2026-07-21
to add `e5_small`, `qwen3_0.6b`, `sct` at its corrected max_seq_length=510)**:

| vs. | recall@10 diff (BM25 − X) | Holm-adj p | significant |
|---|---|---|---|
| sct | +0.4158 | 0.0000 | **yes** |
| m2v | +0.4205 | 0.0000 | **yes** |
| congen | +0.1543 | 0.0080 | **yes** |
| e5_small | +0.1479 | 0.0000 | **yes** |
| e5 | +0.1411 | 0.0000 | **yes** |
| jina_v5 | +0.1174 | 0.0080 | **yes** |
| bge_m3 | +0.0569 | 0.2748 | no |
| qwen3 | +0.0521 | 0.3904 | no |
| qwen3_0.6b | +0.0478 | 0.3904 | no |

**Headline finding (unchanged, now confirmed against the full 9-embedder
matrix)**: **BM25 — free, no GPU, no training — statistically ties the
three best dense embedders (bge-m3, Qwen3-4B, and now also Qwen3-0.6B) and
significantly beats every weaker one (ConGen, e5, e5_small, jina_v5, m2v,
and now also sct)**. Framed for the paper: *only an embedder in the
bge-m3/Qwen3 quality tier is provably worth its inference cost over plain
BM25 for this task; a weaker embedder is not.* **New for sct**: at its
corrected 510-token context (see "Resolved 2026-07-21" above), sct's
dense-alone recall@10 is 0.1519 — statistically indistinguishable from
m2v's 0.1472 (both near-random) — so sct joins m2v as a second embedder
BM25 beats by a very wide margin (+0.42 recall), not just a modest one.
Working hypothesis for why BM25 is this strong: Gold queries are
entity-anchored — even though phrasing is rephrased away from document
titles, the anchor entity's literal name (person/program/faculty) has to
stay verbatim to specify which resolution is being asked about, which gives
exact lexical match a structural advantage on this specific task. Not yet
confirmed against a genuinely paraphrased/thematic-only query set (which
would need higher discrimination than the current thematic queries have).

## Hybrid retrieval (RRF: BM25 + Dense) — the overall best system found

`src/rag_lab/retrievers/hybrid.py` (Reciprocal Rank Fusion, `rrf_k=60`
default) fuses BM25 and dense rankings from the **same** index — no rebuild
needed, every combo already carries both `embeddings.npy` and
`lexical.json`. Full 24-combo matrix run on Gold 73-det
(`tools/eval/run_gold_hybrid_eval.py`), extended 2026-07-21 with 12 more
combos for `e5_small`, `qwen3_0.6b`, `sct` (at its corrected
max_seq_length=510) via `tools/eval/run_gold_hybrid_eval_9way_new.py` — 36
combos total, all 9 embedders now covered.

**Aggregate recall@10 (averaged across the 4 chunkers), hybrid vs. its two components**:

| embedder | hybrid | dense-alone | bm25-alone |
|---|---|---|---|
| **qwen3_0.6b** | **0.6543** | 0.5198 | 0.5676 |
| bge_m3 | 0.6472 | 0.5107 | 0.5676 |
| congen | 0.6426 | 0.4134 | 0.5676 |
| jina_v5 | 0.6383 | 0.4503 | 0.5676 |
| e5 | 0.6264 | 0.4265 | 0.5676 |
| qwen3 | 0.6235 | 0.5155 | 0.5676 |
| e5_small | 0.6228 | 0.4197 | 0.5676 |
| sct | 0.5179 | 0.1519 | 0.5676 |
| m2v | 0.3244 | 0.1472 | 0.5676 |

**Significance** (`tools/eval/hybrid_significance_test_9way.py`, updated
2026-07-21, two 9-test families — hybrid vs. dense-alone, hybrid vs.
BM25-alone — Holm-corrected separately per metric):

- **Hybrid significantly beats dense-alone for every one of the 9
  embedders, on every metric** (Holm-adj p<0.01 in nearly all cases; the
  sole exception is qwen3 on MRR). This is the single most robust finding
  in the whole study, and it **survives the extension to all 9 embedders
  including the two very weak dense-alone models (sct, m2v)**: **adding
  BM25 to a dense retriever never hurts and almost always helps
  significantly** — even a near-random dense signal doesn't make hybrid
  worse than dense-alone.
- **Hybrid significantly beats BM25-alone on recall@10** for 7 of the 9
  embedders (qwen3_0.6b/bge_m3/congen/jina_v5/e5/qwen3/e5_small, Holm-adj
  p≤0.03) — but this does **not** reliably hold on MRR (**none** of the 7
  positive embedders reach significance after Holm correction, best
  Holm-adj p=0.063 for congen) or nDCG@10 (4/7 do: qwen3_0.6b, bge_m3,
  congen, jina_v5). Interpretation: adding dense to BM25 mainly helps
  **recall** (surfacing more relevant docs somewhere in top-10), not
  **ranking precision** (getting the single best doc to rank #1) — those
  are genuinely different benefits, don't conflate them in the writeup.
- **m2v and now also sct are cautionary counter-examples**: hybrid with
  m2v is significantly *worse* than BM25 alone (recall diff=−0.2433,
  p<0.001), and — new 2026-07-21 finding — **hybrid with sct is also
  significantly worse than BM25 alone** (recall diff=−0.0497, p=0.0312;
  even more lopsided on MRR: −0.1349, p<0.001). This is a real RRF failure
  mode when one fused signal is weak enough that equal-weight RRF lets it
  actively demote true positives that BM25 alone got right — and it now has
  **two independent confirmations**: m2v (a non-transformer static
  embedding never intended for this) and sct (a PhayaThaiBERT model whose
  training regime, per the "Resolved 2026-07-21" section above, makes it
  perform near dense-alone-random at 510 tokens, recall@10=0.1519, tied
  with m2v's 0.1472). **RRF fusion is not automatically safe; it assumes
  both signals are individually competent** — a dense signal this weak
  drags hybrid below BM25-alone even though it still doesn't drag hybrid
  below dense-alone (previous bullet).
- **Top single-combo tier across the entire study (dense-alone, BM25-alone,
  and hybrid, all 76 configurations across 9 embedders × 4 chunkers × 3
  retrieval modes)**: all under `semantic × hybrid`, a tight, **not yet
  per-chunker significance-tested** cluster — `qwen3_0.6b` **0.6935**
  (numerically highest), `bge_m3` 0.6845, `e5_small` 0.6821, `qwen3` 0.6797,
  `jina_v5` 0.6796 — a spread of 0.014 across the top five. The per-chunker
  check (Open item #8) is now done (see "Cost / latency characterization"
  below); the aggregate hybrid table above shows the same ordering
  (`qwen3_0.6b` 0.6543 > `bge_m3` 0.6472). **No embedder in this cluster can
  be cited as "the best combo"** until a dedicated per-chunker significance
  test runs — don't promote `qwen3_0.6b` past the others, but don't keep
  asserting `bge_m3` is on top either; that claim was never significance-
  tested in the first place, it was just the highest number known before
  this check. All five clearly beat the best dense-alone combo,
  `semantic × qwen3` (0.6581). Notable: `bge_m3` and `qwen3_0.6b` both
  overtake `qwen3` once hybridized despite `qwen3` being the strongest (or
  tied-strongest) dense-alone embedder — suggests error-pattern
  complementarity with BM25 varies by embedder, though the reason hasn't
  been investigated further (Open item #2).

**Headline system recommendation for the paper**: the best-performing
configuration overall is not a pure embedder choice but **semantic chunking
+ hybrid retrieval (BM25 + dense, RRF)** — lexical and dense signals are
genuinely complementary here, not redundant, contrary to the concern raised
before running this experiment. Which embedder to pair it with is an open,
untested horse race among the top five above (see Open item #8) — the
system-level claim (semantic chunking, hybrid retrieval) is what's robust,
not a specific embedder pick.

## Cost / latency characterization

Full data + methodology: `tools/eval/cost_latency_pareto.py` (run with
`--reuse-latency-cache` to reuse a prior measurement instead of repeating
~20 min of sequential model loading), rendered report at
`data/results/cost_latency_pareto.md` (gitignored, regenerate by rerunning
the script). All numbers below are measured on each embedder's `semantic`-
chunker combo — the same combos the quality numbers elsewhere in this doc
that are chunker-specific refer to — so cost and quality columns in the
table below are apples-to-apples with each other, unlike an earlier
internal draft of this table which paired semantic-chunker latency against
cross-chunker-*aggregate* recall (a mismatch caught before being cited
anywhere).

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
   norms) doesn't change between queries. Measured cost: 39ms (dim=384) /
   119ms (dim=1024) / 287ms (dim=2560), out of a ~270-670ms total dense
   search — roughly a third of dense search time is this one avoidable
   recomputation.
2. `HybridRetriever.retrieve()` (`src/rag_lab/retrievers/hybrid.py`) asks
   **both** sub-retrievers for `k=n` (the entire 81,489-chunk corpus, not a
   bounded candidate pool) before RRF-fusing and truncating to the caller's
   actual k=10 — and `BM25Retriever.retrieve()` (`src/rag_lab/retrievers/bm25.py`)
   separately rebuilds a fresh `BM25Okapi` index from the tokenized corpus
   on **every single query** instead of caching it once per loaded index.
   Measured: BM25 rebuild-from-scratch = ~1.0s vs. `get_scores`-only on an
   already-built index = ~43ms (23x); `DenseRetriever.retrieve(k=n)` = ~765ms
   vs. `retrieve(k=10)` = ~277ms, with the ~490ms gap being `RankedChunk`
   construction (full chunk text included) for tens of thousands of chunks
   nobody will ever look at. Together these two effects — not RRF fusion
   itself — explain most of the measured ~2.3-2.9s hybrid query latency.
   Comparing measured hybrid total to intrinsic hybrid estimate directly:
   the *additive* gap is ~2.15-2.26s for every one of the 9 embedders (the
   tightest-clustered number in this whole table) — confirming the overhead
   is corpus-scanning cost, constant regardless of embedder, not an
   embedder-dependent effect. As a *ratio* the same fixed overhead looks
   very different depending on the embedder's own baseline cost — ~4x for
   `qwen3` (2914ms measured vs. 727ms intrinsic, the most expensive embedder)
   up to ~18x for `e5_small` (2326ms vs. 130ms, the cheapest) — so lead with
   the additive number; the ratio is an artifact of which embedder you
   divide by, not a real difference in how much overhead hybrid retrieval
   carries.

**Because of this, the honest cost signal for a quality-vs-cost comparison
is query-*encode* time (the one component that's genuinely embedder-
dependent and not an artifact of these implementation choices), not the
measured search/hybrid totals.** The table below reports both:

| embedder | dim | encode p50 (ms) | intrinsic dense¹ (ms) | measured dense total p50 (ms) | intrinsic hybrid² (ms) | measured hybrid total p50 (ms) | recall@10 dense (semantic) | recall@10 hybrid (semantic) |
|---|---|---|---|---|---|---|---|---|
| qwen3 | 2560 | 264 | 685 | 947 | 727 | 2914 | **0.6581** | 0.6797 |
| qwen3_0.6b | 1024 | 187 | 360 | 462 | 402 | 2585 | 0.6364 | **0.6935** |
| jina_v5 | 1024 | 167 | 340 | 440 | 382 | 2548 | 0.5845 | 0.6796 |
| bge_m3 | 1024 | 167 | 340 | 447 | 383 | 2630 | 0.5822 | 0.6845 |
| e5_small | 384 | 25 | **87** | **126** | **130** | 2326 | 0.4822 | 0.6821 |
| congen | 1024 | 62 | 235 | 333 | 278 | 2538 | 0.4726 | 0.6653 |
| e5 | 1024 | 188 | 361 | 458 | 403 | 2551 | 0.4675 | 0.6383 |
| sct | 1024 | 62 | 235 | 331 | 277 | 2508 | 0.2089 | 0.5750 |
| m2v | 1024 | **2** | 175 | 276 | 218 | 2391 | 0.1983 | 0.3966 |
| bm25 | — | 0 | — | — | — | 1203 (measured; 43 intrinsic) | — | 0.5902 (recall@10 alone) |

¹ intrinsic dense = encode p50 + dot-product-and-sort at that dim (norms
cached, not recomputed). ² intrinsic hybrid = encode p50 + dot-product-and
-sort + BM25 `get_scores`-only (BM25 index cached, no k=n over-fetch on
either side; bounded-pool RRF fuse is <5ms, not separately measured). Build
cost (`embed_seconds`, `chunks_per_sec`, index size on disk) and the full
p50/p95 breakdowns for every number above: `data/results/cost_latency_pareto.md`.

**Reading this table**: `m2v` (Model2Vec static embedding) is by far the
cheapest to encode (2ms) but also the weakest embedder in the whole matrix
(see Embedders compared above) — not a real Pareto contender. Among
genuinely competitive embedders, `e5_small` is the standout: intrinsic
dense cost of 87ms (2-8x cheaper than every other option in the top two
quality tiers) for recall@10=0.4822 dense / 0.6821 hybrid — within 0.02 of
the hybrid headline number despite the cheapest intrinsic cost by far in
that tier. `qwen3` (4B) is the most expensive per query in every column and
does not lead on hybrid recall despite that — its cost is justified mainly
by the entity-type robustness finding above (no significant weak spot),
not by raw recall@10.

**New finding, surfaced by computing semantic-chunker-specific numbers for
this table (resolves Open item #8)**: `qwen3_0.6b × semantic × hybrid`
(recall@10=**0.6935**) is numerically *higher* than every other single
combo in the whole study, including `bge-m3 × semantic × hybrid`
(recall@10=0.6845, previously cited as the top combo — a +0.009 gap).
**This has not been significance-tested per-chunker** (the existing
significance tests compare cross-chunker aggregates, where qwen3_0.6b
already numerically leads bge_m3 too — 0.6543 vs 0.6472 — but that pairwise
comparison hasn't been significance-tested either; the hybrid significance
tests run so far only test each embedder against dense-alone and against
BM25-alone, not against each other). Given every other close pairwise
comparison in this study that looked large in a raw descriptive table (e.g.
the ConGen/SCT truncation question) turned out to need an explicit
bootstrap test before citing, the same caution applies here: **treat
`qwen3_0.6b × semantic × hybrid` as numerically ahead, not yet confirmed
ahead** — a real ~0.009 gap on 73 queries is well within the range that
could flip under a paired bootstrap given how close every top-tier hybrid
combo is (0.65-0.69 across five embedders: qwen3_0.6b, bge_m3, e5_small,
qwen3, jina_v5). Note that `bge-m3` never had a stronger evidentiary basis
for "best" than `qwen3_0.6b` does now — it was simply the highest number
known before this check, not a significance-tested claim either. So the
correct interim stance is to **crown neither**: report the top five as an
untested cluster, name `qwen3_0.6b` as numerically highest, and let the
system-level recommendation (semantic chunking + hybrid retrieval) carry
the paper's headline rather than a specific embedder pick — until a
dedicated per-chunker significance test (analogous to
`hybrid_significance_test_9way.py` but restricted to the semantic chunker)
resolves the ordering. Not yet run, added as a new Open item.

## Open items (not yet done, needed before the numbers above are "final")

1. Per-chunker point comparison of BM25 vs. embedder (not averaged across
   chunkers) not yet significance-tested — raw numbers there look more
   favorable to BM25 than the aggregate view; worth checking if that's real
   or a chunker-selection artifact.
2. Why bge-m3 overtakes qwen3 specifically under hybrid despite tying it as
   dense-alone — not investigated (error-pattern complementarity with BM25
   is a guess, not verified).
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
   report tables now render precision@k and map columns too. **Not yet
   done**: no eval script has actually been *re-run* with a multi-k list
   (e.g. `k=[1,3,5,10]`) — the capability exists but every number in this
   doc is still k=10 only; re-running with multi-k and citing MAP/P@k
   numbers is a follow-up, not part of this change.
5. RQ3 (normalization/segmentation ablation) and RQ4 (end-to-end RAG answer
   quality) are explicitly out of scope for this first paper per the gap
   analysis — later phase.
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
   not yet checked per-chunker~~ — CHECKED 2026-07-21 (see "Cost / latency
   characterization" section above): `qwen3_0.6b × semantic × hybrid` =
   0.6935, numerically the highest single combo in the whole study — ahead
   of `bge-m3 × semantic × hybrid` (0.6845, the number previously cited as
   "best") and `e5_small × semantic × hybrid` (0.6821), also close. **Still
   open**: none of these gaps have been significance-tested per-chunker
   (existing significance tests only cover cross-chunker aggregates and each
   embedder vs. its own dense-alone/BM25-alone baseline, not embedder-vs-
   embedder within the semantic chunker) — treat the top five hybrid combos
   as an untested cluster (see "Top single-combo tier" bullet above); don't
   cite any one of them, `bge-m3` included, as the confirmed best until that
   test runs. New Open item: a per-chunker (semantic-only) pairwise
   significance test across the top hybrid combos (qwen3_0.6b, bge_m3,
   e5_small, jina_v5, qwen3), analogous to `hybrid_significance_test_9way.py`
   but restricted to one chunker instead of averaging across all 4.

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
- `tools/eval/congen_sct_truncation_fix_eval.py` — before/after eval for the
  ConGen/SCT max_seq_length investigation
- `tools/eval/gold_embedder_breakdown_73det.py` — per-entity_type breakdown, original 6 embedders
- `tools/eval/embedder_significance_test.py` — 15-pair embedder significance, original 6-embedder version
- `tools/eval/embedder_significance_test_by_entity_type.py` — same, split by entity_type, original 6-embedder version
- Raw result files referenced above all live under `data/results/` (gitignored) —
  regenerate by rerunning the scripts above against `data/index/chunker_compare_full/`.
