# ColBERT / late interaction — notes and a pre-registered test

Status: **STARTED 2026-08-13** at the user's request. Encoder + MaxSim written
(`src/rag_lab/colbert/`), lengths profiled, checkpoint qualified, and the encoder
cross-checked against pylate — which found a real defect the 11-check gate could
not (`mask_punctuation` was masking whitespace, not punctuation). Nothing is
indexed and nothing is measured yet, so **the pre-registered prediction in
§"A pre-registered prediction" is still open**. See **§"Build log (2026-08-13)"**
for what is settled so far, and read it *after* the prediction, not before.

Everything before the build log is the original 2026-07-30 write-up, kept as
written: it is the motivation the prediction was registered against, and
rewriting it after the fact would destroy the only reason the prediction means
anything. This is a *new research axis*, not a small experiment; the intended
order was RQ4 first.

**2026-08-07 — RQ4 is complete, so the stated blocker is gone; this is still not
the next thing to do.** Three cheaper items now rank above it, all justified by
this project's own measurements rather than by the literature: (1) caching
`BM25Okapi` instead of rebuilding it per query, which returns a fixed ~1.9–2.0 s
of every hybrid query for no research risk; (2) sweeping the fusion weight —
`HybridRetriever` has a `method="weighted"` path with `dense_weight`/`bm25_weight`
that **has never been used**, so every hybrid number in the project sits at an
implicit 50:50, and the "RRF helps the weaker arm and taxes the stronger one"
rule (r = −0.921) predicts where a sweep pays; (3) the two cheap graph edges.
ColBERT stays ranked where it is because it is the only remaining item whose
index-side cost is comparable to a full rebuild. The pre-registered prediction
below is unchanged and still the point of the exercise.

## What it is

Three retrieval architectures, of which this project currently uses the outer two:

| architecture | encodes | scoring | in this project |
|---|---|---|---|
| bi-encoder (dense) | whole chunk → **one** vector | dot product, precomputed | all 9 embedders |
| **ColBERT (late interaction)** | **each token → its own vector** | MaxSim: each query token takes its best-matching document token, summed | — |
| cross-encoder | nothing precomputed; query+doc go through the model together | full attention | `bge-reranker-v2-m3` (tested, **harmful**) |

The name is the mechanism: interaction between query and document is *deferred*
to query time, but the document side is still precomputed, so it remains indexable.
A bi-encoder destroys token-level detail at index time by pooling a whole chunk
into one vector; a cross-encoder keeps everything but cannot precompute anything.
ColBERT keeps per-token vectors and pays for it in storage instead of latency.

Primary references (verify details before citing):
- Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via
  Contextualized Late Interaction over BERT*, SIGIR 2020 — the architecture.
- Santhanam et al., *ColBERTv2: Effective and Efficient Retrieval via Lightweight
  Late Interaction*, NAACL 2022 — residual compression, which is what makes the
  storage argument below work.
- `jina-colbert-v2` (2024) — multilingual late-interaction checkpoint; the
  practical option here, since most ColBERT checkpoints are English-only.

## Why it is a good fit for *this* project specifically

Not a generic "newer is better" argument — three of this project's own measured
results point at it.

1. **Our cross-encoder reranking failed, and late interaction is the principled
   alternative to it.** `bge-reranker-v2-m3` significantly *hurt* hybrid MRR
   (0.7775 → 0.6775, Holm-adj p=0.0048), with the literature's "phantom hits"
   mechanism: a reranker trained on first-stage candidates disrupts early ranks
   when applied to fused ones. ColBERT delivers the token-level matching a
   cross-encoder provides, but as the **retriever itself** rather than as a
   truncate-and-replace stage on top of someone else's ranking — so the specific
   failure mode we measured does not apply.

2. **It targets the exact complementarity we quantified.** The per-entity_type
   breakdown found BM25 carries `person` outright (0.8147, beating every dense
   embedder's best of 0.5735) while collapsing on `program` (0.3484), where dense
   nearly doubles it (0.6023). That is the classic exact-match-vs-semantics split,
   and MaxSim is exact-ish matching *in embedding space* — the one mechanism that
   could plausibly do both without RRF. Worth noting we also measured that RRF is
   not free: fusing a weak dense signal with BM25 (`m2v`, `sct`) is significantly
   *worse* than BM25 alone.

3. **It sidesteps Thai word segmentation on the lexical side.** BM25 here depends
   on `newmm` segmentation; ColBERT scores over subword tokens with no word
   boundaries required. RQ3 found word-aware segmentation made no difference to
   *chunking*, but that says nothing about lexical matching, where segmentation is
   load-bearing.

## Storage: not the blocker it is assumed to be

Sized against a real index (`plain__semantic__local__834c4336`):

| | value |
|---|---|
| chunks | 74,819 |
| text | 23.4M characters (mean 313/chunk) |
| estimated tokens | ~8-12M (Thai at roughly 2-3 chars/token on multilingual subword vocabularies — **measure with the actual tokenizer before relying on this**) |
| current single-vector index | 1024-dim float32 = **306 MB** |
| ColBERT raw, 128-dim fp16 | ~2.5-3 GB |
| ColBERT with ColBERTv2 residual compression (~32 B/token) | **~300-400 MB** |

So a compressed late-interaction index lands in the same order of magnitude as
the single-vector index we already ship. On a 12 GB card this is workable.

## The real blockers

1. **It breaks an index invariant we now enforce.** `chunks.parquet` and
   `embeddings.npy` are strictly row-aligned, one row per chunk — audit check I1
   verifies exactly that, and that alignment is what makes `resolution_id`
   attribution safe. ColBERT is many vectors per chunk, so it needs a different
   artifact shape plus an I1 variant that checks *chunk→token-block* alignment
   instead. This is the actual work; the retriever itself is a registry entry
   (ADR-0001 open/closed — new file + register, no runner edit).
2. **Checkpoint choice.** `jina-colbert-v2` is the realistic multilingual option
   and conveniently the same family as the `jina_v5` embedder already in the
   matrix. Whether it handles Thai council prose well is an empirical question, not
   an assumption — a pilot on a subset should precede any full index build
   (`feedback_scan_before_broad_preprocessing_fix`).
3. **Scope.** ColBERT is a retrieval-side axis that sits alongside RQ1-RQ2. This
   used to read "RQ4 is unstarted and is the gap that blocks the paper";
   **RQ4 completed 2026-08-03** (and has been refreshed twice since), so that is
   no longer the reason to defer. What replaces it is a sharper one, measured
   after these notes were written: the routed oracle over a P=50 pool delivers
   **0.8331** against arm C's 0.6831 (`data/results/reranker_rrf_routed_test.md`),
   i.e. **+0.1500 of headroom that a better ranker could reach and
   `bge-reranker-v2-m3` reaches 1% of** — and swapping the model moves recall@10
   by 20x the anchor's own effect (`reranker_model_comparison.md`). So the axis is
   motivated by a *measured* ceiling now, not by a gap in the study.

## A pre-registered prediction, so this is falsifiable

If the motivation in §2 is right, ColBERT-alone should show a **specific**
signature on the per-entity_type breakdown, not merely a better aggregate:

> ColBERT-alone ties or beats **BM25** on `person` **and** ties or beats the best
> **dense** embedder on `program`, in the same run.

That is the thing neither family can currently do alone. Writing it down now
matters: an aggregate improvement could equally come from simply being a bigger,
better-trained model, which would be a much weaker claim. If ColBERT wins overall
but keeps the same person/program split, the honest conclusion is "a stronger
retriever", not "late interaction resolves the complementarity".

Evaluation cost is low — it reuses `gold_query_set_73det.yaml`, the existing
paired-bootstrap + Holm machinery, and `bm25_hybrid_entity_type_breakdown.py`
unchanged. Only the index build and a `ColbertRetriever` are new.

> **OUTCOME (2026-08-13, re-run 2026-08-20 against rebuild #4 with the verdict
> unchanged): the prediction FAILED — `person` clears as a tie, `program` loses
> by −0.3337 (Holm 0.0000), verdict STOP.** Everything above this
> line is the untouched pre-registration; the numbers and the mechanism are in
> §"The pilot ran" below. The paragraph two above turned out to describe this run
> exactly: ColBERT *does* carry the best overall figure (0.5555) while keeping the
> person/program split — so the honest conclusion is "a stronger retriever", which
> is why it was written down first.

## Build log (2026-08-13)

### What exists

| artifact | what it is |
|---|---|
| `src/rag_lab/colbert/encoder.py` | `ColbertEncoder` — marker insertion, query augmentation, the projection head `AutoModel` does not load, L2 normalisation, and the rotary repair below. Deliberately **not** a `BaseEmbedder`: that interface is one row per text and `Index` is row-aligned on it (audit check I1), while ColBERT is many vectors per chunk. |
| `src/rag_lab/colbert/scoring.py` | `maxsim` (packed `np.maximum.reduceat`) plus `maxsim_reference`, the definition written out one document at a time and kept as an independent check of the optimisation. |
| `tools/eval/colbert_length_profile.py` → `data/results/colbert_length_profile.md` | what the caps cost this corpus. |
| `tools/eval/qualify_colbert_model.py` → `data/results/colbert_model_qualification.md` | the gate. Nothing measured with this checkpoint is citable until it passes in both directions. |

### The length decision, and it is a stated confound rather than a default

`doc_maxlen`/`query_maxlen` are **ColBERT conventions, not model limits** — this
checkpoint is rotary and its card claims 8192 tokens — so truncation here is a
*choice*, and one applied to the treatment alone: the dense arms ColBERT is
compared against read the whole chunk (bge-m3 8192, qwen3 32k). Measured with
the model's own tokenizer over all four chunkers (2.96–2.98 chars/token, not the
2–3 these notes guessed at, and nothing like the 4.79 a hand-written Thai probe
sentence gives — sizing from a probe would have under-counted tokens by ~60%):

- **`doc_maxlen=300`** (the checkpoint's own default) truncates **1.1%
  (recursive) to 7.4% (semantic)** of chunks. 512 would cost 0.0–3.3% for **+3.7%
  storage** — cheap, and deliberately *not* taken.
- **`query_maxlen=32`** truncates **8%** of Gold queries, by at most 5 tokens
  (max 34 and 45 over the two sets). 48 truncates none.
- Storage at 300 across all four chunkers: 30.7M tokens, **7.3 GB** at 128-dim
  fp16 — one chunker at a time fits a 12 GB card without residual compression,
  all four at once does not.

**Run at the checkpoint's own numbers and report both rates as confounds.** Both
point *against* the treatment, so a ColBERT win is not bought by the setting, and
a deviation is an *unmeasured* configuration where this project has repeatedly
found unmeasured to be worse than handicapped. Pre-registered fallback: rerun at
512/48 **only if** ColBERT loses and truncation is a plausible cause — written
down now so it cannot become a post-hoc rescue of a bad result.

### The checkpoint arrives broken, and G1 could not see it

`jina-colbert-v2` loads through `jinaai/xlm-roberta-flash-implementation`, remote
code written for transformers 4.43 and run here under 5.12 — the same path on
which `gte-multilingual-reranker-base` came back position-blind on 2026-08-09
([[feedback_qualify_a_model_before_measuring_with_it]]). **It has the same bug.**
`RotaryEmbedding.inv_freq` is a *non-persistent* buffer, absent from the
safetensors and rebuilt by `__init__`; 5.x materialises it from the meta device
and never re-runs that code, so all 24 layers come up holding uninitialised
memory — `cos = 1`, `sin = 0`, i.e. the rotation is the identity.

Four things worth keeping, in the order they were learned.

1. **It was found by a gate failing, not by reasoning.** The first run rejected
   the real encoder on G2 (a document and its token-reversal scoring 20.7081 vs
   20.7078, |Δ| = 2.8e-04 — the same magnitude as fp16 padding noise). Nothing
   else about the model looked wrong.
2. **The buffer audit passed on the broken model.** G1's float rule was "finite
   and not identically zero", and the garbage was 30 zeros plus 2.6e-29 and
   1.0e-42 — finite, non-zero, arithmetically indistinguishable from zero. G1 now
   also flags a float buffer whose largest magnitude is under 1e-20, but the
   lesson stands: **a smell test cannot decide this.** `inv_freq` is a
   deterministic function of `(dim, base)` written in the checkpoint's own code,
   so there is a right answer to compare against — **C7** does exactly that, per
   layer, and it is the check that decides.
3. **A behavioural check alone is not sufficient either — and this stopped being
   a hypothetical the same day.** The prediction was that uninitialised memory
   which happened to be *large* would give pseudo-random but stable rotations,
   under which the model looks position-sensitive while being just as wrong.
   It was then observed: on a later load layer 0 held `-5.2e+02` and
   `unrepaired` **passed G2** at |Δ| = 4.09e-01. G2 is the backstop, not the gate.
4. **The corruption is nondeterministic across loads, and that decides which weak
   check fires.** An earlier probe found all 24 buffers non-zero and concluded
   "the gte zeroing did not happen"; the contents were garbage anyway, and
   successive loads produced zeros, then 2.6e-29, then 1.6e-30. Four full
   qualification runs on 2026-08-13 make the consequence concrete — layer 0 came
   up `2.6e-29` (G1 passed, G2 caught it), `-5.2e+02` (the mirror image: G2
   passed, G1 caught it), `1.3e-01` (both caught it, G2 by 4.79e-02 against its
   5e-02 threshold, i.e. within 5% of passing) and `-2.7e-23` (both). **C7 fired
   all four times.** So `colbert_model_qualification.md`'s `unrepaired` row is a
   sample of one load, not a property of the bug: re-running reproduces `real`'s
   row exactly (the repair is deterministic) and will not reproduce
   `unrepaired`'s G1/G2 cells. **A one-off probe of a buffer is not evidence
   about the next load** — the check has to run at load time, which is why the
   repair lives in `_load()` and reports how many layers it rebuilt.

`_repair_rotary` recomputes the buffer with the model's own `_compute_inv_freq`
and invalidates the cos/sin cache (which otherwise rebuilds only when the
sequence length grows, so a corrected `inv_freq` alone would be ignored). That is
**restoration, not modification**: no trained information is involved and it is
exactly what `__init__` would have produced. It is also self-retiring — a future
transformers that loads the buffer correctly makes it return 0 with nothing else
changing.

### Gate result: QUALIFIED, in both directions

11 checks × 4 variants. The real encoder passes all 11; three controls built from
the same weights each fail the check written for them:

| control | is | must fail | did |
|---|---|---|---|
| `bag_of_words` | word embeddings only, no attention | G2 | ✓ at **exactly 0.00e+00** |
| `unnormalised` | no L2 step | C3 | ✓ (max ‖v‖−1 = 31.3) |
| `unrepaired` | `repair_rotary=False` — the live bug, not a synthetic sabotage | C7 | ✓ 24 of 24 layers |

**G2 is unusually sharp for late interaction and that is not luck — it is the
mechanism.** MaxSim is permutation-invariant over document tokens, so a
position-blind model must score a document and its token-reversal *identically*,
not merely similarly. The reversal is done on **ids**, not on words: a word-level
reversal retokenizes to a slightly different multiset, so the control would only
have been *approximately* caught and the gate would have rested on a threshold.
Reversing ids holds the multiset exactly — hence the control's 0.00e+00 against
the repaired encoder's 0.767.

**The most useful single number in the report is one of the passes, not the
failures.** `unrepaired` — fully position-blind — scores the hand-written Thai
relevance example **24.4580 vs 12.7192**, a *wider* margin than the working
encoder's 20.7382 vs 17.1936. A broken model did not merely look acceptable here;
on the check a human would have written first, it looked **better**. That is the
whole argument for qualifying a model before measuring with it, restated in this
project's own numbers.

One smaller thing the gate settled, faithful-to-reference rather than convenient:
original ColBERT's marker route (tokenize `". " + text`, overwrite `ids[:,1]`)
assumes `". "` is one token, which is true on WordPiece and **false** on this
SentencePiece vocabulary — C1 pins that the two routes disagree, leaving a stray
`.`. It also recorded a second SentencePiece note, that `mask_punctuation` builds
its skiplist from the *first* token of each ASCII symbol so it "mostly drops
whitespace and keeps `.` — what pylate does, just not what the name suggests."
**The second half of that sentence was wrong, and it was a defect rather than a
quirk. See the next section.**

### Verified against pylate — which found a defect 11 checks could not

`maxsim_reference` reproduces `maxsim`, but it is in-repo. pylate cannot go into
`.venv` (it pins `transformers<=5.3.0` against 5.12.1), so it ran in a throwaway
CPU venv encoding one fixed Thai query and two fixed documents, saved to `.npz`,
against which our encoder was run on the same texts.

**It is now `tools/eval/colbert_pylate_crosscheck.py` →
`data/results/colbert_pylate_crosscheck.md` (7 self-checks, all PASS), and the
throwaway venv is deleted.** The check survives that deletion because the only
thing the other environment produces is the reference tensors themselves —
`--reference OUT.npz` writes one (run it there), the default mode reads
`data/results/colbert_pylate_ref_t{453,530}.npz` and compares, `--render`
re-derives the report from the cached comparison with no model load. It was
promoted out of the scratchpad for the reason the D-family exists: all the
figures below were quoted in three documents while **no artifact on disk
supported them**, and `audit_doc_claims.py`'s D5 flagged `0 of 24` as
untraceable the moment it was written down. Both directions are exercised —
S1 requires the reference to read `0 of 24` (a broken reference is not a
control) and S6 requires the second cell to *differ*, so a pass there would be
a finding, not a relief.

**Finding A — the rotary bug reaches the reference library, and pinning
transformers is what makes an external check possible at all.** pylate on
**transformers 5.3.0** reports `rotary: 24 of 24 layers wrong; layer holds
[2233450102784.0, 1.9323905823039227e-42, 0.0]`; on **4.53.2**, `0 of 24`. So
this is not a bug in our loading — anyone running `pylate` + `jina-colbert-v2` on
transformers 5.x is silently serving a position-blind model — and 4.53.2 is a
reference that is *correct by construction* rather than merely independent.

**A comparison against a second broken model is not a control.** The cell built
to isolate the encoding conventions from the rotary question — `unrepaired` vs
pylate@5.3.0, both position-blind on identical weights — **cannot work**, because
the uninitialised buffer differs from load to load (query `max|Δ|` = 2.7e-01).
Two independently-broken models are not the same model.

**Finding B — the query side matched exactly, and that is five things at once.**
`repaired` vs pylate@4.53.2: query `max|Δ| = 0.000e+00`, min per-token cosine
`1.000000` over (32, 128). Bitwise agreement externally validates the marker
insertion, the augmentation to 32, `attend_to_mask_tokens`, the hand-loaded
projection head, the L2 step **and** `_repair_rotary` — the repaired buffer
reproduces the correctly-loaded one exactly, so it is restoration and not merely
a self-consistent substitute.

**Finding C — the documents did *not* match, and the cause was ours.** Ours
returned **19 and 21** vectors against pylate's **21 and 22**. The two skiplists
turned out to be **disjoint**:

| | rule | drops here |
|---|---|---|
| ours | `encode(sym)[0]` | `▁` (id 6) — **whitespace**: 2 and 3 tokens |
| pylate | `convert_tokens_to_ids(sym)` | `.` (id 5) — **punctuation**: 0 and 2 tokens |

Original ColBERT uses both forms and they coincide on WordPiece; on SentencePiece
encoding a standalone symbol prepends the boundary marker, so `encode(".")[0]`
is `▁` and never `.`. **`mask_punctuation=True` was masking no punctuation at
all.** Fixed to the symbol's own id; `tests/colbert/test_colbert_skiplist.py`
pins the rule in both directions against a stub tokenizer carrying the property
that makes them disagree, so it states the rule rather than recording today's
vocabulary. After the fix all three tensors match: `max|Δ|` 1.2e-04 with min
per-token cosine 0.999936 (fp32, ours batched-and-padded against pylate's
unpadded singles), MaxSim **20.8212 / 17.5484** against **20.8213 / 17.5487**.

The lesson is the one this project keeps re-learning from a different angle:
**the 11-check gate is a battery of *self*-consistency tests, and a convention
that is uniformly wrong on both sides of a comparison is invisible to every one
of them.** The wrong skiplist produced a plausible number of plausible vectors
and ranked the relevance example correctly. It took an implementation that had
never seen our code to see it — and note the direction of the surprise: the
*query* was where an exact match was least expected (markers, augmentation,
mask attention) and it matched bitwise, while the documents, the simpler path,
were where the defect sat.

### The pilot ran, and the prediction FAILED: verdict STOP

`tools/eval/colbert_pilot.py` → `data/results/colbert_pilot.md`. **Re-run
2026-08-20 against rebuild #4** (figures below are that run; the 2026-08-13
originals are in `data/results/_pre_2026_08_18_rebuild4_refresh/`).
`recursive` only, doc300/q32, all 106 Gold queries, unrouted, k=10, ColBERT-alone.
Build 11.8 min (70,250 chunks → 7,364,358 token vectors, `docset_hash
091b7a0ad8a5cfbe`); score run exit 0 with **7/7 self-checks PASS**. The bars are
recomputed **at `recursive`** by `colbert_pilot_baselines.py` rather than taken
from the published cross-chunker aggregates — comparing a one-chunker treatment
against a nine-chunker bar is the wrong-pair trap that killed per-`entity_type`
alpha and rrf4 — and S1/S2 reproduce those aggregates exactly (0.8147 / 0.6034)
from the same code path.

| cell | comparator | n | ColBERT | bar | diff | 95% CI | Holm p | verdict |
|---|---|---:|---:|---:|---:|---|---:|---|
| `person` | BM25 | 30 | 0.8360 | 0.8053 | **+0.0308** | [−0.0429, +0.1030] | 0.3974 | **clears** |
| `program` | dense `qwen3_0.6b` | 30 | 0.2749 | 0.6086 | **−0.3337** | [−0.4433, −0.2278] | 0.0000 | fails |

**The prediction is a conjunction and it half-held.** `person` clears — but as a
*tie*, not a win: the CI spans zero and +0.0308 is inside the bar's own
cross-chunker spread (0.0283), so by the frozen rider it counts at `recursive`
only and never as an axis-level claim. `program` fails by **6.7x** the STOP
margin. Descriptively, across all four types:

| entity_type | n | ColBERT | BM25 | best dense | ceiling |
|---|---:|---:|---:|---:|---:|
| course | 33 | 0.6176 | 0.4280 | 0.5759 | 0.8729 |
| faculty_adjunct_aggregate | 13 | 0.3978 | 0.4477 | 0.4375 | 0.6810 |
| person | 30 | 0.8360 | 0.8053 | 0.4281 | 0.9760 |
| program | 30 | 0.2749 | 0.3278 | 0.6086 | 0.8979 |
| **overall** | 106 | **0.5555** | 0.5088 | 0.5264 | 0.8856 |

**The mechanism is the finding, and it is the axis's own motivation answered in
the negative.** ColBERT sits nearer BM25 than dense on 2 of 4 types — not a
majority, but **those two are exactly the cells the prediction is decided on**,
and the direction is identical on both: strong where the lexical arm is strong
(`person` 0.8360 ≈ BM25 0.8053, against dense's 0.4281), weak where the lexical
arm is weak (`program` 0.2749 ≈ BM25 0.3278, against dense's 0.6086). Late
interaction was proposed here to **cover** the person/program arm split; it
**inherits one side of it** instead. It is not a purely lexical model either —
on `course` it beats both arms (0.6176 vs 0.5759 / 0.4280) — which is why the
per-type table matters more than the verdict line.

**ColBERT carries the highest overall figure in the table (0.5555 vs BM25 0.5088
and dense 0.5264), and that is precisely the reading the conjunctive
pre-registration exists to refuse.** An aggregate win licenses "a stronger
retriever"; it never licenses "late interaction resolves the complementarity".
Had the prediction been written as an aggregate, this run would have been
published as a success.

**The length rider was executed and does not fire.** `DECISION_RULE`'s 512/48
fallback is conditioned on the losing cell's truncation being "materially above"
the corpus rate — and picking what counts as material *after* seeing −0.3337 is
the favourable re-reading a frozen rule exists to prevent. So it is answered as
an arithmetic bound instead (`truncation_rider`, §3b of the report): grant
truncation the most damage it could possibly do — assume a gold resolution with
**any** truncated chunk is destroyed outright and can never be retrieved. Over
`program`'s 221 gold resolutions / 7,659 chunks, **32 are truncated (0.42%,
below the corpus 1.11%)**, touching 14 resolutions (6.3%), and a total loss of
all 14 could explain at most **0.0837** of recall@10 against a **0.3337** gap.
Both readings agree and neither needed a threshold: the bound is 4x short, and
the cell's own truncation rate is *below* the corpus rate, so the rule's literal
wording says no too. **300/32 stands and the truncation stays a stated confound
pointing against the treatment.**

Cost, for the record: 1,578.9 ms query latency p50 (against a 475.6 ms routed
hybrid query), 1.89 GB fp16 for one chunker.

**What this does and does not close.** It closes the pre-registered question at
`recursive` and stops the axis under its own frozen cost rule — the other three
chunkers are not built. It does **not** show late interaction is worthless here:
the `course` cell and the overall figure both point the other way, and nothing
was measured against the shipped hard router, under fusion with BM25, or on a
second checkpoint. Those are new questions with new predictions, not a
continuation of this one — the asymmetry that lets a null close an axis
([[project_hyde_axis]] used the same rule).

### Should it ship? No — and the decisive reason is the failed cell, not the cost

Recorded 2026-08-13 because "the axis is closed" and "do not deploy it" are two
different decisions and only the first one is what `DECISION_RULE` answers. The
rule stops us *spending more GPU on the question*; it says nothing by itself
about putting the artifact in front of users. The recommendation is **do not
adopt**, on four grounds in descending order of weight.

**1. The cell it fails is the one the shipped system depends on.** `program`
loses by −0.3337 at Holm 0.0000, and `program` is precisely where the shipped
router hands the query to a dense specialist because BM25 collapses there
(0.3278). Adopting ColBERT as the retriever would trade away the capability the
system already has, to buy a capability it already has for free from BM25 —
`person` only **ties** (+0.0308, CI spans zero). That is the inherit-not-cover
mechanism above, restated as a deployment consequence.

**2. It was never shown to beat the shipped system, and — say this precisely —
it was never *measured* against it either.** The pilot's bars are BM25 and
best-dense at `recursive`, by pre-registration; **hybrid at the same chunker was
never a bar**, and neither was the router. So the claim is *not* "ColBERT loses
to what we ship". It is that the only two comparisons it won are against two
single arms, and the shipped configuration is neither of them. Indicatively —
**not** like-for-like, since these are different chunker/embedder systems —
unrouted hybrid publishes 0.6229 and the shipped routed hybrid 0.6811 against
ColBERT's 0.5555. For a ship decision that asymmetry is already enough: the
burden is on the candidate to beat the incumbent, and it has not been asked to.

**3. Cost points the same way.** 1,578.9 ms p50 against the shipped routed
hybrid's 475.6 ms (~3.3x), 1.89 GB fp16 for **one** chunker (7.3 GB for all
four, which does not co-reside on a 12 GB card), plus a permanent maintenance
liability: the checkpoint arrives with all 24 rotary layers uninitialised and
`_repair_rotary` has to restore them at load time, keyed to how a given
`transformers` version materialises buffers.

**4. The one genuinely interesting residue is a hypothesis, and its prior is
bad.** ColBERT beats both arms on `course` (0.6176 vs 0.5759 / 0.4280). But that
is a *per-`entity_type` repair*, and per-type repairs have died against the hard
router twice in this project — per-`entity_type` alpha and rrf4 — by the same
mechanism both times: routing already hands each route a specialist that has no
weak spot there. Anyone proposing "route `course` to ColBERT" is proposing a new
pre-registered experiment against arm C, not reading a result off this table.

For completeness, what a reader should be told to do *instead* is nothing
expensive: the only intervention that has beaten the shipped router is the
reranker fine-tuned on hybrid-fused candidates (+0.0654, Holm 0.0000), and its
own free lexical control sits **0.0043 behind it (ns)** — so on current evidence
no GPU-heavy component has earned a place in the serving path.

### Still open

- ~~An I1-variant alignment check for chunk→token-block~~ — **done**: `S4` in
  the pilot runs the L1a–L6 artifact/index alignment check (7/7 PASS).
- ~~`maxsim` against a genuinely external implementation~~ — **done**, see the
  section above; the encoder matches pylate@4.53.2 exactly on queries and to
  1.2e-04 on documents, once a real defect it exposed was fixed.
- ~~`ColbertRetriever` as a registry entry, then a pilot on one chunker with a
  continuation rule fixed before it runs~~ — **done, and the pilot returned
  STOP** (above).
- Nothing is owed. Anything further on this axis needs a *new* pre-registered
  prediction; do not reopen it as a continuation of the one that failed.
- ~~The artifact predates rebuild #4~~ — **done 2026-08-20**: rebuilt and re-scored,
  verdict and ship decision unchanged. Only 3 of the 5 ColBERT scripts read the
  corpus at all (`colbert_length_profile`, `colbert_pilot`, `colbert_pilot_baselines`);
  `colbert_pylate_crosscheck` and `qualify_colbert_model` run on a hand-written query
  and hand-written documents, so a rebuild cannot stale them and they were not re-run.

## Where this belongs in the paper regardless

Even unbuilt, late interaction should appear in related work. An IR reviewer will
ask why a project that tested a cross-encoder reranker did not test ColBERT, and
the answer is a strength rather than an excuse: **we measured the cross-encoder
reranker making hybrid worse** when it replaces the ranking, and — after these
notes were written — that it adds **+0.0017 (ns)** on top of the shipped router
even when fused as a fourth RRF signal, against an oracle **+0.1500** over the
same pool. State it that way rather than as the bare "it hurt": the evidence that
motivates late interaction is that the *evidence is reachable and this model does
not reach it*, which is exactly what a scoring mechanism change is for.
