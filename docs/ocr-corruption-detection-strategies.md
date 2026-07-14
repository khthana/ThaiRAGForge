# OCR non-repetitive-corruption detection: multi-pass strategy recommendations

Status: research/decision-support doc. No code changed. Companion to `docs/llm-ocr-scan-log.md`
(Thai-language experiment journal) and `tools/corpus_prep/llm_ocr_scan.py` (the scanner script).

## 1. Empirical situation so far

Three local Ollama models (`phi4-mini:latest`, `phi4:latest`, `gemma4:e4b`, all zero-shot,
`temperature=0.0`→`0.3` on retry, `num_ctx=8192`, `num_predict=800`, structured JSON output via
`format`) were asked to flag non-repetitive OCR garbling on Thai academic-resolution pages — a
defect class the existing regex tool (`scan_ocr_repetition.py`) cannot see because it only catches
token/phrase loops.

| Run | Set | phi4-mini | phi4 | gemma4:e4b |
|---|---|---|---|---|
| Floor check (known-bad `.bak` files, repetition-shaped) | cumulative | 156/156 flagged | 28/40 (22 call errors) | 15/20 (2 call errors) |
| Real sample (`--sample 30 --seed 1`, regex-clean, 143 pages) | page-level | **143/143 (100%)** | 75/143 (52.4%) | **18/143 (12.6%)** |

Sources: `academic_resolutions/llm_ocr_scan/floor__*.jsonl`, `academic_resolutions/llm_ocr_scan/sample__*.jsonl`,
human review at `academic_resolutions/llm_ocr_scan/gemma_flags_review.txt` and
`academic_resolutions/llm_ocr_review.md`.

Interpretation already established in the handoff log and confirmed by direct jsonl inspection:

- **phi4-mini is a degenerate constant-true classifier** on this task — it flags 100% of pages
  regardless of content, including grammatically clean institutional Thai prose. Its 156/156 floor
  "recall" is retroactively meaningless (a constant-true classifier trivially catches every
  positive in any set it's shown). It carries zero discriminative signal.
- **phi4** flags roughly half of all pages, with a characterized false-positive pattern: English
  citation lists / course codes (e.g. flagging "PREREQUISITE: NONE" as corrupted). It is slow:
  ~4.5 page-chunks/minute observed during the sample run, and had 22 call errors on the floor set
  (`Unterminated string` — malformed/truncated JSON).
- **gemma4:e4b** is the most selective: 18/143 flags, of which manual review judged ~15/18
  plausible genuine corruption and 1-2 likely false positives (one flagged span concerns an
  eccentric-but-apparently-real elective course title — "introductory to zodiac-related
  fortune-telling" — in a curriculum independently confirmed to contain whimsical elective names
  like "Charm School" and "Digital Quotient").
- Two files were flagged by **all three models independently** on the same/adjacent pages, and the
  human-read spans there look like genuine garbled OCR (Thai table cells breaking into word salad,
  mangled English citations/DOIs) — the strongest signal in the dataset so far.

No timestamps exist in `academic_resolutions/llm_ocr_scan/sample_run.log`, so a directly
comparable gemma4:e4b throughput-per-minute figure is **not derivable from the artifacts on disk**
— only phi4's ~4.5 chunks/min is documented. This gap is called out again in §6.

---

## 2. Cascade / tiered filtering

**What it is.** Route cheap/fast classifiers first; only escalate to a slower or more expensive
stage for cases the cheap stage flags (or is unsure about). Two well-established lineages:

- **Cascade classifiers (computer vision).** Viola & Jones, *Rapid Object Detection using a
  Boosted Cascade of Simple Features*, CVPR 2001 (https://web.cs.ucdavis.edu/~yjlee/teaching/ecs289h-fall2014/violajones.pdf).
  A chain of increasingly expensive/accurate stage-classifiers, each trained to reject obvious
  negatives cheaply and pass ambiguous cases forward, so most of the input is discarded by the
  first (cheapest) stage and only a small fraction reaches the last (most expensive) stage.
- **LLM cascades / model routing.** Chen, Zaharia, Zou, *FrugalGPT: How to Use Large Language
  Models While Reducing Cost and Improving Performance*, arXiv:2305.05176
  (https://arxiv.org/abs/2305.05176). Proposes querying a sequence of LLMs of increasing cost,
  stopping as soon as a response is deemed adequate (via a learned scoring function), reserving
  the expensive model only for cases the cheap model can't resolve confidently. Reports matching
  best-single-LLM accuracy at up to 98% lower cost. A broader related line: AutoMix (Madaan et
  al., 2023) does self-verification + confidence-based routing between a small and large model
  (see arXiv:2410.10347 for a recent unified treatment of routing vs. cascading:
  https://arxiv.org/html/2410.10347v3).

**Verdict for this project.** This is exactly the shape of the problem, but inverted from the
literature's usual cost axis: FrugalGPT/AutoMix cascade cheap→expensive because expensive means
*dollars per token*. Here, all three models are free (local/electricity), so "expensive" means
*wall-clock* (phi4 is ~14B and slow; gemma4:e4b and phi4-mini are smaller/faster) and, separately,
*signal quality* — phi4-mini already disqualifies itself (§6 explains why it should not be a
cascade stage at all). The right cascade for this corpus is: **gemma4:e4b as the sole first-pass
filter over the full corpus** (cheapest and highest-precision — inverting the usual cascade order
of "weakest model first" because here the *most* selective model is also comparatively cheap), with
a second-stage confirmation pass — either a stronger LLM judge (§4) or human review — applied only
to the small flagged subset (12.6% of pages in-sample), never to the full corpus. This is a coarse-
to-fine filter in the Viola-Jones sense: reject the easy majority cheaply, spend the expensive
step's compute only on the ambiguous minority.

---

## 3. Self-consistency / repeated sampling

**What it is.** Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou, *Self-Consistency
Improves Chain of Thought Reasoning in Language Models*, ICLR 2023, arXiv:2203.11171
(https://arxiv.org/abs/2203.11171). The actual mechanism: sample *multiple diverse reasoning
paths* (chains of thought) at temperature > 0 for the same question, then marginalize over the
final answers each path arrives at and take the majority. The paper's gains (+17.9% GSM8K, etc.)
come specifically from diversity in the *intermediate reasoning steps* — different chains explore
different problem-solving strategies and often converge on the same correct final answer even when
individual chains err differently.

**Honest verdict on fit.** The scanner's task is a single-shot binary classification
(`flag: bool`) with no intermediate chain-of-thought exposed to the sampler — one JSON object,
produced by one forward pass per call (gemma runs with `think=False` specifically to *suppress*
the intermediate reasoning that self-consistency depends on diversifying). Resampling this task at
temperature > 0 doesn't diversify multiple *reasoning paths* the way the paper's benchmark tasks
do; it just resamples token-level output noise around a single implicit judgment, closer to plain
Monte Carlo repeated-trials averaging than to the mechanism the paper describes and validates.
**This does not transfer cleanly.** It may still have a real, if weaker, use as a variance-
reduction trick (temperature > 0, N=3-5 runs, keep majority flag) to filter one-off sampling noise
from a single model — but call it what it is (repeated-sampling / self-consistency-*inspired*
majority vote on a single-step classifier), not an application of the self-consistency paper's
actual claim. Given gemma4:e4b's already-low 12.6% flag rate and small absolute page counts, the
wall-clock cost of 3-5x resampling every page for a single model is unlikely to be worth it versus
just running the (cheap) model once and sending flagged pages to stage 2 confirmation (§2, §4).

---

## 4. Ensemble / majority voting across distinct models

**What it is / known pitfalls.** Classic ensemble theory (Dietterich, *Ensemble Methods in Machine
Learning*, 2000, summarized in multiple diversity-measure surveys, e.g.
https://www.researchgate.net/publication/220344230) shows that ensemble gains from majority voting
depend on member errors being *independent/uncorrelated*; when errors are correlated (e.g. models
sharing training data, instruction-tuning recipes, or the same systematic blind spot), the
ensemble's effective error-cancellation collapses toward that of a single correlated model, well
short of the naive "N independent draws" binomial-variance-reduction expectation. See also the
"Unified Theory of Diversity in Ensemble Learning" (arXiv:2301.03962) for a modern formal treatment
of the same point.

**Applied directly to this project's numbers.** A 2-of-3 or 3-of-3 vote across
{phi4-mini, phi4, gemma4:e4b} is **not** meaningfully strengthened by phi4-mini's participation:
it flags every page unconditionally, so it contributes exactly zero bits of information to any
vote it's part of — mathematically equivalent to a constant "yes" input to an AND/OR/majority
gate. Concretely: requiring "N models agree" where one member always says yes reduces to
"do the remaining (informative) models agree," i.e. **phi4-mini's presence in a vote is
indistinguishable from its absence**, except it wastes GPU time computing a call whose outcome is
already known. The real question is whether phi4 and gemma4:e4b are independent enough for their
agreement to be strong evidence — and the data partially answers this already: gemma4:e4b's 18
flags and phi4's 75 flags are drawn from very different rates (12.6% vs 52.4%), and the 2 files
flagged by *all three* models (including the uninformative phi4-mini) sit inside gemma4:e4b's more
selective flag set — i.e., phi4-mini adds nothing there either; the real multi-model consensus is
gemma4:e4b ∩ phi4. Because phi4 and gemma4:e4b are different model families/sizes with a
documented divergent false-positive pattern (citation lists vs. eccentric course titles), their
agreement is plausibly closer to independent evidence than same-model repeated-sampling (§3) is —
agreement between two structurally different models is a qualitatively different (and generally
stronger) signal than resampling one model, precisely because the failure modes differ. **Verdict:
drop phi4-mini from any voting scheme entirely (it is dead weight, not a third independent voter);
treat gemma4:e4b∧phi4 agreement as a high-confidence "very likely genuine corruption" signal usable
to auto-triage without human review, and gemma4:e4b-only flags as "needs human/second-pass
confirmation."**

---

## 5. LLM-as-judge / two-stage prompting

**What it is.** Zheng, Chiang, Sheng, et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena*, NeurIPS 2023, arXiv:2306.05685 (https://arxiv.org/abs/2306.05685). Uses a strong LLM to
score/judge another model's output against a rubric, and documents systematic judge biases:
position bias, verbosity bias, self-enhancement bias (a model favoring outputs similar to its own
style), and limited reasoning ability on tasks requiring careful step-by-step judgment. The paper
finds GPT-4-as-judge reaches >80% agreement with human preference — comparable to human-human
agreement — but only after accounting for these biases (e.g. randomizing answer order to fight
position bias).

**Verdict for this project.** Splitting "detect a suspicious span" (recall-oriented, can be noisy)
from "judge whether that span is truly incoherent" (precision-oriented, given the span in
isolation plus surrounding context) is a reasonable two-stage design and matches the judge-pattern
literature's spirit: use a narrower, more constrained second prompt rather than trusting one
model's single combined flag+reason+span output. Concretely: stage 1 = gemma4:e4b (or gemma4:e4b∧
phi4 per §4) proposes a span; stage 2 = a fresh prompt, ideally the *largest* model comfortably
running on the RTX 3060 (phi4 14B is already the largest currently pulled, or a purpose-fetched
~7-8B instruction model if one's added later), given only the flagged span + a couple sentences of
context, asked a narrower yes/no ("is this specific span incoherent Thai/garbled OCR, yes or no")
rather than open detection. This reduces the judge's task complexity relative to phi4's original
single-shot job (find *and* explain *and* span-extract), which should reduce the false-positive
rate the citation-list/course-code pattern demonstrates. Known bias to watch for: self-enhancement
bias, if the same model family is used for both detection and judging (not a concern here since
gemma and phi are different families) — but verbosity/position bias is not very applicable since
this is a binary judgment over a fixed span rather than a pairwise comparison of free-form
answers, so the paper's mitigations (randomize order, etc.) mostly don't apply; the main
transferable lesson is "narrow the task per stage," not the anti-bias mechanics.

---

## 6. Few-shot calibration

**What it is.** Brown et al., *Language Models are Few-Shot Learners* (GPT-3), NeurIPS 2020,
arXiv:2005.14165 (https://arxiv.org/abs/2005.14165). Demonstrates that placing a handful of
labeled input→output examples directly in the prompt (no gradient update) substantially improves
task performance over zero-shot, especially for tasks requiring the model to infer an implicit
decision boundary from examples rather than from instructions alone — precisely this project's
situation (the instruction "flag incoherent prose, not repetition, not table oddities" is already
in the zero-shot prompt, but the model still misfires on categories the instructions don't
enumerate, like course codes). Complementary primary-source guidance: current prompting docs from
model providers (e.g. Anthropic's and OpenAI's public prompt-engineering guides) consistently
recommend a small number (roughly 2-5) of concrete positive *and* negative examples for
classification-style tasks, particularly to pin down edge cases textual instructions alone tend to
under-specify — exactly the "PREREQUISITE: NONE" / eccentric-course-title edge cases already
observed here.

**Verdict for this project, with caveats specific to small local models.** This corpus already has
exactly the right raw material for few-shot calibration without inventing anything: 2-3 confirmed
positive examples (the multi-model-consensus flags — genuine garbled Thai table cells / mangled
DOIs) and 2 confirmed near-miss negatives ("PREREQUISITE: NONE"-style citation/course-code text;
the zodiac-elective-course span) drawn straight from `gemma_flags_review.txt` and
`llm_ocr_review.md`. Caveats:
- **Context budget.** `num_ctx=8192` is already being spent on page text (up to 6000 chars) plus
  the instruction prompt; each few-shot example (page excerpt + label + reasoning) consumes
  context that competes with the page under evaluation. Keep examples short — trimmed excerpts
  (a paragraph, not a full page) around the specific flagged/unflagged span, not entire pages.
- **Small-model pattern-matching risk.** Sub-10B local models are more prone to overfitting to
  surface features of the few-shot examples (e.g., "if it mentions an English course code, don't
  flag" as a literal rule) rather than generalizing the underlying "coherent vs. garbled" judgment
  — a known small-model few-shot brittleness, consistent with why phi4-mini is already a
  degenerate classifier without any few-shot examples at all. Recommend testing few-shot gemma4:e4b
  on a held-out slice of the existing 30-file sample (not the same pages the few-shot examples were
  drawn from) before trusting it on new corpus files, to confirm it improves rather than just
  shifts the false-positive/negative pattern.
- Given gemma4:e4b is already the highest-precision model zero-shot, few-shot calibration is best
  spent tightening *its* boundary further (fewer of the ~1-2 residual false positives), rather than
  attempting to rescue phi4 (whose issue — a broad, non-degenerate but noisy false-positive class —
  might respond better to few-shot than phi4-mini's issue, which is total signal collapse that
  few-shot is unlikely to fix; see §6 recommendation to not invest further in phi4 first).

---

## 7. Practical sequencing recommendation

**Immediate verdicts:**
- **Retire phi4-mini from this pipeline entirely.** It is empirically a constant-true classifier
  on this task (143/143 flags across all page content types, including clean institutional prose).
  It contributes no information to any downstream vote (§4) and its floor-check "recall" is not
  evidence of real capability. No further tuning (few-shot, resampling) is likely to fix a
  100%-positive-rate model cheaply — that would require materially changing its precision profile,
  which is a bigger intervention than this project needs when gemma4:e4b already works.
- **gemma4:e4b runs the full-corpus first pass alone.** It has the best precision observed
  (~15/18 ≈ 83% plausible-true-positive rate on manual review) and, being a smaller model than
  phi4 (14B), is presumed faster, though this is not directly measurable from the current
  artifacts — `sample_run.log` has no timestamps, so **there is no logged gemma4:e4b
  chunks/minute figure to compare against phi4's observed ~4.5/min.** Recommend capturing wall-
  clock timing (e.g. wrapping the per-file loop with `time.monotonic()` deltas written to the jsonl
  metadata) on the next run so this gap can be closed cheaply, rather than assuming a specific
  speedup.
- **Do not decide phi4's fate by hand-reviewing its full 52.4% flag set.** That set is already
  characterized as noisy (English citation/course-code false-positive pattern) and reviewing
  thousands of files' worth of ~50%-positive-rate output by hand does not scale and isn't needed to
  answer the real question, which is narrower: *does phi4 catch anything gemma4:e4b misses, net of
  its known FP pattern?* A cheap, statistically defensible test:
  1. Take the **existing 143-page sample** (already scored by both models — no new LLM calls
     needed). Compute the set difference: pages phi4 flagged that gemma4:e4b did not
     (`phi4_only`, expected roughly 75 − (overlap with gemma's 18) pages).
  2. Human-review only `phi4_only` (not all 75 phi4 flags) — a bounded, one-sitting task — and
     tag each as genuine-corruption-gemma-missed vs. false-positive-matching-the-known-pattern
     (citation lists, course codes, or other patterns discovered during review).
  3. If the genuine-catch rate in `phi4_only` is near zero (i.e. essentially all of it is the
     already-known citation/course-code FP pattern), phi4 adds no marginal recall worth its ~14B/
     slow wall-clock cost — retire it from the full-corpus pipeline, keep it only as an optional
     stage-2 confirmer (§5) if a second, structurally-different judge model is later wanted for
     high-value auto-triage.
  4. If `phi4_only` contains a non-trivial number of genuine catches gemma4:e4b missed, that
     defines a *specific* residual defect class gemma4:e4b is blind to — worth characterizing
     (what makes those pages different?) before deciding whether to run phi4 as a full second pass
     or fix gemma4:e4b's prompt/few-shot examples (§6) to close that specific gap instead of paying
     phi4's wall-clock cost on the whole corpus.
  This reuses data already on disk (`sample__phi4_latest.jsonl`, `sample__gemma4_e4b.jsonl`) — zero
  new model calls required to run steps 1-2.

**Recommended pipeline (ordered, handoff-ready):**

1. **Stage 0 (already exists, zero-cost re-check).** Regex repetition scan
   (`scan_ocr_repetition.py`) still runs first on the full corpus as today — it's cheap and catches
   a disjoint defect class.
2. **Stage 1 — full-corpus pass, gemma4:e4b only**, zero-shot prompt as currently written (or
   lightly few-shot-calibrated per §6 once validated on held-out sample pages), `think=False`,
   `temperature=0.0`, one call per page-chunk, resumable jsonl per the existing script design.
   This is the coarse filter (§2): expect roughly the observed ~12.6% flag rate to carry over
   corpus-wide, meaning ~87% of pages are cheaply dismissed after one small-model pass.
3. **Stage 2 — confirmation only on Stage-1 flags** (not the full corpus): a narrower LLM-as-judge
   prompt (§5) — ideally on a structurally different model (phi4, once its `Unterminated string`
   truncation issue is fixed by raising `num_predict`, or a fresh model if one is added later per
   the queued-embedder-models memory) asking only "is this specific flagged span incoherent,
   yes/no" with the span + minimal context, not full-page re-detection. Where gemma4:e4b and the
   Stage-2 judge agree, auto-triage as high-confidence corruption (§4's independent-evidence
   argument); where they disagree, route to human review.
4. **Stage 3 — human spot-check**, bounded to Stage-2 disagreements plus a fixed-size random sample
   of Stage-1 non-flags (to periodically re-estimate gemma4:e4b's false-negative rate, which the
   current sample can't measure — there's no ground truth for pages gemma4:e4b *didn't* flag).
5. **Before scaling to the full corpus**, run the phi4-vs-gemma4:e4b diff test above (bounded,
   reuses existing jsonl, no new calls) to settle whether Stage 2 should default to phi4, a
   different model, or be skipped in favor of routing straight to human review for low-volume
   flag sets.
6. **Fix the truncation bug independent of the above.** phi4's 22 `Unterminated string` errors on
   the floor set are consistent with `num_predict=800` being too tight for phi4's more verbose
   JSON (span + reason) at busy pages; Ollama's structured-outputs docs
   (https://docs.ollama.com/capabilities/structured-outputs) constrain the *grammar* of the output
   to match the schema but do not document any special handling for output truncated mid-generation
   by hitting `num_predict` — the practical implication (confirmed by the observed error mode) is
   that a schema-constrained generation cut off early by the token budget simply produces
   incomplete/invalid JSON like any other truncated generation; the schema constrains what can be
   generated at each step, not that generation completes in budget. If phi4 is retained for Stage 2
   confirmation, raise `num_predict` for phi4 specifically (e.g. 1200-1500) given its longer
   average output, and keep the existing retry-at-temperature-0.3 behavior for the remaining
   failures.

**Wall-clock reasoning.** With phi4 measured at ~4.5 page-chunks/minute and gemma4:e4b's rate
unknown (see gap noted above), the only safe capacity planning statement without inventing a number
is: routing the **full corpus** through gemma4:e4b once (Stage 1) and reserving phi4 or another
judge for only the ~12-15% Stage-1 flags (Stage 2) should already be dramatically cheaper in
wall-clock than running phi4 across the whole corpus, purely from the ~85-88% volume reduction
before phi4 is ever invoked — independent of gemma4:e4b's absolute throughput. Recommend capturing
per-model chunks/minute on the very next run (simple timing instrumentation, no design change) so
future capacity estimates for a corpus of "thousands of documents × several pages each" don't rely
on an assumed speedup.

---

## 8. Addendum (2026-07-12): the cascade recommendation above was wrong — file-level diff overturns it

The §7 sequencing above was written from **page-level** flag rates alone (gemma 12.6%, phi4
52.4%). A same-day follow-up computed the **file-level** rollup on the identical 143-page/30-file
sample (zero new model calls — pure re-aggregation of `sample__phi4_latest.jsonl` and
`sample__gemma4_e4b.jsonl`), and it changes the conclusion:

| | files flagged (≥1 page) |
|---|---|
| gemma4:e4b | 9/30 |
| phi4 | 22/30 |
| both | 9/30 |
| **phi4 flags, gemma flags zero pages (gemma file-blind)** | **13/30 (43%)** |
| gemma flags, phi4 flags zero pages | 0/30 |

Manual read of the strongest phi4 span in each of the 13 gemma-blind files: **roughly 9-10 of the
13 contain a plausible genuine defect** phi4 caught and gemma missed entirely (not just under a
different page) — some are minor single-character drops in English course titles ("DATATRUCTURE",
"MUSCIANSHIP", "Digital Quotence"), one is a severe long digit-repetition string phi4 flagged that
`scan_ocr_repetition.py` also missed (see below). Only 2-3 of the 13 files' phi4 flags are
exclusively the known citation-list/course-code false-positive pattern with nothing genuine
underneath.

**This overturns §7's "gemma-first, escalate only its flags" cascade.** Page-level precision (what
§7 optimized for) is the wrong metric for a stage-1 filter; **stage-1 recall** is the binding
constraint, because anything stage 1 drops never reaches stage 2 or a human. gemma's file-level
recall (missing 13/30 = 43% of files with any phi4 signal, ~9-10/30 = 30%+ with a *genuine* one) is
not high enough to serve as a sole full-corpus first pass for a discovery/audit tool. **The
corrected architecture is a union — both models run across the full corpus independently — not a
cascade.** This also means the cascade's cost rationale (§2 — escalate the slow model only for the
cheap model's ~12-15% flags) does not hold: if phi4 must see gemma's misses too, phi4 has to run on
everything, at which point there is nothing left for it to be a "confirmation stage" over.

**Real throughput** (measured via NTFS `CreationTime`/`LastWriteTime` on the three
`sample__*.jsonl` files — the run processes one model fully before starting the next, so the gap
between one file's creation and the next file's creation is that model's phase duration; zero new
GPU calls needed): **gemma4:e4b ≈ 53.6 pages/min** (143 pages / 160s), **phi4 ≈ 14.1 pages/min**
this run (143 pages / 609s) — three times faster than the ~4.5 pages/min observed earlier during
the floor check under different concurrent-load conditions. The discrepancy between the two phi4
measurements is unresolved; treat 4.5-14.1/min as the plausible range until re-measured cleanly.

**Corpus scale** (random 200-file sample of the 2,856 non-backup `.md` files under
`academic_resolutions/`): ~3.5 page-chunks/file average → **~10,000 page-chunks corpus-wide**.
Projected full-corpus wall-clock, sequential on one GPU (both models can't run concurrently without
contention):
- gemma4:e4b alone: ~186 min ≈ **3.1 hours**
- phi4 alone: ~709-2,222 min ≈ **11.8-37 hours** (range reflects the unresolved throughput
  discrepancy above)
- **Union (both, full corpus, the now-corrected architecture): ~15-40 hours total**
- Cascade as originally specified in §7 (gemma full pass + phi4 only on gemma's flags): ~4.6-7.8
  hours — cheaper, but now known to silently miss ~30-43% of files with real defects, so this
  number is not a valid comparison point for a completeness-oriented tool.

**Independent fixes applied to `llm_ocr_scan.py` while investigating (do not block on the
architecture decision above):**
- Retired `phi4-mini:latest` from the default `MODELS` list — confirmed constant-true classifier
  (143/143 page flags in-sample), contributes zero information to any ensemble vote (§4).
- Raised `num_predict` to 1500 for `phi4:latest` specifically (was 800 for all models, causing
  ~6% `Unterminated string` truncation errors on the floor set); other models keep 800.
- Added per-call `elapsed_s` timing to every jsonl record, so the next run produces real
  chunks/minute data instead of requiring the timestamp-inference trick used above.

**Side-finding, out of scope for this experiment:** `scan_ocr_repetition.py` splits on whitespace
(`\S+` tokens) and flags a *token* repeated ≥8 times in a row — it cannot see character-level
repetition glued into a single whitespace-free token (e.g. the same Thai digit repeated 20+ times
with no separators, found in `2568\ครั้งที่ 10\...วิทยาลัยวิศวกรรมสังคีต.md`, page 2). Confirmed by
reading the scanner's token-matching logic, not by running it. This is a third defect shape
(intra-token repetition), distinct from both the inter-token repetition the regex scanner targets
and the non-repetitive corruption this whole document is about. Not fixed here — flagged for a
future, separate pass.

**Status: the full-corpus union run (~15-40 GPU-hours on the user's own machine) has not been
launched.** That is a decision for the user, made with the real numbers above, not something to
start silently on the strength of "cost is just electricity."
