# RQ4 prompt truncation — measured 2026-08-10

**สรุปสั้น (ไทย):** `rq4_generate.py` ตั้ง `num_ctx=8192` มาตลอด แต่ prompt ที่ยาวที่สุด
ยาว **14,721 tokens** — ollama ไม่ได้เตือนอะไรเลย มันแค่ **ตัดหัว prompt ทิ้ง** เหลือ
`num_ctx/2` tokens แล้วเก็บ *ท้าย* ไว้ ซึ่งคือกรณีที่แย่ที่สุดสำหรับเรา เพราะ `build_prompt`
วางเอกสารอันดับดีที่สุดไว้ **หน้าสุด** และวางกติกาไว้ **ท้ายสุด** → สิ่งที่ถูกลบคือหลักฐาน
ที่ดีที่สุด ส่วนกติกาอยู่ครบเสมอ คำตอบจึงยังออกมา "ดูดี" และไม่มีอาการอะไรให้เห็น
กระทบคำตอบที่เผยแพร่แล้ว **81 จาก 1,590 เซลล์** (แก้จาก 80 เมื่อ 2026-08-10 ดู §4) ตอนนี้ซ่อมที่ต้นเหตุแล้ว (ยกเป็น 16384
+ pre-flight ที่ *วัด* ไม่ใช่ *สมมติ* + ตัวจับลายเซ็นรายคำตอบ + บันทึก `prompt_eval_count`
ลงทุกไฟล์คำตอบ) และ **สร้างคำตอบทั้ง 81 เซลล์ใหม่แล้วเมื่อ 2026-08-10** ที่
`num_ctx=16384` — ผลคือ verdict พลิก 1 จาก 57 ในรายงานหลัก (`hybrid > dense` ด้าน
citation recall ภายใต้ `cite_all` กลายเป็น **ไม่นัยสำคัญ** ซึ่งคือทิศทางที่ทำนายไว้) และ
พลิก 3 จาก 57 ในรายงาน guarded (ดีขึ้นทั้งสาม) ส่วน **headline ของ prompt ablation ไม่ขยับ**
— รายละเอียดทั้งหมดอยู่ที่ §4b

## 1. What was found

`tools/eval/rq4_generate.py` shipped with `--num-ctx` defaulting to **8192**, and its
own help string already asserted that the value *"MUST exceed the longest prompt"*.
Nothing checked it. The longest RQ4 prompt is **14,721 tokens**.

This was found by the pre-run verification the user's standing instruction requires
("ห้ามรันโดยไม่ตรวจสอบก่อนเด็ดขาด") before adding the two entity arms — not by any
symptom in the outputs, and not by any existing audit.

## 2. The rule, measured rather than read from the docs

ollama 0.32.6 / `phi4`. One 14,721-token prompt, three context windows:

| `num_ctx` | reported `prompt_eval_count` | interpretation |
|---|---|---|
| 4,096 | 2,050 | truncated to `num_ctx//2 + 2` |
| 8,192 | 4,098 | truncated to `num_ctx//2 + 2` |
| 16,384 | 14,721 | fed whole |

An earlier hypothesis of mine — *"ollama never feeds more than `num_ctx/2`"* — was
**refuted by its own control**: prompts of 5,651 / 6,885 / 7,508 tokens are fed **whole**
at `num_ctx=8192`. The rule that reproduces all six points is:

> **fits `num_ctx` → fed whole; exceeds `num_ctx` → cut to `num_ctx//2 + 2` tokens,
> keeping the tail.**

So the threshold is **8,192 tokens, not 4,098**, and `prompt_eval_count == num_ctx//2 + 2`
is an exact, detectable **truncation signature**.

## 3. Why front-truncation is the worst possible cut here

`build_prompt` lays the retrieved documents out `[1]`…`[k]` **best-ranked first** and
puts the rules **last**. That layout is deliberate — it is the 2026-08-03 fix for
[[feedback_llm_prompt_truncates_from_front]], which put the instructions after the
context precisely so a truncation could not delete them.

The consequence, unnoticed until now: that fix made truncation **invisible instead of
harmless**. When a prompt overflows,

- the rules always survive → the answer is well-formed, cites in the right format,
  abstains when told to;
- the **highest-ranked documents** are the ones deleted → the answer is evidence-poor
  for reasons nothing in the output reveals.

`tests/tools/test_rq4_prompt_truncation.py` pins both the signature arithmetic and the
document-before-rules layout, so a future layout flip fails loudly rather than quietly
making this explanation wrong.

## 4. Blast radius on the published table

Exact token counts for every (variant, arm, query) prompt, measured with ollama itself
at `num_ctx=16384, num_predict=1`. Screening used the corpus's **minimum** observed
chars/token, not the mean — see §5.

> **Corrected 2026-08-10 while building the regeneration worklist: the published arms hold
> 81 truncated cells, not 80.** `tools/eval/rq4_find_truncated_answers.py` re-derived the
> list (it needs the *cells*, not the count) and found one more,
> `cite_all_guarded / dense_qwen3_0.6b_semantic / q001` — 8,258 tokens, fed 4,098 at
> `num_ctx=8192`. The dense row below is corrected 16 → 17. The cause is §5's: this table's
> screen divided characters by 1.046, which the data violates, so a cell 66 tokens over the
> line was never measured. All three arms' 8,192–8,300-token band was affected; the other
> two boundary cells (`sentence_cap/dense/q009` 8,212 and `cite_all/hybrid_m2v/q025` 8,269)
> this table *did* catch, and a first re-derivation using the same 1.046 screen missed both,
> which is what exposed the constant. **The entity-arm rows were screened the same way and
> are likely undercounts too**; they were never generated at 8192 (that run used 16384), so
> nothing on disk depends on them and they are left as measured.
>
> The re-derivation is stronger evidence than this table, not merely different: each of the
> 81 is confirmed twice — the exact signature `prompt_eval_count == 4,098` *at
> `num_ctx=8192`*, i.e. the old run reproduced rather than proxied, plus a whole-prompt
> measurement at 16384. Per-cell evidence: `data/results/rq4_truncated_cells.{json,md}`,
> probe cache `rq4_truncated_cells_raw.json`, log
> `data/logs/rq4_find_truncated_2026_08_10_screen095.log`.

| variant | arm | prompts | > 8,192 tok (truncated) | worst | % kept, worst |
|---|---|---|---|---|---|
| sentence_cap | hybrid_qwen3_0.6b_semantic | 106 | 0 (0%) | 7,702 | 100% |
| sentence_cap | dense_qwen3_0.6b_semantic | 106 | 14 (13%) | 13,433 | 31% |
| sentence_cap | bm25_semantic | 106 | 5 (5%) | 11,396 | 36% |
| sentence_cap | hybrid_m2v_semantic | 106 | 6 (6%) | 9,901 | 41% |
| sentence_cap | closed_book | 106 | 0 (0%) | 0 | 100% |
| sentence_cap | entity_lookup_semantic | 106 | 48 (45%) | 13,842 | 30% |
| sentence_cap | entity_boost_semantic | 106 | 47 (44%) | 14,721 | 28% |
| cite_all | hybrid_qwen3_0.6b_semantic | 106 | 0 (0%) | 7,793 | 100% |
| cite_all | dense_qwen3_0.6b_semantic | 106 | 15 (14%) | 13,433 | 31% |
| cite_all | bm25_semantic | 106 | 5 (5%) | 11,396 | 36% |
| cite_all | hybrid_m2v_semantic | 106 | 7 (7%) | 9,901 | 41% |
| cite_all | closed_book | 106 | 0 (0%) | 0 | 100% |
| cite_all | entity_lookup_semantic | 106 | 49 (46%) | 13,842 | 30% |
| cite_all | entity_boost_semantic | 106 | 49 (46%) | 14,721 | 28% |
| cite_all_guarded | hybrid_qwen3_0.6b_semantic | 106 | 0 (0%) | 7,999 | 100% |
| cite_all_guarded | dense_qwen3_0.6b_semantic | 106 | **17** (16%) | 13,433 | 31% |
| cite_all_guarded | bm25_semantic | 106 | 5 (5%) | 11,396 | 36% |
| cite_all_guarded | hybrid_m2v_semantic | 106 | 7 (7%) | 9,901 | 41% |
| cite_all_guarded | closed_book | 106 | 0 (0%) | 0 | 100% |
| cite_all_guarded | entity_lookup_semantic | 106 | 53 (50%) | 13,842 | 30% |
| cite_all_guarded | entity_boost_semantic | 106 | 51 (48%) | 14,721 | 28% |

**Published answers affected: 81 of 1,590 (query, arm, variant) cells (5.1%)** (was 80
before the correction above), over **29 (arm, query) pairs / 25 distinct queries**, in
**3 of the 5 published arms**.

Read the shape of that table before reading the headline, because the shape carries the
argument:

- **`hybrid`, the headline arm, is untouched at 0/106 in all three variants** — its worst
  prompt is 7,999 tokens, 193 short of the line. Nothing in the design put it there; it is
  luck, and `cite_all_guarded` came within 2.4% of losing it.
- **`closed_book` is 0/106 by construction** (no documents), which is what makes "the
  damage tracks context length" a measurement and not an assumption.
- **The `hybrid` vs `dense` vs `bm25` ordering is the one claim most at risk**: dense is
  the most-truncated published arm (17/106) and hybrid the least (0/106), so truncation
  pushed in exactly the direction the published ordering reports. The published finding
  `hybrid > {dense, bm25}` is therefore **not confirmed and not refuted by this** — it is
  measured under a confound that flatters it, and 5.1% of cells is small but the confound
  is not random with respect to the comparison.
- **The two entity arms — the ones #1 exists to measure — would have been ~45-50%
  truncated at 8192.** That run has not happened, which is the whole point of catching
  this before it did.

## 4b. Regenerated 2026-08-10 — what actually moved

All **81** cells were regenerated at `num_ctx=16,384` by
`tools/eval/rq4_regenerate_truncated.py` (3 variants × 3 arms, 9,501 s, exit 0, post-check:
every regenerated answer carries `num_ctx=16384` and none matches the truncation
signature). The originals are **moved, not deleted**, to
`data/rq4/_truncated_backup_2026_08_10/` with a manifest — they are the only surviving
copies of what an evidence-stripped answer looked like.

The method is the same paired regeneration the 2026-08-07 rebuild-#3 refresh used:
`rq4_generate.py` skips an answer file that already exists, so moving exactly the 81 bad
files regenerates exactly those and freezes all 1,509 others byte-for-byte. That matters
because temperature 0 is **not** reproducible here
([[feedback_temperature_zero_is_not_reproducible]], 14/24 identical citation sets under
`cite_all`) — any answer re-rolled without cause would add noise a paired test cannot tell
from signal. Both reports were then re-scored and **verdict-diffed** against a pre-run
baseline (`data/results/_rq4_baseline_2026_08_10/`) with
`tools/eval/diff_significance_reports.py`, because the same noise floor makes eyeballing
useless.

**The internal control passes, and it is the first thing to read.** `hybrid` had 0
truncated cells and `closed_book` 0 by construction; every single one of their numbers is
**byte-identical** across the re-score, in all three variants. Only the three truncated
arms moved. So the movement below is the repair, not drift.

**Every truncated arm improved, in the direction predicted.** Under `sentence_cap`, dense
precision 0.6413 → **0.6549** and recall 0.2201 → **0.2261**, bm25 0.6463 → **0.6607** /
0.2203 → **0.2265**; under `cite_all`, dense 0.6629 → **0.6798** / 0.3206 → **0.3356**,
bm25 0.5968 → **0.6104** / 0.2938 → **0.2961**; under `cite_all_guarded`, dense 0.6530 →
**0.6746** / 0.3323 → **0.3460**, bm25 0.6177 → **0.6217** / 0.2743 → **0.2798**. `m2v` is
the exception and moved *down* on precision (`cite_all_guarded` 0.5138 → **0.4817**),
which is consistent with it being the arm whose restored context still holds little gold.

**Verdict flips: 1 of 57 in `rq4_score.md`, 3 of 57 in `rq4_score_guarded.md`.**

1. **The one loss is exactly the claim this section flagged as most at risk.** Under
   `cite_all`, `hybrid[recall] vs dense[recall]` goes **yes → no**: −0.0756 (Holm 0.0132)
   becomes **−0.0606, CI [−0.1115, −0.0098], Holm 0.0760**. Dense was the most-truncated
   arm (17/106) and hybrid the least (0/106); giving dense its evidence back narrowed the
   gap until it no longer clears Holm at m=12. **This is the confound discharging in the
   predicted direction** — the ordering still holds numerically
   (hybrid 0.3962 > dense 0.3356 > bm25 0.2961 > m2v 0.2038) but `hybrid > dense` on
   citation recall is now a **bound**, not a result: it rules out dense beating hybrid by
   more than 0.0098 and hybrid beating dense by more than 0.1115.
2. **The three gains are all `no → yes` under the guard**, all against `m2v`
   (`bm25 vs m2v` precision and recall, `dense vs m2v` precision), because the two repaired
   arms rose while m2v fell. Family 1b separation under `cite_all_guarded` therefore goes
   **3/12 → 6/12**, which *reverses* the documented reading that the guard compresses the
   spread — see `docs/rq4-design.md`.

Family counts, before → after: `sentence_cap` (1a) **2/12 → 2/12**, `cite_all` (1b)
**9/12 → 8/12**, `cite_all_guarded` (1b) **3/12 → 6/12**.

**The prompt-ablation headline — the deliverable — is untouched.** Family 2 under
`cite_all`: hybrid **+0.1181**, dense +0.1005 → **+0.1095**, bm25 +0.0734 → **+0.0696**,
all Holm 0.0000; m2v ns (+0.0217). Under `cite_all_guarded`: dense +0.1123 → **+0.1198**
(0.0000), hybrid **+0.0706** (Holm 0.0144 → **0.0192**, still significant in family 2 and
still ns in family 3's m=24 — the family-size lesson survives intact), bm25 and m2v ns.
And **no new fabrication**: phantom counts are unchanged in every cell (dense `cite_all`
still 4, on a denominator that grew 359 → 391; `sentence_cap` is still **0 of 981** total
citations, up from 954).

One further movement worth recording because it is quoted as the guard's cost: the m2v
abstention 2×2 under the guard changes from (correct-abstain 19, hallucinated 10) to
(**16**, **13**), so the guard's effect on m2v is now correct-abstain 13 → 16 and
hallucination 16 → 13, milder than the previously published 13 → 19 / 16 → 10.

## 5. Method note: chars/token is not usable as an estimator here

The obvious screen — convert a character budget to tokens with an average ratio — fails on
this corpus. Observed chars/token across the *same* prompt family spans **1.046** (Thai
prose) to **3.151** (English course-code comparison tables). One 15,915-character
`entity_boost` prompt is only 5,051 tokens.

So: screen with a bound that can only over-select, then measure everything above the line
**exactly** with ollama's own `prompt_eval_count`.

**And an observed minimum is not such a bound — corrected 2026-08-10.** The 1.046 above was
measured on the two entity arms and then used as if it held for every prompt. It does not:
over the 228 prompts screened by `tools/eval/rq4_find_truncated_answers.py`, **15 fall
below it**, the minimum being **1.0098** (`bm25_semantic/q001`, 11,208 chars / 11,099
tokens). The published range is wrong at both ends — the realized spread is
**1.0098 – 4.0175**.

The consequence is worse than a mis-sorted probe queue: an unsound "upper bound" removes
prompts from the candidate list *entirely*, so they are never measured. That is how two
reconstruction runs of §4 missed cells at the 8,192-token boundary. The screen is now the
only bound that is provable rather than sampled — **one token per UTF-8 byte**, since a
byte-level BPE token consumes at least one byte. It is ~3x loose on Thai, and loose is the
one direction a safety screen may err in.

**The first `preflight()` got this wrong, and the entity arms are what exposed it
(2026-08-10).** It measured the single longest prompt *in characters* and cleared the whole
run on that one reading. But the longest-by-characters prompt need not be the longest in
tokens, and on these arms it is not close: `entity_boost/q090` is 15,689 chars / **4,860
tokens**, while the arm's true worst is **14,721 tokens**. At `num_ctx=8192` the old screen
would have measured 4,860, declared "fits", and started a run in which ~45-50% of prompts
were truncated -- the exact failure the pre-flight exists to prevent, passed by its own
check. It now screens on the **upper bound** (UTF-8 byte length, see above): every prompt
whose bound fits is provably safe and is never probed, and only prompts that *could* exceed
`num_ctx` get a forward pass, largest first. When nothing can exceed it the run clears
without touching the GPU. Note that ordering *by* an upper bound sorts by its own slack
rather than by truth, and only the first `max_probes` candidates are probed — so preflight
is a cheap early exit, and the sound guard remains the per-answer `prompt_eval_count` check
in `generate()`, which sees every prompt as it is actually sent.
`tests/tools/test_rq4_prompt_truncation.py` pins the bound in both directions. 742 (arm, query) pairs
reduce to 199 candidates that way, and one measurement per pair suffices because the three
variants differ only in the rules block — borderline pairs get all three measured.

## 6. The repair

Four changes in `tools/eval/rq4_generate.py`, in order of how much they are worth:

1. **`preflight()`** — before generating anything, build every prompt, take the longest,
   send it once with `num_predict=1`, and **refuse to start** (`SystemExit`) if the
   returned `prompt_eval_count` matches the truncation signature. One extra call per run.
   This is the change that converts the help string's assertion into a measurement.
2. **A per-answer guard** — every response's `prompt_eval_count` is compared against
   `truncated_to(num_ctx)`; a match prints `[truncated] arm/qNNN` naming the prompt, and
   the run **exits non-zero** at the end with the count. Belt and braces: the pre-flight
   checks the longest prompt as built, the guard checks each prompt as actually fed.
3. **`num_ctx` and `prompt_eval_count` are recorded in every answer JSON.** The 8192-era
   answers carry no such field, which is exactly why their damage had to be reconstructed
   prompt by prompt instead of read off. Any future audit gets it for free.
4. **Default `--num-ctx` raised 8192 → 16384**, with the measured longest prompt (14,721)
   named in the help string so the next person can see the margin.

`--skip-preflight` exists and is documented as "do not use"; it is there so the guard can
be tested, not so a run can dodge it.

## 7. The reusable lesson

An invariant that is **asserted in a comment or a help string, and checked nowhere**, is
not an invariant — it is a wish. This one had been written down, in the right place, in
capital letters (`MUST exceed the longest prompt`), for months, while being false in
production the entire time.

Two aggravating features worth recognising elsewhere in this project:

- **The failure is silent by construction.** A truncated prompt returns a fluent,
  correctly-formatted, correctly-citing answer. There is no exception, no warning, no
  malformed output — the same shape as this project's other silent-corruption bugs
  (stale caches, `resolution_id` collisions, corpus-discovery contamination):
  *two artifacts produced at different times by different components, never crashing,
  just making a number wrong*.
- **A previous fix moved the damage instead of removing it.** Putting the instructions
  last was correct, and it removed the only symptom anyone could have seen. When a fix
  works by making a failure mode's *output* look normal, add a check for the failure
  mode itself at the same time.

## 8. Provenance

- Measurements: ollama 0.32.6, `phi4`, RTX 3060 12GB, 2026-08-10.
- Repair + tests: `tools/eval/rq4_generate.py`,
  `tests/tools/test_rq4_prompt_truncation.py` (3 tests).
- Raw token counts cached at `rq4_exact_tokens.json` (scratchpad; the measurement is
  reproducible from `rq4_generate.build_prompt` + one ollama call per prompt).
