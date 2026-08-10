# RQ4 prompt truncation — measured 2026-08-10

**สรุปสั้น (ไทย):** `rq4_generate.py` ตั้ง `num_ctx=8192` มาตลอด แต่ prompt ที่ยาวที่สุด
ยาว **14,721 tokens** — ollama ไม่ได้เตือนอะไรเลย มันแค่ **ตัดหัว prompt ทิ้ง** เหลือ
`num_ctx/2` tokens แล้วเก็บ *ท้าย* ไว้ ซึ่งคือกรณีที่แย่ที่สุดสำหรับเรา เพราะ `build_prompt`
วางเอกสารอันดับดีที่สุดไว้ **หน้าสุด** และวางกติกาไว้ **ท้ายสุด** → สิ่งที่ถูกลบคือหลักฐาน
ที่ดีที่สุด ส่วนกติกาอยู่ครบเสมอ คำตอบจึงยังออกมา "ดูดี" และไม่มีอาการอะไรให้เห็น
กระทบคำตอบที่เผยแพร่แล้ว **80 จาก 1,590 เซลล์** ตอนนี้ซ่อมที่ต้นเหตุแล้ว (ยกเป็น 16384
+ pre-flight ที่ *วัด* ไม่ใช่ *สมมติ* + ตัวจับลายเซ็นรายคำตอบ + บันทึก `prompt_eval_count`
ลงทุกไฟล์คำตอบ)

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
| cite_all_guarded | dense_qwen3_0.6b_semantic | 106 | 16 (15%) | 13,433 | 31% |
| cite_all_guarded | bm25_semantic | 106 | 5 (5%) | 11,396 | 36% |
| cite_all_guarded | hybrid_m2v_semantic | 106 | 7 (7%) | 9,901 | 41% |
| cite_all_guarded | closed_book | 106 | 0 (0%) | 0 | 100% |
| cite_all_guarded | entity_lookup_semantic | 106 | 53 (50%) | 13,842 | 30% |
| cite_all_guarded | entity_boost_semantic | 106 | 51 (48%) | 14,721 | 28% |

**Published answers affected: 80 of 1,590 (query, arm, variant) cells (5.0%)**, over
**28 (arm, query) pairs / 25 distinct queries**, in **3 of the 5 published arms**.

Read the shape of that table before reading the headline, because the shape carries the
argument:

- **`hybrid`, the headline arm, is untouched at 0/106 in all three variants** — its worst
  prompt is 7,999 tokens, 193 short of the line. Nothing in the design put it there; it is
  luck, and `cite_all_guarded` came within 2.4% of losing it.
- **`closed_book` is 0/106 by construction** (no documents), which is what makes "the
  damage tracks context length" a measurement and not an assumption.
- **The `hybrid` vs `dense` vs `bm25` ordering is the one claim most at risk**: dense is
  the most-truncated published arm (16/106) and hybrid the least (0/106), so truncation
  pushed in exactly the direction the published ordering reports. The published finding
  `hybrid > {dense, bm25}` is therefore **not confirmed and not refuted by this** — it is
  measured under a confound that flatters it, and 5.0% of cells is small but the confound
  is not random with respect to the comparison.
- **The two entity arms — the ones #1 exists to measure — would have been ~45-50%
  truncated at 8192.** That run has not happened, which is the whole point of catching
  this before it did.

Not yet re-measured: whether regenerating those 80 cells moves any verdict. Note the
measured generator noise floor makes this harder than a diff
([[feedback_temperature_zero_is_not_reproducible]]: only 14/24 identical citation sets at
temperature 0), so a re-run must be scored, not eyeballed.

## 5. Method note: chars/token is not usable as an estimator here

The obvious screen — convert a character budget to tokens with an average ratio — fails on
this corpus. Observed chars/token across the *same* prompt family spans **1.046** (Thai
prose, the phi4 tokenizer's worst case) to **3.151** (English course-code comparison
tables). One 15,915-character `entity_boost` prompt is only 5,051 tokens.

So: screen with the **minimum** ratio (which can only over-select), then measure everything
above the line **exactly** with ollama's own `prompt_eval_count`. 742 (arm, query) pairs
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
