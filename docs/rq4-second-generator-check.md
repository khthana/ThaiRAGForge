# RQ4 second-generator robustness check — `gemma4:e4b` vs `phi4`

> **Not a script-generated report.** The two score tables it compares are
> generated (`tools/eval/rq4_score.py` → `data/results/rq4_score_gemma4.md` and
> `rq4_score_gemma4_guarded.md`, against the published `rq4_score.md` /
> `rq4_score_guarded.md`); this file is the hand-written comparison of the four.
> Run 2026-08-12.

## 0. Why it was run, and what it can and cannot answer

The check was deferred on a **pre-registered rule** in `rq4_score.py`'s family 2:
a second generator is the right next test *only if* citation recall stays flat
under `cite_all`, because a flat line is the shape a real generator ceiling
makes. Recall did not stay flat — it rose (hybrid +0.1181, Holm 0.0000) — so by
that rule the question the check was queued to answer was already closed, and it
was run on explicit instruction rather than on the rule.

That makes it worth being precise about what it *does* answer. Not "is the
generator the bottleneck" (closed: it was the prompt). What it answers is a
question nothing else here has touched:

> **Does the arm ordering — retrieval quality surviving into answer quality —
> depend on which local model writes the answer?**

## 1. Setup

| | |
|---|---|
| model | `gemma4:e4b` (9.6 GB), against the published `phi4` |
| variants | `cite_all` **and** `cite_all_guarded`, 530 answers each |
| arms | the 5 published arms (`--arms`, so the two entity arms stay out of family 1's Holm size) |
| contexts | identical files — the same `data/rq4/contexts/` the phi4 answers were written from, so the comparison is paired by construction |
| `num_ctx` | 16,384 |
| `think` | supported by this model → **disabled** on all 530×2 answers |
| wall clock | 2,231 s + 2,338 s = 76 min for 1,060 answers |

`sentence_cap` is not available: `rq4_generate.py` refuses it for any model but
`phi4`, because those 530 answers are keyed to that model/prompt pair. So
family 1a (arm ordering under `sentence_cap`), family 2 (prompt ablation vs the
`sentence_cap` baseline) and family 3 (all pairwise variants, needs ≥3) all
**skip** for `gemma4:e4b` — by design, not by failure. **Family 1b is the whole
deliverable here.**

Prompt-fit was clean and is recorded rather than assumed: **0 of 1,060** answers
carry the `prompt_eval_count == num_ctx//2 + 2` truncation signature, longest
prompt **6,714** tokens. Note that is well under even the retired 8,192 default,
where the same contexts drove `phi4` to 7,999 — the two models' tokenizers
disagree substantially on this corpus, so a prompt-fit result **does not
transfer between models**.

## 2. Result A — arm ordering

### Means (family 1b, n=106, Holm m=12)

**Citation precision — the ordering is identical in all four positions, under
both prompts.**

| variant | model | hybrid | dense | bm25 | m2v |
|---|---|---|---|---|---|
| `cite_all` | phi4 | 0.7268 | 0.6798 | 0.6104 | 0.5278 |
| `cite_all` | gemma4:e4b | 0.7417 | 0.7375 | 0.6850 | 0.6279 |
| `cite_all_guarded` | phi4 | 0.6900 | 0.6746 | 0.6217 | 0.4817 |
| `cite_all_guarded` | gemma4:e4b | 0.7208 | 0.7371 | 0.6028 | 0.5787 |

**Citation recall — `gemma4:e4b` scores higher on every arm, and the top two
swap.**

| variant | model | hybrid | dense | bm25 | m2v |
|---|---|---|---|---|---|
| `cite_all` | phi4 | 0.3962 | 0.3356 | 0.2961 | 0.2038 |
| `cite_all` | gemma4:e4b | 0.4846 | **0.5074** | 0.3991 | 0.2478 |
| `cite_all_guarded` | phi4 | 0.3487 | 0.3460 | 0.2798 | 0.1943 |
| `cite_all_guarded` | gemma4:e4b | 0.5155 | **0.5263** | 0.4154 | 0.2800 |

### Verdict agreement

| variant | cells agreeing | significant (phi4) | significant (gemma) |
|---|---|---|---|
| `cite_all` | **10 of 12** | 8 | 8 |
| `cite_all_guarded` | **7 of 12** | 6 | 9 |

**No flip is a reversal.** Every disagreement is one model resolving a
comparison the other leaves inconclusive, in the *same* direction:

- `cite_all`: `bm25 > m2v [precision]` significant for phi4 only (−0.1148,
  Holm 0.0132) and `dense > bm25 [recall]` for gemma only (−0.1083, Holm 0.0320).
- `cite_all_guarded`: gemma separates **bm25 from both strong arms on both
  metrics** (four cells, Holm 0.0000–0.0048) where phi4 cannot (0.1056–0.5466);
  phi4 separates `bm25 > m2v [precision]` where gemma cannot.

### The one pair whose sign disagrees

Checked mechanically across all 24 cells (12 comparisons × 2 variants): the sign
of the effect agrees between the two models **everywhere except `hybrid` vs
`dense`**, which disagrees in all four of its cells.

| variant | metric | phi4 | gemma4:e4b |
|---|---|---|---|
| `cite_all` | recall | −0.0606, CI [−0.1115, −0.0098] | **+0.0228**, CI [−0.0258, +0.0711] |
| `cite_all` | precision | −0.0648 | +0.0030 |
| `cite_all_guarded` | recall | −0.0028 | +0.0108 |
| `cite_all_guarded` | precision | −0.0295 | +0.0118 |

Three of the four are near-zero and Holm-ns under both models, so the
disagreement is noise there. **The fourth is not free**: `phi4`/`cite_all` is the
cell CLAUDE.md tells the reader to cite as a **bound** — Holm 0.0760, ns, but a
CI that excludes zero, i.e. "rules out dense beating hybrid, do not read it as
hybrid winning". Under `gemma4:e4b` that CI straddles zero and the point estimate
points the other way. **The bound does not replicate.**

### What is citable

> Under both local generators and both live prompts, citation grounding
> preserves the ordering **{hybrid, dense} > bm25 > m2v** — m2v is last on both
> metrics in all four runs, and both strong arms beat bm25 numerically in all
> four. **`hybrid` vs `dense` is unresolved and generator-dependent**: it is
> Holm-ns in all four cells and its sign flips with the generator.

What must **not** be said: that gemma "confirms" the phi4 table. It does not
confirm the one bound in it, and its recall level is 0.09–0.18 higher on every
arm, so the absolute figures are model-specific even where the ordering is not.

## 3. Result B — the zero-document guard, tested on a second model

This was not the reason for the run; it is the larger finding.

`cite_all` has no rule telling the model what to do when **no documents are
supplied at all**. For `phi4` that cost 2 hallucinations and 5 phantom citations
out of 106 closed-book queries, and rule 5 of `cite_all_guarded` (abstain, cite
`-`, and this outranks rule 4) repaired it to 0/0. On a second model the same
hole is an order of magnitude wider:

| model | variant | correctly abstained | hallucinated | phantom citations |
|---|---|---|---|---|
| phi4 | `cite_all` | 104 / 106 | 2 | 5 / 5 |
| phi4 | `cite_all_guarded` | 106 / 106 | 0 | 0 / 0 |
| gemma4:e4b | `cite_all` | 82 / 106 | **24** | **37 / 37** |
| gemma4:e4b | `cite_all_guarded` | **105 / 106** | **1** | 1 / 1 |

**Rule 5 generalises: 24 → 1.** It does not reach zero the way it does for phi4,
and the survivor is the same shape as the other 24.

### The hallucinations are one query type, in both models

| closed-book hallucinations | course | person | program | faculty |
|---|---|---|---|---|
| gemma4:e4b `cite_all` | **24 / 33** | 0 / 30 | 0 / 30 | 0 / 13 |
| phi4 `cite_all` | 2 / 33 | 0 / 30 | 0 / 30 | 0 / 13 |

100% of closed-book hallucination in both models is `course` queries. The
failure is not diffuse; a course code invites the model to invent a meeting
number and a citation label for it. Example (`gemma4:e4b`, `cite_all`, q079,
**zero documents supplied**):

> คำตอบ: รายวิชา ECOLOGY, CONSERVATION AND ENVIRONMENTALISM ถูกกล่าวถึงใน
> การประชุมสภาสถาบัน ครั้งที่ 1/2567 [3] และในการประชุมสภาสถาบัน ครั้งที่ 2/2567 [3]

The abstention detector was checked before the number was believed: it is a
substring test for `ไม่พบข้อมูล`, so a model phrasing its refusal differently
would be miscounted as a hallucination. All 24 were read; every one asserts a
meeting number and cites a label that cannot exist. The instrument is right.

### The guard's cost is model-specific — and the published cost is phi4's

CLAUDE.md records that the guard "is not free": for `phi4` it pushes the weak
arms toward abstention (m2v *missed* 10 → 18). **That does not reproduce.** For
`gemma4:e4b` the guard *lowered* missed on every arm (hybrid 10 → 7, dense
12 → 7, bm25 14 → 11, m2v 15 → 14) and *raised* recall on every arm (e.g.
hybrid 0.4846 → 0.5155). Its cost here lands on precision instead, and only on
the weak arms (bm25 0.6850 → 0.6028, m2v 0.6279 → 0.5787).

So the trade the guard makes is a property of the (model, prompt) pair, not of
the prompt. State which model any such figure came from.

## 4. Conclusions

1. **The paper's arm ordering is robust to a generator swap, except for the
   `hybrid` vs `dense` pair, which was already the weakest claim in the table.**
   Report `{hybrid, dense} > bm25 > m2v` as generator-independent and
   `hybrid` vs `dense` as unresolved under both.
2. **The `hybrid > dense` bound from `phi4`/`cite_all` does not replicate.** It
   should be stated as a phi4 result, not a system result.
3. **Rule 5 is load-bearing, not cosmetic.** Its value was measured on the model
   that barely needed it; on a second model it prevents 23 of 24 hallucinations.
   That is an argument for `cite_all_guarded` as the paper's prompt considerably
   stronger than the one that originally selected it.
4. **`course` queries are the entire closed-book failure surface** in both
   models — a target for a rule 7 if this is ever pushed further, and consistent
   with `course` being the entity type with the most retrieval headroom.
5. **Absolute citation figures do not transfer between generators** (recall
   +0.09 to +0.18 across the board here), and neither does prompt-fit (6,714
   tokens vs 7,999 on identical contexts). Only orderings transfer, and only
   the ones above.

## 5. Reproduce

```bash
PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_generate.py \
  --model gemma4:e4b --variant cite_all \
  --arms hybrid_qwen3_0.6b_semantic,dense_qwen3_0.6b_semantic,bm25_semantic,hybrid_m2v_semantic,closed_book
# ... and again with --variant cite_all_guarded

PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_score.py \
  --model gemma4_e4b --out data/results/rq4_score_gemma4.md
PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_score.py \
  --model gemma4_e4b --treatment-variant cite_all_guarded \
  --out data/results/rq4_score_gemma4_guarded.md
```

Note `--model gemma4_e4b` (underscore) when scoring — the answers directory is
named from `model.replace(":", "_")`. And `--out` is **mandatory** here: the
guard that refuses to overwrite the published `rq4_score.md` keys on `--arms`,
not on `--model`, so a default-path run with a different model would clobber it.

## Refreshed 2026-08-20 against rebuild #4 — the conclusion holds, two supporting figures do not

All 466 changed `gemma4:e4b` cells were regenerated (76 min, 0 errors, 0 capped)
and both reports re-scored: **1 verdict flip of 12 under `cite_all`, 0 of 12
under the guard.** The pre-refresh reports are snapshotted at
`data/results/_pre_2026_08_18_rebuild4_refresh/rq4_score_gemma4{,_guarded}.md`,
so every figure below is diffable rather than remembered.

**§4's four-position precision ordering no longer holds for gemma.** `cite_all`
now reads dense **0.7314** > hybrid **0.7277** > bm25 **0.6879** > m2v **0.6270**
— the top two swapped — while phi4 still puts hybrid clearly first (0.7185 >
0.6381). Positions 3 and 4 are unchanged in both models.

**Verdict agreement moved in both directions**, which is why the count alone was
never the finding: **10 of 12 → 7 of 12** under `cite_all`, **7 of 12 → 8 of 12**
under the guard.

| | before | after |
|---|---|---|
| `cite_all` verdicts agreeing | 10 / 12 | **7 / 12** |
| `cite_all_guarded` verdicts agreeing | 7 / 12 | **8 / 12** |
| sign disagreements over all 24 cells | 4 (all `hybrid` vs `dense`) | **2 (both `hybrid` vs `dense`, both under `cite_all`)** |

**The mechanical sign check is now the sharper statement.** Both remaining
disagreements are the `hybrid` vs `dense` pair under `cite_all` (precision and
recall); under the guard all 12 signs agree. And the phi4 side of that pair is
**significant again** after the rebuild — recall −0.0678 (Holm **0.0410**),
precision −0.0798 (Holm 0.0410), both hybrid-favouring — which reverses the
2026-08-10 "now a bound, not a result" wording, while gemma stays ns and points
the other way (+0.0093 recall, +0.0026 precision).

**So §4's guidance survives a second index generation and is now the whole
finding: cite `{hybrid, dense} > bm25 > m2v` as generator-independent, and
`hybrid > dense` as a phi4 result, not a system result.**

**Two controls held.** (1) `closed_book` is **byte-identical** across the
refresh — 24 hallucinations under `cite_all`, 1 under the guard, phantom 37/37 →
1/1 — which is expected, since its context is empty and no rebuild can change
it, and is what separates repair from generator drift in an unpaired-looking
table. Result B is therefore untouched. (2) Every verdict disagreement outside
the `hybrid`/`dense` pair is still one model resolving what the other leaves
inconclusive, never a reversal.

**One level worth carrying, because it cuts against the obvious reading of a
re-OCR.** gemma's `bm25` citation recall **fell** 0.3991 → **0.3784** under
`cite_all`, and that is what strengthened both strong arms' margins over it
(`hybrid vs bm25` Holm 0.0320 → 0.0000, `dense vs bm25` 0.0320 → 0.0016) — the
opposite direction to phi4's dense arm, which *gained* from the same re-OCR. **A
rebuild is not uniformly helpful per arm**; read each arm's own movement before
attributing a margin to the treatment.
