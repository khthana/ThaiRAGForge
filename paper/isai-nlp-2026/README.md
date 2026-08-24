# iSAI-NLP 2026 submission — draft

**Target:** iSAI-NLP 2026, Bangkok, 19–21 Nov 2026.
**Paper deadline: 1 September 2026** (extended from 15 Aug).
**Status: DRAFT, 2026-08-24.** Complete argument, all figures verified against
their reports. Not compiled — see *Blockers* below.

Venue reasoning, and why this slice rather than another, is in
`docs/publication-landscape.md` §6.

---

## What this paper is, and what it deliberately is not

**It is** three prompt-ordering findings from RQ4, each measured with the rest of
the system held fixed:

1. Prompt layout interacts with the runtime's front-truncation policy — the
   standard "instructions last" fix converts a loud failure into a silent one.
   81 of 1,590 answers damaged, no symptom.
2. A brevity instruction is a citation-recall cap. `cite_all` raises recall
   significantly on 2 of 4 arms (up to +0.0871, Holm 0.0000, m=9), with no
   retrieval change.
3. A prohibition already in the prompt is defeated by a later rule. Moving it
   after the offender and declaring precedence: 24 → 1 hallucinations on
   `gemma4:e4b`, phantom citations 37/37 → 1/1, answering arms unmoved.

**It is not** the routing result, the test collection, the embedder/chunker
comparison, or any retrieval headline. Those are the spine of the primary paper
(SIGIR-AP 2027) and **must not be consumed here**. The only retrieval facts this
paper uses are the ones it cannot avoid: the four arms exist, and the query set
is 106 queries / 1,046 judgments / 9.87 relevant per query.

---

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper. IEEEtran `conference`, English throughout. |
| `refs.bib` | 10 references, **every one verified against a primary source** during drafting. |
| `prompts_thai.tex` | The prompt rules verbatim in Thai. **Not included by `main.tex`** — pdflatex cannot typeset Thai. See below. |
| `check_paper_figures.py` | Asserts all 79 figures in `main.tex` against the reports that generate them. |

## Compiling

**There is no LaTeX toolchain on this machine** (`pdflatex`, `xelatex` and
`latexmk` are all absent), so this draft has **never been compiled**. Two
consequences you must not assume away:

- **The page count is unverified.** The target is ~6 pages. This draft has 7
  sections, **6 tables and 3,687 words of body text** (counted, not estimated),
  which lands at roughly 5.5-7 pages in IEEE two-column depending on how the
  tables set. Budget for cutting. The cheapest cuts, in order:
  Section II's third paragraph (retrieval components), Finding 1's
  "Blast radius" subsection, and Table~III collapsed to the two `cite_all` rows
  per arm.
- **Nothing has been checked for LaTeX errors.** `\newline` inside a `p{}`
  column and the `\textsc` in a table cell are the two most likely to complain.

Upload the directory to Overleaf and compile there, or install MiKTeX/TeX Live.

### The Thai prompts

`prompts_thai.tex` holds the actual instruction block. It needs XeLaTeX plus a
Thai font:

```latex
\usepackage{fontspec}
\newfontfamily\thaifont{Noto Sans Thai}[Script=Thai]
```

**Recommendation: include it.** At a Thai NLP venue the original prompt is
evidence; the English gloss in Table II is a translation a reviewer cannot
check. Keep both — the gloss for readability, the original for verifiability.
Its source of truth is `tools/eval/rq4_generate.py` (`_RULE4`,
`build_instructions`); regenerate rather than hand-edit if that file changes.

---

## Figure provenance

Run this before every submission, and after any RQ4 re-score:

```
.venv/Scripts/python.exe paper/isai-nlp-2026/check_paper_figures.py
```

It is **not** the repo's `audit_doc_claims.py` D2 check, and the difference
matters. D2 asks whether a figure appears in *some* report — a bag of numbers,
which clears a wrong 4-decimal value 77% of the time. This asserts
(report, row, column) → value, so it names the quantity. It also verifies the
value still appears in `main.tex`, so a figure quietly deleted from the paper
cannot pass.

Driven over three counterfactuals on 2026-08-24, all of which fired:
a wrong figure in the paper, a figure dropped from the paper, and a report
moving under the paper.

| Claim in the paper | Source |
|---|---|
| All `phi4` citation precision/recall/phantom (Table III) | `data/results/rq4_score.md`, descriptive table |
| `closed_book` abstention, `phi4` (Table V) | `rq4_score.md`, abstention 2×2 |
| `gemma4:e4b` 24 → 1, phantom 37/37 → 1/1 (Table V) | `data/results/rq4_score_gemma4.md` |
| The control: answering arms unmoved by the guard | `rq4_score_gemma4.md`, abstention 2×2 |
| Significance family 2, m=9 (Table IV) | `rq4_score.md`, "Significance family 2" section only |
| 106 queries / 1,046 judgments / 9.87 mean | recomputed from `config/eval/gold_query_set_73det.yaml` |
| Truncation table, 14,721-token prompt, 81 of 1,590 cells | `docs/rq4-prompt-truncation.md` §2, §4 |
| Pilot 0/4 vs 4/4 citations, instructions-first | `tools/eval/rq4_generate.py` `build_prompt` docstring |
| Noise floor 14/24 identical citation sets | `docs/rq4-design.md`; generator `tools/eval/rq4_determinism_check.py` |
| `num_predict` cap 4,096; 3 of 1,272 cells | `CLAUDE.md` RQ4 bullet; enforced in `rq4_generate.py` |
| 100% of closed-book hallucination is `course` | `docs/rq4-second-generator-check.md` |

**One trap avoided, recorded so it is not re-introduced.** `docs/rq4-design.md`
carries *older* numbers than the reports (hybrid 0.2781/0.3962 there against
0.2952/0.3823 in `rq4_score.md`) because the 2026-08-20 rebuild-#4 refresh moved
them and the narrative was not re-copied. **Cite the reports, never the design
doc.** This is the same failure the repo's D8 check exists to catch.

---

## Blockers before submission

1. **Compile it.** Page count and LaTeX validity are both unverified.
2. **Confirm iSAI-NLP is Scopus-indexed.** The 2026 site states **no** page
   limit, **no** registration fee and **no** indexing statement. "IEEE Xplore
   therefore Scopus" is a general practice, not a per-series guarantee, and the
   whole reason for choosing this venue is the Scopus requirement. Check the
   Scopus source list for the series directly. **If it is not indexed, do not
   submit** — the constraint is not satisfied and the effort is better spent on
   the primary paper.
3. **Get the page limit and template** from the organisers; `IEEEtran` is an
   assumption from prior editions.
4. **Author block, affiliation, acknowledgments** are placeholders.
5. **Decide on the Thai figure** (see above).
6. **Anonymity:** the title block says ANONYMOUS FOR REVIEW. Confirm whether
   iSAI-NLP reviews double-blind; if it does not, fill the names in.

## Deliberately absent

- **No dataset release claim.** The corpus can be released, but the minutes name
  real people and that is an unresolved privacy decision
  (`docs/publication-landscape.md` §6.4). This paper needs no release and makes
  no promise of one.
- **No "first" claim of any kind.** A manual IEEE Xplore + TCI-ThaiJo prior-art
  sweep is still owed (`docs/publication-landscape.md` risk #14), and nothing in
  this paper depends on being first.
