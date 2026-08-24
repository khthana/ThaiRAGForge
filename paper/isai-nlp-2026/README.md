# iSAI-NLP 2026 submission — draft

**Target:** iSAI-NLP 2026, Bangkok, 19–21 Nov 2026.
**Paper deadline: 1 September 2026** (extended from 15 Aug).
**Status: DRAFT, 2026-08-24. Compiles clean at 5 pages.** 0 overfull boxes,
0 undefined references, 10 references typeset, and all 79 figures verified
against their reports. Under the ~6-page target with room to spare — the Thai
prompt figure fits if wanted. Remaining blockers are below and none are about
the text.

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
| `build.ps1` | One-command build: the full pdflatex/bibtex cycle, then reports overfull boxes, undefined references and the figure check. |
| `main.pdf` | The built paper. **Tracked on purpose** — a reviewer-facing artifact that exists on one machine only is a failure mode this repo knows well. |

## Building

MiKTeX 25.12 is installed **per-user** (`winget install --id MiKTeX.MiKTeX -e
--scope user`), so its `bin` is not on the PATH of any shell that started
before the install. `build.ps1` calls the binaries by full path and does not
care:

```
powershell -File paper\isai-nlp-2026\build.ps1
```

It runs `pdflatex` → `bibtex` → `pdflatex` ×2, then **reports the failures that
do not stop a compile**: overfull hboxes (text in the margin) and undefined
citations (which print as `[?]`). pdflatex exits 0 on both, so they must be
checked rather than inferred from the exit code. It finishes by running
`check_paper_figures.py`.

**A silent-failure lesson from building this, worth keeping.** An earlier edit
turned every row terminator in the variants table from `\\` into `\`, which
LaTeX reads as a control space: all rows merged into one paragraph, rules 4–6
ran together, and `pdflatex` reported **0 errors, 0 overfull boxes and the same
5 pages**. It was found by *looking at the rendered page*, not by any log. The
cause was the Bash tool's heredoc collapsing `\\` to `\` in this environment —
so **write LaTeX through a file, never through a heredoc**, and read the PDF
after any table edit.

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
| All `phi4` citation precision/recall/phantom (Table IV) | `data/results/rq4_score.md`, descriptive table |
| `closed_book` abstention, `phi4` (Table VI) | `rq4_score.md`, abstention 2×2 |
| `gemma4:e4b` 24 → 1, phantom 37/37 → 1/1 (Table VI) | `data/results/rq4_score_gemma4.md` |
| The control: answering arms unmoved by the guard | `rq4_score_gemma4.md`, abstention 2×2 |
| Significance family 2, m=9 (Table V) | `rq4_score.md`, "Significance family 2" section only |
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

1. **Confirm iSAI-NLP is Scopus-indexed. This one gates everything else.** The
   2026 site states **no** page limit, **no** registration fee and **no**
   indexing statement. "IEEE Xplore therefore Scopus" is a general practice,
   not a per-series guarantee, and Scopus indexing is the entire reason for
   choosing this venue. Check the Scopus source list for the series directly.
   **If it is not indexed, do not submit** — the constraint is not satisfied
   and the effort belongs on the primary paper instead.
2. **Get the page limit and template** from the organisers. `IEEEtran` and
   "~6 pages" are both inferred from prior editions; the paper is at 5, so a
   6-page limit is comfortable and a 4-page limit would mean real cuts.
3. **Author block, affiliation, acknowledgments** are placeholders.
4. **Decide on the Thai figure** (see above). There is room for it at 5 pages.
5. **Anonymity:** the title block says ANONYMOUS FOR REVIEW. Confirm whether
   iSAI-NLP reviews double-blind; if it does not, fill the names in.

Done: the paper compiles clean (0 overfull, 0 undefined, 5 pages), the
NitiBench reference has its full 7-author list and page range, and the figure
check is green.

## Deliberately absent

- **No dataset release claim.** The corpus can be released, but the minutes name
  real people and that is an unresolved privacy decision
  (`docs/publication-landscape.md` §6.4). This paper needs no release and makes
  no promise of one.
- **No "first" claim of any kind.** A manual IEEE Xplore + TCI-ThaiJo prior-art
  sweep is still owed (`docs/publication-landscape.md` risk #14), and nothing in
  this paper depends on being first.
