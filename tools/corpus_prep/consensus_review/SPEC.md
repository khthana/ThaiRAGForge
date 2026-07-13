# Spec: Consensus Review App

## Problem Statement

The full-corpus LLM OCR-corruption scan (`tools/corpus_prep/llm_ocr_scan.py`) has
finished all 6 years and produced `consensus_priority.md`: 569 Resolution files,
872 pages that both models (phi4:latest and gemma4:e4b) independently flagged as
corrupted. This is already the highest-confidence subset (single-model-only flags
are known to be noisy and are excluded).

Even narrowed to this list, reviewing it by eye is slow and error-prone:

- The flagged text is only a short quoted span, not the full page — for garbled
  prose that's often enough, but for tables the span alone strips the row/column
  context that would let a person tell corruption from a merely eccentric-but-real
  layout.
- Raw OCR Markdown tables are hard to read as flat text even when correct —
  pipe-table source doesn't visually align into rows the way a rendered table does.
- There is no place to record a verdict per file, and no resulting list of "these
  files need re-OCR" to hand off to the next step in the pipeline.

## Solution

A local, single-user Streamlit tool that walks the 569 consensus-flagged files
(already ranked by consensus-page count in `consensus_priority.md`), renders each
flagged page's full content as properly formatted Markdown — pulled live from the
actual corpus Resolution file, not the short quoted span — alongside both models'
flag reasons, and lets the reviewer record one verdict per file. Verdicts persist
to an append-only log so the review session is resumable across restarts. A
derived worklist file lists exactly which Resolutions need re-OCR, ready for the
next step (the existing manual re-OCR-diff verification process).

## User Stories

1. As the reviewer, I want the app to open directly to the highest-priority
   unreviewed file (most consensus pages first, matching `consensus_priority.md`
   order), so that I spend my limited review time on the strongest signals first.
2. As the reviewer, I want every consensus-flagged page of a file rendered as real
   Markdown (tables as tables, not raw pipes), so that I can actually read the
   content instead of parsing OCR-format text by eye.
3. As the reviewer, I want the full page content shown — not just the short quoted
   span each model flagged — so that I have enough surrounding context (the rest
   of the table row, the rest of the paragraph) to judge whether it's really
   corrupted.
4. As the reviewer, I want to see both models' flag reason (and which model said
   what) next to the rendered page, so that I understand why the page was flagged
   without cross-referencing another file.
5. As the reviewer, I want a visible badge when a file is one piece of a
   multi-piece split document (`__1.md`, `__2.md`, ...), so that I remember this
   is a known higher-risk category and think to check sibling pieces too.
6. As the reviewer, I want to record one verdict per file — not per page — with
   three options ("ควร re-OCR" / "false positive" / "ไม่แน่ใจ"), so that my
   decision matches the actual unit of remediation (re-OCR runs per file).
7. As the reviewer, I want an optional short note field per decision, so that I
   can leave myself a hint (e.g. "เฉพาะหน้า 3") for when I act on the worklist
   later.
8. As the reviewer, I want my decisions saved immediately and durably, so that
   closing the app mid-review loses no progress.
9. As the reviewer, I want the app to default to showing only files I haven't
   decided on yet, with a visible progress counter (e.g. "40/569"), so that I
   always know how much is left and never re-review the same file by accident.
10. As the reviewer, I want the option to switch to a view that includes already-
    decided files, so that I can revisit and change a verdict I got wrong or
    changed my mind about.
11. As the reviewer, when I change a verdict on an already-decided file, I want
    the app to treat the newest decision as authoritative while keeping the old
    one in the log, so that I get an undo-equivalent without any destructive edit.
12. As the reviewer, I want a generated plain-language worklist of exactly the
    files marked "ควร re-OCR", so that I can hand it directly to the existing
    re-OCR-diff verification process without re-deriving it myself.
13. As the maintainer, I want the review logic (parsing, page extraction, decision
    persistence, worklist generation) importable and testable independently of
    Streamlit, so that the core behavior has real automated test coverage rather
    than only manual smoke-testing.
14. As the maintainer, I want this tool fully contained under `tools/corpus_prep/`
    and read-only with respect to the actual corpus, so that it fits the existing
    corpus-prep tooling boundary and carries no risk of mutating Resolution
    content.

## Implementation Decisions

- **Location**: new package `tools/corpus_prep/consensus_review/`, sibling to the
  existing flat scripts in `tools/corpus_prep/` (`llm_ocr_scan.py`,
  `ocr_pdf_to_md.py`, etc.) but grouped in its own subfolder since it's more than
  one file. Fully standalone — not wired into `app/`'s multipage Streamlit nav
  (different concern: corpus-prep triage, not retrieval/query).
- **Module split**:
  - `logic.py` — pure, Streamlit-free functions:
    - Parse `consensus_priority.md` into structured entries: year, Resolution
      file path, list of consensus pages, and for each page the per-model
      (reason, quoted span). Preserve the file's existing descending-consensus-
      count order rather than re-sorting.
    - Detect split-document siblings (`__1.md`, `__2.md`, ...) for the badge —
      reuse the existing `_source_key` / `SPLIT_PIECE` grouping logic already in
      `llm_ocr_scan.py` rather than reimplementing the pattern.
    - Extract one full page's Markdown from the actual corpus Resolution file,
      splitting on `## Page N` headers the same way `llm_ocr_scan.split_pages`
      does (reuse that function directly if its signature allows, rather than
      duplicating the split rule).
    - Decision log: append one JSON record per verdict action
      (`{year, file, verdict, note, timestamp}`) to
      `academic_resolutions/llm_ocr_scan/review_decisions.jsonl`. Resolution of
      "current" state is "last record per file wins" — no in-place rewriting of
      earlier lines.
    - Worklist generation: filter resolved decisions to verdict ==
      "ควร re-OCR", write `academic_resolutions/llm_ocr_scan/reocr_worklist.md`
      as a plain list of full relative Resolution file paths (regenerated fully
      each time, not appended).
  - `review_app.py` — thin Streamlit UI layer that calls only into `logic.py`:
    sidebar/toggle for "undecided only" (default) vs "all", progress counter,
    per-file detail view (split-cluster badge, each consensus page rendered via
    `st.markdown()` with both models' reasons shown alongside), the three verdict
    buttons + optional note field, and a manual "regenerate worklist" action.
- **Data flow boundary**: `consensus_priority.md` is read-only input (never
  rewritten by this tool); actual corpus `.md` files are read-only (never
  rewritten); only `review_decisions.jsonl` and `reocr_worklist.md` are written,
  both already-gitignored paths under `academic_resolutions/llm_ocr_scan/`.
- **Out-of-band context reused, not rebuilt**: page-splitting rule and
  split-piece detection both already exist in `llm_ocr_scan.py` — import and
  reuse rather than re-derive, to avoid the two implementations drifting apart.

## Testing Decisions

- Good tests here check observable outcomes only: given a small fixture
  `consensus_priority.md` and a small fixture corpus tree (a handful of `.md`
  files with `## Page N` sections, mirroring the `_write_corpus` helper pattern
  in `tests/test_streamlit_build_run.py`), verify the parsed structure, the
  extracted page text, the resolved "latest verdict wins" behavior, and the
  generated worklist's contents — not how `logic.py` is internally organized.
- `logic.py` gets direct pytest unit tests (no Streamlit involved), following the
  existing pattern of testing `rag_lab.config` / `rag_lab.runner` directly rather
  than through any UI (`tests/test_config.py`, `tests/test_manifest.py` are prior
  art for this style of fixture-driven, filesystem-based unit test).
- `review_app.py` gets exactly one `streamlit.testing.v1.AppTest` smoke test
  (prior art: `tests/test_streamlit_build_run.py`), driving the real widgets for
  one end-to-end scenario: load a fixture file, click a verdict button, assert
  the decision landed in `review_decisions.jsonl` and the progress counter moved.
  No per-widget unit testing beyond that single smoke path, matching the existing
  project convention noted in `CLAUDE.md` for `app/pages/`.
- No test runs against the real ~2,853-file corpus or the real
  `consensus_priority.md` — all tests use small synthetic fixtures under
  `tmp_path`.

## Out of Scope

- Rendering the source PDF page image for side-by-side comparison — the actual
  problem reported was Markdown/table readability, not needing to see the
  original scan; deferred as a possible v2 if text-only review turns out to be
  insufficient for some cases.
- Reviewing single-model-only flags (the noisier, non-consensus pool) — v1 is
  consensus-only; a lower-confidence phase 2 queue is a separate future effort.
- Performing the re-OCR itself, or automating the re-OCR-diff verification loop
  — both stay the existing manual process (`ocr_pdf_to_md.py` on a copied PDF,
  diffed by hand) run by the reviewer after consulting the worklist this tool
  produces.
- Any write access to actual corpus Resolution `.md` files — strictly read-only.
- Multi-user support, authentication, or a hosted deployment — single local
  reviewer, run via `streamlit run`.
- Integration into the existing `app/` multipage Streamlit UI — this stays a
  fully separate script under `tools/corpus_prep/`.

## Further Notes

- This spec was reached via an interview (`/grilling`) rather than upfront
  requirements; all listed decisions were explicitly confirmed by the user in
  that session, not assumed.
- Feeds into the project's own documented "remaining work" list for the OCR scan
  (see `docs/llm-ocr-scan-log.md` Addendum 2/3 and the
  `project-llm-ocr-scan-experiment` memory entry): this tool is the mechanism for
  step 1 of that list ("read the consensus list by eye"), and its worklist output
  feeds step 2 ("re-OCR-diff verification on confident finds").
- Per the user's explicit request, this spec is saved as a file in the repo
  (`tools/corpus_prep/consensus_review/SPEC.md`) rather than published to the
  GitHub issue tracker, which is `/to-spec`'s default target.
