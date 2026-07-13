# Tickets: Consensus Review App

Builds the tool described in `tools/corpus_prep/consensus_review/SPEC.md` — a
local Streamlit app for triaging the 569 consensus-flagged files from the
full-corpus LLM OCR-corruption scan.

Work the **frontier**: any ticket whose blockers are all done. This chain is
linear, so that means top to bottom.

## Browse consensus-flagged pages, rendered readably

**What to build:** Parse `consensus_priority.md` into ordered file entries,
preserving its existing descending-consensus-page-count order. For each
consensus-flagged page, extract the *full* page Markdown live from the actual
corpus Resolution file (reusing `llm_ocr_scan.split_pages`'s `## Page N`
splitting rule rather than re-deriving it), and render it as real formatted
Markdown — tables as tables — alongside both models' flag reasons. Badge any
file that is one piece of a multi-piece split document (`__1.md`, `__2.md`,
...), reusing the existing `_source_key` / `SPLIT_PIECE` grouping logic from
`llm_ocr_scan.py`. Read-only: no verdict recording yet. This is the seam that
turns unreadable raw OCR text into something a reviewer can actually judge, and
is independently useful on its own.

**Blocked by:** None — can start immediately.

- [x] `logic.py` parses `consensus_priority.md` into structured entries (year,
      file, ordered consensus pages, per-model reason/span), preserving file
      order
- [x] `logic.py` extracts one full page's Markdown from the real corpus `.md`
      file given (year, relative path, page id), reusing
      `llm_ocr_scan.split_pages`
- [x] `logic.py` detects split-document siblings for a given file, reusing
      `_source_key` / `SPLIT_PIECE`
- [x] `review_app.py` walks files in the preserved order, rendering each
      flagged page's full Markdown (tables render as tables) with both models'
      reasons visible, and a badge when the file has split-document siblings
- [x] `logic.py` functions covered by direct pytest unit tests against small
      synthetic fixtures (no real corpus, no Streamlit)
- [x] One `AppTest` smoke test (pattern: `tests/test_streamlit_build_run.py`)
      confirms the app loads and renders the first fixture file's page content

## Record and persist per-file verdicts

**What to build:** Three verdict controls per file — "ควร re-OCR" / "false
positive" / "ไม่แน่ใจ" — plus an optional short note field. Each verdict action
appends one record (`{year, file, verdict, note, timestamp}`) to
`academic_resolutions/llm_ocr_scan/review_decisions.jsonl`; the log is
append-only, and the current state of a file is always its most recent record
(no in-place rewriting). The app defaults to showing only files with no
recorded verdict yet, with a visible progress counter (e.g. "40/569"); a
toggle switches to a view that includes already-decided files so a verdict can
be revisited and changed by simply recording a new one.

**Blocked by:** Browse consensus-flagged pages, rendered readably (needs the
per-file view to attach the verdict controls to).

- [x] `logic.py` appends a verdict record to `review_decisions.jsonl` given
      (year, file, verdict, note)
- [x] `logic.py` resolves the log to current per-file state, where the latest
      record for a file wins over any earlier one
- [x] `review_app.py` shows the three verdict buttons + note field per file,
      and writes through to the log on click
- [x] `review_app.py` defaults to an "undecided only" view with a progress
      counter, and offers a toggle to an "all files" view
- [x] Changing a verdict on an already-decided file (via the "all files" view)
      is reflected as the new current state without deleting the prior record
- [x] `logic.py` append/resolve functions covered by direct pytest unit tests
- [x] `AppTest` smoke test extended to cover: click a verdict, confirm it lands
      in the log, confirm the progress counter moves

## Generate the re-OCR worklist

**What to build:** Resolve the decision log to current per-file state, filter
to files whose verdict is "ควร re-OCR", and write
`academic_resolutions/llm_ocr_scan/reocr_worklist.md` as a plain list of full
relative Resolution file paths — regenerated in full each time, not appended.
Wire a "regenerate worklist" action into the app so the reviewer can produce an
up-to-date list on demand without leaving the tool. This is the final handoff
artifact the rest of the OCR-scan remaining-work list (re-OCR-diff
verification) consumes.

**Blocked by:** Record and persist per-file verdicts (needs decisions to exist
before anything can be filtered into a worklist).

- [ ] `logic.py` generates the worklist content from resolved decisions,
      filtering to verdict == "ควร re-OCR"
- [ ] `review_app.py` exposes a "regenerate worklist" action that writes
      `reocr_worklist.md`
- [ ] `logic.py` worklist generation covered by direct pytest unit tests
      (given a small fixture decision log, assert exact worklist contents)
- [ ] `AppTest` smoke test confirms triggering the regenerate action produces
      the expected file contents for a fixture scenario
