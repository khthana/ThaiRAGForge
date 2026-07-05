# Corpus Preparation Tooling

Scripts that produced the `academic_resolutions/` corpus from the KMITL academic-office
website. These are **one-off data-prep utilities**, not part of the RAG framework
(`src/rag_lab/`). Kept for reproducibility.

Run them from the **repository root** (some paths are relative to it). Several scripts
have a hardcoded `BASE_DIR` / `TARGET_SUBFOLDER` at the top — edit those to target a
specific year/session before running.

## Pipeline order

1. **`scrape_kmitl.py`** — Scrapes the KMITL academic-office page for a given ปี พ.ศ.,
   downloads each resolution's PDF, and writes a sibling `<name>_LINK.txt` containing
   the source Google Drive URL. Output → `academic_resolutions/<year>/ครั้งที่ N/`.

2. **`ocr_pdf_to_md.py`** — Main OCR. Converts each PDF to Markdown using a local
   Ollama model (`scb10x/typhoon-ocr1.5-3b`), with image preprocessing, retries, and
   bad-output detection. Writes `<name>.md` next to each PDF. Requires Poppler
   (`POPPLER_PATH`) and the Ollama model built from `Modelfile`.

3. **`delete_bad_ocr.py`** — Cleanup. Scans a session folder for `.md` files containing
   the OCR error marker and deletes them (has a `DRY_RUN` guard — defaults matter, read
   before running).

4. **`check_ocr_coverage.py`** — Diagnostic. Lists PDFs in a session folder and whether
   a matching `.md` already exists. Useful to see what still needs OCR.

5. **`split_curriculum_bundles.py`** — Curriculum splitting (ADR-0004). Some
   resolutions bundle several curricula into one มติ (e.g. one "ปรับปรุงหลักสูตร"
   file covering 3 curricula); this splits each into one physical `.md` file per
   curriculum, patches `meeting_manifest.json`, and archives the original as
   `*.md.dup`. Detection + boundary validation is content-based with a hard
   length/count guard — anything it can't split cleanly goes to
   `academic_resolutions/curriculum_split_review.md` for manual handling rather
   than being guessed at. Dry-run by default; pass `--apply` to write. Run
   **before** `rebuild_manifests.py` so the new files get picked up as ordinary
   corpus entries.

6. **`rebuild_manifests.py`** — Reconciliation (ADR-0003). Scans the corpus tree and
   writes per-meeting `meeting_manifest.json` (full titles + URLs — the metadata
   source of truth) and `academic_resolutions/master_list.csv`. If the original agenda
   capture `1.docx` is still at the repo root it is reconciled in (join on Google Drive
   file IDs); once retired, the tool runs corpus-only and keeps titles from the existing
   manifests. Dry-run by default; pass `--apply` to write. Re-run after adding/renaming
   corpus files (including after a curriculum split); see
   `docs/corpus-reconciliation-log.md` for the reconciliation history.

## Supporting files

- **`Modelfile`** — Ollama model definition for the OCR model (system prompt +
  parameters). Build with `ollama create ... -f Modelfile`.
- **`ocr_legacy.py`** — Earlier OCR script, superseded by `ocr_pdf_to_md.py` (no image
  preprocessing / retry logic). Kept for reference only.

## Dependencies

Covered by the base `pyproject.toml` dependencies (`uv sync`): `ollama`, `pdf2image`,
`pillow`, `typhoon-ocr`, `requests`, `beautifulsoup4`. Also needs a local **Ollama**
install and **Poppler** on the system PATH.
