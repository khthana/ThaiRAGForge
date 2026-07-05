# Meeting manifest is the metadata source of truth, not the filename

---
Status: accepted
---

## Context

Resolution metadata (the เรื่อง/title and the source URL used to link back to the
original PDF) originally lived **in the filename**: the loader took `<stem>.md` as
the title and read the URL from a sibling `<stem>_LINK.txt`. Reconciling the corpus
against the agenda capture (`1.docx`, joined on Google Drive file IDs) showed this
is lossy:

- The download tooling truncates filenames to ~100 characters — **598 of ~2,320**
  titles were cut mid-word; the full titles existed only in the agenda capture.
- Renaming files back to full titles would re-hit the same limits (Windows
  `MAX_PATH`, POSIX-layer tools such as Git Bash mangle long Thai names).
- Special sessions (วาระพิเศษ) were foldered under three different naming schemes,
  and the session parser collapsed them onto the regular session number, colliding
  `resolution_id`s across distinct meetings (breaking ADR-0002's stable Resolution
  identity).

## Decision

Each meeting folder carries a **`meeting_manifest.json`**: a list of
`{file, title, url, title_source}` entries mapping every `.md` file to its full
recovered title and provenance URL. Loaders (via `loaders/common.py`) prefer the
manifest and fall back to filename + `_LINK.txt` when a file is not listed, so the
corpus remains loadable without manifests.

Folder naming is normalized to `<year>/ครั้งที่ N` and `<year>/ครั้งที่ Ns` for
special sessions; the session parser keeps the `s` suffix so `2566/4s/...` never
collides with `2566/4/...`.

Filenames are demoted to opaque pointers: they stay as-downloaded (truncated is
fine) and are never edited to carry metadata.

## Consequences

- `resolution_id`s and Silver-query titles use full, correct titles; link-back URLs
  survive even where `_LINK.txt` was broken or duplicated.
- The reconciled corpus inventory lives in `academic_resolutions/master_list.csv`
  (one row per resolution: meeting, full title, URL, file, status), which now
  supersedes `1.docx` as the master inventory (the agenda capture has been retired;
  `rebuild_manifests.py` runs corpus-only and keeps titles from the manifests).
- Fixing a title or URL means editing one manifest entry — no file renames, no
  re-download.
- Manifests are read once per process (cached); a live Streamlit session must be
  restarted to pick up manifest edits.
