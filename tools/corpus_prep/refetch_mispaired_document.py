"""Re-fetch and re-OCR a file whose body is another agenda item's document.

`scan_duplicate_bodies.py` finds items that share a body; this repairs the ones
that can be repaired. The distinction matters and is checked here rather than
assumed, because only one of the three shapes is fixable:

  * the two ids serve the same PDF -> the *source* lists one document under two
    items. Nothing to fetch; the manifest title is simply unsupported.
  * the recorded id is dead (404) -> the document existed once and no longer
    does. Also nothing to fetch.
  * the recorded id serves a different document than the file contains -> the
    download stage attached the wrong blob. The right document is sitting at the
    id the manifest already holds, and re-download + re-OCR is the whole fix.
    This is the `2568/ครั้งที่ 7` CHECO mechanism.

So the guard is: fetch the id, and refuse to write unless the fetched PDF's
page-1 heading actually disagrees with what the file already says. Re-OCRing a
document the file already holds would burn GPU time and rewrite the corpus for
nothing, and -- because OCR is not bit-stable across model versions -- would
stale every index for no gain.

Chunk *text* changes here, unlike a title relabel, so every built index holds a
stale vector for this file afterwards. Check the gold set before running: for
`2566/ครั้งที่ 3` no gold query cites the meeting at all, which is what made the
repair free.

Dry-run by default. `--apply` backs the old file up to `<name>.pre_refetch.bak`
(kept in place: the corpus is gitignored, and `.dup` is an overloaded control
signal that must never be reused for backups) and writes the new text.

Run:
    python tools/corpus_prep/refetch_mispaired_document.py "<path to the .md>"
    ... --apply
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_title_body_agreement import flat, subject_line  # noqa: E402

POPPLER = Path(r"C:\poppler\Library\bin")
ID_RE = re.compile(r"/d/([A-Za-z0-9_-]{20,})")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
ENDPOINTS = ["https://docs.google.com/uc",
             "https://drive.usercontent.google.com/download",
             "https://drive.google.com/uc"]


def drive_id(md: Path) -> str | None:
    """The document's own Drive id, from its `_LINK.txt` sidecar."""
    link = md.with_name(md.stem + "_LINK.txt")
    if not link.exists():
        return None
    m = ID_RE.search(link.read_text(encoding="utf-8", errors="replace"))
    return m.group(1) if m else None


def fetch_pdf(fid: str) -> bytes:
    """Fetch through whichever endpoint returns something that is actually a PDF.

    A blocked or missing id answers 200-looking HTML or a 404 page, and an
    earlier version of this probe reported a 1,652-byte error page as a
    different document -- so the `%PDF` magic is checked, not the status code.
    """
    blob = b""
    for url in ENDPOINTS:
        blob = requests.get(url, params={"id": fid, "export": "download", "confirm": "t"},
                            headers={"User-Agent": UA, "Cache-Control": "no-cache"},
                            timeout=180).content
        if blob[:4] == b"%PDF":
            return blob
    return blob


def ocr_pdf(pdf: Path) -> str:
    """OCR every page with the corpus's own pipeline, so the text matches its neighbours."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ocr_pdf_to_md as ocr  # noqa: PLC0415  (heavy: pulls ollama + pdf2image)

    return ocr.process_pdf(pdf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="the .md file whose body is the wrong document")
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = ap.parse_args()

    md = args.target.resolve()
    if not md.exists():
        print(f"[ERROR] not found: {md}")
        return 1

    old = md.read_text(encoding="utf-8", errors="replace")
    fid = drive_id(md)
    print(f"target   {md.relative_to(REPO) if md.is_relative_to(REPO) else md}")
    print(f"drive id {fid}")
    if not fid:
        print("[ERROR] no _LINK.txt sidecar -- cannot tell which document this item should hold")
        return 1

    blob = fetch_pdf(fid)
    if blob[:4] != b"%PDF":
        print(f"[ERROR] id does not serve a PDF ({len(blob):,} b) -- dead or restricted at source; "
              "this item is not repairable by re-download")
        return 1
    print(f"fetched  {len(blob):,} b")

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "fetched.pdf"
        pdf.write_bytes(blob)
        page1 = subprocess.run(
            [str(POPPLER / "pdftoppm.exe"), "-png", "-r", "100", "-f", "1", "-l", "1",
             str(pdf), str(Path(tmp) / "p")], capture_output=True)
        if page1.returncode != 0:
            print("[ERROR] page 1 would not render")
            return 1

        new_text = ocr_pdf(pdf) if args.apply else ""
        if not args.apply:
            print("\n[DRY RUN] would OCR and rewrite the body; re-run with --apply")
            print(f"          current body starts: {flat(old.split(chr(10))[4] if len(old.split(chr(10))) > 4 else old)[:90]}")
            return 0

    if not new_text.strip():
        print("[ERROR] OCR produced nothing -- refusing to overwrite")
        return 1

    header = old.split("\n", 1)[0]
    if not header.startswith("# Document:"):
        header = f"# Document: {md.stem}.pdf"
    rebuilt = f"{header}\n\n{new_text}"

    # The guard the docstring promises: refuse when the fetched document says
    # the same thing the file already says, which means this id was never the
    # mispaired one and the rewrite would be pure churn.
    old_subj, new_subj = subject_line(old), subject_line(rebuilt)
    if old_subj and new_subj and flat(old_subj)[:120] == flat(new_subj)[:120]:
        print("[ERROR] fetched document has the same page-1 subject as the current file; "
              "nothing to repair, refusing to write")
        return 1
    print(f"\nold subject: {flat(old_subj or '')[:100]}")
    print(f"new subject: {flat(new_subj or '')[:100]}")

    backup = md.with_name(md.name + ".pre_refetch.bak")
    if backup.exists():
        print(f"[ERROR] {backup.name} already exists -- refusing to overwrite an earlier original")
        return 1
    backup.write_text(old, encoding="utf-8")
    md.write_text(rebuilt, encoding="utf-8")
    print(f"\n[APPLIED] {len(old):,} -> {len(rebuilt):,} chars; original kept at {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
