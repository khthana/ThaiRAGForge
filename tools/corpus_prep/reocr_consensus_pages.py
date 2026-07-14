"""Phase 1 of the consensus-flagged re-OCR pipeline: re-run OCR fresh on every
page that both phi4:latest and gemma4:e4b flagged (`consensus_priority.md`),
staging results without touching the real corpus at all. A later phase
(deferred) has a local LLM cross-check each staged result against the current
corpus text and decide which is right; only then does anything get written
back to `academic_resolutions/`.

Split-document siblings (`__1.md`, `__2.md`, ...) frequently share a flagged
page from the same source PDF -- OCR is run once per unique (pdf, page) pair,
not once per corpus .md entry, to avoid redundant model calls.

Resumable: re-running skips (pdf, page) pairs already present in the staging
JSONL, same convention as `llm_ocr_scan.py`'s own scan.

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/reocr_consensus_pages.py
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "consensus_review"))
from llm_ocr_scan import _source_key  # noqa: E402
from ocr_pdf_to_md import ocr_image, POPPLER_PATH, PDF_DPI  # noqa: E402
import logic  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
DEFAULT_SRC_ROOT = Path(r"D:/academic_resolutions (ข้อมูลดิบ + OCR)")
CONSENSUS_FILE = CORPUS_ROOT / "llm_ocr_scan" / "consensus_priority.md"
STAGING_FILE = CORPUS_ROOT / "llm_ocr_scan" / "reocr_pages_staging.jsonl"
UNRESOLVED_FILE = CORPUS_ROOT / "llm_ocr_scan" / "reocr_unresolved_files.txt"

_SPECIAL_SESSION = re.compile(r"^ครั้งที่ (\d+)s$")
_PAGE_INT = re.compile(r"^Page (\d+)(?:\.\d+)?$")
_DEDUP_SUFFIX = re.compile(r"\s\(\d+\)$")
_HEADER_LINE = re.compile(
    r"^# Document:\s*(?:\d+s?-\d{4}\s+)?(?P<filename>.+\.pdf)(?:\s*\(ส่วนที่\s*\d+/\d+\))?\s*$"
)


def resolve_src_dir(src_root: Path, year: str, meeting: str) -> Path | None:
    """The corpus names special-session folders `ครั้งที่ Ns`; the raw-data
    drive names the same session `ครั้งที่ N_YYYY` (or, for 2564, `วาระพิเศษ
    ครั้งที่ N_YYYY`). Try the corpus name directly first since it matches
    for every regular session."""
    year_dir = src_root / year
    direct = year_dir / meeting
    if direct.exists():
        return direct
    m = _SPECIAL_SESSION.match(meeting)
    if m:
        n = m.group(1)
        for candidate in (f"ครั้งที่ {n}_{year}", f"วาระพิเศษ ครั้งที่ {n}_{year}"):
            candidate_dir = year_dir / candidate
            if candidate_dir.exists():
                return candidate_dir
    return None


def parse_document_header(md_path: Path) -> str | None:
    """Every OCR'd corpus file opens with `# Document: <original filename>`
    (optionally prefixed with `<meeting>-<year> ` and suffixed with `(ส่วนที่
    N/M)` for a curriculum-split piece). That original filename sometimes
    differs from the corpus .md stem -- e.g. it still carries an agenda-item
    number/prefix that got stripped when the .md was named -- so it's worth
    trying as a fallback source-PDF name. None if the file is missing or the
    header doesn't match the expected shape."""
    try:
        first_line = md_path.open(encoding="utf-8").readline()
    except (FileNotFoundError, UnicodeDecodeError):
        return None
    m = _HEADER_LINE.match(first_line.strip())
    return m.group("filename") if m else None


def resolve_pdf_path(src_root: Path, relpath: str) -> Path | None:
    """The source PDF for a corpus-relative .md path, collapsing split-piece
    siblings back to their shared pre-split document via `_source_key`. None
    if the session folder or the PDF itself can't be found -- callers should
    treat that file as needing manual identification, not an error.

    Tries three increasingly-indirect candidates: the corpus stem as-is; the
    stem with a trailing corpus-only disambiguation suffix like " (2)"
    stripped (added when the corpus held two same-titled documents); and the
    original filename recovered from the file's own `# Document:` header."""
    parent, stem, _ = _source_key(Path(relpath))
    parts = parent.parts
    if len(parts) < 2:
        return None
    src_dir = resolve_src_dir(src_root, parts[0], parts[1])
    if src_dir is None:
        return None

    direct = src_dir / (stem + ".pdf")
    if direct.exists():
        return direct

    deduped_stem = _DEDUP_SUFFIX.sub("", stem)
    if deduped_stem != stem:
        deduped = src_dir / (deduped_stem + ".pdf")
        if deduped.exists():
            return deduped

    header_name = parse_document_header(CORPUS_ROOT / relpath)
    if header_name:
        by_header = src_dir / header_name
        if by_header.exists():
            return by_header

    return None


def page_label_to_int(label: str) -> int | None:
    """"Page 12" -> 12. "Page 7.1" is a sub-chunk of physical page 7 (created
    when `llm_ocr_scan.split_pages` breaks an oversized page into pieces for
    the scan's own context-window budget) -- also 7, since re-OCR-ing the
    physical page covers every sub-chunk. None for anything else (e.g. the
    "chunk1" fallback label `split_pages` emits when a file has no page
    markers at all -- there's no physical page number to re-render)."""
    m = _PAGE_INT.match(label)
    return int(m.group(1)) if m else None


def _default_page_count(pdf_path: Path) -> int:
    from pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(str(pdf_path), poppler_path=POPPLER_PATH)
    return int(info["Pages"])


@dataclass(frozen=True)
class WorkItem:
    pdf: str
    page: int
    files: tuple[str, ...]


def build_work_items(
    entries: list[logic.FileEntry],
    src_root: Path,
    page_count_fn=_default_page_count,
) -> tuple[list[WorkItem], list[str]]:
    """Group every consensus-flagged page by its physical (source PDF, page
    number), deduplicating split-document siblings that flag the same page.
    Returns (work items, files whose source PDF could not be resolved).

    A resolved path is only trusted once its own page count covers every
    flagged page number -- some of `resolve_pdf_path`'s fallback matches
    (e.g. a corpus-only "(2)" disambiguation suffix stripped back to a bare
    title) can land on a genuinely different document that happens to share
    the exact same title. A too-short PDF is the cheapest signal that a
    fallback found the wrong file; treat it exactly like an unresolved file
    rather than risk staging OCR text from the wrong document."""
    grouped: dict[tuple[str, int], list[str]] = {}
    unresolved: list[str] = []

    for entry in entries:
        pdf_path = resolve_pdf_path(src_root, entry.file)
        if pdf_path is None:
            unresolved.append(entry.file)
            continue
        page_nums = [n for p in entry.pages if (n := page_label_to_int(p.page)) is not None]
        if not page_nums:
            continue
        try:
            total_pages = page_count_fn(pdf_path)
        except Exception:
            unresolved.append(entry.file)
            continue
        if max(page_nums) > total_pages:
            unresolved.append(entry.file)
            continue
        for page_num in page_nums:
            grouped.setdefault((str(pdf_path), page_num), []).append(entry.file)

    items = [
        WorkItem(pdf=pdf, page=page, files=tuple(sorted(set(files))))
        for (pdf, page), files in sorted(grouped.items())
    ]
    return items, unresolved


@dataclass(frozen=True)
class PageResult:
    pdf: str
    page: int
    files: tuple[str, ...]
    new_text: str
    timestamp: str


def load_done_keys(staging_file: Path) -> set[tuple[str, int]]:
    if not staging_file.exists():
        return set()
    done = set()
    for line in staging_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        done.add((d["pdf"], d["page"]))
    return done


def append_result(staging_file: Path, result: PageResult) -> None:
    staging_file.parent.mkdir(parents=True, exist_ok=True)
    with staging_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def ocr_page(pdf_path: Path, page_num: int, tmp_dir: Path) -> str:
    from pdf2image import convert_from_path

    images = convert_from_path(
        str(pdf_path), poppler_path=POPPLER_PATH, dpi=PDF_DPI,
        first_page=page_num, last_page=page_num,
    )
    tmp_img = tmp_dir / f"_tmp_reocr_page_{page_num}.png"
    images[0].save(tmp_img, "PNG")
    try:
        return ocr_image(str(tmp_img))
    finally:
        tmp_img.unlink(missing_ok=True)


def main() -> None:
    entries = logic.parse_consensus_priority(CONSENSUS_FILE)
    items, unresolved = build_work_items(entries, DEFAULT_SRC_ROOT)

    if unresolved:
        UNRESOLVED_FILE.parent.mkdir(parents=True, exist_ok=True)
        UNRESOLVED_FILE.write_text("\n".join(unresolved) + "\n", encoding="utf-8")
        print(f"[SKIP] {len(unresolved)} file(s) with no resolvable source PDF -> {UNRESOLVED_FILE}")

    done = load_done_keys(STAGING_FILE)
    todo = [item for item in items if (item.pdf, item.page) not in done]
    print(f"[INFO] {len(items)} unique (pdf, page) pairs, {len(done)} already staged, {len(todo)} remaining")

    for i, item in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {Path(item.pdf).name} -- page {item.page} ({len(item.files)} file(s))")
        try:
            new_text = ocr_page(Path(item.pdf), item.page, STAGING_FILE.parent)
        except Exception as ex:
            print(f"   [ERROR] {ex}")
            continue

        result = PageResult(
            pdf=item.pdf,
            page=item.page,
            files=item.files,
            new_text=new_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        append_result(STAGING_FILE, result)

    print("[FINISH]")


if __name__ == "__main__":
    main()
