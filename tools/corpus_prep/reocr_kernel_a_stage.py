"""Phase 1 (kernel A) of the re-OCR remediation pipeline: extends
`reocr_consensus_pages.py`'s Phase 1 to the "kernel A" population -- pages
flagged by exactly one of {phi4:latest, gemma4:e4b} whose file was never
touched by the original consensus-AND-gate pipeline at all (1,329 files /
2,101 unique (pdf, page) work items after PDF-path resolution + sibling
dedup). See docs/llm-ocr-scan-log.md §7-§8 for how this population was
sized and sampled before committing to a full run (n=100 span-confirmed
true-positive rate: 56%).

Writes to the SAME `reocr_pages_staging.jsonl` used by the original
pipeline (fully resumable, same (pdf, page) key space, no collision risk
since kernel-A files were never touched before) so that `reocr_adjudicate.py`
and `reocr_apply.py` need no changes at all -- they already process
"every staged page not yet adjudicated/applied", so once staged here,
kernel-A pages flow through Phase 2 and Phase 3 exactly like the original
872-page batch.

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/reocr_kernel_a_stage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "consensus_review"))
import logic  # noqa: E402
from reocr_consensus_pages import (  # noqa: E402
    DEFAULT_SRC_ROOT,
    STAGING_FILE,
    PageResult,
    append_result,
    build_work_items,
    load_done_keys,
    load_manual_pdf_overrides,
    ocr_page,
)
from datetime import datetime, timezone  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
SCAN_DIR = CORPUS_ROOT / "llm_ocr_scan"
ADJUDICATION_FILE = SCAN_DIR / "reocr_adjudication.jsonl"
ITEMS_CACHE_FILE = SCAN_DIR / "reocr_kernel_a_items_cache.json"
YEARS = ["2564", "2565", "2566", "2567", "2568", "2569"]


def load_flagged(model_suffix: str) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    for year in YEARS:
        path = SCAN_DIR / f"full_{year}__{model_suffix}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["flag"]:
                files.setdefault(r["file"], []).append(r["page"])
    return files


def load_kernel_a_items() -> list:
    if ITEMS_CACHE_FILE.exists():
        cached = json.loads(ITEMS_CACHE_FILE.read_text(encoding="utf-8"))
        items = [
            type("Item", (), {"pdf": d["pdf"], "page": d["page"], "files": tuple(d["files"])})()
            for d in cached
        ]
        print(f"[INFO] loaded {len(items)} cached kernel-A work items from {ITEMS_CACHE_FILE.name}")
        return items

    phi4_files = load_flagged("phi4_latest")
    gemma_files = load_flagged("gemma4_e4b")

    phi4_pages = {(f, p) for f, ps in phi4_files.items() for p in ps}
    gemma_pages = {(f, p) for f, ps in gemma_files.items() for p in ps}
    phi4_only_pages = phi4_pages - gemma_pages
    gemma_only_pages = gemma_pages - phi4_pages
    phi4_only_files = {f for f, p in phi4_only_pages}
    gemma_only_files = {f for f, p in gemma_only_pages}

    # Kernel-A membership must be decided against a frozen snapshot of "files
    # touched before kernel-A work began" -- NOT the live ADJUDICATION_FILE,
    # which kernel-A processing itself writes into. Checking the live file
    # would make a file's own not-yet-processed pages disappear from the pool
    # the moment any ONE of its pages gets adjudicated (self-shrinking bug,
    # found 2026-07-26: merging the 100-sample sizing results into the real
    # adjudication file before this ran had already dropped 334 real kernel-A
    # pages across 102 files that way). Sample-sizing keys are excluded from
    # the snapshot since they were merged in from a scoped kernel-A sample,
    # not the original pre-kernel-A batch.
    sample_result_file = SCAN_DIR / "sample_kernelA_result.jsonl"
    sample_keys: set[tuple[str, int]] = set()
    if sample_result_file.exists():
        for line in sample_result_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                sample_keys.add((d["pdf"], d["page"]))

    adjudicated_files: set[str] = set()
    for line in ADJUDICATION_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if (r["pdf"], r["page"]) in sample_keys:
            continue
        for f in r.get("files", []):
            adjudicated_files.add(f)

    never_touched_phi4 = phi4_only_files - adjudicated_files
    never_touched_gemma = gemma_only_files - adjudicated_files

    pool_phi4 = [(f, p, "phi4:latest") for f, p in phi4_only_pages if f in never_touched_phi4]
    pool_gemma = [(f, p, "gemma4:e4b") for f, p in gemma_only_pages if f in never_touched_gemma]
    pool = pool_phi4 + pool_gemma
    print(f"[INFO] kernel-A pool: {len(pool)} (file, page) pairs "
          f"({len(pool_phi4)} phi4-only, {len(pool_gemma)} gemma-only)")

    by_year_file: dict[str, dict[str, list[logic.PageEntry]]] = {}
    for f, page_label, flagger in pool:
        year = f.split("\\")[0]
        by_year_file.setdefault(year, {}).setdefault(f, []).append(
            logic.PageEntry(page=page_label, models={flagger: logic.ModelFlag(reason="")})
        )

    entries = [
        logic.FileEntry(year=year, file=f, pages=pages)
        for year, files in by_year_file.items()
        for f, pages in files.items()
    ]

    overrides = load_manual_pdf_overrides()
    items, unresolved = build_work_items(entries, DEFAULT_SRC_ROOT, overrides=overrides)
    print(f"[INFO] resolved to {len(items)} unique (pdf, page) work items "
          f"({len(unresolved)} unresolved files skipped)")

    ITEMS_CACHE_FILE.write_text(json.dumps([
        {"pdf": it.pdf, "page": it.page, "files": list(it.files)} for it in items
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] cached {len(items)} work items to {ITEMS_CACHE_FILE.name}")
    return items


def main() -> None:
    items = load_kernel_a_items()
    done = load_done_keys(STAGING_FILE)
    todo = [item for item in items if (item.pdf, item.page) not in done]
    print(f"[INFO] {len(items)} total, {len(done)} already staged (incl. from real pipeline + kernel-A), "
          f"{len(todo)} remaining")

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
