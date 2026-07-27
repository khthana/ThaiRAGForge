"""One-off sizing script (not part of the production pipeline): randomly sample
100 (pdf, page) pairs from the "kernel A" pool -- pages flagged by exactly one
of {phi4:latest, gemma4:e4b} whose file was never touched by the existing
consensus-AND-gate remediation pipeline at all -- and run them through the
same Phase 1 (fresh re-OCR) + Phase 2 (dual-model old-vs-new adjudication)
logic the production pipeline uses, writing to separate sample-only output
files so the real staging/adjudication jsonls are untouched.

Purpose: estimate what fraction of phi4-only-flagged pages are genuine OCR
defects, before committing to a ~16-hour full run over all ~2,684 kernel-A
pages. See docs/llm-ocr-scan-log.md §7-§8.

**Two things fixed 2026-07-26 after the first 35 samples**:
1. Split into two phases (`--phase=ocr` / `--phase=adjudicate`) matching how
   the production pipeline actually runs -- interleaving one OCR call + two
   adjudicate calls per item forces 3 model evictions per item on a 12GB
   GPU, which is why throughput collapsed after the first chunk. Running all
   OCR first, then all adjudication, needs only 3 model loads total.
2. The raw new/new verdict overstates the true defect-fix rate, because the
   adjudicator compares whole pages while phi4 flags one `span` within the
   page -- a page can read "better overall" without the flagged span ever
   being touched. `--phase=report` cross-checks each result's flagged span
   (already in the raw scan jsonl, no extra LLM calls) against the fresh
   OCR text and reports the span-confirmed rate alongside raw verdicts.

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/sample_kernel_a_check.py --phase=ocr
    .venv/Scripts/python.exe tools/corpus_prep/sample_kernel_a_check.py --phase=adjudicate
    .venv/Scripts/python.exe tools/corpus_prep/sample_kernel_a_check.py --phase=report
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "consensus_review"))
import logic  # noqa: E402
from reocr_consensus_pages import (  # noqa: E402
    DEFAULT_SRC_ROOT,
    build_work_items,
    load_manual_pdf_overrides,
    ocr_page,
)
from reocr_adjudicate import call_compare_model, load_full_page_text  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
SCAN_DIR = CORPUS_ROOT / "llm_ocr_scan"
ADJUDICATION_FILE = SCAN_DIR / "reocr_adjudication.jsonl"
SAMPLE_STAGING_FILE = SCAN_DIR / "sample_kernelA_staging.jsonl"
SAMPLE_RESULT_FILE = SCAN_DIR / "sample_kernelA_result.jsonl"
SAMPLE_CACHE_FILE = SCAN_DIR / "sample_kernelA_items_cache.json"
YEARS = ["2564", "2565", "2566", "2567", "2568", "2569"]
SAMPLE_SIZE = 100
SEED = 42


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


def load_sample() -> list:
    if SAMPLE_CACHE_FILE.exists():
        cached = json.loads(SAMPLE_CACHE_FILE.read_text(encoding="utf-8"))
        sample = [
            type("Item", (), {"pdf": d["pdf"], "page": d["page"], "files": tuple(d["files"])})()
            for d in cached
        ]
        print(f"[INFO] loaded {len(sample)} cached sampled work items from {SAMPLE_CACHE_FILE.name}")
    else:
        phi4_files = load_flagged("phi4_latest")
        gemma_files = load_flagged("gemma4_e4b")

        phi4_pages = {(f, p) for f, ps in phi4_files.items() for p in ps}
        gemma_pages = {(f, p) for f, ps in gemma_files.items() for p in ps}
        phi4_only_pages = phi4_pages - gemma_pages
        gemma_only_pages = gemma_pages - phi4_pages
        phi4_only_files = {f for f, p in phi4_only_pages}
        gemma_only_files = {f for f, p in gemma_only_pages}

        adjudicated_files: set[str] = set()
        for line in ADJUDICATION_FILE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
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

        random.seed(SEED)
        sample = random.sample(items, min(SAMPLE_SIZE, len(items)))
        print(f"[INFO] sampled {len(sample)} work items (seed={SEED})")

        SAMPLE_CACHE_FILE.write_text(json.dumps([
            {"pdf": it.pdf, "page": it.page, "files": list(it.files)} for it in sample
        ], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] cached sample to {SAMPLE_CACHE_FILE.name}")

    return sample


def load_staged_text() -> dict[tuple, str]:
    staged_text: dict[tuple, str] = {}
    if SAMPLE_STAGING_FILE.exists():
        for line in SAMPLE_STAGING_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                staged_text[(d["pdf"], d["page"])] = d["new_text"]
    return staged_text


def load_done_results() -> set:
    done_results = set()
    if SAMPLE_RESULT_FILE.exists():
        for line in SAMPLE_RESULT_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done_results.add((d["pdf"], d["page"]))
    return done_results


def run_ocr_phase(sample: list) -> None:
    staged_text = load_staged_text()
    for i, item in enumerate(sample, 1):
        key = (item.pdf, item.page)
        if key in staged_text:
            continue
        print(f"[OCR {i}/{len(sample)}] {Path(item.pdf).name} -- page {item.page}")
        try:
            new_text = ocr_page(Path(item.pdf), item.page, SAMPLE_STAGING_FILE.parent)
        except Exception as ex:
            print(f"   [ERROR OCR] {ex}")
            continue
        SAMPLE_STAGING_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SAMPLE_STAGING_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "pdf": item.pdf, "page": item.page, "files": list(item.files),
                "new_text": new_text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False) + "\n")
    print("[DONE OCR PHASE]")


def run_adjudicate_phase(sample: list) -> None:
    staged_text = load_staged_text()
    done_results = load_done_results()
    for i, item in enumerate(sample, 1):
        key = (item.pdf, item.page)
        if key in done_results:
            continue
        if key not in staged_text:
            print(f"[ADJ {i}/{len(sample)}] [SKIP] not yet OCR'd -- run --phase=ocr first")
            continue
        new_text = staged_text[key]
        print(f"[ADJ {i}/{len(sample)}] {Path(item.pdf).name} -- page {item.page}")

        old_text = None
        old_source = None
        for f in item.files:
            text = load_full_page_text(CORPUS_ROOT, f, item.page)
            if text is not None:
                old_text, old_source = text, f
                break

        if old_text is None:
            print("   [SKIP] could not resolve old text")
            continue

        verdicts = {model: call_compare_model(model, old_text, new_text) for model in ["phi4:latest", "gemma4:e4b"]}
        result = {
            "pdf": item.pdf, "page": item.page, "files": list(item.files),
            "old_text_source": old_source,
            "verdicts": verdicts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with SAMPLE_RESULT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"   verdicts: {verdicts.get('phi4:latest', {}).get('verdict')} / "
              f"{verdicts.get('gemma4:e4b', {}).get('verdict')}")
    print("[DONE ADJUDICATE PHASE]")


def build_span_lookup() -> dict:
    span_lookup: dict[tuple, list] = {}
    for year in YEARS:
        for suffix in ["phi4_latest", "gemma4_e4b"]:
            path = SCAN_DIR / f"full_{year}__{suffix}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["flag"]:
                    span_lookup.setdefault((r["file"], r["page"]), []).append((suffix, r.get("span", "")))
    return span_lookup


def run_report_phase() -> None:
    """Report both the raw verdict distribution and the span-confirmed rate.

    The raw new/new rate overstates the true defect-fix rate: the adjudicator
    compares whole pages, but phi4/gemma flag one span within the page, so a
    page can verdict "new" because the rest of it OCR'd cleaner, without the
    flagged span itself ever changing. Cross-checking the flagged span
    (already in the scan jsonl) against the fresh OCR text is a free way to
    tell the two apart -- see docs/llm-ocr-scan-log.md §8.
    """
    results = []
    if SAMPLE_RESULT_FILE.exists():
        results = [json.loads(l) for l in SAMPLE_RESULT_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    staged_text = load_staged_text()
    span_lookup = build_span_lookup()

    from collections import Counter
    verdict_counts = Counter()
    span_changed = 0
    span_unchanged = 0
    span_missing = 0
    new_new_span_changed = 0
    new_new_span_unchanged = 0

    for res in results:
        key = (res["pdf"], res["page"])
        new_text = staged_text.get(key, "")
        v_phi4 = res["verdicts"].get("phi4:latest", {}).get("verdict")
        v_gemma = res["verdicts"].get("gemma4:e4b", {}).get("verdict")
        verdict_counts[(v_phi4, v_gemma)] += 1

        found_spans = None
        for f in res["files"]:
            for (sf, sp), spans in span_lookup.items():
                if sf != f:
                    continue
                m = re.match(r"Page (\d+)", sp)
                if m and int(m.group(1)) == res["page"]:
                    found_spans = spans
                    break
            if found_spans:
                break

        if not found_spans:
            span_missing += 1
            continue

        any_changed = False
        for _model, span in found_spans:
            if not span:
                continue
            span_norm = "".join(span.split())
            new_norm = "".join(new_text.split())
            if span_norm and span_norm not in new_norm:
                any_changed = True
        if any_changed:
            span_changed += 1
            if v_phi4 == "new" and v_gemma == "new":
                new_new_span_changed += 1
        else:
            span_unchanged += 1
            if v_phi4 == "new" and v_gemma == "new":
                new_new_span_unchanged += 1

    print(f"[REPORT] total results: {len(results)}")
    print(f"[REPORT] raw verdict distribution: {dict(verdict_counts)}")
    both_new = verdict_counts[("new", "new")]
    print(f"[REPORT] (new,new): {both_new} ({both_new / len(results):.0%})" if results else "")
    checked = span_changed + span_unchanged
    print(f"[REPORT] span-confirmed changed: {span_changed}/{checked} "
          f"({span_changed / checked:.0%})" if checked else "[REPORT] no spans checked")
    if both_new:
        print(f"[REPORT] of (new,new) verdicts: span actually changed in "
              f"{new_new_span_changed}/{both_new} ({new_new_span_changed / both_new:.0%})")
    print(f"[REPORT] span not found for {span_missing} results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["ocr", "adjudicate", "report"], default="ocr")
    args = parser.parse_args()

    if args.phase == "report":
        run_report_phase()
        return

    sample = load_sample()
    if args.phase == "ocr":
        run_ocr_phase(sample)
    else:
        run_adjudicate_phase(sample)


if __name__ == "__main__":
    main()
