"""Extra independent re-OCR + adjudication round for pages whose Phase 2
verdict wasn't unanimous "new" and where the free span-check (see
`sample_kernel_a_check.py`) gave no clean evidence either way -- see
docs/llm-ocr-scan-log.md addendum "Kernel-A review queue: tie-break rounds"
(2026-07-27).

Round 2 (temperature=0.0, identical to Phase 1's call) was run first on the
premise that re-OCR is non-deterministic in practice. That premise was
FALSIFIED: round 2 reproduced Phase 1's OCR text byte-for-byte and the
adjudicator's per-model verdicts byte-for-byte, for all 137/137 pool pages.
`ocr_image` at temperature=0.0 is fully deterministic for this pipeline --
the original corpus text must differ from Phase-1 re-OCR text for some
other reason (script/prompt/model-pull drift since original ingestion),
not run-to-run noise. Re-running the identical call again (a literal
"round 3") would add nothing.

Round 3 therefore perturbs the input instead of repeating it exactly:
`--temperature` (default 0.3, matching the retry-temperature convention
already used in `reocr_adjudicate.call_compare_model`) is passed through to
`ocr_image` so the sample is genuinely independent this time.

Reuses `reocr_consensus_pages.ocr_page` and `reocr_adjudicate`'s
`call_compare_model` / `resolve_old_text` unmodified. Writes to
ROUND-specific staging/adjudication files (never touches the real
`reocr_pages_staging.jsonl` / `reocr_adjudication.jsonl`) so this is fully
separate from and safe to run alongside the main pipeline.

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/reocr_tiebreak_round.py --round 3 --pool <pool.json> --phase ocr --temperature 0.3
    .venv/Scripts/python.exe tools/corpus_prep/reocr_tiebreak_round.py --round 3 --pool <pool.json> --phase adjudicate
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reocr_consensus_pages import ocr_page  # noqa: E402
from reocr_adjudicate import MODELS, call_compare_model, resolve_old_text, StagedPage  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
SCAN_DIR = CORPUS_ROOT / "llm_ocr_scan"


def staging_file(round_n: int) -> Path:
    return SCAN_DIR / f"reocr_tiebreak_r{round_n}_staging.jsonl"


def adjudication_file(round_n: int) -> Path:
    return SCAN_DIR / f"reocr_tiebreak_r{round_n}_adjudication.jsonl"


def load_pool(pool_file: Path) -> list[dict]:
    return json.loads(pool_file.read_text(encoding="utf-8"))


def load_done_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        done.add((d["pdf"], d["page"]))
    return done


def run_ocr_stage(round_n: int, pool: list[dict], temperature: float = 0.0, dpi: int | None = None) -> None:
    out = staging_file(round_n)
    done = load_done_keys(out)
    todo = [item for item in pool if (item["pdf"], item["page"]) not in done]
    print(f"[OCR round {round_n}] temperature={temperature} dpi={dpi} {len(pool)} total, {len(done)} already staged, {len(todo)} remaining")

    for i, item in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {Path(item['pdf']).name} -- page {item['page']}")
        try:
            kwargs = {"temperature": temperature}
            if dpi is not None:
                kwargs["dpi"] = dpi
            new_text = ocr_page(Path(item["pdf"]), item["page"], out.parent, **kwargs)
        except Exception as ex:
            print(f"   [ERROR] {ex}")
            continue
        record = {
            "pdf": item["pdf"], "page": item["page"], "files": item["files"],
            "new_text": new_text, "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[OCR round {round_n}] [FINISH]")


def run_adjudicate_stage(round_n: int) -> None:
    staged_records = [json.loads(l) for l in staging_file(round_n).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = adjudication_file(round_n)
    done = load_done_keys(out)
    todo = [r for r in staged_records if (r["pdf"], r["page"]) not in done]
    print(f"[ADJUDICATE round {round_n}] {len(staged_records)} staged, {len(done)} already adjudicated, {len(todo)} remaining")

    for i, rec in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {Path(rec['pdf']).name} -- page {rec['page']}")
        staged = StagedPage(pdf=rec["pdf"], page=rec["page"], files=tuple(rec["files"]),
                             new_text=rec["new_text"], timestamp=rec["timestamp"])
        old_text, diverging = resolve_old_text(CORPUS_ROOT, staged)
        if old_text is None:
            print("   [SKIP] old text not found in corpus")
            continue
        verdicts = {model: call_compare_model(model, old_text, staged.new_text) for model in MODELS}
        result = {
            "pdf": rec["pdf"], "page": rec["page"], "files": rec["files"],
            "diverging_siblings": diverging, "verdicts": verdicts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[ADJUDICATE round {round_n}] [FINISH]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, choices=[2, 3])
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--phase", choices=["ocr", "adjudicate"], required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dpi", type=int, default=None)
    args = parser.parse_args()

    pool = load_pool(args.pool)
    if args.phase == "ocr":
        run_ocr_stage(args.round, pool, temperature=args.temperature, dpi=args.dpi)
    else:
        run_adjudicate_stage(args.round)


if __name__ == "__main__":
    main()
