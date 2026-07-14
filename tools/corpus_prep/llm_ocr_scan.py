# -*- coding: utf-8 -*-
"""Experiment: can a local LLM catch OCR corruption that scan_ocr_repetition.py
structurally cannot see?

scan_ocr_repetition.py only catches one *shape* of defect: a token/phrase/line
repeated many times in a row (the OCR model got "stuck" looping). It is blind
by construction to a garbled sentence or paragraph that occurs only *once* --
misread characters, mid-word breaks, non-sequitur insertions -- since there is
no repetition for a regex to key on. This script asks 3 local Ollama models
(phi4-mini, phi4, gemma4:e4b -- already pulled) to read each page of a
document and flag exactly that: prose that reads as corrupted/incoherent,
NOT repetition (a separate, already-solved problem).

Two file sets, two different jobs:

1. **Floor sanity check** -- the .pre_reocr.bak/.corrupted_ocr.bak files
   (pre-fix copies of documents scan_ocr_repetition.py found, kept as backups
   after the 2026-07 re-OCR). These are known-corrupted, but with the
   *repetition* shape -- so "does the LLM flag them" is only a floor ("if it
   can't even catch this, the approach is dead"), not proof it catches the
   non-repetitive class. Do not read recall on this set as the headline
   result.

2. **The actual experiment** -- a random sample of regex-clean files (every
   file in the live corpus passes scan_ocr_repetition.py as of 2026-07-11).
   Every flag here is either a false positive or a genuine novel-class find;
   this is the list meant for human review, same spirit as
   academic_resolutions/ocr_repetition_review.md but for a defect class that
   script cannot see.

Read-only: never edits or deletes a corpus file. Writes results to
academic_resolutions/llm_ocr_scan/ (gitignored, same as the rest of the
corpus) as it goes, so a crash mid-run doesn't lose progress.

Run from the repo root, e.g.:

    python tools/corpus_prep/llm_ocr_scan.py --floor
    python tools/corpus_prep/llm_ocr_scan.py --sample 40 --seed 1
    python tools/corpus_prep/llm_ocr_scan.py --report
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import ollama

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "academic_resolutions"
OUT_DIR = ROOT / "llm_ocr_scan"
REPORT_FILE = ROOT / "llm_ocr_review.md"

# phi4-mini retired 2026-07-12: on the 143-page clean sample it flagged
# 143/143 (100%) -- a constant-true classifier, zero discriminative signal,
# contributes nothing to any downstream vote. See docs/ocr-corruption-detection-strategies.md.
MODELS = ["phi4:latest", "gemma4:e4b"]

# phi4's JSON (span+reason) is more verbose than gemma's; 800 truncated it
# mid-generation on ~6% of floor-set calls (`Unterminated string` errors).
NUM_PREDICT = {"phi4:latest": 1500}
DEFAULT_NUM_PREDICT = 800

PAGE_HEADER = re.compile(r"^## Page \d+\s*$", re.M)
PAGE_CHAR_BUDGET = 6000  # further split an oversized page so it fits num_ctx comfortably

SCHEMA = {
    "type": "object",
    "properties": {
        "flag": {"type": "boolean"},
        "span": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["flag", "span", "reason"],
}

PROMPT_TEMPLATE = (
    "You are checking OCR output (Thai academic-committee meeting resolutions) "
    "for transcription errors.\n\n"
    "Flag ONLY incoherent or garbled prose: misread characters, words broken "
    "mid-way into nonsense, a sentence that doesn't parse grammatically, or "
    "text that reads as unrelated noise inserted into the middle of a "
    "sentence. This is text corrupted by a bad OCR read, not a style issue.\n\n"
    "Do NOT flag: repeated words/phrases/lines (a separate tool already "
    "checks for that -- ignore repetition entirely), table/checkbox "
    "formatting oddities, markdown artifacts (##, ---, <table> tags), or "
    "text that is merely informal/abbreviated Thai academic language.\n\n"
    "If nothing is wrong, set flag=false and leave span/reason empty.\n\n"
    "Respond with JSON only: {\"flag\": bool, \"span\": \"the exact corrupted "
    "text, verbatim, <=200 chars\", \"reason\": \"one short sentence\"}.\n\n"
    "--- PAGE TEXT ---\n%%TEXT%%\n--- END PAGE TEXT ---"
)


def build_prompt(text: str) -> str:
    return PROMPT_TEMPLATE.replace("%%TEXT%%", text)


def split_pages(text: str) -> list[tuple[str, str]]:
    """Return (label, chunk_text) pairs. Splits on '## Page N' headers (the
    ocr_pdf_to_md.py convention); falls back to fixed-size slices for files
    with no page markers or a single oversized page."""
    headers = list(PAGE_HEADER.finditer(text))
    if not headers:
        chunks = [text[i : i + PAGE_CHAR_BUDGET] for i in range(0, len(text), PAGE_CHAR_BUDGET)]
        return [(f"chunk{i+1}", c) for i, c in enumerate(chunks) if c.strip()]

    pages: list[tuple[str, str]] = []
    for i, m in enumerate(headers):
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        label = m.group().strip("# ").strip()
        body = text[start:end].strip()
        if not body:
            continue
        if len(body) <= PAGE_CHAR_BUDGET:
            pages.append((label, body))
        else:
            sub_chunks = [body[i : i + PAGE_CHAR_BUDGET] for i in range(0, len(body), PAGE_CHAR_BUDGET)]
            for j, c in enumerate(sub_chunks):
                if c.strip():
                    pages.append((f"{label}.{j+1}", c))
    return pages


def call_model(model: str, text: str, retries: int = 2) -> dict:
    last_err = ""
    num_predict = NUM_PREDICT.get(model, DEFAULT_NUM_PREDICT)
    start = time.monotonic()
    for attempt in range(1, retries + 1):
        try:
            # temperature=0 makes a retry after a truncated/malformed response
            # deterministic and pointless (same failure every time) -- nudge
            # it on retry so there's a chance of a shorter/different output.
            temperature = 0.0 if attempt == 1 else 0.3
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": build_prompt(text)}],
                format=SCHEMA,
                think=False,
                options={"temperature": temperature, "num_ctx": 8192, "num_predict": num_predict},
            )
            content = response["message"]["content"]
            parsed = json.loads(content)
            return {
                "flag": bool(parsed.get("flag", False)),
                "span": str(parsed.get("span", ""))[:300],
                "reason": str(parsed.get("reason", ""))[:300],
                "error": None,
                "elapsed_s": round(time.monotonic() - start, 2),
            }
        except Exception as e:  # malformed JSON, model error, timeout, etc.
            last_err = str(e)
            time.sleep(1)
    return {"flag": False, "span": "", "reason": "", "error": last_err, "elapsed_s": round(time.monotonic() - start, 2)}


SPLIT_PIECE = re.compile(r"__\d+$")
BAK_SUFFIXES = (".pre_reocr.bak", ".corrupted_ocr.bak")


def _source_key(path: Path) -> tuple[Path, str, bool]:
    """Collapse a backup filename back to (parent, base document stem,
    is_split_piece) -- so curriculum-split siblings of the same source
    document (each backed up separately) count as ONE floor-set entry, not
    N. Backup naming seen in the corpus: '<stem>.md.dup.pre_reocr.bak' (the
    pre-split whole document) and '<stem>__3.md.pre_reocr.bak' (one split
    piece) both reduce to the same (parent, stem)."""
    stem = path.name
    for suffix in BAK_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if stem.endswith(".dup"):
        stem = stem[: -len(".dup")]
    if stem.endswith(".md"):
        stem = stem[: -len(".md")]
    is_split = bool(SPLIT_PIECE.search(stem))
    stem = SPLIT_PIECE.sub("", stem)
    return (path.parent, stem, is_split)


def bak_files() -> list[Path]:
    """One representative backup file per corrupted *source document* --
    excludes '_LINK.txt' sidecars (URL-only, no document text) and collapses
    curriculum-split pieces of the same source down to one entry (preferring
    the whole pre-split document when both exist, since it contains every
    piece's text).

    Note: this resolves to ~121 source documents, not the ~26 you'd count
    with a naive `find -iname` in Git Bash on Windows -- that undercounts
    badly on these Thai filenames (Unicode/codepage handling issue in that
    environment), Path.rglob() here does not. 121 is close to the ~112
    documents memory records as originally found by scan_ocr_repetition.py,
    which is the expected ballpark. Use --floor-sample to keep the sanity
    check small rather than re-scanning all of them."""
    candidates = [
        f
        for pattern in BAK_SUFFIXES
        for f in ROOT.rglob(f"*{pattern}")
        if "_LINK.txt" not in f.name
    ]
    groups: dict[tuple[Path, str], list[tuple[bool, Path]]] = {}
    for f in candidates:
        parent, stem, is_split = _source_key(f)
        groups.setdefault((parent, stem), []).append((is_split, f))

    reps = []
    for (parent, stem), entries in groups.items():
        entries.sort(key=lambda e: (e[0], str(e[1])))  # whole docs (is_split=False) first
        reps.append(entries[0][1])
    return sorted(reps)


def clean_sample(n: int, seed: int) -> list[Path]:
    """Random sample of live, never-flagged .md files. As of 2026-07-11 the
    whole corpus passes scan_ocr_repetition.py (0 candidates), so no explicit
    exclusion list is needed beyond .dup/.bak siblings -- but re-run
    scan_ocr_repetition.py first if it's been a while, and pass its flagged
    stems via --exclude-review to be safe."""
    all_files = [
        f for f in ROOT.rglob("*.md") if not f.name.endswith(".dup")
    ]
    rng = random.Random(seed)
    rng.shuffle(all_files)
    return all_files[:n]


def year_files(year: str) -> list[Path]:
    """Every live .md file under academic_resolutions/<year>/, sorted for a
    deterministic, resumable full-corpus production scan (no sampling -- the
    corpus is batched by year, not by row count, per the union architecture
    decision in docs/ocr-corruption-detection-strategies.md §8)."""
    year_dir = ROOT / year
    if not year_dir.is_dir():
        raise SystemExit(f"no such year folder: {year_dir}")
    return sorted(f for f in year_dir.rglob("*.md") if not f.name.endswith(".dup"))


def out_path(model: str, run: str) -> Path:
    safe_model = model.replace(":", "_").replace("/", "_")
    return OUT_DIR / f"{run}__{safe_model}.jsonl"


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["key"])
        except Exception:
            continue
    return done


def run_scan(files: list[Path], run: str, models: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in models:
        path = out_path(model, run)
        done = already_done(path)
        print(f"\n=== {model} ({run}) -- {len(files)} file(s), {len(done)} chunk(s) already done ===")
        with path.open("a", encoding="utf-8") as out:
            for fi, f in enumerate(files):
                try:
                    text = f.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    text = f.read_text(encoding="utf-8-sig")
                rel = str(f.relative_to(ROOT))
                pages = split_pages(text)
                for label, chunk in pages:
                    key = f"{rel}::{label}"
                    if key in done:
                        continue
                    result = call_model(model, chunk)
                    record = {"key": key, "file": rel, "page": label, "model": model, **result}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    if result["flag"]:
                        print(f"  [FLAG] {rel} [{label}] -- {result['reason']}")
                print(f"  ({fi+1}/{len(files)}) {rel} -- {len(pages)} page(s)", end="\r")
        print()


def load_results(run: str) -> list[dict]:
    records = []
    for model in MODELS:
        path = out_path(model, run)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def write_report() -> None:
    floor = load_results("floor")
    sample = load_results("sample")

    lines = [
        "# LLM OCR-corruption scan (experiment)",
        "",
        "Auto-generated by `tools/corpus_prep/llm_ocr_scan.py`. Read-only -- "
        "nothing here has been edited or deleted. Looks for *non-repetitive* "
        "OCR corruption (garbled/incoherent prose) -- the class "
        "`scan_ocr_repetition.py` cannot see by construction.",
        "",
        "## 1. Floor sanity check (known-corrupted `.bak` files)",
        "",
        "Not the headline result -- these are corrupted with the repetition "
        "shape the regex scanner already catches, so this only checks each "
        "model isn't blind to an obvious case.",
        "",
    ]

    bak_stems = {r["file"] for r in floor}
    for model in MODELS:
        model_records = [r for r in floor if r["model"] == model]
        if not model_records:
            continue
        flagged_files = {r["file"] for r in model_records if r["flag"]}
        total_files = {r["file"] for r in model_records}
        errors = sum(1 for r in model_records if r.get("error"))
        lines.append(
            f"- **{model}**: flagged {len(flagged_files)}/{len(total_files)} known-bad files"
            + (f" ({errors} call errors)" if errors else "")
        )
    lines.append("")

    lines += [
        "## 2. Novel-class candidates (regex-clean sample)",
        "",
        "Every flag below is on a file `scan_ocr_repetition.py` already "
        "passed as clean -- so it's either a false positive or a genuine "
        "find. Read the span before trusting it.",
        "",
    ]

    sample_files = sorted({r["file"] for r in sample})
    flags_by_file: dict[str, list[dict]] = {}
    for r in sample:
        if r["flag"]:
            flags_by_file.setdefault(r["file"], []).append(r)

    lines.append(f"Scanned {len(sample_files)} regex-clean file(s) across {len(MODELS)} model(s).")
    lines.append(f"{len(flags_by_file)} file(s) got at least one flag from at least one model:")
    lines.append("")

    for file in sorted(flags_by_file):
        lines.append(f"### `{file}`")
        for r in flags_by_file[file]:
            lines.append(f"- **[{r['model']}]** page `{r['page']}` -- {r['reason']}")
            lines.append(f"  > {r['span']}")
        lines.append("")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {REPORT_FILE}")


def write_year_review(year: str) -> None:
    """Full-corpus production scan, one file per year (batched per the union
    architecture decision -- see docs/ocr-corruption-detection-strategies.md
    §8: gemma4:e4b alone misses ~43% of files phi4 catches, so both models
    run over every file, not just a sample)."""
    records = load_results(f"full_{year}")
    files = sorted({r["file"] for r in records})
    flags_by_file: dict[str, list[dict]] = {}
    for r in records:
        if r["flag"]:
            flags_by_file.setdefault(r["file"], []).append(r)

    lines = [
        f"# LLM OCR-corruption scan -- {year} (production)",
        "",
        "Auto-generated by `tools/corpus_prep/llm_ocr_scan.py --year`. Read-only. "
        "Union of phi4:latest + gemma4:e4b (phi4-mini retired -- see "
        "docs/ocr-corruption-detection-strategies.md). Every flag is either a "
        "false positive or a genuine non-repetitive OCR defect; read the span "
        "before trusting it -- known false-positive patterns: standalone "
        "course codes, \"PREREQUISITE: NONE\"-style boilerplate, coherent "
        "citation lists, eccentric-but-real elective course names.",
        "",
        f"Scanned {len(files)} file(s) across {len(MODELS)} model(s).",
        f"{len(flags_by_file)} file(s) got at least one flag from at least one model:",
        "",
    ]
    for file in sorted(flags_by_file):
        lines.append(f"### `{file}`")
        for r in flags_by_file[file]:
            lines.append(f"- **[{r['model']}]** page `{r['page']}` -- {r['reason']}")
            lines.append(f"  > {r['span']}")
        lines.append("")

    out = OUT_DIR / f"full_review_{year}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", action="store_true", help="Run the .bak floor sanity check")
    parser.add_argument(
        "--floor-sample",
        type=int,
        default=20,
        help="Cap the floor set to N random known-bad source docs (there are "
        "~121 after dedup, not 26 -- see bak_files() docstring; this is meant "
        "to stay a small sanity check, not an exhaustive re-scan)",
    )
    parser.add_argument("--floor-all", action="store_true", help="Use all deduped .bak source docs, ignore --floor-sample")
    parser.add_argument("--sample", type=int, default=0, help="Run N random regex-clean files")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--year",
        nargs="*",
        default=[],
        help="Full-corpus production scan, batched by year folder (e.g. "
        "--year 2564 2565) -- every live file, both models, no sampling. "
        "Resumable per-year jsonl; writes academic_resolutions/llm_ocr_scan/"
        "full_review_<year>.md after each year.",
    )
    parser.add_argument("--models", nargs="*", default=MODELS)
    parser.add_argument("--limit", type=int, default=0, help="Cap file count (testing)")
    parser.add_argument("--report", action="store_true", help="(Re)generate the markdown report only")
    args = parser.parse_args()

    if args.report and not args.floor and not args.sample and not args.year:
        write_report()
        return

    if args.floor:
        files = bak_files()
        if not args.floor_all:
            rng = random.Random(args.seed)
            rng.shuffle(files)
            files = files[: args.floor_sample]
        if args.limit:
            files = files[: args.limit]
        run_scan(files, "floor", args.models)

    if args.sample:
        files = clean_sample(args.sample, args.seed)
        if args.limit:
            files = files[: args.limit]
        run_scan(files, "sample", args.models)

    for year in args.year:
        files = year_files(year)
        if args.limit:
            files = files[: args.limit]
        run_scan(files, f"full_{year}", args.models)
        write_year_review(year)

    if not args.year:
        write_report()


if __name__ == "__main__":
    main()
