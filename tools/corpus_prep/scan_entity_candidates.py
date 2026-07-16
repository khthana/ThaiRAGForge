"""Scan the corpus body text for candidate entity names identified by a Thai
prefix -- e.g. คณะ/วิทยาลัย/สถาบัน/วิทยาเขต for organizational units, or หลักสูตร
for degree programs. Generic Thai NER (pythainlp/wangchanberta/phayathaibert)
does not reliably catch either category (confirmed: it shatters the
institution's own name into multiple ORG fragments) -- these are closed or
semi-closed domain vocabularies better handled as an explicit dictionary the
corpus itself is scanned to build, cross-checked by frequency so one-off OCR
misspellings sink to the bottom instead of polluting the list.

Read-only: never writes to the corpus. Prints a frequency-sorted candidate
report for human review -- the actual canonical dictionary (e.g.
data/entity_dictionaries/faculties.json) is a separate, hand-confirmed file
this script's output feeds into, not something it writes automatically.

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/scan_entity_candidates.py --prefix คณะ วิทยาลัย สถาบัน วิทยาเขต
    .venv/Scripts/python.exe tools/corpus_prep/scan_entity_candidates.py --prefix หลักสูตร --top 300
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"

# "มหาวิทยาลัย" (university) ends in "วิทยาลัย" (college) -- a prefix search for
# "วิทยาลัย" alone matches its tail ("มหาวิทยาลัยมหิดล" -> "วิทยาลัยมหิดล") unless
# excluded, which silently pulls in every OTHER university mentioned in the
# corpus (MOUs, curriculum benchmarking) as if it were an internal unit.
_NOT_PRECEDED_BY_MAHA = "(?<!มหา)"

# Trailing junk a naive \S+ boundary picks up: file extensions from the
# "# Document: ...pdf" header line, a stray closing backtick from Markdown
# link syntax, or leftover HTML table tags.
_NOISE_TAIL = re.compile(r"(\.pdf|\.md|\.docx|`|</?\w+.*|\)\s*</.*)$")

# "คณะกรรมการ" (committee) and "คณะอนุกรรมการ" (subcommittee) both start with
# "คณะ" but are never an organizational unit -- filtered separately from the
# noise-tail cleanup since they're valid, complete matches of a different
# meaning entirely, not corrupted ones.
_COMMITTEE_PREFIXES = ("คณะกรรมการ", "คณะอนุกรรมการ")


def build_prefix_pattern(prefixes: tuple[str, ...]) -> re.Pattern:
    alternation = "|".join(re.escape(p) for p in prefixes)
    # One whitespace-delimited token after the prefix. Thai has no spaces
    # *within* a name, so this correctly captures a single-word entity
    # whole -- but some real entity names are themselves multiple
    # space-separated words (e.g. "คณะสถาปัตยกรรม ศิลปะและการออกแบบ"), and
    # there is no reliable way to tell "next word is still part of the
    # name" from "next word is just the following sentence" without either
    # a dictionary of known names or per-entity boundary anchors (see
    # extract_program_candidates for the หลักสูตร case, which has enough
    # structure to anchor on). This function deliberately stays single-token
    # and simple: it's a candidate *generator* for human review, not a
    # boundary-perfect extractor -- multi-word names get confirmed by hand
    # once (see data/entity_dictionaries/faculties.json) rather than guessed
    # here on every scan.
    return re.compile(rf"{_NOT_PRECEDED_BY_MAHA}({alternation})(\S+)")


def extract_candidates(text: str, prefixes: tuple[str, ...]) -> list[str]:
    """Every candidate entity name found in `text`, in order, cleaned of
    trailing filename/markup noise. `text` should already have any leading
    "# Document: ..." header line stripped (see `scan_corpus`) -- that line
    is a filename, not prose, and its extension would itself match as a
    false "entity"."""
    pattern = build_prefix_pattern(prefixes)
    candidates = []
    for m in pattern.finditer(text):
        candidate = m.group(0)
        # stacked noise (e.g. ".md`") needs repeated stripping -- a single
        # sub() only removes the one alternative matching right at the
        # current end of string, not everything noise-shaped behind it.
        while True:
            stripped = _NOISE_TAIL.sub("", candidate)
            if stripped == candidate:
                break
            candidate = stripped
        candidate = candidate.rstrip("),.:;<>\"'/")
        if candidate.startswith(_COMMITTEE_PREFIXES):
            continue
        if 3 <= len(candidate) <= 60:
            candidates.append(candidate)
    return candidates


def scan_corpus(corpus_root: Path, prefixes: tuple[str, ...]) -> Counter:
    """Frequency count of every candidate across every non-`.dup` corpus
    file. `.dup` files are excluded corpus-wide (ADR-0004 convention, see
    llm_ocr_scan._source_key) -- they're not part of the live corpus."""
    counter: Counter = Counter()
    for f in corpus_root.rglob("*.md"):
        if f.name.endswith(".dup"):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")
        body = text.split("\n", 2)[-1] if text.startswith("# Document:") else text
        counter.update(extract_candidates(body, prefixes))
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", nargs="+", required=True, help="Thai prefix(es) to scan for")
    parser.add_argument("--top", type=int, default=150, help="How many top candidates to print")
    args = parser.parse_args()

    counter = scan_corpus(CORPUS_ROOT, tuple(args.prefix))
    print(f"[INFO] {len(counter)} unique candidates for prefixes {args.prefix}")
    for candidate, n in counter.most_common(args.top):
        print(f"{n}\t{candidate}")


if __name__ == "__main__":
    main()
