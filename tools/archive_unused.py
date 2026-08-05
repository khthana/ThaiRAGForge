"""Move superseded artifacts off the working drive, by category, with evidence.

The corpus and the built indices both accumulate archives: pre-split originals,
pre-re-OCR backups, superseded index combos, retired result sets. Some are
genuinely inert; some are still read by a tool that has simply not been run
lately. This script keeps that distinction explicit instead of leaving it to
whoever is looking at a directory listing.

Every category below records *why* its verdict is what it is, checked against
the code rather than assumed:

  SAFE   -- no code path reads these. `.dup` in particular is only ever used as
            an exclusion (`if name.endswith(".dup"): skip`) in
            `loaders/common.py`, `llm_ocr_scan.py`, `llm_thematic_bootstrap.py`,
            `excise_ocr_loops.py`, `canonicalize_people.py` -- never opened.
  GATED  -- something does read them, or they are the only copy of something.
            Each needs its own flag, and the reason is printed when you pass it.

Nothing is deleted: everything moves under `--dest`, preserving its relative
path, so ADR-0004's "recoverable, not deleted" property survives the move --
it just lives on the data drive with the raw scans instead of in the repo.

Run:
    python tools/archive_unused.py                      # dry run, all categories
    python tools/archive_unused.py --apply              # move the SAFE ones
    python tools/archive_unused.py --apply --reocr-baks --index-backups \
        --superseded-combos --retired-results           # include GATED ones
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(".").resolve()
CORPUS = Path("academic_resolutions")
DEFAULT_DEST = Path(r"D:/academic_resolutions (ข้อมูลดิบ + OCR)/_superseded_from_repo")

# The 8 combo dirs excluded from every current eval by
# embedder_matrix_9way.py::_EXCLUDED_COMBO_DIRS -- superseded 128-cap sct and
# rejected 510-cap congen variants, kept only so their labels stop colliding
# with the correct counterparts.
SUPERSEDED_COMBOS = [
    "plain__fixed_size__local__9d03b361", "plain__recursive__local__31293c05",
    "plain__sentence__local__d6c1f8e1", "plain__semantic__local__9576aa59",
    "plain__fixed_size__local__e6048946", "plain__recursive__local__4a350a4e",
    "plain__sentence__local__26622ae7", "plain__semantic__local__6b33a155",
]

# Result sets the invariant audit flags as older than the indices they name.
# `gold_full_embedder_matrix` is read by no script at all; the rest are read
# only by the pre-9-way scripts kept for reference.
RETIRED_RESULTS = [
    "gold_full_embedder_matrix", "silver_chunker_compare", "gold_chunker_compare",
    "gold_chunker_compare_73det", "gold_embedder_compare", "congen_sct_truncation_fix",
    "mode_b_routed",
]


@dataclass
class Category:
    key: str
    label: str
    verdict: str  # "SAFE" | "GATED"
    why: str
    paths: list[Path] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(
            sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir()
            else p.stat().st_size
            for p in self.paths
        )


def collect() -> list[Category]:
    def corpus_files(*suffixes: str) -> list[Path]:
        return sorted(
            p for p in CORPUS.rglob("*")
            if p.is_file() and any(p.name.endswith(s) for s in suffixes)
        )

    cats = [
        Category(
            "dup", "pre-split / pre-rename originals (*.md.dup, *.txt.dup, *.dup.superseded)",
            "SAFE",
            "every reference in the codebase skips these by name; none opens one. "
            "ADR-0004 archives a split bundle's original this way instead of deleting it",
            corpus_files(".md.dup", ".txt.dup", ".dup.superseded"),
        ),
        Category(
            "manual-fix", "manual-fix backups (*.pre_manual_fix.bak)", "SAFE",
            "no reference anywhere in src/, tools/, app/ or tests/",
            corpus_files(".pre_manual_fix.bak"),
        ),
        Category(
            "reocr-baks", "pre-re-OCR originals (*.pre_reocr.bak, *.corrupted_ocr.bak)",
            "GATED",
            "llm_ocr_scan.py reads these (BAK_SUFFIXES) for its floor sanity check -- "
            "comparing current text against the pre-re-OCR original. They are also the "
            "only copy of that original, and reocr_apply.py backs a file up only once, "
            "so a future re-OCR run would archive the *current* text instead. Moving "
            "them is fine while the remediation stays closed; re-running the scan's "
            "floor check afterwards means pointing it at the archive",
            corpus_files(".pre_reocr.bak", ".corrupted_ocr.bak"),
        ),
        Category(
            "index-backups", "resolution_id relabel backups (chunks.parquet.pre_relabel.bak)",
            "GATED",
            "written by relabel_index_resolution_ids.py today; its --results-only phase "
            "builds its exact old->new id map from them. Keep until the re-run of the "
            "eval suite has confirmed the relabel, then they are dead weight",
            sorted(Path("data/index").glob("*/*/chunks.parquet.pre_relabel.bak")),
        ),
        Category(
            "superseded-combos", "superseded index combos (8 dirs)", "GATED",
            "excluded from every current eval by _EXCLUDED_COMBO_DIRS, and they still "
            "hold 21 bogus resolutions from the pre-fix corpus walk. Inert, but they are "
            "the only record of the ConGen/SCT max_seq_length comparison's losing arm",
            [d for name in SUPERSEDED_COMBOS
             if (d := Path("data/index/chunker_compare_full") / name).exists()],
        ),
        Category(
            "retired-results", "retired result sets (7 dirs)", "GATED",
            "all older than the indices they name (invariant audit E4). "
            "gold_full_embedder_matrix is read by no script; the others only by the "
            "superseded pre-9-way eval scripts -- which would silently report pre-fix "
            "numbers if re-run against them",
            [d for name in RETIRED_RESULTS if (d := Path("data/results") / name).exists()],
        ),
        Category(
            "qdrant-demo", "Qdrant vertical-slice demo data (person_slice, 2026-07-16)", "SAFE",
            "tools/eval/build_qdrant_person_slice.py only writes this path, never reads "
            "it back; no other script or test references it (test_qdrant_retriever.py "
            "seeds its own collection under pytest's tmp_path). Regenerable by re-running "
            "the build script",
            [p for p in [
                Path("data/qdrant/fixed_size_e5_person_slice"),
                Path("data/qdrant/person_slice_demo_result.json"),
            ] if p.exists()],
        ),
    ]
    return [c for c in cats if c.paths]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--apply", action="store_true")
    for flag in ("reocr-baks", "index-backups", "superseded-combos", "retired-results"):
        ap.add_argument(f"--{flag}", action="store_true", help=f"include the GATED '{flag}' category")
    args = ap.parse_args()
    enabled = {
        "reocr-baks": args.reocr_baks,
        "index-backups": args.index_backups,
        "superseded-combos": args.superseded_combos,
        "retired-results": args.retired_results,
    }

    cats = collect()
    total = 0
    for c in cats:
        on = c.verdict == "SAFE" or enabled.get(c.key, False)
        mark = "MOVE" if on else "keep"
        print(f"\n[{c.verdict}] [{mark}] {c.label}")
        print(f"    {len(c.paths)} path(s), {c.size / 1e6:.1f} MB")
        print(f"    why: {c.why}")
        if on:
            total += c.size

    print(f"\ndestination: {args.dest}")
    print(f"{'moving' if args.apply else 'would move'} {total / 1e6:.1f} MB")
    if not args.apply:
        print("dry run -- pass --apply to move")
        return 0

    moved = 0
    for c in cats:
        if not (c.verdict == "SAFE" or enabled.get(c.key, False)):
            continue
        for p in c.paths:
            rel = p.resolve().relative_to(REPO)
            target = args.dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                print(f"  skip (exists): {rel}")
                continue
            shutil.move(str(p), str(target))
            moved += 1
    print(f"moved {moved} path(s) to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
