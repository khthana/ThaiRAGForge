"""Seal the index directories that predate `ArtifactStore.seal` (2026-08-21).

`ArtifactStore.save` now writes `_complete.json` declaring the four artifacts to
be one build, and `index_cache` refuses to serve a directory whose artifacts do
not match that declaration -- which is how a serving read detects that it landed
between a writer's `chunks.parquet` and its `embeddings.npy`
(`data/results/serving_concurrency.md` section 6: with a 150 ms inter-file gap
that mixed pairing was served on the majority of reads, and stamping the read at
both ends could not see it).

Every index on disk today was written before that, so every one is **unsealed**:
the cache serves it, but only with the older, narrower guarantee. This walks the
index tree and seals them.

**What sealing does and does not claim.** It records the artifacts' current
`(mtime_ns, size)`; it does not verify that they came from one build. That is
sound here only because these directories are quiescent and audited
(`audit_pipeline_invariants.py` I1 checks the row alignment itself), and it is
exactly why a directory whose artifacts were touched in the last
`--min-age` seconds is REFUSED rather than sealed -- sealing something that is
still being written would bless the very pairing this exists to catch.

Dry run by default, like `tools/archive_unused.py`:

    python tools/seal_index_dirs.py                 # list what would be sealed
    python tools/seal_index_dirs.py --apply
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from rag_lab.io.artifact_store import (  # noqa: E402
    ARTIFACT_FILES,
    artifact_stamp,
    read_seal,
    seal,
)

DEFAULT_ROOTS = ("data/index",)


def index_dirs(roots: list[str]) -> list[Path]:
    """Every directory holding a chunks.parquet, at any depth under the roots."""
    out: list[Path] = []
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        out.extend(sorted(p.parent for p in r.rglob(ARTIFACT_FILES[0])))
    return out


def classify(d: Path, min_age_s: float) -> tuple[str, str]:
    """(verdict, detail) for one directory. Verdicts:

    sealed     already matches its seal -- nothing to do
    stale      has a seal that does not match: rebuilt or edited since
    unsealed   no seal at all: what every pre-2026-08-21 index is
    too-new    an artifact was touched recently; refuse rather than bless it
    """
    stamp = artifact_stamp(d)
    newest = max((e[0] for e in stamp if e is not None), default=0)
    age_s = time.time() - newest / 1e9
    if age_s < min_age_s:
        return "too-new", f"newest artifact is {age_s:.0f}s old (< {min_age_s:.0f}s)"
    current = read_seal(d)
    if current is None:
        return "unsealed", "no _complete.json"
    if current == stamp:
        return "sealed", "matches"
    return "stale", "seal does not match the artifacts"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="*", default=list(DEFAULT_ROOTS))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument(
        "--min-age",
        type=float,
        default=60.0,
        help="refuse to seal a directory whose artifacts changed this recently",
    )
    args = ap.parse_args()

    dirs = index_dirs(args.roots)
    counts: dict[str, int] = {}
    to_seal: list[Path] = []
    for d in dirs:
        verdict, detail = classify(d, args.min_age)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict in ("unsealed", "stale"):
            to_seal.append(d)
        if verdict != "sealed":
            print(f"  {verdict:9s} {d}  ({detail})")

    print(
        f"\n{len(dirs)} index directories: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    )
    if not args.apply:
        print(f"dry run -- {len(to_seal)} would be sealed. Re-run with --apply.")
        return
    for d in to_seal:
        seal(d)
    print(f"sealed {len(to_seal)}")
    # Re-classify, so the exit status reports the state on disk rather than the
    # state this script believes it produced.
    bad = [d for d in to_seal if classify(d, 0.0)[0] != "sealed"]
    if bad:
        print(f"FAILED to seal {len(bad)}: {bad[:3]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
