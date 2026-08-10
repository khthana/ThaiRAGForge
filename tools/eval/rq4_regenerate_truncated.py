"""RQ4 step 2b: regenerate the answers whose prompt was truncated at num_ctx=8192.

Consumes the worklist `tools/eval/rq4_find_truncated_answers.py` writes (which
gates itself against `docs/rq4-prompt-truncation.md` section 4, so the set is the
published blast radius, not a plausible re-derivation).

**Why delete-then-generate rather than a dedicated writer.** `rq4_generate.py`
skips an answer file that already exists -- that is its resume mechanism -- so
removing exactly the bad files and re-invoking it regenerates exactly those and
freezes every other answer byte-for-byte. That is the same move the 2026-08-07
rebuild-#3 refresh made (regenerate only the cells whose context changed), and it
matters for the same reason: temperature 0 is **not** reproducible here (14/24
identical citation sets under `cite_all`), so any answer re-rolled without cause
adds noise to a paired comparison that the scorer cannot tell from signal.

The originals are **moved, not deleted** (ADR-0004 recoverability): they go to
`data/rq4/_truncated_backup_<stamp>/` with their `<variant_dir>/<arm>/qNNN.json`
path preserved, alongside a `manifest.json`. They are evidence -- the only copies
of what an evidence-stripped answer looked like -- not garbage.

One invocation per variant (not per arm): `rq4_generate.py` unloads the model
when it returns, so folding the three arms into one `--arms` list saves two
model loads per variant. The resident-model guard is honoured between variants
by waiting for `ollama ps` to empty, since only one GPU job may run at a time on
this machine.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_regenerate_truncated.py
    ... --dry-run     # print the plan, touch nothing
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rq4_generate import resident_models  # noqa: E402

_WORKLIST = REPO / "data" / "results" / "rq4_truncated_cells.json"
_ANSWERS = REPO / "data" / "rq4" / "answers"
_PY = REPO / ".venv" / "Scripts" / "python.exe"
_GEN = REPO / "tools" / "eval" / "rq4_generate.py"

NUM_CTX = 16384         # every cell in the worklist exceeds 8192; none exceeds this
# docs/rq4-prompt-truncation.md section 4, as corrected 2026-08-10: that table
# published 80, and re-deriving the list with a sound screen found one more
# (`cite_all_guarded/dense/q001`, 8,258 tokens). The finder gates on the same
# number from the other side, so a disagreement here means the worklist on disk
# is not the one it wrote.
EXPECTED = 81


def wait_for_idle_gpu(timeout: float = 300.0) -> None:
    """Block until no model is resident, so the next run's guard can pass."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not resident_models():
            return
        time.sleep(5)
    raise SystemExit("a model is still resident after 5 min; refusing to start a "
                     "second GPU job on a 12 GB card")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default="phi4")
    args = ap.parse_args()

    if not _WORKLIST.is_file():
        raise SystemExit(f"no worklist at {_WORKLIST}; run "
                         "tools/eval/rq4_find_truncated_answers.py first")
    cells = json.loads(_WORKLIST.read_text(encoding="utf-8"))
    if len(cells) != EXPECTED:
        raise SystemExit(f"worklist holds {len(cells)} cells, expected {EXPECTED} "
                         "(docs/rq4-prompt-truncation.md section 4) -- reconcile "
                         "before regenerating")
    missing = [c for c in cells if not (REPO / c["answer_path"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} worklist cells have no answer file on "
                         f"disk, e.g. {missing[0]['answer_path']} -- the worklist "
                         "does not describe this checkout")

    by_variant: dict[str, list[dict]] = {}
    for c in cells:
        by_variant.setdefault(c["variant"], []).append(c)
    per_arm = Counter((c["variant"], c["arm"]) for c in cells)

    print(f"{len(cells)} truncated cells to regenerate at num_ctx={NUM_CTX:,}")
    for (variant, arm), n in sorted(per_arm.items()):
        print(f"  {variant:18s} {arm:28s} {n:3d}")
    if args.dry_run:
        print("\n--dry-run: nothing moved, nothing generated")
        return 0

    stamp = datetime.now().strftime("%Y_%m_%d")
    backup = REPO / "data" / "rq4" / f"_truncated_backup_{stamp}"
    for c in cells:
        src = REPO / c["answer_path"]
        dst = backup / Path(c["answer_path"]).relative_to("data/rq4/answers")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    (backup / "manifest.json").write_text(
        json.dumps({"moved_at": datetime.now().isoformat(timespec="seconds"),
                    "reason": "generated at num_ctx=8192 from a truncated prompt; "
                              "see docs/rq4-prompt-truncation.md",
                    "cells": cells}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nmoved {len(cells)} answers -> {backup.relative_to(REPO)}")

    arms = ",".join(sorted({c["arm"] for c in cells}))
    for variant, group in by_variant.items():
        wait_for_idle_gpu()
        print(f"\n=== {variant}: regenerating {len(group)} answers ===", flush=True)
        rc = subprocess.run(
            [str(_PY), str(_GEN), "--model", args.model, "--variant", variant,
             "--arms", arms, "--num-ctx", str(NUM_CTX)],
            cwd=str(REPO), env={**__import__("os").environ, "PYTHONPATH": "src",
                                "PYTHONIOENCODING": "utf-8"},
        ).returncode
        if rc != 0:
            raise SystemExit(f"rq4_generate.py exited {rc} for variant {variant} -- "
                             "stopping; the remaining variants are untouched and the "
                             "backup still holds every original")

    still = [c for c in cells if not (REPO / c["answer_path"]).is_file()]
    if still:
        raise SystemExit(f"{len(still)} answers were not regenerated, e.g. "
                         f"{still[0]['answer_path']}")
    bad = []
    for c in cells:
        rec = json.loads((REPO / c["answer_path"]).read_text(encoding="utf-8"))
        if rec.get("num_ctx") != NUM_CTX or rec.get("prompt_eval_count") in (
                None, NUM_CTX // 2 + 2):
            bad.append(c["answer_path"])
    if bad:
        raise SystemExit(f"{len(bad)} regenerated answers are still truncated or "
                         f"carry no num_ctx, e.g. {bad[0]}")
    print(f"\nall {len(cells)} regenerated at num_ctx={NUM_CTX:,}, none truncated; "
          "re-score with rq4_score.py and diff the verdicts before citing anything")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
