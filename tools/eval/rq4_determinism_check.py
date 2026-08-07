"""Is the RQ4 generator reproducible? Re-generate answers for prompts that did
not change and compare against what was stored.

Written 2026-08-07, during the refresh of RQ4 against `chunker_compare_full`
rebuild #3. `rq4_generate.py`'s docstring asserts "**Temperature 0.** One pass,
no sampling variance to average over", and the project had a precedent for
trusting that (re-OCR at temperature 0 reproduced its input byte-for-byte, see
docs/llm-ocr-scan-log.md). The assertion does not hold for `phi4` through
Ollama: a first 12-prompt spot check found 4 answers whose prose differed from
the stored one on a byte-identical prompt.

Greedy decoding is deterministic in exact arithmetic, but GPU reductions are
not associative, so two near-tied logits can swap between runs and the
continuation diverges from there. Nothing is wrong with the pipeline; the
docstring's claim is just stronger than the hardware supports.

**Prose divergence is not the number that matters.** `rq4_score.py` reads only
the set of `[n]` labels and whether the abstention token appears, so the
question for every claim built on that report is whether *those* are stable.
This script measures all three levels separately:

    text identical          strictest, expected to fail sometimes
    citation set identical  what 4a (precision/recall) actually consumes
    abstention identical    what 4b's 2x2 actually consumes

Sampling is restricted to (arm, query) pairs whose context is byte-identical
between the backup and the current build, since only those have a stored answer
that is comparable at all.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_determinism_check.py
    ... --per-arm 5 --model phi4
    ... --baseline data/rq4/_pre_2026_08_07_refresh   # backup to compare against
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import ollama  # noqa: E402

from rq4_generate import build_prompt  # noqa: E402
from rq4_score import is_abstained, parse_citations  # noqa: E402

_CONTEXTS = REPO / "data" / "rq4" / "contexts"


def unchanged_pairs(baseline: Path, per_arm: int) -> list[tuple[str, str, dict]]:
    """(arm, filename, context) for contexts identical to the baseline copy."""
    out = []
    for arm_dir in sorted(p for p in _CONTEXTS.iterdir() if p.is_dir()):
        old_dir = baseline / "contexts" / arm_dir.name
        if not old_dir.is_dir():
            continue
        n = 0
        for path in sorted(arm_dir.glob("q*.json")):
            new = json.loads(path.read_text(encoding="utf-8"))
            if not new["blocks"]:
                continue  # closed_book: no context to vary, not informative here
            old = json.loads((old_dir / path.name).read_text(encoding="utf-8"))
            sig = lambda c: [(b["label"], b["resolution_id"], b["text"]) for b in c["blocks"]]
            if sig(old) != sig(new):
                continue
            out.append((arm_dir.name, path.name, new))
            n += 1
            if n >= per_arm:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="phi4")
    ap.add_argument("--per-arm", type=int, default=5, help="prompts per arm per variant")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--baseline", default="data/rq4/_pre_2026_08_07_refresh",
                    help="directory holding the contexts/ and answers/ to compare against")
    args = ap.parse_args()

    baseline = REPO / args.baseline
    picks = unchanged_pairs(baseline, args.per_arm)
    print(f"{len(picks)} unchanged contexts sampled from {baseline.name}\n")

    variants = [("sentence_cap", args.model), ("cite_all", f"{args.model}_cite_all")]
    tally = {v: {"n": 0, "text": 0, "cites": 0, "abstain": 0} for v, _ in variants}
    diffs = []

    for arm, name, ctx in picks:
        for variant, model_dir in variants:
            prev_path = baseline / "answers" / model_dir / arm / name
            if not prev_path.exists():
                continue
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            resp = ollama.chat(
                model=args.model,
                messages=[{"role": "user", "content": build_prompt(ctx, variant)}],
                options={"temperature": 0.0, "num_ctx": args.num_ctx},
            )
            now = resp["message"]["content"].strip()

            label_map = prev["label_map"]
            c_old, p_old = parse_citations(prev["answer"], label_map)
            c_new, p_new = parse_citations(now, label_map)
            t = tally[variant]
            t["n"] += 1
            t["text"] += (now == prev["answer"])
            t["cites"] += ((c_old, p_old) == (c_new, p_new))
            t["abstain"] += (is_abstained(prev["answer"]) == is_abstained(now))
            if (c_old, p_old) != (c_new, p_new):
                diffs.append((variant, arm, name, sorted(p_old | {str(i) for i in range(0)}),
                              len(c_old), len(c_new)))
            print(f"  {variant:12} {arm}/{name}  text={'=' if now == prev['answer'] else 'X'} "
                  f"cites={'=' if (c_old, p_old) == (c_new, p_new) else 'X'} "
                  f"({len(c_old)}->{len(c_new)})")

    print("\n| variant | n | identical text | identical citation set | identical abstention |")
    print("|---|---|---|---|---|")
    for variant, _ in variants:
        t = tally[variant]
        if not t["n"]:
            continue
        print(f"| {variant} | {t['n']} | {t['text']}/{t['n']} ({t['text']/t['n']:.0%}) | "
              f"{t['cites']}/{t['n']} ({t['cites']/t['n']:.0%}) | "
              f"{t['abstain']}/{t['n']} ({t['abstain']/t['n']:.0%}) |")

    print("\nThe middle column is the noise floor for 4a: a re-run moves this "
          "fraction of queries even with no data change at all. The right column "
          "is the same for 4b. Read any RQ4 before/after diff against these.")

    ollama.generate(model=args.model, prompt="", keep_alive=0)
    print(f"\nunloaded {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
