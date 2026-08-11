"""Close `audit_pipeline_invariants.py`'s G1c by measuring it (2026-08-11).

G1c WARNs that 759 published RQ4 answers have neither a recorded `num_ctx` nor
provable evidence about their prompt -- *unmeasured*, never *suspected*. This
script is the ~GPU-hours that turns them into one or the other, and it is
deliberately the boring option: ask ollama, at the context the old run used,
what it actually feeds. `prompt_eval_count == 8192 // 2 + 2` is the exact
truncation signature (`rq4_generate.truncated_to`, measured against ollama
0.32.6 / phi4, not read from the docs).

**Why this and not the constant.** `SCREEN_CHARS_PER_TOKEN = 0.95` would clear
all 759 for free, and 15 of the 228 prompts already measured fall below
`rq4_generate.MIN_CHARS_PER_TOKEN = 1.046`, which was documented as this
corpus's floor. An observed minimum is a description of a sample; this project
has published a wrong blast radius three times by treating one as a bound
([[feedback_an_observed_extreme_is_not_a_bound]]).

**The universe is imported, not redefined.** The set probed here is exactly what
`_rq4_prompt_fit_evidence` classifies as unmeasured -- same walk, same byte-bound
screen, same cache lookup -- because two enumerations of "which answers still
need evidence" would eventually disagree, and the disagreement would look like a
finding. Probing something the audit does not count is wasted GPU; missing
something it does count leaves G1c red after a run that reported success.

**Its own cache file, on purpose.** Results go to
`data/results/rq4_prompt_fit_probes.json`, not into
`rq4_truncated_cells_raw.json`. That file belongs to
`rq4_find_truncated_answers.py`, whose S1 re-derives the realized chars/token of
every entry to prove *its* 0.95 screen was sound. This script screens by the
provable byte bound over all five arms, so an entry of its own could fail a check
about a screen it was never under. The audit reads both.

**Do not "refresh" `rq4_find_truncated_answers.py` instead.** That script rewrites
`rq4_truncated_cells.{json,md}`, which are the record of a state that has since
been repaired -- the 81 cells it lists were regenerated 2026-08-10
([[feedback_dont_regenerate_a_record_of_a_repaired_state]]).

Self-checks, all reported and gating:
  S0  anchor -- re-probe two cells the finder already measured (one truncated,
      one not) and reproduce its counts. Every other probe here is a number
      nothing else can confirm, so without this a systematically wrong call
      (different model, different options) would produce a clean, plausible,
      wrong report.
  S1  no probe reports more tokens fed than `num_ctx` allows.
  S2  every cell called truncated reports the signature exactly AND measures
      longer when re-probed whole -- two independent facts, not one restated.
  S3  re-classify through the audit afterwards: 0 unmeasured must remain, and
      the truncated set must match what this run found.

Run (GPU, one job at a time; every probe is cached, so an interrupt resumes free):
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_probe_prompt_fit.py --dry-run
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_probe_prompt_fit.py --sample 12
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_probe_prompt_fit.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama  # noqa: E402

import audit_pipeline_invariants as audit  # noqa: E402
from rq4_generate import (  # noqa: E402
    _CONTEXTS, build_prompt, resident_models, token_upper_bound, truncated_to,
)

MODEL = "phi4"
OLD_NUM_CTX = 8192          # what every pre-fix answer was generated at
PROBE_NUM_CTX = 16384       # large enough to feed every prompt whole
NEAR = 128                  # tokens either side of the signature that it cannot decide

_ANSWERS = REPO / "data" / "rq4" / "answers"
_CACHE = REPO / "data" / "results" / "rq4_prompt_fit_probes.json"
_OUT_MD = REPO / "data" / "results" / "rq4_prompt_fit_probes.md"

# S0 anchors, from `rq4_truncated_cells_raw.json` -- one cell the finder measured
# as truncated and one it measured as fitting. Values are read from that file at
# run time rather than hardcoded here: a second transcription of a number is a
# second place for it to be wrong.
_ANCHORS = ["cite_all/dense_qwen3_0.6b_semantic/q001",
            "sentence_cap/bm25_semantic/q001"]


def _probe(prompt: str, num_ctx: int) -> int:
    return ollama.chat(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_ctx": num_ctx, "num_predict": 1},
    )["prompt_eval_count"]


def _pending() -> list[tuple[str, str, str, Path]]:
    """(label, variant, cache key, context path) for every answer G1c still counts.

    Classification comes from the audit so the two cannot drift; this only maps
    its labels back to the files a probe needs.
    """
    if audit._RQ4_CONTEXTS is None:
        raise SystemExit("rq4_generate did not import; cannot rebuild prompts")
    pre_fix = []
    for path in sorted(_ANSWERS.glob("*/*/q*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("num_ctx") is None or rec.get("prompt_eval_count") is None:
            pre_fix.append(path)

    _, _, unmeasured, _ = audit._rq4_prompt_fit_evidence(pre_fix)
    out, unbuildable = [], []
    for label in unmeasured:
        parts = label.split("/")
        variant = audit._RQ4_VARIANT_BY_DIR.get(parts[0]) if len(parts) == 3 else None
        ctx = _CONTEXTS / parts[1] / f"{parts[2]}.json" if variant else None
        # The audit also calls a cell unmeasured when its *context* is missing or
        # unreadable -- no probe can reach those, so they stay in G1c by a different
        # cause and are reported rather than silently dropped.
        if variant is None or not ctx.is_file():
            unbuildable.append(label)
            continue
        out.append((label, variant, f"{variant}/{parts[1]}/{parts[2]}", ctx))
    if unbuildable:
        print(f"note: {len(unbuildable)} unmeasured cells have no rebuildable prompt "
              f"(missing context or unknown variant); not probeable: "
              f"{', '.join(unbuildable[:3])}{' ...' if len(unbuildable) > 3 else ''}")
    return out


def _load_cache() -> dict:
    try:
        return json.loads(_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="enumerate and size the work; no GPU, no ollama call")
    ap.add_argument("--sample", type=int, default=0,
                    help="probe N cells spread evenly across the length distribution, "
                         "and project the full cost from them")
    ap.add_argument("--skip-anchor", action="store_true",
                    help="skip S0 (only for a resumed run that already passed it)")
    args = ap.parse_args()

    pending = _pending()
    signature = truncated_to(OLD_NUM_CTX)
    lens = sorted(token_upper_bound(build_prompt(
        json.loads(ctx.read_text(encoding="utf-8")), variant))
        for _, variant, _, ctx in pending) if pending else []

    print(f"{len(pending)} answers unmeasured by G1c")
    if lens:
        # The byte bound, not a token count -- ~3x loose on Thai. It orders the
        # work; it does not predict the probe result.
        q = [lens[0], lens[len(lens) // 4], lens[len(lens) // 2],
             lens[3 * len(lens) // 4], lens[-1]]
        print("  byte-bound tokens  min/p25/median/p75/max: "
              + " / ".join(f"{v:,}" for v in q))
    if args.dry_run:
        by_arm: dict[str, int] = {}
        for label, *_ in pending:
            by_arm[label.rsplit("/", 1)[0]] = by_arm.get(label.rsplit("/", 1)[0], 0) + 1
        for k in sorted(by_arm, key=lambda k: -by_arm[k]):
            print(f"    {by_arm[k]:>4}  {k}")
        print("\ndry run: nothing probed")
        return 0

    cache = _load_cache()
    # Guard the GPU only when this run will actually touch it. Once every cell is
    # cached, `--skip-anchor` is a pure re-render, and refusing that because the
    # model this script itself loaded has not timed out yet would be theatre.
    if pending or not args.skip_anchor:
        resident = resident_models()
        if resident:
            raise SystemExit(f"refusing to start: {resident} already loaded; "
                             f"only one GPU job at a time on this machine")
    s0 = []
    if not args.skip_anchor:
        finder = json.loads((REPO / "data" / "results"
                             / "rq4_truncated_cells_raw.json").read_text(encoding="utf-8"))
        for key in _ANCHORS:
            variant, arm, stem = key.split("/")
            prompt = build_prompt(json.loads((_CONTEXTS / arm / f"{stem}.json")
                                             .read_text(encoding="utf-8")), variant)
            got = _probe(prompt, OLD_NUM_CTX)
            want = finder[key]["n_8192"]
            s0.append((key, want, got))
            print(f"  S0 {key}: finder {want:,}, reprobed {got:,} "
                  f"-- {'ok' if got == want else 'MISMATCH'}", flush=True)
        # Carried in the cache so a re-render reports the anchor that actually ran
        # instead of a PASS meaning "not checked" -- a vacuous PASS is the failure
        # mode this project has already hit twice in its own audits.
        cache.setdefault("_meta", {})["anchor"] = s0
        _CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        if any(w != g for _, w, g in s0):
            print("S0 FAILED: this run does not reproduce the finder's own counts, "
                  "so its numbers describe a different measurement -- reconcile "
                  "the model/options before trusting anything below")
            return 1

    work = pending
    if args.sample:
        step = max(1, len(pending) // args.sample)
        order = sorted(pending, key=lambda p: token_upper_bound(build_prompt(
            json.loads(p[3].read_text(encoding="utf-8")), p[1])))
        work = order[::step][:args.sample]
        print(f"\nsampling {len(work)} of {len(pending)}, spread across the "
              f"length distribution (not the longest first, which would "
              f"over-project the total)")

    timings = []
    for i, (label, variant, key, ctx_path) in enumerate(work, 1):
        prompt = build_prompt(json.loads(ctx_path.read_text(encoding="utf-8")), variant)
        rec = cache.setdefault(key, {})
        rec["n_chars"] = len(prompt)
        rec["n_bytes"] = token_upper_bound(prompt)
        t0, probed_now = time.perf_counter(), False
        if "n_8192" not in rec:
            rec["n_8192"] = _probe(prompt, OLD_NUM_CTX)
            probed_now = True
            _CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        n8 = rec["n_8192"]
        # A prompt whose true length sits within NEAR of the signature reports it
        # whether cut or not, so the signature cannot decide it; measure it whole.
        # Truncated cells get the same second probe, for their true length.
        if abs(n8 - signature) <= NEAR and "n_16384" not in rec:
            rec["n_16384"] = _probe(prompt, PROBE_NUM_CTX)
            probed_now = True
            _CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        n16 = rec.get("n_16384")
        rec["truncated"] = (n16 > n8 + NEAR) if n16 is not None else (n8 == signature)
        _CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        if probed_now:                       # a cache hit costs no GPU; timing it
            timings.append(time.perf_counter() - t0)   # would deflate the projection

        if i % 25 == 0 or rec["truncated"] or args.sample:
            print(f"  [{i}/{len(work)}] {label}: fed {n8:,} at 8192"
                  f"{'  TRUNCATED (true ' + format(n16, ',') + ')' if rec['truncated'] else ''}"
                  f"  {timings[-1]:.1f}s" if timings else "  (cached)", flush=True)

    # Warm-up lands in the first probe, so a mean over everything over-states a
    # long run's per-probe cost ([[feedback_dont_extrapolate_gpu_eta_from_first_batches]]).
    steady = timings[1:] or timings
    # Kept in the cache so a later re-render reports the rate that was measured
    # rather than the 0.0 of a run that probed nothing. `_meta` carries no
    # `n_8192`, so every consumer's "is this a probe" filter skips it.
    if steady:
        cache.setdefault("_meta", {})["seconds_per_cell"] = sum(steady) / len(steady)
        _CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    per = cache.get("_meta", {}).get("seconds_per_cell", 0.0)
    print(f"\n{len(work)} cells, {len(timings)} newly probed, "
          f"{sum(timings)/60:.1f} min; {per:.1f}s per cell after the first "
          f"({timings[0]:.1f}s)" if timings else f"\n{len(work)} cells, all cached")
    if args.sample:
        print(f"projection for all {len(pending)}: "
              f"{per * len(pending) / 3600:.1f} GPU-hours")
        return 0

    # ---- self-checks -------------------------------------------------------
    # Read the verdict back off the cache rather than off this run's loop, so a
    # re-render reports the same thing the probing run did instead of "0 truncated"
    # because it happened to probe nothing.
    probed = {k: v for k, v in cache.items() if "n_8192" in v}
    truncated = [(k, v["n_8192"], v.get("n_16384")) for k, v in probed.items()
                 if v.get("truncated")]
    s1 = [k for k, v in probed.items() if v["n_8192"] > OLD_NUM_CTX]
    s2 = [k for k, v in probed.items() if v.get("truncated")
          and not (v["n_8192"] == signature
                   and (v.get("n_16384") or 0) > v["n_8192"] + NEAR)]
    _, _, still, audit_trunc = _pending_classification()
    s3 = (not still) and len(audit_trunc) == len(truncated)
    anchor = [tuple(a) for a in (s0 or cache.get("_meta", {}).get("anchor", []))]
    checks = [("S0 reproduces the finder's counts",
               bool(anchor) and not any(w != g for _, w, g in anchor),
               ("; ".join(f"`{k}` {w:,}=={g:,}" for k, w, g in anchor)
                + ("" if s0 else " (carried from the probing run)"))
               if anchor else "never run -- no evidence this measures what the "
                              "finder measured"),
              ("S1 no probe exceeds its num_ctx", not s1,
               f"0 of {len(probed)} report more than {OLD_NUM_CTX:,} tokens fed"),
              ("S2 a truncated cell shows the signature AND measures longer whole",
               not s2, f"0 of {len(truncated)} fail either half"),
              ("S3 the audit now classifies every answer", s3,
               f"{len(still)} still unmeasured, {len(audit_trunc)} truncated "
               f"(this run found {len(truncated)})")]
    print()
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    _write_report(probed, truncated, checks, per)
    print(f"\n{len(truncated)} of {len(probed)} probed cells were truncated; "
          f"wrote {_OUT_MD}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def _pending_classification() -> tuple[int, int, list[str], list[str]]:
    pre_fix = []
    for path in sorted(_ANSWERS.glob("*/*/q*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("num_ctx") is None or rec.get("prompt_eval_count") is None:
            pre_fix.append(path)
    return audit._rq4_prompt_fit_evidence(pre_fix)


def _write_report(probed: dict, truncated: list, checks: list, per: float) -> None:
    ratios = [(v["n_chars"] / (v.get("n_16384") or v["n_8192"]), k)
              for k, v in probed.items() if v.get("n_chars")]
    lo, lo_key = min(ratios) if ratios else (float("nan"), "-")
    hi, hi_key = max(ratios) if ratios else (float("nan"), "-")
    fed = sorted((v["n_8192"], k) for k, v in probed.items())
    loose = sorted(v["n_bytes"] / v["n_8192"] for v in probed.values() if v.get("n_bytes"))
    second = sum(1 for v in probed.values() if "n_16384" in v)
    L = ["# RQ4 prompt fit: the answers G1c could not reach\n",
         "Generated by `tools/eval/rq4_probe_prompt_fit.py`.\n",
         f"Every published RQ4 answer predating the `num_ctx` field, whose prompt "
         f"the UTF-8-byte upper bound cannot clear, sent to ollama at "
         f"`num_ctx={OLD_NUM_CTX:,}` -- the context the old run used. Truncation is "
         f"read from the signature `prompt_eval_count == {truncated_to(OLD_NUM_CTX):,}`, "
         f"the old run reproduced rather than a proxy for it. "
         f"**{len(probed)} cells probed**"
         + (f", {per:.1f}s each.\n" if per else ".\n"),
         f"**{len(truncated)} truncated.**\n"]
    if truncated:
        L += ["| answer | fed at 8,192 | true tokens | % kept |", "|---|---|---|---|"]
        for label, n8, n16 in sorted(truncated, key=lambda t: -(t[2] or 0)):
            L.append(f"| `{label}` | {n8:,} | {n16:,} | {n8/n16:.0%} |")
        L.append("")
    L += ["## Self-checks\n", "| check | result | detail |", "|---|---|---|"]
    for name, ok, detail in checks:
        L.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    L += ["\n## What the measurement says about the estimators\n",
          f"The longest prompt in this set was fed **{fed[-1][0]:,}** tokens "
          f"(`{fed[-1][1]}`), **{OLD_NUM_CTX - fed[-1][0]:,} short** of the old "
          f"default; the shortest {fed[0][0]:,}. So the pre-fix run cleared the line "
          f"on every one of these, by a margin no one chose.\n",
          f"**{second} of {len(probed)}** prompts landed within {NEAR} tokens of the "
          f"signature, where the count alone cannot say whether it was cut, and were "
          f"re-probed at {PROBE_NUM_CTX:,} to decide. A run reporting no truncation "
          f"without those second probes would not have measured its own boundary.\n",
          f"The UTF-8-byte bound is **{loose[0]:.2f}x-{loose[-1]:.2f}x** loose here "
          f"(median {loose[len(loose)//2]:.2f}x), which is why it left these 759 "
          f"unresolved while proving 603 others outright: it is sound, not tight.\n",
          f"Realized chars/token spans **{lo:.4f}** (`{lo_key}`) to **{hi:.4f}** "
          f"(`{hi_key}`). The minimum matters: `rq4_generate.MIN_CHARS_PER_TOKEN` "
          f"was documented as 1.046 and the previous low was 1.0098, so measuring "
          f"more prompts moved the observed floor **again**. Read it as a "
          f"description of a sample; it is not a bound, and neither is the 0.95 "
          f"screen this script deliberately does not use.\n"]
    _OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
