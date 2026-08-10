"""Identify exactly which published RQ4 answers were generated from a truncated
prompt, so they can be regenerated (2026-08-10).

`docs/rq4-prompt-truncation.md` §4 established the blast radius -- 80 of 1,590
cells -- with an ad-hoc script that was not kept. Regenerating them needs the
*list*, not the count, and a list nobody can re-derive is worth little, so this
recomputes both and **gates itself against that table**: if the counts per
(variant, arm) do not reproduce it, it exits non-zero rather than handing back a
plausible but different set.

**The gate fired, and §4 was the side that was wrong: the radius is 81.** The
extra cell is `cite_all_guarded / dense_qwen3_0.6b_semantic / q001`, 8,258
tokens, 66 past the line -- §4's screen divided characters by 1.046 and never
measured it (see below). The doc is corrected there rather than here, and
`EXPECTED_CELLS` now carries the measured radius as a regression pin.

Method (a two-stage screen, for the reason `rq4_generate.preflight` uses one --
chars/token spans 1.0098..4.0175 on this corpus, so no ratio estimate is usable):

1. **Screen** on an upper bound. A prompt whose bound fits 8,192 fit the old
   default and cannot have been truncated; it is never probed.
2. **Reproduce the old run**: send each survivor at `num_ctx=8192` and look for
   the truncation *signature*, `prompt_eval_count == 8192//2 + 2 == 4098`.

**The first version measured at 16384 and flagged `> 8192` instead, and that is
a proxy, not the counterfactual.** It reconstructed 78 where section 4 published
80, both misses off by exactly one cell and both in the boundary region -- which
is the shape the proxy fails in: `prompt_eval_count` reports tokens *evaluated*,
excluding any prefix served from llama.cpp's prompt cache (a real request here
shows `task.n_tokens = 9391` against `prompt eval ... 9378 tokens`, the 13-token
chat-template header). A few tokens of slack decides nothing in the middle of the
range and decides everything within ~13 of the line. Asking ollama at 8192 what
it does removes the question: the signature is exact, and it is the same rule
`rq4_generate.truncated_to()` and `audit_pipeline_invariants.py` G1a already use.

The one place the signature is *not* self-evident is a prompt whose true length
sits near 4,098 tokens -- it would report ~4,098 whether cut or not. Those are
disambiguated with a second probe at 16384 (cut => the 16384 count is far
larger), rather than resolved by a threshold.

The published answers themselves carry no `num_ctx` field (that was added by the
same fix), which is precisely why the damage has to be reconstructed prompt by
prompt; see `audit_pipeline_invariants.py` G1b.

**The screen constant `rq4_generate.MIN_CHARS_PER_TOKEN = 1.046` was NOT a bound,
and that -- not the proxy above -- is why two runs came back with 78.** It was
documented as this corpus's lowest chars/token but measured on the two entity
arms alone; of the **228** prompts probed here **15 fall below it**, the minimum
being **1.0098** (`bm25_semantic/q001`, 11,208 chars / 11,099 tokens). An "upper
bound" the data violates does not merely mis-sort candidates -- it *excludes*
cells from being measured at all, and all three cells it hid sit in the same
8,192-8,300-token band: `sentence_cap/dense/q009` (8,212),
`cite_all/hybrid_m2v/q025` (8,269, and 8,475 under `cite_all_guarded`, where it
did screen in) and `cite_all_guarded/dense/q001` (8,258, the one §4 also missed).
So the screen here is `SCREEN_CHARS_PER_TOKEN`, set with real headroom under the
observed floor, and **S1 re-derives the realized ratio of every probed prompt and
fails if any lands below it** -- an empirical constant with a check that it held,
rather than an asserted one ([[feedback_an_asserted_invariant_is_not_a_check]]).
The only *provable* bound is one token per UTF-8 byte, which screens in 759 of
954 cells; that is the honest fallback if S1 ever fires, and it is what
`rq4_generate.token_upper_bound` was changed to, since a shipped guard cannot
rely on a self-check that only this script runs.

Every probe is cached to `data/results/rq4_truncated_cells_raw.json` as it is
taken, so an interrupted run resumes free and the evidence outlives the console
log -- the first run's did not, and reconciling its two mismatches without it
meant re-measuring everything. Widening the screen afterwards then cost only the
53 newly-admitted probes.

Writes `data/results/rq4_truncated_cells.json` (the worklist consumed by
`rq4_regenerate_truncated.py`) and `data/results/rq4_truncated_cells.md`.

Run (GPU, ~280 forward passes of 1 token each; cached, so a re-run is free):
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_find_truncated_answers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ollama  # noqa: E402

from rq4_generate import (  # noqa: E402
    _CONTEXTS, build_prompt, resident_models, truncated_to,
)

# Deliberately below the 1.0098 minimum realized over 228 measured prompts (see
# the module docstring): a screen has to be wrong in the direction of probing too
# much. S1 checks that it stayed below every realized ratio.
SCREEN_CHARS_PER_TOKEN = 0.95

_ANSWERS = REPO / "data" / "rq4" / "answers"
_OUT_JSON = REPO / "data" / "results" / "rq4_truncated_cells.json"
_OUT_MD = REPO / "data" / "results" / "rq4_truncated_cells.md"
_RAW = REPO / "data" / "results" / "rq4_truncated_cells_raw.json"

OLD_NUM_CTX = 8192          # what every published answer was generated at
PROBE_NUM_CTX = 16384       # large enough to feed every prompt whole

VARIANT_DIR = {
    "sentence_cap": "phi4",
    "cite_all": "phi4_cite_all",
    "cite_all_guarded": "phi4_cite_all_guarded",
}
# Only arms that can be truncated at all. hybrid tops out at 7,999 tokens and
# closed_book carries no documents -- both 0/106 in all three variants (§4), so
# probing them would burn GPU to confirm a structural zero.
ARMS = ["dense_qwen3_0.6b_semantic", "bm25_semantic", "hybrid_m2v_semantic"]

# docs/rq4-prompt-truncation.md section 4, transcribed -- **with one cell
# corrected 2026-08-10**: `cite_all_guarded / dense` is 17, not the 16 published
# there. That table was screened with the same unsound `chars / 1.046` bound, so
# `q001` (8,258 tokens, 66 over the line) was never measured; here it reports the
# exact signature at num_ctx=8192. The doc is corrected to match, and the
# correction is recorded there rather than silently absorbed.
#
# **What this constant is for, after that change.** It began as a cross-check
# against an independent measurement; overwriting it with what this script
# measures would make it a tautology. It is now a *regression pin*: 81 cells,
# each confirmed twice (signature at 8192 + whole-prompt count at 16384), so a
# future run that finds a different number has either found a real change or
# broken the screen -- and S1 below is what tells those two apart.
EXPECTED_CELLS = {
    ("sentence_cap", "dense_qwen3_0.6b_semantic"): 14,
    ("sentence_cap", "bm25_semantic"): 5,
    ("sentence_cap", "hybrid_m2v_semantic"): 6,
    ("cite_all", "dense_qwen3_0.6b_semantic"): 15,
    ("cite_all", "bm25_semantic"): 5,
    ("cite_all", "hybrid_m2v_semantic"): 7,
    ("cite_all_guarded", "dense_qwen3_0.6b_semantic"): 17,   # section 4 said 16
    ("cite_all_guarded", "bm25_semantic"): 5,
    ("cite_all_guarded", "hybrid_m2v_semantic"): 7,
}


def _probe(model: str, prompt: str, num_ctx: int) -> int:
    return ollama.chat(
        model=model, messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_ctx": num_ctx, "num_predict": 1},
    )["prompt_eval_count"]


def main() -> int:
    model = "phi4"
    resident = resident_models()
    if resident:
        raise SystemExit(f"refusing to start: {resident} already loaded; "
                         f"only one GPU job at a time on this machine")

    signature = truncated_to(OLD_NUM_CTX)       # 4,098
    raw = json.loads(_RAW.read_text(encoding="utf-8")) if _RAW.is_file() else {}

    cells, n_probed = [], 0
    for variant, vdir in VARIANT_DIR.items():
        for arm in ARMS:
            for path in sorted((_CONTEXTS / arm).glob("q*.json")):
                ctx = json.loads(path.read_text(encoding="utf-8"))
                prompt = build_prompt(ctx, variant)
                if len(prompt) / SCREEN_CHARS_PER_TOKEN <= OLD_NUM_CTX:
                    continue        # fitted; no forward pass needed
                key = f"{variant}/{arm}/{path.stem}"
                rec = raw.setdefault(key, {})
                rec["n_chars"] = len(prompt)

                if "n_8192" not in rec:
                    rec["n_8192"] = _probe(model, prompt, OLD_NUM_CTX)
                    n_probed += 1
                    _RAW.write_text(json.dumps(raw, indent=1), encoding="utf-8")
                n8 = rec["n_8192"]

                # A prompt whose true length is itself near 4,098 reports ~4,098
                # cut or not, so the signature cannot decide it. Measure it whole.
                if abs(n8 - signature) <= 128 and "n_16384" not in rec:
                    rec["n_16384"] = _probe(model, prompt, PROBE_NUM_CTX)
                    n_probed += 1
                    _RAW.write_text(json.dumps(raw, indent=1), encoding="utf-8")
                is_trunc = (n8 == signature if "n_16384" not in rec
                            else rec["n_16384"] > n8 + 128)

                if is_trunc:
                    if "n_16384" not in rec:
                        rec["n_16384"] = _probe(model, prompt, PROBE_NUM_CTX)
                        n_probed += 1
                        _RAW.write_text(json.dumps(raw, indent=1), encoding="utf-8")
                    answer = _ANSWERS / vdir / arm / path.name
                    cells.append({
                        "variant": variant, "arm": arm, "query": path.stem,
                        "prompt_tokens": rec["n_16384"],
                        "fed_at_8192": n8,
                        "kept_frac": round(n8 / rec["n_16384"], 4),
                        "answer_path": str(answer.relative_to(REPO)).replace("\\", "/"),
                        "answer_exists": answer.is_file(),
                    })
                print(f"  {key}: fed {n8:,} tok at 8192"
                      f"{'  TRUNCATED (true ' + format(rec['n_16384'], ',') + ')' if is_trunc else ''}",
                      flush=True)

    # S1 -- the screen is only sound while every prompt's realized chars/token
    # stays at or above SCREEN_CHARS_PER_TOKEN; below it, a cell longer than
    # 8,192 tokens would be screened out and never probed (which is exactly how
    # 1.046 lost hybrid_m2v_semantic/q025). Re-derive the ratio from the cache
    # rather than trusting the constant.
    ratios = []
    for key, rec in raw.items():
        true_tok = rec.get("n_16384") or rec.get("n_8192")
        if rec.get("n_chars") and true_tok:
            ratios.append((rec["n_chars"] / true_tok, key))
    s1_min, s1_key = min(ratios) if ratios else (float("nan"), "-")
    s1_ok = bool(ratios) and s1_min >= SCREEN_CHARS_PER_TOKEN
    print(f"\nS1 screen soundness: min realized chars/token = {s1_min:.4f} "
          f"({s1_key}) vs screen {SCREEN_CHARS_PER_TOKEN} over {len(ratios)} "
          f"measured prompts -- {'PASS' if s1_ok else 'FAIL'}")

    got = {}
    for c in cells:
        got[(c["variant"], c["arm"])] = got.get((c["variant"], c["arm"]), 0) + 1
    mismatches = [(k, EXPECTED_CELLS[k], got.get(k, 0)) for k in EXPECTED_CELLS
                  if EXPECTED_CELLS[k] != got.get(k, 0)]

    _OUT_JSON.write_text(json.dumps(cells, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# RQ4 cells generated from a truncated prompt\n",
         "Generated by `tools/eval/rq4_find_truncated_answers.py`.",
         f"{n_probed} forward passes this run over {len(raw)} screened cells "
         f"(of 954); the rest provably fitted {OLD_NUM_CTX:,} tokens by upper "
         f"bound. Truncation is read from the signature "
         f"`prompt_eval_count == {signature:,}` at `num_ctx={OLD_NUM_CTX:,}` -- "
         f"the old run reproduced, not a proxy for it.\n",
         f"**S1 (screen soundness)**: the screen keeps a cell only if "
         f"`len(prompt)/{SCREEN_CHARS_PER_TOKEN}` exceeds {OLD_NUM_CTX:,} tokens, "
         f"so it is sound only while no prompt realizes fewer chars per token "
         f"than that. Minimum realized over {len(ratios)} measured prompts: "
         f"**{s1_min:.4f}** (`{s1_key}`) -- {'PASS' if s1_ok else '**FAIL**'}. "
         f"(`rq4_generate.MIN_CHARS_PER_TOKEN = 1.046` is violated by this same "
         f"data and is not a bound.)\n",
         "| variant | arm | published (section 4) | reconstructed | agrees |",
         "|---|---|---|---|---|"]
    for k in EXPECTED_CELLS:
        L.append(f"| {k[0]} | `{k[1]}` | {EXPECTED_CELLS[k]} | {got.get(k, 0)} | "
                 f"{'yes' if EXPECTED_CELLS[k] == got.get(k, 0) else '**NO**'} |")
    L.append(f"\n**{len(cells)} cells** to regenerate "
             f"({sum(1 for c in cells if not c['answer_exists'])} have no answer file).\n")
    L.append("| variant | arm | query | true prompt tokens | fed at 8192 | % kept |")
    L.append("|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda c: -c["prompt_tokens"]):
        L.append(f"| {c['variant']} | `{c['arm']}` | {c['query']} | "
                 f"{c['prompt_tokens']:,} | {c['fed_at_8192']:,} | "
                 f"{c['kept_frac']:.0%} |")
    _OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"\n{len(cells)} truncated cells; wrote {_OUT_JSON} and {_OUT_MD}")
    if not s1_ok:
        print("S1 FAILED: a probed prompt realizes fewer chars/token than the "
              "screen assumes, so cells may have been excluded unprobed. Lower "
              "SCREEN_CHARS_PER_TOKEN (the provable fallback is one token per "
              "UTF-8 byte) and re-run before trusting the count")
        return 1
    if mismatches:
        for k, pub, got_n in mismatches:
            print(f"  [MISMATCH] {k}: published {pub}, reconstructed {got_n}")
        print("refusing to hand back a worklist that does not reproduce the "
              "published blast radius -- reconcile before regenerating")
        return 1
    print("gate: reproduces EXPECTED_CELLS exactly (= docs/rq4-prompt-truncation.md "
          "section 4 with its cite_all_guarded/dense cell corrected 16 -> 17)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
