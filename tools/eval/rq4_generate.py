"""RQ4 step 2: generate an answer per (query, arm) with a local Ollama model.

No external API (docs/rq4-design.md): generation runs on the local RTX 3060, so
the two constraints below are correctness requirements, not hygiene.

**One resident model at a time.** A full-size chat model needs ~9-10 GB of a
12 GB card. Two of them either fail outright or silently spill to CPU, which
would make every latency number meaningless and could change output. So this
script refuses to start while `ollama ps` shows anything resident, unloads
(`keep_alive=0`) after each arm, and never loads a second model itself.

**Resumable per query.** Each answer is written the moment it is produced, and an
existing file is skipped, so an interrupted run loses only the generation in
flight -- the pattern the thematic bootstrap used across an 80-session run.

**Temperature 0 -- which is NOT the same as reproducible.** Greedy decoding is
deterministic in exact arithmetic, but GPU reductions are not associative, so two
near-tied logits can swap between runs and the continuation diverges from there.
This docstring used to claim "one pass, no sampling variance to average over";
that was measured false 2026-08-07 (`tools/eval/rq4_determinism_check.py`, 24
byte-identical prompts per variant): re-running an *unchanged* prompt reproduces
the citation set 21/24 under `sentence_cap` but only 14/24 under `cite_all`.
Consequences: (1) still set temperature 0 -- a nonzero one would be strictly
worse; (2) when re-generating after an index rebuild, regenerate **only** the
queries whose context actually changed and leave the rest frozen, so the
comparison stays paired; (3) quote that noise floor beside any before/after
movement -- an RQ4 diff cannot be read like the deterministic retrieval refreshes,
where "0 verdict flips" meant something exact.

The prompt asks for a fixed two-line shape so 4a is parseable at all:

    คำตอบ: <answer, or the abstention token>
    อ้างอิง: [1], [3]

Abstention is a *first-class* output, not a failure: 4b's whole point is whether
the model declines when the context lacks the answer, and the closed-book arm has
no context by construction, so ไม่พบข้อมูล is the correct answer there 106 times
out of 106. Nothing in the prompt tells the model which arm it is in.

Run:
    PYTHONPATH=src python tools/eval/rq4_generate.py --model phi4 \
        --variant cite_all_guarded

A second generator (the deferred gemma4:e4b robustness check) must name a live
variant -- `sentence_cap` is refused for any model but phi4, since the ablation
that would have justified it already fired the other way:

    PYTHONPATH=src python tools/eval/rq4_generate.py --model gemma4:e4b \
        --variant cite_all_guarded --limit 20
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import ollama

REPO = Path(__file__).resolve().parents[2]
_CONTEXTS = REPO / "data" / "rq4" / "contexts"
_OUT = REPO / "data" / "rq4" / "answers"

ABSTAIN = "ไม่พบข้อมูล"

# Rule 4 is the one line that changes between variants. The default
# ("sentence_cap") is what the original 530-generation run used; docs/rq4-design.md's
# "Correction (same day)" section found citation recall flat at ~0.41 across every
# retrieval arm and traced it to this rule fighting the gold set's aggregation
# queries (mean 9.87 relevant docs) rather than to a generator comprehension limit.
# "cite_all" was the ablation that decided it, under a rule registered before the
# number existed: recall rising => prompt artifact, recall staying ~0.41 => a real
# generator ceiling, and only *then* is a second generator (gemma4:e4b) the right
# next step. **Recall rose** (hybrid +0.1181 / dense +0.1095 / bm25 +0.0696, all
# Holm 0.0000), so that branch never opened and "sentence_cap" is a retired
# baseline, kept only because the 530 answers on disk are keyed to it. Running a
# *different model* under it therefore reproduces a retired artifact rather than
# answering anything -- see `_refuse_retired_variant`.
#
# "cite_all_guarded" is that ablation with its one measured cost repaired. The
# ablation worked (recall rose significantly for hybrid/dense/bm25, no precision
# cost) but it bought that with two failures rule 4 induces by *recency* -- it is
# the last rule before the question, so it outranks rule 3 in practice:
#   - closed-book abstention fell 106/106 -> 104/106. Both failures are the same
#     shape: zero documents supplied (`label_map == {}`), and the model answers
#     anyway, citing [1]/[2]/[5] that cannot exist. "Cite every relevant document"
#     with nothing to cite reads as an instruction to produce citations.
#   - the dense arm produced 4/359 phantom labels ([6]-[9] against 5 supplied
#     documents) -- the same over-production, bounded by a real context.
# So the guard is two clauses, aimed at exactly those two: an explicit zero-document
# case, and an explicit ceiling on which labels are citable. Rule 3 already covers
# the first in principle; it is restated inside rule 4 because position, not
# absence, is what defeated it.
_RULE4 = {
    "sentence_cap": "4. ตอบสั้น ๆ ไม่เกิน 3 ประโยค",
    "cite_all": "4. อ้างอิงเอกสารที่เกี่ยวข้องทุกฉบับที่พบในเอกสารที่ให้มา ไม่ใช่แค่ฉบับเดียว "
                "ความยาวคำตอบไม่จำกัด ตราบใดที่ครอบคลุมทุกฉบับที่เกี่ยวข้อง",
    "cite_all_guarded":
        "4. อ้างอิงเอกสารที่เกี่ยวข้องทุกฉบับที่พบในเอกสารที่ให้มา ไม่ใช่แค่ฉบับเดียว "
        "ความยาวคำตอบไม่จำกัด ตราบใดที่ครอบคลุมทุกฉบับที่เกี่ยวข้อง\n"
        f"5. ถ้าไม่มีเอกสารประกอบมาให้เลย ให้ตอบว่า {ABSTAIN} และอ้างอิงเป็น - เท่านั้น "
        "ห้ามอ้างอิงหมายเลขใด ๆ ทั้งสิ้น ข้อนี้สำคัญกว่าข้อ 4\n"
        "6. อ้างอิงได้เฉพาะหมายเลขที่ปรากฏจริงในเอกสารที่ให้มาข้างต้นเท่านั้น "
        "ห้ามสร้างหมายเลขขึ้นเอง",
}


def build_instructions(variant: str) -> str:
    return f"""คุณคือผู้ช่วยตอบคำถามจากมติที่ประชุมสภาวิชาการ

กติกา:
1. ตอบจากเอกสารที่ให้มาเท่านั้น ห้ามใช้ความรู้อื่น
2. อ้างอิงเอกสารที่ใช้ด้วยหมายเลขในวงเล็บเหลี่ยม เช่น [1] หรือ [2], [5]
3. ถ้าเอกสารที่ให้มาไม่มีคำตอบ ให้ตอบว่า {ABSTAIN} เท่านั้น ห้ามเดา
{_RULE4[variant]}

รูปแบบคำตอบ (ต้องมีสองบรรทัดนี้เสมอ):
คำตอบ: <คำตอบของคุณ>
อ้างอิง: [หมายเลข] หรือ - ถ้าไม่มี"""


def build_prompt(ctx: dict, variant: str = "sentence_cap") -> str:
    """Context first, instructions last.

    The pilot put the instructions first and got 0/4 citations on prompts that
    carried context, while getting 4/4 correct format on the short closed-book
    prompts -- same model, same instructions, so it was not a capability limit.
    Ollama truncates an over-long prompt from the *front*, so on long prompts the
    rules were being cut away before the model ever saw them. Raising num_ctx
    fixes the truncation; putting the rules after the documents fixes the recency
    problem underneath it, and costs nothing when the prompt does fit."""
    if ctx["blocks"]:
        docs = "\n\n".join(f"[{b['label']}] {b['text']}" for b in ctx["blocks"])
        body = f"เอกสาร:\n{docs}"
    else:
        body = "(ไม่มีเอกสารประกอบ)"
    return f"{body}\n\n{build_instructions(variant)}\n\nคำถาม: {ctx['query']}\n\nคำตอบ:"


# Every published RQ4 answer was generated with this model. It is the only model
# for which "sentence_cap" names a real, comparable baseline.
_BASELINE_MODEL = "phi4"

# The longest prompt in the current context set measures 14,721 tokens, so 8192 --
# the default until 2026-08-10 -- truncates silently, keeping the tail and deleting
# the highest-ranked documents. preflight() probes for exactly that, but it probes
# at most `max_probes` prompts and can be skipped, so the floor is enforced
# separately: a bound that holds for every prompt beats a probe of five of them.
_MIN_NUM_CTX = 16384


def _refuse_retired_variant(model: str, variant: str, allow: bool) -> None:
    """`sentence_cap` names a baseline that exists only for `phi4`.

    The pre-registered rule in `rq4_score.py` made a second generator worth
    running only if the flat ~0.41 citation recall survived the `cite_all`
    ablation. It did not, so a gemma4:e4b run under `sentence_cap` would spend
    ~5 GPU-hours reproducing a retired artifact: nothing scores that pair (the
    published families are baseline-vs-variant within `phi4`), and the 3-sentence
    cap is the very instruction the ablation showed to be the confound.

    Containment, not a ban -- `--allow-retired-variant` exists because the run is
    legitimate if someone deliberately wants a same-prompt cross-model reading.
    """
    if variant != "sentence_cap" or allow or model.split(":")[0] == _BASELINE_MODEL:
        return
    raise SystemExit(
        f"refusing to start: --variant sentence_cap with --model {model}.\n"
        f"sentence_cap is a retired baseline kept for {_BASELINE_MODEL}'s 530 "
        "answers; its rule 4 is the confound the cite_all ablation identified "
        "(recall rose for hybrid/dense/bm25 at Holm 0.0000), so the branch that "
        "made a second generator interesting never opened.\n"
        "Use --variant cite_all (the arm-ordering result) or cite_all_guarded "
        "(the paper's prompt), or pass --allow-retired-variant if you really "
        "mean to reproduce the retired pair."
    )


def _refuse_small_ctx(num_ctx: int, allow: bool) -> None:
    """Enforce the context floor for every prompt, not just the probed ones.

    preflight() already refuses to start on the truncation signature, but it
    probes at most `max_probes` candidates and `--skip-preflight` turns it off,
    so on its own it is a sample. This is the bound.
    """
    if num_ctx >= _MIN_NUM_CTX or allow:
        return
    raise SystemExit(
        f"refusing to start: --num-ctx {num_ctx:,} is below {_MIN_NUM_CTX:,}. "
        "The longest prompt in this context set measures 14,721 tokens, and "
        "ollama cuts an over-long prompt to num_ctx//2+2 keeping the TAIL, so "
        "the best-ranked documents are deleted and the rules survive -- there "
        "is no symptom in the answer. Raise --num-ctx, or pass "
        "--allow-small-ctx to reproduce a historical run on purpose."
    )


def supports_thinking(model: str) -> bool:
    """Whether ollama reports a `thinking` capability for `model`.

    Asked rather than assumed, because the answer decides how tokens are spent
    and the two models this project uses differ: `phi4:latest` reports
    `['completion']`, `gemma4:e4b` reports `[..., 'thinking']`. A thinking model
    left unconfigured emits a `thinking` field that `generate()` never reads --
    measured 2026-08-12 on one real RQ4 prompt, gemma4:e4b spends **1058 eval
    tokens / 27.2 s** unset against **243 / 4.9 s** at `think=False`, i.e. 77% of
    the generated tokens are discarded, a 5.6x slowdown. `llm_ocr_scan.py` has
    passed `think=False` since the OCR scan for exactly this reason; this script
    never did, which was harmless only while `phi4` was the only model used.

    Gated on the capability rather than sent unconditionally: `think=False` is in
    fact accepted by `phi4:latest` today (checked), but a flag that is a silent
    no-op for one model and load-bearing for another should be visible in the
    run's own record, which is why `generate()` writes both fields per answer.
    """
    try:
        return "thinking" in (getattr(ollama.show(model), "capabilities", None) or [])
    except Exception as exc:
        print(f"  [warn] cannot read capabilities for {model}: {exc}; assuming none")
        return False


def resident_models() -> list[str]:
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # ollama not on PATH is itself worth failing loudly on
        raise SystemExit(f"cannot query `ollama ps`: {exc}")
    return [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]


def unload(model: str) -> None:
    try:
        ollama.generate(model=model, prompt="", keep_alive=0)
        time.sleep(2)
    except Exception as exc:
        print(f"  [warn] unload {model}: {exc}")


def truncated_to(num_ctx: int) -> int:
    """The prompt-token count ollama reports when it HAS truncated.

    Measured on 2026-08-10 against ollama 0.32.6, not read from the docs: one
    14,721-token prompt reports 2050 / 4098 / 14721 at num_ctx 4096 / 8192 /
    16384, and eight shorter prompts are fed whole at 8192. So the rule is
    "fits => whole, exceeds => num_ctx/2, keeping the tail", and this exact
    value in `prompt_eval_count` is the truncation signature.
    """
    return num_ctx // 2 + 2


def token_upper_bound(text: str) -> int:
    """Most tokens `text` can possibly tokenize to: its length in UTF-8 bytes.

    Every token of a byte-level BPE vocabulary consumes at least one byte, so
    tokens <= bytes holds for any text and any such tokenizer. It is loose --
    Thai is 3 bytes/char at ~1.0 chars/token, so ~3x -- and loose is the only
    direction a safety screen may err in.

    **This replaced `int(n_chars / 1.046) + 1` on 2026-08-10, and the reason is
    that 1.046 was not a bound.** It was documented as this corpus's lowest
    chars/token, measured on the two entity arms; over the 228 prompts screened
    by `rq4_find_truncated_answers.py` **15 fall below it**, the minimum being
    **1.0098** (`bm25_semantic/q001`, 11,208 chars / 11,099 tokens). An unsound
    "upper bound" does not merely mis-sort candidates for probing -- it removes
    prompts from the candidate list entirely, which is exactly how two
    reconstruction runs missed `hybrid_m2v_semantic/q025` (8,475 tokens).
    An empirical extreme is a description of a sample, never a bound on the next
    input ([[feedback_an_asserted_invariant_is_not_a_check]]).
    """
    return len(text.encode("utf-8"))


def preflight(model: str, arms: list[str], variant: str, num_ctx: int,
              max_probes: int = 5, chat_kwargs: dict | None = None) -> None:
    """Refuse to start if any prompt does not fit `num_ctx`.

    The old `--num-ctx` help already *said* it must exceed the longest prompt.
    Nothing checked, and the default 8192 was in fact exceeded by prompts up to
    ~14.7k tokens, so every long prompt lost its highest-ranked documents (the
    cut keeps the tail, and blocks are laid out best-first).

    **Screen by an upper bound, not by picking the longest prompt.** The first
    version measured the single longest prompt *in characters* and cleared the
    run on that one result. That is unsound in a way this corpus actually
    exhibits: chars/token spans 1.046 (Thai) to 3.151 (English course tables),
    so the longest-in-characters prompt need not be the longest in tokens. On
    the two entity arms it is not even close -- the longest by characters is
    15,689 chars / 4,860 tokens, while the true worst is 14,721 tokens. At
    num_ctx=8192 the old screen would have measured 4,860, declared "fits", and
    started a run in which ~45-50% of prompts were silently truncated.

    So: every prompt whose *upper bound* fits is provably safe and needs no
    forward pass; only prompts that could exceed `num_ctx` are probed, largest
    first. When nothing can exceed it, the run is cleared without touching the
    GPU at all.

    Ordering by the bound is a heuristic (an upper bound orders by its own
    slack, not by truth), and only `max_probes` candidates are probed, so
    preflight is a cheap early exit -- **the sound guard is the per-answer
    `prompt_eval_count` check in `generate()`**, which sees every prompt as it
    is actually sent.
    """
    prompts = []  # (bound_tokens, label, text)
    for arm in arms:
        for path in sorted((_CONTEXTS / arm).glob("q*.json")):
            p = build_prompt(json.loads(path.read_text(encoding="utf-8")), variant)
            prompts.append((token_upper_bound(p), f"{arm}/{path.name}", p))
    if not prompts:
        return

    candidates = [t for t in prompts if t[0] > num_ctx]
    candidates.sort(reverse=True)
    biggest = max(prompts)
    print(f"preflight: {len(prompts)} prompts, longest {len(biggest[2]):,} chars "
          f"(<= {biggest[0]:,} tokens); {len(candidates)} could "
          f"exceed num_ctx={num_ctx:,}")
    if not candidates:
        print("preflight: no prompt can exceed num_ctx -- cleared without probing")
        return

    for n_bound, label, text in candidates[:max_probes]:
        n = ollama.chat(model=model, messages=[{"role": "user", "content": text}],
                        options={"temperature": 0.0, "num_ctx": num_ctx,
                                 "num_predict": 1},
                        **(chat_kwargs or {}))["prompt_eval_count"]
        print(f"preflight: {label} = {len(text):,} chars (<= {n_bound:,} tok) "
              f"-> {n:,} prompt tokens")
        if n == truncated_to(num_ctx):
            raise SystemExit(
                f"refusing to start: {label} reported prompt_eval_count {n:,}, "
                f"exactly num_ctx//2+2, the truncation signature. Raise "
                f"--num-ctx (try {num_ctx * 2:,}) or lower --k / --max-chars in "
                f"rq4_build_contexts.py."
            )
    if len(candidates) > max_probes:
        print(f"preflight: {len(candidates) - max_probes} further candidate(s) "
              f"left to the per-answer guard")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="", help="comma-separated; default all")
    ap.add_argument("--limit", type=int, default=0, help="first N queries per arm (pilot)")
    ap.add_argument("--out", default=str(_OUT))
    ap.add_argument("--num-ctx", type=int, default=16384,
                    help="context window. MUST exceed the longest prompt in TOKENS: "
                    "ollama feeds a fitting prompt whole but cuts an over-long one "
                    "to num_ctx/2, keeping the tail (see build_prompt and the "
                    "pre-flight below). Raised 8192 -> 16384 on 2026-08-10, when "
                    "the pre-flight measured the longest prompt at 14,721 tokens. "
                    f"Values below {_MIN_NUM_CTX:,} are refused: see --allow-small-ctx.")
    ap.add_argument("--allow-small-ctx", action="store_true",
                    help=f"permit --num-ctx below {_MIN_NUM_CTX:,}. Only for "
                    "deliberately reproducing a historical run (the 8192 era); a "
                    "real generation at 8192 truncates the longest prompts silently")
    ap.add_argument("--allow-retired-variant", action="store_true",
                    help="permit --variant sentence_cap with a non-phi4 model "
                    "(reproduces a retired artifact; see _refuse_retired_variant)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="do not measure the longest prompt first (do not use)")
    ap.add_argument("--allow-resident", action="store_true",
                    help="skip the resident-model guard (do not use on a 12 GB card)")
    ap.add_argument("--variant", default="sentence_cap", choices=sorted(_RULE4),
                    help="which rule-4 wording to use. Non-default variants write to "
                    "a separate answers/<model>_<variant>/ dir so they never clobber "
                    "the baseline run.")
    args = ap.parse_args()

    _refuse_retired_variant(args.model, args.variant, args.allow_retired_variant)
    _refuse_small_ctx(args.num_ctx, args.allow_small_ctx)

    resident = resident_models()
    if resident and not args.allow_resident:
        raise SystemExit(
            f"refusing to start: {resident} already resident in VRAM.\n"
            "Two full-size models will not fit on this card and may spill to CPU "
            "silently. Unload first, or pass --allow-resident if you are sure."
        )

    arms = [a for a in (args.arms.split(",") if args.arms else
                        sorted(p.name for p in _CONTEXTS.iterdir() if p.is_dir())) if a]
    thinking = supports_thinking(args.model)
    chat_kwargs = {"think": False} if thinking else {}
    print(f"model={args.model}  arms={arms}  limit={args.limit or 'all'}  "
          f"num_ctx={args.num_ctx}  variant={args.variant}  "
          f"thinking={'supported -> disabled' if thinking else 'not supported'}")

    model_dir = args.model.replace(":", "_")
    if args.variant != "sentence_cap":
        model_dir += f"_{args.variant}"

    if not args.skip_preflight:
        preflight(args.model, arms, args.variant, args.num_ctx,
                  chat_kwargs=chat_kwargs)

    t_start, truncated = time.time(), 0
    for arm in arms:
        files = sorted((_CONTEXTS / arm).glob("q*.json"))
        if args.limit:
            files = files[: args.limit]
        out_dir = Path(args.out) / model_dir / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        done = skipped = 0
        t0 = time.time()

        for path in files:
            dst = out_dir / path.name
            if dst.exists():
                skipped += 1
                continue
            ctx = json.loads(path.read_text(encoding="utf-8"))
            t1 = time.time()
            try:
                resp = ollama.chat(
                    model=args.model,
                    messages=[{"role": "user", "content": build_prompt(ctx, args.variant)}],
                    options={"temperature": 0.0, "num_ctx": args.num_ctx},
                    **chat_kwargs,
                )
                answer, error = resp["message"]["content"].strip(), None
                n_prompt = resp.get("prompt_eval_count")
            except Exception as exc:
                answer, error, n_prompt = "", str(exc), None
            if n_prompt == truncated_to(args.num_ctx):
                truncated += 1
                print(f"  [truncated] {arm}/{path.name}: fed {n_prompt:,} tokens "
                      f"(num_ctx//2+2) -- the front of this prompt was discarded")

            dst.write_text(json.dumps({
                "query": ctx["query"],
                "arm": arm,
                "model": args.model,
                "variant": args.variant,
                "entity_type": ctx["entity_type"],
                "relevant_resolution_ids": ctx["relevant_resolution_ids"],
                # label -> resolution_id, so scoring maps a cited [n] back without
                # re-reading the context file
                "label_map": {str(b["label"]): b["resolution_id"] for b in ctx["blocks"]},
                "context_has_gold": ctx["context_has_gold"],
                "answer": answer,
                "error": error,
                # recorded so truncation is auditable after the fact rather than
                # only at run time -- the 8192 runs have no such field, which is
                # why their damage had to be re-measured prompt by prompt
                "num_ctx": args.num_ctx,
                "prompt_eval_count": n_prompt,
                # same reasoning as num_ctx: a generation setting that changes the
                # answer must be readable off the answer, not inferred from the
                # date. `think=False` is not cosmetic -- on gemma4:e4b it changes
                # the wording, not just the discarded `thinking` field.
                "thinking_supported": thinking,
                "thinking_disabled": bool(chat_kwargs),
                "seconds": round(time.time() - t1, 2),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            done += 1
            if done % 10 == 0:
                print(f"  [{arm}] {done}/{len(files) - skipped}  "
                      f"{(time.time() - t0) / done:.1f}s/query")

        print(f"[{arm}] {done} generated, {skipped} already present, "
              f"{time.time() - t0:.0f}s total")

    unload(args.model)
    print(f"\nunloaded {args.model}; total {time.time() - t_start:.0f}s")
    print(f"answers -> {Path(args.out) / model_dir}")
    if truncated:
        print(f"\n!! {truncated} prompt(s) were TRUNCATED at num_ctx={args.num_ctx}: "
              "each lost its front, i.e. its highest-ranked documents. Re-run those "
              f"answers at --num-ctx {args.num_ctx * 2} before scoring.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
