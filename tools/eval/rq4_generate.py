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

**Temperature 0.** One pass, no sampling variance to average over. Worth stating
because a nonzero temperature would silently turn every arm comparison into a
noisy one and invite averaging runs that were never run.

The prompt asks for a fixed two-line shape so 4a is parseable at all:

    คำตอบ: <answer, or the abstention token>
    อ้างอิง: [1], [3]

Abstention is a *first-class* output, not a failure: 4b's whole point is whether
the model declines when the context lacks the answer, and the closed-book arm has
no context by construction, so ไม่พบข้อมูล is the correct answer there 106 times
out of 106. Nothing in the prompt tells the model which arm it is in.

Run (pilot first -- the generator choice is an open decision in the design doc):
    PYTHONPATH=src python tools/eval/rq4_generate.py --model phi4 --limit 20
    PYTHONPATH=src python tools/eval/rq4_generate.py --model gemma4:e4b --limit 20
    PYTHONPATH=src python tools/eval/rq4_generate.py --model <winner>
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
# "cite_all" is the pending ablation: if recall rises, the flat line was a prompt
# artifact; if it stays ~0.41, it's a real generator ceiling worth testing gemma4:e4b
# against.
_RULE4 = {
    "sentence_cap": "4. ตอบสั้น ๆ ไม่เกิน 3 ประโยค",
    "cite_all": "4. อ้างอิงเอกสารที่เกี่ยวข้องทุกฉบับที่พบในเอกสารที่ให้มา ไม่ใช่แค่ฉบับเดียว "
                "ความยาวคำตอบไม่จำกัด ตราบใดที่ครอบคลุมทุกฉบับที่เกี่ยวข้อง",
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="", help="comma-separated; default all")
    ap.add_argument("--limit", type=int, default=0, help="first N queries per arm (pilot)")
    ap.add_argument("--out", default=str(_OUT))
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="context window. MUST exceed the longest prompt: ollama "
                    "truncates from the front, which silently removes the "
                    "instructions and makes 4a unmeasurable (see build_prompt).")
    ap.add_argument("--allow-resident", action="store_true",
                    help="skip the resident-model guard (do not use on a 12 GB card)")
    ap.add_argument("--variant", default="sentence_cap", choices=sorted(_RULE4),
                    help="which rule-4 wording to use. Non-default variants write to "
                    "a separate answers/<model>_<variant>/ dir so they never clobber "
                    "the baseline run.")
    args = ap.parse_args()

    resident = resident_models()
    if resident and not args.allow_resident:
        raise SystemExit(
            f"refusing to start: {resident} already resident in VRAM.\n"
            "Two full-size models will not fit on this card and may spill to CPU "
            "silently. Unload first, or pass --allow-resident if you are sure."
        )

    arms = [a for a in (args.arms.split(",") if args.arms else
                        sorted(p.name for p in _CONTEXTS.iterdir() if p.is_dir())) if a]
    print(f"model={args.model}  arms={arms}  limit={args.limit or 'all'}  "
          f"num_ctx={args.num_ctx}  variant={args.variant}")

    model_dir = args.model.replace(":", "_")
    if args.variant != "sentence_cap":
        model_dir += f"_{args.variant}"

    t_start = time.time()
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
                )
                answer, error = resp["message"]["content"].strip(), None
            except Exception as exc:
                answer, error = "", str(exc)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
