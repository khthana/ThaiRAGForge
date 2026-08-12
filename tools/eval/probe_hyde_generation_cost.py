"""Price one HyDE generation, so the axis is costed by measurement not by analogy.

This is **not** a HyDE implementation and nothing imports it. It answers one
question -- *how long would running the HyDE axis take* -- before any of it is
built, because the figure this project had on file could not survive being
looked at: `docs/hyde-axis-notes.md` inherited **15.6 s/query** from the RQ4
generation log, and an RQ4 prompt carries ~8k tokens of retrieved context while
a HyDE prompt carries only the query. A number measured on one prompt shape
does not transfer to another, however similar the model and the machine
([[feedback_state_the_input_size_with_any_timing]]).

What it measures, and why in this shape:

* **Two variants per query, uncapped and `num_predict=256`.** The uncapped arm
  is the honest cost of the obvious implementation; the capped arm exists
  because the first probe found the model spending 576-808 output tokens on a
  prompt that asks for at most five sentences. If the cost is output-bound the
  cap is nearly free to adopt, and *which* it is decides whether the axis costs
  half an hour or two. Reporting only the uncapped figure would have priced the
  axis at twice what it needs to cost.
* **Queries spread across `entity_type`**, not the first N of the gold file --
  those are all `program`, and a HyDE paragraph about a curriculum is a
  different generation length from one about a person.
* **The first call is reported separately.** It pays the model load (~9 GB) and
  is not the per-query cost of a 106-query run; folding it into the mean would
  inflate a 106-query estimate by the one-off.

The timings are a *price*, not a fixture. `temperature=0` is not reproducible
here ([[feedback_temperature_zero_is_not_reproducible]]), so the output-token
counts -- and therefore the wall times -- move between runs; the figure that
transfers is the tokens/s and the ratio between the two variants.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import ollama
import yaml

REPO = Path(__file__).resolve().parents[2]
GOLD = REPO / "config/eval/gold_query_set_73det.yaml"
OUT = REPO / "data/results/hyde_generation_cost.md"
# Every run's output-token counts, appended. The reproducibility sentence in the
# report is derived from this file rather than typed, because a cross-run claim
# is exactly the kind that is true the day it is written and never re-checked
# ([[feedback_verify_status_numbers_against_the_artifact]]).
RUNS = REPO / "data/results/hyde_generation_cost_runs.json"

MODEL = "phi4"
CAP = 256  # num_predict for the capped arm

# A plausible HyDE prompt for this corpus: write the passage that WOULD answer
# the question, in the register of the minutes, so the embedding lands near real
# minutes text. Deliberately unengineered -- this prices the axis, it does not
# tune it.
PROMPT = """คุณคือผู้ช่วยที่เขียนร่างข้อความในรายงานการประชุมสภาวิชาการของมหาวิทยาลัย

จงเขียนย่อหน้าสั้น ๆ (ไม่เกิน 5 ประโยค) ที่มีลักษณะเหมือนข้อความในรายงานการประชุม
ซึ่งน่าจะเป็นคำตอบของคำถามข้างล่างนี้ ไม่ต้องอธิบายอะไรเพิ่ม ตอบเป็นย่อหน้าเดียว

คำถาม: {q}

ย่อหน้า:"""


def pick_queries(n: int) -> list[dict]:
    """One query per `entity_type`, in file order, up to `n`."""
    gold = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    by_type: dict[str, dict] = {}
    for row in gold:
        by_type.setdefault(row.get("entity_type", "?"), row)
    return list(by_type.values())[:n]


def run_one(query: str, num_predict: int | None) -> dict:
    opts = {"temperature": 0.0, "num_ctx": 8192}
    if num_predict is not None:
        opts["num_predict"] = num_predict
    t0 = time.perf_counter()
    resp = ollama.generate(model=MODEL, prompt=PROMPT.format(q=query), options=opts)
    wall = time.perf_counter() - t0
    return {
        "wall": wall,
        "prompt_tok": resp.get("prompt_eval_count"),
        "out_tok": resp.get("eval_count"),
        "text": (resp.get("response") or "").strip(),
    }


def history_summary() -> tuple[int, list[str], int, int]:
    """(n_runs, disagreements, #distinct signatures, leading identical streak).

    A key is `(entity_type, cap)`; a *signature* is a whole run's map of them.
    The **streak** -- how many runs from the start share the first run's
    signature -- is what makes the "N identical runs, then a different one"
    shape visible. It is derived rather than typed because that streak is
    precisely the evidence that would have licensed the opposite claim: after
    the third identical run, "this prompt family is deterministic" looked safe
    to write down, and the fourth disagreed on every query.
    """
    history = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else []
    keys = {k for h in history for k in h["out_tok"]}
    disagree = []
    for key in sorted(keys):
        seen = {h["out_tok"][key] for h in history if key in h["out_tok"]}
        if len(seen) > 1:
            disagree.append(f"{key} -> {sorted(seen)}")
    sigs = [json.dumps(h["out_tok"], sort_keys=True) for h in history]
    streak = 0
    for sig in sigs:
        if sig != sigs[0]:
            break
        streak += 1
    return len(history), disagree, len(set(sigs)), streak


def record_run(rows: list[dict]) -> tuple[int, list[str], int, int]:
    """Append this run's output-token counts, then summarise the history.

    A disagreement is reported, never silenced: if generation stops being
    reproducible on this prompt family the report has to say so, since the
    per-query time is `out_tok / throughput` and only throughput is stable.
    """
    history = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else []
    this = {f"{r['entity_type']}|{r['cap']}": r["out_tok"] for r in rows}
    # `rows` is kept whole so `--render` can rebuild the report without a GPU;
    # `out_tok` stays a flat map because the reproducibility check reads it and
    # the two seeded entries predate the timings being persisted.
    history.append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "out_tok": this,
                    "rows": [{k: v for k, v in r.items() if k != "text"} for r in rows]})
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    RUNS.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return history_summary()


def render(rows: list[dict], n_gold: int, n_thematic: int,
           n_runs: int, disagree: list[str], n_sigs: int, streak: int) -> str:
    uncapped = [r for r in rows if r["cap"] is None]
    capped = [r for r in rows if r["cap"] is not None]
    warm = uncapped[1:] or uncapped  # first call pays the model load

    def mean(xs: list[float]) -> float:
        return statistics.fmean(xs)

    m_un = mean([r["wall"] for r in warm])
    m_cap = mean([r["wall"] for r in capped])
    tps = mean([r["out_tok"] / r["wall"] for r in warm if r["out_tok"]])

    L: list[str] = []
    w = L.append
    w("# HyDE — ราคาของการ generate หนึ่งคำถาม")
    w("")
    w(f"Generated by `tools/eval/probe_hyde_generation_cost.py` · model `{MODEL}` · "
      f"{len(uncapped)} คำถามจริงจาก `gold_query_set_73det.yaml` (คนละ `entity_type`) · "
      f"temperature 0 · num_ctx 8,192")
    w("")
    w("**นี่คือการตั้งราคา ไม่ใช่การทดลอง HyDE** — สคริปต์นี้ไม่ได้ implement HyDE "
      "และไม่มีอะไร import มัน มันตอบคำถามเดียวคือ *ถ้าจะทำ axis นี้ต้องใช้เวลาเท่าไร* "
      "ก่อนจะเขียนอะไรสักบรรทัด")
    w("")
    w("**ทำไมต้องวัดใหม่ทั้งที่มีตัวเลขอยู่แล้ว** — บันทึกเดิมยกเลข **15.6 วินาที/คำถาม** "
      "มาจาก log ของ RQ4 แต่ prompt ของ RQ4 แบก context ที่ retrieve มา ~8,000 token "
      "ส่วน prompt ของ HyDE มีแค่คำถาม (~300 token) เวลาที่วัดจาก prompt รูปหนึ่ง "
      "ใช้กับอีกรูปหนึ่งไม่ได้ แม้จะเป็นโมเดลเดียวกันบนเครื่องเดียวกัน")
    w("")
    w("## 1. ต่อคำถาม")
    w("")
    w("| # | entity_type | num_predict | wall (s) | prompt tok | out tok | tok/s |")
    w("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        cap = "—" if r["cap"] is None else str(r["cap"])
        tok_s = (r["out_tok"] / r["wall"]) if r["out_tok"] else float("nan")
        note = " *(cold — โหลดโมเดล)*" if i == 1 else ""
        w(f"| {i} | {r['entity_type']}{note} | {cap} | {r['wall']:.2f} | "
          f"{r['prompt_tok']} | {r['out_tok']} | {tok_s:.1f} |")
    w("")
    w(f"**warm mean (ไม่ cap) = {m_un:.2f} s** · **cap {CAP} token = {m_cap:.2f} s** · "
      f"throughput ≈ {tps:.1f} tok/s")
    w("")
    w("## 2. ข้อค้นพบที่เปลี่ยนราคา")
    w("")
    w("ต้นทุนเกือบทั้งหมดเป็น **output** ไม่ใช่ prompt: prompt ~300 token แต่โมเดลเขียนออกมา "
      f"{min(r['out_tok'] for r in uncapped)}–{max(r['out_tok'] for r in uncapped)} token "
      "ทั้งที่ prompt สั่งว่า “ไม่เกิน 5 ประโยค” — คำสั่งความยาวที่เขียนเป็นภาษาคนไม่ได้บังคับอะไร "
      f"การใส่ `num_predict={CAP}` จึงลดเวลาลงเหลือราว {m_cap / m_un:.0%} "
      "และยัง**ตรง**กับเจตนาของ HyDE มากกว่า (เอกสารสมมติสั้น ๆ ไม่ใช่เรียงความ)")
    w("")
    w("## 3. ประเมินทั้ง axis")
    w("")
    w("| ชุดคำถาม | generate (ไม่ cap) | generate (cap) |")
    w("|---|---|---|")
    for name, n in (("73det", n_gold), ("thematic", n_thematic)):
        w(f"| {name} ({n:,} คำถาม) | {n * m_un / 60:.0f} นาที | {n * m_cap / 60:.0f} นาที |")
    w("")
    w("**generate ครั้งเดียวใช้ได้กับทุก embedder** เพราะ HyDE เป็น query transform ล้วน ๆ "
      "ไม่ขึ้นกับ index — cache ผลลง JSON แล้ว sweep กี่ combo ก็ไม่ต้อง generate ใหม่ "
      "และ**ไม่ต้อง rebuild index เลย** ต้นทุนที่เหลือคือ retrieval pass ตามปกติ "
      "(ราว 20–40 นาทีต่อชุด) กับการเขียนโค้ด ซึ่งเป็นก้อนที่ใหญ่กว่า GPU")
    w("")
    w("## 4. อ่านตัวเลขนี้ยังไง")
    w("")
    tps_all = [r["out_tok"] / r["wall"] for r in rows if r["out_tok"]]
    w(f"**สิ่งที่ยกไปใช้ต่อได้คือ throughput ไม่ใช่ “วินาทีต่อคำถาม”** — tok/s นิ่งมาก "
      f"({min(tps_all):.1f}–{max(tps_all):.1f} ในรอบนี้ ทั้งแบบ cap และไม่ cap) "
      "แต่เวลาต่อคำถาม = `out_tok / throughput` และ `out_tok` คือสิ่งที่โมเดลเลือกเอง "
      "ประมาณการในตารางที่ 3 จึงเป็น **ราคา ไม่ใช่ค่าคงที่**")
    w("")
    if n_runs <= 1:
        repro = "probe ตัวนี้เพิ่งรันรอบแรก ยังไม่มีข้อมูลว่ารันซ้ำแล้วได้เท่าเดิมหรือไม่"
    elif not disagree:
        repro = (f"รันมาแล้ว **{n_runs} รอบ** จำนวน output token ตรงกันทุกรอบทุกคำถาม "
                 f"({' / '.join(str(r['out_tok']) for r in uncapped)}) — "
                 "ซึ่ง**ยังไม่ใช่หลักฐานว่า deterministic** ดูย่อหน้าถัดไป")
    else:
        repro = (f"รันมาแล้ว **{n_runs} รอบ** ได้ลายเซ็น output token **{n_sigs} แบบ** "
                 f"โดย {streak} รอบแรกเหมือนกันเป๊ะ แล้วจึงต่าง — ต่างกันดังนี้: "
                 + "; ".join(disagree))
    w("**`temperature=0` ไม่การันตีว่ารันซ้ำแล้วได้ token เท่าเดิม แม้กับ prompt สั้น ๆ แบบนี้** "
      "(ประโยคนี้ derive จาก `data/results/hyde_generation_cost_runs.json` ไม่ได้พิมพ์มือ): "
      + repro)
    if disagree and streak >= 3:
        w("")
        w(f"บทเรียนอยู่ที่รูปร่างของมัน ไม่ใช่ที่ข้อสรุป: **{streak} รอบแรกเหมือนกันเป๊ะทุกคำถาม** "
          "ซึ่งมากพอจะเขียนลงเอกสารได้อย่างสบายใจว่า “prompt สั้นแบบนี้ deterministic” "
          f"แล้วรอบที่ {streak + 1} ก็ต่างทุกคำถาม — **รันซ้ำได้เท่าเดิมหลายรอบไม่ใช่หลักฐานว่า deterministic** "
          "เหตุผลเดียวที่รายงานนี้ไม่ได้พูดผิดคือประโยคข้างบนถูกให้สคริปต์ derive แทนที่จะพิมพ์มือ "
          "(RQ4 เจอข้อเดียวกันจากอีกทาง: รันซ้ำ prompt เดิม ได้ชุด citation เหมือนเดิมแค่ 14/24, "
          "[[feedback_temperature_zero_is_not_reproducible]])")
    w("")
    w("ตัวเลขนี้ไม่ได้บอกว่า HyDE **ควร**ทำ — คำทำนายที่ลงทะเบียนไว้ใน "
      "`docs/hyde-axis-notes.md` ยังบอกว่ามันน่าจะ*แย่ลง*บน 73det บอกแค่ว่าถ้าทำ จะจ่ายเท่าไร")
    return "\n".join(L) + "\n"


def query_set_sizes() -> tuple[int, int]:
    n_gold = len(yaml.safe_load(GOLD.read_text(encoding="utf-8")))
    thematic = REPO / "config/eval/gold_query_set.yaml"
    n_thematic = 179
    if thematic.exists():
        rows_t = yaml.safe_load(thematic.read_text(encoding="utf-8")) or []
        n_thematic = sum(1 for r in rows_t if r.get("entity_type") in (None, "thematic")) or 179
    return n_gold, n_thematic


def main() -> int:
    # Console output is ASCII on purpose: this machine's console is cp874 and a
    # stray "." separator killed a run *after* the GPU work and *before* the
    # report was written. The report itself is written as UTF-8.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=4, help="how many queries (one per entity_type)")
    ap.add_argument("--render", action="store_true",
                    help="rebuild the report from the last recorded run (no GPU)")
    ap.add_argument("--keep-loaded", action="store_true",
                    help="leave the model resident (default: unload to free VRAM)")
    args = ap.parse_args()

    n_gold, n_thematic = query_set_sizes()

    if args.render:
        history = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else []
        last = next((h for h in reversed(history) if h.get("rows")), None)
        if last is None:
            print("no recorded run carries timings yet -- run without --render once")
            return 1
        n_runs, disagree, n_sigs, streak = history_summary()
        OUT.write_text(
            render(last["rows"], n_gold, n_thematic, n_runs, disagree, n_sigs, streak),
            encoding="utf-8")
        print(f"rendered from run {last['at']} ({n_runs} runs recorded)")
        print(f"wrote {OUT.relative_to(REPO)}")
        return 0

    picks = pick_queries(args.n)
    print(f"model={MODEL}  n={len(picks)}  (uncapped + num_predict={CAP} per query)")
    rows: list[dict] = []
    for cap in (None, CAP):
        for row in picks:
            r = run_one(row["query"], cap)
            r["cap"] = cap
            r["entity_type"] = row.get("entity_type", "?")
            rows.append(r)
            print(f"  cap={cap}  type={r['entity_type']:<28} wall={r['wall']:6.2f}s  "
                  f"prompt={r['prompt_tok']}  out={r['out_tok']}")

    n_runs, disagree, n_sigs, streak = record_run(rows)
    print(f"\nruns recorded: {n_runs} ({n_sigs} distinct signatures, "
          f"first {streak} identical) - output tokens "
          + ("agree across all runs" if not disagree else "DISAGREE: " + "; ".join(disagree)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(rows, n_gold, n_thematic, n_runs, disagree, n_sigs, streak),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")

    if not args.keep_loaded:
        ollama.generate(model=MODEL, prompt="", keep_alive=0)
        print("unloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
