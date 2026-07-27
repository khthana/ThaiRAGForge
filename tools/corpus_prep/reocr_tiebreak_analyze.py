"""Analysis step for the round-2 tie-break: joint (round1_lean, round2_lean)
distribution over the 137-page pool, plus the round1-vs-round2 new_text
length-delta check the advisor flagged as a blocking gap (round 2's verdict
is cast about round 2's OCR text, not round 1's -- if the two texts diverge,
a round1==round2=="new" agreement isn't evidence for applying round 1's
staged text). Read-only, writes nothing to the real pipeline files.

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/reocr_tiebreak_analyze.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
SCAN_DIR = CORPUS_ROOT / "llm_ocr_scan"

POOL_FILE = Path(
    r"C:\Users\Terry\AppData\Local\Temp\claude\C--Users-Terry-Desktop-Code-RAG"
    r"\5cc0badb-64ee-467a-b45f-a68268ee9e38\scratchpad\tiebreak_r2_pool.json"
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def round_lean(verdicts: dict) -> str:
    votes = [v.get("verdict") for v in verdicts.values()]
    new_votes = votes.count("new")
    old_votes = votes.count("old")
    if new_votes > old_votes:
        return "new"
    if old_votes > new_votes:
        return "old"
    return "neutral"


def main() -> None:
    pool = json.loads(POOL_FILE.read_text(encoding="utf-8"))
    pool_keys = {(it["pdf"], it["page"]) for it in pool}
    print(f"[INFO] pool size: {len(pool_keys)}")

    r1_adj = {(r["pdf"], r["page"]): r for r in load_jsonl(SCAN_DIR / "reocr_adjudication.jsonl")}
    r2_adj = {(r["pdf"], r["page"]): r for r in load_jsonl(SCAN_DIR / "reocr_tiebreak_r2_adjudication.jsonl")}
    r1_staging = {(r["pdf"], r["page"]): r["new_text"] for r in load_jsonl(SCAN_DIR / "reocr_pages_staging.jsonl")}
    r2_staging = {(r["pdf"], r["page"]): r["new_text"] for r in load_jsonl(SCAN_DIR / "reocr_tiebreak_r2_staging.jsonl")}

    joint = Counter()
    rows = []
    missing_r1 = missing_r2 = 0
    for key in pool_keys:
        r1 = r1_adj.get(key)
        r2 = r2_adj.get(key)
        if r1 is None:
            missing_r1 += 1
            continue
        if r2 is None:
            missing_r2 += 1
            continue
        r1_lean = round_lean(r1.get("verdicts", {}))
        r2_lean = round_lean(r2.get("verdicts", {}))
        joint[(r1_lean, r2_lean)] += 1

        t1 = r1_staging.get(key, "")
        t2 = r2_staging.get(key, "")
        len1, len2 = len(t1), len(t2)
        denom = max(len1, len2, 1)
        rel_delta = abs(len1 - len2) / denom
        rows.append({
            "pdf": key[0], "page": key[1],
            "r1_lean": r1_lean, "r2_lean": r2_lean,
            "r1_len": len1, "r2_len": len2, "rel_delta": round(rel_delta, 4),
        })

    print(f"[INFO] missing round1 adjudication: {missing_r1}, missing round2: {missing_r2}")
    print("\n[JOINT DISTRIBUTION] (round1_lean, round2_lean) -> count")
    for k, v in sorted(joint.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")

    agree_directional = [r for r in rows if r["r1_lean"] == r["r2_lean"] and r["r1_lean"] != "neutral"]
    disagree_directional = [r for r in rows if {r["r1_lean"], r["r2_lean"]} == {"new", "old"}]
    other = [r for r in rows if r not in agree_directional and r not in disagree_directional]

    print(f"\n[SUMMARY] agree (new/new or old/old): {len(agree_directional)}")
    print(f"[SUMMARY] true disagreement (new vs old): {len(disagree_directional)}")
    print(f"[SUMMARY] other (any neutral involved, not pure disagreement): {len(other)}")

    print("\n[TEXT DELTA] r1_new_text vs r2_new_text length rel_delta, for agree_directional rows:")
    for r in agree_directional:
        flag = " <-- DIVERGENT TEXT" if r["rel_delta"] > 0.01 else ""
        print(f"  [{r['r1_lean']}/{r['r2_lean']}] rel_delta={r['rel_delta']:.4f} "
              f"len1={r['r1_len']} len2={r['r2_len']} {Path(r['pdf']).name} p{r['page']}{flag}")

    out_path = Path(
        r"C:\Users\Terry\AppData\Local\Temp\claude\C--Users-Terry-Desktop-Code-RAG"
        r"\5cc0badb-64ee-467a-b45f-a68268ee9e38\scratchpad\tiebreak_r2_analysis.json"
    )
    out_path.write_text(json.dumps({
        "joint": {f"{k[0]}|{k[1]}": v for k, v in joint.items()},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[INFO] wrote full analysis to {out_path}")


if __name__ == "__main__":
    main()
