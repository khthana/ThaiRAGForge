"""Status / purge helper for the supervised RQ4 refresh.

Two jobs, both about the same hazard: `rq4_generate.py` writes an answer file
even when generation FAILED (answer="", error=<exc>), and an existing file is
skipped on resume -- so a stalled cell would be frozen as an empty answer and
scored as one. So (a) completion is counted over answers with error=None, never
over file count, and (b) --purge deletes the error-carrying ones so the next
attempt regenerates them.
"""
import json, os, sys

ARMS = ["hybrid_qwen3_0.6b_semantic", "dense_qwen3_0.6b_semantic",
        "bm25_semantic", "hybrid_m2v_semantic"]
N = 106
BASE = "data/rq4/answers"


def scan(vdir):
    ok = bad = capped = 0
    bad_files = []
    for arm in ARMS:
        d = os.path.join(BASE, vdir, arm)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            p = os.path.join(d, f)
            try:
                a = json.load(open(p, encoding="utf-8"))
            except Exception:
                bad += 1; bad_files.append(p); continue
            if a.get("error") or not a.get("answer"):
                bad += 1; bad_files.append(p)
            else:
                ok += 1
                # generation stopped at the num_predict cap, so the answer is cut
                # and may have lost its citation line -- countable, not silent
                if a.get("done_reason") == "length":
                    capped += 1
    return ok, bad, bad_files, capped


def main():
    vdir = sys.argv[1]
    purge = "--purge" in sys.argv
    ok, bad, files, capped = scan(vdir)
    if purge:
        for p in files:
            os.remove(p)
        print(f"purged {len(files)} error-carrying answer(s) from {vdir}")
        ok, bad, files, capped = scan(vdir)
    target = N * len(ARMS)
    print(f"{vdir}: {ok}/{target} valid, {bad} bad, {capped} capped at num_predict")
    return 0 if ok >= target and bad == 0 else 1


sys.exit(main())
