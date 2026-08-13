"""Check `ColbertEncoder` against pylate's ColBERT on identical texts.

WHY THIS EXISTS
---------------
`qualify_colbert_model.py` runs 11 checks and every one of them compares the
model **against itself** -- a control built from the same weights, a reversal of
the same ids, a norm of its own output. So a convention that is uniformly wrong
on *both* sides of every internal comparison cancels and is invisible to all 11.
That is not hypothetical: this cross-check is what found `mask_punctuation`
masking whitespace and no punctuation at all (see §3), after the 11-check gate
had passed the encoder.

The reference must be **correct by construction, not merely independent.**
pylate under transformers 5.3.0 hits the same uninitialised-`inv_freq` bug the
repo does (24 of 24 layers), so comparing against it would be comparing two
independently-broken models -- and since the garbage differs per load, they are
not even the *same* broken model. Pinned to transformers 4.53.2 the buffer loads
correctly (0 of 24) and the comparison means something. Both cells are run and
reported: cell 1 gates, cell 2 is kept as the demonstration that it cannot.

RUNNING IT
----------
pylate pins `transformers<=5.3.0` against this repo's 5.12.1, so it cannot go
into `.venv`. Two steps, and only the first needs the throwaway environment::

    # once, in a separate CPU venv (pylate + transformers==4.53.2):
    <other-venv>/python.exe tools/eval/colbert_pylate_crosscheck.py \
        --reference data/results/colbert_pylate_ref_t453.npz
    # and again with transformers==5.3.0 -> ..._ref_t530.npz

    # then, in .venv:
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/colbert_pylate_crosscheck.py

The reference `.npz` files are the only thing the other venv produces, so the
comparison stays re-runnable after it is deleted. `--render` re-derives the
report from `colbert_pylate_crosscheck_raw.json` with no model load at all.

WHAT AN EXACT MATCH BUYS
------------------------
One number. `max|Δ| = 0` on the query side externally validates marker
insertion, augmentation to `query_maxlen`, `attend_to_mask_tokens`, the 1024->128
projection head `AutoModel` does not load, L2 normalisation **and**
`_repair_rotary` at once -- the last of those being the interesting one, since
reproducing a correctly-loaded buffer's output bit for bit is what separates
*restoration* from a self-consistent substitute.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data" / "results"
RAW = RESULTS / "colbert_pylate_crosscheck_raw.json"
REPORT = RESULTS / "colbert_pylate_crosscheck.md"

# One query and two documents, fixed. Small on purpose: the point is an exact
# comparison of conventions, and a disagreement has to be readable token by
# token. Doc 1 carries `.` and `พ.ศ.` because punctuation is where the two
# implementations turned out to disagree.
QUERY = "อาจารย์สมชาย ใจดี สังกัดคณะวิศวกรรมศาสตร์"
DOCS = [
    "ที่ประชุมมีมติอนุมัติให้ อาจารย์สมชาย ใจดี เป็นอาจารย์ผู้รับผิดชอบหลักสูตร",
    "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า ฉบับปรับปรุง พ.ศ. ๒๕๖๔",
]

TOL = 5e-3  # fp32 on two different transformers versions; not a fitted number.

CELLS = [
    {
        "key": "t453",
        "npz": "colbert_pylate_ref_t453.npz",
        "transformers": "4.53.2",  # used only if the cache predates the version stamp
        "repair": True,
        "gates": True,
        "label": "`repaired` vs pylate @ transformers 4.53.2",
        "why": "the rotary buffer loads correctly there, so this is an external "
               "witness that `_repair_rotary` restores the *right* values",
    },
    {
        "key": "t530",
        "npz": "colbert_pylate_ref_t530.npz",
        "transformers": "5.3.0",
        "repair": False,
        "gates": False,
        "label": "`unrepaired` vs pylate @ transformers 5.3.0",
        "why": "intended as 'both position-blind on the same weights, so a match "
               "isolates the encoding conventions'. It cannot work, and that is "
               "the result -- kept as a demonstration, never as a gate",
    },
]


# --------------------------------------------------------------------------
# the reference side -- runs in the OTHER venv, imports pylate, nothing else
# --------------------------------------------------------------------------
def emit_reference(out: Path) -> int:
    import torch
    from pylate import models

    model = models.ColBERT(
        model_name_or_path="jinaai/jina-colbert-v2",
        trust_remote_code=True,
        device="cpu",
    )
    model.eval()

    bad = tot = 0
    first: list[float] | None = None
    for m in model[0].auto_model.modules():
        rot = getattr(m, "rotary_emb", None)
        if rot is None or not hasattr(rot, "_compute_inv_freq"):
            continue
        good = rot._compute_inv_freq(device=rot.inv_freq.device)
        if not torch.equal(rot.inv_freq.float(), good):
            bad += 1
            if first is None:
                first = rot.inv_freq.float().flatten()[:3].tolist()
        tot += 1
    print(f"rotary: {bad} of {tot} layers wrong"
          + (f"; layer 0 holds {first}" if first else ""))

    with torch.no_grad():
        qv = model.encode([QUERY], is_query=True, convert_to_numpy=True, padding=False)
        dv = model.encode(DOCS, is_query=False, convert_to_numpy=True, padding=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        q=np.asarray(qv[0], dtype=np.float32),
        d0=np.asarray(dv[0], dtype=np.float32),
        d1=np.asarray(dv[1], dtype=np.float32),
        rotary_bad=np.array([bad, tot]),
        transformers=np.array(__import__("transformers").__version__),
    )
    print("wrote", out)
    return 0


# --------------------------------------------------------------------------
# our side
# --------------------------------------------------------------------------
def encode_ours(repair: bool):
    from rag_lab.colbert import ColbertConfig, ColbertEncoder

    enc = ColbertEncoder(replace(ColbertConfig(), repair_rotary=repair),
                         device="cpu", dtype="float32")
    q = enc.encode_queries([QUERY])[0]
    vecs, lengths = enc.encode_documents(DOCS)
    off = np.concatenate([[0], np.cumsum(lengths)])
    docs = [vecs[off[i]:off[i + 1]] for i in range(len(DOCS))]
    repaired = enc.rotary_repaired  # None when repair_rotary=False, i.e. nothing was touched
    enc.release()
    return q, docs, int(repaired or 0)


def run_cell(cell: dict) -> dict:
    from rag_lab.colbert import maxsim

    ref = np.load(RESULTS / cell["npz"], allow_pickle=False)
    q, docs, repaired = encode_ours(cell["repair"])
    bad, tot = (int(x) for x in ref["rotary_bad"])

    out = {
        "key": cell["key"],
        "label": cell["label"],
        "why": cell["why"],
        "gates": cell["gates"],
        "pylate_transformers": (str(ref["transformers"]) if "transformers" in ref.files
                                else cell["transformers"]),
        "pylate_rotary_bad": bad,
        "pylate_rotary_total": tot,
        "ours_layers_repaired": int(repaired),
        "tensors": [],
    }
    agree = True
    for name, mine, theirs in [("query", q, ref["q"]),
                               ("doc0", docs[0], ref["d0"]),
                               ("doc1", docs[1], ref["d1"])]:
        row = {"name": name, "ours": list(mine.shape), "pylate": list(theirs.shape)}
        if mine.shape != theirs.shape:
            row["shape_mismatch"] = True
            agree = False
        else:
            row["max_abs_delta"] = float(np.abs(mine - theirs).max())
            row["min_cosine"] = float((mine * theirs).sum(axis=1).min())
            agree &= row["max_abs_delta"] < TOL
        out["tensors"].append(row)

    lengths = np.array([len(d) for d in docs], dtype=np.int64)
    out["maxsim_ours"] = [round(float(x), 4) for x in maxsim(q, np.vstack(docs), lengths)]
    out["maxsim_pylate"] = [
        round(float(x), 4)
        for x in maxsim(ref["q"], np.vstack([ref["d0"], ref["d1"]]),
                        np.array([len(ref["d0"]), len(ref["d1"])], dtype=np.int64))
    ]
    out["agree"] = bool(agree)
    return out


# --------------------------------------------------------------------------
# self-checks -- both directions, per the qualification script's own rule
# --------------------------------------------------------------------------
def self_checks(cells: list[dict]) -> list[tuple[str, bool, str]]:
    by = {c["key"]: c for c in cells}
    gate, demo = by["t453"], by["t530"]
    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "S1 the reference is correct by construction",
        gate["pylate_rotary_bad"] == 0,
        f"pylate @ {gate['pylate_transformers']}: "
        f"{gate['pylate_rotary_bad']} of {gate['pylate_rotary_total']} rotary layers wrong "
        f"(must be 0 -- otherwise this is a second broken model, not a control)",
    ))
    checks.append((
        "S2 shapes agree",
        all("shape_mismatch" not in t for t in gate["tensors"]),
        "; ".join(f"{t['name']} {tuple(t['ours'])} vs {tuple(t['pylate'])}"
                  for t in gate["tensors"])
        + " -- a token-count mismatch names the convention that differs, which a "
          "value delta cannot",
    ))
    checks.append((
        "S3 values agree",
        gate["agree"],
        "; ".join(f"{t['name']} max|Δ|={t.get('max_abs_delta', float('nan')):.3e}"
                  for t in gate["tensors"]) + f" (tol {TOL:.0e})",
    ))
    q = next(t for t in gate["tensors"] if t["name"] == "query")
    checks.append((
        "S4 the query side matches exactly",
        q.get("max_abs_delta") == 0.0,
        f"max|Δ| = {q.get('max_abs_delta', float('nan')):.3e} over "
        f"{tuple(q['ours'])} -- exact, so `_repair_rotary` reproduces a "
        f"correctly-loaded buffer rather than substituting a consistent one",
    ))
    checks.append((
        "S5 MaxSim agrees on the score, not only the vectors",
        max(abs(a - b) for a, b in zip(gate["maxsim_ours"], gate["maxsim_pylate"])) < 1e-2,
        f"ours {gate['maxsim_ours']} vs pylate {gate['maxsim_pylate']}",
    ))
    checks.append((
        "S6 two independently-broken models do NOT agree",
        not demo["agree"],
        "; ".join(f"{t['name']} max|Δ|={t.get('max_abs_delta', float('nan')):.3e}"
                  for t in demo["tensors"])
        + " -- this cell is REQUIRED to differ: the uninitialised buffer is drawn "
          "fresh at each load, so 'both position-blind' is not 'the same model'",
    ))
    checks.append((
        "S7 the demonstration cell really is the broken path",
        demo["pylate_rotary_bad"] == demo["pylate_rotary_total"] > 0
        and demo["ours_layers_repaired"] == 0,
        f"pylate @ {demo['pylate_transformers']}: {demo['pylate_rotary_bad']} of "
        f"{demo['pylate_rotary_total']} wrong; ours repaired "
        f"{demo['ours_layers_repaired']} layers (`repair_rotary=False`)",
    ))
    return checks


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def render(raw: dict) -> str:
    cells = raw["cells"]
    checks = self_checks(cells)
    gate = next(c for c in cells if c["gates"])
    L: list[str] = []
    A = L.append

    A("# ColBERT encoder vs pylate")
    A("")
    A("Generated by `tools/eval/colbert_pylate_crosscheck.py` — the external half of "
      "the ColBERT gate. `colbert_model_qualification.md` is 11 checks of the encoder "
      "against *itself*; this is one check against somebody else's.")
    A("")
    A("**Why that distinction is not pedantic: this cross-check found a defect all 11 "
      "gates passed.** `mask_punctuation=True` was masking whitespace and no "
      "punctuation at all — see §3. Every internal check compares the model to a "
      "control built from the same weights, a reversal of the same ids, or a norm of "
      "its own output, so a convention that is wrong on *both* sides of each "
      "comparison cancels.")
    A("")
    A(f"Inputs are one query and two documents, fixed in the script: `{QUERY}` and "
      f"{len(DOCS)} Thai minutes-style sentences. Tolerance {TOL:.0e} (fp32, two "
      "different `transformers` versions).")
    A("")

    A("## 1. the two cells")
    A("")
    A("| cell | pylate `transformers` | pylate rotary | ours | verdict |")
    A("|---|---|---|---|---|")
    for c in cells:
        rot = f"{c['pylate_rotary_bad']} of {c['pylate_rotary_total']} layers wrong"
        ours = (f"repaired {c['ours_layers_repaired']} layers"
                if c["ours_layers_repaired"] else "`repair_rotary=False`")
        verdict = "**MATCH**" if c["agree"] else "DIFFER"
        if not c["gates"]:
            verdict += " (expected — see §2)"
        A(f"| {c['label']} | {c['pylate_transformers']} | {rot} | {ours} | {verdict} |")
    A("")
    A("**Only the first cell gates**, and the reason is the rotary column: pylate under "
      "5.3.0 hits the same uninitialised-`inv_freq` bug this repo does, so anyone "
      "running pylate + `jina-colbert-v2` on 5.x is silently serving a position-blind "
      "model. Pinning 4.53.2 is what makes the reference *correct by construction* "
      "rather than merely independent.")
    A("")

    for c in cells:
        A(f"**{c['label']}** — {c['why']}")
        A("")
        A("| tensor | ours | pylate | max\\|Δ\\| | min per-token cosine |")
        A("|---|---|---|---|---|")
        for t in c["tensors"]:
            if "shape_mismatch" in t:
                A(f"| `{t['name']}` | {tuple(t['ours'])} | {tuple(t['pylate'])} | "
                  f"— | — (shape mismatch) |")
            else:
                A(f"| `{t['name']}` | {tuple(t['ours'])} | {tuple(t['pylate'])} | "
                  f"{t['max_abs_delta']:.3e} | {t['min_cosine']:.6f} |")
        A("")
        A(f"MaxSim ours `{c['maxsim_ours']}` vs pylate `{c['maxsim_pylate']}`.")
        A("")

    A("## 2. two independently-broken models are not a control")
    A("")
    A("The second cell was built on the reasoning that if both sides are position-blind "
      "on identical weights, a match would isolate the *encoding conventions* from the "
      "rotary question. It cannot: the corrupt `inv_freq` is uninitialised memory, drawn "
      "fresh at every load, so the two sides are not running the same broken model. "
      "S6 requires this cell to differ — a gate that passed here would mean the "
      "corruption had become reproducible, which is a different and worse finding.")
    A("")

    A("## 3. what the comparison found")
    A("")
    A("The query side matched **bitwise** on the first attempt. The document side did "
      "not: **19 and 21 vectors against pylate's 21 and 22**, and the cause was ours. "
      "`mask_punctuation` built its skiplist from `encode(sym)[0]`, which on this "
      "checkpoint's SentencePiece tokenizer is the `▁` word-boundary marker and never "
      "the symbol itself — so the two skiplists were **disjoint**, and the flag masked "
      "whitespace while keeping every punctuation mark, the exact inverse of its name. "
      "Original ColBERT uses both forms and they coincide only on WordPiece. Fixed to "
      "`convert_tokens_to_ids(sym)`; `tests/colbert/test_colbert_skiplist.py` pins the "
      "rule against a stub tokenizer carrying the property that makes the two rules "
      "disagree, so it states the rule rather than recording today's vocabulary ids.")
    A("")
    A("**The surprise ran backwards**, which is worth keeping: the elaborate path — "
      "marker insertion, augmentation to 32, `attend_to_mask_tokens`, the hand-loaded "
      "projection head — matched exactly, and the simple one held the defect.")
    A("")

    A("## self-checks")
    A("")
    for name, ok, detail in checks:
        A(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    A("")
    n_fail = sum(1 for _, ok, _ in checks if not ok)
    A(f"{len(checks)} checks: {len(checks) - n_fail} pass, {n_fail} fail.")
    A("")
    if gate["agree"] and n_fail == 0:
        A("**Verdict: the encoder reproduces an independent, correctly-loaded "
          "implementation.** That is a statement about these conventions on these "
          "inputs, not a proof of correctness at scale — the corpus-scale question is "
          "the artifact-alignment check, which is separate.")
    else:
        A("**Verdict: NOT reproduced.** Nothing measured with this encoder is citable.")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path,
                    help="run the pylate side (requires the other venv) and write "
                         "this .npz; nothing else runs")
    ap.add_argument("--render", action="store_true",
                    help="re-render the report from the cached comparison, no model load")
    args = ap.parse_args()

    if args.reference:
        return emit_reference(args.reference)

    if args.render:
        if not RAW.exists():
            print(f"no cache at {RAW}; run without --render", file=sys.stderr)
            return 2
        raw = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        missing = [c["npz"] for c in CELLS if not (RESULTS / c["npz"]).exists()]
        if missing:
            print("missing reference file(s): " + ", ".join(missing)
                  + "\nproduce them with --reference from a venv holding pylate "
                    "(transformers 4.53.2 and 5.3.0 respectively); see the module "
                    "docstring.", file=sys.stderr)
            return 2
        raw = {"cells": [run_cell(c) for c in CELLS]}
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    md = render(raw)
    REPORT.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten to {REPORT}")
    checks = self_checks(raw["cells"])
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
