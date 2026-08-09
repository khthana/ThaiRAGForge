"""Is a reranker model actually working before we spend GPU hours measuring it?

WHY THIS EXISTS
---------------
On 2026-08-09, setting up a second reranker to test whether this project's
`+0.0017 on top of the router` is a property of `bge-reranker-v2-m3` or of
cross-encoder reranking in general, `Alibaba-NLP/gte-multilingual-reranker-base`
loaded, ran, and ranked a hand-written Thai example correctly -- while being
**completely position-blind**. Reversing the word order of a sentence produced a
**bit-identical** score.

Cause: `transformers` 5.x materialises a model's **non-persistent buffers** from
the meta device, and a buffer that is not in the checkpoint is filled with
uninitialised memory rather than re-run through the module's `__init__`. That
remote modeling code builds `position_ids` (`torch.arange`) and the RoPE
`inv_freq`/`cos_cached`/`sin_cached` tables there, so all of them arrive as
garbage: `cos_cached` was **all zeros**, i.e. rotary position encoding
multiplied everything by 0, and nothing recomputed it because the lazy refresh
only triggers past `max_seq_len_cached`. `position_ids` held
`3633978736640`, which is the only reason anything crashed at all.

**That is the dangerous shape**: a broken model that still returns plausible
numbers would have produced a low score for the new reranker, and the conclusion
"a second, independent cross-encoder also fails to beat the router" would have
been published as a family-level claim on the strength of a bag-of-words model.
A crash is safe; a plausible number is not.

WHAT IT CHECKS
--------------
G1  every non-persistent buffer is sane -- integer buffers hold a usable index,
    float buffers are finite and not identically zero (see `audit_buffers` for
    why the integer rule is *in range*, not *equals arange*)
G2  **position sensitivity** -- reversing the token order of the document must
    change the score. This is the behavioural gate, and it is the one that
    catches a dead RoPE table; G1 only catches it if you already know which
    buffers to look at.
G3  relevance direction -- a document containing the queried entity must outscore
    an unrelated one, on Thai text, twice over
G4  determinism -- the same pair scored twice gives the same number
G5  padding independence -- a pair's score must not move when it is batched with
    a much longer pair, beyond one fp16 ulp

VALIDATED IN BOTH DIRECTIONS, which is the point: `bge-reranker-v2-m3` (the
model every published number in this project came from) must PASS, and the two
models found broken on 2026-08-09 must FAIL -- `gte-multilingual-reranker-base`
on G2, `jina-reranker-v2-base-multilingual` at load. A gate that only ever says
PASS is not evidence of anything. `tests/tools/test_qualify_reranker_model.py`
pins the probes themselves.

Run:
    .venv/Scripts/python.exe tools/eval/qualify_reranker_model.py            # default set
    .venv/Scripts/python.exe tools/eval/qualify_reranker_model.py MODEL ...  # ad hoc
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402,F401  (imported so a child fails fast if the env is broken)

OUT = REPO / "data" / "results" / "reranker_model_qualification.md"
CHILD_TAG = "@@QUALIFY@@"  # how a child hands its verdict back to the parent

# (model, trust_remote_code). Ordered cheap-first; the two known-broken ones are
# kept in the default set on purpose so the gate is exercised in both directions.
# Both break on the same underlying cause -- remote modeling code pinned to a
# `transformers` 4.x internal -- but at different depths, which is exactly why
# G2 has to exist. `jina` dies at import (it wants
# `transformers.models.xlm_roberta.modeling_xlm_roberta.create_position_ids_from_input_ids`,
# deleted in 5.x), so nothing can be measured with it by accident. `gte` gets
# all the way to producing numbers. Note `jina` is rejected before any optional
# dependency of its own is reached, so none is installed for it.
DEFAULT_MODELS = [
    ("BAAI/bge-reranker-v2-m3", False),
    ("BAAI/bge-reranker-base", False),
    ("BAAI/bge-reranker-large", False),
    ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", False),
    ("Alibaba-NLP/gte-multilingual-reranker-base", True),
    ("jinaai/jina-reranker-v2-base-multilingual", True),
]

Q_PERSON = "อาจารย์สมชาย ใจดี"
D_MATCH = "ที่ประชุมเห็นชอบการแต่งตั้ง ผศ.ดร.สมชาย ใจดี เป็นอาจารย์ประจำหลักสูตร"
D_OTHER = "ที่ประชุมรับรองรายงานการประชุมครั้งที่ 3/2566 โดยไม่มีการแก้ไข"
Q_PROG = "หลักสูตรวิศวกรรมไฟฟ้า"
D_PROG = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า ฉบับปรับปรุง พ.ศ. 2565"
# G2: same bag of words, opposite order. A model with working position encoding
# cannot score these identically; a bag-of-words model must.
Q_ORDER = "ใครดำรงตำแหน่งแทนใคร"
D_ORDER = "ที่ประชุมอนุมัติให้ นาย ก ดำรงตำแหน่งแทน นาย ข ตั้งแต่วันที่ 1 ตุลาคม"
D_ORDER_REV = " ".join(D_ORDER.split()[::-1])
# G5: long enough to force padding of the short pair in the same batch.
D_LONG = "ระเบียบวาระที่ ๔.๑ " + ("เรื่องเสนอเพื่อพิจารณา การขออนุมัติปรับปรุงหลักสูตร " * 40)

# G2/G5 tolerances. Both are deliberately loose: fp32 attention reorders a few
# additions when the batch shape changes, so ~1e-6 drift is expected and
# meaningless, while the failures these gates exist to catch are gross (a dead
# position encoding gives *exactly* 0.0; an ignored attention mask moves a score
# by tenths). A tight tolerance here would only manufacture false alarms.
POS_MIN_DELTA = 1e-4
PAD_TOL = 1e-3


def audit_buffers(model) -> tuple[bool, list[str]]:
    """G1. A buffer absent from the state_dict was built in `__init__`, so under
    a meta-device load it is uninitialised unless the loader re-ran that code.

    The test for an integer buffer is **not** "equals `arange`" -- that was the
    first version and it rejected the anchor, because XLM-R's `token_type_ids`
    is legitimately all zeros (one segment type). These buffers are *indices*, so
    what actually distinguishes uninitialised memory is holding a value that
    could not index anything: `gte`'s `position_ids` came back as
    `3633978736640` over 512 slots. Flag out-of-range, not un-`arange`."""
    sd = dict(model.state_dict())
    bad = []
    for name, buf in model.named_buffers():
        if name in sd or buf.numel() == 0:
            continue
        if buf.is_floating_point():
            if not bool(torch.isfinite(buf).all()):
                bad.append(f"{name}: not finite")
            elif not bool(buf.any()):
                bad.append(f"{name}: identically zero")
        else:
            lo, hi = int(buf.min()), int(buf.max())
            if lo < 0 or hi >= max(buf.numel(), 2):
                bad.append(f"{name}: index out of range [0,{buf.numel()}) "
                           f"— min={lo} max={hi}")
    return not bad, bad


def qualify(name: str, trc: bool) -> dict:
    from rag_lab.rerankers.cross_encoder import CrossEncoderReranker

    row: dict = {"model": name, "gates": {}, "note": ""}
    t0 = time.time()
    try:
        rr = CrossEncoderReranker(model_name=name, batch_size=8, trust_remote_code=trc)
        m = rr._load()
    except Exception as e:  # a crash at load is the SAFE failure -- record it as one
        row["load"] = False
        row["note"] = f"{type(e).__name__}: {str(e)[:110]}"
        return row
    row["load"] = True

    ok, bad = audit_buffers(m.model)
    row["gates"]["G1 buffers sane"] = (ok, "; ".join(bad)[:110] if bad else "all clean")

    def p(pairs):
        return [float(x) for x in m.predict(pairs, batch_size=8, show_progress_bar=False)]

    def alone(q, d):
        """Score one pair on its own. Every gate below except G5 uses this, so
        that batch composition -- which changes the padded length, and which
        sentence-transformers may reorder by length -- can never be mistaken for
        the effect a gate is testing."""
        return p([(q, d)])[0]

    s_ord, s_rev = alone(Q_ORDER, D_ORDER), alone(Q_ORDER, D_ORDER_REV)
    row["gates"]["G2 position-sensitive"] = (
        abs(s_ord - s_rev) > POS_MIN_DELTA,
        f"forward {s_ord:.6f} vs reversed {s_rev:.6f} (|d|={abs(s_ord-s_rev):.2e}, "
        f"need >{POS_MIN_DELTA:.0e})")

    s_m, s_o = alone(Q_PERSON, D_MATCH), alone(Q_PERSON, D_OTHER)
    s_p, s_pn = alone(Q_PROG, D_PROG), alone(Q_PROG, D_OTHER)
    row["gates"]["G3 relevance direction"] = (
        s_m > s_o and s_p > s_pn, f"person {s_m:.4f}>{s_o:.4f}, program {s_p:.4f}>{s_pn:.4f}")

    again = alone(Q_PERSON, D_MATCH)
    row["gates"]["G4 deterministic"] = (again == s_m, f"|d|={abs(again-s_m):.2e}")

    batched = p([(Q_PERSON, D_MATCH), (Q_PERSON, D_LONG)])[0]
    row["gates"]["G5 padding-independent"] = (
        abs(batched - s_m) <= PAD_TOL, f"alone {s_m:.6f} vs batched-with-long {batched:.6f} "
                                       f"(|d|={abs(batched-s_m):.2e}, tol {PAD_TOL:.0e})")

    row["secs"] = time.time() - t0
    rr.release()
    return row


def run_child(name: str, trc: bool) -> dict:
    """One model per process, and not for tidiness: `gte`'s garbage buffers
    raise a **CUDA device-side assert**, which poisons the whole context -- every
    model loaded afterwards in the same process reports bogus failures. A
    single-process loop would therefore reject healthy models as a side effect of
    the order they happen to sit in. It also guarantees only one model is
    resident at a time on a 12GB card."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", name]
    if trc:
        cmd.append("--trust-remote-code")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sys.stderr.write(proc.stderr)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(CHILD_TAG):
            return json.loads(line[len(CHILD_TAG):])
    return {"model": name, "load": False, "gates": {},
            "note": f"child process died (exit {proc.returncode}): "
                    f"{proc.stderr.strip().splitlines()[-1][:110] if proc.stderr.strip() else '?'}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="qualify a reranker model before measuring with it")
    ap.add_argument("models", nargs="*", help="model ids (default: the standard set)")
    ap.add_argument("--trust-remote-code", action="store_true", help="for ad-hoc models")
    ap.add_argument("--child", metavar="MODEL", help=argparse.SUPPRESS)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    if args.child:
        print(CHILD_TAG + json.dumps(qualify(args.child, args.trust_remote_code)))
        return 0

    todo = ([(m, args.trust_remote_code) for m in args.models] if args.models
            else DEFAULT_MODELS)
    rows = []
    for name, trc in todo:
        print(f"=== {name}", file=sys.stderr)
        rows.append(run_child(name, trc))

    gates = ["G1 buffers sane", "G2 position-sensitive", "G3 relevance direction",
             "G4 deterministic", "G5 padding-independent"]
    lines = [
        "# Reranker model qualification",
        "",
        "Generated by `tools/eval/qualify_reranker_model.py` · a model must pass every gate "
        "before any number measured with it is citable.",
        "",
        "G2 is the load-bearing one, and `gte-multilingual-reranker-base` is why. Under "
        "`transformers` 5.x its non-persistent buffers are materialised from the meta device "
        "as uninitialised memory: `position_ids` holds a pointer-sized integer (which is what "
        "makes it crash, and a crash is the *safe* failure) and the RoPE tables `cos_cached`/"
        "`sin_cached` are **all zeros**. Repair `position_ids` by hand and it stops crashing, "
        "ranks a hand-written Thai relevance example correctly — and scores a sentence and its "
        "reversal **bit-identically**, because rotary position encoding is multiplying by zero. "
        "A plausible number from a bag-of-words model is the failure this gate exists to catch: "
        "it would have been published as *a second cross-encoder also fails to beat the router*.",
        "",
        "| model | loads | " + " | ".join(g.split(" ", 1)[0] for g in gates) + " | verdict |",
        "|---|---|" + "---|" * (len(gates) + 1),
    ]
    verdicts = {}
    for r in rows:
        if not r["load"]:
            verdicts[r["model"]] = False
            lines.append(f"| `{r['model']}` | ✗ | " + " | ".join(["–"] * len(gates))
                         + " | **REJECTED (load)** |")
            continue
        cells = ["✓" if r["gates"][g][0] else "✗" for g in gates]
        ok = all(v[0] for v in r["gates"].values())
        verdicts[r["model"]] = ok
        lines.append(f"| `{r['model']}` | ✓ | " + " | ".join(cells)
                     + f" | {'**QUALIFIED**' if ok else '**REJECTED**'} |")
    lines += ["", "## evidence", ""]
    for r in rows:
        lines.append(f"**`{r['model']}`**" + (f" — {r['note']}" if r["note"] else ""))
        for g in gates:
            if g in r["gates"]:
                okg, ev = r["gates"][g]
                lines.append(f"- [{'PASS' if okg else 'FAIL'}] {g} — {ev}")
        lines.append("")

    # The gate is only evidence if it is exercised in both directions.
    anchor = verdicts.get("BAAI/bge-reranker-v2-m3")
    broken = [verdicts.get("Alibaba-NLP/gte-multilingual-reranker-base"),
              verdicts.get("jinaai/jina-reranker-v2-base-multilingual")]
    both_ways = None
    if anchor is not None and any(b is not None for b in broken):
        both_ways = anchor is True and not any(b for b in broken if b is not None)
        lines += [
            "## the gate is exercised in both directions",
            "",
            f"- [{'PASS' if both_ways else 'FAIL'}] the published anchor qualifies and both "
            f"models found broken on 2026-08-09 are rejected — anchor "
            f"{'QUALIFIED' if anchor else 'REJECTED'}, "
            f"gte {'QUALIFIED' if verdicts.get('Alibaba-NLP/gte-multilingual-reranker-base') else 'REJECTED'}, "
            f"jina {'QUALIFIED' if verdicts.get('jinaai/jina-reranker-v2-base-multilingual') else 'REJECTED'}",
            "",
        ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {OUT}")
    # Exit 1 if the anchor stopped qualifying (every published reranker number
    # came from it) or if the both-directions check stopped holding -- a gate
    # that no longer rejects a model known to be broken has gone vacuous, which
    # is this project's recurring way of losing a check without noticing.
    return 0 if anchor and both_ways is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
