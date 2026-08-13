"""Is `jina-colbert-v2` actually working, and is our encoder actually ColBERT?

WHY THIS EXISTS
---------------
Two failures, of the same shape, both already paid for in this project.

**The model.** On 2026-08-09 `Alibaba-NLP/gte-multilingual-reranker-base` loaded,
ran, and ranked a hand-written Thai example correctly while being completely
position-blind: `transformers` 5.x materialises non-persistent buffers from the
meta device as uninitialised memory, and its RoPE tables came back all zeros. A
crash is the safe failure; a plausible number is not. `jina-colbert-v2` runs
through the same remote-code path (`jinaai/xlm-roberta-flash-implementation`,
written for transformers 4.43, run here under 5.12), so it gets the same gate.

**And it has the same bug.** That was not a guess and it was not caught by
reasoning: the first run of this script failed G2 on the real encoder, and the
probe that followed found `inv_freq` holding uninitialised memory in every layer
(30 of 32 entries exactly 0, 2 denormal), making `cos = 1`, `sin = 0` — the
rotation is the identity. `ColbertEncoder` repairs it at load
(`_repair_rotary`), and the repair is checked here by **C7**, which compares
every layer against the checkpoint's own `_compute_inv_freq`. Two earlier,
weaker checks are kept precisely because each one *missed* this, and **which one
missed depends on the load**: the garbage is uninitialised memory, so it differs
every time the model is built. Three loads, three different layer-0 values and
three different outcomes -- `[2.6e-29, 1.0e-42, 0]` (G1 **passed**, finite and
not identically zero; G2 caught it), `[-5.2e+02, 2.1e-42, 0]` (the mirror image:
G2 **passed** at |d|=4.09e-01, looking position-sensitive while being just as
wrong; G1 caught it), `[1.3e-01, 1.4e-42, 0]` (both caught it, G2 by |d|=4.79e-02
against a 5e-02 threshold, i.e. within 5% of passing). C7 caught it all three
times, because it is the only check whose rule does not depend on what the
garbage happened to be. **Read the `unrepaired` row below as one sample of one
load** -- re-running this script will not reproduce its G1/G2 cells, only its C7
one, and that is the finding rather than flakiness.

**The encoder.** ColBERT has five places to be quietly wrong -- marker
insertion, query augmentation, attending to the expansion tokens, the 1024->128
projection `AutoModel` silently does not load, and L2 normalisation -- and each
of them leaves a model that still returns a believable ranking. So the gates
below cover our code, not only the checkpoint.

BOTH DIRECTIONS, which is the whole point of a gate
---------------------------------------------------
A PASS-only gate is not evidence. Here the negative direction is not a second
checkpoint but two **deliberately sabotaged encoders** built from the same
weights, each reproducing one of the failures above:

  `bag_of_words`  -- word embeddings only, no transformer stack, i.e. exactly
                     what a dead position encoding degrades to. Must FAIL G2.
  `unnormalised`  -- skips L2 normalisation. Must FAIL C3.
  `unrepaired`    -- `repair_rotary=False`, i.e. the checkpoint exactly as
                     transformers 5.x hands it over. Must FAIL C7. This one is
                     not synthetic: it is the live bug, so it also demonstrates
                     that C7 is capable of failing rather than passing vacuously.
                     If a future transformers restores the buffer correctly this
                     control will pass, which is not a gate failure -- it is
                     classified and reported as "the upstream bug is gone".

They run in the same process as the real encoder on purpose. The one-model-per-
subprocess rule in `qualify_reranker_model.py` exists because `gte`'s garbage
buffers raise a CUDA device-side assert that poisons every later load; there is
no such hazard here, and three variants of one checkpoint sharing a process is
what keeps only one model resident on a 12 GB card.

Run:
    .venv/Scripts/python.exe tools/eval/qualify_colbert_model.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from rag_lab.colbert import ColbertConfig, ColbertEncoder, maxsim, maxsim_reference  # noqa: E402

OUT = REPO / "data" / "results" / "colbert_model_qualification.md"

Q_PERSON = "อาจารย์สมชาย ใจดี"
D_MATCH = "ที่ประชุมเห็นชอบการแต่งตั้ง ผศ.ดร.สมชาย ใจดี เป็นอาจารย์ประจำหลักสูตร"
D_OTHER = "ที่ประชุมรับรองรายงานการประชุมครั้งที่ 3/2566 โดยไม่มีการแก้ไข"
Q_PROG = "หลักสูตรวิศวกรรมไฟฟ้า"
D_PROG = "หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้า ฉบับปรับปรุง พ.ศ. 2565"
# G2: the same document, token order reversed. The reversal is done on the *ids*
# rather than on the words, which is what makes the negative control decisive
# instead of merely suggestive: a word-level reversal retokenizes to a slightly
# different multiset, so a position-blind model would score it *nearly* alike and
# the gate would rest on a threshold. Reversing the ids holds the multiset exactly,
# and MaxSim is permutation-invariant over document tokens -- so a position-blind
# model must score the two **identically**, to the bit, while a working one cannot.
Q_ORDER = "ใครดำรงตำแหน่งแทนใคร"
D_ORDER = "ที่ประชุมอนุมัติให้ นาย ก ดำรงตำแหน่งแทน นาย ข ตั้งแต่วันที่ 1 ตุลาคม"
D_LONG = "ระเบียบวาระที่ ๔.๑ " + ("เรื่องเสนอเพื่อพิจารณา การขออนุมัติปรับปรุงหลักสูตร " * 40)

# Scores are a sum of `query_maxlen` cosines, so they live in [0, 32] and a real
# effect moves them by tenths. Both tolerances are loose for the same reason as
# the reranker gate: fp16 attention reorders a few additions when the batch shape
# changes, so ~1e-3 drift is expected and meaningless, while every failure these
# gates exist to catch is gross.
POS_MIN_DELTA = 5e-2
PAD_TOL = 5e-2
NORM_TOL = 1e-3


class BagOfWordsEncoder(ColbertEncoder):
    """NEGATIVE CONTROL. Word embeddings, projection, normalise -- no attention,
    no position information at all. This is what `gte-multilingual-reranker-base`
    effectively became when its rotary tables arrived as zeros."""

    def _forward(self, input_ids, attention_mask):  # noqa: D102
        emb = self._model.embeddings.word_embeddings
        with torch.no_grad():
            v = emb(input_ids) @ self._linear.T
            return torch.nn.functional.normalize(v.float(), p=2, dim=-1)


class UnnormalisedEncoder(ColbertEncoder):
    """NEGATIVE CONTROL. Everything real except the L2 normalisation, which is
    the step `AutoModel` cannot do for us and whose absence turns cosine MaxSim
    into a length-weighted dot product."""

    def _forward(self, input_ids, attention_mask):  # noqa: D102
        with torch.no_grad():
            h = self._model(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state
            return (h @ self._linear.T).float()


def score(enc: ColbertEncoder, q: str, docs: list[str]) -> np.ndarray:
    qv = enc.encode_queries([q])[0]
    vecs, lengths = enc.encode_documents(docs)
    return maxsim(qv, vecs, lengths)


def alone(enc: ColbertEncoder, q: str, d: str) -> float:
    """One document per batch, so batch composition -- which changes the padded
    length -- can never be mistaken for the effect a gate is testing."""
    return float(score(enc, q, [d])[0])


def score_ids(enc: ColbertEncoder, qv: np.ndarray, ids) -> float:
    """Score a document given its token ids directly, so G2 can permute the ids
    without going back through the tokenizer. Uses `maxsim_reference`, the
    written-out definition, rather than the packed path -- the packed path is
    what C5 checks, and a gate should not depend on the optimisation it is
    meant to be independent of."""
    am = torch.ones_like(ids)
    v = enc._forward(ids.to(enc._device), am.to(enc._device)).cpu().numpy()[0]
    return float(maxsim_reference(qv, [v.astype(np.float32)])[0])


def audit_buffers(model) -> tuple[bool, str]:
    """G1. A buffer absent from the state_dict was built in `__init__`, so a
    meta-device load leaves it uninitialised unless the loader re-ran that code.
    The integer rule is *in range*, not *equals arange*: XLM-R's `token_type_ids`
    is legitimately all zeros, while `gte`'s `position_ids` held 3633978736640.

    The float rule gained a third clause after this gate passed on a model that
    was fully position-blind: uninitialised memory is not reliably *zero*. What
    was actually there was 30 zeros and two values of 2.6e-29 and 1.0e-42, which
    is finite, non-zero, and arithmetically indistinguishable from zero. So a
    float buffer whose largest magnitude is below 1e-20 is flagged as
    *effectively* zero. This is still only a smell test -- C7 is the check that
    decides -- but the version without it was worse than useless, because it
    reported "all sane" about the exact buffer that was wrong.
    """
    sd = dict(model.state_dict())
    bad, n = [], 0
    for name, buf in model.named_buffers():
        if name in sd or buf.numel() == 0:
            continue
        n += 1
        if buf.is_floating_point():
            peak = float(buf.abs().max())
            if not bool(torch.isfinite(buf).all()):
                bad.append(f"{name}: not finite")
            elif not bool(buf.any()):
                bad.append(f"{name}: identically zero")
            elif peak < 1e-20:
                bad.append(f"{name}: effectively zero (max |x| = {peak:.2e})")
        else:
            lo, hi = int(buf.min()), int(buf.max())
            if lo < 0 or hi >= max(buf.numel(), 2):
                bad.append(f"{name}: index out of range [0,{buf.numel()}) min={lo} max={hi}")
    return not bad, "; ".join(bad)[:140] if bad else f"{n} non-persistent buffers, all sane"


def audit_rotary(model) -> tuple[bool, str]:
    """C7. `inv_freq` is a deterministic function of `(dim, base)` written in
    this checkpoint's own code, so there is a *right answer* to compare against
    and no reason to settle for a heuristic. Everything weaker missed the live
    bug: G1 called the corrupt buffer sane, and a behavioural check would accept
    large garbage as position-sensitive.

    Reported per layer rather than as a boolean because the corruption is
    **nondeterministic across loads** -- the same checkpoint came up with zeros
    in one process and denormals in the next -- so the count is a fact worth
    keeping, not a formality.
    """
    n, bad, ev = 0, [], ""
    for module in model.modules():
        re_ = getattr(module, "rotary_emb", None)
        if re_ is None or not hasattr(re_, "_compute_inv_freq"):
            continue
        good = re_._compute_inv_freq(device=re_.inv_freq.device)
        if not torch.equal(re_.inv_freq.float(), good):
            if not bad:
                got = re_.inv_freq.float().flatten()[:3].tolist()
                want = good.flatten()[:3].tolist()
                ev = (f"layer {n} holds {[f'{x:.3e}' for x in got]} where "
                      f"{[f'{x:.3e}' for x in want]} is required")
            bad.append(n)
        n += 1
    if n == 0:
        return False, "no rotary_emb found — this checkpoint is supposed to be rotary"
    if bad:
        return False, f"{len(bad)} of {n} layers wrong; {ev}"
    return True, f"all {n} layers' inv_freq == _compute_inv_freq exactly"


def gates(enc: ColbertEncoder) -> dict[str, tuple[bool, str]]:
    g: dict[str, tuple[bool, str]] = {}
    enc._load()
    cfg = enc.config

    ok, ev = audit_buffers(enc._model)
    g["G1 buffers sane"] = (ok, ev)

    ok, ev = audit_rotary(enc._model)
    if enc.rotary_repaired is not None:
        ev += f"; `_repair_rotary` rebuilt {enc.rotary_repaired} of them at load"
    g["C7 rotary inv_freq exact"] = (ok, ev)

    qv_ord = enc.encode_queries([Q_ORDER])[0]
    ids_ord = enc._tok([f"{cfg.document_prefix} {D_ORDER}"],
                       return_tensors="pt")["input_ids"]
    ids_rev = ids_ord.clone()
    ids_rev[0, 2:-1] = ids_ord[0, 2:-1].flip(0)   # keep <s>, the marker and </s> in place
    s_ord, s_rev = score_ids(enc, qv_ord, ids_ord), score_ids(enc, qv_ord, ids_rev)
    g["G2 position-sensitive"] = (
        abs(s_ord - s_rev) > POS_MIN_DELTA,
        f"forward {s_ord:.4f} vs id-reversed {s_rev:.4f} "
        f"(|d|={abs(s_ord-s_rev):.2e}, need >{POS_MIN_DELTA:.0e}; identical token "
        f"multiset, so a position-blind model scores exactly 0 apart)")

    s_m, s_o = alone(enc, Q_PERSON, D_MATCH), alone(enc, Q_PERSON, D_OTHER)
    s_p, s_pn = alone(enc, Q_PROG, D_PROG), alone(enc, Q_PROG, D_OTHER)
    g["G3 relevance direction"] = (
        s_m > s_o and s_p > s_pn,
        f"person {s_m:.4f}>{s_o:.4f}, program {s_p:.4f}>{s_pn:.4f}")

    again = alone(enc, Q_PERSON, D_MATCH)
    g["G4 deterministic"] = (again == s_m, f"|d|={abs(again-s_m):.2e}")

    batched = float(score(enc, Q_PERSON, [D_MATCH, D_LONG])[0])
    g["G5 padding-independent"] = (
        abs(batched - s_m) <= PAD_TOL,
        f"alone {s_m:.4f} vs batched-with-long {batched:.4f} "
        f"(|d|={abs(batched-s_m):.2e}, tol {PAD_TOL:.0e})")

    # ---- ColBERT-specific -------------------------------------------------
    tok = enc._tok
    ids = tok([f"{cfg.query_prefix} {Q_PERSON}"], return_tensors="pt")["input_ids"][0].tolist()
    marker_id = tok.convert_tokens_to_ids(cfg.query_prefix)
    # the route original ColBERT uses, which assumes ". " is one token
    other = tok([". " + Q_PERSON], return_tensors="pt")["input_ids"][0].tolist()
    other[1] = marker_id
    g["C1 marker at position 1"] = (
        ids[1] == marker_id and ids[0] == tok.cls_token_id,
        f"ids[:3]={ids[:3]}, marker={marker_id}; the ids[:,1]-overwrite route "
        f"{'agrees' if other[: len(ids)] == ids else 'DISAGREES'} "
        f"(it leaves {tok.convert_ids_to_tokens([other[2]])[0]!r} behind on SentencePiece)")

    qv = enc.encode_queries([Q_PERSON])
    dv, dl = enc.encode_documents([D_MATCH])
    norms = np.concatenate([np.linalg.norm(qv[0], axis=1),
                            np.linalg.norm(dv.astype(np.float32), axis=1)])
    g["C3 unit-norm vectors"] = (
        bool(np.abs(norms - 1.0).max() <= NORM_TOL),
        f"max |‖v‖-1| = {np.abs(norms - 1.0).max():.2e} over {len(norms)} vectors "
        f"(tol {NORM_TOL:.0e}; unprojected norms are ~15, so this is not vacuous)")

    g["C4 query augmentation"] = (
        qv.shape[1] == cfg.query_maxlen,
        f"query is {qv.shape[1]} vectors for a {len(tok(Q_PERSON)['input_ids'])}-token "
        f"question (want {cfg.query_maxlen}: pads become <mask> and are kept)")

    # C5: the packed reduceat path against the definition, one document at a time
    vecs, lengths = enc.encode_documents([D_MATCH, D_OTHER, D_PROG, D_LONG])
    off = np.concatenate([[0], np.cumsum(lengths)])
    per = [vecs[off[i]:off[i + 1]] for i in range(len(lengths))]
    a = maxsim(qv[0], vecs, lengths)
    b = maxsim_reference(qv[0], per)
    g["C5 packed == definition"] = (
        bool(np.abs(a - b).max() < 1e-4),
        f"max |Δ| = {np.abs(a - b).max():.2e} over {len(a)} documents")

    n_long = int(lengths[-1])
    g["C6 doc_maxlen truncates"] = (
        n_long <= cfg.doc_maxlen,
        f"a {len(tok(D_LONG)['input_ids']):,}-token document yields {n_long} vectors "
        f"(cap {cfg.doc_maxlen}; punctuation masking removes a few more)")
    return g


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    cfg = ColbertConfig()
    variants = [
        ("real", ColbertEncoder(cfg)),
        ("bag_of_words", BagOfWordsEncoder(cfg)),
        ("unnormalised", UnnormalisedEncoder(cfg)),
        ("unrepaired", ColbertEncoder(replace(cfg, repair_rotary=False))),
    ]
    names = ["G1 buffers sane", "C7 rotary inv_freq exact", "G2 position-sensitive",
             "G3 relevance direction", "G4 deterministic", "G5 padding-independent",
             "C1 marker at position 1", "C3 unit-norm vectors", "C4 query augmentation",
             "C5 packed == definition", "C6 doc_maxlen truncates"]

    results: dict[str, dict[str, tuple[bool, str]]] = {}
    for label, enc in variants:
        print(f"=== {label}", file=sys.stderr)
        t0 = time.time()
        try:
            results[label] = gates(enc)
        except Exception as e:  # noqa: BLE001 - a crash is a recordable outcome
            results[label] = {"LOAD": (False, f"{type(e).__name__}: {str(e)[:140]}")}
        print(f"    {time.time()-t0:.1f}s", file=sys.stderr)
        enc.release()

    real = results["real"]
    real_ok = all(v[0] for v in real.values())
    # The negative direction: each sabotage must be caught by the gate written
    # for it, and by that gate specifically.
    bow_caught = not results["bag_of_words"].get("G2 position-sensitive", (True, ""))[0]
    unn_caught = not results["unnormalised"].get("C3 unit-norm vectors", (True, ""))[0]
    # `unrepaired` is the live bug rather than a synthetic sabotage, so it has a
    # second legitimate outcome: if it passes C7, transformers has started
    # restoring the buffer and the repair is a no-op. That is classified, not
    # failed -- the [[feedback_cleanup_can_break_an_audit]] rule, one layer down.
    unrep = results["unrepaired"].get("C7 rotary inv_freq exact", (True, ""))
    unrep_caught = not unrep[0]
    unrep_note = ("caught — the upstream bug is live and C7 can fail" if unrep_caught
                  else "**passed** — transformers now restores `inv_freq` itself; "
                       "`_repair_rotary` is a no-op and C7's failing branch is "
                       "unexercised here (`bag_of_words` still exercises G2's)")
    both_ways = real_ok and bow_caught and unn_caught

    L = [
        "# ColBERT model + encoder qualification",
        "",
        "Generated by `tools/eval/qualify_colbert_model.py` — nothing measured with "
        "`jina-colbert-v2` is citable until every gate here passes.",
        "",
        "Two failures are being guarded against at once. The **checkpoint** runs through "
        "`jinaai/xlm-roberta-flash-implementation`, remote code written for transformers "
        "4.43 and executed here under 5.x — the same path on which "
        "`gte-multilingual-reranker-base` came back with all-zero rotary tables, ranked a "
        "Thai example correctly, and scored a sentence and its reversal bit-identically. "
        "The **encoder** has five places to be wrong while still ranking plausibly "
        "(marker insertion, query augmentation, attending to the expansion tokens, the "
        "1024→128 projection `AutoModel` does not load, L2 normalisation).",
        "",
        "The negative direction is three encoders built from the same weights: "
        "`bag_of_words` (word embeddings only — what a dead position encoding degrades "
        "to) must fail **G2**, `unnormalised` (no L2 step) must fail **C3**, and "
        "`unrepaired` (`repair_rotary=False`, the checkpoint exactly as transformers "
        "hands it over) must fail **C7**. G2 is unusually sharp for late interaction: "
        "MaxSim is permutation-invariant over document tokens, so a position-blind model "
        "scores a document and its reversal *identically*, not merely similarly.",
        "",
        "**`unrepaired` is not a hypothetical.** The first run of this script rejected "
        "the real encoder on G2, and the cause was the `gte` bug again: every layer's "
        "`inv_freq` came up as uninitialised memory, so `cos = 1`, `sin = 0` and the "
        "rotation was the identity. `ColbertEncoder._repair_rotary` rebuilds the buffer "
        "from the checkpoint's own `_compute_inv_freq` at load. **Which check catches "
        "it is a coin toss, because the garbage is uninitialised memory and differs "
        "every load.** Three loads gave three layer-0 values and three outcomes: "
        "`[2.6e-29, 1.0e-42, 0]` — G1 *passed* (finite, not identically zero), G2 "
        "caught it; `[-5.2e+02, 2.1e-42, 0]` — the mirror image, G2 *passed* at "
        "|Δ|=4.09e-01 looking position-sensitive while being just as wrong, G1 caught "
        "it; `[1.3e-01, 1.4e-42, 0]` — both caught it, G2 by |Δ|=4.79e-02 against a "
        "5e-02 threshold, within 5% of passing. C7 caught it all three times. So read "
        "the `unrepaired` row above as **one sample of one load**: re-running this "
        "script will not reproduce its G1/G2 cells, only its C7 one. That is the "
        "finding, not flakiness — only the exact check is a property of the bug rather "
        "than of whatever was in memory.",
        "",
        "| variant | " + " | ".join(n.split(" ", 1)[0] for n in names) + " | verdict |",
        "|---|" + "---|" * (len(names) + 1),
    ]
    for label, _ in variants:
        r = results[label]
        if "LOAD" in r:
            L.append(f"| `{label}` | " + " | ".join(["–"] * len(names)) + " | **CRASHED** |")
            continue
        cells = ["✓" if r[n][0] else "✗" for n in names]
        ok = all(v[0] for v in r.values())
        if label == "real":
            verdict = "**QUALIFIED**" if ok else "**REJECTED**"
        else:
            verdict = "control — must fail its own gate"
        L.append(f"| `{label}` | " + " | ".join(cells) + f" | {verdict} |")

    L += ["", "## evidence", ""]
    for label, _ in variants:
        L.append(f"**`{label}`**")
        for n, (okg, ev) in results[label].items():
            L.append(f"- [{'PASS' if okg else 'FAIL'}] {n} — {ev}")
        L.append("")

    L += [
        "## the gate is exercised in both directions",
        "",
        f"- [{'PASS' if real_ok else 'FAIL'}] the real encoder passes every gate",
        f"- [{'PASS' if bow_caught else 'FAIL'}] `bag_of_words` is caught by G2 "
        "(a position-blind encoder must not qualify)",
        f"- [{'PASS' if unn_caught else 'FAIL'}] `unnormalised` is caught by C3 "
        "(an unnormalised encoder must not qualify)",
        f"- [{'PASS' if unrep_caught else 'note'}] `unrepaired` vs C7 — {unrep_note}",
        "",
        f"Configuration under test: `doc_maxlen={cfg.doc_maxlen}`, "
        f"`query_maxlen={cfg.query_maxlen}`, `dim={cfg.dim}`, "
        f"`mask_punctuation={cfg.mask_punctuation}`, "
        f"`attend_to_mask_tokens={cfg.attend_to_mask_tokens}` — the checkpoint's own "
        "`artifact.metadata` defaults. What those caps cost this corpus is measured "
        "separately in `colbert_length_profile.md`.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {OUT}")
    return 0 if both_ways else 1


if __name__ == "__main__":
    raise SystemExit(main())
