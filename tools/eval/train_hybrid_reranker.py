"""Fine-tune a cross-encoder on hybrid-fused candidates — follow-up (a), phase 2.

Pre-registration: `docs/reranker-trained-on-hybrid-design.md`. Training data:
`tools/eval/build_reranker_training_data.py` -> `data/results/reranker_train/`.
Nothing here chooses an arm, a metric or a verdict; those are fixed in the design
doc and measured by `tools/eval/reranker_trained_test.py`.

WHAT IS ACTUALLY BEING CHANGED, AND WHAT IS HELD FIXED
------------------------------------------------------
The published null (`reranker_rrf_routed_test.md`: the reranker adds **+0.0017**
on top of the shipped router, against a routed-pool oracle of **+0.1500**) was
attributed to the *model* rather than to the axis, on two independent pieces of
evidence -- the oracle column, and a 4-model swap whose spread (0.0355) is ~20x
the anchor's whole effect. Follow-up (a) asks whether the model is weak *because
it never saw this candidate distribution*. So exactly one thing varies: the
cross-encoder's weights. The pool, the fusion, the routing, `w`, `P` and `k` all
stay at the values the published arms used, and the eval script re-anchors
0.6831 / 0.6847 / 0.8331 / 0.9054 rather than trusting that claim.

THE GPU BUDGET, AND WHY THE WORD EMBEDDINGS ARE FROZEN
------------------------------------------------------
`bge-reranker-v2-m3` is XLM-R-large: ~560M parameters, of which ~256M sit in the
250,002 x 1024 word-embedding matrix. Full AdamW over all of them needs ~9 GB of
optimizer state alone on a 12 GB card. Freezing that one matrix leaves ~304M
trainable (~4.9 GB of params+grads+moments) and costs nothing this experiment
cares about: a fixed multilingual vocabulary is not what "trained on hybrid-fused
candidates" means. `C1` checks the matrix is bit-identical after training rather
than trusting `requires_grad_(False)` -- an asserted invariant is not a check.

WHY THE SCORING PATH IS ANCHORED AGAINST sentence-transformers
--------------------------------------------------------------
Every published reranker arm scored pairs through `CrossEncoderReranker`, i.e.
`sentence_transformers.CrossEncoder.predict`. Training needs raw
`transformers` (ST's trainer is not what this needs), so there are now two
scoring paths, and two paths that are meant to agree are this project's signature
way of being silently wrong. `C2` scores the same pairs both ways on the
untouched model and requires the **same delivered top-K** (ST applies a sigmoid
for `num_labels=1`, which is monotonic, so values differ by a known transform
while order must not).

That check also settled a question this experiment could otherwise have got
silently wrong: **inference runs in fp32 even though training runs under bf16
autocast**, because bf16 is not accurate enough to rank this pool. Measured on
three real dev pools -- fp32 vs ST: max |Δ| 2e-7..1e-6, top-10 identical **3 of
3**; bf16 vs ST: max |Δ| up to 9.8e-3, top-10 **reordered on 2 of 3**. A
checkpoint selected on bf16 rankings would be selected on noise the size of the
effect under test. The top-K bound is measured too: fp32 at batch 16 vs batch 8
already permutes positions past K (~6e-6), so full-pool order equality is a
property of BLAS reduction order, not of the model.

TRAINING LENGTH vs EVAL LENGTH
------------------------------
The anchor accepts 8,192 tokens and `reranker_model_comparison.md` measured only
**1.9%** of routed-pool pairs above 512 (longest 2,755). Training pads to the
longest pair in each group and caps at `MAX_LEN`; evaluation runs at the model's
own native maximum, exactly as every published arm did. `C5` reports the training
truncation rate from the tokenizer alone so the mismatch is a measured number in
the report rather than an assumption, and the dev metric is additionally
recomputed at the eval-time length so a length regression cannot hide.

Run (GPU: nothing else may be resident -- the script refuses a busy card):
    .venv/Scripts/python.exe tools/eval/train_hybrid_reranker.py --smoke
    .venv/Scripts/python.exe tools/eval/train_hybrid_reranker.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

IN_POOLS = REPO / "data" / "results" / "reranker_train" / "train_pools.json"
IN_META = REPO / "data" / "results" / "reranker_train" / "train_meta.json"
OUT_MODEL = REPO / "data" / "models" / "reranker_hybrid_trained"
OUT_LOG = REPO / "data" / "results" / "reranker_train" / "train_log.json"
REPORT = REPO / "data" / "results" / "reranker_training_run.md"

BASE_MODEL = "BAAI/bge-reranker-v2-m3"

# Group-wise softmax cross-entropy: one positive against N negatives drawn from
# the SAME query's pool. Negatives are top-50 routed-hybrid hits by construction,
# so every one of them is a hard negative -- no mining step is needed or wanted.
GROUP_NEG = 7
MAX_POS_PER_QUERY = 4          # so a query with 20 gold chunks cannot dominate
GROUPS_PER_STEP = 2            # gradient accumulation; effective batch 16 seqs
MAX_LEN = 1024                 # see the docstring; C5 reports what it costs
EPOCHS = 3
LR = 1.0e-5
WARMUP_FRAC = 0.1
WEIGHT_DECAY = 0.01
K = 10                         # the eval's budget; the dev metric uses the same
SEED = 42

# Refuse to start on a card someone else is using. The standing rule on this
# machine is one GPU job at a time, and OOM half an hour into a fine-tune is the
# expensive way to discover it.
MIN_FREE_GB = 9.0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_pools() -> tuple[list[dict], list[dict], dict]:
    if not IN_POOLS.exists():
        raise SystemExit(
            f"missing {IN_POOLS.relative_to(REPO)} — run "
            "tools/eval/build_reranker_training_data.py first (without --smoke)"
        )
    pools = json.loads(IN_POOLS.read_text(encoding="utf-8"))
    meta = json.loads(IN_META.read_text(encoding="utf-8"))
    train = [r for r in pools if r["split"] == "train"]
    dev = [r for r in pools if r["split"] == "dev"]
    return train, dev, meta


def build_groups(records: list[dict], epoch: int) -> list[tuple[str, str, list[str]]]:
    """(query, positive text, negative texts) triples for one epoch.

    Re-sampled per epoch (seeded on the epoch, so a re-run reproduces it) rather
    than fixed once: with ~43 negatives per pool and 7 shown per group, a fixed
    sample would leave most of the pool unseen for the whole run."""
    rng = random.Random(f"{SEED}:groups:{epoch}")
    groups = []
    for r in records:
        pos = [c for c in r["candidates"] if c["label"]]
        neg = [c for c in r["candidates"] if not c["label"]]
        if not pos or len(neg) < GROUP_NEG:
            continue
        for p in (pos if len(pos) <= MAX_POS_PER_QUERY
                  else rng.sample(pos, MAX_POS_PER_QUERY)):
            groups.append((r["query"], p["text"],
                           [c["text"] for c in rng.sample(neg, GROUP_NEG)]))
    rng.shuffle(groups)
    return groups


def recall_at_k(order_rids: list[str], relevant: set[str], k: int, n_relevant: int) -> float:
    """The project's recall@k, at the level relevance is judged at (ADR-0002):
    distinct relevant `resolution_id`s among the top-k CHUNKS, over the query's
    FULL qrels. Chunks sharing an id are the same document twice, and the
    denominator is `n_relevant` rather than what the pool holds -- a reranker
    must not be credited for evidence retrieval never delivered to it."""
    if n_relevant <= 0:
        return 0.0
    return len(set(order_rids[:k]) & relevant) / n_relevant


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_pairs(model, tok, pairs, device, max_len: int, batch: int = 16) -> np.ndarray:
    """Raw logits for (query, passage) pairs, in input order. Inference only.

    **fp32, deliberately, even though training runs under bf16 autocast.** That
    was measured, not assumed: against `sentence_transformers` on three real dev
    pools, fp32 agrees to 2e-7..1e-6 with an identical top-10 every time, while
    bf16 disagrees by up to 9.8e-3 and **reorders the delivered top-10 on 2 of
    3 pools**. A checkpoint selected on bf16 rankings would be selected on noise
    of the same size as the effect under test."""
    import torch

    out = np.empty(len(pairs), dtype=np.float64)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(0, len(pairs), batch):
            chunk = pairs[s: s + batch]
            enc = tok([a for a, _ in chunk], [b for _, b in chunk],
                      padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits
            out[s: s + len(chunk)] = logits.float().squeeze(-1).cpu().numpy()
    model.train(was_training)
    return out


def dev_metric(model, tok, dev: list[dict], device, max_len: int) -> tuple[float, dict]:
    """Rerank each dev pool and score the delivered top-K, per route as well as
    overall. This is the ONLY metric a checkpoint is selected on -- selecting on
    the 106 eval queries would turn the whole experiment into an argmax."""
    per_route: dict[str, list[float]] = {}
    vals = []
    for r in dev:
        rel = {c["resolution_id"] for c in r["candidates"] if c["label"]}
        sc = score_pairs(model, tok, [(r["query"], c["text"]) for c in r["candidates"]],
                         device, max_len)
        order = np.argsort(-sc, kind="stable")
        rids = [r["candidates"][i]["resolution_id"] for i in order]
        # Denominator is the query's full qrels, not what the pool happens to
        # hold: a reranker must not be credited for evidence nobody retrieved.
        v = recall_at_k(rids, rel, K, r["n_relevant"])
        vals.append(v)
        per_route.setdefault(r["route"], []).append(v)
    return (float(np.mean(vals)) if vals else 0.0,
            {k: round(float(np.mean(v)), 4) for k, v in sorted(per_route.items())})


def baseline_metric(dev: list[dict]) -> tuple[float, dict]:
    """The pool's own routed-hybrid ordering, no reranking. Everything below has
    to beat this to be worth a GPU."""
    per_route: dict[str, list[float]] = {}
    vals = []
    for r in dev:
        rel = {c["resolution_id"] for c in r["candidates"] if c["label"]}
        rids = [c["resolution_id"] for c in sorted(r["candidates"], key=lambda c: c["rank"])]
        v = recall_at_k(rids, rel, K, r["n_relevant"])
        vals.append(v)
        per_route.setdefault(r["route"], []).append(v)
    return (float(np.mean(vals)) if vals else 0.0,
            {k: round(float(np.mean(v)), 4) for k, v in sorted(per_route.items())})


# --------------------------------------------------------------------------- #
# checks that need the untouched model
# --------------------------------------------------------------------------- #
def check_scoring_path_agrees(model, tok, dev: list[dict], device) -> tuple[bool, str]:
    """C2: this file's raw-transformers scoring must rank a pool as
    `sentence_transformers.CrossEncoder.predict` does, because every published
    arm was scored the other way. ST applies a sigmoid when `num_labels == 1`, so
    the values differ by a monotonic transform and the ORDER is what must match.

    It is the **delivered top-K** order that is checked, not all P, and that
    bound was measured rather than chosen for convenience: scoring the same pool
    in fp32 at batch 16 vs batch 8 already reorders positions past K (values move
    ~6e-6, and a pool of 50 near-ties has to break somewhere). Requiring all 50
    to agree would be a check on BLAS reduction order, which no published number
    depends on; requiring the top-10 to agree is a check on the thing every arm
    actually delivers."""
    from sentence_transformers import CrossEncoder

    st = CrossEncoder(BASE_MODEL, device=device, max_length=None)
    n_same, n_pairs, worst = 0, 0, 0.0
    probe = dev[:3]
    for r in probe:
        pairs = [(r["query"], c["text"]) for c in r["candidates"]]
        mine = score_pairs(model, tok, pairs, device, int(tok.model_max_length))
        theirs = np.asarray(st.predict(pairs, batch_size=8, show_progress_bar=False),
                            dtype=np.float64)
        n_same += int(list(np.argsort(-mine, kind="stable"))[:K]
                      == list(np.argsort(-theirs, kind="stable"))[:K])
        worst = max(worst, float(np.abs(1.0 / (1.0 + np.exp(-mine)) - theirs).max()))
        n_pairs += len(pairs)
    del st
    if device == "cuda":
        import torch as _t
        _t.cuda.empty_cache()
    return (n_same == len(probe),
            f"{n_same} of {len(probe)} pools rank identically ({n_pairs} pairs); "
            f"max |sigmoid(logit) - ST score| = {worst:.2e}")


def truncation_stats(tok, records: list[dict], max_len: int) -> dict:
    """C5: what the training cap costs, measured from the tokenizer (CPU)."""
    n, trunc, longest = 0, 0, 0
    for r in records:
        lens = [len(x) for x in tok([r["query"]] * len(r["candidates"]),
                                    [c["text"] for c in r["candidates"]],
                                    truncation=False)["input_ids"]]
        n += len(lens)
        trunc += sum(x > max_len for x in lens)
        longest = max(longest, max(lens))
    return {"n_pairs": n, "truncated": trunc,
            "truncated_pct": round(100.0 * trunc / max(n, 1), 2),
            "longest_pair_tokens": longest}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="24 groups, 1 epoch, no checkpoint written")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="skip the free-VRAM guard (one GPU job at a time is the rule)")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.time()
    checks: list[tuple[str, bool, str]] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
        print(f"  GPU free {free_gb:.1f} GB", file=sys.stderr)
        if free_gb < MIN_FREE_GB and not args.allow_busy_gpu:
            raise SystemExit(
                f"only {free_gb:.1f} GB free (< {MIN_FREE_GB} GB) — another GPU job is "
                "resident. Wait for it, or pass --allow-busy-gpu."
            )

    train, dev, meta = load_pools()
    if args.smoke:
        train, dev = train[:24], dev[:4]
    print(f"  {len(train)} train / {len(dev)} dev pools  "
          f"routes={dict(Counter(r['route'] for r in train))}", file=sys.stderr)

    checks.append((
        "C4 the dev pools are disjoint from the train pools",
        not ({r["query"] for r in dev} & {r["query"] for r in train}),
        f"dev {len(dev)}, train {len(train)}, "
        f"{len({r['query'] for r in dev} & {r['query'] for r in train})} shared",
    ))

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.to(device)

    emb = model.get_input_embeddings().weight
    emb.requires_grad_(False)
    emb_before = emb.detach().cpu().clone()   # on CPU: 1 GB of VRAM is not free here
    n_all = sum(p.numel() for p in model.parameters())
    n_train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {n_all/1e6:.0f}M params, {n_train_p/1e6:.0f}M trainable "
          f"({emb.numel()/1e6:.0f}M frozen in the word embeddings)", file=sys.stderr)

    trunc = truncation_stats(tok, train, args.max_len)
    checks.append((
        f"C5 the training cap of {args.max_len} tokens is reported, not assumed",
        True,
        f"{trunc['truncated']} of {trunc['n_pairs']:,} training pairs truncated "
        f"({trunc['truncated_pct']}%), longest pair {trunc['longest_pair_tokens']} tokens; "
        f"eval runs at the model's own {int(tok.model_max_length)}",
    ))

    ok, detail = check_scoring_path_agrees(model, tok, dev, device)
    checks.append(("C2 this file's scoring ranks a pool as sentence-transformers does",
                   ok, detail))

    base_v, base_by_route = baseline_metric(dev)
    step0_v, step0_by_route = dev_metric(model, tok, dev, device, int(tok.model_max_length))
    print(f"  dev recall@{K}: hybrid {base_v:.4f} · off-the-shelf {step0_v:.4f}",
          file=sys.stderr)

    # ---- train ----------------------------------------------------------- #
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and not (n.endswith("bias") or "LayerNorm" in n)]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and (n.endswith("bias") or "LayerNorm" in n)]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    n_steps = max(1, sum(len(build_groups(train, e)) for e in range(args.epochs))
                  // GROUPS_PER_STEP)
    warm = max(1, int(n_steps * WARMUP_FRAC))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else max(0.0, (n_steps - s) / max(1, n_steps - warm)))

    history = [{"epoch": 0, "dev_recall": round(step0_v, 4), "by_route": step0_by_route,
                "loss": None, "note": "off-the-shelf"}]
    # Selection is over the TRAINED epochs, with the off-the-shelf score kept
    # beside them as a reference: if training turned out to hurt, that belongs in
    # the eval's report as a result, not silently swapped back to the base model
    # (which would make arm T the already-published anchor under a new name).
    best = (-1.0, -1)
    step, losses = 0, []
    for epoch in range(1, args.epochs + 1):
        groups = build_groups(train, epoch)
        model.train()
        opt.zero_grad(set_to_none=True)
        for gi, (q, pos, negs) in enumerate(groups):
            texts = [pos] + negs
            enc = tok([q] * len(texts), texts, padding=True, truncation=True,
                      max_length=args.max_len, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**enc).logits.squeeze(-1)
                # The positive is at index 0 by construction; softmax over the
                # group is the "pick the gold chunk out of its own pool" task the
                # reranker actually performs at eval time.
                loss = torch.nn.functional.cross_entropy(
                    logits.float().unsqueeze(0),
                    torch.zeros(1, dtype=torch.long, device=logits.device))
            (loss / GROUPS_PER_STEP).backward()
            losses.append(loss.detach().item())
            if (gi + 1) % GROUPS_PER_STEP == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    print(f"    epoch {epoch} step {step}/{n_steps} "
                          f"loss {np.mean(losses[-200:]):.4f}  {time.time()-t0:.0f}s",
                          file=sys.stderr)
        v, by_route = dev_metric(model, tok, dev, device, int(tok.model_max_length))
        history.append({"epoch": epoch, "dev_recall": round(v, 4), "by_route": by_route,
                        "loss": round(float(np.mean(losses)), 4), "note": ""})
        print(f"  epoch {epoch}: loss {np.mean(losses):.4f}  dev recall@{K} {v:.4f} "
              f"(best {best[0]:.4f} @ epoch {best[1]})  {time.time()-t0:.0f}s",
              file=sys.stderr)
        if v > best[0]:
            best = (v, epoch)
            if not args.smoke:
                OUT_MODEL.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(OUT_MODEL)
                tok.save_pretrained(OUT_MODEL)
        losses = []

    emb_after = model.get_input_embeddings().weight.detach().cpu()
    checks.append((
        "C1 the frozen word-embedding matrix is bit-identical after training",
        bool(torch.equal(emb_before, emb_after)),
        f"{emb.numel()/1e6:.0f}M weights, max |Δ| = "
        f"{float((emb_before - emb_after).abs().max()):.3e}",
    ))
    checks.append((
        "C3 a checkpoint is selected on held-out TRAINING queries, never on the eval set",
        best[1] >= 1,
        f"best dev recall@{K} {best[0]:.4f} at epoch {best[1]} of {args.epochs} "
        f"(off-the-shelf {step0_v:.4f}, routed-hybrid baseline {base_v:.4f})"
        + ("" if best[0] > step0_v else
           " — training did NOT beat the off-the-shelf model on dev; that is a "
           "finding for the eval to report, not a reason to substitute the base model"),
    ))

    log = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": BASE_MODEL,
        "data": {"train_pools": len(train), "dev_pools": len(dev),
                 "built_at": meta.get("at"), "fingerprint": meta.get("fingerprint"),
                 "pool_depth": meta.get("pool_depth")},
        "hyper": {"group_neg": GROUP_NEG, "max_pos_per_query": MAX_POS_PER_QUERY,
                  "groups_per_step": GROUPS_PER_STEP, "max_len": args.max_len,
                  "epochs": args.epochs, "lr": args.lr, "warmup_frac": WARMUP_FRAC,
                  "weight_decay": WEIGHT_DECAY, "seed": SEED,
                  "trainable_params": n_train_p, "frozen_embedding_params": int(emb.numel())},
        "truncation": trunc,
        "dev": {"k": K, "baseline_hybrid": round(base_v, 4),
                "baseline_by_route": base_by_route,
                "off_the_shelf": round(step0_v, 4),
                "best": round(best[0], 4), "best_epoch": best[1]},
        "history": history,
        "beats_off_the_shelf_on_dev": bool(best[0] > step0_v),
        "checkpoint": str(OUT_MODEL.relative_to(REPO)).replace("\\", "/")
        if best[1] >= 1 and not args.smoke else None,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    if args.smoke:
        print("  [smoke] no checkpoint, no log, no report written", file=sys.stderr)
    else:
        OUT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT.write_text(render_report(log, checks), encoding="utf-8")

    print()
    for name, ok_, detail in checks:
        print(f"[{'PASS' if ok_ else 'FAIL'}] {name} — {detail}")
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n{(time.time()-t0)/60:.1f} min")
    return 0 if all(ok_ for _, ok_, _ in checks) else 1


def render_report(log: dict, checks) -> str:
    d, h = log["dev"], log["hyper"]
    L = [
        "# เทรน cross-encoder บน hybrid-fused candidates — บันทึกการเทรน",
        "",
        "Generated by `tools/eval/train_hybrid_reranker.py` · "
        "ออกแบบไว้ล่วงหน้าที่ `docs/reranker-trained-on-hybrid-design.md` · "
        "ข้อมูลจาก `tools/eval/build_reranker_training_data.py`",
        "",
        "**ไฟล์นี้ไม่ใช่ผลการทดลอง** ตัวเลข dev ข้างล่างวัดบนคำถามที่กันไว้จาก "
        "*ชุดเทรน* เพื่อเลือก checkpoint เท่านั้น ห้ามนำไปเทียบกับตัวเลขที่ตีพิมพ์ "
        "ผลจริงอยู่ที่ `tools/eval/reranker_trained_test.py` ซึ่งวัดบน 106 คำถามเดิม",
        "",
        f"| โมเดลตั้งต้น | `{log['base_model']}` |",
        "|---|---|",
        f"| pool ที่เทรน | routed hybrid top-{log['data']['pool_depth']} "
        f"({log['data']['train_pools']} คำถาม) |",
        f"| dev (เลือก checkpoint) | {log['data']['dev_pools']} คำถาม |",
        f"| พารามิเตอร์ที่เทรน | {h['trainable_params']/1e6:.0f}M "
        f"(แช่แข็ง word embedding {h['frozen_embedding_params']/1e6:.0f}M) |",
        f"| loss | group-wise softmax CE, 1 positive vs {h['group_neg']} negative "
        "จาก pool ของคำถามเดียวกัน |",
        f"| epoch / lr / max_len | {h['epochs']} / {h['lr']} / {h['max_len']} |",
        f"| เวลา | {log['minutes']} นาที |",
        "",
        f"## dev recall@{d['k']} (ระดับ resolution)",
        "",
        "| | recall | หมายเหตุ |",
        "|---|---|---|",
        f"| routed hybrid ไม่ rerank | {d['baseline_hybrid']:.4f} | ลำดับเดิมของ pool |",
        f"| off-the-shelf | {d['off_the_shelf']:.4f} | โมเดลก่อนเทรน |",
        f"| **best checkpoint** | **{d['best']:.4f}** | epoch {d['best_epoch']} |",
        "",
        "| epoch | loss | dev recall | แยกตาม route |",
        "|---|---|---|---|",
    ]
    for e in log["history"]:
        L.append(f"| {e['epoch']}{' (' + e['note'] + ')' if e['note'] else ''} | "
                 f"{e['loss'] if e['loss'] is not None else '—'} | {e['dev_recall']:.4f} | "
                 + ", ".join(f"{k} {v:.4f}" for k, v in e["by_route"].items()) + " |")
    t = log["truncation"]
    L += [
        "",
        f"ตัดความยาวตอนเทรนที่ {h['max_len']} token — โดน {t['truncated']:,} คู่ "
        f"จาก {t['n_pairs']:,} ({t['truncated_pct']}%) คู่ยาวสุด "
        f"{t['longest_pair_tokens']:,} token · ตอนวัดผลใช้ความยาวเต็มของโมเดลเอง "
        "เหมือนทุก arm ที่ตีพิมพ์ไปแล้ว",
        "",
        "## self-check",
        "",
    ]
    for name, ok, detail in checks:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    L += ["", f"เทรนเมื่อ {log['at']} · checkpoint `{log['checkpoint']}`", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
