"""RQ4 step 3: score generated answers on 4a (citation grounding) and 4b
(abstention correctness), and significance-test the arm/prompt comparisons
docs/rq4-design.md calls for -- the deliverable that section left open.

Two independent comparisons, each its own Holm family (they test different
hypotheses, so pooling them would just dilute both):

1. **Arm ordering** (docs/rq4-design.md 4c): with the prompt held fixed at the
   original ("sentence_cap") wording, does citation precision/recall order the
   same way recall@10 did across the five arms? Pairwise paired bootstrap over
   all arm pairs that have citations at all (closed_book excluded -- it never
   cites anything by construction).
2. **Prompt ablation** ("Correction (same day)" in rq4-design.md): citation
   recall was flat at ~0.41 across every arm, and looked like a fixed citation
   budget rather than a comprehension limit (mean 2.65 citations/answer
   regardless of how much gold was available, recall falling as more gold
   appeared). Rule 4 (`ตอบสั้น ๆ ไม่เกิน 3 ประโยค`) is the suspect. This
   compares the original prompt against a "cite every relevant document"
   variant, paired per query, for whichever arms have both
   (hybrid_qwen3_0.6b_semantic, bm25_semantic). Recall rising => prompt
   artifact; recall staying ~0.41 => a real generator ceiling (then the
   deferred gemma4:e4b check is the right next step, not before).

Citation parsing: every `[n]` in the answer text is a citation, deduped per
query (rq4-design.md's scoring caveat -- models sometimes cite a label twice,
once inline and once in the `อ้างอิง:` line). `n` outside 1..len(blocks) is a
phantom citation (a fabrication this setup can detect exactly, because the
supplied context is known).

Precision/recall are macro-averaged (mean of per-query ratios), consistent
with how recall@k is computed everywhere else in this project
(rag_lab.metrics). Precision is undefined (excluded, not zero) for a query
where the model cited nothing at all -- a 0/0, not a miss.

Abstention detection: the ABSTAIN token appearing anywhere in the answer text.
The prompt asks for it as the entire `คำตอบ:` line, but models don't always
follow the two-line format exactly (see pilot answers), so a substring check
is the robust choice; it can't false-positive here because ABSTAIN is a
multi-character Thai phrase that wouldn't otherwise appear in a substantive
answer about council resolutions.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_score.py
    ... --model phi4  # only this model has been generated so far
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from embedder_matrix_9way import bootstrap_pvalue, holm_correct  # noqa: E402

_ANSWERS = REPO / "data" / "rq4" / "answers"
_OUTPUT = REPO / "data" / "results" / "rq4_score.md"
_ABSTAIN = "ไม่พบข้อมูล"
_CITE_RE = re.compile(r"\[(\d+)\]")
N_BOOT = 10_000
SEED = 42

# The five design-doc arms, in the order recall@10 established (docs/rq4-design.md
# 4c / paper-results-summary.md). closed_book never cites anything -- kept in the
# descriptive table, excluded from the precision/recall significance family.
ARM_ORDER = [
    "hybrid_qwen3_0.6b_semantic",
    "dense_qwen3_0.6b_semantic",
    "bm25_semantic",
    "hybrid_m2v_semantic",
    "closed_book",
]


def parse_citations(answer: str, label_map: dict[str, str]) -> tuple[set[str], set[str]]:
    """-> (cited resolution_ids that resolve, phantom labels that don't)."""
    labels = set(_CITE_RE.findall(answer))
    valid = {lbl for lbl in labels if lbl in label_map}
    phantom = labels - valid
    return {label_map[lbl] for lbl in valid}, phantom


def is_abstained(answer: str) -> bool:
    return _ABSTAIN in answer


def load_arm(model_dir: str, arm: str) -> dict[str, dict]:
    """-> {file stem ("q000".."q105"): record}, so different variants of the
    same arm can be paired by stem (both are built from the same gold-set
    order, one file per query index)."""
    d = _ANSWERS / model_dir / arm
    if not d.is_dir():
        return {}
    out = {}
    for path in sorted(d.glob("q*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("error"):
            print(f"  [warn] {path}: generation error {rec['error']!r}, skipped")
            continue
        out[path.stem] = rec
    return out


class ArmScore:
    """Per-query 4a/4b arrays for one (model_dir, arm), keyed by file stem so
    two variants of the same arm can be diffed pairwise later."""

    def __init__(self, records: dict[str, dict]):
        self.stems = sorted(records)
        self.precision = {}   # stem -> float, only where citations exist
        self.recall = {}      # stem -> float, always defined
        self.n_cited = self.n_phantom = self.n_gold_present = 0
        self.cell_counts = defaultdict(int)  # (has_gold, abstained) -> count

        for stem in self.stems:
            r = records[stem]
            gold = set(r["relevant_resolution_ids"])
            cited_rids, phantom = parse_citations(r["answer"], r["label_map"])
            self.n_cited += len(cited_rids) + len(phantom)
            self.n_phantom += len(phantom)
            if cited_rids:
                self.precision[stem] = len(cited_rids & gold) / len(cited_rids)
            self.recall[stem] = len(cited_rids & gold) / len(gold) if gold else float("nan")
            abstained = is_abstained(r["answer"])
            self.cell_counts[(r["context_has_gold"], abstained)] += 1
            self.n_gold_present += int(r["context_has_gold"])

    @property
    def n(self) -> int:
        return len(self.stems)

    def mean_precision(self) -> tuple[float, int]:
        vals = list(self.precision.values())
        return (float(np.mean(vals)) if vals else float("nan"), len(vals))

    def mean_recall(self) -> float:
        vals = [v for v in self.recall.values() if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def phantom_rate(self) -> tuple[int, int]:
        return self.n_phantom, self.n_cited

    def cell(self, has_gold: bool, abstained: bool) -> int:
        return self.cell_counts.get((has_gold, abstained), 0)


def paired_arrays(a: ArmScore, b: ArmScore, metric: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Aligned (a_vals, b_vals) over stems where the metric is defined in both."""
    common = sorted(set(a.stems) & set(b.stems))
    src_a = a.precision if metric == "precision" else a.recall
    src_b = b.precision if metric == "precision" else b.recall
    stems = [s for s in common if s in src_a and s in src_b
             and not (isinstance(src_a[s], float) and np.isnan(src_a[s]))
             and not (isinstance(src_b[s], float) and np.isnan(src_b[s]))]
    return np.array([src_a[s] for s in stems]), np.array([src_b[s] for s in stems]), len(stems)


def run_family(pairs_data: list[tuple[str, str, np.ndarray, np.ndarray]], rng, n_boot: int, alpha: float):
    """pairs_data: (label_a, label_b, vals_a, vals_b) -> Holm-corrected rows."""
    pairs = []
    for label_a, label_b, va, vb in pairs_data:
        diffs = vb - va
        observed, p, ci = bootstrap_pvalue(diffs, rng, n_boot)
        pairs.append((label_a, label_b, observed, p, ci))
    return holm_correct(pairs, alpha=alpha) if pairs else []


def fmt_rows(rows, mean_lookup) -> list[str]:
    lines = []
    for a, b, diff, p, ci, holm_p, sig in sorted(rows, key=lambda x: x[5]):
        mark = "**yes**" if sig else "no"
        lines.append(
            f"| {a} vs {b} | {mean_lookup(a):.4f} | {mean_lookup(b):.4f} | {diff:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} | {holm_p:.4f} | {mark} |"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="phi4")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--treatment-variant", default="cite_all",
                    help="which prompt variant is the treatment arm of the ablation "
                    "(families 1b and 2), as the suffix on the answers dir. Default "
                    "`cite_all` reproduces the published run. Pass "
                    "`cite_all_guarded` to score the zero-document-guarded rewrite "
                    "against the same baseline. The descriptive/abstention table "
                    "above is unaffected -- it enumerates whatever is on disk.")
    ap.add_argument("--out", default=str(_OUTPUT),
                    help="report path. Defaults to the published rq4_score.md; a "
                    "non-default --treatment-variant should pass a different path "
                    "rather than clobber it, the same way rq4_generate.py gives each "
                    "prompt variant its own answers dir.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lines = ["# RQ4 scoring: citation grounding (4a) and abstention correctness (4b)", ""]

    # ---- descriptive table, all arms x variants present on disk ----
    variants = sorted(p.name for p in _ANSWERS.iterdir()
                       if p.is_dir() and (p.name == args.model or p.name.startswith(args.model + "_")))
    print(f"model={args.model}  variants found: {variants}")

    scores: dict[tuple[str, str], ArmScore] = {}  # (variant, arm) -> ArmScore
    for variant in variants:
        for arm in ARM_ORDER:
            recs = load_arm(variant, arm)
            if recs:
                scores[(variant, arm)] = ArmScore(recs)

    lines += [
        "## Descriptive: citation precision/recall and phantom rate, per (prompt variant, arm)",
        "",
        "Precision is macro-averaged over queries with >=1 citation only (see module "
        "docstring for the 0/0 handling); recall is macro-averaged over all queries. "
        "Phantom = citation to a label outside the supplied context, reported as "
        "count/total-citations (micro).",
        "",
        "| variant | arm | n | mean precision | (n with citations) | mean recall | phantom / total citations |",
        "|---|---|---|---|---|---|---|",
    ]
    for (variant, arm), s in sorted(scores.items(), key=lambda kv: (kv[0][0], ARM_ORDER.index(kv[0][1]))):
        prec, n_prec = s.mean_precision()
        ph, tot = s.phantom_rate()
        prec_str = f"{prec:.4f}" if not np.isnan(prec) else "—"
        lines.append(f"| {variant} | {arm} | {s.n} | {prec_str} | {n_prec} | {s.mean_recall():.4f} | {ph}/{tot} |")
    lines.append("")

    lines += [
        "## Descriptive: 4b abstention 2x2, per (prompt variant, arm)",
        "",
        "Rows: whether the context contained >=1 gold document. Cols: whether the model "
        "abstained. Bottom-right (no gold, no abstain) = hallucination; top-right "
        "(gold present, abstained) = missed.",
        "",
        "| variant | arm | answered w/ gold (expected) | missed (gold present, abstained) | "
        "hallucinated (no gold, answered) | correctly abstained (no gold) |",
        "|---|---|---|---|---|---|",
    ]
    for (variant, arm), s in sorted(scores.items(), key=lambda kv: (kv[0][0], ARM_ORDER.index(kv[0][1]))):
        lines.append(
            f"| {variant} | {arm} | {s.cell(True, False)} | {s.cell(True, True)} | "
            f"{s.cell(False, False)} | {s.cell(False, True)} |"
        )
    lines.append("")

    # ---- family 1: arm ordering, one sub-family per prompt variant that has
    # >=2 arms with citations. Originally sentence_cap-only; now that the
    # cite_all extension covers all 5 arms, run it there too so the arm
    # ordering (does citation grounding track recall@10?) can be checked to
    # still hold under the prompt that actually raised recall.
    base_variant = "phi4" if args.model == "phi4" else args.model
    cite_all_variant = f"{base_variant}_{args.treatment_variant}"

    def arm_ordering_family(variant: str, label: str) -> list[str]:
        out_lines = []
        ordering_arms = [a for a in ARM_ORDER if a != "closed_book" and (variant, a) in scores]
        if len(ordering_arms) < 2:
            return [f"## Significance family 1{label}: skipped (fewer than 2 arms "
                    f"with citations found for variant {variant})\n"]
        pairs_data = []
        for arm_a, arm_b in combinations(ordering_arms, 2):
            sa, sb = scores[(variant, arm_a)], scores[(variant, arm_b)]
            for metric in ("precision", "recall"):
                va, vb, n = paired_arrays(sa, sb, metric)
                if n > 0:
                    pairs_data.append((f"{arm_a}[{metric}]", f"{arm_b}[{metric}]", va, vb))
        rows = run_family(pairs_data, rng, args.n_boot, args.alpha)
        mean_of = {}
        for arm in ordering_arms:
            s = scores[(variant, arm)]
            mean_of[f"{arm}[precision]"] = s.mean_precision()[0]
            mean_of[f"{arm}[recall]"] = s.mean_recall()
        out_lines += [
            f"## Significance family 1{label}: arm ordering under `{variant}` (does citation "
            "grounding order the way recall@10 did?)",
            "",
            f"Paired bootstrap over queries common to both arms (n_boot={args.n_boot}, "
            f"seed={args.seed}), Holm-corrected across all {len(rows)} tests in this family.",
            "",
            "| comparison | mean(a) | mean(b) | diff(b-a) | 95% CI | raw p | Holm-adj p | significant |",
            "|---|---|---|---|---|---|---|---|",
        ]
        out_lines += fmt_rows(rows, lambda k: mean_of[k])
        out_lines.append("")
        return out_lines

    lines += arm_ordering_family(base_variant, "a")
    lines += arm_ordering_family(cite_all_variant, "b")

    # ---- family 2: prompt ablation, cite_all vs sentence_cap, per arm ----
    ablation_arms = [a for a in ARM_ORDER
                      if (base_variant, a) in scores and (cite_all_variant, a) in scores]
    if ablation_arms:
        tv = args.treatment_variant
        pairs_data = []
        for arm in ablation_arms:
            sa, sb = scores[(base_variant, arm)], scores[(cite_all_variant, arm)]
            for metric in ("recall", "precision"):
                va, vb, n = paired_arrays(sa, sb, metric)
                if n > 0:
                    pairs_data.append((f"{arm}[{metric}]:sentence_cap",
                                       f"{arm}[{metric}]:{tv}", va, vb))
        rows = run_family(pairs_data, rng, args.n_boot, args.alpha)
        mean_of = {}
        for arm in ablation_arms:
            sa, sb = scores[(base_variant, arm)], scores[(cite_all_variant, arm)]
            mean_of[f"{arm}[recall]:sentence_cap"] = sa.mean_recall()
            mean_of[f"{arm}[recall]:{tv}"] = sb.mean_recall()
            mean_of[f"{arm}[precision]:sentence_cap"] = sa.mean_precision()[0]
            mean_of[f"{arm}[precision]:{tv}"] = sb.mean_precision()[0]
        lines += [
            "## Significance family 2: prompt ablation (rule 4 'answer in <=3 sentences' "
            f"vs 'cite every relevant document') -- treatment variant `{tv}`",
            "",
            "This is the deliverable that answers whether the flat ~0.41 citation recall "
            "found under the original prompt was a fixed citation budget (recall rises here) "
            "or a real generator ceiling (recall stays ~0.41, and then gemma4:e4b is the "
            "right next test -- not before this result).",
            "",
            f"Paired bootstrap over queries common to both variants (n_boot={args.n_boot}, "
            f"seed={args.seed}), Holm-corrected across all {len(rows)} tests in this family.",
            "",
            f"| comparison | mean(sentence_cap) | mean({tv}) | diff | 95% CI | raw p | Holm-adj p | significant |",
            "|---|---|---|---|---|---|---|---|",
        ]
        lines += fmt_rows(rows, lambda k: mean_of[k])
        lines.append("")
    else:
        lines.append(
            f"## Significance family 2: skipped (no arm has both {base_variant} and "
            f"{cite_all_variant} answers on disk yet)\n"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
