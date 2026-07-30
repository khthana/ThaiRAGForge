"""Statistical power / minimum-detectable-effect for the 106-query Gold set.

Every significance test in this project reports whether a difference was
detected. None reports what size of difference it *could* have detected. That
gap matters more here than in most retrieval papers, because this project's
central claims are largely **null results** -- "the top-4 embedders are fully
tied", "`semantic` never significantly beats any chunker", "normalization and
word-aware segmentation do nothing". A reviewer's first question about a null
result is whether the effect is absent or merely invisible at n=106, and
"we found no difference" is a much weaker sentence than "we can rule out
differences larger than X".

This script answers the second form. For each comparison the study actually
makes, it reports:

  * sd of the per-query paired differences -- the quantity that drives power;
  * **MDE**, the smallest true difference detectable at 80%/90% power, both at
    a nominal alpha and at the Holm worst case (alpha/m), since every headline
    test in this project is Holm-corrected inside a family;
  * **n required** to detect the *observed* difference, for ties whose observed
    effect is nonzero but unresolved;
  * a verdict per non-significant pair: `ruled out` when the observed |diff| is
    below the MDE (the tie is informative -- any real effect is smaller than
    MDE), vs `underpowered` when it exceeds the MDE (the tie says little, and
    should not be cited as evidence of equivalence).

That last distinction is the point of the whole script. A non-significant
result is only evidence of absence in the first case, and this project has
been citing ties in both.

**MDE is closed-form, then verified by simulation against the real test.** The
formula (z_{1-a/2} + z_{power}) * sd / sqrt(n) assumes the test statistic is
normal; the tests here are percentile paired bootstraps. Rather than assume the
approximation holds, `--verify` re-runs the actual bootstrap test on samples
resampled from the observed (mean-centered) difference vector shifted by the
computed MDE, and reports achieved power. Agreement near 0.80 validates the
closed form; disagreement would mean the closed-form numbers are wrong and the
simulated ones should be cited instead.

Pure recompute from already-persisted retrieval results -- no retrieval, no GPU.
Like every other persisted-results consumer, it must be re-run after an index
rebuild (CLAUDE.md, "refresh every retrieval path").

Run with:
    .venv/Scripts/python.exe tools/eval/power_analysis.py
    .venv/Scripts/python.exe tools/eval/power_analysis.py --verify
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_lab.metrics import ndcg_at_k, reciprocal_rank, recall_at_k  # noqa: E402
from rag_lab.query_sets import load_gold_query_set  # noqa: E402
from rag_lab.results import load_retrieval_result  # noqa: E402
from embedder_matrix_9way import (  # noqa: E402
    _INDEX_DIR,
    EMBEDDER_ORDER,
    bootstrap_pvalue,
    build_combo_to_chunker_embedder,
    holm_correct,
)

_GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
_DENSE_DIR = REPO / "data" / "results" / "gold_73det_full_embedder_matrix"
_HYBRID_DIR = REPO / "data" / "results" / "gold_hybrid_73det"
_BM25_DIR = REPO / "data" / "results" / "gold_bm25_73det"
_OUTPUT = REPO / "data" / "results" / "power_analysis.md"

CHUNKERS = ["fixed_size", "recursive", "semantic", "sentence"]
METRICS = ("recall", "mrr", "ndcg")

# Normal quantiles, hardcoded so the script needs no scipy (the framework does
# not depend on it and one constant is not worth a dependency).
_Z = {0.80: 0.8416212, 0.90: 1.2815516}


def _z_two_sided(alpha: float) -> float:
    """z_{1-alpha/2} by bisection on the normal CDF via erf."""
    from math import erf, sqrt

    target = 1 - alpha / 2
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_per_query(query_idx: dict[str, int], qrels: dict[str, list[str]], k: int):
    """(mode, chunker, embedder) -> {metric: per-query array}.

    BM25 ignores the embedder axis entirely (a chunker's lexical index is
    identical across its 9 embedder variants), so its results are keyed by
    chunker with embedder=None and duplicate runs average to themselves.
    """
    n_q = len(query_idx)
    combo_ce = build_combo_to_chunker_embedder(_INDEX_DIR)
    combo_ce = {c[: -len("__dense")]: v for c, v in combo_ce.items()}

    sums: dict[tuple, dict[str, np.ndarray]] = defaultdict(
        lambda: {m: np.zeros(n_q) for m in METRICS}
    )
    counts: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_q))

    for directory, suffix, mode in (
        (_DENSE_DIR, "__dense", "dense"),
        (_HYBRID_DIR, "__hybrid", "hybrid"),
        (_BM25_DIR, "__bm25", "bm25"),
    ):
        n_used = 0
        for path in directory.glob("*.json"):
            r = load_retrieval_result(path)
            if not r.combination_id.endswith(suffix):
                continue
            base = r.combination_id[: -len(suffix)]
            if base not in combo_ce:
                continue
            chunker, embedder = combo_ce[base]
            if chunker not in CHUNKERS:
                continue
            qi = query_idx.get(r.query)
            if qi is None:
                continue
            key = (mode, chunker, None if mode == "bm25" else embedder)
            relevant = qrels[r.query]
            sums[key]["recall"][qi] += recall_at_k(r, relevant, k)
            sums[key]["mrr"][qi] += reciprocal_rank(r, relevant)
            sums[key]["ndcg"][qi] += ndcg_at_k(r, relevant, k)
            counts[key][qi] += 1
            n_used += 1
        print(f"  {mode:7s}: {n_used:,} result files used")

    out = {}
    for key, s in sums.items():
        c = counts[key]
        if (c == 0).any():
            print(f"  WARNING: {key} missing {(c == 0).sum()}/{n_q} queries -- skipped")
            continue
        out[key] = {m: s[m] / c for m in METRICS}
    return out


def average_over(pq: dict, keys: list[tuple], metric: str) -> np.ndarray:
    return np.mean(np.stack([pq[k][metric] for k in keys]), axis=0)


def build_families(pq: dict, metric: str) -> dict[str, list[tuple[str, str, np.ndarray]]]:
    """The comparison families the study actually reports, as difference vectors.

    Each mirrors an existing significance test's aggregation convention (average
    across the *other* axis per query, then difference), so the MDE computed
    here applies to that test's published numbers rather than to some other
    aggregation that happens to be easier.
    """
    embedders = [e for e in EMBEDDER_ORDER if ("dense", "semantic", e) in pq]
    families: dict[str, list[tuple[str, str, np.ndarray]]] = {}

    def cross_chunker(mode: str, embedder: str | None) -> np.ndarray:
        keys = [(mode, c, embedder) for c in CHUNKERS if (mode, c, embedder) in pq]
        return average_over(pq, keys, metric)

    # 1. embedder vs embedder, dense-alone, averaged across chunkers
    dense = {e: cross_chunker("dense", e) for e in embedders}
    families["embedder pairs (dense, cross-chunker)"] = [
        (a, b, dense[a] - dense[b]) for a, b in itertools.combinations(embedders, 2)
    ]

    # 2. hybrid vs dense-alone, per embedder
    hybrid = {e: cross_chunker("hybrid", e) for e in embedders}
    families["hybrid vs dense (per embedder)"] = [
        (f"hybrid_{e}", f"dense_{e}", hybrid[e] - dense[e]) for e in embedders
    ]

    # 3. hybrid vs BM25-alone, per embedder
    bm25 = cross_chunker("bm25", None)
    families["hybrid vs BM25 (per embedder)"] = [
        (f"hybrid_{e}", "bm25", hybrid[e] - bm25) for e in embedders
    ]

    # 4. chunker vs chunker, hybrid, averaged across embedders
    by_chunker = {
        c: average_over(pq, [("hybrid", c, e) for e in embedders if ("hybrid", c, e) in pq], metric)
        for c in CHUNKERS
    }
    families["chunker pairs (hybrid, cross-embedder)"] = [
        (a, b, by_chunker[a] - by_chunker[b]) for a, b in itertools.combinations(CHUNKERS, 2)
    ]
    return families


def simulate_power(d0: np.ndarray, delta: float, alpha: float, n_boot: int,
                   trials: int, rng: np.random.Generator, batch: int = 25) -> float:
    """Achieved power of the *actual* percentile bootstrap test at true effect `delta`.

    Resamples from the mean-centered observed differences, so the simulated
    population keeps the empirical dispersion and shape (both far from normal
    here -- recall@10 differences are discrete and heavily zero-inflated)
    instead of assuming a Gaussian the closed form would take for granted.
    """
    n = len(d0)
    rejects = 0
    for start in range(0, trials, batch):
        b = min(batch, trials - start)
        samp = d0[rng.integers(0, n, size=(b, n))] + delta
        idx = rng.integers(0, n, size=(b, n_boot, n))
        boot = np.take_along_axis(np.broadcast_to(samp[:, None, :], idx.shape), idx, axis=2).mean(axis=2)
        p = np.minimum(2 * np.minimum((boot <= 0).mean(axis=1), (boot >= 0).mean(axis=1)), 1.0)
        rejects += int((p < alpha).sum())
    return rejects / trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--verify", action="store_true",
                    help="simulate achieved power at the closed-form MDE (slow, ~1 min/family)")
    ap.add_argument("--verify-trials", type=int, default=400)
    ap.add_argument("--verify-boot", type=int, default=2_000)
    args = ap.parse_args()

    query_set = load_gold_query_set(_GOLD)
    qrels = {e.query: e.relevant_resolution_ids for e in query_set}
    query_idx = {q: i for i, q in enumerate(qrels)}
    n_q = len(query_idx)
    print(f"{n_q} queries, {sum(len(v) for v in qrels.values()):,} relevance judgments")

    pq = load_per_query(query_idx, qrels, args.k)
    metric_labels = {"recall": f"recall@{args.k}", "mrr": "mrr", "ndcg": f"ndcg@{args.k}"}

    lines = [
        "# Statistical power and minimum detectable effect (Gold 73-det, n=106)",
        "",
        f"Paired bootstrap, n_boot={args.n_boot}, seed={args.seed}, alpha={args.alpha}, "
        f"Holm-corrected within each family. Recomputed from persisted retrieval results.",
        "",
        "**MDE** = smallest true difference detectable at the stated power, "
        "`(z_(1-a/2) + z_power) * sd / sqrt(n)`. `MDE(Holm)` uses `alpha/m` for the "
        "family's size m, the worst case a Holm-corrected test faces.",
        "",
        "**Verdict on a non-significant pair** -- the reason this report exists:",
        "",
        "- `ruled out` -- observed |diff| < MDE(80%). The tie is *informative*: any true "
        "effect is smaller than the MDE, which can be cited as a bound.",
        "- `underpowered` -- observed |diff| >= MDE(80%). The tie is *uninformative*: an "
        "effect of the observed size would have been missed more often than not, so this "
        "pair must not be cited as evidence of equivalence.",
        "",
        "**`CI bound`** is the wider end of the 95% bootstrap CI in absolute value -- the "
        "largest difference the data is consistent with. For a tie this is the number "
        "worth citing, because it bounds the effect directly and needs no power argument; "
        "MDE describes what the *design* could have detected, the CI bound what the *data* "
        "actually excludes.",
        "",
    ]

    verify_rows: list[tuple] = []
    tie_bounds: list[tuple] = []

    for metric in METRICS:
        families = build_families(pq, metric)
        lines.append(f"## {metric_labels[metric]}")
        lines.append("")
        for fam_name, pairs in families.items():
            m = len(pairs)
            alpha_holm = args.alpha / m
            z_a = _z_two_sided(args.alpha)
            z_a_holm = _z_two_sided(alpha_holm)

            rng = np.random.default_rng(args.seed)
            tested = []
            for a, b, diffs in pairs:
                observed, p, ci = bootstrap_pvalue(diffs, rng, args.n_boot)
                tested.append((a, b, observed, p, ci))
            corrected = holm_correct(tested, alpha=args.alpha)

            lines.append(f"### {fam_name} (m={m} pairs)")
            lines.append("")
            lines.append(
                f"| A | B | observed | sd(diff) | Holm-adj p | sig | MDE 80% | MDE 90% | "
                f"MDE(Holm) 80% | CI bound | n needed | verdict |"
            )
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

            for (a, b, diffs), (_, _, observed, p, ci, holm_p, sig) in zip(pairs, corrected):
                sd = float(diffs.std(ddof=1))
                mde80 = (z_a + _Z[0.80]) * sd / np.sqrt(n_q)
                mde90 = (z_a + _Z[0.90]) * sd / np.sqrt(n_q)
                mde80_holm = (z_a_holm + _Z[0.80]) * sd / np.sqrt(n_q)
                if abs(observed) > 1e-9:
                    n_needed = int(np.ceil(((z_a + _Z[0.80]) * sd / abs(observed)) ** 2))
                    n_needed_s = f"{n_needed:,}"
                else:
                    n_needed_s = "inf"
                if sig:
                    verdict = "significant"
                elif abs(observed) < mde80_holm:
                    verdict = "**ruled out**"
                else:
                    verdict = "underpowered"
                # The citable bound. MDE says what the *design* could detect; this
                # says what the data actually rules out, and is the stronger claim
                # for a tie -- "the 95% CI excludes differences larger than X" needs
                # no power argument at all.
                ci_bound = max(abs(ci[0]), abs(ci[1]))
                lines.append(
                    f"| {a} | {b} | {observed:+.4f} | {sd:.4f} | {holm_p:.4f} | "
                    f"{'yes' if sig else 'no'} | {mde80:.4f} | {mde90:.4f} | {mde80_holm:.4f} | "
                    f"{ci_bound:.4f} | {n_needed_s} | {verdict} |"
                )
                if not sig:
                    tie_bounds.append((metric, fam_name, a, b, observed, ci_bound))
                if args.verify and metric == "recall" and len(verify_rows) < 6:
                    verify_rows.append((fam_name, a, b, diffs - diffs.mean(), mde80_holm, alpha_holm))
            lines.append("")

            sds = np.array([d.std(ddof=1) for _, _, d in pairs])
            lines.append(
                f"Family median sd {np.median(sds):.4f} -> median MDE(Holm, 80%) "
                f"**{(z_a_holm + _Z[0.80]) * float(np.median(sds)) / np.sqrt(n_q):.4f}** "
                f"{metric_labels[metric]}."
            )
            lines.append("")

    # ---- what every tie in the study is actually worth, in one table ----
    lines += [
        "## Every non-significant pair, as a bound",
        "",
        "The claim to make about a tie is not \"no difference\" but \"no difference larger "
        "than this\". Sorted by bound, tightest first -- a small bound is a strong "
        "equivalence claim, a large one is a weak one that should be reported as "
        "inconclusive rather than as a tie.",
        "",
        "| metric | family | A | B | observed | rules out differences > |",
        "|---|---|---|---|---|---|",
    ]
    for metric, fam, a, b, observed, bound in sorted(tie_bounds, key=lambda r: r[5]):
        lines.append(
            f"| {metric_labels[metric]} | {fam.split(' (')[0]} | {a} | {b} | "
            f"{observed:+.4f} | **{bound:.4f}** |"
        )
    lines.append("")
    if tie_bounds:
        worst = max(tie_bounds, key=lambda r: r[5])
        lines += [
            f"Weakest tie in the study: {worst[2]} vs {worst[3]} on "
            f"{metric_labels[worst[0]]}, consistent with a difference as large as "
            f"**{worst[5]:.4f}**. Any equivalence claim about that pair is the "
            "least defensible one in the paper.",
            "",
        ]

    if args.verify:
        print(f"\nverifying {len(verify_rows)} closed-form MDEs by simulation "
              f"({args.verify_trials} trials x {args.verify_boot} bootstraps each)...")
        lines += [
            "## Simulation check of the closed form",
            "",
            "The MDE formula assumes a normal test statistic; the tests here are "
            "percentile paired bootstraps on discrete, zero-inflated per-query "
            "differences. Each row resamples from the observed mean-centered difference "
            "vector shifted by its own MDE(Holm, 80%) and re-runs the real bootstrap "
            "test, so achieved power near 0.80 means the closed form is safe to cite.",
            "",
            f"({args.verify_trials} trials x {args.verify_boot} bootstraps per row; "
            f"Monte-Carlo se ~{np.sqrt(0.8 * 0.2 / args.verify_trials):.3f})",
            "",
            "| family | A | B | MDE(Holm, 80%) | achieved power |",
            "|---|---|---|---|---|",
        ]
        rng = np.random.default_rng(args.seed + 1)
        for fam, a, b, d0, delta, alpha_holm in verify_rows:
            power = simulate_power(d0, delta, alpha_holm, args.verify_boot, args.verify_trials, rng)
            print(f"  {a} vs {b}: delta={delta:.4f} -> power {power:.3f}")
            lines.append(f"| {fam} | {a} | {b} | {delta:.4f} | {power:.3f} |")
        lines.append("")

    _OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten to {_OUTPUT}")


if __name__ == "__main__":
    main()
