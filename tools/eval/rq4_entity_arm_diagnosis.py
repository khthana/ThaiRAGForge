"""Why `entity_lookup` collapsed at the generation stage (2026-08-10).

`rq4_score.py --arms ...,entity_lookup_semantic,entity_boost_semantic` answers
*whether* the entity arms beat shipped hybrid end-to-end (they don't: see
`data/results/rq4_score_entity.md`). It cannot say *why*, and the why decides
what the null licenses.

The pre-registration in `docs/rq4-design.md` treats `entity_lookup` as a
structural upper bound: exhaustive retrieval over the same dictionaries a
relation graph would be built on, scored against qrels partly derived from
those dictionaries. If that arm can't lift citation quality, the argument
goes, nothing built on the dictionaries can. **That inference is only valid if
the arm failed for lack of evidence.** If instead the evidence reached the
context and the generator declined to use it, the failure is about *ranking
into a fixed context budget*, and it bounds unranked dictionary retrieval
rather than the dictionaries themselves.

So this script measures where the evidence actually was, from the artifacts on
disk (contexts are recorded inside each answer JSON as `label_map`, so no
retrieval and no GPU):

* **gold-in-context density** -- of the blocks actually supplied to the model,
  what fraction were gold. `context_has_gold` (which `rq4_score.py` already
  reports) is a *presence* bit and is the wrong instrument here: an arm can
  score 1.0 on it while burying one gold block among nine distractors.
* **the 4b abstention 2x2**, split by whether gold was present -- specifically
  the `has_gold & abstained` cell, i.e. answers the model refused to give while
  holding the evidence.

Both are descriptive; nothing here is significance-tested, because the
comparison that carries a verdict already lives in `rq4_score_entity.md` and
re-testing the same data in a second family would just be an uncorrected
second look.

Run (no GPU, seconds):
    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/rq4_entity_arm_diagnosis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rq4_score import is_abstained, parse_citations  # noqa: E402

_ANSWERS = REPO / "data" / "rq4" / "answers"
_OUTPUT = REPO / "data" / "results" / "rq4_entity_arm_diagnosis.md"
_SCORE_REPORT = REPO / "data" / "results" / "rq4_score_entity.md"
_VARIANT = "phi4_cite_all"
_ARMS = [
    "hybrid_qwen3_0.6b_semantic",
    "entity_lookup_semantic",
    "entity_boost_semantic",
]


def score_report_recalls() -> tuple[dict[str, float], str]:
    """Mean citation recall per arm, *read* from `rq4_score_entity.md`.

    The first version of this script retyped the pair (`0.4379 vs 0.1431`) into
    its own prose, which survives exactly until the arms are regenerated -- and
    a wrong number in a `data/results/*.md` file is worse than one in a doc,
    because that directory is `audit_doc_claims.py`'s *haystack*: a stale figure
    here would go on to clear the same figure quoted anywhere else.

    Returns `({}, why)` rather than a number when the report is missing, or when
    it predates the answers measured below. The staleness branch is the
    load-bearing one: quoting a recall computed from a previous generation run
    beside *these* abstention counts is this project's signature
    two-artifacts-from-different-days failure, and it never crashes -- it just
    makes one sentence quietly wrong.
    """
    if not _SCORE_REPORT.exists():
        return {}, f"{_SCORE_REPORT.name} not found"
    newest_answer = max(
        (p.stat().st_mtime for arm in _ARMS for p in (_ANSWERS / _VARIANT / arm).glob("q*.json")),
        default=0.0,
    )
    if _SCORE_REPORT.stat().st_mtime < newest_answer:
        return {}, f"{_SCORE_REPORT.name} is older than the answers measured here -- re-run rq4_score.py"

    out: dict[str, float] = {}
    col: int | None = None
    for line in _SCORE_REPORT.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # Locate `mean recall` by NAME, re-reading the header of every table.
        # Keying on (variant, arm) alone is not enough: this report repeats the
        # same two labels in its abstention 2x2, whose column 5 is a *count*, so
        # a positional read silently returns 3 instead of 0.1431. Same trap
        # `diff_significance_reports.py` keys its rows around.
        if "arm" in cells:                       # a header row: a new table starts
            col = cells.index("mean recall") if "mean recall" in cells else None
            continue
        if col is not None and len(cells) > col and cells[0] == _VARIANT and cells[1] in _ARMS:
            try:
                out[cells[1]] = float(cells[col])
            except ValueError:
                continue
    missing = [a for a in _ARMS if a not in out]
    if missing:
        return {}, f"{_SCORE_REPORT.name} has no {_VARIANT} row for {', '.join(missing)}"
    return out, ""


def measure(arm: str) -> dict:
    d = _ANSWERS / _VARIANT / arm
    rows = []
    for path in sorted(d.glob("q*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("error"):
            continue
        gold = set(r["relevant_resolution_ids"])
        blocks = r["label_map"]              # label -> resolution_id, one per block
        n_blocks = len(blocks)
        n_gold_blocks = sum(1 for rid in blocks.values() if rid in gold)
        cited, _ = parse_citations(r["answer"], blocks)
        rows.append({
            "n_blocks": n_blocks,
            "density": n_gold_blocks / n_blocks if n_blocks else 0.0,
            "has_gold": bool(r["context_has_gold"]),
            "abstained": is_abstained(r["answer"]),
            "n_gold": len(gold),
            "cited_any": bool(cited),
        })

    n = len(rows)
    cell = lambda g, a: sum(1 for x in rows if x["has_gold"] is g and x["abstained"] is a)  # noqa: E731
    return {
        "arm": arm,
        "n": n,
        "blocks": sum(x["n_blocks"] for x in rows) / n,
        "density": sum(x["density"] for x in rows) / n,
        "has_gold": sum(x["has_gold"] for x in rows),
        "n_gold": sum(x["n_gold"] for x in rows) / n,
        "abstained": sum(x["abstained"] for x in rows),
        "cited_any": sum(x["cited_any"] for x in rows),
        "answered_with_gold": cell(True, False),
        "missed": cell(True, True),
        "hallucinated": cell(False, False),
        "correct_abstain": cell(False, True),
    }


def main() -> None:
    m = [measure(a) for a in _ARMS]

    L = []
    L.append("# RQ4 entity arms: where the evidence was\n")
    L.append(f"Generated by `tools/eval/rq4_entity_arm_diagnosis.py`. Variant `{_VARIANT}`.")
    L.append("Descriptive only -- the verdicts live in `rq4_score_entity.md`.\n")

    L.append("## 1. Context composition\n")
    L.append("| arm | n | blocks/query | **gold density** | ctx holds gold | #gold/query |")
    L.append("|---|---|---|---|---|---|")
    for x in m:
        L.append(f"| `{x['arm']}` | {x['n']} | {x['blocks']:.1f} | **{x['density']:.4f}** | "
                 f"{x['has_gold']}/{x['n']} | {x['n_gold']:.1f} |")
    L.append("")
    L.append("`gold density` is the fraction of *supplied blocks* that were gold; "
             "`ctx holds gold` is the presence bit `rq4_score.py` reports. The two "
             "can disagree, which is the point of measuring the first one.\n")

    L.append("## 2. Abstention 2x2\n")
    L.append("| arm | answered w/ gold | **missed** (gold present, abstained) | "
             "hallucinated (no gold, answered) | correctly abstained | cited >0 |")
    L.append("|---|---|---|---|---|---|")
    for x in m:
        L.append(f"| `{x['arm']}` | {x['answered_with_gold']} | **{x['missed']}** | "
                 f"{x['hallucinated']} | {x['correct_abstain']} | {x['cited_any']} |")
    L.append("")

    lookup = next(x for x in m if x["arm"] == "entity_lookup_semantic")
    hybrid = next(x for x in m if x["arm"] == "hybrid_qwen3_0.6b_semantic")
    L.append("## 3. Reading\n")
    boost = next(x for x in m if x["arm"] == "entity_boost_semantic")
    L.append(
        f"`entity_lookup` supplied a **higher** gold density than shipped hybrid "
        f"({lookup['density']:.4f} vs {hybrid['density']:.4f}) and still abstained on "
        f"**{lookup['missed']}** queries whose context held gold (hybrid: "
        f"{hybrid['missed']}). So its collapse is **not** an evidence-availability "
        f"failure, and the pre-registered upper-bound inference does not go through "
        f"on this arm.\n"
    )
    L.append(
        "**The high density is itself the circularity, and the abstentions are what "
        "it costs.** These qrels define a `person`/`program`/`course` document as "
        "relevant when it *contains the entity*, and `entity_lookup` retrieves "
        "exactly the documents containing the entity -- so a near-pure gold context "
        "is true by construction, not a sign of a good context. The generator was "
        f"then handed ~{lookup['blocks']:.0f} documents that all name the entity, and on "
        f"{lookup['missed']} queries it "
        "judged that none of them answered the question asked. That is direct "
        "evidence, from an independent judge, that string containment over-counts "
        "relevance for this query shape -- the same threat "
        "`docs/eval-validity-threats.md` raises for the entity arms, here visible "
        "rather than argued.\n"
    )
    recalls, why_no_recalls = score_report_recalls()
    if recalls:
        recall_clause = (
            f"citation recall {recalls['entity_boost_semantic']:.4f} vs "
            f"{recalls['entity_lookup_semantic']:.4f}, "
        )
    else:
        recall_clause = f"citation recall: see `{_SCORE_REPORT.name}` ({why_no_recalls}), "
    L.append(
        f"**What separates the two entity arms is ranking, and it is worth more than "
        f"the dictionaries are.** Both draw on the same dictionaries and both supply "
        f"entity-bearing contexts ({lookup['density']:.4f} vs {boost['density']:.4f} "
        f"density); the difference is that `entity_boost` orders them by hybrid "
        f"relevance, so what fills the budget also answers the question "
        f"({recall_clause}missed {boost['missed']} vs "
        f"{lookup['missed']}). That gap is far larger than `entity_boost`'s entire "
        f"non-significant margin over shipped hybrid. **An exhaustive retriever's "
        f"advantage does not survive a fixed context budget it cannot rank into** -- "
        f"so `entity_lookup` bounds unranked dictionary retrieval, and "
        f"`entity_boost` is the arm that bounds the dictionaries themselves."
    )
    _OUTPUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {_OUTPUT}")


if __name__ == "__main__":
    main()
