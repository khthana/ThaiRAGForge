"""Verify every figure in main.tex against the report that generates it.

Why this exists, and why it is not the repo's `audit_doc_claims.py` D2 check.
D2 asks whether a figure in prose appears *somewhere* in *some* report -- a bag
of numbers. Scored against its own perturbations it clears a wrong 4-decimal
number 77% of the time, because ten other reports carry the same value for a
different quantity. A paper cannot be defended that way: a reviewer asks what
*this* number is, of *this* quantity, and a figure that is a real value of the
wrong subject is the exact error this project has already made once in its own
guidance.

So every claim below is a (report, row, column) -> value assertion: it names
the quantity, reads the current artifact, and compares. If a report is
refreshed and a value moves, this goes red and names the figure -- which is the
whole point, since a paper is written once and the reports keep moving.

Run:  .venv/Scripts/python.exe paper/isai-nlp-2026/check_paper_figures.py
Exit 1 on any mismatch. Run it before every submission and after any RQ4
re-score.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "data" / "results"
PAPER = Path(__file__).resolve().parent / "main.tex"

# (label, report, (variant, arm) matched by EXACT CELL equality, n_cols, col, expected)
#
# Cell equality, not substring containment in the line: `phi4` is a prefix of
# `phi4_cite_all` and `gemma4_e4b_cite_all` of `..._guarded`, so a substring
# match silently selects three rows where it means one. The first version of
# this file did exactly that and reported 38 failures against a paper whose
# figures were all correct -- the instrument was wrong, not the numbers.
#
# n_cols disambiguates the descriptive table from the abstention 2x2, which
# carry the same (variant, arm) pair.
CLAIMS: list[tuple[str, str, tuple[str, str], int, int, str]] = []


def desc(model_report: str, variant: str, arm: str, prec: str, rec: str, phantom: str) -> None:
    for col, val, what in ((4, prec, "precision"), (6, rec, "recall"), (7, phantom, "phantom")):
        CLAIMS.append((f"{variant}/{arm} {what}", model_report, (variant, arm), 9, col, val))


def abst(model_report: str, variant: str, arm: str, halluc: str, abstained: str) -> None:
    CLAIMS.append((f"{variant}/{arm} hallucinated", model_report, (variant, arm), 8, 5, halluc))
    CLAIMS.append((f"{variant}/{arm} abstained", model_report, (variant, arm), 8, 6, abstained))


# --- Table III: phi4 citation precision/recall (rq4_score.md, descriptive) ---
R = "rq4_score.md"
desc(R, "phi4", "hybrid_qwen3_0.6b_semantic", "0.7013", "0.2952", "0/293")
desc(R, "phi4", "dense_qwen3_0.6b_semantic", "0.6867", "0.2738", "0/311")
desc(R, "phi4", "bm25_semantic", "0.6591", "0.2329", "0/235")
desc(R, "phi4", "hybrid_m2v_semantic", "0.5417", "0.1807", "0/228")
desc(R, "phi4_cite_all", "hybrid_qwen3_0.6b_semantic", "0.7185", "0.3823", "0/416")
desc(R, "phi4_cite_all", "dense_qwen3_0.6b_semantic", "0.6381", "0.3145", "4/378")
desc(R, "phi4_cite_all", "bm25_semantic", "0.5801", "0.3118", "0/354")
desc(R, "phi4_cite_all", "hybrid_m2v_semantic", "0.5057", "0.1989", "0/297")
desc(R, "phi4_cite_all_guarded", "hybrid_qwen3_0.6b_semantic", "0.6945", "0.3794", "0/406")
desc(R, "phi4_cite_all_guarded", "dense_qwen3_0.6b_semantic", "0.6650", "0.3481", "0/386")
desc(R, "phi4_cite_all_guarded", "bm25_semantic", "0.5968", "0.2733", "0/303")
desc(R, "phi4_cite_all_guarded", "hybrid_m2v_semantic", "0.4539", "0.1761", "0/273")

# --- Table V: closed_book abstention, phi4 ---
abst(R, "phi4", "closed_book", "0", "106")
abst(R, "phi4_cite_all", "closed_book", "2", "104")
abst(R, "phi4_cite_all_guarded", "closed_book", "0", "106")
CLAIMS.append(("phi4 cite_all closed_book phantom", R,
                ("phi4_cite_all", "closed_book"), 9, 7, "5/5"))
CLAIMS.append(("phi4 guarded closed_book phantom", R,
                ("phi4_cite_all_guarded", "closed_book"), 9, 7, "0/0"))

# --- gemma4:e4b -- the 24 -> 1 result and its control ---
G = "rq4_score_gemma4.md"
abst(G, "gemma4_e4b_cite_all", "closed_book", "24", "82")
abst(G, "gemma4_e4b_cite_all_guarded", "closed_book", "1", "105")
CLAIMS.append(("gemma cite_all closed_book phantom", G,
                ("gemma4_e4b_cite_all", "closed_book"), 9, 7, "37/37"))
CLAIMS.append(("gemma guarded closed_book phantom", G,
                ("gemma4_e4b_cite_all_guarded", "closed_book"), 9, 7, "1/1"))
# the control: the answering arms must NOT move
for variant in ("gemma4_e4b_cite_all", "gemma4_e4b_cite_all_guarded"):
    abst(G, variant, "hybrid_qwen3_0.6b_semantic", "2", "2")
    abst(G, variant, "dense_qwen3_0.6b_semantic", "4", "2")
# levels quoted in the text
CLAIMS.append(("gemma cite_all hybrid recall", G,
                ("gemma4_e4b_cite_all", "hybrid_qwen3_0.6b_semantic"), 9, 6, "0.4956"))
CLAIMS.append(("gemma cite_all dense recall", G,
                ("gemma4_e4b_cite_all", "dense_qwen3_0.6b_semantic"), 9, 6, "0.5049"))
CLAIMS.append(("gemma guarded hybrid recall", G,
                ("gemma4_e4b_cite_all_guarded", "hybrid_qwen3_0.6b_semantic"), 9, 6, "0.5260"))

# --- Table IV: significance family 2, scoped to that section ---
FAM2 = [
    ("hybrid recall", "hybrid_qwen3_0.6b_semantic[recall]", "+0.0871", "[+0.0438, +0.1307]", "0.0000"),
    ("bm25 recall", "bm25_semantic[recall]", "+0.0789", "[+0.0440, +0.1174]", "0.0000"),
    ("dense recall", "dense_qwen3_0.6b_semantic[recall]", "+0.0407", "[-0.0133, +0.0932]", "0.6610"),
    ("m2v recall", "hybrid_m2v_semantic[recall]", "+0.0182", "[-0.0136, +0.0489]", "0.7398"),
    ("bm25 precision", "bm25_semantic[precision]", "-0.0644", "[-0.1155, -0.0134]", "0.0798"),
    ("hybrid precision", "hybrid_qwen3_0.6b_semantic[precision]", "+0.0043", "[-0.0559, +0.0636]", "1.0000"),
]


def read(name: str) -> str:
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"FATAL: {p} is missing -- the paper cannot be checked")
    return p.read_text(encoding="utf-8")


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.split("|")]


def check_table_claims() -> list[str]:
    failures = []
    cache: dict[str, list[str]] = {}
    for label, report, (variant, arm), ncols, col, expected in CLAIMS:
        lines = cache.setdefault(report, read(report).splitlines())
        hits = [c for ln in lines
                if len(c := cells(ln)) == ncols and c[1] == variant and c[2] == arm]
        if len(hits) != 1:
            failures.append(f"{label}: matched {len(hits)} rows in {report}, expected exactly 1")
            continue
        got = hits[0][col]
        if got != expected:
            failures.append(f"{label}: paper says {expected}, {report} says {got}")
    return failures


def check_family2() -> list[str]:
    text = read("rq4_score.md")
    start = text.find("## Significance family 2")
    end = text.find("## Significance family 3")
    if start < 0 or end < 0 or end < start:
        return ["family 2 section not found in rq4_score.md (renamed? re-scope this check)"]
    section = text[start:end]
    failures = []
    for label, key, diff, ci, holm in FAM2:
        pat = f"{key}:sentence_cap vs {key}:cite_all"
        rows = [ln for ln in section.splitlines() if pat in ln]
        if len(rows) != 1:
            failures.append(f"family2 {label}: matched {len(rows)} rows, expected 1")
            continue
        c = cells(rows[0])
        for what, idx, want in (("diff", 4, diff), ("CI", 5, ci), ("Holm p", 7, holm)):
            got = c[idx]
            if got.replace(" ", "") != want.replace(" ", ""):
                failures.append(f"family2 {label} {what}: paper says {want}, report says {got}")
    return failures


def check_paper_has_them() -> list[str]:
    """Every expected value must actually appear in main.tex.

    Without this the checks above are vacuous in the dangerous direction: a
    figure silently dropped from the paper still 'passes' every assertion about
    the report.
    """
    tex = PAPER.read_text(encoding="utf-8")
    wanted = {e for *_, e in CLAIMS if re.fullmatch(r"0\.\d{4}", e)}
    wanted |= {d for _, _, d, _, _ in FAM2}
    missing = sorted(w for w in wanted if w.lstrip("+") not in tex and w not in tex)
    return [f"value {m} is asserted here but appears nowhere in main.tex" for m in missing]


def main() -> int:
    failures = check_table_claims() + check_family2() + check_paper_has_them()
    n = len(CLAIMS) + len(FAM2) * 3
    if failures:
        print(f"FAIL -- {len(failures)} of {n} paper figures do not match their report:\n")
        for f in failures:
            print(f"  {f}")
        print("\nA report was refreshed, or a figure was typed by hand. Fix main.tex, "
              "not this file.")
        return 1
    print(f"PASS -- all {n} figures in main.tex match the current reports "
          f"(rq4_score.md, rq4_score_gemma4.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
