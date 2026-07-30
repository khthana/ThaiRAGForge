"""Flag manifest titles that disagree with the document's own subject line.

A first version of this check was **prototyped and rejected on measurement**
(2026-07-30): comparing each manifest title to its document's page-1 heading with
whole-string similarity gave a median agreement of 0.660 over 2,820 files and 544
files below 0.5, nearly all false alarms. Shipping it as a gate would have meant
544 alarms to triage and no way to find the ~2 real defects among them.

The rejection was of the *metric*, not the idea. The three documented false-alarm
classes are all artifacts of the comparison rather than disagreements of substance:

  1. an agenda-number prefix on one side only (`20. `, `1.3 `, Thai numerals);
  2. one side being an abbreviation of the other (ปรับปรุง vs ปรับปรุงแก้ไข);
  3. a title truncated mid-subject -- incomplete, but not *wrong*.

This version removes all three by construction:

  * numbering and the `เรื่อง` marker are stripped from both sides first;
  * comparison is **token containment, not string similarity**, so an abbreviation
    contained in the longer form scores as agreement;
  * the score is **asymmetric** -- what fraction of the *title's* content words the
    subject line supports -- so a truncated title scores 1.0 (all of its words are
    there) instead of being punished for the words it is missing.

The defect this is actually for is topical disjointness: the title says A, the
document is about B. That is the `2568/ครั้งที่ 7` CHECO shape (a download fetched
the wrong Drive id), and it drives coverage toward 0 rather than toward 0.5 -- so
the threshold belongs near zero, and the check is tuned for precision over recall.

**Calibrated against a known case first, which changed the design.** The obvious
formulation -- "do the title's words appear anywhere in the document" -- was tried
against `2568/8`'s *ระบบ E-Portfolio*, a known real mismatch, and **fails**: the
string `Portfolio` does occur in that body, as a sub-bullet of a project whose
actual subject line is โครงการพัฒนาเว็บไซต์...Student Journey. Whole-document search
cannot distinguish "this document is about X" from "X is mentioned in passing",
so the page-1 subject line stays the comparison target.

Files whose subject line cannot be located are counted and reported **with their
denominator** rather than silently passing -- a check that quietly stops examining
its subject matter becomes a vacuous PASS (the C4 lesson in CLAUDE.md).

Run:
    PYTHONIOENCODING=utf-8 PYTHONPATH=src python tools/corpus_prep/audit_title_body_agreement.py
    ... --threshold 0.34 --report docs/title-body-agreement.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from pythainlp.tokenize import word_tokenize  # noqa: E402

CORPUS = REPO / "academic_resolutions"

# the document's own subject line, as OCR'd onto page 1
_PAGE1_HEADING = re.compile(r"มติคณะกรรมการสภาวิชาการ.{0,80}?เรื่อง\s(.{10,300})", re.S)
# agenda numbering in either digit system, e.g. "20. ", "1.3 ", "๔.๒ "
_AGENDA_NUM = re.compile(r"^[\s\d๐-๙]+(?:[.．][\s\d๐-๙]*)*[\s.)]*")
_LEAD_MARKER = re.compile(r"^(?:เรื่อง|วาระที่|ระเบียบวาระที่)\s*")
# tokens that carry no topical information, so their presence/absence says nothing
_STOP = {
    "การ", "ความ", "ของ", "และ", "หรือ", "ที่", "ใน", "เพื่อ", "กับ", "ให้", "จาก",
    "โดย", "ตาม", "เป็น", "มี", "ได้", "ต่อ", "ด้วย", "แก่", "แห่ง", "ซึ่ง", "นี้",
    "เรื่อง", "ขอ", "ทาง", "อัน", "ณ", "พ.ศ.", "ปี",
}


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s.replace(" ", " "))).strip()


def strip_prefix(s: str) -> str:
    """Remove agenda numbering and the `เรื่อง` marker, in either order."""
    prev = None
    while prev != s:
        prev = s
        s = _LEAD_MARKER.sub("", _AGENDA_NUM.sub("", s)).strip()
    return s


def subject_line(text: str) -> str | None:
    m = _PAGE1_HEADING.search(flat(text[:4000]))
    if not m:
        return None
    # the subject runs to the end of the heading sentence; body prose follows a
    # blank line in the source, which flat() collapsed, so cut at a sane length
    return strip_prefix(flat(m.group(1)))[:300]


def content_tokens(s: str) -> list[str]:
    out = []
    for tok in word_tokenize(s, engine="newmm"):
        tok = tok.strip()
        if len(tok) < 2 or tok in _STOP or tok.isdigit():
            continue
        if not any(ch.isalnum() for ch in tok):
            continue
        out.append(tok)
    return out


def coverage(title: str, subject: str) -> tuple[float, list[str]]:
    """Fraction of the title's content words the subject line supports.

    Substring containment against the whole subject string, not set intersection
    of two tokenizations: Thai has no word boundaries, so the two sides can be
    segmented differently for the same text and a token-set comparison would
    report a spurious miss. Containment is segmentation-independent."""
    toks = content_tokens(strip_prefix(title))
    if not toks:
        return 1.0, []
    hay = subject.replace(" ", "")
    missing = [t for t in toks if t.replace(" ", "") not in hay]
    return 1 - len(missing) / len(toks), missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.34,
                    help="flag files at or below this coverage")
    ap.add_argument("--report", type=str, default="")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    scored: list[tuple[float, str, str, str, list[str]]] = []
    no_subject: list[str] = []
    n_files = 0

    for manifest in sorted(CORPUS.glob("*/*/meeting_manifest.json")):
        folder = manifest.parent
        rel = f"{folder.parent.name}/{folder.name}"
        for entry in json.loads(manifest.read_bytes().decode("utf-8-sig")):
            md = folder / entry["file"]
            if not md.exists():
                continue
            n_files += 1
            subject = subject_line(md.read_text(encoding="utf-8"))
            if subject is None:
                no_subject.append(f"{rel}/{entry['file']}")
                continue
            cov, missing = coverage(entry["title"], subject)
            scored.append((cov, rel, entry["title"], subject, missing))

    scored.sort(key=lambda r: r[0])
    covs = [s[0] for s in scored]
    flagged = [s for s in scored if s[0] <= args.threshold]

    lines = [
        "# Title-vs-subject-line agreement",
        "",
        f"- {n_files:,} files; subject line located for {len(scored):,}, "
        f"not located for {len(no_subject):,} (reported below, not flagged)",
        f"- coverage = fraction of the *title's* content words the page-1 subject "
        f"line supports; asymmetric, so a truncated title scores 1.0",
        f"- median **{statistics.median(covs):.3f}**, mean {statistics.fmean(covs):.3f}; "
        f"perfect (1.0) for {sum(1 for c in covs if c == 1.0):,} "
        f"({sum(1 for c in covs if c == 1.0) / len(covs):.1%})",
        f"- **{len(flagged)} flagged at coverage <= {args.threshold}** "
        f"({len(flagged) / len(scored):.2%} of files with a locatable subject line)",
        "",
        "For comparison, the rejected first formulation (whole-string similarity vs "
        "the same heading) had median 0.660 and 544 files below 0.5.",
        "",
        "## Flagged",
        "",
        "| coverage | meeting | manifest title | page-1 subject line | title words absent |",
        "|---|---|---|---|---|",
    ]
    for cov, rel, title, subject, missing in flagged:
        lines.append(
            f"| {cov:.2f} | {rel} | {title[:70]} | {subject[:70]} | {', '.join(missing[:6])} |"
        )
    if no_subject:
        lines += ["", f"## Subject line not located ({len(no_subject)})", ""]
        lines += [f"- {p}" for p in no_subject[:60]]
        if len(no_subject) > 60:
            lines.append(f"- ... and {len(no_subject) - 60} more")

    report = "\n".join(lines) + "\n"
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"written to {args.report}")
    print("\n".join(lines[: 14 + args.show]))

    dist = [(lo, sum(1 for c in covs if lo <= c < lo + 0.1)) for lo in [i / 10 for i in range(10)]]
    print("\ncoverage distribution:")
    for lo, n in dist:
        print(f"  {lo:.1f}-{lo + 0.1:.1f}  {n:5,}  {'#' * min(60, n * 60 // max(1, len(covs)))}")
    print(f"  ==1.0     {sum(1 for c in covs if c == 1.0):5,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
