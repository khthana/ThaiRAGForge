"""Find agenda items that have no document of their own.

Two items in one meeting cannot both be described by one document, so when two
files carry the same content exactly one title is right and the other item was
never really fetched. That is the `2568/ครั้งที่ 7` CHECO defect, which was found
one file at a time; this script asks the question of the whole corpus at once.

**Two signals, because one of them undercounts.** The obvious check is a hash of
the OCR text (dropping the filename-derived `# Document:` header line, which
differs even when the OCR is identical). It is exact and proves the two files came
from one PDF -- but it only fires when the *same* PDF was fetched twice. When the
source holds two separate scans or exports of one document, the OCR differs in a
few characters and the hash misses it: `2564/ครั้งที่ 5` items 18-21 are four
files, four distinct hashes, and one subject line between them.

So the second signal is the document's own page-1 `เรื่อง` subject line, compared
within a meeting. It is segmentation-independent (no threshold, no tokenizer) and
catches the separate-export case the hash cannot. It is a strict superset in
practice, but both are reported because they mean different things: a hash
collision identifies *which PDF* was duplicated, a subject collision only says at
most one of the titles can be right.

`audit_title_body_agreement.py` does not see any of this. It scores each title
against its *own* body, so two items that share a body both score against that
one subject line -- and where the titles differ only in a faculty name, the
shared boilerplate carries both well above its 0.34 threshold (items 20 and 21
score 0.692 and 0.583 against a subject line naming a third faculty entirely).
Comparing items to *each other* is what makes the defect visible.

Hash groups are classified, because not every collision is a defect:

  * `same-item-variant` -- members are one agenda item under decorated
    filenames: the `__N` piece index `split_curriculum_bundles.py` appends
    (ADR-0004), or the ` (N)` suffix the download stage adds when it fetches one
    document twice. Duplicate text is expected here and is not a missing item.
  * `same-meeting` -- different titles inside one meeting folder. The defect shape.
  * `cross-meeting` -- the same text filed under two meetings. Usually a genuine
    re-tabled item; judge case by case.

Report only -- writes nothing, so it is safe to run at any time.

Run:
    PYTHONIOENCODING=utf-8 python tools/corpus_prep/scan_duplicate_bodies.py
    ... --show same-item-variant         # include the expected class too
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_title_body_agreement import flat, subject_line  # noqa: E402

CORPUS = REPO / "academic_resolutions"
# Filename decorations that mark a *variant of one agenda item* rather than a
# different item: `__N` is the piece index `split_curriculum_bundles.py` appends
# (ADR-0004), ` (N)` is the browser-style suffix the download stage adds when it
# fetches the same document twice. Both are stripped before asking whether two
# files belong to different items, and both may repeat legitimately.
SPLIT_SUFFIX = re.compile(r"(?:\s*\(\d+\))?(?:__\d+)?$")


def body_of(path: Path) -> str:
    """File text minus the filename-derived `# Document:` header line."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    first, sep, rest = raw.partition("\n")
    return rest if sep and first.startswith("# Document:") else raw


def classify(members: list[Path]) -> str:
    stems = {SPLIT_SUFFIX.sub("", m.stem) for m in members}
    if len(stems) == 1 and len(members) > 1:
        return "same-item-variant"
    return "same-meeting" if len({m.parent for m in members}) == 1 else "cross-meeting"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--show", action="append", default=None,
                    help="hash classes to print (repeatable); default all but same-item-variant")
    args = ap.parse_args()

    files = sorted(args.corpus.rglob("*.md"))
    by_hash: dict[str, list[Path]] = defaultdict(list)
    by_subject: dict[tuple[Path, str], list[Path]] = defaultdict(list)
    no_subject = 0

    for f in files:
        body = body_of(f)
        by_hash[hashlib.sha256(body.encode("utf-8")).hexdigest()].append(f)
        subj = subject_line(body)
        if subj is None:
            no_subject += 1
            continue
        # Compared at full length, deliberately. Truncating to a prefix looks
        # attractive (the subject runs on into body prose) and is a trap: at 60
        # characters this reports 229 groups / 1,255 orphans, because curriculum
        # items share a long boilerplate opening and the faculty that
        # distinguishes them falls past the cut. The count collapses 229 -> 28
        # between 60 and 80 and is then flat from 100 to full length (13 -> 11
        # groups), with both known separate-export cases grouped at every window.
        # Full length is the stable end of that plateau and needs no constant.
        by_subject[(f.parent, flat(subj))].append(f)

    hash_groups = [(h, m) for h, m in by_hash.items() if len(m) > 1]
    counts: dict[str, int] = defaultdict(int)
    for _, members in hash_groups:
        counts[classify(members)] += 1

    print(f"{len(files):,} files scanned  ({no_subject} with no locatable subject line)\n")
    print(f"A. identical OCR text -- {len(hash_groups)} groups")
    for cls in ("same-meeting", "cross-meeting", "same-item-variant"):
        print(f"     {cls:<18} {counts[cls]}")

    subj_groups = [(k, m) for k, m in by_subject.items()
                   if len(m) > 1 and classify(m) != "same-item-variant"]
    orphans = sum(len(m) - 1 for _, m in subj_groups)
    print(f"\nB. one subject line, several agenda items -- {len(subj_groups)} groups, "
          f"{orphans} items with no document of their own")

    show = set(args.show) if args.show else {"same-meeting", "cross-meeting"}
    for h, members in sorted(hash_groups, key=lambda g: str(g[1][0])):
        cls = classify(members)
        if cls in show:
            print(f"\n[A {cls}] {h[:16]}  {len(members)} files")
            for m in members:
                print(f"    {m.relative_to(args.corpus)}")

    for (folder, subj), members in sorted(subj_groups, key=lambda g: str(g[0][0])):
        print(f"\n[B] {folder.relative_to(args.corpus)}  {len(members)} items share one subject")
        print(f"    subject: {subj}")
        for m in members:
            print(f"      {m.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
