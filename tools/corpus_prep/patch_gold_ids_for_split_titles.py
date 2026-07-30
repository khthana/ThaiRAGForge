"""Re-point gold-set `relevant_resolution_ids` at the repaired 2567/1 split ids.

`fix_manifest_title_collisions.py` gave the two `__1`/`__2` pieces of the
2567/ครั้งที่ 1 คณะวิศวกรรมศาสตร์ bundle their distinct per-curriculum titles
(they had both been patched with the same one). Their `resolution_id`s change
with the title, so the single merged id the gold sets cite would otherwise
dangle -- silently costing the affected queries a relevant document they
should be credited for.

One old id becomes two: both pieces are revisions of the same
วิศวกรรมชีวการแพทย์ curriculum (differing only in ฉบับปี ๒๕๖๓ vs ๒๕๖๔) and both
contain the two graded course names, so under
`build_gold_candidates.py`'s rules (program history = title names the program;
course = resolution is tagged with the course) each piece is relevant on its
own. That raises those queries' relevant-set size by one, which is the point:
there really are two distinct relevant resolutions where the merged id
presented one.

Run:
    python tools/corpus_prep/patch_gold_ids_for_split_titles.py           # dry run
    python tools/corpus_prep/patch_gold_ids_for_split_titles.py --apply
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# Both files were machine-dumped, so a load/modify/dump round-trip is safe --
# but only at the width each was originally written with, verified below to
# reproduce the untouched file byte-for-byte. Anything else rewraps every long
# Thai id in the file and buries a 2-line change in a whole-file diff.
GOLD = {
    Path("config/eval/gold_query_set_73det.yaml"): 100,
    Path("config/eval/gold_query_set.yaml"): 1000,
}

_STEM = (
    "2567/1/เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) "
    "คณะวิศวกรรมศาสตร์ — หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมชีวการแพทย์ "
    "(หลักสูตรนานาชาติ)"
)
OLD = _STEM
NEW = [
    f"{_STEM} (พหุวิทยาการ) (การปรับปรุงแก้ไขหลักสูตร ฉบับปี พ.ศ. ๒๕๖๓)",
    f"{_STEM} (พหุวิทยาการ) (การปรับปรุงแก้ไขหลักสูตร ฉบับปี พ.ศ. ๒๕๖๔)",
]


def patch(entries: list[dict]) -> int:
    """Replace the merged id with both new ones, in place. Returns the number
    of queries touched."""
    n = 0
    for entry in entries:
        ids = entry.get("relevant_resolution_ids") or []
        if OLD not in ids:
            continue
        at = ids.index(OLD)
        entry["relevant_resolution_ids"] = ids[:at] + list(NEW) + ids[at + 1 :]
        n += 1
        print(f"  {entry['query'][:78]}")
        print(f"    relevant ids {len(ids)} -> {len(entry['relevant_resolution_ids'])}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    total = 0
    for path, width in GOLD.items():
        original = path.read_text(encoding="utf-8")
        entries = yaml.safe_load(original)
        if yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, width=width) != original:
            raise SystemExit(
                f"{path} does not round-trip at width={width}; re-check the dump "
                "settings before rewriting it"
            )
        print(f"{path}:")
        n = patch(entries)
        total += n
        if n and args.apply:
            path.write_text(
                yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, width=width),
                encoding="utf-8",
            )
            print(f"  -> wrote {path}")
    print(f"\n{total} query/queries {'written' if args.apply else 'found (dry run)'}")
    if not total:
        print("nothing to do -- already patched, or the id no longer appears")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
