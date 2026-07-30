"""Name the meeting in every thematic gold query, so the query is answerable.

All 179 `thematic` entries in `config/eval/gold_query_set.yaml` are
**meeting-scoped**: each one's `relevant_resolution_ids` come from exactly one
meeting (verified: 179 of 179 span a single `<year>/<session>`). But every one of
them phrases the question as "ในการประชุม**ครั้งนี้** ..." -- *this* meeting -- and
"this" is never stated. A retriever therefore cannot know which of ~120 meetings
is meant, and any meeting that discussed the same theme is an equally good answer
while being scored wrong.

That is the root cause behind two things previously recorded as separate:

  * the audit's E2 warning about 5 duplicated query strings. Those are simply the
    cases where two different meetings produced byte-identical text; their gold
    sets are **disjoint** (shared=0 for all 5 pairs), which is the giveaway that
    they were never meant to be the same question.
  * the thematic subset's near-zero chunker discrimination (t=0.02, 67% ties,
    docs/paper-results-summary.md). Not evidence that chunking does not matter --
    evidence that the queries were unanswerable as posed, so the scores were
    noise.

The fix is a rewrite, not a deletion: the meeting identity is already present in
each entry's own gold ids, so it can be moved into the query text without adding
information or re-judging relevance. Special sessions (วาระพิเศษ, the `s` suffix
per ADR-0003) are spelled out rather than left as a bare "4s", which would not be
a question anyone would type.

Safe by construction:
  * the whole file is round-tripped through yaml first and the result compared
    byte-for-byte with the original, so a formatting difference aborts before any
    write (the same guard patch_gold_ids_for_split_titles.py uses -- long Thai
    scalars wrap across lines, so line-based patching silently misses entries);
  * only `query` strings change -- no `relevant_resolution_ids`, no entity
    metadata, no non-thematic entry;
  * idempotent: an already-qualified query contains no "ครั้งนี้" to replace;
  * no persisted result set is invalidated. Results are keyed by
    `sha256(query)`, so rewriting a query would orphan its cache -- checked
    first: **no live result directory answers a thematic query** (they all lived
    in the retired sets archived off-repo on 2026-07-30).

Run:
    PYTHONPATH=src python tools/eval/qualify_thematic_queries.py          # dry run
    PYTHONPATH=src python tools/eval/qualify_thematic_queries.py --apply
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

GOLD = Path("config/eval/gold_query_set.yaml")
WIDTH = 1000  # matches how the file is currently wrapped; verified by round-trip
NEEDLE = "การประชุมครั้งนี้"


def meeting_phrase(year: str, session: str) -> str:
    """'2566', '5s' -> 'การประชุมวาระพิเศษ ครั้งที่ 5/2566'."""
    if session.endswith("s"):
        return f"การประชุมวาระพิเศษ ครั้งที่ {session[:-1]}/{year}"
    return f"การประชุมครั้งที่ {session}/{year}"


def qualify(entry: dict) -> str | None:
    """The entry's query with 'this meeting' replaced by the meeting it means."""
    ids = entry.get("relevant_resolution_ids") or []
    meetings = {tuple(str(r).split("/")[:2]) for r in ids}
    if len(meetings) != 1:
        return None  # not meeting-scoped; leave alone rather than guess
    (year, session), = meetings
    query = entry["query"]
    if NEEDLE not in query:
        return None
    return query.replace(NEEDLE, meeting_phrase(year, session))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    original = GOLD.read_text(encoding="utf-8")
    entries = yaml.safe_load(original)
    if yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, width=WIDTH) != original:
        raise SystemExit(
            f"{GOLD} does not round-trip through yaml unchanged -- refusing to rewrite it. "
            "Fix the dump settings first; a mismatch here would reformat all 252 entries."
        )

    changed, skipped = 0, []
    for entry in entries:
        if entry.get("entity_type") != "thematic":
            continue
        new = qualify(entry)
        if new is None:
            skipped.append(entry["query"][:60])
            continue
        if new != entry["query"]:
            entry["query"] = new
            changed += 1

    dupes = {q: n for q, n in Counter(e["query"] for e in entries).items() if n > 1}
    print(f"{len(entries)} entries; {changed} thematic queries qualified with their meeting")
    if skipped:
        print(f"  {len(skipped)} thematic entries left alone (not meeting-scoped or already qualified):")
        for q in skipped[:5]:
            print(f"    {q}")
    print(f"  duplicate query strings after rewrite: {len(dupes)}")
    for q, n in list(dupes.items())[:5]:
        print(f"    x{n} {q[:70]}")
    sample = next((e for e in entries if e.get("entity_type") == "thematic"), None)
    if sample:
        print(f"\n  example: {sample['query'][:110]}")

    if not args.apply:
        print("\ndry run -- pass --apply to write")
        return 0
    if dupes:
        raise SystemExit("refusing to write: duplicates remain, the rewrite did not resolve them")

    GOLD.write_text(
        yaml.safe_dump(entries, allow_unicode=True, sort_keys=False, width=WIDTH),
        encoding="utf-8",
    )
    print(f"\nwrote {GOLD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
