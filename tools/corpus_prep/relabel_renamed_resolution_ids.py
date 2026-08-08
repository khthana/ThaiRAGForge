"""Relabel `resolution_id`/`chunk_id` after a 1:1 title rename.

Sibling of `relabel_index_resolution_ids.py`, for the *other* shape of title
repair. That script handles the hard case -- several files sharing one id, split
apart by `#N` -- where nothing in a stored row says which source file it came
from, so it has to attribute chunks by `chunk_index` block structure and
cross-check against text. Here every old id maps to exactly one new id, so
attribution is a dict lookup and none of that machinery is needed or wanted.

Driven by `fix_manifest_title_mispairings.py`, which repaired the 4 mispairings
`audit_title_body_agreement.py` found. Chunk *text* did not change and
embeddings are a function of text alone, so this is a rename, not a rebuild: no
GPU, no re-embedding, and rewriting a persisted result is exactly equivalent to
re-running retrieval against the relabelled index.

**The mapping must be applied atomically.** Two of the four ids are a mutual
A<->B swap (`2565/8`: the credit-transfer and online-teaching entries had each
other's titles). Applied as two sequential renames, A->B would land on B's rows
before B->A ran and both documents would collapse onto one id -- the precise
defect this repair exists to remove. Reading the old value and writing the new
one in a single pass over the original column makes that unrepresentable, and
`--check` asserts the row counts per id are preserved across the swap.

No `chunks.parquet` backup is written, unlike the shared-id script. There it is
needed because splitting one id into several is not invertible from the result
alone; a 1:1 rename is a bijection (checked: old and new sets are the same size
and every id stays unique), so the repair is undone by re-running with the map
inverted -- a few kilobytes of JSON instead of ~1.5 GB of parquet copies.

The map is *derived*, never hand-written: `--map` takes the diff of two
`resolution_id` snapshots taken with the real loader either side of the manifest
edit, so a typo in a 90-character Thai title cannot silently mint a third id.

Run:
    PYTHONPATH=src python tools/corpus_prep/relabel_renamed_resolution_ids.py --map m.json
    PYTHONPATH=src python tools/corpus_prep/relabel_renamed_resolution_ids.py --map m.json --apply
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INDEX_ROOT = Path("data/index")
RESULTS_ROOT = Path("data/results")
GOLD_ROOT = Path("config/eval")


def combos() -> list[Path]:
    return [
        d
        for parent in sorted(INDEX_ROOT.iterdir())
        if parent.is_dir()
        for d in sorted(parent.iterdir())
        if (d / "chunks.parquet").exists()
    ]


def relabel_indices(mapping: dict[str, str], apply: bool) -> tuple[int, int]:
    """Rewrite both id columns in every `chunks.parquet` holding an affected id.

    Only `chunks.parquet` carries ids: `lexical.json` is a row-aligned list
    (positions, not labels), `embeddings.npy` is row-aligned too, and
    `meta.json`/`manifest.json` record the combo, not its documents. So a rename
    touches exactly one file per index.
    """
    touched = rows_changed = 0
    for d in combos():
        table = pq.read_table(d / "chunks.parquet")
        cols = table.to_pydict()
        rids, cidx, cids = cols["resolution_id"], cols["chunk_index"], cols["chunk_id"]
        positions = [i for i, r in enumerate(rids) if r in mapping]
        if not positions:
            continue

        # A new id already present on rows we are NOT relabelling would be a
        # different document, and renaming onto it merges the two -- invisibly,
        # since the row counts among affected rows would still balance. The
        # corpus-wide snapshot says the new ids are unique, but an index can
        # hold ids the current corpus no longer mints, so check the artifact
        # rather than trusting the corpus to speak for it.
        affected = set(positions)
        incoming = set(mapping.values())
        squatters = {rids[i] for i in range(len(rids))
                     if i not in affected and rids[i] in incoming}
        if squatters:
            raise SystemExit(
                f"{d}: {len(squatters)} new id(s) already present on unaffected rows "
                "-- relabelling would merge two documents:\n"
                + "\n".join(f"  {s}" for s in sorted(squatters))
            )

        before = Counter(rids[i] for i in positions)
        for i in positions:
            new_id = mapping[rids[i]]
            # a stored chunk_id that isn't <resolution_id>:<chunk_index> means
            # the convention changed and rebuilding it here would corrupt it
            expected = f"{rids[i]}:{cidx[i]}"
            if cids[i] != expected:
                raise SystemExit(
                    f"{d}: row {i} chunk_id {cids[i]!r} != {expected!r} -- "
                    "chunk_id convention changed, refusing to rewrite"
                )
            rids[i], cids[i] = new_id, f"{new_id}:{cidx[i]}"
        after = Counter(rids[i] for i in positions)
        # the swap-safety check: a rename permutes labels, it never merges them
        if sorted(before.values()) != sorted(after.values()) or len(before) != len(after):
            raise SystemExit(f"{d}: id row-counts changed across relabel -- ids merged")

        touched += 1
        rows_changed += len(positions)
        print(f"  {d.parent.name}/{d.name}: {len(positions)} rows"
              + ("" if apply else " (dry run)"))
        if apply:
            new_table = pa.table({name: cols[name] for name in table.column_names})
            tmp = d / "chunks.parquet.tmp"
            pq.write_table(new_table, tmp)
            tmp.replace(d / "chunks.parquet")
            manifest = d / "manifest.json"
            if manifest.exists():
                man = json.loads(manifest.read_text(encoding="utf-8"))
                man["relabeled_mispairings"] = {
                    "reason": "manifest title mispairing repair 2026-08-08 "
                              "(see docs/title-body-agreement.md)",
                    "rows": len(positions),
                    # read by audit_pipeline_invariants.py's I6: a title repair
                    # edits meeting_manifest.json without touching any .md, so
                    # this is what tells I6 the index was brought current
                    # without being rebuilt
                    "at": datetime.now().isoformat(timespec="seconds"),
                }
                manifest.write_text(
                    json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8"
                )
    return touched, rows_changed


def relabel_results(mapping: dict[str, str], apply: bool) -> tuple[int, int]:
    """Rewrite the same ids in persisted retrieval results.

    Metrics are computed from these files, not from the index, so relabelling
    the index alone would leave every affected row graded against an id nothing
    can match. Unlike the shared-id case this needs no (chunk_id, text) key:
    the old id identifies its document uniquely, so `resolution_id` alone is an
    exact attribution.
    """
    changed_files = changed_rows = 0
    by_dir: Counter[str] = Counter()
    for rdir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        for f in sorted(rdir.rglob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or "results" not in data:
                continue
            touched = 0
            for row in data["results"]:
                new_id = mapping.get(row.get("resolution_id"))
                if new_id is None:
                    continue
                row["resolution_id"] = new_id
                chunk_id = row.get("chunk_id")
                if isinstance(chunk_id, str) and ":" in chunk_id:
                    row["chunk_id"] = f"{new_id}:{chunk_id.rsplit(':', 1)[1]}"
                touched += 1
            if touched:
                changed_files += 1
                changed_rows += touched
                by_dir[rdir.name] += 1
                if apply:
                    f.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
    for name, n in by_dir.most_common():
        print(f"  {name:42} {n:>5} file(s)")
    print(f"  {changed_files} file(s), {changed_rows} row(s) "
          f"{'rewritten' if apply else 'to rewrite (dry run)'}")
    return changed_files, changed_rows


def check_gold(mapping: dict[str, str]) -> None:
    """Report whether any gold query set cites an affected id.

    Printed with its denominator on purpose: '0 affected' is ambiguous between
    'checked and clean' and 'the pattern never could have matched', which is how
    a vacuous check hides (see `audit_pipeline_invariants.py`'s E3 counters).
    """
    total = hits = 0
    for f in sorted(GOLD_ROOT.rglob("*.yaml")):
        text = f.read_text(encoding="utf-8")
        n = sum(text.count(old) for old in mapping)
        m = sum(text.count(new) for new in mapping.values())
        total += text.count("resolution_id")
        if n or m:
            hits += n + m
            print(f"  {f}: {n} old id ref(s), {m} new id ref(s)")
    print(f"  {hits} affected reference(s) across {total} gold resolution_id entries")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, help="JSON object: old id -> new id")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    if not mapping:
        raise SystemExit("empty mapping")
    if set(mapping) & set(mapping.values()):
        print(f"note: mapping contains a swap "
              f"({len(set(mapping) & set(mapping.values()))} id(s) both old and new) "
              "-- applied in one pass over the original column")

    print(f"\nmapping ({len(mapping)} id(s)):")
    for old, new in mapping.items():
        print(f"  - {old}\n  + {new}")

    print("\ngold query sets:")
    check_gold(mapping)

    print("\nindices:")
    touched, rows = relabel_indices(mapping, args.apply)
    print(f"  {touched} index(es), {rows} rows")

    print("\npersisted results:")
    relabel_results(mapping, args.apply)

    if not args.apply:
        print("\ndry run -- re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
