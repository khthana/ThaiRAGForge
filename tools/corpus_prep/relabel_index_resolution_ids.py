"""Relabel `resolution_id`/`chunk_id` in built indices after a title repair.

`fix_manifest_title_collisions.py` changed 5 files' titles and
`make_resolution_id` now ranks 2 more, so 7 of 2,853 files carry a different
`resolution_id` than when the indices were built. Chunk *text* did not change,
and embeddings are a function of text alone -- so this is a rename, not a
rebuild: no GPU, no re-embedding.

Why it cannot be skipped: relevance is judged per `resolution_id` (ADR-0002).
Until the artifacts agree with the gold set, 3 of 106 gold queries are graded
against ids their index does not contain.

**How a chunk is attributed to its source file.** Two files that shared one id
have their chunks stored under that one id, and nothing in the row says which
file each came from -- except `chunk_index`, which the chunker restarts at 0 per
resolution. So within one id, a `chunk_index` back at 0 marks the start of the
next source file, and blocks appear in the loader's order (`sorted()` over
paths). That signal is text-independent, which matters: the 8 pre-2026-07-24
combo dirs hold pre-OCR-remediation text, so matching chunk text against the
corpus as it is now would fail for them even where nothing is wrong. Where the
text *is* current, it is used as a cross-check and a mismatch aborts.

Not handled here: 21 ids in those same 8 superseded combo dirs that trace to the
gitignored tooling reports ingested by the pre-fix corpus walk (Open item #11).
Those are not renamed documents, they are documents that should never have been
indexed -- there is nothing to relabel them to. They are reported, and those 8
dirs are already excluded from every current eval
(`embedder_matrix_9way.py::_EXCLUDED_COMBO_DIRS`).

Run:
    PYTHONPATH=src python tools/corpus_prep/relabel_index_resolution_ids.py
    PYTHONPATH=src python tools/corpus_prep/relabel_index_resolution_ids.py --apply
    ... --apply --no-backup      # skip the ~1.5GB of chunks.parquet backups
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "src")

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_loader  # noqa: E402
from rag_lab.io.artifact_store import seal  # noqa: E402

CORPUS = Path("academic_resolutions")
INDEX_ROOT = Path("data/index")

# Each entry: the id these files used to share -> (folder, filename patterns).
# Patterns are matched against *whitespace-normalized* filenames, and paths are
# resolved and sorted by the script rather than written out here, for two
# reasons that both bite: these filenames mix NBSP (\xa0) with ordinary spaces,
# so a literal copy is unreproducible; and where two files differ *only* in that
# whitespace, one normalized pattern correctly yields both of them. Ordering is
# left to `sorted()` over the real paths so it matches the loader exactly --
# hand-ordering long Thai titles that differ by an invisible character is how
# this would go wrong silently.
_ENG = "เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีกระทบกระเทือนโครงสร้าง) คณะวิศวกรรมศาสตร์"
_ENG_NO = "เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) คณะวิศวกรรมศาสตร์"
_SCI = "เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีไม่กระทบกระเทือนโครงสร้าง) คณะวิทยาศาสตร์"
_EXT = "เรื่อง ขออนุมัติขยายระยะเวลาการศึกษาของนักศึกษาระดับปริญญาตรีและบัณฑิตศึกษา"

SHARED: dict[str, tuple[str, list[str]]] = {
    f"2564/11/{_ENG}": (
        "2564/ครั้งที่ 11",
        [
            f"{_ENG}.md",
            "เรื่อง ขอความเห็นชอบการปรับปรุงหลักสูตร (กรณีกระทบกระเทือนโครงสร้าง) วิทยาลัยนวัตกรรมการผลิตขั้นสูง.md",
        ],
    ),
    f"2567/10/{_ENG}": ("2567/ครั้งที่ 10", [f"{_ENG} (2).md", f"{_ENG}.md"]),
    # one pattern, two files: they differ only by NBSP-vs-space
    f"2567/11/{_ENG}": ("2567/ครั้งที่ 11", [f"{_ENG}.md"]),
    f"2567/9/{_SCI}": (
        "2567/ครั้งที่ 9",
        [
            f"{_SCI}.md",
            "เรื่อง ขออนุมัติแต่งตั้งอาจารย์บัณฑิตประจำ และอาจารย์บัณฑิตพิเศษ คณะวิทยาศาสตร์.md",
        ],
    ),
    f"2568/4/{_EXT}": ("2568/ครั้งที่ 4", [f"{_EXT}.md"]),  # ditto
    (
        f"2567/1/{_ENG_NO} — หลักสูตรวิศวกรรมศาสตรบัณฑิต "
        "สาขาวิชาวิศวกรรมชีวการแพทย์ (หลักสูตรนานาชาติ)"
    ): ("2567/ครั้งที่ 1", [f"{_ENG_NO}__1.md", f"{_ENG_NO}__2.md"]),
}
EXPECTED_FILES_PER_ID = 2


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def source_paths() -> dict[str, list[Path]]:
    """old id -> the files that shared it, in the loader's order."""
    out: dict[str, list[Path]] = {}
    for old, (folder, patterns) in SHARED.items():
        wanted = {_norm(p) for p in patterns}
        paths = sorted(
            (p for p in (CORPUS / folder).glob("*.md") if _norm(p.name) in wanted),
            key=str,
        )
        if len(paths) != EXPECTED_FILES_PER_ID:
            raise SystemExit(
                f"{old[:70]}: matched {len(paths)} file(s) in {folder}, "
                f"expected {EXPECTED_FILES_PER_ID}:\n"
                + "\n".join(f"  {p.name!r}" for p in paths)
            )
        out[old] = paths
    return out


def resolve(paths_by_old: dict[str, list[Path]], loader_spec: dict) -> dict[str, list[tuple[str, str]]]:
    """old id -> [(current id, normalized text as this combo's loader sees it)].

    The text is loaded through the *combo's own* loader, not `plain`: the
    `normalized` loader rewrites text (Thai digits, `pythainlp.util.normalize`),
    so comparing its chunks against plain-loaded text would look like a total
    mismatch when nothing is wrong."""
    loader = build_loader(StrategySpec(**loader_spec))
    out: dict[str, list[tuple[str, str]]] = {}
    for old, paths in paths_by_old.items():
        out[old] = [
            (res.resolution_id, _norm(res.raw_text))
            for res in (loader.load(str(p)) for p in paths)
        ]
    return out


def blocks(indices: list[int]) -> list[range]:
    """Split positions into runs that each start where chunk_index returns to 0."""
    starts = [i for i, v in enumerate(indices) if v == 0] or [0]
    bounds = starts + [len(indices)]
    return [range(bounds[i], bounds[i + 1]) for i in range(len(starts))]


def datetime_ok(timestamp: str | None, newest_corpus: float) -> bool:
    """True when the index was built after the corpus's last edit, i.e. its
    stored chunk text should still match what the loader produces today."""
    if not timestamp:
        return False
    return datetime.fromisoformat(timestamp).timestamp() >= newest_corpus


def combos() -> list[Path]:
    return [
        d
        for parent in sorted(INDEX_ROOT.iterdir())
        if parent.is_dir()
        for d in sorted(parent.iterdir())
        if (d / "chunks.parquet").exists()
    ]


RESULTS_ROOT = Path("data/results")


def relabel_results(apply: bool) -> int:
    """Rewrite the same ids in persisted retrieval results.

    Metrics are computed from these files, not from the index, so relabelling
    the index alone leaves the affected gold queries still graded against ids
    nothing can match. Ranking is unaffected -- text and embeddings never
    changed, only labels -- so rewriting is exactly equivalent to re-running
    retrieval, at no GPU cost.

    Attribution is exact rather than heuristic: each relabelled index kept a
    `chunks.parquet.pre_relabel.bak` whose rows line up positionally with the
    new file, giving a real (old chunk_id, text) -> (new resolution_id, new
    chunk_id) map. Text is part of the key because the old chunk_ids themselves
    collided -- that collision is what is being repaired."""
    lookup: dict[tuple[str, str], tuple[str, str]] = {}
    # rows that legitimately kept their id: within a shared id the first source
    # file keeps it, so a result row pointing there is already correct and must
    # not be counted as a failed attribution
    keep: set[tuple[str, str]] = set()
    affected_old = set(SHARED)
    for d in combos():
        backup = d / "chunks.parquet.pre_relabel.bak"
        if not backup.exists():
            continue
        old = pq.read_table(backup, columns=["chunk_id", "resolution_id", "text"]).to_pydict()
        new = pq.read_table(d / "chunks.parquet", columns=["chunk_id", "resolution_id"]).to_pydict()
        for i, old_rid in enumerate(old["resolution_id"]):
            if old_rid != new["resolution_id"][i]:
                lookup[(old["chunk_id"][i], old["text"][i])] = (
                    new["resolution_id"][i],
                    new["chunk_id"][i],
                )
            elif old_rid in affected_old:
                keep.add((old["chunk_id"][i], old["text"][i]))
    print(f"\nresult relabel: {len(lookup)} distinct (chunk_id, text) keys from index backups")
    if not lookup:
        print("  nothing to do -- run the index relabel with backups first")
        return 0

    changed_files = changed_rows = unmatched = 0
    by_dir: Counter[str] = Counter()
    unmatched_dirs: Counter[str] = Counter()
    for rdir in sorted(p for p in RESULTS_ROOT.iterdir() if p.is_dir()):
        for f in sorted(rdir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict) or "results" not in data:
                continue
            touched = 0
            for row in data["results"]:
                key = (row.get("chunk_id"), row.get("text"))
                hit = lookup.get(key)
                if hit is None:
                    if row.get("resolution_id") in affected_old and key not in keep:
                        unmatched += 1
                        unmatched_dirs[rdir.name] += 1
                    continue
                row["resolution_id"], row["chunk_id"] = hit
                touched += 1
            if touched:
                changed_files += 1
                changed_rows += touched
                by_dir[rdir.name] += 1
                if apply:
                    f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, n in by_dir.most_common():
        print(f"  {name:38} {n:>5} file(s)")
    print(
        f"  {changed_files} file(s), {changed_rows} row(s) "
        f"{'rewritten' if apply else 'to rewrite (dry run)'}"
    )
    if unmatched:
        print(
            f"  {unmatched} row(s) carry an affected id but no exact (chunk_id, text) "
            "match -- pre-OCR-remediation result sets, left alone:"
        )
        for name, n in unmatched_dirs.most_common():
            print(f"    {name:38} {n:>5} row(s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument(
        "--results-only",
        action="store_true",
        help="skip the index phase; only rewrite persisted results (needs the backups)",
    )
    args = ap.parse_args()
    if args.results_only:
        return relabel_results(args.apply)

    paths_by_old = source_paths()
    plan = resolve(paths_by_old, {"type": "plain"})
    print("relabel plan (loader order within each shared id):")
    for old, news in plan.items():
        print(f"\n  OLD  {old[:96]}")
        for i, (new, _) in enumerate(news):
            same = " (unchanged)" if new == old else ""
            print(f"    #{i} -> {new[:92]}{same}")

    newest_corpus = max(p.stat().st_mtime for p in CORPUS.rglob("*.md"))
    by_loader: dict[str, dict[str, list[tuple[str, str]]]] = {}
    touched = skipped = rows_changed = unchecked = 0
    unresolved: Counter[str] = Counter()
    for d in combos():
        manifest_path = d / "manifest.json"
        man = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        loader_spec = (man.get("combo") or {}).get("loader") or {"type": "plain"}
        key = json.dumps(loader_spec, sort_keys=True, ensure_ascii=False)
        if key not in by_loader:
            by_loader[key] = resolve(paths_by_old, loader_spec)
        mapping = by_loader[key]
        # An index built before the corpus's last edit holds pre-OCR-remediation
        # text, which cannot match the corpus as it is now -- the block
        # structure is still valid there, so relabel but skip the text check
        # rather than refusing on evidence that cannot exist.
        text_check = bool(man) and datetime_ok(man.get("timestamp"), newest_corpus)
        table = pq.read_table(d / "chunks.parquet")
        cols = table.to_pydict()
        rids, cidx, cids, texts = (
            cols["resolution_id"],
            cols["chunk_index"],
            cols["chunk_id"],
            cols["text"],
        )
        present = [old for old in mapping if old in set(rids)]
        if not present:
            skipped += 1
            continue

        changes = 0
        problems = []
        for old in present:
            positions = [i for i, r in enumerate(rids) if r == old]
            groups = blocks([cidx[i] for i in positions])
            expected = mapping[old]
            if len(groups) != len(expected):
                problems.append(
                    f"{old[:60]}: found {len(groups)} source block(s), expected {len(expected)}"
                )
                continue
            for block, (new_id, file_text) in zip(groups, expected):
                rows = [positions[j] for j in block]
                if text_check:
                    hits = sum(1 for i in rows if _norm(texts[i]) in file_text)
                    if hits < len(rows) * 0.5:
                        problems.append(
                            f"{old[:50]}: block matched {hits}/{len(rows)} chunks against "
                            f"{new_id[-40:]} -- attribution looks wrong"
                        )
                        continue
                if new_id == old:
                    continue
                for i in rows:
                    rids[i] = new_id
                    cids[i] = f"{new_id}:{cidx[i]}"
                    changes += 1

        if problems:
            print(f"\n  !! {d.parent.name}/{d.name}")
            for p in problems:
                print(f"       {p}")
                unresolved[p.split(':')[0]] += 1
            continue

        touched += 1
        rows_changed += changes
        if not text_check:
            unchecked += 1
        note = "" if text_check else " [block-structure only: text predates the OCR remediation]"
        print(f"  {d.parent.name}/{d.name}: {changes} rows relabelled"
              + ("" if args.apply else " (dry run)") + note)
        if args.apply and changes:
            if not args.no_backup:
                shutil.copy2(d / "chunks.parquet", d / "chunks.parquet.pre_relabel.bak")
            new_table = pa.table({name: cols[name] for name in table.column_names})
            tmp = d / "chunks.parquet.tmp"
            pq.write_table(new_table, tmp)
            tmp.replace(d / "chunks.parquet")
            # Re-declare the four artifacts to be one build. This script is the
            # repo's only in-place writer of an index artifact, so without this
            # the directory would permanently disagree with the seal
            # ArtifactStore.save left, and a serving read would refuse it as
            # half-rebuilt. See ArtifactStore.seal.
            seal(d)
            manifest = d / "manifest.json"
            if manifest.exists():
                man = json.loads(manifest.read_text(encoding="utf-8"))
                man["relabeled"] = {
                    "reason": "resolution_id title repair 2026-07-30 (see ADR-0002 amendment)",
                    "rows": changes,
                }
                manifest.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\n{touched} combo(s) {'relabelled' if args.apply else 'to relabel'}, "
        f"{rows_changed} rows, {skipped} untouched"
    )
    if unresolved:
        print("unresolved (nothing written for these combos):")
        for k, v in unresolved.items():
            print(f"  x{v} {k}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
