"""Sweep for the whole class of bug that `resolution_id` collisions belonged to:
an invariant the pipeline *assumes* but never enforces, whose violation does not
crash anything -- it just makes a reported number quietly wrong.

Three such bugs have now been found by accident rather than by looking
(corpus-discovery contamination 2026-07-23, stale BM25/hybrid result cache
2026-07-29, resolution_id collisions 2026-07-30). They share a shape: a
silent-by-construction mismatch between two artifacts that were produced at
different times by different scripts. This script checks every such pairing
mechanically, so the next one is found by running a command instead of by
noticing an implausible number.

Checks, grouped by layer (C = corpus, I = index, E = eval):

    C1  resolution_id unique per file
    C2  no corpus file loads to empty text
    C3  meeting_manifest hygiene: dead `file` entries, duplicate keys, files
        absent from the manifest, one URL claimed by differently-titled files
    C4  a `*.md.dup` archive with no live counterpart (a file dropped by accident)
    C5  corpus file count vs master_list.csv
    I1  row alignment: embeddings rows == chunk rows == lexical rows
    I2  chunk_id unique within an index
    I3  every index resolution_id exists in the corpus (contamination), and how
        much of the corpus the index covers
    I4  embeddings sane: no NaN/inf, no all-zero rows (sampled), dim consistent
    I5  manifest n_resolutions/docset_hash vs the corpus as it is now
    I6  index built before the corpus it indexes was last modified
    E1  every gold relevant_resolution_id resolves against the corpus
    E2  no duplicate query text within a gold set
    E3  persisted results reference resolution_ids their index actually holds,
        separating ids left over from the corpus-discovery contamination bug
        (expected in a retired result set) from a genuine index mismatch
    E4  persisted results older than the index they were computed from

Read-only. Exits 1 if any check FAILs.

Run:
    PYTHONPATH=src python tools/eval/audit_pipeline_invariants.py
    PYTHONPATH=src python tools/eval/audit_pipeline_invariants.py --quick   # skip I4/E3
    PYTHONPATH=src python tools/eval/audit_pipeline_invariants.py --report out.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, "src")

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import build_loader  # noqa: E402
from rag_lab.loaders.common import _meeting_manifest, iter_corpus_files  # noqa: E402

CORPUS = Path("academic_resolutions")

# resolution_id is "<year>/<session>/<title>" (session may carry a trailing "s" for a
# special session); anything else is make_resolution_id's path fallback.
_WELL_FORMED_ID = re.compile(r"^\d{4}/\d+s?/")

# Where tools/archive_unused.py moved the off-repo archives; C4's subject matter
# lives here now, so the check has to follow it rather than report an empty corpus
# scan as a pass. Missing (different machine, drive not mounted) is tolerated --
# the check then says so instead of pretending.
ARCHIVE_ROOT = Path(r"D:/academic_resolutions (ข้อมูลดิบ + OCR)/_superseded_from_repo")
INDEX_ROOT = Path("data/index")
RESULTS_ROOT = Path("data/results")
GOLD = [
    Path("config/eval/gold_query_set_73det.yaml"),
    Path("config/eval/gold_query_set.yaml"),
]
# smoke/dev fixtures are deliberately tiny subsets -- coverage and count checks
# against the full corpus are meaningless for them
TOY_INDEXES = {"chunker_compare_smoke", "dev_smoke"}

# Result sets no current script reads (see tools/archive_unused.py::RETIRED_RESULTS).
# They were computed against earlier corpus states -- before the manifest rebuild
# renumbered titles, before the corpus-discovery contamination fix, before the
# resolution_id title repair -- so their ids legitimately do not resolve against
# today's indices. Reporting that as a FAIL would leave the gate permanently red
# and therefore useless; the finding that matters is "a *live* result set has
# drifted", so these are recorded separately as a warning instead.
RETIRED_RESULT_DIRS = {
    "gold_full_embedder_matrix", "silver_chunker_compare", "gold_chunker_compare",
    "gold_chunker_compare_73det", "gold_embedder_compare", "congen_sct_truncation_fix",
    "mode_b_routed",
}

findings: list[tuple[str, str, str]] = []  # (check, status, detail)


def record(check: str, ok: bool, detail: str, warn: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    findings.append((check, status, detail))
    print(f"[{status}] {check}: {detail}")


# --------------------------------------------------------------- corpus layer
def audit_corpus() -> tuple[set[str], dict[str, Path]]:
    loader = build_loader(StrategySpec(type="plain"))
    paths = sorted(iter_corpus_files(CORPUS), key=str)
    by_id: dict[str, list[Path]] = defaultdict(list)
    empty: list[Path] = []
    for p in paths:
        res = loader.load(str(p))
        by_id[res.resolution_id].append(p)
        if not res.raw_text.strip():
            empty.append(p)

    dupes = {k: v for k, v in by_id.items() if len(v) > 1}
    record(
        "C1 resolution_id unique",
        not dupes,
        f"{len(paths)} files -> {len(by_id)} ids"
        + (f"; {len(dupes)} shared" if dupes else ""),
    )
    record("C2 no empty document", not empty, f"{len(empty)} empty of {len(paths)}")

    # C3 manifest hygiene
    dead, unlisted, dup_keys = [], [], []
    url_titles: dict[str, set[str]] = defaultdict(set)
    url_files: dict[str, set[str]] = defaultdict(set)
    folders = {p.parent for p in paths}
    for folder in sorted(folders):
        mpath = folder / "meeting_manifest.json"
        if not mpath.exists():
            continue
        raw = json.loads(mpath.read_text(encoding="utf-8-sig"))
        names = [e.get("file") for e in raw if isinstance(e, dict)]
        dup_keys += [f"{folder}/{n}" for n, c in Counter(names).items() if c > 1]
        for entry in raw:
            if not isinstance(entry, dict) or not entry.get("file"):
                continue
            if not (folder / entry["file"]).exists():
                dead.append(f"{folder}/{entry['file']}")
            if entry.get("url"):
                url_titles[entry["url"]].add(entry.get("title") or "")
                url_files[entry["url"]].add(f"{folder}/{entry['file']}")
        listed = set(names)
        unlisted += [str(p) for p in folder.glob("*.md") if p.name not in listed]

    # one PDF legitimately backs every piece of a split bundle (ADR-0004), and
    # those pieces carry *different* titles -- so a shared URL is only suspicious
    # when the titles differ in a way that is not the "— <curriculum>" split
    # suffix, i.e. when the shared-URL files disagree on their base มติ title
    shared_url_conflicts = [
        u
        for u, titles in url_titles.items()
        if len({t.split(" — ")[0] for t in titles}) > 1
    ]
    record("C3a manifest entries point at real files", not dead, f"{len(dead)} dead")
    record("C3b no duplicate file keys", not dup_keys, f"{len(dup_keys)} duplicated")
    record(
        "C3c every corpus file listed in its manifest",
        not unlisted,
        f"{len(unlisted)} unlisted (title falls back to filename stem)",
        warn=True,
    )
    record(
        "C3d one source URL, one base title",
        not shared_url_conflicts,
        f"{len(shared_url_conflicts)} URLs claimed by differently-titled documents",
    )
    for u in shared_url_conflicts[:5]:
        print(f"        {u} -> {sorted(url_files[u])[:3]}")

    # C4 orphaned .md.dup. A split bundle's pre-split original is *supposed* to
    # have no live counterpart -- it was replaced by its `<stem>__N.md` pieces
    # (ADR-0004) -- so only an archive with neither a live file nor any split
    # piece is a file that fell out of the corpus.
    # The archives themselves now live off-repo (tools/archive_unused.py), so
    # scanning only CORPUS would turn this into a vacuous PASS -- 0 archives found
    # is not the same finding as 0 orphans. Look wherever they actually are, and
    # resolve each archive back to the corpus path it would have occupied.
    live = {str(p) for p in paths}
    orphans, n_archives, roots = [], 0, [(CORPUS, CORPUS)]
    if (dest := ARCHIVE_ROOT / CORPUS.name).is_dir():
        roots.append((dest, ARCHIVE_ROOT))
    for scan_root, rel_base in roots:
        for archive in scan_root.rglob("*.md.dup"):
            n_archives += 1
            here = CORPUS / archive.relative_to(rel_base).relative_to(CORPUS.name)
            stem = str(here)[: -len(".dup")]
            if stem in live:
                continue
            if list(here.parent.glob(f"{Path(stem).stem}__*.md")):
                continue
            orphans.append(str(archive))
    record(
        "C4 no orphaned .md.dup archive",
        not orphans,
        f"{len(orphans)} of {n_archives} archives have neither a live file nor split pieces"
        + ("" if len(roots) > 1 else " (in-repo only -- archive root not reachable)"),
        warn=True,
    )
    for o in orphans[:5]:
        print(f"        {o}")

    # C5 master_list.csv
    master = CORPUS / "master_list.csv"
    if master.exists():
        with master.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
        record(
            "C5 master_list.csv row count",
            True,
            f"{len(rows)} rows vs {len(paths)} corpus files (rows are meeting-level, "
            "not file-level -- informational)",
        )
    return set(by_id), {k: v[0] for k, v in by_id.items()}


# ---------------------------------------------------------------- index layer
def combos() -> list[Path]:
    return [
        d
        for parent in sorted(INDEX_ROOT.iterdir())
        if parent.is_dir()
        for d in sorted(parent.iterdir())
        if (d / "chunks.parquet").exists()
    ]


def audit_indexes(corpus_ids: set[str], quick: bool) -> dict[Path, set[str]]:
    newest_corpus = max(p.stat().st_mtime for p in iter_corpus_files(CORPUS))
    misaligned, dup_chunk_ids, contaminated, bad_vectors = [], [], [], []
    stale_vs_corpus, manifest_drift, coverage = [], [], []
    ids_by_combo: dict[Path, set[str]] = {}

    for d in combos():
        toy = d.parent.name in TOY_INDEXES
        table = pq.read_table(d / "chunks.parquet", columns=["chunk_id", "resolution_id"])
        chunk_ids = table.column("chunk_id").to_pylist()
        res_ids = table.column("resolution_id").to_pylist()
        n = len(chunk_ids)

        vecs = np.load(d / "embeddings.npy", mmap_mode="r")
        n_lex = None
        lex_path = d / "lexical.json"
        if lex_path.exists():
            n_lex = len(json.loads(lex_path.read_text(encoding="utf-8")))
        if vecs.shape[0] != n or (n_lex is not None and n_lex != n):
            misaligned.append(f"{d.parent.name}/{d.name}: chunks={n} vecs={vecs.shape[0]} lex={n_lex}")

        dupes = [c for c, k in Counter(chunk_ids).items() if k > 1]
        if dupes:
            dup_chunk_ids.append(f"{d.parent.name}/{d.name}: {len(dupes)} duplicated")

        unique_ids = set(res_ids)
        ids_by_combo[d] = unique_ids
        if not toy:
            unknown = unique_ids - corpus_ids
            if unknown:
                contaminated.append(
                    f"{d.parent.name}/{d.name}: {len(unknown)} ids not in corpus "
                    f"({sum(1 for r in res_ids if r in unknown)} chunks)"
                )
            coverage.append((f"{d.parent.name}/{d.name}", len(unique_ids), len(corpus_ids)))

        if not quick and n:
            idx = np.unique(np.linspace(0, n - 1, min(n, 2000)).astype(int))
            sample = np.asarray(vecs[idx], dtype=np.float64)
            norms = np.linalg.norm(sample, axis=1)
            if not np.isfinite(sample).all() or (norms == 0).any():
                bad_vectors.append(
                    f"{d.parent.name}/{d.name}: "
                    f"{int((~np.isfinite(sample)).any(axis=1).sum())} non-finite, "
                    f"{int((norms == 0).sum())} zero-norm (of {len(idx)} sampled)"
                )

        mpath = d / "manifest.json"
        if mpath.exists():
            man = json.loads(mpath.read_text(encoding="utf-8"))
            built = datetime.fromisoformat(man["timestamp"]).timestamp()
            if not toy and built < newest_corpus:
                stale_vs_corpus.append(
                    f"{d.parent.name}/{d.name}: built "
                    f"{datetime.fromtimestamp(built):%Y-%m-%d %H:%M} < corpus "
                    f"{datetime.fromtimestamp(newest_corpus):%Y-%m-%d %H:%M}"
                )
            if not toy and man.get("n_resolutions") not in (None, len(corpus_ids)):
                manifest_drift.append(
                    f"{d.parent.name}/{d.name}: manifest n_resolutions="
                    f"{man['n_resolutions']} vs corpus {len(corpus_ids)}"
                )

    record("I1 row alignment (chunks/vectors/lexical)", not misaligned, f"{len(misaligned)} misaligned of {len(ids_by_combo)}")
    for m in misaligned[:8]:
        print(f"        {m}")
    record("I2 chunk_id unique within index", not dup_chunk_ids, f"{len(dup_chunk_ids)} indexes with duplicates")
    for m in dup_chunk_ids[:8]:
        print(f"        {m}")
    record("I3a no chunks from outside the corpus", not contaminated, f"{len(contaminated)} indexes contaminated")
    for m in contaminated[:8]:
        print(f"        {m}")
    worst = sorted(coverage, key=lambda c: c[1])[:5]
    record(
        "I3b corpus coverage",
        all(c[1] == c[2] for c in coverage),
        "lowest: " + ", ".join(f"{name.split('/')[0]} {have}/{want}" for name, have, want in worst),
        warn=True,
    )
    if not quick:
        record("I4 embeddings finite and non-zero (sampled)", not bad_vectors, f"{len(bad_vectors)} indexes with bad vectors")
        for m in bad_vectors[:8]:
            print(f"        {m}")
    record("I5 manifest n_resolutions matches corpus", not manifest_drift, f"{len(manifest_drift)} drifted", warn=True)
    for m in manifest_drift[:8]:
        print(f"        {m}")
    record("I6 index newer than corpus", not stale_vs_corpus, f"{len(stale_vs_corpus)} indexes built before the corpus's last edit", warn=True)
    for m in stale_vs_corpus[:8]:
        print(f"        {m}")
    return ids_by_combo


# ----------------------------------------------------------------- eval layer
def audit_eval(corpus_ids: set[str], ids_by_combo: dict[Path, set[str]], quick: bool) -> None:
    all_queries: set[str] = set()
    for path in GOLD:
        entries = yaml.safe_load(path.read_text(encoding="utf-8"))
        refs = [r for e in entries for r in (e.get("relevant_resolution_ids") or [])]
        dangling = [r for r in refs if r not in corpus_ids]
        record(
            f"E1 gold ids resolve ({path.name})",
            not dangling,
            f"{len(refs)} refs, {len(dangling)} dangling",
        )
        for r in dangling[:5]:
            print(f"        {r[:110]}")
        queries = [e["query"] for e in entries]
        dupes = [q for q, c in Counter(queries).items() if c > 1]
        record(
            f"E2 no duplicate query ({path.name})",
            not dupes,
            f"{len(queries)} queries, {len(dupes)} duplicated",
            warn=True,
        )
        for q in dupes[:5]:
            print(f"        {q[:100]}")
        all_queries |= set(queries)

    # index mtime per combo id, for E3/E4
    # toy roots excluded: a freshly rebuilt smoke fixture must not make
    # full-corpus results look stale (same id-ambiguity caveat as above)
    built_at = {
        d.name: json.loads((d / "manifest.json").read_text(encoding="utf-8"))["timestamp"]
        for d in ids_by_combo
        if (d / "manifest.json").exists() and d.parent.name not in TOY_INDEXES
    }
    # A combo id hashes loader+chunker+embedder but NOT the corpus (see
    # combos.py::BuildCombo.id), so the same name exists under several index
    # roots -- a 12-file smoke subset and the 2,853-file corpus are
    # indistinguishable by id alone, and a result file records only the id.
    # Union the ids across roots: attributing a result to the wrong root is what
    # produces a false "unknown id" here, and there is no signal in the data to
    # do better. Reported as its own finding rather than papered over.
    ids_by_name: dict[str, set[str]] = defaultdict(set)
    roots_by_name: dict[str, set[str]] = defaultdict(set)
    for d, v in ids_by_combo.items():
        ids_by_name[d.name] |= v
        roots_by_name[d.name].add(d.parent.name)
    ambiguous = {n: r for n, r in roots_by_name.items() if len(r) > 1}
    record(
        "E0 combo id identifies its index unambiguously",
        not ambiguous,
        f"{len(ambiguous)} combo ids exist under >1 index root "
        "(BuildCombo.id omits the corpus, so results cannot be attributed to one index)",
    )
    for n, r in list(ambiguous.items())[:5]:
        print(f"        {n} -> {sorted(r)}")

    result_dirs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir()]
    stale_dirs, unknown_ids, unknown_queries = [], [], []
    stale_contaminated: list[str] = []
    retired_drift: list[str] = []
    for rdir in sorted(result_dirs):
        files = sorted(rdir.glob("*.json"))
        if not files:
            continue
        newest_result = max(f.stat().st_mtime for f in files)
        combo_names = set()
        checked = 0
        for f in files if not quick else files[:40]:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "results" not in data:
                continue
            checked += 1
            combo = (data.get("combination_id") or "").split("__")
            # combination_id is <loader>__<chunker>__<embedder>__<hash>[__<retriever>...]
            name = "__".join(combo[:4]) if len(combo) >= 4 else ""
            combo_names.add(name)
            known = ids_by_name.get(name) or None
            if known is not None:
                bad = {r["resolution_id"] for r in data["results"] if r.get("resolution_id") not in known}
                # A well-formed id is '<year>/<session>/<title>'; make_resolution_id falls
                # back to the file path when a walk picked up something that is not a
                # resolution at all. Those are the corpus-discovery contamination bug's
                # artifacts, so a *retired* result set citing them is expected, not a
                # mismatch -- it was computed before the fix. Classify rather than
                # conflate: until the 8 superseded combos were deleted, these files were
                # excused only because those indices still held the bogus ids.
                contaminated = {b for b in bad if not _WELL_FORMED_ID.match(b or "")}
                if contaminated:
                    stale_contaminated.append(
                        f"{rdir.name}/{f.name}: {len(contaminated)} pre-fix contamination id(s)")
                drifted = bad - contaminated
                if drifted:
                    where = retired_drift if rdir.name in RETIRED_RESULT_DIRS else unknown_ids
                    where.append(f"{rdir.name}/{f.name}: {len(drifted)} ids not in {name}")
            if data.get("query") and data["query"] not in all_queries:
                unknown_queries.append(f"{rdir.name}: {data['query'][:60]}")
        # E4: results older than the index they name
        for name in combo_names:
            ts = built_at.get(name)
            if rdir.name in RETIRED_RESULT_DIRS:
                continue
            if ts and datetime.fromisoformat(ts).timestamp() > newest_result:
                stale_dirs.append(
                    f"{rdir.name}: results {datetime.fromtimestamp(newest_result):%Y-%m-%d %H:%M}"
                    f" < index {name[:38]} {datetime.fromisoformat(ts).astimezone():%Y-%m-%d %H:%M}"
                )
                break

    record("E3a results reference ids their index holds", not unknown_ids,
           f"{len(unknown_ids)} result files with unknown ids")
    record("E3d retired result sets name ids no index holds", not retired_drift,
           f"{len(retired_drift)} result files in {sorted(RETIRED_RESULT_DIRS)!r} carry "
           f"titles from an earlier corpus state -- retired, not read by any current script",
           warn=True)
    record("E3c retired result sets cite pre-fix contamination ids", not stale_contaminated,
           f"{len(stale_contaminated)} result files cite an id from the corpus-discovery "
           f"contamination bug -- expected for result sets computed before its fix; do not reuse them",
           warn=True)
    for m in unknown_ids[:8]:
        print(f"        {m}")
    record("E3b results answer a known gold query", not unknown_queries, f"{len(set(unknown_queries))} unrecognized queries", warn=True)
    for m in sorted(set(unknown_queries))[:5]:
        print(f"        {m}")
    record("E4 results newer than their index", not stale_dirs, f"{len(stale_dirs)} result sets computed before their index was rebuilt")
    for m in stale_dirs[:10]:
        print(f"        {m}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip embedding sampling; cap result files per dir")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    started = datetime.now(timezone.utc)
    print("=== corpus ===")
    corpus_ids, _ = audit_corpus()
    print("\n=== indexes ===")
    ids_by_combo = audit_indexes(corpus_ids, args.quick)
    print("\n=== eval ===")
    audit_eval(corpus_ids, ids_by_combo, args.quick)

    fails = [f for f in findings if f[1] == "FAIL"]
    warns = [f for f in findings if f[1] == "WARN"]
    print(f"\n{len(findings)} checks: {len(findings) - len(fails) - len(warns)} pass, {len(warns)} warn, {len(fails)} fail")

    if args.report:
        lines = [
            "# Pipeline invariant audit",
            "",
            f"Run {started:%Y-%m-%d %H:%M} UTC. "
            f"{len(findings) - len(fails) - len(warns)} pass / {len(warns)} warn / {len(fails)} fail.",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
        lines += [f"| {c} | {s} | {d} |" for c, s, d in findings]
        args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.report}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
