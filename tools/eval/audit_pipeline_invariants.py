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
    C4  a `*.md.dup` archive with no live counterpart (a file dropped by accident),
        classifying the legitimate reasons one has none: a tail fragment of a
        wrapped title, a rename, a truncated title, a title naming another item
    C5  corpus file count vs master_list.csv
    I1  row alignment: embeddings rows == chunk rows == lexical rows
    I2  chunk_id unique within an index
    I3  every index resolution_id exists in the corpus (contamination), and how
        much of the corpus the index covers
    I4  embeddings sane: no NaN/inf, no all-zero rows (sampled), dim consistent
    I5  manifest n_resolutions/docset_hash vs the corpus as it is now
    I6  index built before the corpus it indexes was last modified
    E0  every persisted result attributes to exactly one built index -- by its
        recorded `index_dir`, by a combo id unique across index roots, or by
        elimination on the resolution_ids it cites (BuildCombo.id omits the
        corpus, so the id alone names a combo, not an index)
    E1  every gold relevant_resolution_id resolves against the corpus
    E2  no duplicate query text within a gold set
    E3  persisted results reference resolution_ids the index E0 attributed them
        to actually holds, separating ids left over from the corpus-discovery
        contamination bug (expected in a retired result set) from a genuine
        index mismatch
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
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
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


_PAGE1_HEADING = re.compile(r"มติคณะกรรมการสภาวิชาการ.{0,80}?(เรื่อง\s.{10,300})", re.S)


def _page1_heading(text: str) -> str:
    """The 'เรื่อง ...' subject line the document states about itself.

    Preferred over the manifest title when asking what a file *is*: the manifest
    can be wrong (that is what the ADR-0002 amendment repaired), the body cannot.
    Falls back to the head of the text when the heading does not parse.
    """
    m = _PAGE1_HEADING.search(_flat(text[:3000]))
    return _flat(m.group(1))[:200] if m else _flat(text[:200])


def _flat(s: str) -> str:
    """Collapse whitespace and NBSP so Thai filenames/titles compare equal.

    Corpus filenames mix U+00A0 with ordinary spaces, and OCR inserts line breaks
    mid-title, so a naive == or `in` misses matches that are the same string.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s.replace(" ", " "))).strip()
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

# Written by the Streamlit UI (query_service only ever saves to these, never reads
# back), so they hold whatever a human typed while clicking around. E3b asks whether
# an *eval* result set answers a query some gold set defines; an interactive dir
# failing that is the design, not drift, so it would be a permanent expected warning.
UI_RESULT_DIRS = {"mode_b", "mode_b_routed"}

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
    # Two further reasons an archive legitimately has no same-named live file,
    # both established by reviewing all 24 orphans on 2026-07-30. Encoded as rules
    # rather than as a list of 24 reviewed paths, so the check keeps working as the
    # corpus changes instead of going stale the moment a file is renamed:
    #   (a) the archive's *filename* is a tail fragment of a longer title. Before
    #       the manifest rebuild, a wrapped title produced one file per line, so
    #       the fragment is a substring of some live file's full manifest title
    #       (21 of 24; e.g. "และมาตรฐานคุณวุฒิสาขา" + "วิชาเภสัชศาสตร์ ระดับ" +
    #       "ปริญญาตรี พ.ศ. ๒๕๖๗" were three files for one agenda item).
    #   (b) the document is live under a different name -- renamed by a title
    #       repair -- which only content can show (3 of 24, ratio 0.90/0.99/1.00).
    #       Compare against the folder only, and note this evidence is one-way:
    #       a *low* ratio proves nothing, because the archive predates the re-OCR.
    live = {str(p) for p in paths}
    orphans, n_archives, roots = [], 0, [(CORPUS, CORPUS)]
    fragment_of_title, renamed = 0, 0
    mistitled: list[str] = []
    truncated_title: list[str] = []
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
            manifest = _meeting_manifest(str(here.parent))
            frag = _flat(Path(stem).stem)
            siblings = [f for f in sorted(here.parent.glob("*.md")) if str(f) in live]
            if frag and any(
                frag in _flat((manifest.get(f.name) or {}).get("title") or "")
                for f in siblings
            ):
                fragment_of_title += 1
                continue
            # Same document under a different name. Compare page-1 headings rather
            # than whole files: the archive predates the re-OCR, so full-text
            # similarity decays (one pair sits at 0.638 while its headings are
            # identical), but the heading is short and stable.
            ahead = _page1_heading(archive.read_text(encoding="utf-8"))
            twin = next(
                (f for f in siblings
                 if SequenceMatcher(
                     None, ahead, _page1_heading(f.read_text(encoding="utf-8"))
                 ).ratio() >= 0.90),
                None,
            )
            if twin is not None:
                # Does the live twin's manifest title actually describe this
                # document? When it does not, nothing fell out of the corpus, but
                # the surviving file is filed under the wrong agenda item -- the
                # same class of defect as the ADR-0002 title repairs. Surface it
                # here rather than absorbing it into the rename count.
                twin_title = _flat((manifest.get(twin.name) or {}).get("title") or "")
                twin_head = _page1_heading(twin.read_text(encoding="utf-8"))
                if SequenceMatcher(None, frag, twin_title).ratio() >= 0.60:
                    renamed += 1
                elif twin_title and SequenceMatcher(
                    None, twin_title, twin_head[: len(twin_title)]
                ).ratio() >= 0.85:
                    # The title is the head of the document's own subject line, cut
                    # short -- incomplete, not wrong. It still becomes a truncated
                    # resolution_id, so it is worth counting separately.
                    truncated_title.append(f"{here.parent}/{twin.name}")
                else:
                    mistitled.append(
                        f"{here.parent}/{twin.name}\n            filed as {twin_title[:60]!r}"
                        f"\n            body says {twin_head[:60]!r}"
                    )
                continue
            orphans.append(str(archive))
    record(
        "C4 no orphaned .md.dup archive",
        not orphans,
        f"{len(orphans)} of {n_archives} archives are unaccounted for "
        f"({fragment_of_title} are tail fragments of a live file's title, {renamed} are "
        f"live under a repaired name, {len(truncated_title)} under a truncated title, "
        f"{len(mistitled)} under a title naming a different agenda item)"
        + ("" if len(roots) > 1 else "; in-repo only -- archive root not reachable"),
        warn=True,
    )
    for o in orphans[:5]:
        print(f"        {o}")
    for m in truncated_title:
        print(f"        TRUNCATED TITLE {m}")
    for m in mistitled:
        print(f"        WRONG TITLE {m}")

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
    # Manifests count as corpus edits, not just the .md files. A `resolution_id`
    # is `<year>/<session>/<title>` and the title comes from meeting_manifest.json
    # (ADR-0003), so repairing a manifest title moves an id without touching a
    # single .md -- exactly what the 2026-08-08 mispairing repair did. Reading
    # only *.md here made that edit invisible to I6, which would then report an
    # index as current while it still held the pre-repair ids: a stale index
    # passing the check written to catch stale indexes.
    newest_corpus = max(
        p.stat().st_mtime
        for p in [*iter_corpus_files(CORPUS), *CORPUS.rglob("meeting_manifest.json")]
    )
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
            # A relabel brings an index current with respect to a title-only
            # corpus edit without rebuilding it: chunk text is unchanged and
            # embeddings are a function of text alone, so only the id columns
            # move. Without counting it, adding manifests to `newest_corpus`
            # above would leave every index permanently warned after any title
            # repair -- a check that is always red is one nobody reads.
            for marker in ("relabeled_mispairings", "relabeled"):
                at = (man.get(marker) or {}).get("at")
                if at:
                    built = max(built, datetime.fromisoformat(at).timestamp())
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
class IndexAttributor:
    """Which built index produced a persisted result (E0).

    `BuildCombo.id` hashes loader+chunker+embedder but NOT the corpus (see
    combos.py), so one name is the directory name of several different indices --
    a 10-file smoke fixture and the 2,854-file corpus are indistinguishable by
    name. A result file naming only that combo therefore does not say which index
    produced it, and that is exactly why the 2026-07-29 stale-cache incident was
    invisible in the data.

    Renaming the indices was the wrong fix: the id *is* the directory name, so
    hashing the corpus into it would rename every index on every corpus edit and
    orphan ~24k persisted results. Attribution is the fix instead -- strongest
    evidence first:

      1. `recorded`     -- the result records `index_dir` outright (written since
                           2026-08-09; schema.RetrievalResult)
      2. `unique name`  -- the combo id exists under exactly one index root
      3. `elimination`  -- exactly one candidate index holds every resolution_id
                           the result cites. Sound because the result *did* come
                           from one of the candidates, so a candidate missing an
                           id it cites is ruled out; and this is the only rule of
                           the three that can fail to decide.

    Two further outcomes are classified rather than folded into a verdict, because
    neither is an ambiguity: `no built index` (nothing by that name exists -- its
    index was deleted, e.g. the 8 superseded combos) and `no candidate fits` (the
    result cites ids none of them holds, which is drift and belongs to E3a).
    Only `ambiguous` -- rule 3 leaving more than one survivor -- is a result that
    genuinely cannot be attributed.

    Note rule 1 does not always apply even to results written after 2026-08-09:
    `router.rrf_merge` leaves `index_dir` unset on purpose, because a merged
    ranking spans several indices and there is no single index to attribute it
    to. Rules 2/3 read that absence correctly -- they attribute by the ids a
    result cites rather than by a field claiming one index answered it.
    """

    def __init__(self, ids_by_combo: dict[Path, set[str]]) -> None:
        self.ids_by_combo = ids_by_combo
        self.dirs_by_name: dict[str, list[Path]] = defaultdict(list)
        self.ids_by_name: dict[str, set[str]] = defaultdict(set)
        for d, v in ids_by_combo.items():
            self.dirs_by_name[d.name].append(d)
            self.ids_by_name[d.name] |= v
        # A result records whatever path its writer was handed, which may be
        # absolute while the scan here is relative -- so match on the resolved
        # path, or every recorded provenance silently degrades to rule 2/3 and
        # rule 1 quietly stops being exercised.
        self.by_resolved = {d.resolve(): d for d in ids_by_combo}

    @property
    def ambiguous_names(self) -> dict[str, list[Path]]:
        return {n: v for n, v in self.dirs_by_name.items() if len(v) > 1}

    def attribute(self, data: dict, name: str) -> tuple[Path | None, str]:
        """The one index that produced this result, and how we know."""
        recorded = data.get("index_dir")
        if recorded and (known := self.by_resolved.get(Path(recorded).resolve())):
            return known, "recorded"
        candidates = self.dirs_by_name.get(name) or []
        if not candidates:
            return None, "no built index"
        if len(candidates) == 1:
            return candidates[0], "unique name"
        cited = {r.get("resolution_id") for r in data.get("results") or []}
        fits = [d for d in candidates if cited <= self.ids_by_combo[d]]
        if len(fits) == 1:
            return fits[0], "elimination"
        return (None, "ambiguous") if fits else (None, "no candidate fits")


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

    # index mtime per index dir, for E4
    built_at = {
        d: json.loads((d / "manifest.json").read_text(encoding="utf-8"))["timestamp"]
        for d in ids_by_combo
        if (d / "manifest.json").exists()
    }
    attributor = IndexAttributor(ids_by_combo)
    ids_by_name = attributor.ids_by_name
    result_dirs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir()]
    attribution = Counter()
    unattributable: list[str] = []
    stale_dirs, unknown_ids, unknown_queries = [], [], []
    stale_contaminated: list[str] = []
    retired_drift: list[str] = []
    # Denominators matter here: these three checks all report a *count of bad
    # files*, so 0 is ambiguous between 'examined and clean' and 'nothing left
    # to examine' -- which is exactly what happened when the retired result sets
    # were archived off-repo and E3c/E3d silently went to a vacuous 0.
    n_examined = 0
    n_examined_retired = 0
    for rdir in sorted(result_dirs):
        files = sorted(rdir.glob("*.json"))
        if not files:
            continue
        newest_result = max(f.stat().st_mtime for f in files)
        index_dirs: set[Path] = set()
        checked = 0
        for f in files if not quick else files[:40]:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or "results" not in data:
                continue
            checked += 1
            n_examined += 1
            if rdir.name in RETIRED_RESULT_DIRS:
                n_examined_retired += 1
            combo = (data.get("combination_id") or "").split("__")
            # combination_id is <loader>__<chunker>__<embedder>__<hash>[__<retriever>...]
            name = "__".join(combo[:4]) if len(combo) >= 4 else ""
            index_dir, how = attributor.attribute(data, name)
            attribution[how] += 1
            if how == "ambiguous":
                unattributable.append(
                    f"{rdir.name}/{f.name}: {name} fits >1 of "
                    f"{sorted(d.parent.name for d in attributor.dirs_by_name[name])}"
                )
            if index_dir is not None:
                index_dirs.add(index_dir)
            # Check the ids against the *attributed* index, not the union over every
            # root sharing the name: the union is a superset, so it would accept an
            # id only the smoke fixture holds. Fall back to the union only when
            # attribution could not decide, which never makes this weaker than the
            # pre-2026-08-09 behaviour.
            known = (
                ids_by_combo[index_dir] if index_dir is not None
                else (ids_by_name.get(name) or None)
            )
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
            if (data.get("query") and data["query"] not in all_queries
                    and rdir.name not in UI_RESULT_DIRS):
                unknown_queries.append(f"{rdir.name}: {data['query'][:60]}")
        # E4: results older than the index that produced them. Keyed on the
        # attributed index dir rather than the combo name, so a rebuilt smoke
        # fixture can no longer make full-corpus results look stale by sharing a
        # name -- toy roots are skipped because they are subsets, not because the
        # name was ambiguous.
        for d in index_dirs:
            ts = built_at.get(d)
            if rdir.name in RETIRED_RESULT_DIRS or d.parent.name in TOY_INDEXES:
                continue
            if ts and datetime.fromisoformat(ts).timestamp() > newest_result:
                stale_dirs.append(
                    f"{rdir.name}: results {datetime.fromtimestamp(newest_result):%Y-%m-%d %H:%M}"
                    f" < index {d.parent.name}/{d.name[:38]} "
                    f"{datetime.fromisoformat(ts).astimezone():%Y-%m-%d %H:%M}"
                )
                break

    record(
        "E0 every result attributes to exactly one index",
        not unattributable,
        f"{len(unattributable)} of {n_examined} result files cannot be attributed to one "
        f"built index ({len(attributor.ambiguous_names)} of {len(attributor.dirs_by_name)} "
        f"combo ids exist under >1 index root, since BuildCombo.id omits the corpus; "
        f"attributed "
        + ", ".join(f"{v} by {k}" for k, v in sorted(attribution.items()))
        + ")",
    )
    for n, v in list(attributor.ambiguous_names.items())[:5]:
        print(f"        {n} -> {sorted(d.parent.name for d in v)}")
    for m in unattributable[:8]:
        print(f"        {m}")

    record("E3a results reference ids their index holds", not unknown_ids,
           f"{len(unknown_ids)} of {n_examined - n_examined_retired} live result files "
           f"reference an id their index does not hold")
    record("E3d retired result sets name ids no index holds", not retired_drift,
           f"{len(retired_drift)} of {n_examined_retired} retired result files carry titles "
           f"from an earlier corpus state (retired sets, read by no current script; 0 of 0 "
           f"means they have been archived off-repo, not that they were checked)",
           warn=True)
    record("E3c retired result sets cite pre-fix contamination ids", not stale_contaminated,
           f"{len(stale_contaminated)} of {n_examined} result files cite an id from the "
           f"corpus-discovery contamination bug -- expected for sets computed before its fix; "
           f"do not reuse them",
           warn=True)
    for m in unknown_ids[:8]:
        print(f"        {m}")
    record("E3b results answer a known gold query", not unknown_queries, f"{len(set(unknown_queries))} unrecognized queries across "
           f"{n_examined} result files ({sorted(UI_RESULT_DIRS)!r} excluded: "
           f"interactive UI queries are not gold by design)", warn=True)
    for m in sorted(set(unknown_queries))[:5]:
        print(f"        {m}")
    record("E4 results newer than their index", not stale_dirs, f"{len(stale_dirs)} result sets computed before their index was rebuilt")
    for m in stale_dirs[:10]:
        print(f"        {m}")


def audit_generation() -> None:
    """G1: no published RQ4 answer may have been generated from a truncated prompt.

    Added 2026-08-10 because `audit_doc_claims.py`'s D4 -- the only thing that had
    been flagging the 80 truncated cells -- turned out to be clearable *without
    fixing them*. D4 compares a report's mtime against its generator's, so merely
    re-running `rq4_score.py` (seconds, no generation) flipped it from FAIL to PASS
    while every truncated answer sat untouched on disk. That is the
    [[feedback_cleanup_can_break_an_audit]] shape one layer up: a real finding
    silently discharged by an unrelated action.

    So this check reads the *artifacts* instead of their timestamps, and nothing but
    regeneration can clear it. `rq4_generate.py` records `num_ctx` and
    `prompt_eval_count` in every answer it writes; ollama truncates an over-long
    prompt to exactly `num_ctx // 2 + 2` tokens (docs/rq4-prompt-truncation.md), so
    that equality is an exact signature. Answers predating the fix carry neither
    field -- they are counted separately as *unverifiable*, not as clean, because
    "no evidence of truncation" and "evidence of no truncation" are different
    ([[feedback_undefined_is_not_zero]]).
    """
    root = Path("data/rq4/answers")
    if not root.is_dir():
        record("G1 no RQ4 answer generated from a truncated prompt", True,
               "0 of 0 answers on disk", warn=True)
        return

    truncated, unverifiable, checked = [], 0, 0
    for path in sorted(root.glob("*/*/q*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        num_ctx, n_prompt = rec.get("num_ctx"), rec.get("prompt_eval_count")
        if num_ctx is None or n_prompt is None:
            unverifiable += 1
            continue
        checked += 1
        if n_prompt == num_ctx // 2 + 2:
            truncated.append(f"{path.relative_to(root)} ({n_prompt} tok at num_ctx={num_ctx})")

    record("G1a no RQ4 answer generated from a truncated prompt", not truncated,
           f"{len(truncated)} truncated of {checked} answers carrying num_ctx")
    for m in truncated[:10]:
        print(f"        {m}")
    # Deliberately a WARN and deliberately not silent: docs/rq4-prompt-truncation.md
    # reconstructed 80 of these prompt by prompt and found them truncated, so this is
    # outstanding work, not merely unmeasured. It clears only by regenerating them --
    # which is the point, since the timestamp check that used to carry this finding
    # could be cleared by re-running the scorer.
    record("G1b every RQ4 answer records the context it was generated at", unverifiable == 0,
           f"{unverifiable} of {checked + unverifiable} answers predate the num_ctx fix "
           f"and cannot be verified either way; 80 of them are KNOWN truncated "
           f"(docs/rq4-prompt-truncation.md) and are still to be regenerated",
           warn=True)


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
    print("\n=== generation ===")
    audit_generation()

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
