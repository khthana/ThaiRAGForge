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
    G1  no published RQ4 answer was generated from a truncated prompt: G1a reads
        the `num_ctx`/`prompt_eval_count` an answer records, G1b re-derives the
        same claim for pre-fix answers from provable evidence about their prompt,
        and G1c reports how many answers neither can reach

Lessons this file's own checks were built from. Read these before editing a
check, because each one is a way a check went quietly wrong here at least once.

  * THE DERIVED-COPY INVENTORY (swept 2026-08-23, after a Qdrant collection was
    found ANSWERING from a previous build). The class is: an artifact computed
    from another artifact, read by something, where drift is silent. Enumerated
    here so a future copy is added to a list rather than discovered:

      - Qdrant collections (a copy of an Index's rows) -- GUARDED at query time
        by QdrantHybridRetriever._verify: row count plus a sample compared by
        identity. Before 2026-08-23 a rebuild without a re-ingest served the
        previous build's payload with no error.
      - data/qdrant/<collection>/vocab.json (term -> sparse id, from the same
        ingest) -- GUARDED BY ORDERING, not by a check. Written BEFORE the
        upsert since 2026-08-23, so a run that dies in between leaves a current
        vocabulary beside a stale collection, which _verify already refuses.
        Written after, the same crash left the undetectable pairing.
      - academic_resolutions/entity_tags/*_by_file.json (corpus x matchers) --
        T1. Read by build_gold_candidates.py, the QRELS generator, and by VALUE
        for people/courses/faculties. Not read by build_relation_graph.py, which
        recomputes from the loaders and says so in its own report; not read by
        entity_loader, which calls the same matchers at build time -- so a built
        index is current with the matchers while these copies need not be.
      - academic_resolutions/entity_tags/gold_candidates.json -- DELIBERATELY
        NOT regenerated: it is dated 2026-07-25 and is the provenance of the
        published gold set. Re-deriving it is a measurement, never a repair.
      - data/index/<combo>/* -- the seal (_complete.json, written last) plus
        index_cache._settle; I6/I7 watch staleness and unsealed dirs.
      - data/results/** -- E0 (a result names the index that produced it) and E4
        (results newer than their index).
      - docs/*.md generated from a script -- D1a (report older than generator)
        and D4 (an input changed after the report) in audit_doc_claims.py.

    The shape to look for when adding one: does the CONSUMER read the copy's own
    bytes? A collection answers from its stored payload and the qrels generator
    reads cached values, so both can be wrong without erroring. A copy nothing
    reads by value is a reporting concern, not a correctness one (T1b).

  * A CHECK WHOSE SUBJECT MATTER MOVES BECOMES A VACUOUS PASS. C4 went 24 -> 0
    the moment the `.md.dup` archives were moved off-repo, so it now follows them
    to ARCHIVE_ROOT and prints "0 of 239" rather than "0". Because 0 is ambiguous
    between "examined and clean" and "nothing left to examine", the E3 checks
    print their denominator too.
  * DON'T LET A KNOWN-RETIRED ARTIFACT KEEP THE GATE RED. Deleting the 8
    superseded combos removed the only indices still holding pre-contamination-fix
    ids, which made E3a jump 7 -> 3,106 for result sets nothing reads. Those are
    classified separately (E3c, E3d, RETIRED_RESULT_DIRS) so a FAIL still means a
    *live* result set has drifted.
  * ATTRIBUTABILITY, NOT RENAMING. `BuildCombo.id` hashes loader+chunker+embedder
    but NOT the corpus, so a smoke-subset combo and a full-corpus combo share an
    id -- which is why the 2026-07-29 stale-cache incident was invisible (12 of 43
    combo ids exist under more than one index root). Hashing the corpus into the
    id is the obvious fix and is DISQUALIFIED: the id *is* the on-disk directory
    name and the prefix of every result filename, so it would rename 55 index
    dirs on every corpus edit, orphan ~24k results, and break combo names
    hardcoded in eval scripts. E0 instead makes each result NAME its source. Its
    third rule, elimination, is sound because the result *did* come from one of
    the candidate indices, so exactly one candidate holding every resolution_id
    it cites identifies it; only >1 survivor is a FAIL, while "no built index"
    and "no candidate fits" are classified, not failed. Elimination alone
    resolves 100% of the live ambiguity, which is why no backfill was needed.
    `provenance` is deliberately kept OUT of `meta` (which is what `save` writes
    back, so a load-time field must not round-trip) and the new result fields are
    optional, because ~24k legacy results must keep validating.
  * A STALENESS CHECK IS ONLY A PROXY; WHEN IT CARRIES A REAL FINDING, ADD THE
    CHECK THAT READS THE THING ITSELF. The G1 family exists because a D4 finding
    about truncated RQ4 answers was discharged by an unrelated re-run of a cheap
    generator while all 80 truncated answers sat untouched on disk. G1 reads the
    artifacts instead: `prompt_eval_count == num_ctx // 2 + 2` is an exact
    truncation signature.
  * "UNVERIFIABLE" IS A BUCKET TO SHRINK, NOT A VERDICT. G1b's predecessor called
    all 1,509 pre-fix answers unverifiable and overstated the hole by 2x; two
    provable sources already on disk (the UTF-8-byte upper bound, and cached
    probes actually sent at the old num_ctx) decided most of them, which priced
    the remainder at one probe per prompt instead of a full regeneration.
    SCREEN_CHARS_PER_TOKEN = 0.95 would have cleared the whole remainder at a
    stroke and is deliberately NOT evidence: it is an observed minimum, and an
    observed extreme is not a bound. G1c counts what neither reaches --
    *unmeasured*, never *suspected* -- with its denominator printed, and the warn
    stays wired rather than deleted so a new pre-fix answer or a deleted probe
    cache reopens it.
  * A CHECK CAN BE BLIND TO A WHOLE CLASS OF INPUT. I6 derived "the corpus's last
    edit" from `*.md` mtimes alone, but a resolution_id is built from the manifest
    title (ADR-0003), so the 2026-08-08 title repair moved 4 ids without touching
    a single `.md` and I6 would have called all 41 affected indices current. It
    now reads `meeting_manifest.json` mtimes too, and counts a recorded relabel
    (`relabeled_mispairings.at`) as bringing an index current without a rebuild --
    without that second half it would sit permanently red after any title repair,
    and an always-red check is one nobody reads.
  * AN UNSEALED INDEX IS A REPORTED GAP, NOT A PASS. I7 watches the writer-side
    seal `_complete.json`; an index built before that convention gets the older,
    narrower staleness guarantee and must say so rather than passing silently.
  * A CHECK THAT CANNOT FAIL ON LIVE DATA IS UNEXERCISED. G1b reports 0 here, so
    its failing branch never runs against the corpus; `tests/tools/` pins all
    three G1 outcomes plus both cache rules, and pins all six E0 outcomes, or
    five of them would be exactly the vacuous PASS the first lesson warns about.

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

# G1b rebuilds RQ4 prompts to bound their token count. `rq4_generate` pulls in the
# ollama client, so the import is guarded: this audit is read-only and makes no
# request, and it must stay runnable where that client is absent. Missing => every
# pre-fix answer falls to G1c as *unmeasured*, which is the honest degradation
# rather than a silent pass.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from rq4_generate import (  # noqa: E402
        _CONTEXTS as _RQ4_CONTEXTS, build_prompt as _rq4_build_prompt,
        token_upper_bound as _rq4_token_upper_bound, truncated_to as _rq4_truncated_to,
    )
except Exception:                                       # pragma: no cover
    _RQ4_CONTEXTS = None

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


# ------------------------------------------------------- derived-copy layer
#: Files sampled per tag file. All four matchers over one corpus file cost
#: ~0.4 s, so a full 2,854-file pass is ~18 min -- far too slow for a sweep
#: meant to run before trusting anything. The sample size is stated in the
#: detail rather than hidden, because a CLEAN result on a small sample is weak
#: evidence while a DIRTY one is conclusive: this check can prove drift, never
#: its absence.
_TAG_SAMPLE = 60
_TAG_SAMPLE_QUICK = 20

#: One cached map per entity type, written by tools/corpus_prep/tag_*.py from
#: the matcher named beside it. `entity_loader` calls the SAME matchers directly
#: at build time, so a built index is current with the matchers while these
#: cached copies need not be -- which is the whole reason this check exists.
_TAG_FILES = ("people", "programs", "courses", "faculties")


def audit_derived_copies(quick: bool) -> None:
    """Do the cached entity-tag files still reproduce from the current matchers?

    **This is the class the Qdrant collection guard came out of, swept.** A
    derived copy that nothing re-derives goes stale silently. The collection did
    it by ANSWERING from a previous build's payload; these files do it by being
    read as ground truth by `build_gold_candidates.py`, which is the qrels
    generator. CLAUDE.md has said "recompute tags from the tested matchers,
    never read entity_tags/*_by_file.json as ground truth" since 2026-08-12 --
    and writing a rule down is not a guard.

    **The date is only a proxy, and is deliberately not what is checked.** Every
    one of these files IS older than its own matcher, which says nothing about
    whether the matcher would now produce something different. So the check
    reads the artifact: it replicates each tagger's own text pipeline exactly --
    `strip_mapping_tables(strip_document_header(text))`, NOT `PlainLoader`,
    which since 2026-08-03 also strips course-comparison tables -- and compares
    tag for tag. A difference is then the matcher or the corpus, never a
    different preprocessing path.
    """
    from rag_lab.loaders.common import strip_document_header, strip_mapping_tables
    from rag_lab.loaders.course_loader import match_courses
    from rag_lab.loaders.faculty_loader import match_faculties
    from rag_lab.loaders.person_loader import match_people
    from rag_lab.loaders.program_loader import match_programs

    matchers = {"people": match_people, "programs": match_programs,
                "courses": match_courses, "faculties": match_faculties}
    tags_dir = CORPUS / "entity_tags"
    cached: dict[str, dict] = {}
    for name in _TAG_FILES:
        path = tags_dir / f"{name}_by_file.json"
        if not path.exists():
            record("T1 cached entity tags reproduce from the current matchers",
                   False, f"{path} is missing")
            return
        cached[name] = json.loads(path.read_text(encoding="utf-8"))

    files = sorted(iter_corpus_files(CORPUS), key=str)
    n = _TAG_SAMPLE_QUICK if quick else _TAG_SAMPLE
    step = max(1, len(files) // n)
    sample = files[::step][:n]

    differ = {k: 0 for k in matchers}
    compared = {k: 0 for k in matchers}
    for f in sample:
        rel = str(f.relative_to(CORPUS).as_posix())
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8-sig")
        text = strip_mapping_tables(strip_document_header(text))
        for name, fn in matchers.items():
            was = cached[name].get(rel)
            if was is None:
                continue
            compared[name] += 1
            now = json.dumps(fn(text), ensure_ascii=False, sort_keys=True)
            before = json.dumps(was, ensure_ascii=False, sort_keys=True)
            if now != before:
                differ[name] += 1

    parts = [f"{k} {differ[k]}/{compared[k]}" for k in _TAG_FILES]
    ok = not any(differ.values())
    remedy = (
        "; build_gold_candidates.py (the qrels generator) reads people/courses/"
        "faculties VALUES and now refuses on this, so nothing can silently derive "
        "qrels from them. RE-RUNNING tag_*.py IS NOT AN OBVIOUS FIX and is a "
        "decision, not a chore: it would make these copies current with today's "
        "corpus while data/index/entity_tags_full still holds tags from its own "
        "build date, i.e. it moves the mismatch rather than removing it, and "
        "CLAUDE.md coupled that index to a tag regeneration for exactly this "
        "reason. Decide the pair together"
    )
    record(
        "T1 cached entity tags reproduce from the current matchers",
        ok,
        f"files differing of files compared, over a {len(sample)}-file sample: "
        + ", ".join(parts) + ("" if ok else remedy),
    )

    # Stated so the T1 detail is not read as "all four are equally load-bearing".
    record(
        "T1b a programs drift cannot move the qrels",
        True,
        "program_candidates() iterates the tag mapping's KEYS and never reads a "
        "value -- audit_program_tag_regeneration.py S2 blanks every value and "
        "requires identical output -- so a programs difference is a reporting "
        "concern, not a qrels one",
    )


# ---------------------------------------------------------------- index layer
def combos() -> list[Path]:
    return [
        d
        for parent in sorted(INDEX_ROOT.iterdir())
        if parent.is_dir()
        for d in sorted(parent.iterdir())
        if (d / "chunks.parquet").exists()
    ]


def audit_docset_hashes() -> None:
    """Indices built by the same loader over the same number of resolutions must
    share a `docset_hash`.

    **The loader qualifier is the whole check, and it came from a false claim in
    CLAUDE.md.** `manifest._docset_hash` hashes `(resolution_id, raw_text)` AFTER
    the loader has run, so it identifies *corpus x loader*, not the corpus. That
    file said the four RQ3 treatment indices "carry the same docset_hash
    091b7a0ad8a5cfbe as both baselines -- so no confound". Three do; the
    `normalized` one carries `574945883e8320d0` and CANNOT carry the other,
    because normalising the text is the treatment. The conclusion survived (`I6`
    is the loader-independent evidence for it) but the evidence given was false,
    and no figure check could see it: the hash quoted is a real hash of a real
    index, just not of the index the sentence named.

    **The key is `(loader, n_resolutions)` and NOT `(root, loader)`, which is
    what the first version used and was wrong.** The claim being pinned is
    cross-ROOT -- the RQ3 treatment roots against the `chunker_compare_full`
    baselines -- so a per-root key cannot express it; and grouped per root the
    check split nowhere even with the loader qualifier removed, i.e. it was
    grouping on a property this fleet does not exercise. Keying on the loader
    globally would then wrongly demand that the smoke roots agree with the full
    corpus, so `n_resolutions` separates them -- on a real property of the build,
    not on a directory name. All 39 full-corpus `plain` indices, both RQ3 plain
    roots included, land in one group and must agree.

    **What it cannot see, stated because a check that looks broader than it is
    would be worse than none**: a group of one index has nothing to disagree
    with, so `entity_tags_full` and `rq3_normalize_ablation` -- each the sole
    holder of its loader -- are outside this check entirely. `I6` and `I7` are
    what watch those. Within a multi-index group, a disagreement is the shape a
    half-finished rebuild leaves, which is what rebuild #4 would have left had it
    died at combo 20 of 40.
    """
    groups: dict[tuple[str, object], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for d in combos():
        manifest_path = d / "manifest.json"
        if not manifest_path.exists():
            continue
        j = json.loads(manifest_path.read_text(encoding="utf-8"))
        loader = j.get("combo", {}).get("loader", {}).get("type", "?")
        groups[(loader, j.get("n_resolutions"))][j.get("docset_hash")].append(
            f"{d.parent.name}/{d.name}"
        )

    if not groups:
        # Not a pass. A comparison over nothing agrees with itself, and this
        # file has already had a check report "0 mixed of 0 checked".
        record("I8 one corpus state per (loader, n_resolutions)", False,
               "no index carries a manifest -- nothing was compared, which is a "
               "gap in the check's input, not a clean fleet")
        return

    split = {k: v for k, v in groups.items() if len(v) > 1}
    sizes = {k: sum(len(n) for n in v.values()) for k, v in groups.items()}
    singletons = sorted(k for k, n in sizes.items() if n == 1)
    n_idx = sum(sizes.values())
    detail = (
        f"{len(groups)} (loader, n_resolutions) groups over {n_idx} indices, "
        f"{len(split)} holding more than one docset_hash; "
        f"largest {max(sizes.values())}; "
        f"{len(singletons)} group(s) of one are UNWATCHED here "
        f"({', '.join(f'{l}/{n}' for l, n in singletons)}) -- I6/I7 cover those"
    )
    if split:
        detail += ". SPLIT: " + "; ".join(
            f"{loader}/{n} -> " + ", ".join(
                f"{h} ({len(names)}: {names[0]}...)" for h, names in sorted(v.items()))
            for (loader, n), v in sorted(split.items(), key=lambda kv: str(kv[0]))
        )
        detail += " -- a half-finished rebuild leaves exactly this"
    record("I8 one corpus state per (loader, n_resolutions)", not split, detail)


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

    # I7. Every index directory must match the build its writer sealed.
    #
    # `ArtifactStore.save` writes chunks.parquet and embeddings.npy in sequence
    # and they are row-aligned (I1 above), so between the two writes a reader
    # can pair one build's chunks with another's vectors -- stably, with no
    # stamp moving, and undetectably downstream. `index_cache` refuses a
    # directory whose artifacts do not match the seal `save` writes last. This
    # is the fleet-level version of that: an UNSEALED directory silently drops
    # the serving cache to the weaker guarantee, and a MISMATCHING one is
    # either mid-build or was rewritten in place without re-sealing, which
    # `relabel_index_resolution_ids.py` is the repo's one writer able to do.
    # WARN rather than FAIL, because unsealed is the pre-2026-08-21 state and a
    # directory can legitimately be mid-build while this runs.
    from rag_lab.io.artifact_store import artifact_stamp, read_seal  # noqa: E402

    unsealed, mismatched = [], []
    for d in combos():
        recorded = read_seal(d)
        if recorded is None:
            unsealed.append(str(d))
        elif recorded != artifact_stamp(d):
            mismatched.append(str(d))
    record(
        "I7 index matches the build its writer sealed",
        not unsealed and not mismatched,
        f"{len(unsealed)} unsealed, {len(mismatched)} mismatching of {len(list(combos()))} "
        f"(fix: python tools/seal_index_dirs.py --apply)",
        warn=True,
    )
    for m in (unsealed + mismatched)[:8]:
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
    been flagging the 81 truncated cells -- turned out to be clearable *without
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
    field -- they are counted separately, not as clean, because "no evidence of
    truncation" and "evidence of no truncation" are different
    ([[feedback_undefined_is_not_zero]]).

    **The pre-fix answers are not one bucket, and calling them one overstated the
    hole by 2x (sharpened 2026-08-11).** The field is not the only *provable*
    evidence about a prompt; two others already exist on disk and cost nothing:

      1. `rq4_generate.token_upper_bound` -- a prompt can be no more tokens than it
         is UTF-8 bytes, for any byte-level BPE vocabulary. A prompt under 8,192
         bytes provably fitted the old default whatever the tokenizer did. Loose
         (~3x on Thai) and that is the safe direction ([[feedback_an_observed_extreme_is_not_a_bound]]).
      2. `rq4_truncated_cells_raw.json` -- `rq4_find_truncated_answers.py` sent 228
         prompts to ollama *at num_ctx=8192* and recorded what it fed. That is the
         old run reproduced, not a proxy for it.

    Both are evidence about the *prompt*, so they need one premise about the answer:
    that pre-fix answers were generated at 8,192 (`docs/rq4-prompt-truncation.md`;
    it was the default until that day). The premise is used in the conservative
    direction only -- 8,192 is the smallest context any published run used, so
    "fits 8,192" implies "fits whatever it actually ran at", and a larger true
    num_ctx cannot turn a proven fit into a truncation.

    What is deliberately NOT used as evidence is `SCREEN_CHARS_PER_TOKEN = 0.95`,
    which would clear every remaining answer at a stroke. It is an observed
    minimum with headroom, and this project has already published a wrong blast
    radius three times by treating one of those as a bound.

    So: G1a is the recorded field, G1b re-derives the same claim from 1 and 2 and
    **fails** if either shows a truncation, and G1c reports what neither reaches --
    with its denominator, because 0 must not be ambiguous between "examined and
    clean" and "nothing left to examine".
    """
    root = Path("data/rq4/answers")
    if not root.is_dir():
        record("G1 no RQ4 answer generated from a truncated prompt", True,
               "0 of 0 answers on disk", warn=True)
        return

    truncated, pre_fix, checked = [], [], 0
    for path in sorted(root.glob("*/*/q*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        num_ctx, n_prompt = rec.get("num_ctx"), rec.get("prompt_eval_count")
        if num_ctx is None or n_prompt is None:
            pre_fix.append(path)
            continue
        checked += 1
        if n_prompt == num_ctx // 2 + 2:
            truncated.append(f"{path.relative_to(root)} ({n_prompt} tok at num_ctx={num_ctx})")

    record("G1a no RQ4 answer generated from a truncated prompt", not truncated,
           f"{len(truncated)} truncated of {checked} answers carrying num_ctx")
    for m in truncated[:10]:
        print(f"        {m}")

    by_bound, by_probe, unmeasured, pre_fix_trunc = _rq4_prompt_fit_evidence(pre_fix)
    proven = by_bound + by_probe
    # Deliberately not silent: docs/rq4-prompt-truncation.md reconstructed 81 of these
    # prompt by prompt and found them truncated (section 4 published 80; re-deriving
    # the list with a sound screen on 2026-08-10 found one more,
    # `cite_all_guarded/dense/q001` at 8,258 tokens, and the doc was corrected). Those
    # 81 were regenerated the same day, so they now carry the field and are counted by
    # G1a -- which is why this reads 0 rather than 81, and why it is a real FAIL if it
    # ever does not.
    record("G1b no pre-fix RQ4 answer is provably truncated", not pre_fix_trunc,
           f"{len(pre_fix_trunc)} truncated of {proven} pre-fix answers carrying "
           f"provable evidence either way about their prompt, none of it needing them "
           f"regenerated ({by_bound} by the UTF-8-byte upper bound, {by_probe} by a "
           f"cached probe at num_ctx=8,192)")
    for m in pre_fix_trunc[:10]:
        print(f"        {m}")
    # Closed 2026-08-11 by measuring it: `tools/eval/rq4_probe_prompt_fit.py` sent all
    # 759 remaining prompts to ollama at the old default (70 min) and none came back
    # cut. The warning stays wired rather than deleted -- a new pre-fix answer, a
    # rebuilt context, or a deleted probe cache all reopen it, and it must reopen as
    # *unmeasured* rather than silently as clean.
    record("G1c every RQ4 answer's prompt fit is established", not unmeasured,
           (f"{len(unmeasured)} of {checked + len(pre_fix)} answers have neither a "
            f"recorded num_ctx nor provable evidence about their prompt; they are "
            f"unmeasured, not suspected (the empirical 0.95 chars/token screen clears "
            f"all of them, but an observed minimum is not a bound). Closing this needs "
            f"a probe at num_ctx=8,192 per prompt, ~1 GPU-hour, not a regeneration")
           if unmeasured else
           (f"0 of {checked + len(pre_fix)} answers are unmeasured -- every published "
            f"RQ4 answer's prompt is now established by a recorded field, the "
            f"UTF-8-byte bound, or a probe at num_ctx=8,192 "
            f"(data/results/rq4_prompt_fit_probes.md), never by the 0.95 screen"),
           warn=True)


_RQ4_LEGACY_NUM_CTX = 8192
# answers/<model_dir>/<arm>/qNNN.json -- the variant lives in the model dir, and it
# is what picks rule 4, so the prompt cannot be rebuilt without it.
_RQ4_VARIANT_BY_DIR = {"phi4": "sentence_cap", "phi4_cite_all": "cite_all",
                       "phi4_cite_all_guarded": "cite_all_guarded"}


def _rq4_prompt_fit_evidence(paths: list[Path]) -> tuple[int, int, list[str], list[str]]:
    """Provable evidence that each pre-fix answer's prompt fitted 8,192 tokens.

    Returns (n_by_byte_bound, n_by_cached_probe, unmeasured, provably_truncated).
    See `audit_generation` for why these two sources and not the chars/token screen.
    """
    unmeasured: list[str] = []
    truncated: list[str] = []
    by_bound = by_probe = 0
    if _RQ4_CONTEXTS is None:
        return 0, 0, [str(p) for p in paths], []

    # Two probe caches, deliberately separate files rather than one shared pool.
    # `rq4_truncated_cells_raw.json` belongs to `rq4_find_truncated_answers.py`,
    # whose S1 self-check re-derives the realized chars/token of *every* entry to
    # prove its own 0.95 screen was sound. `rq4_probe_prompt_fit.py` probes a
    # wider universe (all 5 arms, screened by the provable byte bound instead), so
    # writing its entries into that file would let a hybrid prompt -- which that
    # screen never has to admit -- fail a check about a screen it is not under.
    # Same evidence, different provenance: read both, and let the older file win
    # any key they share so a re-read cannot silently change a published cell.
    raw: dict = {}
    for raw_path in (Path("data/results/rq4_prompt_fit_probes.json"),
                     Path("data/results/rq4_truncated_cells_raw.json")):
        try:
            raw.update(json.loads(raw_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    signature = _rq4_truncated_to(_RQ4_LEGACY_NUM_CTX)

    for path in paths:
        label = f"{path.parent.parent.name}/{path.parent.name}/{path.stem}"
        variant = _RQ4_VARIANT_BY_DIR.get(path.parent.parent.name)
        ctx_path = _RQ4_CONTEXTS / path.parent.name / path.name
        if variant is None or not ctx_path.is_file():
            unmeasured.append(label)
            continue
        try:
            prompt = _rq4_build_prompt(json.loads(ctx_path.read_text(encoding="utf-8")), variant)
        except (OSError, json.JSONDecodeError, KeyError):
            unmeasured.append(label)
            continue
        if _rq4_token_upper_bound(prompt) <= _RQ4_LEGACY_NUM_CTX:
            by_bound += 1
            continue
        probe = raw.get(f"{variant}/{path.parent.name}/{path.stem}") or {}
        n8, n16 = probe.get("n_8192"), probe.get("n_16384")
        if n8 is None:
            unmeasured.append(label)
            continue
        by_probe += 1
        # A prompt whose true length is near the signature reports it cut or not, so
        # where the finder disambiguated with a second probe, believe that one.
        if (n16 > n8 + 128) if n16 is not None else (n8 == signature):
            truncated.append(f"{label} (fed {n8:,} tok at num_ctx=8,192)")
    return by_bound, by_probe, unmeasured, truncated


# The report this project cites. A bare run prints to the terminal and vanishes,
# which is how the 2026-08-11 probe run (G1c closed at 14:55) left the 06:17
# report on disk still saying 26 pass / 2 warn / 0 fail while CLAUDE.md and the
# journey both -- correctly -- claimed 27/1/0. The claim was right and the
# artifact was stale, which is the harder direction to notice.
PUBLISHED_REPORT = Path("docs/pipeline-invariant-audit.md")


def _published_counts() -> tuple[int, int, int] | None:
    """`(pass, warn, fail)` as the report on disk states them, or None."""
    try:
        text = PUBLISHED_REPORT.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"(\d+) pass / (\d+) warn / (\d+) fail", text)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip embedding sampling; cap result files per dir")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    if args.quick and args.report:
        # A capped run reports capped denominators ("0 of 500 result files"),
        # and this project has already been bitten by a check whose subject
        # matter moved under it -- a vacuous PASS reads exactly like a real one.
        raise SystemExit(
            "refusing to write a report from a --quick run: it caps result files "
            "per dir and skips embedding sampling, so its denominators are not the "
            "published ones. Run without --quick to refresh "
            f"{PUBLISHED_REPORT}, or drop --report to see the quick numbers."
        )

    started = datetime.now(timezone.utc)
    print("=== corpus ===")
    corpus_ids, _ = audit_corpus()
    print("\n=== derived copies ===")
    audit_derived_copies(args.quick)
    print("\n=== indexes ===")
    ids_by_combo = audit_indexes(corpus_ids, args.quick)
    audit_docset_hashes()
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
    elif args.quick:
        print(f"\n[note] --quick: counts are not comparable to {PUBLISHED_REPORT}")
    else:
        # Say whether the artifact everyone reads still matches this run. A
        # reminder to pass --report would only be read by whoever already
        # remembered; this is a check on the file itself.
        counts = (len(findings) - len(fails) - len(warns), len(warns), len(fails))
        published = _published_counts()
        if published is None:
            print(f"\n[note] no readable summary in {PUBLISHED_REPORT}")
        elif published != counts:
            print(f"\n[STALE] {PUBLISHED_REPORT} says "
                  f"{published[0]} pass / {published[1]} warn / {published[2]} fail; "
                  f"this run says {counts[0]} / {counts[1]} / {counts[2]}. "
                  f"Re-run with --report {PUBLISHED_REPORT} to publish it.")
        else:
            print(f"\n[ok] {PUBLISHED_REPORT} matches this run "
                  f"({counts[0]} pass / {counts[1]} warn / {counts[2]} fail)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
