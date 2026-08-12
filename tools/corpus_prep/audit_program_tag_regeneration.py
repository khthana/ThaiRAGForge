"""Attribute the 2026-08-12 regeneration of `programs_by_file.json`, and verify
for real the blast radius that `docs/program-matcher-absorption.md` could only
state as a counterfactual.

WHY THIS EXISTS. The cached tag file dated 2026-07-25; the matcher was repaired
2026-08-11 (degree filter + cross-subject guard) and the corpus moved three times
in between (07-28 OCR remediation, 08-08 title repair, 08-09 re-OCR). So a plain
old-vs-new diff of the artifact bundles at least two causes, and this project's
signature failure is exactly a comparison between two artifacts produced on
different days by different code. §2 splits them, using the absorption walk's own
cache as the middle term: that walk ran 2026-08-11 over TODAY's corpus and records,
per mention, both the old winner (`canonical`) and the new one (`selected`), so

    old artifact  --(B) corpus drift-->  cache old  --(A) matcher repair-->  cache new

and S1 pins the right-hand end by requiring the reconstruction to reproduce the
regenerated artifact exactly. Without S1 the middle term is an assumption.

WHAT §3 FOUND, stated up front because it is stronger than the claim it replaces.
CLAUDE.md says `program_candidates()` "seeds from tagged files then gates on
`canonical in resolution_id`". The first half is not what the code does: it
iterates the mapping's KEYS, and `tag_programs.py` writes a key for every live
corpus file including the ones matching zero programs. So the tag VALUES are not
read at all, and the program qrels cannot move with the matcher -- structurally,
not merely by measurement. S2 checks that by blanking every value and requiring
identical output. §3 still recomputes both arms, because a structural argument
that is never executed is the kind this project has been wrong about before.

No GPU, seconds. Run with:
    .venv/Scripts/python.exe tools/corpus_prep/audit_program_tag_regeneration.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO / "academic_resolutions"
TAGS_DIR = CORPUS_ROOT / "entity_tags"
NEW_TAGS = TAGS_DIR / "programs_by_file.json"
OLD_TAGS = TAGS_DIR / "_snapshots" / "programs_by_file.2026-07-25.pre_matcher_repair.json"
ABSORPTION_RAW = REPO / "data" / "results" / "program_matcher_absorption_raw.json"
PUBLISHED_CANDIDATES = TAGS_DIR / "gold_candidates.json"
GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
REPORT = REPO / "docs" / "program-tag-regeneration.md"

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "corpus_prep"))
import build_gold_candidates as BGC  # noqa: E402
from rag_lab.router import classify_query  # noqa: E402

# The published program candidate set was built at this threshold
# (build_gold_candidates.py's --min-program-hits default).
MIN_PROGRAM_HITS = 2

# docs/program-matcher-absorption.md's per-file figures for the two guards
# together, which S3 must reproduce from the same cache this script reads.
PUBLISHED_REPAIR = {"gained": 140, "lost": 594, "stripped_bare": 115, "changed": 446}
# docs/program-matcher-absorption.md §5's cross-tab over the 106-query 73det set.
# Keyed on (gold entity_type, route) rather than on either alone: `faculty` is the
# one place the two vocabularies differ, so a per-type count would hide a
# misrouting that landed on a same-sized route.
PUBLISHED_ROUTES = {
    ("course", "course"): 33,
    ("faculty_adjunct_aggregate", "faculty"): 13,
    ("person", "person"): 30,
    ("program", "program"): 30,
}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# §2  attribution
# --------------------------------------------------------------------------- #
def reconstruct_from_cache(raw: dict) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(old-matcher, new-matcher) tags per file, both over TODAY's corpus.

    `match_programs` returns its canonicals deduped and sorted, so a set over the
    per-mention records is the same object it would return -- which is what S1
    tests rather than trusts.
    """
    old: dict[str, list[str]] = {}
    new: dict[str, list[str]] = {}
    for relpath, hits in raw["files"].items():
        old[relpath] = sorted({h["canonical"] for h in hits})
        new[relpath] = sorted({h["selected"] for h in hits if h["selected"]})
    return old, new


def _tag_delta(before: dict[str, list[str]], after: dict[str, list[str]]) -> dict:
    """Per-file movement between two mappings, over their shared key set."""
    shared = set(before) & set(after)
    gained = lost = changed = stripped = 0
    for f in shared:
        b, a = set(before[f]), set(after[f])
        if b == a:
            continue
        changed += 1
        gained += len(a - b)
        lost += len(b - a)
        if b and not a:
            stripped += 1
    return {
        "files_compared": len(shared),
        "only_before": len(set(before) - set(after)),
        "only_after": len(set(after) - set(before)),
        "changed": changed,
        "gained": gained,
        "lost": lost,
        "stripped_bare": stripped,
        "tags_before": sum(len(v) for f, v in before.items() if f in shared),
        "tags_after": sum(len(v) for f, v in after.items() if f in shared),
        "files_tagged_before": sum(1 for f in shared if before[f]),
        "files_tagged_after": sum(1 for f in shared if after[f]),
    }


# --------------------------------------------------------------------------- #
# §3  blast radius: the program qrels
# --------------------------------------------------------------------------- #
def program_candidates_from(tags: dict[str, list[str]], min_hits: int) -> list[dict]:
    """`build_gold_candidates.program_candidates()` with the mapping injected
    instead of read from disk. Every other step is imported from that module so
    the two cannot drift apart."""
    programs = _load(BGC.DICT_DIR / "programs.json")
    resolution_ids = sorted({BGC._resolution_id_for(relpath) for relpath in tags})

    candidates = []
    for program in programs:
        canonical, field = program["canonical"], program["field"] or ""
        if not canonical or not field:
            continue
        hits = sorted((rid for rid in resolution_ids if canonical in rid), key=BGC._sort_key)
        if len({BGC._event_key(rid) for rid in hits}) < min_hits:
            continue
        candidates.append({"entity": canonical, "relevant_resolution_ids": hits})
    return candidates


def _by_entity(cands: list[dict]) -> dict[str, list[str]]:
    return {c["entity"]: c["relevant_resolution_ids"] for c in cands}


def diff_candidates(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict:
    moved = {e: (a[e], b[e]) for e in set(a) & set(b) if a[e] != b[e]}
    return {
        "n_before": len(a),
        "n_after": len(b),
        "only_before": sorted(set(a) - set(b)),
        "only_after": sorted(set(b) - set(a)),
        "moved": moved,
        "pairs_before": sum(len(v) for v in a.values()),
        "pairs_after": sum(len(v) for v in b.values()),
    }


# --------------------------------------------------------------------------- #
# self-checks
# --------------------------------------------------------------------------- #
def self_checks(new_disk, old_disk, cache_old, cache_new, repair, routes, new_cands,
                qrels, gold_moved, n_gold_pairs, n_published_moved) -> list:
    checks = []

    # S1 -- the attribution's load-bearing assumption. If the cache reconstruction
    # does not reproduce the regenerated artifact, the "cache old" middle term is
    # not the old matcher over today's corpus and §2's split is meaningless.
    mismatch = [f for f, v in new_disk.items()
                if v != (cache_new.get(f, []) if f in cache_new else [])]
    checks.append((
        "S1 cache reconstruction == regenerated artifact",
        not mismatch,
        f"{len(new_disk) - len(mismatch)} of {len(new_disk)} files agree"
        + (f"; first mismatch `{mismatch[0]}`" if mismatch else ""),
    ))

    # S2 -- the structural claim in the docstring, executed. program_candidates()
    # iterates the mapping's keys, so blanking every value must change nothing.
    blanked = _by_entity(program_candidates_from({f: [] for f in new_disk}, MIN_PROGRAM_HITS))
    checks.append((
        "S2 program_candidates reads only the key set",
        blanked == new_cands,
        f"{len(blanked)} candidates from an all-empty mapping, identical: {blanked == new_cands}",
    ))

    # S3 -- the repair's own published per-file figures, re-derived here from the
    # same cache, so a silently rewritten cache cannot pass unnoticed.
    got = {k: repair[k] for k in PUBLISHED_REPAIR}
    checks.append((
        "S3 repair figures reproduce docs/program-matcher-absorption.md",
        got == PUBLISHED_REPAIR,
        f"got {got} vs published {PUBLISHED_REPAIR}",
    ))

    # S4 -- corpus membership is the ONE input that can move the program qrels
    # (S2 having shown the tag values cannot). Membership *changing* is expected
    # -- the corpus is live -- so gate on the consequence, not on the fact: a
    # bare "unchanged" test would sit permanently red after any legitimate
    # addition, which is the one thing that makes a check stop being read.
    only_old, only_new = set(old_disk) - set(new_disk), set(new_disk) - set(old_disk)
    moved = len(qrels["moved"]) + len(qrels["only_before"]) + len(qrels["only_after"])
    membership = (
        f"{len(old_disk)} -> {len(new_disk)} files (removed {len(only_old)}, "
        f"added {len(only_new)}"
        + (": " + "; ".join(sorted(only_new | only_old)) if (only_new | only_old) else "")
        + f"); {moved} program candidates moved"
    )
    checks.append((
        "S4 corpus membership change moves no program candidate",
        moved == 0,
        membership,
    ))

    # S6 -- the 07-25 `gold_candidates.json` is a superseded intermediate: the
    # scored set was regenerated 2026-07-30 for the resolution_id uniqueness fix,
    # so a candidate can legitimately differ from it while the gold set is right.
    # What must hold is that no such difference reaches a SCORED pair.
    checks.append((
        "S6 no scored gold pair is lost by the recomputed candidate set",
        not gold_moved,
        f"{len(gold_moved)} of {n_gold_pairs} program gold pairs lost; "
        f"{n_published_moved} published candidate(s) differ from the 07-25 artifact "
        f"(superseded by the 07-30 gold-set regeneration)",
    ))

    # S5 -- the router, pinned to the same distribution tests/test_router.py holds.
    checks.append((
        "S5 router distribution unchanged",
        routes == PUBLISHED_ROUTES,
        f"got {dict(sorted(routes.items()))}",
    ))
    return checks


# --------------------------------------------------------------------------- #
def main() -> int:
    import yaml

    new_disk = _load(NEW_TAGS)
    old_disk = _load(OLD_TAGS)
    raw = _load(ABSORPTION_RAW)
    cache_old, cache_new = reconstruct_from_cache(raw)

    # Fill the cache's implicit zero-match files so the two arms are comparable
    # over the whole corpus rather than over the 1,710 files that had a hit.
    cache_old_full = {f: cache_old.get(f, []) for f in new_disk}
    cache_new_full = {f: cache_new.get(f, []) for f in new_disk}

    overall = _tag_delta(old_disk, new_disk)
    drift = _tag_delta({f: old_disk[f] for f in old_disk if f in cache_old_full}, cache_old_full)
    repair = _tag_delta(cache_old_full, cache_new_full)

    old_cands = _by_entity(program_candidates_from(old_disk, MIN_PROGRAM_HITS))
    new_cands = _by_entity(program_candidates_from(new_disk, MIN_PROGRAM_HITS))
    qrels = diff_candidates(old_cands, new_cands)

    # gold_candidates.json is keyed by category ("programs"/"people"/...), not a
    # flat list -- so index it, don't filter it.
    published = _by_entity(_load(PUBLISHED_CANDIDATES)["programs"])
    vs_published = diff_candidates(published, new_cands)

    gold = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    routes = Counter((e.get("entity_type"), classify_query(e["query"])) for e in gold)
    prog_gold = [e for e in gold if e.get("entity_type") == "program"]
    gold_pairs = [(e["entity"], rid) for e in prog_gold for rid in e["relevant_resolution_ids"]]
    gold_moved = [(ent, rid) for ent, rid in gold_pairs
                  if rid not in new_cands.get(ent, []) and ent in new_cands]

    checks = self_checks(new_disk, old_disk, cache_old_full, cache_new_full,
                         repair, dict(routes), new_cands,
                         qrels, gold_moved, len(gold_pairs), len(vs_published["moved"]))

    text = render(overall, drift, repair, qrels, vs_published,
                  len(prog_gold), len(gold_pairs), gold_moved, dict(routes), checks)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten -> {REPORT}")
    return 0 if all(ok for _, ok, _ in checks) else 1


def _delta_table(name: str, d: dict) -> list[str]:
    return [
        f"| {name} | {d['files_compared']} | {d['changed']} | +{d['gained']} | "
        f"-{d['lost']} | {d['stripped_bare']} | {d['tags_before']} -> {d['tags_after']} | "
        f"{d['files_tagged_before']} -> {d['files_tagged_after']} |"
    ]


def render(overall, drift, repair, qrels, vs_published,
           n_prog_gold, n_gold_pairs, gold_moved, routes, checks) -> str:
    L = [
        "# programs_by_file.json regeneration (2026-08-12)",
        "",
        "`tools/corpus_prep/audit_program_tag_regeneration.py`. The cached program",
        "tags dated **2026-07-25**; `match_programs` was repaired **2026-08-11**",
        "(degree filter + cross-subject guard, `docs/program-matcher-absorption.md`)",
        "and the corpus moved three times in between. This report regenerates the",
        "artifact, splits the delta into the two causes, and verifies for real the",
        "blast radius that could previously only be stated as a counterfactual --",
        "`build_gold_candidates.py` was reading the *cached* file, so nothing",
        "published had moved yet.",
        "",
        "## 1-2. What changed, and why",
        "",
        "| arm | files | files changed | tags gained | tags lost | stripped bare | total tags | files tagged |",
        "|---|---|---|---|---|---|---|---|",
    ]
    L += _delta_table("**overall** (07-25 artifact -> 08-12 artifact)", overall)
    L += _delta_table("(B) corpus drift (old matcher, 07-25 -> 08-12 corpus)", drift)
    L += _delta_table("(A) matcher repair (08-12 corpus, old -> new matcher)", repair)
    L += [
        "",
        "The middle row is the corpus changes alone (07-28 OCR remediation, 08-09",
        "re-OCR); the 08-08 title repair cannot appear here at all, because",
        "`tag_programs.py` keys on the file path and matches over body text, and a",
        "title repair touches neither. The bottom row is the two guards, and it",
        "reproduces `docs/program-matcher-absorption.md`'s own per-file figures",
        "(S3) -- which is what licenses reading the middle row as drift rather",
        "than as a residue.",
        "",
        "## 3. Does it reach the program qrels?",
        "",
        "**No, and the reason is stronger than a measurement.** `program_candidates()`",
        "iterates the mapping's **keys**, and `tag_programs.py` writes a key for every",
        "live corpus file -- including the ones that match zero programs. The tag",
        "*values* are never read: the function derives a `resolution_id` per file and",
        "gates on `canonical in resolution_id`, an exact substring of the manifest",
        "title. So the matcher cannot move the program qrels **structurally**, not",
        "merely in this sample; S2 executes that claim by blanking every value and",
        "requiring identical output. (CLAUDE.md's wording -- \"seeds from tagged",
        "files\" -- overstated the coupling; only corpus *membership* matters, which",
        "is what S4 watches.)",
        "",
        "Both arms were recomputed anyway:",
        "",
        f"- program candidates from the 07-25 mapping: **{qrels['n_before']}** "
        f"({qrels['pairs_before']} pairs)",
        f"- program candidates from the 08-12 mapping: **{qrels['n_after']}** "
        f"({qrels['pairs_after']} pairs)",
        f"- candidates appearing in only one: **{len(qrels['only_before'])} / "
        f"{len(qrels['only_after'])}**",
        f"- candidates whose resolution_id list moved: **{len(qrels['moved'])}**",
        "",
        "Against the published `gold_candidates.json` (built 2026-07-25, before the",
        "08-08 title repair moved 4 resolution_ids):",
        "",
        f"- published program candidates: **{vs_published['n_before']}**; recomputed: "
        f"**{vs_published['n_after']}**",
        f"- only published / only recomputed: **{len(vs_published['only_before'])} / "
        f"{len(vs_published['only_after'])}**",
        f"- candidates whose list moved: **{len(vs_published['moved'])}**",
        "",
        "Read that last figure against the right artifact. `gold_candidates.json`",
        "dates 2026-07-25 and is a **superseded intermediate**: the scored set was",
        "regenerated 2026-07-30 for the `resolution_id` uniqueness fix, so a",
        "candidate can differ from it while the gold set is correct. The one that",
        "does is `...สาขาวิชาวิศวกรรมชีวการแพทย์`, whose single truncated",
        "`2567/1/...(หลักสูตรนานาชาติ)` id became the two full ids that end",
        "`...ฉบับปี พ.ศ. ๒๕๖๓` / `๒๕๖๔` -- and `gold_query_set_73det.yaml` already",
        "holds both. What must hold is that no difference reaches a *scored* pair,",
        "which is S6.",
        "",
        "The corpus also gained one file since 07-25 (S4):",
        "`2568/ครั้งที่ 7/เรื่อง รับรองรายงานการประชุม.md`, from the 2026-08-09 CHECO",
        "restoration. It matches zero programs and its `resolution_id` contains no",
        "program canonical, so it adds no hit to any candidate -- which is why",
        "membership moved and the qrels did not.",
        "",
        "And against the scored gold set itself (`gold_query_set_73det.yaml`):",
        "",
        f"- program gold queries: **{n_prog_gold}**, gold pairs: **{n_gold_pairs}**",
        f"- gold pairs no longer produced by the recomputed candidate set: "
        f"**{len(gold_moved)}**",
        "",
        "## 4. Does it change a route?",
        "",
        "`classify_query` asks only whether *any* program matched, never which one.",
        "",
        "| gold entity_type | route | queries |",
        "|---|---|---|",
    ]
    for (et, route), v in sorted(routes.items()):
        L.append(f"| {et} | {route} | {v} |")
    L += ["", "## 5. Self-checks", "",
          "| check | result | detail |", "|---|---|---|"]
    for name, ok, detail in checks:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    # Thai + `->` on a cp874 console would traceback *after* the report is
    # already written, losing the self-check exit code.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
