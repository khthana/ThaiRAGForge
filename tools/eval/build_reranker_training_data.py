"""Mint training data for follow-up (a): a reranker trained on hybrid-fused candidates.

Pre-registration: `docs/reranker-trained-on-hybrid-design.md`. Read it before
changing anything here -- the prediction and the decision rule were fixed before
this file existed, and several choices below are load-bearing for them.

WHAT THIS BUILDS, and why each half is the way it is
----------------------------------------------------
The intervention under test is "a reranker trained on hybrid-fused candidate
distributions specifically ... rather than assuming an off-the-shelf
single-retriever-trained cross-encoder transfers"
(`docs/reranker-hybrid-interaction-research.md`). So the training candidates must
come from **the same pool the model will face at eval time**: routed hybrid,
top-P, built by `reranker_rrf_routed_test.rank_one_index` -- imported, never
reimplemented, because two copies of that dense-first RRF tie-break would
eventually disagree (the rule `routed_fetch_depth_test.py` already follows).

The training *queries* are minted by the eval qrels' own generator,
`tools/corpus_prep/build_gold_candidates.py`, over entities **disjoint from the
106-query eval set**. Using the same generator is deliberate and has a cost,
disclosed in the design doc as the central validity threat: training labels and
eval qrels then share one relevance definition, so a gain could mean "learned the
labelling function". That is why the eval script carries a lexical-containment
control. The alternative -- inventing a second labelling rule for training --
would measure a different task and make a null uninterpretable.

Course candidates are recomputed here rather than imported, and the reason is
cost, not disagreement: `build_gold_candidates.course_candidates` additionally
computes `anchor_precision` by searching all 678 dictionary names across every
tagged document, and none of those annotation fields touch
`relevant_resolution_ids`. `_course_relevant_sets` reproduces the label half
exactly (code -> resolution_ids via `courses_by_file.json`), and **T7 checks the
whole pipeline against the shipped eval qrels** rather than trusting that claim:
if the generator reproduces what `gold_query_set_73det.yaml` says for the eval
entities, it is the same working code path that labels the disjoint ones.

WHY THE CANDIDATE POOL IS CACHED, AND WHAT GUARDS THE CACHE
-----------------------------------------------------------
`person_candidates` reads every tagged document, so minting costs minutes. The
pool is cached -- but a cache is this project's signature silent-corruption
shape (two artifacts produced on different days), so the cache stores a
fingerprint of everything it derives from (entity dictionaries, tag snapshots,
and meeting manifests, since `resolution_id` comes from the manifest title per
ADR-0003) and is **re-minted** rather than silently reused when any of them moved.

WHAT IS DROPPED, AND WHY IT IS COUNTED
--------------------------------------
A query whose top-P pool contains **no** gold chunk teaches a group-softmax loss
nothing -- there is no positive to select. Those queries are dropped, and T5
prints how many, because "0 dropped" and "we stopped looking" are the same number
otherwise ([[feedback_undefined_is_not_zero]]).

THE QUERY'S SURFACE FORM IS PART OF THE EXPERIMENT, NOT DECORATION
------------------------------------------------------------------
The generator's own `query` strings are **not** what the shipped eval set says.
`gold_query_set_73det.yaml` was hand-paraphrased, and for `program` it names the
**full canonical** ("หลักสูตรการออกแบบบัณฑิต สาขาวิชาปัญญาออกแบบ...") where
`program_candidates` emits only the bare field ("หลักสูตรปัญญาออกแบบ..."). Measured
over the 106: course 33/33 identical, program 30/30 different, person 29/30,
faculty 13/13. That is not cosmetic -- under the generator's own wording **all 117
disjoint program candidates route to `unmatched`**, so their training pools would
be drawn from a different index than any eval program query ever touches, which
confounds (a) with a query-form mismatch: the very distribution mismatch the
intervention exists to remove. `QUERY_TEMPLATES` therefore renders each candidate
in the eval set's own phrasings (full canonical entity, several paraphrases so the
model cannot key on one fixed frame), and **T8 gates the routes mechanically**
rather than trusting the wording.

Run with (GPU: loads one embedder at a time, never two resident):
    .venv/Scripts/python.exe tools/eval/build_reranker_training_data.py --smoke
    .venv/Scripts/python.exe tools/eval/build_reranker_training_data.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))
sys.path.insert(0, str(REPO / "tools" / "corpus_prep"))

from rag_lab.query_service import discover_indices, resolve_index  # noqa: E402
from rag_lab.router import classify_query, route_targets  # noqa: E402

import build_gold_candidates as bgc  # noqa: E402
from reranker_rrf_routed_test import INDEX_ROOT, P_MAX, rank_one_index  # noqa: E402

GOLD = REPO / "config" / "eval" / "gold_query_set_73det.yaml"
OUT_DIR = REPO / "data" / "results" / "reranker_train"
OUT_CANDS = OUT_DIR / "candidate_pool.json"
OUT_POOLS = OUT_DIR / "train_pools.json"
OUT_META = OUT_DIR / "train_meta.json"
REPORT = REPO / "data" / "results" / "reranker_training_data.md"

# The pool depth the experiment is pre-registered at. Training on exactly the
# depth the eval fuses at is the point of (a); a deeper training pool would show
# the model negatives it never meets at eval time.
P_TRAIN = 50

# Same threshold the shipped eval set was curated under, so a training entity is
# the same kind of object an eval entity is.
MIN_HITS = 2

# Held-out slice of TRAINING queries, used to select the checkpoint. Selecting a
# checkpoint on the 106 eval queries would make the whole experiment an argmax
# (design doc section 5).
DEV_FRACTION = 0.15

SEED = 42

# Modelled on the shipped eval set's own phrasings (the first entry of each list
# is one lifted from `gold_query_set_73det.yaml` verbatim, with the entity
# substituted). Every template names the entity in FULL, because that is what the
# eval queries do and what `classify_query` needs to route them -- see the module
# docstring. `program` canonicals already begin with "หลักสูตร".
QUERY_TEMPLATES: dict[str, list[str]] = {
    "program": [
        "{e}มีการเปลี่ยนแปลงกี่ครั้ง และแต่ละครั้งมีรายละเอียดอย่างไรบ้าง",
        "อยากทราบประวัติการปรับปรุงของ{e}ทั้งหมด มีกี่ครั้ง ครั้งไหนบ้าง",
        "{e}เคยมีการแก้ไขหลักสูตรกี่ครั้ง แต่ละครั้งแก้ไขอะไรบ้าง",
        "ช่วยสรุปการเปลี่ยนแปลงของ{e}ที่ผ่านมาทั้งหมดให้หน่อย",
    ],
    "person": [
        "{e} มีประวัติเกี่ยวข้องกับหลักสูตรใดบ้าง ในช่วงใด",
        "อยากทราบว่า {e} เคยเกี่ยวข้องกับหลักสูตรอะไรบ้าง ตั้งแต่เมื่อไหร่",
        "{e} ปรากฏอยู่ในหลักสูตรใดบ้าง และในการประชุมครั้งไหน",
    ],
    "course": [
        "รายวิชา {e} ถูกกล่าวถึงในการประชุมสภาสถาบันครั้งใดบ้าง ให้แสดงรายละเอียดทั้งหมด",
        "อยากทราบว่ารายวิชา {e} เคยเข้าที่ประชุมสภาสถาบันครั้งไหนบ้าง",
        "รายวิชา {e} ปรากฏในมติของการประชุมครั้งใดบ้าง มีรายละเอียดอะไร",
    ],
    "faculty_adjunct_aggregate": [
        "หลักสูตรไหนของ{e}ที่ใช้อาจารย์พิเศษสอนเกินร้อยละ 50 มากที่สุด กี่คน และมีใครบ้าง",
        "ใน{e} หลักสูตรใดเชิญอาจารย์พิเศษมาสอนเกินร้อยละ 50 บ่อยที่สุด กี่คน ใครบ้าง",
        "{e}มีหลักสูตรใดบ้างที่อาจารย์พิเศษสอนเกินร้อยละ 50 และเป็นใคร",
    ],
}

# What `classify_query` must return for a training query of each type, so its
# pool comes from the same index the eval's own queries of that type reach.
EXPECTED_ROUTE = {
    "program": "program",
    "person": "person",
    "course": "course",
    "faculty_adjunct_aggregate": "faculty",
}

# T8's bar. 0 is the wrong bar: a handful of entities genuinely carry another
# type's anchor text (`gold_anchor_ambiguity.md` §3b found a shorter dictionary
# course name sitting inside another query's own name), so a few misroutes are a
# property of the dictionaries. The defect T8 exists to catch -- a systemic
# wording mismatch -- shows up as tens of percent, not units.
MAX_MISROUTE_RATE = 0.05


# --------------------------------------------------------------------------- #
# inputs and their fingerprint
# --------------------------------------------------------------------------- #
def _max_mtime(paths) -> float:
    return max((p.stat().st_mtime for p in paths), default=0.0)


def input_fingerprint() -> dict:
    """Everything a minted candidate's labels depend on. Manifests are in here
    because a `resolution_id` is built from the manifest title (ADR-0003), so a
    title repair moves ids without touching a single tag file -- the exact class
    of change `I6` was once blind to."""
    return {
        "dicts": round(_max_mtime(bgc.DICT_DIR.glob("*.json")), 3),
        "tags": round(_max_mtime(bgc.TAGS_DIR.glob("*_by_file.json")), 3),
        "manifests": round(
            _max_mtime(bgc.CORPUS_ROOT.glob("*/*/meeting_manifest.json")), 3
        ),
    }


def _load_eval_set() -> tuple[list[dict], set[str], set[str]]:
    raw = yaml.safe_load(GOLD.read_text(encoding="utf-8"))
    return raw, {d["entity"] for d in raw}, {d["query"] for d in raw}


def _course_relevant_sets() -> list[dict]:
    """`build_gold_candidates.course_candidates`'s label half, without its
    anchor-precision annotation pass (which reads every tagged document and does
    not touch `relevant_resolution_ids`). T7 gates this against the shipped
    qrels rather than asking anyone to take that on trust."""
    courses = bgc._load_json(bgc.DICT_DIR / "courses.json")
    by_file = bgc._load_json(bgc.TAGS_DIR / "courses_by_file.json")
    code_to_rids: dict[str, set[str]] = {}
    for relpath, codes in by_file.items():
        rid = bgc._resolution_id_for(relpath)
        for code in codes:
            code_to_rids.setdefault(code, set()).add(rid)
    out = []
    for course in courses:
        rids = code_to_rids.get(course["code"], set())
        if len(rids) < MIN_HITS:
            continue
        out.append({
            "entity_type": "course",
            "entity": course["canonical"],
            "code": course["code"],
            "query": f"รายวิชา {course['canonical']} ถูกกล่าวถึงในการประชุมสภาสถาบันครั้งใดบ้าง ให้แสดงรายละเอียดทั้งหมด",
            "relevant_resolution_ids": sorted(rids, key=bgc._sort_key),
            "hit_count": len(rids),
        })
    return out


def render_queries(pool: dict[str, list[dict]]) -> None:
    """Rewrite every candidate's `query` in the eval set's surface form, in
    place. Deterministic per entity (seeded on the entity string, not on
    iteration order) so re-running picks the same paraphrase for the same entity
    even if the pool is re-minted in a different order."""
    for entity_type, candidates in pool.items():
        templates = QUERY_TEMPLATES[entity_type]
        for c in candidates:
            rng = random.Random(f"{SEED}:{entity_type}:{c['entity']}")
            c["generator_query"] = c["query"]
            c["query"] = rng.choice(templates).format(e=c["entity"])


# --------------------------------------------------------------------------- #
# T7: does today's generator still reproduce the SHIPPED qrels, and if not, why
# --------------------------------------------------------------------------- #
def _person_tag_evidence(canonical: str, tags: dict, alias_index: dict):
    """What the frozen tag snapshot claims about one person, and what the
    CURRENT corpus text still supports.

    `person_candidates` is the only generator that re-reads the document text
    (its secretarial-signature filter needs it), so it is the only one whose
    output can drift while its inputs sit still: `people_by_file.json` is a
    2026-07-25 snapshot recording surnames **as they were OCR'd then**, and the
    07-28 remediation / 08-09 re-OCR corrected some of them in the corpus. A
    mention cached as `ดร.กฤษณะ โฆษชุณหันท์` cannot be found in text that now
    reads `โฆษชุณหนันท์`, so the label silently disappears -- which makes the
    stale artifact the tag cache, NOT the shipped qrels (the person really is
    named in that document; only the cached spelling of them is obsolete)."""
    cached, verified, stale = set(), set(), set()
    for relpath, mentions in tags.items():
        ms = [
            m for m in mentions
            if alias_index.get((m["given_name"], m["surname"])) == canonical
        ]
        if not ms:
            continue
        rid = bgc._resolution_id_for(relpath)
        cached.add(rid)
        text = (bgc.CORPUS_ROOT / relpath).read_text(encoding="utf-8", errors="replace")
        if any(
            bgc._has_non_secretarial_mention(text, m["given_name"], m["surname"])
            for m in ms
        ):
            verified.add(rid)
        elif all(m["surname"] not in text for m in ms):
            stale.add(rid)
    return cached, verified, stale


def explain_qrels_drift(entry: dict, got: list[str] | None, tags, alias_index) -> str | None:
    """Return a mechanism if today's output differs from the shipped qrels for a
    reason already understood, else None (which T7 counts as unexplained and
    fails on). Only ever *subtractive* drift is explainable: an id today's
    generator finds that the shipped set lacks is a different event entirely."""
    if entry["entity_type"] != "person":
        return None
    want, have = set(entry["relevant_resolution_ids"]), set(got or ())
    if have - want:
        return None
    _cached, verified, stale = _person_tag_evidence(entry["entity"], tags, alias_index)
    # When the entity vanished from the pool there is nothing to subtract from,
    # so the comparison is against what the current text still verifies -- an
    # entity disappears when the ids the OCR repair took away drop the survivors
    # under the curation threshold, which is a *combination* of the two effects,
    # not either alone.
    have = have if got is not None else verified
    missing = want - have
    if not missing or not missing <= stale:
        return None
    if got is None:
        events = len({bgc._event_key(r) for r in verified})
        if events >= MIN_HITS:
            return None
        return f"stale tag OCR ({len(missing)} id) → {events} event < min_hits {MIN_HITS}"
    return f"stale tag OCR ({len(missing)} id)"


def mint_candidates(refresh: bool) -> tuple[dict[str, list[dict]], str]:
    fp = input_fingerprint()
    if OUT_CANDS.exists() and not refresh:
        cached = json.loads(OUT_CANDS.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fp:
            return cached["pool"], f"reused cache from {cached['at']}"
        print("  cache fingerprint moved — re-minting", file=sys.stderr)

    t = time.time()
    pool = {
        "program": bgc.program_candidates(MIN_HITS),
        "course": _course_relevant_sets(),
        "faculty_adjunct_aggregate": bgc.faculty_adjunct_candidates(MIN_HITS),
    }
    print(f"  program/course/faculty  {time.time()-t:.0f}s", file=sys.stderr)
    # person_candidates reads every tagged document (its secretarial-signature
    # filter needs the text), so it is the slow one -- run it last so a failure
    # in the cheap types surfaces first.
    pool["person"] = bgc.person_candidates(MIN_HITS)
    print(f"  person  {time.time()-t:.0f}s", file=sys.stderr)

    at = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CANDS.write_text(
        json.dumps({"at": at, "fingerprint": fp, "pool": pool}, ensure_ascii=False),
        encoding="utf-8",
    )
    return pool, f"minted {at} in {time.time()-t:.0f}s"


def choose_training_queries(
    pool: dict[str, list[dict]], eval_entities: set[str], eval_queries: set[str], n_train: int
) -> list[dict]:
    """Disjoint by entity AND by query string, stratified toward the eval set's
    own type mix but never inventing entities that do not exist -- `faculty` has
    almost no disjoint candidate, which the design doc records as a known hole
    rather than papering over."""
    rng = random.Random(SEED)
    disjoint = {
        t: sorted(
            (c for c in cs if c["entity"] not in eval_entities and c["query"] not in eval_queries),
            key=lambda c: c["entity"],
        )
        for t, cs in pool.items()
    }
    # Take every program and faculty candidate (both are scarce), then fill the
    # remainder with course/person in the eval set's own 33:30 ratio.
    picked: list[dict] = []
    for t in ("program", "faculty_adjunct_aggregate"):
        picked += disjoint[t]
    remaining = max(n_train - len(picked), 0)
    n_course = min(len(disjoint["course"]), round(remaining * 33 / 63))
    n_person = min(len(disjoint["person"]), remaining - n_course)
    picked += rng.sample(disjoint["course"], n_course)
    picked += rng.sample(disjoint["person"], n_person)
    rng.shuffle(picked)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="12 queries, no cache written — verifies the path, not the data")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--pool", type=int, default=P_TRAIN)
    ap.add_argument("--refresh-candidates", action="store_true",
                    help="re-mint even if the cached fingerprint still matches")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    if not 1 <= args.pool <= P_MAX:
        raise SystemExit(f"--pool must be within rank_one_index's depth (1..{P_MAX})")

    t0 = time.time()
    checks: list[tuple[str, bool, str]] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_eval, eval_entities, eval_queries = _load_eval_set()
    pool, how = mint_candidates(args.refresh_candidates)
    print(f"  candidates: {how} — "
          + ", ".join(f"{t} {len(c)}" for t, c in pool.items()), file=sys.stderr)

    # ---- T7 first: does this generator reproduce the SHIPPED qrels? ---------
    by_entity = {
        (c["entity_type"], c["entity"]): c["relevant_resolution_ids"]
        for cs in pool.values() for c in cs
    }
    tags = bgc._load_json(bgc.TAGS_DIR / "people_by_file.json")
    alias_index = bgc._build_person_alias_index(bgc._load_json(bgc.DICT_DIR / "people.json"))
    exact, drift, unexplained = 0, [], []
    for d in raw_eval:
        got = by_entity.get((d["entity_type"], d["entity"]))
        if got is not None and sorted(got) == sorted(d["relevant_resolution_ids"]):
            exact += 1
            continue
        why = explain_qrels_drift(d, got, tags, alias_index)
        (drift if why else unexplained).append(f"{d['entity']}: {why or 'unexplained'}")
    checks.append((
        "T7 the label generator reproduces the shipped eval qrels",
        not unexplained,
        f"{exact} exact, {len(drift)} explained by stale tag OCR, "
        f"{len(unexplained)} unexplained, of {len(raw_eval)} eval entities"
        + (f" — {'; '.join(unexplained[:3])}" if unexplained else ""),
    ))
    for line in drift:
        print(f"  [T7 drift] {line}", file=sys.stderr)

    render_queries(pool)
    train = choose_training_queries(pool, eval_entities, eval_queries, args.n_train)
    if args.smoke:
        train = train[:12]
    print(f"  {len(train)} training queries: "
          f"{dict(Counter(c['entity_type'] for c in train))}", file=sys.stderr)

    checks.append((
        "T1 no training entity appears in the eval set",
        not ({c["entity"] for c in train} & eval_entities),
        f"{len({c['entity'] for c in train} & eval_entities)} overlapping of {len(train)}",
    ))
    checks.append((
        "T2 no training query string appears in the eval set",
        not ({c["query"] for c in train} & eval_queries),
        f"{len({c['query'] for c in train} & eval_queries)} overlapping of {len(train)}",
    ))

    # ---- route + retrieve, one index resident at a time ---------------------
    indices = discover_indices(INDEX_ROOT)
    targets = route_targets("hybrid")
    resolved = {r: resolve_index(t, indices) for r, t in targets.items()}
    combo_of = {r: i.combo_id for r, i in resolved.items()}
    dir_of = {i.combo_id: Path(i.dir) for i in resolved.values()}
    # WHICH indices the pools came from, recorded so the pools and the
    # checkpoint trained on them can name their source. `input_fingerprint()`
    # deliberately covers only the LABEL side (dicts/tags/manifests), because
    # that is all a minted candidate's labels depend on -- but a *pool* is a
    # retrieval result, so an index rebuild stales `train_pools.json` while
    # every fingerprint field stays put. Rebuild #4 was exactly that case: the
    # 2026-08-12 pools survived it looking current. `docset_hash` comes from
    # `Index.provenance`, the same field `E0` uses to attribute a result to its
    # index (see CLAUDE.md) -- identify the artifact, never rename it.
    # Read from the index's own manifest, NOT from `IndexRef.provenance`:
    # `discover_indices` returns a lightweight reference and `provenance` is
    # stamped only by `ArtifactStore.load`, so it is `{}` here -- recording it
    # would have written four rows of None and looked like provenance.
    index_provenance = {}
    for ix in resolved.values():
        mf = Path(ix.dir) / "manifest.json"
        m = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
        index_provenance[ix.combo_id] = {
            "dir": str(Path(ix.dir).relative_to(REPO)).replace("\\", "/"),
            "docset_hash": m.get("docset_hash"),
            "n_resolutions": m.get("n_resolutions"),
            "built_at": m.get("timestamp"),
        }

    route_of = {c["query"]: classify_query(c["query"]) for c in train}
    misrouted = [c for c in train if route_of[c["query"]] != EXPECTED_ROUTE[c["entity_type"]]]
    rate = len(misrouted) / max(len(train), 1)
    checks.append((
        "T8 a training query routes where its own entity type routes",
        rate <= MAX_MISROUTE_RATE,
        f"{len(misrouted)} misrouted of {len(train)} ({rate:.1%}, bar {MAX_MISROUTE_RATE:.0%}) — "
        + (str(dict(Counter(
            f"{c['entity_type']}→{route_of[c['query']]}" for c in misrouted))) or "{}"),
    ))
    # Dropped, not merely counted: a misrouted training query's pool comes from
    # an index no eval query of its type ever reaches, which is exactly the
    # confound (a) exists to remove.
    train = [c for c in train if route_of[c["query"]] == EXPECTED_ROUTE[c["entity_type"]]]

    by_combo: dict[str, list[str]] = defaultdict(list)
    for c in train:
        by_combo[combo_of[route_of[c["query"]]]].append(c["query"])
    print(f"  {len(by_combo)} routed indices  "
          f"routes={dict(Counter(route_of[c['query']] for c in train))}", file=sys.stderr)

    checks.append((
        "T3 every training query routes to a built index",
        set(by_combo) <= set(dir_of),
        f"{len(by_combo)} combos, {len(set(by_combo) - set(dir_of))} unresolved",
    ))

    records: list[dict] = []
    qrels_of = {c["query"]: set(c["relevant_resolution_ids"]) for c in train}
    meta_of = {c["query"]: c for c in train}
    short_pool = 0
    for combo, qs in by_combo.items():
        top, _h, cid, rid, _page, text = rank_one_index(dir_of[combo], qs, True)
        for q in qs:
            rows = [int(i) for i in top[q][: args.pool]]
            short_pool += int(len(rows) < args.pool)
            rel = qrels_of[q]
            c = meta_of[q]
            records.append({
                "query": q,
                "entity": c["entity"],
                "entity_type": c["entity_type"],
                "route": route_of[q],
                "combo": combo,
                "n_relevant": len(rel),
                "candidates": [{
                    "chunk_id": cid[i],
                    "resolution_id": rid[i],
                    "rank": r + 1,
                    "label": int(rid[i] in rel),
                    "text": text[i],
                } for r, i in enumerate(rows)],
            })
        print(f"  {combo}  {len(qs)} queries  {time.time()-t0:.0f}s", file=sys.stderr)

    checks.append((
        f"T6 every pool holds {args.pool} candidates",
        short_pool == 0, f"{short_pool} short of {len(records)}",
    ))

    # ---- drop the queries a group-softmax loss cannot learn from ------------
    kept = [r for r in records if any(c["label"] for c in r["candidates"])]
    dropped = len(records) - len(kept)
    checks.append((
        "T5 queries whose pool holds no gold are dropped and counted",
        len(kept) > 0,
        f"{dropped} dropped of {len(records)}; {len(kept)} usable",
    ))

    rng = random.Random(SEED)
    rng.shuffle(kept)
    n_dev = max(1, round(len(kept) * DEV_FRACTION))
    for i, r in enumerate(kept):
        r["split"] = "dev" if i < n_dev else "train"
    checks.append((
        "T4 the dev slice is disjoint from the train slice",
        not ({r["query"] for r in kept if r["split"] == "dev"}
             & {r["query"] for r in kept if r["split"] == "train"}),
        f"dev {n_dev}, train {len(kept) - n_dev}",
    ))

    n_pos = sum(c["label"] for r in kept for c in r["candidates"])
    n_pairs = sum(len(r["candidates"]) for r in kept)
    meta = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fingerprint": input_fingerprint(),
        "pool_depth": args.pool,
        "n_queries_requested": args.n_train,
        "n_misrouted_dropped": len(misrouted),
        "qrels_reproduction": {
            "exact": exact, "explained_stale_tag_ocr": len(drift),
            "unexplained": len(unexplained), "of": len(raw_eval), "drift": drift,
        },
        "n_queries_retrieved": len(records),
        "n_queries_usable": len(kept),
        "n_dropped_no_gold_in_pool": dropped,
        "n_dev": n_dev,
        "n_pairs": n_pairs,
        "n_positive": n_pos,
        "positive_rate": round(n_pos / max(n_pairs, 1), 4),
        "type_mix": dict(Counter(r["entity_type"] for r in kept)),
        "route_mix": dict(Counter(r["route"] for r in kept)),
        "combos": sorted(by_combo),
        "index_provenance": index_provenance,
        "seed": SEED,
        "min_hits": MIN_HITS,
        "eval_set": str(GOLD.relative_to(REPO)).replace("\\", "/"),
    }

    if args.smoke:
        # A cache holding only the smoke subset would silently poison a later
        # training run, and every check here would pass on it.
        print("  [smoke] train_pools.json NOT written", file=sys.stderr)
    else:
        OUT_POOLS.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
        OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        REPORT.write_text(render_report(meta, checks), encoding="utf-8")

    print()
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\n{time.time()-t0:.0f}s")
    return 0 if all(ok for _, ok, _ in checks) else 1


def render_report(meta: dict, checks) -> str:
    L = [
        "# ข้อมูลเทรน reranker บน hybrid-fused candidates (follow-up (a))",
        "",
        "Generated by `tools/eval/build_reranker_training_data.py` · "
        "ออกแบบไว้ล่วงหน้าที่ `docs/reranker-trained-on-hybrid-design.md`",
        "",
        "คำถามเทรนสร้างด้วย **generator ตัวเดียวกับ qrels ของชุด eval** "
        "(`tools/corpus_prep/build_gold_candidates.py`) แต่บน entity ที่ "
        "**ไม่ทับกับ 106 คำถามที่ใช้วัด** · candidate ของแต่ละคำถามคือ "
        f"routed hybrid top-{meta['pool_depth']} — pool เดียวกับที่โมเดลจะเจอตอน eval "
        "ซึ่งคือสาระทั้งหมดของ intervention (a)",
        "",
        "| | |",
        "|---|---|",
        f"| คำถามที่ขอ | {meta['n_queries_requested']} |",
        f"| ตัดทิ้ง (route ไม่ตรงชนิด entity) | {meta['n_misrouted_dropped']} |",
        f"| ดึง pool สำเร็จ | {meta['n_queries_retrieved']} |",
        f"| ใช้ได้จริง | **{meta['n_queries_usable']}** |",
        f"| ตัดทิ้ง (pool ไม่มี gold เลย) | {meta['n_dropped_no_gold_in_pool']} |",
        f"| กันไว้เป็น dev (เลือก checkpoint) | {meta['n_dev']} |",
        f"| คู่ (query, chunk) ทั้งหมด | {meta['n_pairs']:,} |",
        f"| เป็น positive | {meta['n_positive']:,} ({meta['positive_rate']:.1%}) |",
        "",
        "## สัดส่วนตามชนิด entity",
        "",
        "| ชนิด | จำนวนคำถาม |",
        "|---|---|",
    ]
    for t, n in sorted(meta["type_mix"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {t} | {n} |")
    L += [
        "",
        "`faculty_adjunct_aggregate` แทบไม่มี candidate ที่ไม่ทับชุด eval เหลืออยู่เลย "
        "ทั้งคลัง — 13 ใน 106 คำถามที่วัด (12%) จึงมาจาก route ที่โมเดลแทบไม่เคยเทรน "
        "**นี่เป็นข้อจำกัดของคลังข้อมูล ไม่ใช่ตัวเลือก** และต้องรายงานแยก route ไม่ใช่เฉลี่ยรวม",
        "",
        "## คำถามเทรนถูกเขียนใหม่ให้เหมือนชุด eval",
        "",
        "template ของ generator เองไม่ตรงกับที่ `gold_query_set_73det.yaml` ใช้จริง "
        "(ชุด eval เขียนด้วยมือ และเอ่ยชื่อ entity **เต็ม**) — วัดได้ว่า course ตรงกัน "
        "33/33 แต่ program ต่างกัน 30/30, person 29/30, faculty 13/13 "
        "**และผลไม่ใช่แค่รูปประโยค**: ด้วยถ้อยคำของ generator เอง program candidate ที่ไม่ทับ "
        "ชุด eval ทั้ง 117 รายการถูกจัดเป็น `unmatched` ทั้งหมด pool ที่ได้จึงจะมาจากดัชนี "
        "คนละตัวกับที่คำถาม program ในชุด eval ไปถึง — เท่ากับเอา (a) ไปปนกับความไม่เข้ากัน "
        "ของรูปคำถาม ซึ่งเป็นสิ่งเดียวกับที่ (a) ตั้งใจจะกำจัด · `T8` จึงตรวจ route ด้วยเครื่อง",
        "",
        "## qrels ของชุด eval ยังสร้างซ้ำได้หรือไม่",
        "",
        f"`T7` เทียบ generator วันนี้กับ qrels ที่ตีพิมพ์: **{meta['qrels_reproduction']['exact']} "
        f"ตรงเป๊ะ**, {meta['qrels_reproduction']['explained_stale_tag_ocr']} "
        f"อธิบายได้ด้วย tag cache เก่า, {meta['qrels_reproduction']['unexplained']} "
        f"อธิบายไม่ได้ จาก {meta['qrels_reproduction']['of']} entity",
        "",
        "ที่ต่างกันมาจากกลไกเดียว และ **ของที่เก่าคือ tag cache ไม่ใช่ qrels**: "
        "`people_by_file.json` เป็น snapshot 2026-07-25 ที่บันทึกนามสกุล "
        "*ตามที่ OCR อ่านผิดในตอนนั้น* (เช่น `โฆษชุณหันท์`) พอ remediation 07-28 / "
        "re-OCR 08-09 แก้ตัวสะกดในคลังเป็น `โฆษชุณหนันท์` `person_candidates` "
        "ซึ่งเป็น generator ตัวเดียวที่กลับไปอ่านตัวบทจริงจึงหาไม่เจอ ป้ายกำกับเลยหายเงียบ ๆ — "
        "คนคนนั้นถูกเอ่ยชื่อในเอกสารจริง ที่ล้าสมัยคือ *ตัวสะกดที่แคชไว้* "
        "(ดู `programs_by_file.json` ที่ค้างเรื่องเดียวกันใน CLAUDE.md)",
    ]
    for line in meta["qrels_reproduction"]["drift"]:
        L.append(f"- {line}")
    L += [
        "",
        "## self-check",
        "",
    ]
    for name, ok, detail in checks:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")
    L += [
        "",
        f"สร้างเมื่อ {meta['at']} · seed={meta['seed']} · min_hits={meta['min_hits']} · "
        f"fingerprint={meta['fingerprint']}",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
