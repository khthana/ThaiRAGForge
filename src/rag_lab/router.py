"""Query-time routing: classify an incoming query by shape (person, course,
program, faculty, or unmatched) and pick which (chunker, embedder) combo to
query against, informed by the Gold-eval finding that no single combo wins
universally -- program-history favors one combo, person-history favors
another (see docs/chunker-embedder-comparison-log.md and the embedder-axis
follow-up).

Classification deliberately reuses *different* strategies per axis, because
the entity types have different staleness properties:

- Person: `match_people` (loaders/person_loader.py) is a live regex over
  academic rank + name, not a dictionary lookup -- applying it to the query
  text itself means a newly-added person is classified correctly with zero
  dependency on any snapshot. This is the actual fix for the staleness
  concern raised when the routing idea first came up.
- Program: canonical program names have no equivalent structural marker, so
  `match_programs` genuinely needs `programs.json`. New programs are a much
  rarer, deliberate event (curriculum approval) than new people joining, so
  dictionary staleness is a smaller risk here -- but not zero, so a
  lower-confidence structural fallback (`สาขาวิชา` -- present in essentially
  every canonical program's full name template) still routes a
  not-yet-catalogued program correctly rather than falling through to
  "unmatched".
- Course: half live pattern, half dictionary -- `match_courses` reads the
  8-digit code straight out of the text (no staleness at all), and
  `match_courses_by_name` covers the far commoner case of a user naming the
  course by its title, which does need `courses.json`. A brand-new course
  with neither its code nor a catalogued name falls to "unmatched", which is
  the right failure: there is nothing to anchor on.
- Faculty: dictionary-only. Acceptable here in a way it would not be
  elsewhere, because faculties are ~20 stable institutional units that change
  on a timescale of years, not a growing list.

Everything here is pure classification/fusion logic (no I/O beyond reading
the already-loaded dictionaries) -- Streamlit-free per ADR-0001.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from rag_lab.loaders.course_loader import match_courses, match_courses_by_name
from rag_lab.loaders.faculty_loader import match_faculties
from rag_lab.loaders.person_loader import match_people, match_people_by_dictionary
from rag_lab.loaders.program_loader import load_dictionary, match_programs
from rag_lab.schema import RankedChunk, RetrievalResult

# Present in essentially every canonical program name's template
# ("หลักสูตร<degree> สาขาวิชา<field>") -- a program not yet in programs.json
# (freshly approved, dictionary not yet rebuilt) still carries this marker.
_PROGRAM_FALLBACK = re.compile(r"สาขาวิชา")

# match_people (loaders/person_loader.py) requires zero space between a
# title and the name -- a deliberate choice tuned to how this corpus's own
# tables/prose actually write it. A user typing a query naturally writes
# "ผศ. ธนา..." with a space, though, so collapse that spacing here (query
# classification only -- corpus tagging via PersonLoader is untouched)
# before handing off to the same regex.
_TITLE_TRAILING_SPACE = re.compile(
    r"(ผู้ช่วยศาสตราจารย์|รองศาสตราจารย์|ศาสตราจารย์|ผศ\.|รศ\.|ศ\.|ดร\.)\s+(?=[ก-ฮ])"
)


def _collapse_title_spacing(query: str) -> str:
    return _TITLE_TRAILING_SPACE.sub(r"\1", query)

ROUTE_PERSON = "person"
ROUTE_PROGRAM = "program"
ROUTE_COURSE = "course"
ROUTE_FACULTY = "faculty"
ROUTE_UNMATCHED = "unmatched"


@dataclass(frozen=True)
class RouteTarget:
    """Which built index a route should query: matched against an
    IndexInfo's chunker.type / embedder.type / embedder.params['model_name']
    (see query_service.resolve_index). `embedder_model_name=None` matches any
    model under that embedder type (e5 only has one variant in this repo)."""

    chunker_type: str
    embedder_type: str
    embedder_model_name: str | None = None


# Best-performing (chunker, embedder) combo per route. Keyed by RETRIEVER
# first, because the best target for a route is retriever-dependent and a
# single flat dict silently served whichever retriever it was picked under.
# Evidence for every entry: `tools/eval/routing_eval.py` ->
# `data/results/routing_eval.md`, section 2, run per retriever on the
# 106-query 73det Gold set.
#
# Adoption rule used below, so a future refresh has a criterion rather than an
# argmax: take the scan's best combo only when the leave-one-out selector
# picks that same target in essentially every fold (>=29/30). One distinct LOO
# target means the choice does not hinge on any single query, which is what
# makes adopting it a *refresh* rather than a fit to the eval set. Where the
# LOO selector wavers, the incumbent stays even if it is numerically behind.
#
# Consequence to state whenever these numbers are cited: after this refresh
# `routed (shipped)` is chosen ON the 106 queries it is scored on, so it is
# no longer an independent arm -- it now sits near `routed (oracle)` by
# construction (dense 0.6189 vs 0.6293, hybrid 0.6831 vs 0.6868 recall@10).
# **The honest generalisation estimate is `routed (loo)`**, and that arm is
# unchanged by this refresh (+0.0349 dense / +0.0499 hybrid vs the best single
# combo, both ns) because it never read these constants in the first place.
# So the refresh raises the shipped router to what LOO already predicted; it
# does not create new gain.
#
# course -- shipped target is already the argmax under both retrievers (gap
# +0.0000), so nothing to do. It was chosen 2026-08-08 on an embedder-level
# effect, not a lucky cell: qwen3-0.6B leads course under *all four* chunkers
# (0.5105-0.5759) while the next embedder tops out at 0.5114 and bge-m3
# reaches only 0.2681. The chunker within qwen3-0.6B is a near-tie, settled by
# the project's one significance-backed chunker result (recursive is the only
# chunker ever shown to beat another) rather than by argmax.
#
# faculty -- STAYS at the unmatched default under both retrievers, and this is
# the rule above doing its job rather than an oversight. Dense wants
# fixed_size+e5 (+0.0852) but the LOO selector picks 3 different targets
# across 13 folds; hybrid wants recursive+e5_small on a gap of only +0.0305.
# n=13 is inside the embedder family's own MDE (~0.05-0.10). "Checked,
# nothing stable enough" is recorded here in code rather than in a doc.
#
# person / program -- REFRESHED 2026-08-08. Both were picked 2026-07-17 from
# the retired 252-query set and both are beaten on the 73det set under both
# retrievers, unanimously across folds: person 30/30 under each, program 30/30
# dense and 29/30 hybrid. `sentence+congen` in particular was not merely
# stale but actively harmful under hybrid -- routing `program` to it scored
# 0.5321 where NOT routing at all scored 0.6105, i.e. -0.0784 on those 30
# queries, which was most of why hard routing netted out at only +0.0101
# overall (`tools/eval/soft_vs_hard_routing.py`). Re-running that script after
# this refresh flipped its headline verdict: `program` now scores 0.6545
# routed (+0.0440 for routing instead of -0.0784), hard routing wins all four
# routes, and it overtakes soft (per-route alpha) routing. Any number quoted
# from that script predating 2026-08-08 is measuring the stale targets above.
#
# unmatched -- unchanged and still unmeasured: 0/106 Gold queries reach it
# since the course/faculty routes landed, so there is no per-route evidence
# for it either way. bge-m3 on fixed_size is a balanced, cheap default.
_ROUTE_COMBO_DENSE: dict[str, RouteTarget] = {
    ROUTE_PERSON: RouteTarget("semantic", "qwen3", "Qwen/Qwen3-Embedding-4B"),
    ROUTE_PROGRAM: RouteTarget("fixed_size", "qwen3", "Qwen/Qwen3-Embedding-0.6B"),
    ROUTE_COURSE: RouteTarget("recursive", "qwen3", "Qwen/Qwen3-Embedding-0.6B"),
    ROUTE_FACULTY: RouteTarget("fixed_size", "local", "BAAI/bge-m3"),
    ROUTE_UNMATCHED: RouteTarget("fixed_size", "local", "BAAI/bge-m3"),
}

_ROUTE_COMBO_HYBRID: dict[str, RouteTarget] = {
    ROUTE_PERSON: RouteTarget("sentence", "local", "BAAI/bge-m3"),
    ROUTE_PROGRAM: RouteTarget("semantic", "qwen3", "Qwen/Qwen3-Embedding-0.6B"),
    ROUTE_COURSE: RouteTarget("recursive", "qwen3", "Qwen/Qwen3-Embedding-0.6B"),
    ROUTE_FACULTY: RouteTarget("fixed_size", "local", "BAAI/bge-m3"),
    ROUTE_UNMATCHED: RouteTarget("fixed_size", "local", "BAAI/bge-m3"),
}

ROUTE_COMBO_BY_RETRIEVER: dict[str, dict[str, RouteTarget]] = {
    "dense": _ROUTE_COMBO_DENSE,
    "hybrid": _ROUTE_COMBO_HYBRID,
}

# The map used for any retriever the routing eval has not covered (bm25,
# entity_lookup, entity_boost, qdrant). Hybrid rather than dense because
# hybrid is this project's best-measured system, but note it IS an
# extrapolation for those retrievers, not a measurement -- `routing_eval.py`
# only runs the dense and hybrid arms.
ROUTE_COMBO: dict[str, RouteTarget] = _ROUTE_COMBO_HYBRID


def route_targets(retriever_type: str | None = None) -> dict[str, RouteTarget]:
    """The route -> RouteTarget map to use with `retriever_type`.

    Falls back to `ROUTE_COMBO` for retrievers with no measured map of their
    own, so an unknown retriever routes sensibly instead of raising -- the
    cost of being wrong here is a slightly worse index, and a KeyError at
    query time would be worse than that."""
    return ROUTE_COMBO_BY_RETRIEVER.get(retriever_type or "", ROUTE_COMBO)


def _default_program_matcher(text: str) -> list[str]:
    return match_programs(text, dictionary=load_dictionary())


def _default_course_matcher(text: str) -> list[str]:
    return sorted(set(match_courses(text)) | set(match_courses_by_name(text)))


def classify_query(query: str) -> str:
    """Classify `query` as person-, course-, program- or faculty-shaped, else
    unmatched.

    Order is most-precise-signal-first, so a query carrying two signals lands
    on the better-anchored one:

    - Person leads: a query naming a titled person almost never also names a
      specific program, and rank+name is a strong anchor.
    - Course next, ahead of *both* program branches. An 8-digit course code
      (or an exact unique course title) is a far tighter match than the
      program fallback's bare `สาขาวิชา` substring, and misrouting a course
      query into the program route is the single most expensive mistake this
      classifier can make: the program route's ConGen embedder scores
      **0.0000** recall@10 on course queries (it never saw Latin-script
      course titles), versus 0.5759 on the course route.
    - Faculty last, since its dictionary entries ("คณะ..."/"วิทยาลัย...") are
      common enough to appear incidentally inside a program or course query.

    Verified on the 106-query 73det Gold set: 30/30 person, 30/30 program,
    33/33 course, 13/13 faculty, with zero cross-firing in either direction
    (no non-course query matches a course, no non-faculty query a faculty)."""
    if match_people(_collapse_title_spacing(query)):
        return ROUTE_PERSON
    if _default_course_matcher(query):
        return ROUTE_COURSE
    if match_programs(query, dictionary=load_dictionary()):
        return ROUTE_PROGRAM
    if _PROGRAM_FALLBACK.search(query):
        return ROUTE_PROGRAM
    if match_faculties(query):
        return ROUTE_FACULTY
    return ROUTE_UNMATCHED


def detect_entities(
    query: str,
    *,
    people_matcher=match_people,
    people_dict_matcher=match_people_by_dictionary,
    program_matcher=_default_program_matcher,
    course_matcher=_default_course_matcher,
    faculty_matcher=match_faculties,
) -> dict[str, list[str]]:
    """kind -> canonical values actually found in `query` (empty kinds
    omitted). Used by the entity-lookup retrieval mode and the keyword/
    filter-boost pre-filter (retrievers/entity_lookup.py,
    retrievers/filters.py's EntityFilter, query_service.py) -- a different
    question from classify_query's single route label, so this is a sibling
    function, not an extension of it.

    Person: titled match_people first (query classification collapses
    title-trailing spacing the same way classify_query does), falling back
    to the untitled dictionary matcher since a user typing a search query
    usually won't include an academic rank. People are keyed on
    'given_name surname' (title stripped) to match EntityFilter's key, so a
    no-title query match still matches a title-anchored corpus tag.

    Faculty: `match_faculties` is dictionary-based like match_programs, no
    query-side fallback needed -- Gold-eval spot check (2026-07-25, the
    faculty_adjunct_aggregate entity_type) found every real query names a
    faculty/college by its exact canonical text, 13/13 detected unmodified.

    Course: `match_courses` (8-digit code) OR `match_courses_by_name`
    (course_loader.py's unique-name dictionary) -- a real user names a
    course by its title, not its code, so the code-only matcher alone
    (sufficient for corpus tagging) would never fire on a realistic query.

    Matchers are injectable so tests can substitute fakes (e.g. for courses)
    without depending on the real regex/dictionary."""
    people = people_matcher(_collapse_title_spacing(query)) or people_dict_matcher(query)
    programs = program_matcher(query)
    courses = course_matcher(query)
    faculties = faculty_matcher(query)

    detected: dict[str, list[str]] = {}
    if people:
        detected["people"] = sorted({f"{p['given_name']} {p['surname']}" for p in people})
    if programs:
        detected["programs"] = list(programs)
    if courses:
        detected["courses"] = list(courses)
    if faculties:
        detected["faculties"] = list(faculties)
    return detected


def rrf_merge(
    results: list[RetrievalResult], k_rrf: int = 60, top_k: int = 10,
    combination_id: str = "rrf",
) -> RetrievalResult:
    """Reciprocal Rank Fusion over already-ranked RetrievalResults, deduped
    to one entry per resolution_id (ADR-0002: relevance is judged at the
    Resolution level, so a merged ranking shouldn't waste top-k slots on the
    same resolution appearing via two different chunks/combos).

    score(resolution_id) = sum over each result of 1 / (k_rrf + rank).
    The representative chunk kept for a resolution_id is whichever had the
    best (lowest) rank across the merged results, purely for display."""
    scores: dict[str, float] = defaultdict(float)
    best_chunk: dict[str, RankedChunk] = {}
    query = results[0].query if results else ""
    for r in results:
        for rc in r.results:
            scores[rc.resolution_id] += 1.0 / (k_rrf + rc.rank)
            if rc.resolution_id not in best_chunk or rc.rank < best_chunk[rc.resolution_id].rank:
                best_chunk[rc.resolution_id] = rc

    ranked_ids = sorted(scores, key=lambda rid: -scores[rid])[:top_k]
    merged = [
        RankedChunk(
            chunk_id=best_chunk[rid].chunk_id,
            resolution_id=rid,
            page=best_chunk[rid].page,
            score=scores[rid],
            rank=i,
            text=best_chunk[rid].text,
        )
        for i, rid in enumerate(ranked_ids, start=1)
    ]
    return RetrievalResult(
        query=query, combination_id=combination_id, results=merged, top_k=top_k, retriever="rrf",
    )
