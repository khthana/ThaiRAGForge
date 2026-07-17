"""Query-time routing: classify an incoming query by shape (person-history,
program-history, or unmatched) and pick which (chunker, embedder) combo to
query against, informed by the Gold-eval finding that no single combo wins
universally -- program-history favors one combo, person-history favors
another (see docs/chunker-embedder-comparison-log.md and the embedder-axis
follow-up).

Classification deliberately reuses two *different* strategies per axis,
because the two entity types have different staleness properties:

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

Everything here is pure classification/fusion logic (no I/O beyond reading
the already-loaded program dictionary) -- Streamlit-free per ADR-0001.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from rag_lab.loaders.person_loader import match_people
from rag_lab.loaders.program_loader import load_dictionary, match_programs
from rag_lab.schema import RankedChunk, RetrievalResult

# Present in essentially every canonical program name's template
# ("หลักสูตร<degree> สาขาวิชา<field>") -- a program not yet in programs.json
# (freshly approved, dictionary not yet rebuilt) still carries this marker.
_PROGRAM_FALLBACK = re.compile(r"สาขาวิชา")

ROUTE_PERSON = "person"
ROUTE_PROGRAM = "program"
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


# Best-performing (chunker, embedder) combo per route, from the Gold-eval
# breakdown (tools/eval/gold_embedder_breakdown.py): person peaks under
# semantic+bge-m3 (recall@10 0.7504), program peaks under
# sentence+phayathaibert-congen (0.6081). "unmatched" has no clear winner in
# the per-category data, so it defaults to bge-m3 (the most balanced embedder
# across every chunker) on fixed_size (a reasonable, cheap default chunker).
ROUTE_COMBO: dict[str, RouteTarget] = {
    ROUTE_PERSON: RouteTarget("semantic", "local", "BAAI/bge-m3"),
    ROUTE_PROGRAM: RouteTarget("sentence", "local", "kornwtp/ConGen-BGE_M3-model-phayathaibert"),
    ROUTE_UNMATCHED: RouteTarget("fixed_size", "local", "BAAI/bge-m3"),
}


def classify_query(query: str) -> str:
    """Classify `query` as person-shaped, program-shaped, or unmatched.

    Person is checked first: a query naming a titled person almost never
    also happens to name a specific program, and the person pattern is the
    more precise signal (rank + name is a strong anchor; the program
    fallback is a single common substring)."""
    if match_people(query):
        return ROUTE_PERSON
    if match_programs(query, dictionary=load_dictionary()):
        return ROUTE_PROGRAM
    if _PROGRAM_FALLBACK.search(query):
        return ROUTE_PROGRAM
    return ROUTE_UNMATCHED


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
