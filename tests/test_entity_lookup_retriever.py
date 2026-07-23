"""EntityLookupRetriever: exhaustive retrieval over metadata, bypassing
top-k ranking entirely. Tested against a hand-built multi-resolution Index
with an injected fake detector -- no dependency on the real regex/dictionary
matchers or a built corpus."""
from __future__ import annotations

import numpy as np

from rag_lab.retrievers.entity_lookup import EntityLookupRetriever
from rag_lab.schema import Chunk, Index, Query


def _index():
    # res-1 has two chunks both tagged with the target program -- exercises
    # dedup-to-one-per-resolution (ADR-0002).
    metadatas = [
        {"programs": ["หลักสูตรบริหารธุรกิจบัณฑิต"]},
        {"programs": ["หลักสูตรบริหารธุรกิจบัณฑิต"]},
        {"programs": ["หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา"]},
        {"programs": []},
    ]
    resolution_ids = ["res-1", "res-1", "res-2", "res-3"]
    chunks = [
        Chunk(chunk_id=f"c{i}", resolution_id=rid, text=f"t{i}", chunk_index=0, page=1, metadata=m)
        for i, (rid, m) in enumerate(zip(resolution_ids, metadatas))
    ]
    embeddings = np.zeros((len(chunks), 2))
    lexical = [[] for _ in chunks]
    return Index(chunks=chunks, embeddings=embeddings, meta={}, lexical=lexical)


def _query(text: str = "q") -> Query:
    return Query(text=text, vector=None, tokens=None)


def test_returns_every_matching_resolution_deduped_to_one_chunk_each():
    retriever = EntityLookupRetriever(detector=lambda q: {"programs": ["หลักสูตรบริหารธุรกิจบัณฑิต"]})

    results = retriever.retrieve(_query(), _index(), k=10)

    assert [r.resolution_id for r in results] == ["res-1"]


def test_uniform_score_and_monotonic_rank():
    retriever = EntityLookupRetriever(
        detector=lambda q: {"programs": ["หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา"]}
    )

    [result] = retriever.retrieve(_query(), _index(), k=10)

    assert result.score == 1.0
    assert result.rank == 1


def test_k_is_ignored_all_matches_returned():
    # both res-1 chunks share the same program tag -- only res-1 exists as
    # a target here so k's normal "slice to this many" role would be a
    # no-op regardless; the real point is retrieve() never raises/truncates
    # based on k for this retriever (BaseRetriever.exhaustive = True)
    retriever = EntityLookupRetriever(detector=lambda q: {"programs": ["หลักสูตรบริหารธุรกิจบัณฑิต"]})

    results = retriever.retrieve(_query(), _index(), k=0)

    assert len(results) == 1


def test_empty_detection_returns_empty_list():
    retriever = EntityLookupRetriever(detector=lambda q: {})

    assert retriever.retrieve(_query(), _index(), k=10) == []


def test_exhaustive_flag_is_true():
    assert EntityLookupRetriever().exhaustive is True


def test_name_property():
    assert EntityLookupRetriever().name == "entity_lookup"
