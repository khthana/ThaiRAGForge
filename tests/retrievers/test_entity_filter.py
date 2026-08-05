"""EntityFilter restricts candidates by list-valued entity metadata
(metadata['people']/['programs']/['courses']) before any retriever ranks --
parallel to MetadataFilter, which is exact-equality-only and can't express
list-membership. Tested in isolation with hand-set chunk.metadata (no
loader / no build)."""
from __future__ import annotations

import numpy as np

from rag_lab.retrievers.filters import EntityFilter
from rag_lab.schema import Chunk, Index


def _index():
    metadatas = [
        {
            "people": [{"given_name": "ธนา", "surname": "หงษ์สุวรรณ", "title": "ผศ."}],
            "programs": ["หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา"],
        },
        {
            "people": [{"given_name": "สมชาย", "surname": "ใจดี", "title": "รศ."}],
            "programs": [],
        },
        {
            "people": [{"given_name": "ธนา", "surname": "หงษ์สุวรรณ", "title": "ดร."}],
            "programs": ["หลักสูตรบริหารธุรกิจบัณฑิต"],
        },
        {"programs": ["หลักสูตรบริหารธุรกิจบัณฑิต"]},  # no 'people' key at all
    ]
    chunks = [
        Chunk(chunk_id=f"c{i}", resolution_id=f"r{i}", text=f"t{i}", chunk_index=0, page=1, metadata=m)
        for i, m in enumerate(metadatas)
    ]
    embeddings = np.zeros((len(chunks), 2))
    lexical = [[] for _ in chunks]
    return Index(chunks=chunks, embeddings=embeddings, meta={}, lexical=lexical)


def test_matches_person_regardless_of_title_in_metadata():
    # a no-title query match (e.g. via match_people_by_dictionary) must
    # still match a title-anchored corpus tag ("ผศ." or "ดร.") -- the key
    # is given_name+surname, title stripped
    filtered = EntityFilter({"people": ["ธนา หงษ์สุวรรณ"]}).apply(_index())
    assert [c.chunk_id for c in filtered.chunks] == ["c0", "c2"]


def test_matches_string_valued_programs_by_exact_membership():
    filtered = EntityFilter({"programs": ["หลักสูตรบริหารธุรกิจบัณฑิต"]}).apply(_index())
    assert [c.chunk_id for c in filtered.chunks] == ["c2", "c3"]


def test_and_across_kinds_requires_both_to_match():
    filtered = EntityFilter(
        {"people": ["ธนา หงษ์สุวรรณ"], "programs": ["หลักสูตรบริหารธุรกิจบัณฑิต"]}
    ).apply(_index())
    assert [c.chunk_id for c in filtered.chunks] == ["c2"]


def test_missing_key_is_treated_as_empty_not_an_error():
    # chunk c3 has no 'people' key at all -- must not raise, must not match
    filtered = EntityFilter({"people": ["ธนา หงษ์สุวรรณ"]}).apply(_index())
    assert "c3" not in [c.chunk_id for c in filtered.chunks]


def test_no_matches_returns_empty_index():
    filtered = EntityFilter({"people": ["ไม่มีใครชื่อนี้"]}).apply(_index())
    assert filtered.chunks == []
