"""Cycles B–C — BM25Retriever (lexical ranking over stored per-chunk tokens)."""
from __future__ import annotations

import numpy as np
import pytest

from rag_lab.config import StrategySpec
from rag_lab.factory import build_retriever
from rag_lab.schema import Chunk, Index, Query


def _chunk(i: int) -> Chunk:
    return Chunk(chunk_id=f"c{i}", resolution_id=f"r{i}", text=f"t{i}", chunk_index=i, page=1)


def test_bm25_ranks_by_token_overlap():
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    lexical = [
        ["ค่าธรรมเนียม", "การศึกษา"],
        ["หลักสูตร", "วิศวกรรม"],
        ["การศึกษา", "ทั่วไป"],
    ]
    index = Index(chunks=chunks, embeddings=np.zeros((3, 1)), meta={}, lexical=lexical)

    ranked = build_retriever(StrategySpec(type="bm25")).retrieve(
        Query(text="ค่าธรรมเนียม", tokens=["ค่าธรรมเนียม"]), index, k=3
    )

    assert ranked[0].chunk_id == "c0"  # only c0 carries the query term


def test_bm25_without_lexical_index_raises():
    index = Index(chunks=[_chunk(0)], embeddings=np.zeros((1, 1)), meta={})  # lexical=None
    with pytest.raises(ValueError):
        build_retriever(StrategySpec(type="bm25")).retrieve(
            Query(text="x", tokens=["x"]), index, k=1
        )


def _lexical_index() -> Index:
    lexical = [
        ["ค่าธรรมเนียม", "การศึกษา"],
        ["หลักสูตร", "วิศวกรรม"],
        ["การศึกษา", "ทั่วไป"],
    ]
    return Index(
        chunks=[_chunk(0), _chunk(1), _chunk(2)],
        embeddings=np.zeros((3, 1)),
        meta={},
        lexical=lexical,
    )


def test_bm25_scorer_is_built_once_per_index():
    # The scorer walks the whole corpus to build its IDF table, which cost ~26x
    # a single get_scores call on the real index -- so paying it per query put a
    # fixed corpus-sized tax on every BM25 and hybrid retrieval. Two queries
    # against one Index must share one scorer object.
    index = _lexical_index()
    retriever = build_retriever(StrategySpec(type="bm25"))

    retriever.retrieve(Query(text="ค่าธรรมเนียม", tokens=["ค่าธรรมเนียม"]), index, k=3)
    first = index.lexical_scorer[1]
    retriever.retrieve(Query(text="หลักสูตร", tokens=["หลักสูตร"]), index, k=3)

    assert index.lexical_scorer[1] is first


def test_bm25_memo_survives_a_fresh_retriever_instance():
    # The memo lives on the Index, not the retriever, so it must still hit when
    # the caller builds a retriever per query (query_indices does) as long as
    # the same loaded Index is reused.
    index = _lexical_index()
    q = Query(text="ค่าธรรมเนียม", tokens=["ค่าธรรมเนียม"])

    build_retriever(StrategySpec(type="bm25")).retrieve(q, index, k=3)
    first = index.lexical_scorer[1]
    build_retriever(StrategySpec(type="bm25")).retrieve(q, index, k=3)

    assert index.lexical_scorer[1] is first


def test_bm25_rebuilds_when_the_token_list_is_replaced():
    # `Index.lexical` is a mutable field. Serving a memo built from different
    # tokens would score the wrong rows silently -- the failure shape this repo
    # keeps finding -- so the memo is keyed on the token list's identity.
    index = _lexical_index()
    retriever = build_retriever(StrategySpec(type="bm25"))
    retriever.retrieve(Query(text="ค่าธรรมเนียม", tokens=["ค่าธรรมเนียม"]), index, k=3)
    stale = index.lexical_scorer[1]

    index.lexical = [["หลักสูตร"], ["หลักสูตร"], ["ค่าธรรมเนียม"]]
    ranked = retriever.retrieve(Query(text="ค่าธรรมเนียม", tokens=["ค่าธรรมเนียม"]), index, k=3)

    assert index.lexical_scorer[1] is not stale
    assert ranked[0].chunk_id == "c2"  # scored against the NEW tokens, not the old


def test_sub_index_does_not_inherit_the_full_corpus_scorer():
    # Index.select builds a fresh Index, so a scorer derived from all rows can
    # never leak into a narrowed one -- that would make BM25 corpus-relative to
    # the wrong corpus, which is precisely what the class docstring promises not
    # to do.
    index = _lexical_index()
    build_retriever(StrategySpec(type="bm25")).retrieve(
        Query(text="ค่าธรรมเนียม", tokens=["ค่าธรรมเนียม"]), index, k=3
    )
    assert index.lexical_scorer is not None

    sub = index.select([1, 2])

    assert sub.lexical_scorer is None


def test_memo_does_not_change_scores_or_order():
    # The whole change is a timing change; a memoised run must be identical to
    # a cold one, score for score.
    q = Query(text="การศึกษา", tokens=["การศึกษา"])
    retriever = build_retriever(StrategySpec(type="bm25"))

    cold = retriever.retrieve(q, _lexical_index(), k=3)
    warm_index = _lexical_index()
    retriever.retrieve(q, warm_index, k=3)
    warm = retriever.retrieve(q, warm_index, k=3)

    assert [(r.chunk_id, r.score, r.rank) for r in cold] == [
        (r.chunk_id, r.score, r.rank) for r in warm
    ]
