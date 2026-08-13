"""Pin that the sparse decomposition reproduces `BM25Okapi.get_scores`.

The whole point of `bm25_sparse` is that the served lexical arm is the *measured*
one rather than whatever the engine's own BM25 computes, so the equality is the
contract -- not an implementation detail. Each rule that could silently break it
is pinned separately, because each of them still returns plausible rankings when
wrong.
"""
from __future__ import annotations

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from rag_lab.retrievers import bm25_sparse


# "มติ" sits in 4 of 5 documents on purpose: BM25Okapi's idf goes negative once a
# term is in more than half the corpus, and that is the branch the floor test below
# needs to be non-vacuous.
CORPUS = [
    ["มติ", "สภา", "วิชาการ", "อนุมัติ", "หลักสูตร"],
    ["มติ", "หลักสูตร", "วิศวกรรมศาสตรบัณฑิต", "สาขาวิชา", "ไฟฟ้า", "หลักสูตร"],
    ["มติ", "อนุมัติ", "ปริญญา", "แก่", "ผู้", "สำเร็จ", "การศึกษา"],
    ["มติ", "สภา", "วิชาการ", "รับรอง", "รายงาน", "การประชุม"],
    ["ไฟฟ้า", "ไฟฟ้า", "ไฟฟ้า"],
]


@pytest.fixture(scope="module")
def fitted():
    scorer = BM25Okapi(CORPUS)
    return scorer, bm25_sparse.build_vocabulary(scorer)


@pytest.mark.parametrize(
    "query",
    [
        ["หลักสูตร"],
        ["สภา", "วิชาการ"],
        ["ไฟฟ้า", "ไฟฟ้า"],  # repeated term: get_scores adds it twice
        ["อนุมัติ", "หลักสูตร", "ไฟฟ้า"],
        ["คำที่ไม่มีในคลัง"],  # unknown term contributes 0
        ["หลักสูตร", "คำที่ไม่มีในคลัง"],
    ],
)
def test_sparse_dot_reproduces_get_scores(fitted, query):
    scorer, vocab = fitted
    expected = scorer.get_scores(query)
    qvec = bm25_sparse.query_sparse_vector(query, vocab)
    got = np.array(
        [
            bm25_sparse.dot(bm25_sparse.document_sparse_vector(scorer, i, vocab), qvec)
            for i in range(scorer.corpus_size)
        ]
    )
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_repeated_query_term_is_counted_not_flattened(fitted):
    """The query weight is the term's count. Weighting every term 1.0 instead
    looks right on distinct-token queries and is wrong exactly when a term
    repeats -- which real Thai queries do."""
    scorer, vocab = fitted
    once = scorer.get_scores(["ไฟฟ้า"])
    twice = scorer.get_scores(["ไฟฟ้า", "ไฟฟ้า"])
    assert np.allclose(twice, 2.0 * once)

    indices, values = bm25_sparse.query_sparse_vector(["ไฟฟ้า", "ไฟฟ้า"], vocab)
    assert len(indices) == 1
    assert values == [2.0]


def test_unknown_query_term_is_dropped_not_defaulted(fitted):
    scorer, vocab = fitted
    indices, _ = bm25_sparse.query_sparse_vector(["ไม่เคยเห็น"], vocab)
    assert indices == []


def test_document_vector_omits_absent_terms(fitted):
    """A zero-frequency term contributes exactly 0, so it must not be stored --
    otherwise every document vector is the size of the vocabulary."""
    scorer, vocab = fitted
    indices, values = bm25_sparse.document_sparse_vector(scorer, 4, vocab)
    assert len(indices) == 1  # doc 4 is ["ไฟฟ้า"] x 3, one distinct term
    assert all(v > 0 for v in values)


def test_vocabulary_is_order_independent():
    """Ids are persisted next to the collection and re-read at query time; if
    they depended on corpus insertion order, a rebuild would silently re-key
    every stored vector while still returning results."""
    a = bm25_sparse.build_vocabulary(BM25Okapi(CORPUS))
    b = bm25_sparse.build_vocabulary(BM25Okapi(list(reversed(CORPUS))))
    assert a == b


def test_idf_floor_is_taken_from_the_scorer_not_recomputed(fitted):
    """`BM25Okapi` floors negative idf at `epsilon * average_idf`. A term in more
    than half the documents therefore carries a *positive* floored weight, which
    an engine-side IDF (Qdrant's `Modifier.IDF`) would not reproduce. Assert the
    floor is actually exercised by this fixture, so the equality test above is
    not passing on a corpus where the branch never fires."""
    scorer, vocab = fitted
    floored = [t for t, v in scorer.idf.items() if v == scorer.epsilon * scorer.average_idf]
    assert floored, "fixture no longer exercises the negative-idf floor"
    for term in floored:
        _, values = bm25_sparse.document_sparse_vector(
            scorer, next(i for i, d in enumerate(scorer.doc_freqs) if term in d), vocab
        )
        assert all(v > 0 for v in values)


def test_float32_storage_keeps_agreement_within_tolerance(fitted):
    """The engine stores sparse values as f32. Equality is therefore to ~1e-6
    relative, not bitwise -- state the tolerance here rather than discovering it
    as a ranking difference in the pilot."""
    scorer, vocab = fitted
    query = ["หลักสูตร", "ไฟฟ้า", "สภา"]
    expected = scorer.get_scores(query)
    qvec = bm25_sparse.query_sparse_vector(query, vocab)
    got = []
    for i in range(scorer.corpus_size):
        idx, val = bm25_sparse.document_sparse_vector(scorer, i, vocab)
        val32 = np.asarray(val, dtype=np.float32).astype(np.float64).tolist()
        got.append(bm25_sparse.dot((idx, val32), qvec))
    assert np.allclose(got, expected, rtol=1e-6, atol=1e-6)
