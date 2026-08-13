"""`ColbertRetriever`: the ranking, and the two ways it can be silently wrong.

The ranking itself is pinned against `maxsim_reference`, the naive one-document-
at-a-time definition, because the packed `reduceat` path is an optimisation whose
failure mode (a segment boundary off by one) returns plausible scores.

The other two rules are refusals, and both are refusals of things that would
otherwise *work*:

* a 1-D `query.vector` is a dense query, and reshaping it to `(1, dim)` would
  turn MaxSim into max-pooled cosine and still return a ranking;
* an index the artifact was not built from -- a `select()` sub-index above all --
  still has chunks to attribute scores to, so scoring it produces a complete,
  well-ordered, wrong result.
"""
from __future__ import annotations

import numpy as np
import pytest

from rag_lab.colbert.scoring import maxsim_reference
from rag_lab.colbert.store import ColbertArtifact, ColbertStore
from rag_lab.config import StrategySpec
from rag_lab.factory import build_retriever
from rag_lab.retrievers.colbert import ColbertRetriever
from rag_lab.schema import Chunk, Index, Query

DIM = 8
LENGTHS = [3, 5, 2, 4]


def _unit(rows: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(rows, DIM)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)


def _index() -> Index:
    chunks = [
        Chunk(chunk_id=f"c{i}", resolution_id=f"r{i // 2}", text=f"t{i}",
              chunk_index=i, page=1, metadata={})
        for i in range(len(LENGTHS))
    ]
    return Index(chunks=chunks, embeddings=np.zeros((len(chunks), DIM)), meta={})


def _artifact(ids=None) -> ColbertArtifact:
    lengths = np.asarray(LENGTHS, dtype=np.int64)
    return ColbertArtifact(
        chunk_ids=list(ids or [f"c{i}" for i in range(len(LENGTHS))]),
        vecs=_unit(int(lengths.sum()), seed=1),
        lengths=lengths,
        meta={"dim": DIM, "doc_maxlen": 300},
    )


def _query(seed: int = 2, maxlen: int = 6) -> Query:
    return Query(text="q", vector=_unit(maxlen, seed).astype(np.float32))


def test_ranking_matches_the_naive_definition():
    art, index = _artifact(), _index()
    got = ColbertRetriever(artifact=art).retrieve(_query(), index, k=len(LENGTHS))

    docs, at = [], 0
    for n in LENGTHS:
        docs.append(np.asarray(art.vecs[at:at + n], dtype=np.float32))
        at += n
    ref = maxsim_reference(np.asarray(_query().vector, dtype=np.float32), docs)
    want = [f"c{i}" for i in np.argsort(-ref)]

    assert [r.chunk_id for r in got] == want
    assert [r.rank for r in got] == [1, 2, 3, 4]
    assert got[0].score == pytest.approx(float(ref.max()), abs=1e-4)
    assert got[0].resolution_id == index.chunks[int(np.argmax(ref))].resolution_id


def test_k_truncates_without_disturbing_the_order():
    art, index = _artifact(), _index()
    full = ColbertRetriever(artifact=art).retrieve(_query(), index, k=4)
    top2 = ColbertRetriever(artifact=art).retrieve(_query(), index, k=2)
    assert [r.chunk_id for r in top2] == [r.chunk_id for r in full[:2]]


def test_a_one_dimensional_query_vector_is_refused_not_reshaped():
    """It would score as max-pooled cosine and return a plausible ranking."""
    with pytest.raises(ValueError, match="query_maxlen"):
        ColbertRetriever(artifact=_artifact()).retrieve(
            Query(text="q", vector=np.ones(DIM, dtype=np.float32)), _index(), k=2
        )


def test_a_missing_query_vector_is_refused():
    with pytest.raises(ValueError, match="requires query.vector"):
        ColbertRetriever(artifact=_artifact()).retrieve(Query(text="q"), _index(), k=2)


def test_a_sub_index_is_refused_rather_than_scored():
    """`select()` re-slices the row-aligned arrays; it cannot re-slice a packed
    per-token artifact, so the pair is no longer the pair that was built."""
    with pytest.raises(ValueError, match="does not align"):
        ColbertRetriever(artifact=_artifact()).retrieve(
            _query(), _index().select([0, 2]), k=2
        )


def test_an_index_of_the_same_size_but_different_chunks_is_refused():
    """Shape agreement is not alignment -- the failure a length check cannot see."""
    art = _artifact(ids=["c0", "c1", "c2", "cX"])
    with pytest.raises(ValueError, match="L1b"):
        ColbertRetriever(artifact=art).retrieve(_query(), _index(), k=2)


def test_verification_runs_once_per_index_not_once_per_query():
    r, index = ColbertRetriever(artifact=_artifact()), _index()
    assert r._verified_for is None
    r.retrieve(_query(), index, k=2)
    assert r._verified_for is index.chunks
    r.retrieve(_query(seed=3), index, k=2)
    assert r._verified_for is index.chunks


def test_a_second_index_is_re_verified_and_can_still_be_refused():
    """The memo is keyed on identity, so a different index does not inherit the
    first one's clean bill of health."""
    r = ColbertRetriever(artifact=_artifact())
    r.retrieve(_query(), _index(), k=2)
    with pytest.raises(ValueError, match="does not align"):
        r.retrieve(_query(), _index().select([0, 1]), k=2)


def test_an_empty_index_returns_nothing():
    empty = Index(chunks=[], embeddings=np.zeros((0, DIM)), meta={})
    assert ColbertRetriever(artifact=_artifact()).retrieve(_query(), empty, k=5) == []


def test_registered_and_buildable_from_a_strategy_spec(tmp_path):
    """ADR-0001: a new retriever is reachable by name from config, with no edit
    to the runner."""
    art = _artifact()
    ColbertStore().save(tmp_path, art.chunk_ids, art.vecs, art.lengths, art.meta)
    r = build_retriever(StrategySpec(type="colbert", params={"artifact_dir": str(tmp_path)}))
    assert isinstance(r, ColbertRetriever) and r.name == "colbert"
    assert [x.chunk_id for x in r.retrieve(_query(), _index(), k=4)] == [
        x.chunk_id for x in ColbertRetriever(artifact=art).retrieve(_query(), _index(), k=4)
    ]


def test_exactly_one_source_of_the_artifact_is_required():
    with pytest.raises(ValueError, match="exactly one"):
        ColbertRetriever()
    with pytest.raises(ValueError, match="exactly one"):
        ColbertRetriever(artifact_dir="x", artifact=_artifact())
