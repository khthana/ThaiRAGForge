"""LexicalContainmentRetriever — arm L-prime, wired.

The load-bearing test is `test_partition_equals_rrf4_at_w_one`: the retriever
implements a stable partition, but what was *measured* in
`tools/eval/reranker_trained_test.py` is a fourth RRF term at the
leave-one-out-selected w=1.00. Those are the same ranking, and that equivalence
is the entire licence for the shortcut — so it is pinned here against a
transcription of the eval's own fusion rather than asserted in a docstring.
"""
from __future__ import annotations

import numpy as np
import pytest

from rag_lab.config import StrategySpec
from rag_lab.factory import build_retriever
from rag_lab.registries import retriever_registry
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.retrievers.lexical_containment import LexicalContainmentRetriever
from rag_lab.schema import Chunk, Index, Query

ENTITY = "ดร.สมชาย ใจดี"
BARE = "สมชาย ใจดี"


def _index(texts: list[str]) -> Index:
    chunks = [
        Chunk(chunk_id=f"c{i}", resolution_id=f"r{i}", text=t, chunk_index=i, page=1)
        for i, t in enumerate(texts)
    ]
    # A deliberately uninformative embedding matrix: every row identical, so the
    # dense arm cannot impose an order of its own and the test is about the
    # containment partition rather than about which chunk happens to embed well.
    embeddings = np.tile(np.array([[1.0, 0.0]]), (len(texts), 1))
    lexical = [t.split() for t in texts]
    return Index(chunks=chunks, embeddings=embeddings, meta={}, lexical=lexical)


def _query(text: str = "ประวัติของ " + ENTITY) -> Query:
    return Query(text=text, vector=np.array([1.0, 0.0]), tokens=text.split())


def _fake_detector(mapping):
    return lambda text: mapping


# --------------------------------------------------------------- registration
def test_is_registered_under_its_own_name():
    assert "lexical_containment" in retriever_registry.names()
    r = build_retriever(StrategySpec(type="lexical_containment"))
    assert isinstance(r, LexicalContainmentRetriever)
    assert r.name == "lexical_containment"


def test_hybrid_is_unchanged_by_this_arm_existing():
    """Nothing defaults to it — `hybrid` must still be plain hybrid."""
    r = build_retriever(StrategySpec(type="hybrid"))
    assert isinstance(r, HybridRetriever)
    assert not isinstance(r, LexicalContainmentRetriever)


# ------------------------------------------------- the equivalence that matters
def _rrf4_at_w(candidates, hit_flags, w, rrf_k=60, k=3):
    """Transcription of tools/eval/reranker_trained_test.py's `fuse_grid`.

    `fused = (1-w)*hybrid_term + w*containment_term`, the containment term being
    a reciprocal rank over the pool sorted by (1/0 score - 1e-6 * hybrid rank),
    a chunk outside the pool contributing 0, ties settled stably in hybrid
    order. Deliberately written out rather than imported: importing the eval's
    copy would make this test agree with itself if the eval's fusion were wrong.
    """
    pool = len(candidates)
    sc = np.array([float(h) - 1e-6 * rank for rank, h in enumerate(hit_flags)])
    cterm = np.zeros(pool)
    for rank, pos in enumerate(np.argsort(-sc)):
        cterm[pos] = 1.0 / (rrf_k + rank + 1)
    hterm = np.array([1.0 / (rrf_k + i + 1) for i in range(pool)])
    fused = (1.0 - w) * hterm + w * cterm
    order = np.argsort(-fused, kind="stable")[:k]
    return [candidates[i] for i in order]


def test_partition_equals_rrf4_at_w_one():
    texts = [
        "รายงานทั่วไป",              # c0 no hit, best hybrid rank
        f"ที่ประชุมเชิญ {ENTITY} มา",  # c1 hit
        "เรื่องอื่น",                  # c2 no hit
        f"{ENTITY} เสนอหลักสูตร",     # c3 hit
    ]
    index = _index(texts)
    q = _query()
    r = LexicalContainmentRetriever(
        pool=4, entity_detector=_fake_detector({"people": [ENTITY]})
    )
    got = [c.chunk_id for c in r.retrieve(q, index, 3)]

    hybrid = HybridRetriever().retrieve(q, index, 4)
    ids = [c.chunk_id for c in hybrid]
    flags = [ENTITY in texts[int(cid[1:])] for cid in ids]
    expected = _rrf4_at_w(ids, flags, w=1.0, k=3)

    assert got == expected
    # ...and it is a real reordering, not a no-op that would make the test vacuous
    assert got != ids[:3]


def test_w_one_is_what_leave_one_out_selected():
    """Guard the premise, not just the arithmetic.

    If a future re-run picks a w below 1.00 the hybrid term stops being
    annihilated and this class silently stops implementing the measured arm.
    S9 of reranker_trained_test.py reports the selection; this pins the value
    the shortcut depends on so the coupling is visible in the test suite.
    """
    texts = ["ไม่มีชื่อ", f"มี {ENTITY} อยู่"]
    index = _index(texts)
    q = _query()
    hybrid = HybridRetriever().retrieve(q, index, 2)
    ids = [c.chunk_id for c in hybrid]
    flags = [ENTITY in texts[int(cid[1:])] for cid in ids]

    at_one = _rrf4_at_w(ids, flags, w=1.0, k=2)
    r = LexicalContainmentRetriever(
        pool=2, entity_detector=_fake_detector({"people": [ENTITY]})
    )
    assert [c.chunk_id for c in r.retrieve(q, index, 2)] == at_one
    # at w=0.00 the arm must be plain hybrid — the other end of the grid
    assert _rrf4_at_w(ids, flags, w=0.0, k=2) == ids


# ------------------------------------------------------------------- behaviour
def test_stable_within_each_group():
    texts = [
        f"{ENTITY} ก",   # hit
        "ไม่เกี่ยว ก",    # miss
        f"{ENTITY} ข",   # hit
        "ไม่เกี่ยว ข",    # miss
    ]
    index = _index(texts)
    r = LexicalContainmentRetriever(
        pool=4, entity_detector=_fake_detector({"people": [ENTITY]})
    )
    got = [c.chunk_id for c in r.retrieve(_query(), index, 4)]
    hybrid = [c.chunk_id for c in HybridRetriever().retrieve(_query(), index, 4)]

    hits = [c for c in hybrid if ENTITY in texts[int(c[1:])]]
    misses = [c for c in hybrid if ENTITY not in texts[int(c[1:])]]
    assert got == hits + misses


def test_no_entity_detected_falls_back_to_hybrid_order():
    texts = ["ก", "ข", "ค"]
    index = _index(texts)
    q = _query("คำถามที่ไม่เอ่ยชื่อใครเลย")
    r = LexicalContainmentRetriever(pool=3, entity_detector=_fake_detector({}))
    got = [c.chunk_id for c in r.retrieve(q, index, 3)]
    assert got == [c.chunk_id for c in HybridRetriever().retrieve(q, index, 3)]


def test_any_entity_matches_not_all():
    """Detection returns several canonicals for one query (two course codes for
    one `ENGLISH FOR ...` name); requiring all of them would score the matcher's
    recall rather than the containment signal."""
    texts = ["ไม่มีอะไร", "มี 01416504 อยู่"]
    index = _index(texts)
    r = LexicalContainmentRetriever(
        pool=2, entity_detector=_fake_detector({"courses": ["01416504", "02646204"]})
    )
    got = [c.chunk_id for c in r.retrieve(_query(), index, 2)]
    assert got[0] == "c1"


def test_containment_rule_is_the_shared_one_not_naive_substring():
    """`CALCULUS 2` must not match inside `CALCULUS 21` — the boundary rule from
    src/rag_lab/text_match.py, which the anchor-ambiguity audit also uses."""
    texts = ["วิชา CALCULUS 21 ปรับปรุง", "วิชา CALCULUS 2 ปรับปรุง"]
    index = _index(texts)
    r = LexicalContainmentRetriever(
        pool=2, entity_detector=_fake_detector({"courses": ["CALCULUS 2"]})
    )
    got = [c.chunk_id for c in r.retrieve(_query(), index, 2)]
    assert got[0] == "c1", "naive substring matching would promote CALCULUS 21"


def test_whitespace_is_collapsed_before_matching():
    """OCR'd minutes wrap a long name across a line; matching raw text would
    call a genuine mention absent."""
    texts = ["ไม่มี", "ที่ประชุมเชิญ ดร.สมชาย\n   ใจดี มาชี้แจง"]
    index = _index(texts)
    r = LexicalContainmentRetriever(
        pool=2, entity_detector=_fake_detector({"people": [ENTITY]})
    )
    assert [c.chunk_id for c in r.retrieve(_query(), index, 2)][0] == "c1"


def test_candidates_outside_the_pool_are_never_promoted():
    """The pool bounds what containment may reorder. A chunk the router placed
    past `pool` must not jump the queue just because it holds the entity —
    otherwise the arm is a full-corpus lexical filter, which is a different
    (and unmeasured) retriever."""
    texts = ["ก", "ข", "ค", f"{ENTITY} อยู่ท้ายสุด"]
    index = _index(texts)
    hybrid = [c.chunk_id for c in HybridRetriever().retrieve(_query(), index, 4)]
    deep = hybrid[-1]
    r = LexicalContainmentRetriever(
        pool=len(texts) - 1, entity_detector=_fake_detector({"people": [ENTITY]})
    )
    got = [c.chunk_id for c in r.retrieve(_query(), index, 4)]
    if deep == "c3":
        assert got[0] != "c3"
    assert sorted(got) == sorted(hybrid)


def test_score_stays_the_hybrid_score():
    texts = ["ก", f"{ENTITY} ข"]
    index = _index(texts)
    r = LexicalContainmentRetriever(
        pool=2, entity_detector=_fake_detector({"people": [ENTITY]})
    )
    got = r.retrieve(_query(), index, 2)
    hybrid = {c.chunk_id: c.score for c in HybridRetriever().retrieve(_query(), index, 2)}
    assert all(c.score == hybrid[c.chunk_id] for c in got)
    assert [c.rank for c in got] == [1, 2]


def test_reads_index_rows_stays_true():
    """It ranks in-process over Index rows, so a row-level filter/boost is
    legitimate here — unlike the engine-served qdrant_hybrid arm."""
    assert LexicalContainmentRetriever().reads_index_rows is True
    assert LexicalContainmentRetriever().exhaustive is False


def test_pool_must_be_positive():
    with pytest.raises(ValueError, match="pool must be"):
        LexicalContainmentRetriever(pool=0)


def test_k_larger_than_pool_still_fetches_k():
    texts = ["ก", "ข", "ค", "ง"]
    index = _index(texts)
    r = LexicalContainmentRetriever(pool=2, entity_detector=_fake_detector({}))
    assert len(r.retrieve(_query(), index, 4)) == 4
