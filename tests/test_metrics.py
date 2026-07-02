"""Cycle 15 — Resolution-level metrics (ADR-0002): a Query's ground truth is a set
of relevant resolution_ids; metrics are computed by mapping each retrieved top-k
Chunk back to its Resolution. Top-k is a window over CHUNKS (the retrieval unit,
per CONTEXT.md) — dedup by resolution_id happens *after* slicing to k, not before,
so redundant chunks from one Resolution can legitimately crowd a different
relevant Resolution out of the window.
"""
from __future__ import annotations

import math

import pytest

from rag_lab.metrics import evaluate, ndcg_at_k, reciprocal_rank, recall_at_k
from rag_lab.schema import RankedChunk, RetrievalResult


def _result(
    resolution_ids: list[str], query: str = "q", combination_id: str = "combo"
) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        combination_id=combination_id,
        top_k=len(resolution_ids),
        retriever="dense",
        results=[
            RankedChunk(
                chunk_id=f"c{i}",
                resolution_id=rid,
                page=1,
                score=1.0 - i * 0.01,
                rank=i + 1,
                text="x",
            )
            for i, rid in enumerate(resolution_ids)
        ],
    )


def test_recall_at_k_counts_a_hit_within_top_k():
    result = _result(["r1", "r2", "r3"])

    assert recall_at_k(result, relevant_resolution_ids=["r2"], k=3) == 1.0


def test_recall_at_k_slices_chunks_before_deduping_by_resolution():
    # top-3 chunks are r1, r1, r2 — the relevant r3 sits at chunk rank 4, outside
    # the window, even though only 3 *distinct* resolutions have appeared by then.
    result = _result(["r1", "r1", "r2", "r3", "r5"])

    assert recall_at_k(result, relevant_resolution_ids=["r3"], k=3) == 0.0


def test_reciprocal_rank_uses_the_real_chunk_rank_of_the_first_hit():
    # the first r3 chunk is at rank 4 (1-indexed); deduping resolutions first
    # would wrongly report rank 3 (r1, r2, r3 are the first 3 *distinct* ids).
    result = _result(["r1", "r1", "r2", "r3", "r5"])

    assert reciprocal_rank(result, relevant_resolution_ids=["r3"]) == 1 / 4


def test_reciprocal_rank_is_zero_with_no_hit():
    result = _result(["r1", "r2"])

    assert reciprocal_rank(result, relevant_resolution_ids=["r9"]) == 0.0


def test_ndcg_at_k_is_one_when_the_only_relevant_resolution_ranks_first():
    result = _result(["r1", "r2", "r3"])

    assert ndcg_at_k(result, relevant_resolution_ids=["r1"], k=3) == 1.0


def test_ndcg_at_k_is_zero_when_relevant_resolution_is_crowded_out_of_the_window():
    result = _result(["r1", "r1", "r2", "r3", "r5"])

    assert ndcg_at_k(result, relevant_resolution_ids=["r3"], k=3) == 0.0


def test_multiple_chunks_from_one_relevant_resolution_count_as_a_single_hit():
    # AC: "chunk หลายตัวจากมติเดียวใน top-k นับเป็น hit เดียว" — 3 chunks from the
    # one relevant resolution must not inflate recall past 1.0.
    result = _result(["r1", "r1", "r1"])

    assert recall_at_k(result, relevant_resolution_ids=["r1"], k=3) == 1.0
    assert ndcg_at_k(result, relevant_resolution_ids=["r1"], k=3) == 1.0


def test_ndcg_at_k_credits_each_relevant_resolution_once_at_its_first_rank():
    # top-3 chunks: r1(rank1), r3(rank2), r1(rank3, ignored - r1 not relevant anyway
    # and already not distinct). r5 is relevant but sits at rank 4, outside the window.
    result = _result(["r1", "r3", "r1", "r5"])

    dcg = 1.0 / math.log2(2 + 1)  # r3 hits at rank 2
    idcg = 1.0 / math.log2(1 + 1) + 1.0 / math.log2(2 + 1)  # 2 relevant ids, k=3
    assert ndcg_at_k(result, relevant_resolution_ids=["r3", "r5"], k=3) == pytest.approx(
        dcg / idcg
    )


def test_evaluate_averages_metrics_across_queries_within_a_combination():
    results = [
        _result(["r1", "r2"], query="q1", combination_id="combo-a"),  # hit at rank 1
        _result(["r5", "r6"], query="q2", combination_id="combo-a"),  # hit at rank 2
    ]
    qrels = {"q1": ["r1"], "q2": ["r6"]}

    scores = evaluate(results, qrels, k=2)

    assert scores["combo-a"]["recall@2"] == 1.0
    assert scores["combo-a"]["mrr"] == pytest.approx((1 / 1 + 1 / 2) / 2)


def test_evaluate_scores_a_query_with_no_persisted_result_as_zero():
    # q2 is in the query set (qrels) but no RetrievalResult was persisted for it —
    # that must drag the average down, not shrink the denominator by skipping it.
    results = [_result(["r1"], query="q1", combination_id="combo-a")]
    qrels = {"q1": ["r1"], "q2": ["r9"]}

    scores = evaluate(results, qrels, k=1)

    assert scores["combo-a"]["recall@1"] == 0.5
