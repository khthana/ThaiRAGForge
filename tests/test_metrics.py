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

from rag_lab.metrics import (
    average_precision_at_k,
    evaluate,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
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


def test_precision_at_k_divides_by_the_chunk_window_not_by_hit_count():
    # 1 relevant resolution found within a window of 3 chunks -> 1/3, not 1/1.
    result = _result(["r1", "r2", "r3"])

    assert precision_at_k(result, relevant_resolution_ids=["r2"], k=3) == pytest.approx(1 / 3)


def test_precision_at_k_slices_chunks_before_deduping_by_resolution():
    result = _result(["r1", "r1", "r2", "r3", "r5"])

    assert precision_at_k(result, relevant_resolution_ids=["r3"], k=3) == 0.0


def test_precision_at_k_counts_one_relevant_resolution_once_despite_repeat_chunks():
    result = _result(["r1", "r1", "r1"])

    assert precision_at_k(result, relevant_resolution_ids=["r1"], k=3) == pytest.approx(1 / 3)


def test_average_precision_is_one_when_every_relevant_resolution_ranks_first():
    result = _result(["r1", "r2", "r3"])

    assert average_precision_at_k(result, relevant_resolution_ids=["r1"], k=3) == 1.0


def test_average_precision_is_zero_with_no_hit_in_window():
    result = _result(["r1", "r2"])

    assert average_precision_at_k(result, relevant_resolution_ids=["r9"], k=2) == 0.0


def test_average_precision_averages_precision_at_each_hit_rank_over_total_relevant():
    # relevant = {r2, r4}. r2 hits at chunk rank 2 (precision so far 1/2), r4 hits
    # at chunk rank 4 (precision so far 2/4) -- AP divides the sum by the total
    # relevant count (2), matching recall_at_k's denominator convention.
    result = _result(["r1", "r2", "r3", "r4"])

    expected = (1 / 2 + 2 / 4) / 2
    assert average_precision_at_k(
        result, relevant_resolution_ids=["r2", "r4"], k=4
    ) == pytest.approx(expected)


def test_average_precision_only_credits_the_first_occurrence_of_a_resolution():
    # r1 repeats at chunk ranks 1 and 3; must count once (hit=1 at rank 1), not
    # inflate the AP sum with a second credit at rank 3.
    result = _result(["r1", "r2", "r1"])

    assert average_precision_at_k(result, relevant_resolution_ids=["r1"], k=3) == 1.0


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


def test_evaluate_single_int_k_still_reports_precision_and_map():
    # backward-compat: existing callers pass a plain int k. New keys (precision@k,
    # map) must appear alongside the original recall@k/mrr/ndcg@k, not replace them.
    results = [_result(["r1", "r2"], query="q1", combination_id="combo-a")]
    qrels = {"q1": ["r1"]}

    scores = evaluate(results, qrels, k=2)

    assert scores["combo-a"]["recall@2"] == 1.0
    assert scores["combo-a"]["precision@2"] == pytest.approx(1 / 2)
    assert scores["combo-a"]["map"] == 1.0


def test_evaluate_accepts_a_list_of_k_and_reports_every_cutoff():
    results = [_result(["r1", "r2", "r3"], query="q1", combination_id="combo-a")]
    qrels = {"q1": ["r3"]}  # relevant resolution sits at chunk rank 3

    scores = evaluate(results, qrels, k=[1, 3])

    assert scores["combo-a"]["recall@1"] == 0.0
    assert scores["combo-a"]["recall@3"] == 1.0
    assert scores["combo-a"]["precision@1"] == 0.0
    assert scores["combo-a"]["precision@3"] == pytest.approx(1 / 3)
    # map is computed once at max(k)=3, not once per cutoff
    assert scores["combo-a"]["map"] == pytest.approx(1 / 3)
