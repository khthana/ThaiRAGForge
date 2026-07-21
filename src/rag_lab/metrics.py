"""Resolution-level retrieval metrics (ADR-0002).

A Query's ground truth is "these resolution_ids are relevant." Every metric maps
retrieved Chunks back to their Resolution and works from there. Top-k is always a
window over CHUNKS first (the retrieval unit) — deduping to distinct resolution_ids
happens after slicing, never before, so redundant chunks from one Resolution can
legitimately crowd another relevant Resolution out of the window.
"""
from __future__ import annotations

import math

from rag_lab.schema import RetrievalResult


def _chunks_within_k(result: RetrievalResult, k: int) -> list:
    return [rc for rc in sorted(result.results, key=lambda rc: rc.rank) if rc.rank <= k]


def recall_at_k(
    result: RetrievalResult, relevant_resolution_ids: list[str], k: int
) -> float:
    relevant = set(relevant_resolution_ids)
    if not relevant:
        return 0.0
    hit_resolutions = {rc.resolution_id for rc in _chunks_within_k(result, k)} & relevant
    return len(hit_resolutions) / len(relevant)


def reciprocal_rank(result: RetrievalResult, relevant_resolution_ids: list[str]) -> float:
    relevant = set(relevant_resolution_ids)
    for rc in sorted(result.results, key=lambda rc: rc.rank):
        if rc.resolution_id in relevant:
            return 1.0 / rc.rank
    return 0.0


def precision_at_k(
    result: RetrievalResult, relevant_resolution_ids: list[str], k: int
) -> float:
    relevant = set(relevant_resolution_ids)
    if not relevant:
        return 0.0
    hit_resolutions = {rc.resolution_id for rc in _chunks_within_k(result, k)} & relevant
    return len(hit_resolutions) / k


def average_precision_at_k(
    result: RetrievalResult, relevant_resolution_ids: list[str], k: int
) -> float:
    """Standard IR Average Precision, resolution-level and chunk-windowed to k.

    Divides by the total relevant count (like recall_at_k), not by the number
    of hits found -- a query that never surfaces all its relevant resolutions
    within the window is penalized, matching recall_at_k's convention.
    """
    relevant = set(relevant_resolution_ids)
    if not relevant:
        return 0.0
    seen: set[str] = set()
    hits = 0
    precision_sum = 0.0
    for rc in _chunks_within_k(result, k):
        if rc.resolution_id in relevant and rc.resolution_id not in seen:
            seen.add(rc.resolution_id)
            hits += 1
            precision_sum += hits / rc.rank
    return precision_sum / len(relevant)


def ndcg_at_k(
    result: RetrievalResult, relevant_resolution_ids: list[str], k: int
) -> float:
    relevant = set(relevant_resolution_ids)
    if not relevant:
        return 0.0

    seen: set[str] = set()
    dcg = 0.0
    for rc in _chunks_within_k(result, k):
        if rc.resolution_id in relevant and rc.resolution_id not in seen:
            seen.add(rc.resolution_id)
            dcg += 1.0 / math.log2(rc.rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    results: list[RetrievalResult], qrels: dict[str, list[str]], k: int | list[int]
) -> dict[str, dict[str, float]]:
    """Average recall/precision/ndcg@k, mrr, and map over every query in qrels,
    per combination_id.

    `k` may be a single int (original behavior, one recall@k/precision@k/ndcg@k
    triple) or a list of ints to report multiple cutoffs (e.g. [1, 3, 5, 10]) in
    one pass. MAP is computed once per combination, windowed to max(k) -- not
    per-cutoff, since Average Precision already aggregates over a ranking.

    A query in qrels with no matching persisted result scores 0 on every metric
    for that combination rather than being excluded from the average — a gap in
    a batch run must show up as a worse score, not a smaller, quietly-easier one.
    """
    ks = sorted({k}) if isinstance(k, int) else sorted(set(k))
    k_for_map = max(ks)
    by_key = {(r.combination_id, r.query): r for r in results}
    combination_ids = sorted({r.combination_id for r in results})

    scores: dict[str, dict[str, float]] = {}
    for combination_id in combination_ids:
        recalls = {kk: [] for kk in ks}
        precisions = {kk: [] for kk in ks}
        ndcgs = {kk: [] for kk in ks}
        rrs, aps = [], []
        for query, relevant in qrels.items():
            result = by_key.get((combination_id, query))
            if result is None:
                for kk in ks:
                    recalls[kk].append(0.0)
                    precisions[kk].append(0.0)
                    ndcgs[kk].append(0.0)
                rrs.append(0.0)
                aps.append(0.0)
                continue
            for kk in ks:
                recalls[kk].append(recall_at_k(result, relevant, kk))
                precisions[kk].append(precision_at_k(result, relevant, kk))
                ndcgs[kk].append(ndcg_at_k(result, relevant, kk))
            rrs.append(reciprocal_rank(result, relevant))
            aps.append(average_precision_at_k(result, relevant, k_for_map))

        combo_scores = {"mrr": sum(rrs) / len(rrs), "map": sum(aps) / len(aps)}
        for kk in ks:
            combo_scores[f"recall@{kk}"] = sum(recalls[kk]) / len(recalls[kk])
            combo_scores[f"precision@{kk}"] = sum(precisions[kk]) / len(precisions[kk])
            combo_scores[f"ndcg@{kk}"] = sum(ndcgs[kk]) / len(ndcgs[kk])
        scores[combination_id] = combo_scores
    return scores
