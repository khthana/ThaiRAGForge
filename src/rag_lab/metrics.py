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
    results: list[RetrievalResult], qrels: dict[str, list[str]], k: int
) -> dict[str, dict[str, float]]:
    """Average recall@k/mrr/ndcg@k over every query in qrels, per combination_id.

    A query in qrels with no matching persisted result scores 0 on every metric
    for that combination rather than being excluded from the average — a gap in
    a batch run must show up as a worse score, not a smaller, quietly-easier one.
    """
    by_key = {(r.combination_id, r.query): r for r in results}
    combination_ids = sorted({r.combination_id for r in results})

    scores: dict[str, dict[str, float]] = {}
    for combination_id in combination_ids:
        recalls, rrs, ndcgs = [], [], []
        for query, relevant in qrels.items():
            result = by_key.get((combination_id, query))
            if result is None:
                recalls.append(0.0)
                rrs.append(0.0)
                ndcgs.append(0.0)
                continue
            recalls.append(recall_at_k(result, relevant, k))
            rrs.append(reciprocal_rank(result, relevant))
            ndcgs.append(ndcg_at_k(result, relevant, k))
        scores[combination_id] = {
            f"recall@{k}": sum(recalls) / len(recalls),
            "mrr": sum(rrs) / len(rrs),
            f"ndcg@{k}": sum(ndcgs) / len(ndcgs),
        }
    return scores
