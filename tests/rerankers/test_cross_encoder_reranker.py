"""CrossEncoderReranker: lazy-loads a sentence-transformers CrossEncoder,
scores (query, candidate) pairs, and re-orders by score. A fake CrossEncoder
model is injected so the test never loads a real model."""
from __future__ import annotations

from rag_lab.rerankers.cross_encoder import CrossEncoderReranker
from rag_lab.schema import Query, RankedChunk


class _FakeCrossEncoderModel:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.predict_calls: list[dict] = []

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        self.predict_calls.append({"pairs": list(pairs), "batch_size": batch_size})
        return [self._scores[text] for _query, text in pairs]


def _chunk(i: int, text: str) -> RankedChunk:
    return RankedChunk(chunk_id=f"c{i}", resolution_id=f"r{i}", page=1, score=0.0, rank=i + 1, text=text)


def test_rerank_orders_by_cross_encoder_score():
    candidates = [_chunk(0, "low"), _chunk(1, "high"), _chunk(2, "mid")]
    model = _FakeCrossEncoderModel({"low": 0.1, "high": 0.9, "mid": 0.5})
    reranker = CrossEncoderReranker(model=model)

    ranked = reranker.rerank(Query(text="q"), candidates, k=3)

    assert [r.chunk_id for r in ranked] == ["c1", "c2", "c0"]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert ranked[0].score == 0.9


def test_rerank_limits_to_k():
    candidates = [_chunk(0, "low"), _chunk(1, "high"), _chunk(2, "mid")]
    model = _FakeCrossEncoderModel({"low": 0.1, "high": 0.9, "mid": 0.5})
    reranker = CrossEncoderReranker(model=model)

    ranked = reranker.rerank(Query(text="q"), candidates, k=2)

    assert len(ranked) == 2
    assert [r.chunk_id for r in ranked] == ["c1", "c2"]


def test_rerank_pairs_query_text_with_each_candidate_text():
    candidates = [_chunk(0, "a"), _chunk(1, "b")]
    model = _FakeCrossEncoderModel({"a": 0.0, "b": 0.0})
    reranker = CrossEncoderReranker(model=model)

    reranker.rerank(Query(text="the query"), candidates, k=2)

    assert model.predict_calls[0]["pairs"] == [("the query", "a"), ("the query", "b")]


def test_rerank_passes_configured_batch_size():
    candidates = [_chunk(0, "a")]
    model = _FakeCrossEncoderModel({"a": 0.0})
    reranker = CrossEncoderReranker(model=model, batch_size=2)

    reranker.rerank(Query(text="q"), candidates, k=1)

    assert model.predict_calls[0]["batch_size"] == 2


def test_rerank_on_empty_candidates_returns_empty_without_loading_model():
    reranker = CrossEncoderReranker()  # no injected model -- must not try to load one

    ranked = reranker.rerank(Query(text="q"), [], k=5)

    assert ranked == []
    assert reranker._model is None


def test_release_drops_the_loaded_model_so_it_reloads_on_next_use():
    model = _FakeCrossEncoderModel({})
    reranker = CrossEncoderReranker(model=model)

    reranker.release()

    assert reranker._model is None


def test_release_on_a_never_loaded_reranker_is_a_no_op():
    reranker = CrossEncoderReranker()  # never loaded

    reranker.release()  # must not raise

    assert reranker._model is None


def test_name_is_cross_encoder():
    assert CrossEncoderReranker().name == "cross_encoder"
