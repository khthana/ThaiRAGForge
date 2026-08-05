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
