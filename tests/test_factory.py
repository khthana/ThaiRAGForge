"""Cycle 5 — the factory builds registered strategies from {type, params}."""
from __future__ import annotations

import pytest

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.config import StrategySpec
from rag_lab.factory import build_chunker, build_embedder


def test_builds_chunker_with_params():
    chunker = build_chunker(StrategySpec(type="fixed_size", params={"chunk_size": 256}))
    assert isinstance(chunker, FixedSizeChunker)
    assert chunker.chunk_size == 256


def test_builds_registered_embedder_by_name():
    embedder = build_embedder(StrategySpec(type="hashing"))
    assert embedder.model_id.startswith("hashing")


def test_builds_e5_embedder_selectable_via_config():
    from rag_lab.embedders import E5Embedder

    embedder = build_embedder(StrategySpec(type="e5"))
    assert isinstance(embedder, E5Embedder)


def test_unknown_type_raises():
    with pytest.raises(KeyError):
        build_chunker(StrategySpec(type="no-such-chunker"))
