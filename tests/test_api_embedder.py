"""Cycle 13 — APIEmbedder interface (OpenAI/Voyage/Cohere): stays inert until a
key is configured. Must raise clearly at construction, not surface as an
unexplained crash mid-batch (run_experiment's error isolation catches this,
see test_runner.py).
"""
from __future__ import annotations

import pytest

from rag_lab.embedders.api_embedder import APIEmbedder, MissingAPIKeyError


def test_raises_clear_error_when_key_env_is_absent(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        APIEmbedder(provider="openai", model="text-embedding-3-large")


def test_constructs_when_key_env_is_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    embedder = APIEmbedder(provider="openai", model="text-embedding-3-large")

    assert embedder.model_id == "api:openai:text-embedding-3-large"
