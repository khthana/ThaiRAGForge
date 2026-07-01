"""Build strategy instances from {type, params} specs via the registries.

Imports the strategy packages so their decorator registrations fire (a
decorator-based registry is silent if the module is never imported).
"""
from __future__ import annotations

import rag_lab.chunkers  # noqa: F401  (register strategies)
import rag_lab.embedders  # noqa: F401
import rag_lab.loaders  # noqa: F401
import rag_lab.retrievers  # noqa: F401
from rag_lab.config import StrategySpec
from rag_lab.registries import (
    chunker_registry,
    embedder_registry,
    loader_registry,
    retriever_registry,
)


def build_loader(spec: StrategySpec):
    return loader_registry.get(spec.type)(**spec.params)


def build_chunker(spec: StrategySpec):
    return chunker_registry.get(spec.type)(**spec.params)


def build_embedder(spec: StrategySpec):
    return embedder_registry.get(spec.type)(**spec.params)


def build_retriever(spec: StrategySpec):
    return retriever_registry.get(spec.type)(**spec.params)
