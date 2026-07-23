"""The per-stage registry instances that strategies register themselves into.

Kept separate from the strategy modules (which import these) so there is no
import cycle: this module imports only the Registry class.
"""
from __future__ import annotations

from rag_lab.registry import Registry

loader_registry = Registry("loader")
chunker_registry = Registry("chunker")
embedder_registry = Registry("embedder")
retriever_registry = Registry("retriever")
reranker_registry = Registry("reranker")
