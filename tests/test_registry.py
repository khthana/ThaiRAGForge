"""Cycle 8 — the decorator registry that makes strategies pluggable (OCP).

New strategies register by name without the runner being edited.
"""
from __future__ import annotations

import pytest

from rag_lab.registry import Registry


def test_register_then_get_by_name():
    reg = Registry("chunker")

    @reg.register("fixed")
    class Fixed:
        pass

    assert reg.get("fixed") is Fixed
    assert "fixed" in reg.names()


def test_unknown_name_raises_keyerror():
    reg = Registry("chunker")
    with pytest.raises(KeyError):
        reg.get("does-not-exist")


def test_duplicate_registration_raises():
    reg = Registry("chunker")

    @reg.register("dup")
    class A:
        pass

    with pytest.raises(ValueError):

        @reg.register("dup")
        class B:
            pass


def test_reranker_registry_has_cross_encoder_registered():
    from rag_lab.registries import reranker_registry
    from rag_lab.rerankers.cross_encoder import CrossEncoderReranker

    assert reranker_registry.get("cross_encoder") is CrossEncoderReranker
    assert "cross_encoder" in reranker_registry.names()


def test_retriever_registry_has_entity_lookup_registered():
    from rag_lab.registries import retriever_registry
    from rag_lab.retrievers.entity_lookup import EntityLookupRetriever

    assert retriever_registry.get("entity_lookup") is EntityLookupRetriever
    assert "entity_lookup" in retriever_registry.names()


def test_loader_registry_has_entity_tags_and_course_registered():
    from rag_lab.loaders.course_loader import CourseLoader
    from rag_lab.loaders.entity_loader import EntityTagLoader
    from rag_lab.registries import loader_registry

    assert loader_registry.get("entity_tags") is EntityTagLoader
    assert loader_registry.get("course") is CourseLoader
    assert {"entity_tags", "course"} <= set(loader_registry.names())
