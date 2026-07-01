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
