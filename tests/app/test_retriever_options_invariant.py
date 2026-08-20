"""A retriever offered in the UI that quietly ignores the depth shown beside it.

`lexical_containment` was added to the Retriever selectbox on 2026-08-20 and
`_retriever_spec()` still read `if retriever != "hybrid"`, so the new arm ran at
whole-corpus depth while the sidebar displayed 200. Nothing failed: the answers
were fine, just slower and computed at a depth the operator did not choose. That
is the same silent shape as the router shipping with three routes while the gold
set had four entity types, and `tests/test_router.py` pins that one structurally
rather than by smoke — so this pins its UI analogue.

The rule: **every retriever the UI offers whose class takes `fetch_depth` must be
in `_FUSED_RETRIEVERS`**, because that set is what decides whether the widget's
value is passed on. The app is read with `ast` rather than imported: importing it
executes Streamlit page setup, and this invariant is a property of the source.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from rag_lab.registries import retriever_registry
import rag_lab.retrievers  # noqa: F401  (populates the registry)

_APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


def _app_ast() -> ast.Module:
    return ast.parse(_APP.read_text(encoding="utf-8"))


def _offered_retrievers() -> list[str]:
    """The `Retriever` selectbox's option list."""
    for node in ast.walk(_app_ast()):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "selectbox"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "Retriever":
            continue
        assert len(node.args) >= 2 and isinstance(node.args[1], ast.List)
        return [e.value for e in node.args[1].elts]
    raise AssertionError("could not find the Retriever selectbox in the app source")


def _named_set(name: str) -> set[str]:
    for node in ast.walk(_app_ast()):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return {e.value for e in node.value.elts}
    raise AssertionError(f"could not find {name} in the app source")


def _takes_fetch_depth(retriever_name: str) -> bool:
    cls = retriever_registry.get(retriever_name)
    return "fetch_depth" in inspect.signature(cls.__init__).parameters


def test_the_parsers_find_something():
    """Guard against the whole file passing because `ast` matched nothing."""
    offered = _offered_retrievers()
    assert len(offered) >= 4 and "hybrid" in offered
    assert "hybrid" in _named_set("_FUSED_RETRIEVERS")
    assert "qdrant_hybrid" in _named_set("_ENGINE_RETRIEVERS")


@pytest.mark.parametrize("name", _offered_retrievers())
def test_every_offered_retriever_is_registered(name):
    assert name in retriever_registry.names()


def test_depth_capable_retrievers_are_all_in_the_fused_set():
    fused = _named_set("_FUSED_RETRIEVERS")
    missing = [
        n for n in _offered_retrievers() if _takes_fetch_depth(n) and n not in fused
    ]
    assert not missing, (
        f"{missing} accept fetch_depth but are not in _FUSED_RETRIEVERS, so the "
        "sidebar would show a depth it never passes on — the arm silently runs "
        "at whole-corpus depth"
    )


def test_the_fused_set_holds_nothing_that_would_reject_the_depth():
    """The converse: a retriever in the set whose class has no `fetch_depth`
    would raise a TypeError at query time instead of running wrong."""
    bad = [n for n in _named_set("_FUSED_RETRIEVERS") if not _takes_fetch_depth(n)]
    assert not bad, f"{bad} are in _FUSED_RETRIEVERS but take no fetch_depth"


def test_lexical_containment_is_offered():
    """It is opt-in, so it must actually appear — and must not be the default."""
    offered = _offered_retrievers()
    assert "lexical_containment" in offered
    assert offered[0] == "dense", "nothing new may become the UI default"
