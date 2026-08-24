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
    """The `Retriever` selectbox's option list.

    Accepts either an inline list or the name of one, because the app now
    passes `_RETRIEVER_OPTIONS`: the default is chosen BY NAME rather than by
    being element 0, so the option list had to become a name too.
    """
    tree = _app_ast()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "selectbox"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "Retriever":
            continue
        assert len(node.args) >= 2
        arg = node.args[1]
        if isinstance(arg, ast.List):
            return [e.value for e in arg.elts]
        assert isinstance(arg, ast.Name), f"unexpected options node {type(arg).__name__}"
        return list(_named_list(arg.id))
    raise AssertionError("could not find the Retriever selectbox in the app source")


def _named_list(name: str) -> list[str]:
    for node in ast.walk(_app_ast()):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return [e.value for e in node.value.elts]
    raise AssertionError(f"could not find {name} in the app source")


def _named_str(name: str) -> str:
    for node in ast.walk(_app_ast()):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            assert isinstance(node.value, ast.Constant), f"{name} is not a literal"
            return node.value.value
    raise AssertionError(f"could not find {name} in the app source")


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
    """It is opt-in, so it must actually appear -- and must not be the default."""
    assert "lexical_containment" in _offered_retrievers()


def test_the_default_retriever_is_a_choice_not_a_position():
    """This assertion used to read `offered[0] == "dense"`, which conflated two
    different things: the first element of a list, and what a user gets. They
    were the same only because the widget passed `index=0`, so reordering the
    list for readability would have silently changed the shipped default and
    this test would have caught nothing.

    What is pinned now is the decision (2026-08-24, docs/serving-architecture.md
    section 10): the default is `hybrid`, which with Smart routing on is
    `routed hybrid` -- and the two arms that score higher stay opt-in, each for
    a stated reason rather than for lack of a number.
    """
    default = _named_str("_DEFAULT_RETRIEVER")
    offered = _offered_retrievers()
    assert default in offered, "the default is not one of the offered options"
    assert default == "hybrid"
    assert default not in {"lexical_containment", "qdrant_hybrid"}, (
        "lexical_containment's score partly measures its own rule (the "
        "person/program/faculty qrels were derived by string containment too) "
        "and qdrant_hybrid adds a container plus a stale collection that "
        "ANSWERS rather than fails -- neither may become the default silently"
    )


def test_the_selectbox_actually_uses_the_named_default():
    """A named constant nothing reads is worse than a magic number: it looks
    like a decision while the widget still opens on element 0."""
    src = _APP.read_text(encoding="utf-8")
    assert "_RETRIEVER_OPTIONS.index(_DEFAULT_RETRIEVER)" in src, (
        "the Retriever selectbox must resolve its index from _DEFAULT_RETRIEVER"
    )
