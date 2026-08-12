"""Pins the three pre-run guards a *second* RQ4 generator has to clear.

All three exist because the published run used one model (`phi4`) under one
default, and each default is wrong for any other model in a way that produces
plausible output rather than an error:

  - `sentence_cap` is a retired baseline. The pre-registered rule in
    `rq4_score.py` made a second generator worth running only if citation recall
    stayed flat at ~0.41 through the `cite_all` ablation. It rose, so a run under
    `sentence_cap` costs ~5 GPU-hours and answers nothing.
  - `think` was never passed, which is a no-op for `phi4` (capabilities
    `['completion']`) and expensive for `gemma4:e4b` (`[..., 'thinking']`).
  - `--num-ctx` below 16,384 truncates the longest prompts silently, keeping the
    tail, so the best-ranked documents are the ones deleted.

Every guard is pinned in **both** directions: a guard that only ever refuses is
as useless as one that never does ([[feedback_an_asserted_invariant_is_not_a_check]]).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))

import rq4_generate  # noqa: E402
from rq4_generate import (  # noqa: E402
    _MIN_NUM_CTX,
    _refuse_retired_variant,
    _refuse_small_ctx,
    supports_thinking,
)


# --- guard 1: the retired variant -------------------------------------------

@pytest.mark.parametrize("model", ["phi4", "phi4:latest"])
def test_baseline_model_may_still_use_the_retired_variant(model):
    """The 530 published answers are keyed to this exact pair; re-running or
    resuming it must stay possible, tag or no tag."""
    _refuse_retired_variant(model, "sentence_cap", allow=False)


def test_other_model_under_sentence_cap_is_refused():
    with pytest.raises(SystemExit) as exc:
        _refuse_retired_variant("gemma4:e4b", "sentence_cap", allow=False)
    # the message has to name the way out, or the guard just blocks the work
    assert "cite_all" in str(exc.value)


@pytest.mark.parametrize("variant", ["cite_all", "cite_all_guarded"])
def test_other_model_under_a_live_variant_is_allowed(variant):
    _refuse_retired_variant("gemma4:e4b", variant, allow=False)


def test_the_escape_hatch_works():
    """Containment, not a ban: reproducing the retired pair on purpose is a
    legitimate thing to want."""
    _refuse_retired_variant("gemma4:e4b", "sentence_cap", allow=True)


# --- guard 2: thinking capability -------------------------------------------

class _Show:
    def __init__(self, capabilities):
        self.capabilities = capabilities


def test_thinking_is_read_from_capabilities_both_ways(monkeypatch):
    """The two real answers this project sees, as reported by ollama 0.32.6."""
    caps = {"phi4:latest": ["completion"],
            "gemma4:e4b": ["completion", "vision", "audio", "tools", "thinking"]}
    monkeypatch.setattr(rq4_generate.ollama, "show", lambda m: _Show(caps[m]))
    assert supports_thinking("phi4:latest") is False
    assert supports_thinking("gemma4:e4b") is True


def test_unreadable_capabilities_assume_no_thinking(monkeypatch):
    """Fail toward the behaviour every published answer was generated with.

    Guessing `True` would send `think=False` to a model that may reject it and
    kill a multi-hour run; guessing `False` only forfeits a saving.
    """
    def boom(_model):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(rq4_generate.ollama, "show", boom)
    assert supports_thinking("anything") is False


def test_capabilities_may_be_absent_or_none(monkeypatch):
    """Older ollama-python returns a payload with no `capabilities` at all."""
    monkeypatch.setattr(rq4_generate.ollama, "show", lambda m: _Show(None))
    assert supports_thinking("x") is False
    monkeypatch.setattr(rq4_generate.ollama, "show", lambda m: object())
    assert supports_thinking("x") is False


# --- guard 3: the context floor ---------------------------------------------

def test_the_historical_default_is_refused():
    """8192 is not a neutral smaller value -- it is the setting that produced
    the 80 truncated cells."""
    with pytest.raises(SystemExit) as exc:
        _refuse_small_ctx(8192, allow=False)
    assert "--allow-small-ctx" in str(exc.value)


@pytest.mark.parametrize("num_ctx", [_MIN_NUM_CTX, _MIN_NUM_CTX * 2])
def test_the_floor_and_above_pass(num_ctx):
    _refuse_small_ctx(num_ctx, allow=False)


def test_small_ctx_is_reachable_on_purpose():
    """`rq4_probe_prompt_fit.py` deliberately probes at the old 8,192 to
    re-derive G1b, so the floor cannot be absolute."""
    _refuse_small_ctx(8192, allow=True)


def test_floor_matches_the_shipped_default():
    """If the default ever drops below the floor, every run refuses to start."""
    assert rq4_generate._MIN_NUM_CTX <= 16384
