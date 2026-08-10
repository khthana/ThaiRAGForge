"""Pins the `--arms` opt-in that lets the entity arms be scored (2026-08-10).

`rq4_score.py` had `ARM_ORDER` hardcoded to the five published arms, so the two
entity arms on disk would have been **silently dropped** from the report -- no
error, just a table missing the rows the run was for. The flag fixes that, and
brings two hazards of its own that both fail silently if unguarded:

* a mistyped arm name would reproduce the original bug (arm quietly absent);
* family 1 is Holm-corrected across *arm pairs*, so scoring a different arm set
  into `rq4_score.md` would re-adjust every published p-value without touching
  a single answer file.

Both are checked here rather than trusted to the docstring
([[feedback_an_asserted_invariant_is_not_a_check]]).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "eval" / "rq4_score.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(REPO / "src"), "PYTHONIOENCODING": "utf-8"}
    import os

    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, **env}, cwd=str(REPO), timeout=300,
    )


def test_an_unknown_arm_is_rejected_rather_than_dropped():
    r = _run(["--model", "phi4", "--arms", "hybrid_qwen3_0.6b_semanticc",
              "--out", "unused.md"])
    assert r.returncode != 0
    assert "unknown arm" in (r.stdout + r.stderr)


def test_a_nondefault_arm_set_cannot_write_the_published_report():
    """No --out, non-default arms => refuse, because family 1's Holm size moves."""
    r = _run(["--model", "phi4", "--arms",
              "hybrid_qwen3_0.6b_semantic,entity_lookup_semantic"])
    assert r.returncode != 0
    assert "refusing to write the published" in (r.stdout + r.stderr)


def test_the_entity_arms_are_known_but_not_default():
    """They must be addressable, and must not join the published five by default.

    Adding them to ARM_ORDER would have been the one-line change; it would also
    have silently re-adjusted every published verdict.
    """
    sys.path.insert(0, str(REPO / "tools" / "eval"))
    from rq4_score import ARM_ORDER, EXTRA_ARMS  # noqa: PLC0415

    assert "entity_lookup_semantic" in EXTRA_ARMS
    assert "entity_boost_semantic" in EXTRA_ARMS
    assert not set(EXTRA_ARMS) & set(ARM_ORDER)
    assert len(ARM_ORDER) == 5
