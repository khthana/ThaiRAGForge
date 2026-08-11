"""Pins how G1b/G1c classify an RQ4 answer that predates the `num_ctx` field.

Written with the check (2026-08-10). G1b's whole point is that it can FAIL on an
answer nobody regenerated -- but on the artifacts as they stand it reports **0
truncated of 750**, because the 81 known bad cells were regenerated and now carry
the field. A check whose failing branch is never exercised would pass identically
if it silently ignored its evidence, so the truncated branch is exercised here
instead ([[feedback_anchor_a_check_where_the_mechanism_is_live]]).

The other half is the three-way split. Two buckets would let "no evidence" read
as "no truncation" ([[feedback_undefined_is_not_zero]]); each test below states
which of the three a case must land in, in both directions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "eval"))

import audit_pipeline_invariants as A  # noqa: E402

ARM = "bm25_semantic"
VARIANT_DIR = "phi4"          # -> "sentence_cap"


@pytest.fixture()
def bench(tmp_path, monkeypatch):
    """A contexts dir + a raw probe cache, both where the helper looks for them.

    The answer file itself is never opened -- the helper is asked about the
    *prompt* -- so only its path shape has to be real.
    """
    contexts = tmp_path / "contexts"
    (contexts / ARM).mkdir(parents=True)
    monkeypatch.setattr(A, "_RQ4_CONTEXTS", contexts)
    monkeypatch.chdir(tmp_path)         # the cache is read cwd-relative
    (tmp_path / "data" / "results").mkdir(parents=True)

    def write(query: str, chars: int) -> Path:
        blocks = ([{"label": 1, "resolution_id": "r1", "text": "ก" * chars}]
                  if chars else [])
        (contexts / ARM / f"{query}.json").write_text(
            json.dumps({"query": "คำถาม", "blocks": blocks}), encoding="utf-8")
        return Path("data/rq4/answers") / VARIANT_DIR / ARM / f"{query}.json"

    def cache(entries: dict) -> None:
        (tmp_path / "data" / "results" / "rq4_truncated_cells_raw.json").write_text(
            json.dumps(entries), encoding="utf-8")

    return write, cache


def test_a_cached_probe_carrying_the_signature_is_reported_truncated(bench):
    """The failing branch, which no artifact on disk currently exercises.

    `prompt_eval_count == 8192 // 2 + 2` is what ollama reports when it HAS cut
    the prompt, so a pre-fix answer whose prompt probed to exactly that was
    generated from evidence-stripped context -- G1b must say so and fail.
    """
    write, cache = bench
    path = write("q000", 4_000)                    # ~12 kB: past the byte bound
    cache({f"sentence_cap/{ARM}/q000": {"n_8192": A._rq4_truncated_to(8192)}})

    by_bound, by_probe, unmeasured, truncated = A._rq4_prompt_fit_evidence([path])

    assert truncated == [f"{VARIANT_DIR}/{ARM}/q000 (fed 4,098 tok at num_ctx=8,192)"]
    assert (by_bound, by_probe, unmeasured) == (0, 1, [])


def test_a_probe_short_of_the_signature_proves_the_prompt_fitted(bench):
    write, cache = bench
    path = write("q001", 4_000)
    cache({f"sentence_cap/{ARM}/q001": {"n_8192": 5_000}})

    by_bound, by_probe, unmeasured, truncated = A._rq4_prompt_fit_evidence([path])

    assert (by_bound, by_probe, unmeasured, truncated) == (0, 1, [], [])


def test_a_second_probe_decides_a_prompt_whose_true_length_is_near_the_signature(bench):
    """A prompt of ~4,098 real tokens reports ~4,098 whether it was cut or not.

    `rq4_find_truncated_answers.py` disambiguates those with a probe at 16,384:
    a cut prompt reads far larger there, an uncut one reads the same. Believe the
    second probe where it exists, or the boundary cases invert.
    """
    write, cache = bench
    uncut = write("q002", 4_000)
    cut = write("q003", 4_000)
    cache({f"sentence_cap/{ARM}/q002": {"n_8192": 4_098, "n_16384": 4_100},
           f"sentence_cap/{ARM}/q003": {"n_8192": 4_098, "n_16384": 9_400}})

    _, _, _, truncated = A._rq4_prompt_fit_evidence([uncut, cut])

    assert [t.split()[0] for t in truncated] == [f"{VARIANT_DIR}/{ARM}/q003"]


def test_a_long_prompt_with_no_probe_is_unmeasured_not_clean(bench):
    """The bucket the sharpening exists to keep separate.

    Nothing here says this prompt was truncated -- and nothing says it wasn't.
    It must not be absorbed into either count.
    """
    write, cache = bench
    path = write("q004", 4_000)
    cache({})

    by_bound, by_probe, unmeasured, truncated = A._rq4_prompt_fit_evidence([path])

    assert unmeasured == [f"{VARIANT_DIR}/{ARM}/q004"]
    assert (by_bound, by_probe, truncated) == (0, 0, [])


def test_the_byte_bound_clears_a_short_prompt_with_no_probe_at_all(bench):
    """The only evidence that needs no GPU and no cache.

    A prompt under 8,192 UTF-8 bytes cannot exceed 8,192 tokens for any
    byte-level BPE vocabulary, so it provably fitted the old default.
    """
    write, cache = bench
    path = write("q005", 0)                        # closed-book shape: no documents
    cache({})

    by_bound, by_probe, unmeasured, truncated = A._rq4_prompt_fit_evidence([path])

    assert (by_bound, by_probe, unmeasured, truncated) == (1, 0, [], [])


def test_the_screen_constant_is_not_used_as_evidence(bench):
    """0.95 chars/token would clear all 759 unmeasured answers at a stroke.

    It is an observed minimum with headroom, not a bound
    ([[feedback_an_observed_extreme_is_not_a_bound]]), and this project has
    published a wrong blast radius from exactly that mistake. A 4,000-char Thai
    prompt sits far under 8,192 by that screen and must still come back
    unmeasured.
    """
    write, cache = bench
    path = write("q006", 4_000)
    cache({})

    assert 4_000 / 0.95 < 8_192                    # the screen would clear it
    assert A._rq4_token_upper_bound("ก" * 4_000) > 8_192

    _, _, unmeasured, _ = A._rq4_prompt_fit_evidence([path])
    assert unmeasured == [f"{VARIANT_DIR}/{ARM}/q006"]
