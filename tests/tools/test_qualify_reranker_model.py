"""Pure-logic tests for qualify_reranker_model.py's gate rules and probes.

The gate exists because a broken reranker returns *plausible* numbers rather
than crashing (`gte-multilingual-reranker-base`, 2026-08-09: dead RoPE tables,
correct-looking Thai ranking, bit-identical scores for a sentence and its
reversal). So the gate itself has to be trustworthy, and it can fail in two
directions:

* too strict -- G1's first version rejected the published anchor, because
  XLM-R's `token_type_ids` is legitimately all zeros. A gate that rejects the
  model every published number came from is worse than no gate.
* too loose -- the recurring failure in this repo (C4 in
  `audit_pipeline_invariants.py`, D2's exemptions in `audit_doc_claims.py`): a
  check that can no longer say FAIL has gone vacuous without anyone noticing.

Both directions are pinned below. No model downloads and no GPU: the buffer
rule takes a hand-built module, and the text probes are plain strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "eval"))
import qualify_reranker_model as q  # noqa: E402


def _mod(**buffers) -> nn.Module:
    """A module whose buffers are all non-persistent, i.e. absent from the
    state_dict -- exactly the class of buffer that a meta-device load leaves
    uninitialised, and the only class `audit_buffers` inspects."""
    m = nn.Module()
    for name, t in buffers.items():
        m.register_buffer(name, t, persistent=False)
    return m


class TestBufferAudit:
    def test_accepts_a_real_arange(self):
        ok, bad = q.audit_buffers(_mod(position_ids=torch.arange(512).reshape(1, 512)))
        assert (ok, bad) == (True, [])

    def test_accepts_all_zero_token_type_ids(self):
        # The anchor's own buffer. XLM-R has one segment type, so all-zeros is
        # correct -- this is the case that broke the first version of the rule.
        ok, _ = q.audit_buffers(_mod(token_type_ids=torch.zeros(1, 512, dtype=torch.long)))
        assert ok

    def test_rejects_a_pointer_sized_integer(self):
        # gte's actual failure: 512 slots holding 3633978736640, which cannot
        # index anything.
        ok, bad = q.audit_buffers(
            _mod(position_ids=torch.full((1, 512), 3633978736640, dtype=torch.long)))
        assert not ok and "out of range" in bad[0]

    def test_rejects_a_negative_index(self):
        ok, bad = q.audit_buffers(_mod(idx=torch.full((8,), -99, dtype=torch.long)))
        assert not ok and "out of range" in bad[0]

    def test_rejects_an_all_zero_float_buffer(self):
        # gte's cos_cached/sin_cached: rotary encoding multiplying by zero.
        ok, bad = q.audit_buffers(_mod(cos_cached=torch.zeros(512, 64)))
        assert not ok and "identically zero" in bad[0]

    def test_rejects_a_non_finite_float_buffer(self):
        ok, bad = q.audit_buffers(_mod(inv_freq=torch.tensor([1.0, float("nan"), 3.0])))
        assert not ok and "not finite" in bad[0]

    def test_accepts_a_healthy_float_buffer(self):
        ok, _ = q.audit_buffers(_mod(inv_freq=1.0 / (10000 ** (torch.arange(0, 64, 2) / 64))))
        assert ok

    def test_skips_persistent_buffers(self):
        # A buffer that IS in the checkpoint was loaded from it, so it is not
        # this failure mode -- and gating on it would flag legitimate content
        # such as a learned all-zero bias.
        m = nn.Module()
        m.register_buffer("loaded", torch.zeros(4), persistent=True)
        assert "loaded" in dict(m.state_dict())
        assert q.audit_buffers(m) == (True, [])

    def test_skips_empty_buffers(self):
        assert q.audit_buffers(_mod(empty=torch.empty(0, dtype=torch.long))) == (True, [])

    def test_a_two_slot_buffer_may_hold_its_own_length(self):
        # max(numel, 2) exists so a 1-element buffer holding 1 is not flagged;
        # without it every scalar shape counter would look like garbage.
        assert q.audit_buffers(_mod(n=torch.tensor([1])))[0]


class TestPositionProbe:
    def test_reversal_is_a_permutation_not_a_different_text(self):
        # If the two strings differed in *content*, G2 could pass on a
        # bag-of-words model and the gate would be worthless.
        assert sorted(q.D_ORDER.split()) == sorted(q.D_ORDER_REV.split())

    def test_reversal_is_not_a_palindrome(self):
        # ...and if they were identical, G2 could never pass at all.
        assert q.D_ORDER != q.D_ORDER_REV

    def test_the_two_names_swap_places(self):
        # The probe's meaning: "ก replaces ข" vs "ข replaces ก". A model that
        # cannot tell these apart cannot answer this corpus's person queries.
        assert q.D_ORDER.index("นาย ก") < q.D_ORDER.index("นาย ข")
        assert q.D_ORDER_REV.index("ข") < q.D_ORDER_REV.index("ก")


class TestRelevanceProbe:
    def test_the_queried_entity_is_present_in_the_match_only(self):
        assert "สมชาย ใจดี" in q.D_MATCH and "สมชาย" not in q.D_OTHER
        assert "วิศวกรรมไฟฟ้า" in q.D_PROG and "วิศวกรรมไฟฟ้า" not in q.D_OTHER

    def test_the_distractor_is_shared_by_both_directions(self):
        # One distractor for two queries, so G3 cannot be passed by a model
        # that has merely learned to dislike one particular sentence.
        assert q.D_OTHER not in (q.D_MATCH, q.D_PROG)

    def test_the_padding_probe_is_much_longer_than_the_short_pair(self):
        # G5 only tests padding if batching the two actually forces padding.
        assert len(q.D_LONG) > 20 * len(q.D_MATCH)


class TestTolerances:
    @pytest.mark.parametrize("tol", [q.POS_MIN_DELTA, q.PAD_TOL])
    def test_tolerances_are_far_above_fp32_noise_and_far_below_the_failures(self, tol):
        # Loose on purpose: attention reorders additions when the batch shape
        # changes (~1e-6), while the failures are gross -- a dead position
        # encoding gives exactly 0.0.
        assert 1e-5 <= tol <= 1e-2


class TestDefaultSet:
    def test_the_anchor_is_first_and_needs_no_remote_code(self):
        assert q.DEFAULT_MODELS[0] == ("BAAI/bge-reranker-v2-m3", False)

    def test_both_known_broken_models_stay_in_the_default_set(self):
        # This is the both-directions guarantee. Drop them and the report's
        # "the gate is exercised in both directions" line silently disappears
        # (`both_ways` goes None) and the gate becomes PASS-only.
        names = [m for m, _ in q.DEFAULT_MODELS]
        assert "Alibaba-NLP/gte-multilingual-reranker-base" in names
        assert "jinaai/jina-reranker-v2-base-multilingual" in names

    def test_every_measured_model_is_in_the_default_set(self):
        # The comparison script's S8 looks each of its models up in this
        # script's report; a model measured but never qualified would make S8
        # fail rather than silently pass, but keeping the sets aligned here
        # means that never happens in the first place.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "eval"))
        import reranker_model_comparison as rmc  # noqa: PLC0415

        names = {m for m, _ in q.DEFAULT_MODELS}
        assert {m for m, _, _ in rmc.MODELS} <= names
