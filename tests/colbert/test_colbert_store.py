"""Both directions of the ColBERT artifact's alignment invariant (the L family).

I1 has a row correspondence to check; late interaction has only a cumulative
sum, and every way of getting it wrong produces finite, plausible scores. So each
rule here is exercised in the failing direction too -- a PASS-only gate is not
evidence, the rule this repo applies to `qualify_reranker_model.py` and the
pylate cross-check.

Two of these are load-bearing beyond their own line:

* **L1b** is the only check that can see the two-artifacts-from-different-days
  shape. `L1a` compares the artifact against itself, so it passes for any pair of
  builds of the same corpus; `test_l1b_catches_a_reordering_l1a_cannot_see` is
  what says the two are not the same check.
* **L3** exists because `np.maximum.reduceat` does not raise on an empty segment,
  it reads the next one -- `test_reduceat_really_borrows_the_next_document`
  demonstrates the corruption the check is written against, so the check states a
  measured mechanism rather than a suspicion.
"""
from __future__ import annotations

import numpy as np
import pytest

from rag_lab.colbert.store import ColbertArtifact, ColbertStore, verify_alignment

DIM = 4


def _unit(rows: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(rows, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float16)


def _artifact(lengths=(3, 2, 4), *, ids=None, vecs=None, meta=None) -> ColbertArtifact:
    lengths = np.asarray(lengths, dtype=np.int64)
    return ColbertArtifact(
        chunk_ids=list(ids or [f"c{i}" for i in range(len(lengths))]),
        vecs=_unit(int(lengths.sum())) if vecs is None else vecs,
        lengths=lengths,
        meta={"dim": DIM, "doc_maxlen": 300, **(meta or {})},
    )


def _verdicts(art, index_ids=None):
    return {name.split()[0]: ok for name, ok, _ in verify_alignment(art, index_ids)}


def _ids(art):
    return list(art.chunk_ids)


def test_a_clean_artifact_passes_every_check():
    art = _artifact()
    assert all(_verdicts(art, _ids(art)).values())


def test_l1a_catches_a_truncated_length_vector():
    art = _artifact()
    art.chunk_ids = art.chunk_ids + ["c3"]
    assert _verdicts(art, _ids(art))["L1a"] is False


def test_l1b_catches_a_reordering_l1a_cannot_see():
    """The whole point of storing chunk_ids: same count, same packing, different
    documents. Every other check still passes, which is what makes this the one
    with teeth."""
    art = _artifact()
    shuffled = ["c1", "c0", "c2"]
    v = _verdicts(art, shuffled)
    assert v["L1b"] is False
    assert v["L1a"] is True and v["L2"] is True and v["L3"] is True


def test_l1b_reports_unchecked_rather_than_passing_when_no_index_is_supplied():
    """A check that passes for lack of an input reads as clean; it is not."""
    name, ok, detail = verify_alignment(_artifact())[1]
    assert name.startswith("L1b") and ok is False and "UNCHECKED" in detail


def test_l2_catches_a_packing_mismatch():
    art = _artifact()
    art.vecs = art.vecs[:-1]
    assert _verdicts(art, _ids(art))["L2"] is False


def test_l3_catches_a_zero_length_document():
    art = _artifact(lengths=(3, 0, 4))
    art.vecs = _unit(7)
    assert _verdicts(art, _ids(art))["L3"] is False


def test_reduceat_really_borrows_the_next_document():
    """The mechanism L3 is written against, stated as a measurement.

    `reduceat` does not raise on an empty segment and does not return -inf; it
    returns the row at that offset, which belongs to the *next* document. The
    borrowed score is finite and plausible, so nothing downstream can notice.
    """
    sim = np.array([[1.0], [2.0], [9.0]], dtype=np.float32)
    borrowed = np.maximum.reduceat(sim, np.array([0, 2, 2]), axis=0)
    assert borrowed[1, 0] == 9.0          # the empty document scores the next one's row
    assert borrowed[1, 0] == borrowed[2, 0]


def test_l4_catches_a_chunk_longer_than_the_cap():
    """A length over the cap means the cap was not applied, so the truncation
    rates `colbert_length_profile.md` reports -- the pilot's stated confound --
    describe a run that did not happen."""
    art = _artifact(lengths=(3, 2, 4), meta={"doc_maxlen": 3})
    assert _verdicts(art, _ids(art))["L4"] is False


def test_l4_reports_unchecked_when_the_cap_was_never_recorded():
    art = _artifact(meta={"doc_maxlen": None})
    name, ok, detail = next(c for c in verify_alignment(art, _ids(art)) if c[0].startswith("L4"))
    assert ok is False and "UNCHECKED" in detail


def test_l5_catches_unnormalised_vectors():
    """MaxSim terms are cosines only if both sides are unit-norm; this is the
    `unnormalised` control of the qualification gate, arriving on disk."""
    art = _artifact()
    art.vecs = (np.asarray(art.vecs, dtype=np.float32) * 3.0).astype(np.float16)
    assert _verdicts(art, _ids(art))["L5"] is False


def test_l5_catches_a_non_finite_vector():
    art = _artifact()
    v = np.asarray(art.vecs, dtype=np.float32)
    v[0, 0] = np.nan
    art.vecs = v.astype(np.float16)
    assert _verdicts(art, _ids(art))["L5"] is False


def test_l6_catches_a_width_that_disagrees_with_the_recorded_dim():
    art = _artifact(meta={"dim": DIM + 1})
    assert _verdicts(art, _ids(art))["L6"] is False


def test_round_trip_preserves_every_field_and_passes(tmp_path):
    art = _artifact()
    ColbertStore().save(tmp_path, art.chunk_ids, art.vecs, art.lengths, art.meta)
    back = ColbertStore().load(tmp_path)
    assert back.chunk_ids == art.chunk_ids
    assert back.meta == art.meta
    assert np.array_equal(np.asarray(back.lengths), np.asarray(art.lengths))
    assert np.array_equal(np.asarray(back.vecs), np.asarray(art.vecs))
    assert all(_verdicts(back, _ids(art)).values())


def test_load_memory_maps_by_default(tmp_path):
    """The audit reads `lengths` and a sample of `vecs`; a chunker's artifact is
    ~1.8 GB, and a check nobody can afford to run is not a check."""
    art = _artifact()
    ColbertStore().save(tmp_path, art.chunk_ids, art.vecs, art.lengths, art.meta)
    assert isinstance(ColbertStore().load(tmp_path).vecs, np.memmap)
    assert not isinstance(ColbertStore().load(tmp_path, mmap=False).vecs, np.memmap)


def test_maxsim_refuses_the_artifacts_verify_rejects():
    """The two layers agree: what L2/L3 fail on, `maxsim` will not score."""
    from rag_lab.colbert.scoring import maxsim

    q = _unit(2, seed=1).astype(np.float32)
    with pytest.raises(ValueError):
        maxsim(q, _unit(7), np.array([3, 0, 4], dtype=np.int64))
    with pytest.raises(ValueError):
        maxsim(q, _unit(6), np.array([3, 2, 4], dtype=np.int64))
