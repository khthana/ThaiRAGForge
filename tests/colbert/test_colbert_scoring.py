"""Pin the packed MaxSim against the written-out definition, both ways.

`maxsim`'s failure mode is a segment boundary off by one, which produces
plausible scores rather than an error -- so the naive path is not a nicety, it is
the only independent check in the repo until pylate can be installed somewhere
(it pins `transformers<=5.3.0` against the 5.12.1 in `.venv`).
"""
from __future__ import annotations

import numpy as np
import pytest

from rag_lab.colbert import maxsim, maxsim_reference, offsets_from_lengths


def _docs(rng, lengths, dim=8):
    return [rng.standard_normal((int(n), dim)).astype(np.float32) for n in lengths]


def test_packed_equals_the_definition():
    rng = np.random.default_rng(0)
    lengths = np.array([1, 5, 3, 12, 2], dtype=np.int64)
    docs = _docs(rng, lengths)
    q = rng.standard_normal((4, 8)).astype(np.float32)

    packed = maxsim(q, np.vstack(docs), lengths)
    assert np.allclose(packed, maxsim_reference(q, docs), atol=1e-5)


def test_a_length_of_one_is_scored_as_itself():
    """The boundary `reduceat` is most likely to get wrong."""
    rng = np.random.default_rng(1)
    lengths = np.array([1, 1, 1], dtype=np.int64)
    docs = _docs(rng, lengths)
    q = rng.standard_normal((3, 8)).astype(np.float32)
    assert np.allclose(maxsim(q, np.vstack(docs), lengths), maxsim_reference(q, docs), atol=1e-5)


def test_offsets_start_at_zero_and_exclude_the_total():
    off = offsets_from_lengths(np.array([2, 3, 4], dtype=np.int64))
    assert off.tolist() == [0, 2, 5]


def test_a_row_count_mismatch_is_refused():
    q = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="packed vectors"):
        maxsim(q, np.zeros((6, 8), dtype=np.float32), np.array([3, 4], dtype=np.int64))


def test_a_zero_length_document_is_refused_not_scored():
    """`reduceat` would silently hand it the next document's first row."""
    q = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="zero-length"):
        maxsim(q, np.zeros((5, 8), dtype=np.float32), np.array([2, 0, 3], dtype=np.int64))
