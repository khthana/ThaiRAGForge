"""MaxSim: the late-interaction score.

    score(q, d) = sum over query tokens of ( max over document tokens of q_i . d_j )

Both sides are L2-normalised by `ColbertEncoder`, so each inner term is a cosine
and the score is bounded by `query_maxlen`. It is **not** normalised by query
length -- ColBERT sums rather than averages, and every query here is padded to
the same `query_maxlen` anyway, so the sum is comparable across queries of one
run and not across runs with different `query_maxlen`.

Documents arrive packed (see `encoder.encode_documents`): one `(total, dim)`
matrix plus a length per document. `np.maximum.reduceat` reduces per segment in
one pass, which is what makes an exhaustive scan over ~30M token vectors
affordable and removes the need for an ANN index at this corpus size.
"""
from __future__ import annotations

import numpy as np


def offsets_from_lengths(lengths: np.ndarray) -> np.ndarray:
    """Start row of each document in the packed matrix."""
    off = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=off[1:])
    return off


def maxsim(q: np.ndarray, vecs: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """`(n_docs,)` scores for one query.

    `q` is `(query_maxlen, dim)`, `vecs` is `(total_tokens, dim)`, and
    `lengths` sums to `total_tokens`. A zero-length document is rejected rather
    than scored: `reduceat` silently reads the *next* segment's first row for an
    empty one, which would attribute another document's best match to it.
    """
    if vecs.shape[0] != int(lengths.sum()):
        raise ValueError(
            f"{vecs.shape[0]} packed vectors for {int(lengths.sum())} claimed tokens")
    if (lengths <= 0).any():
        raise ValueError("zero-length document: reduceat would borrow the next document's row")
    sim = vecs.astype(np.float32) @ q.astype(np.float32).T   # (total, query_maxlen)
    per_doc = np.maximum.reduceat(sim, offsets_from_lengths(lengths), axis=0)
    return per_doc.sum(axis=1)


def maxsim_reference(q: np.ndarray, docs: list[np.ndarray]) -> np.ndarray:
    """The definition, written out one document at a time.

    Deliberately naive and deliberately kept: the packed `reduceat` path above is
    an optimisation whose failure mode (a segment boundary off by one) produces
    plausible scores, so it needs something independent to be checked against.
    """
    return np.asarray(
        [float((d.astype(np.float32) @ q.astype(np.float32).T).max(axis=0).sum()) for d in docs],
        dtype=np.float64,
    )
