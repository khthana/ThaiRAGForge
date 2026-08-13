"""Express `BM25Okapi.get_scores` as a plain sparse dot product.

Why this exists: a served deployment wants the lexical arm inside the vector DB
(one round trip, no 253 ms of single-threaded Python per query -- see
`docs/paper-results-summary.md` on `rank_bm25` cost), but every published BM25
number in this project comes from `rank_bm25`'s `BM25Okapi` over *our* Thai
tokenization. Letting the engine supply its own BM25 would change the measured
arm and turn an infrastructure swap into an unmeasured quality change --
exactly the confound this project keeps getting hurt by. So the arm is
reproduced **by construction** instead, the same discipline as
`fetch_depth=None` reproducing k=n and `alpha=0.50` reproducing plain RRF.

The algebra. `BM25Okapi.get_scores` is

    score(d) = sum over q in query of
                 idf[q] * ( f(q,d)*(k1+1) ) / ( f(q,d) + k1*(1 - b + b*|d|/avgdl) )

Every factor depends on **the document and the term only** -- nothing in it
depends on the rest of the query. So the per-(doc, term) quantity can be
precomputed at ingestion into a sparse document vector, and the query side is
whatever multiplies it: `get_scores` iterates `query` as a *list*, so a term
repeated twice contributes twice, and the query weight is the term's **count**
in the query, not 1.0. The score is then a plain dot product.

Two consequences worth stating out loud, because both are ways this could go
quietly wrong:

* **Do not ask the engine to apply IDF** (Qdrant's `Modifier.IDF`, Elasticsearch's
  own similarity). Qdrant computes a different IDF than `BM25Okapi` does
  (`BM25Okapi` uses `log(N-df+0.5) - log(df+0.5)` and *floors negatives* at
  `epsilon * average_idf`), so an engine-side IDF would silently re-score the arm.
  The IDF here is read out of the fitted scorer's own table.
* **Terms absent from the query vocabulary contribute 0**, matching
  `self.idf.get(q) or 0`, so an unseen query term is dropped rather than
  defaulted.

What this module does *not* promise is bitwise equality with `get_scores`:
storage in an engine is f32 and the summation order differs (per-term over all
docs there, per-doc over terms here), so agreement is to ~1e-6 relative. That is
a measurement to report, not an assumption -- see
`tests/retrievers/test_bm25_sparse.py`, which pins both the formula and the
tolerance.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterator


def build_vocabulary(scorer: Any) -> dict[str, int]:
    """Map every term the fitted scorer knows to a stable integer id.

    Sorted rather than insertion-ordered so the same corpus yields the same ids
    on any machine -- the ids are persisted alongside the collection and a query
    built against a differently-ordered vocabulary would score garbage while
    still returning plausible results.
    """
    return {term: i for i, term in enumerate(sorted(scorer.idf))}


def document_sparse_vector(
    scorer: Any, doc_index: int, vocab: dict[str, int]
) -> tuple[list[int], list[float]]:
    """The precomputed BM25 weight of every term occurring in document `doc_index`.

    Terms with zero frequency contribute exactly 0 to `get_scores` and are
    omitted, which is what makes the vector sparse.
    """
    freqs = scorer.doc_freqs[doc_index]
    denom_len = scorer.k1 * (1.0 - scorer.b + scorer.b * scorer.doc_len[doc_index] / scorer.avgdl)
    indices: list[int] = []
    values: list[float] = []
    for term, f in freqs.items():
        idf = scorer.idf.get(term)
        if idf is None:
            continue
        indices.append(vocab[term])
        values.append(float(idf) * (f * (scorer.k1 + 1.0)) / (f + denom_len))
    return indices, values


def iter_document_sparse_vectors(
    scorer: Any, vocab: dict[str, int]
) -> Iterator[tuple[list[int], list[float]]]:
    for i in range(scorer.corpus_size):
        yield document_sparse_vector(scorer, i, vocab)


def query_sparse_vector(
    tokens: list[str], vocab: dict[str, int]
) -> tuple[list[int], list[float]]:
    """Query side of the dot product: each known term weighted by its **count**.

    The count (rather than 1.0) is what reproduces `get_scores`, which loops over
    the query as a list and therefore adds a repeated term's contribution once
    per occurrence.
    """
    counts = Counter(t for t in tokens if t in vocab)
    indices = [vocab[t] for t in counts]
    values = [float(c) for c in counts.values()]
    return indices, values


def dot(
    doc: tuple[list[int], list[float]], query: tuple[list[int], list[float]]
) -> float:
    """Reference dot product, used by tests and self-checks rather than in the
    query path (the engine computes this one for real)."""
    dvec = dict(zip(doc[0], doc[1]))
    return sum(dvec.get(i, 0.0) * v for i, v in zip(query[0], query[1]))
