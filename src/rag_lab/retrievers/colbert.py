"""Late-interaction retrieval: MaxSim over a packed per-token artifact.

Registered rather than wired into the runner (ADR-0001), so the axis costs no
edit to anything already measured.

**Why the artifact is held by the retriever and not by `Index`.** Every other
retriever reads arrays that `Index` keeps row-aligned to `chunks`, and
`Index.select` re-slices them together. A ColBERT artifact cannot ride along
there: it has many rows per chunk, so `select` would have to translate row
indices through a cumulative sum, and a `select` that quietly *forgot* to would
leave scores that are finite, ordered and attributed to the wrong documents --
the exact silent-corruption shape `store.verify_alignment` exists for. So the
artifact stays outside `Index`, and this retriever **checks that the index in
front of it is the one the artifact was built from** on every call, refusing
rather than scoring when it is not. A sub-index from `select()` therefore raises
instead of returning plausible nonsense.

The check is memoised on the *identity* of `index.chunks`, the rule
`BM25Retriever._scorer` already applies: a replaced chunk list is a different
index and re-verifies; the one case identity cannot see is mutation in place,
which nothing here does.
"""
from __future__ import annotations

import numpy as np

from rag_lab.colbert.scoring import maxsim
from rag_lab.colbert.store import ColbertArtifact, ColbertStore, verify_alignment
from rag_lab.registries import retriever_registry
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.schema import Index, Query, RankedChunk


@retriever_registry.register("colbert")
class ColbertRetriever(BaseRetriever):
    """Scores every chunk by MaxSim against the query's token matrix.

    Exhaustive by design: at this corpus size `reduceat` over ~30M token vectors
    is affordable, so there is no ANN stage and therefore no approximation to
    confound the pilot's comparison against BM25 and dense.
    """

    def __init__(
        self,
        artifact_dir: str | None = None,
        artifact: ColbertArtifact | None = None,
    ) -> None:
        if (artifact_dir is None) == (artifact is None):
            raise ValueError("give exactly one of artifact_dir or artifact")
        self.artifact = artifact if artifact is not None else ColbertStore().load(artifact_dir)
        # No opt-out: the check is already paid once per index rather than per
        # query, so a `verify=False` hatch would buy nothing and would be the
        # easiest way to make the guard vacuous.
        self._verified_for: list | None = None

    @property
    def name(self) -> str:
        return "colbert"

    def _check(self, index: Index) -> None:
        """Refuse an index the artifact was not built from.

        `verify_alignment` is run once per distinct chunk list, not once per
        query: L5 samples 4,096 vectors and L1b walks every id, which is cheap
        per index and not per query.
        """
        if self._verified_for is index.chunks:
            return
        ids = [c.chunk_id for c in index.chunks]
        failed = [
            f"{name}: {detail}"
            for name, ok, detail in verify_alignment(self.artifact, ids)
            if not ok
        ]
        if failed:
            raise ValueError(
                "ColBERT artifact does not align with this index; "
                + "; ".join(failed)
            )
        self._verified_for = index.chunks

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        if query.vector is None:
            raise ValueError("ColbertRetriever requires query.vector")
        q = np.asarray(query.vector, dtype=np.float32)
        if q.ndim != 2:
            # A 1-D vector is a *dense* query. Reshaping one to (1, dim) would
            # silently turn MaxSim into plain max-pooled cosine and still return
            # a ranking, so this is refused rather than accommodated.
            raise ValueError(
                f"ColbertRetriever needs a (query_maxlen, dim) matrix, got shape {q.shape}"
            )
        if not index.chunks:
            return []
        self._check(index)

        scores = maxsim(q, self.artifact.vecs, np.asarray(self.artifact.lengths))
        order = np.argsort(-scores)[:k]
        return [
            RankedChunk(
                chunk_id=index.chunks[i].chunk_id,
                resolution_id=index.chunks[i].resolution_id,
                page=index.chunks[i].page,
                score=float(scores[i]),
                rank=rank + 1,
                text=index.chunks[i].text,
            )
            for rank, i in enumerate(order)
        ]
