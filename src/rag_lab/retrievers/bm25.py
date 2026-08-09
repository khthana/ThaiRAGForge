from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from rag_lab.registries import retriever_registry
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.schema import Index, Query, RankedChunk


@retriever_registry.register("bm25")
class BM25Retriever(BaseRetriever):
    """Lexical BM25 over the index's per-chunk tokens. Corpus-relative: when run
    over a filtered sub-index it scores against that subset (Index.select carries
    the aligned lexical tokens along).

    The scorer is memoised on the Index (see `_scorer`), so the corpus-sized
    build is paid once per loaded Index instead of once per query. This changes
    only *when* work happens, never the ranking: the same BM25Okapi answers the
    same query with the same scores, so every persisted result reproduces."""

    @property
    def name(self) -> str:
        return "bm25"

    @staticmethod
    def _scorer(index: Index) -> BM25Okapi:
        """The index's BM25Okapi, built once and memoised on the Index itself.

        Constructing it walks every chunk to build the term frequencies and IDF
        table, costing ~1.01s over 74,816 chunks -- so rebuilding it per query,
        as this did until 2026-08-08, put a fixed corpus-sized cost on every
        BM25 and hybrid retrieval regardless of embedder.

        **Quote that saving in seconds, not as a multiple of `get_scores`.**
        `rank_bm25` loops over query *terms* in Python, so scoring is linear in
        query length (~12 ms/token here) while the build is not: the ratio is
        ~26x for a 3-token query but ~4x for the 20-token Gold queries this
        project evaluates. This docstring quoted the 26x on 2026-08-08 without
        naming its token count; re-measured 2026-08-09 (`cost_latency_pareto.py`,
        1007ms build vs 252ms scoring at 20 tokens median). The ~1s removed from
        every query is the part that does not depend on query shape.

        The memo is validated by *identity of the token list*, not by presence:
        `Index.lexical` is a mutable field, and serving a scorer built from a
        different token list would silently return scores for the wrong rows --
        the exact shape of silent corruption this project keeps finding. A
        replaced `lexical` therefore rebuilds; a sub-index from `Index.select`
        is a new Index and starts empty. The one case identity cannot see is
        `lexical` mutated *in place*; nothing does that (every producer --
        ArtifactStore.load, pipeline.build_index, Index.select -- hands over a
        freshly built list), and the alternative, hashing 74k token lists on
        every query, would cost more than the rebuild it guards.
        """
        memo = index.lexical_scorer
        if memo is not None and memo[0] is index.lexical:
            return memo[1]
        scorer = BM25Okapi(index.lexical)
        index.lexical_scorer = (index.lexical, scorer)
        return scorer

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        if index.lexical is None:
            raise ValueError(
                "BM25Retriever needs a lexical index; rebuild the index for BM25"
            )
        if query.tokens is None:
            raise ValueError("BM25Retriever requires query.tokens")
        if not index.chunks:
            return []

        scores = self._scorer(index).get_scores(query.tokens)
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
