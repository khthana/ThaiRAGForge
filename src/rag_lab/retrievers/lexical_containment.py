"""Arm L-prime: re-rank a hybrid pool by whether the query's entity appears in
the chunk. No GPU, no model, no training.

Measured in `tools/eval/reranker_trained_test.py` (2026-08-20, against index
rebuild #4) as **arm L-prime**, and it beats the shipped hard router
significantly on every metric at zero GPU cost:

    arm C (router, shipped)   recall@10 0.6811   MRR 0.8618   nDCG@10 0.7473
    arm L' (this retriever)             0.7300        0.9055           0.8188
    L' vs C                             +0.0489       +0.0437          +0.0714
    Holm-adj p (family 3, m=9)           0.0000        0.0084           0.0000

Three things must be read with those numbers, and none of them is optional.

**1. This is the DEPLOYABLE arm, and the published "arm L" is not.** The paper's
arm L reads the entity straight out of the gold query set, so it is handed the
very string the qrels were derived from, while arms C/D/T see only the query
text. That asymmetry is what this class exists to remove: it recovers the entity
with `router.detect_entities`, the shipped extractor. Losing the oracle string
costs `L' vs L` = -0.0138 recall@10, not significant (Holm 0.1950, CI rules out
a loss worse than 0.0304) -- so the oracle was not carrying the result, but the
deployable number is the one above, never arm L's 0.7438.

**2. The metric partly measures this retriever's own rule.** The qrels for
`person`/`program`/`faculty` were themselves derived by string containment
(`docs/eval-validity-threats.md` section 2), so a scorer implementing that rule
is closer to the labelling generator than to "relevance". The corpus owner's
domain judgement is that for this query shape -- a specific named person,
course, faculty or programme -- relevance genuinely does require the entity to
appear, which is why this is defensible to ship even though it inflates the
score. It is NOT evidence that lexical matching beats learned ranking in
general, and the trained cross-encoder separates from it on all three metrics
(`T vs L'` +0.0241 / +0.0516 / +0.0423, Holm 0.0290 / 0.0424 / 0.0042).

**3. `w` is 1.00, and that is why this class is so small.** The eval fuses the
containment signal as a fourth RRF term, `fused = (1-w)*hybrid + w*lexical`,
sweeping w. Leave-one-out picked **w=1.00 on all 106 folds** (oracle-on-all also
1.00, so there is no fitting premium), and at w=1.00 the hybrid term drops out
of the *score* entirely. What remains is exactly a stable partition: candidates
containing the entity first, everything else after, hybrid order preserved
inside each group -- because the eval breaks ties in the binary signal by hybrid
rank (`- 1e-6 * rank`), and a stable sort on a 2-valued key does the same thing.
This class therefore implements the partition directly rather than reproducing
an RRF sum whose weight annihilates one term; `tests/retrievers/
test_lexical_containment.py` pins the equivalence so the shortcut cannot drift
from the arm that was measured.

Cost: `detect_entities` is ~100 ms/query on this corpus (the course matcher is
~75 ms of it) on top of a hybrid fetch of `fetch_depth` instead of `k`. Against
a 475 ms routed hybrid query that is roughly +20%, and it touches no GPU -- the
layer that actually saturates under load (`data/results/qdrant_concurrency.md`).

Not wired as anyone's default: `dense`/`hybrid` still ship unchanged, and this
is opt-in by name, exactly like `qdrant_hybrid`.
"""
from __future__ import annotations

from rag_lab.registries import retriever_registry
from rag_lab.retrievers.base import BaseRetriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.router import detect_entities
from rag_lab.schema import Index, Query, RankedChunk
from rag_lab.text_match import collapse_ws, contains_phrase

#: The pool depth the arm was measured at. Not `k`: the whole point is that the
#: containment signal reorders candidates the router put below the cut.
DEFAULT_POOL = 50


@retriever_registry.register("lexical_containment")
class LexicalContainmentRetriever(BaseRetriever):
    """Hybrid, then a stable partition on entity containment. See module docstring."""

    def __init__(
        self,
        pool: int = DEFAULT_POOL,
        method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        fetch_depth: int | None = None,
        entity_detector=None,
        include_field_matches: bool = True,
    ) -> None:
        if pool < 1:
            raise ValueError(f"pool must be >= 1, got {pool}")
        self.pool = pool
        self._hybrid = HybridRetriever(
            method=method,
            rrf_k=rrf_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            fetch_depth=fetch_depth,
        )
        # Injectable so tests can substitute a fake without loading the real
        # dictionaries, the same convention router.detect_entities itself uses.
        #
        # `include_field_matches` defaults ON *here* and OFF in
        # `detect_entities` itself. A person types the field
        # ("วิศวกรรมคอมพิวเตอร์"), not the 60-character canonical, and without
        # it this arm has no signal to apply and silently degrades to plain
        # hybrid on exactly the queries a deployment sees most. It stays off in
        # `detect_entities` because that function also feeds `entity_lookup`
        # and `EntityFilter`, whose published numbers were measured without it.
        # Measured -- and it did NOT change nothing until it was narrowed.
        # This comment asserted "changes nothing on the Gold set" before anyone
        # ran it; gated on `not programs` the fallback fired on 5 of the 106
        # queries (the faculty ones whose faculty name contains a programme
        # field) and would have moved the published arm L' silently. The
        # fallback now fires only when the query resolved to nothing at all, so
        # the claim holds BY CONSTRUCTION (all 106 detect something) and is
        # pinned in both directions by tests/test_program_field_matching.py.
        # What remains is a deployment fix the eval is structurally unable to
        # score, not an improvement it declined to show.
        self._detect = entity_detector or (
            lambda text: detect_entities(text, include_field_matches=include_field_matches)
        )

    @property
    def name(self) -> str:
        return "lexical_containment"

    def retrieve(self, query: Query, index: Index, k: int) -> list[RankedChunk]:
        candidates = self._hybrid.retrieve(query, index, max(self.pool, k))

        found = self._detect(query.text)
        entities = [e for values in found.values() for e in values]
        if not entities:
            # Nothing recovered from the query: fall back to the hybrid order
            # untouched. That is the honest deployed behaviour -- a query naming
            # no known entity has no containment signal to apply, and inventing
            # one would be worse than passing the router's answer through.
            return _renumber(candidates[:k])

        # `any`, not `all`: detection legitimately returns several canonicals for
        # one query (two course codes for an `ENGLISH FOR ...` name), and
        # requiring all of them would punish the arm for the matcher's recall
        # rather than measure the containment signal.
        def hit(c: RankedChunk) -> bool:
            text = collapse_ws(c.text or "")
            return any(contains_phrase(text, e) for e in entities)

        # Stable: hybrid order is preserved inside each group. See docstring (3).
        ordered = sorted(candidates[: self.pool], key=lambda c: not hit(c))
        ordered += candidates[self.pool :]
        return _renumber(ordered[:k])


def _renumber(chunks: list[RankedChunk]) -> list[RankedChunk]:
    """Re-issue 1-based ranks after reordering; scores are the hybrid's own.

    The score is deliberately NOT recomputed into a containment score. It stays
    the fused hybrid score so a caller comparing this arm against `hybrid` sees
    the same scale, and so nothing downstream can mistake a 1/0 flag for a
    relevance estimate. Rank carries the reordering; score carries provenance.
    """
    return [
        RankedChunk(
            chunk_id=c.chunk_id,
            resolution_id=c.resolution_id,
            page=c.page,
            score=c.score,
            rank=i + 1,
            text=c.text,
        )
        for i, c in enumerate(chunks)
    ]
