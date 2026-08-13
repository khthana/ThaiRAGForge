"""Ingest one built `Index` into a Qdrant **server** collection (dense + sparse).

Scope: the pilot (2026-08-13) ingested one collection -- the `person` route's
shipped hybrid target, `chunker_compare_full/plain__sentence__local__bf8b7ebb`
(sentence chunker x BAAI/bge-m3). The other **three** routed collections were
ingested with this same script later the same day, on the user's go-ahead after
the pilot's and the concurrency run's numbers; all four are verified end to end
against the shipped router by `tools/eval/qdrant_routed_check.py` ->
`data/results/qdrant_routed_check.md`. Re-run both after any index rebuild:
a collection is a copy of an `Index`'s rows and goes stale with it.

Three things this script is careful about, each of which would otherwise make the
subsequent measurement return a clean, plausible, wrong answer:

1. **The server, not embedded mode.** `QdrantClient(path=...)` is exact brute
   force (`LocalCollection.search` -> `calculate_distance` over every vector +
   `np.argsort`) while *reporting* an `HnswConfig`, so an ANN-vs-exact question
   asked there can only answer "identical". `--url` is required; there is no
   embedded path here on purpose.
2. **HNSW must actually exist.** Qdrant only builds a vector index once a segment
   passes `indexing_threshold` (default 20,000). Below that it plain-scans, and
   the "ANN" arm would again be exact by accident. The threshold is set
   explicitly and `S5` blocks on `indexed_vectors_count > 0` -- if it never
   becomes non-zero, ingestion FAILS rather than handing the pilot a vacuous arm.
3. **The lexical arm is reproduced by construction, not re-implemented.** The
   sparse vectors hold precomputed `BM25Okapi` weights (see
   `rag_lab.retrievers.bm25_sparse`), fitted here on the *same* `index.lexical`
   token lists `BM25Retriever` scores at eval time, so the served arm is the
   measured arm. The engine's own IDF is deliberately not used -- no
   `Modifier.IDF` anywhere below.

Row alignment is the load-bearing invariant: **point id == row index** in
`index.chunks` / `index.embeddings` / `index.lexical`, which is what lets S3/S4
compare a retrieved point against the array row it came from. The payload carries
`row` explicitly rather than trusting the id, so a re-ingest that changed ids
would be caught rather than silently re-keying every vector.

The vocabulary is written next to the collection (`data/qdrant/<collection>/
vocab.json`) and is what `QdrantSparseRetriever` reads at query time. It is a
sorted enumeration, not a hash: ~200k terms hashed into 2^32 expects ~4.7
collisions, and a collision voids the exactness claim while still returning a
plausible ranking.

Usage (verify before running, per the project's standing rule):

    PYTHONPATH=src .venv/Scripts/python.exe tools/eval/qdrant_pilot_ingest.py \
        --index data/index/chunker_compare_full/plain__sentence__local__bf8b7ebb \
        --url http://127.0.0.1:6333

`--dry-run` does everything except talk to the server (fits BM25, builds the
vocabulary, runs the formula self-check) so the expensive half can be verified
first.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.retrievers import bm25_sparse

REPO = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = REPO / "data/index/chunker_compare_full/plain__sentence__local__bf8b7ebb"
VOCAB_ROOT = REPO / "data/qdrant"

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# Qdrant's own default is 20,000 vectors per segment; stated explicitly so the
# ANN arm's existence is a decision recorded here rather than a default that
# could change under us.
INDEXING_THRESHOLD = 20_000


def fit_scorer(index) -> BM25Okapi:
    """Fit on `index.lexical` -- the same pre-tokenized Thai token lists
    `BM25Retriever` scores against, so the served arm cannot drift from the
    measured one through a tokenizer difference."""
    if index.lexical is None:
        raise SystemExit("index has no lexical field; cannot build the sparse arm")
    return BM25Okapi(index.lexical)


def check_formula(index, scorer, vocab, n_queries: int = 5, seed: int = 0) -> dict:
    """S1: the sparse dot product reproduces `BM25Okapi.get_scores` on this
    corpus, at f32 storage precision.

    `tests/retrievers/test_bm25_sparse.py` pins the algebra on a toy corpus; this
    re-checks it on the real 57k-chunk one, where the vocabulary is ~4 orders of
    magnitude larger and float error has room to accumulate. Queries are drawn
    from actual chunks so they carry realistic term repetition.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(index.lexical), size=n_queries, replace=False)
    queries = [list(index.lexical[int(r)])[:20] for r in rows]
    expected = np.array([scorer.get_scores(q) for q in queries])
    qvecs = [dict(zip(*bm25_sparse.query_sparse_vector(q, vocab))) for q in queries]

    # One pass over the corpus scoring all queries at once: building 57k document
    # vectors per query would be five times the work for the same answer.
    got = np.zeros_like(expected)
    for i in range(scorer.corpus_size):
        idx, val = bm25_sparse.document_sparse_vector(scorer, i, vocab)
        # f32 exactly as the engine stores it, so the tolerance reported here is
        # the one the pilot will actually see.
        val32 = np.asarray(val, dtype=np.float32)
        for qi, qvec in enumerate(qvecs):
            got[qi, i] = sum(
                v * w for j, v in zip(idx, val32.tolist()) if (w := qvec.get(j))
            )
    denom = np.maximum(np.abs(expected), 1e-9)
    worst = float(np.max(np.abs(got - expected) / denom))
    ok = worst < 1e-5
    return {"check": "S1 sparse dot == get_scores (f32)", "ok": ok, "detail": f"worst rel err {worst:.3e} over {n_queries} queries"}


def check_vocabulary(scorer, vocab) -> dict:
    """S2: the vocabulary covers exactly the fitted scorer's terms and is a
    bijection onto 0..N-1. A missing term scores 0 forever; a duplicated id
    merges two terms' weights."""
    ok = set(vocab) == set(scorer.idf) and sorted(vocab.values()) == list(range(len(vocab)))
    return {"check": "S2 vocabulary is a bijection over the fitted terms", "ok": ok, "detail": f"{len(vocab):,} terms"}


def build_points(index, scorer, vocab, models):
    """Yield one `PointStruct` per row. Point id == row index (see module docstring)."""
    emb = index.embeddings
    for i, chunk in enumerate(index.chunks):
        indices, values = bm25_sparse.document_sparse_vector(scorer, i, vocab)
        yield models.PointStruct(
            id=i,
            vector={
                DENSE_VECTOR: emb[i].tolist(),
                SPARSE_VECTOR: models.SparseVector(indices=indices, values=values),
            },
            payload={
                "row": i,
                "chunk_id": chunk.chunk_id,
                "resolution_id": chunk.resolution_id,
                "page": chunk.page,
                "text": chunk.text,
            },
        )


def ingest(client, models, collection: str, index, scorer, vocab, batch_size: int) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=index.embeddings.shape[1], distance=models.Distance.COSINE
            )
        },
        # No `modifier` on purpose: the stored values are already full BM25
        # document weights, so the engine must do a plain dot product. Asking it
        # for IDF would re-score the arm with a different IDF than BM25Okapi's.
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=INDEXING_THRESHOLD),
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="resolution_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    batch = []
    sent = 0
    t0 = time.time()
    for point in build_points(index, scorer, vocab, models):
        batch.append(point)
        if len(batch) >= batch_size:
            client.upsert(collection_name=collection, points=batch, wait=True)
            sent += len(batch)
            batch = []
            print(f"  upserted {sent:,}/{len(index.chunks):,} ({time.time() - t0:.0f}s)", flush=True)
    if batch:
        client.upsert(collection_name=collection, points=batch, wait=True)
        sent += len(batch)
    print(f"  upserted {sent:,}/{len(index.chunks):,} ({time.time() - t0:.0f}s)", flush=True)


def check_count(client, collection: str, index) -> dict:
    """S3: every row arrived. A short collection silently truncates recall."""
    got = client.count(collection_name=collection, exact=True).count
    return {"check": "S3 point count == index rows", "ok": got == len(index.chunks), "detail": f"{got:,} vs {len(index.chunks):,}"}


def check_alignment(client, models, collection: str, index, n: int = 200, seed: int = 0) -> dict:
    """S4: the payload at point id i really is row i.

    This is the invariant every later measurement rests on: if ids and rows
    disagree, the ANN-vs-exact comparison compares two different documents and
    reports a disagreement that is not about ANN at all.
    """
    rng = np.random.default_rng(seed)
    ids = sorted(int(x) for x in rng.choice(len(index.chunks), size=min(n, len(index.chunks)), replace=False))
    records = client.retrieve(collection_name=collection, ids=ids, with_payload=True, with_vectors=True)
    by_id = {int(r.id): r for r in records}
    bad = []
    for i in ids:
        rec = by_id.get(i)
        chunk = index.chunks[i]
        if rec is None or rec.payload["row"] != i or rec.payload["chunk_id"] != chunk.chunk_id:
            bad.append(i)
            continue
        stored = np.asarray(rec.vector[DENSE_VECTOR], dtype=np.float32)
        # Cosine collections normalise on write, so compare direction, not norm.
        a, b = stored, index.embeddings[i].astype(np.float32)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        if cos < 1 - 1e-5:
            bad.append(i)
    return {"check": "S4 payload/vector at id i == row i", "ok": not bad, "detail": f"{len(ids)} sampled, {len(bad)} mismatched"}


def check_hnsw(client, collection: str, timeout_s: int = 900) -> dict:
    """S5: an HNSW index actually got built.

    Without this the ANN arm is a full scan wearing an ANN's name, and the pilot
    would report "ANN is identical to exact" for a reason that has nothing to do
    with ANN (cf. feedback_a_guards_precondition_biases_its_own_test).
    """
    t0 = time.time()
    indexed = 0
    status = ""
    while time.time() - t0 < timeout_s:
        info = client.get_collection(collection)
        indexed = info.indexed_vectors_count or 0
        status = str(getattr(info.status, "value", info.status))
        if indexed > 0 and status == "green":
            break
        time.sleep(5)
    return {
        "check": "S5 HNSW built (indexed_vectors_count > 0)",
        "ok": indexed > 0,
        "detail": f"{indexed:,} indexed, status {status}, after {time.time() - t0:.0f}s",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--url", default="http://127.0.0.1:6333")
    ap.add_argument("--collection", default=None, help="defaults to the index directory name")
    ap.add_argument("--grpc-port", type=int, default=6334)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true", help="fit + self-check the sparse arm, touch no server")
    args = ap.parse_args()

    collection = args.collection or args.index.name
    print(f"index      : {args.index}")
    print(f"collection : {collection}")

    index = ArtifactStore().load(args.index)
    print(f"rows       : {len(index.chunks):,}  dim {index.embeddings.shape[1]}")

    t0 = time.time()
    scorer = fit_scorer(index)
    vocab = bm25_sparse.build_vocabulary(scorer)
    print(f"bm25 fitted: {len(vocab):,} terms ({time.time() - t0:.0f}s)")

    checks = [check_vocabulary(scorer, vocab), check_formula(index, scorer, vocab)]

    if not args.dry_run:
        from qdrant_client import QdrantClient, models

        # gRPC for ingestion: 57k x 1024 floats over REST is ~700 MB of JSON
        # text, where gRPC sends them as binary. Transport is an ingestion-speed
        # concern only -- it does not touch what is stored, and the retrievers
        # query over REST.
        client = QdrantClient(
            url=args.url, grpc_port=args.grpc_port, prefer_grpc=True, timeout=300
        )
        ingest(client, models, collection, index, scorer, vocab, args.batch_size)
        checks.append(check_count(client, collection, index))
        checks.append(check_alignment(client, models, collection, index))
        checks.append(check_hnsw(client, collection))

        vocab_path = VOCAB_ROOT / collection / "vocab.json"
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        print(f"vocab      : {vocab_path} ({vocab_path.stat().st_size / 1e6:.1f} MB)")

    print("\nself-checks")
    for c in checks:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']} -- {c['detail']}")
    failed = [c for c in checks if not c["ok"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
