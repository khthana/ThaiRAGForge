"""`ArtifactStore.load` reads chunks.parquet a batch at a time.

Motivation is measured, not stylistic: the whole-table read
(`pq.read_table(...).to_pydict()`) held **395 MB** of a 57k-chunk index's rows
against streaming's **253 MB**, at no cost in time
(`data/results/serving_cache_memory.md` 1b/1c and the curve in
`artifact_store._BATCH_ROWS`). With four indices resident in a serving process
that is over half a gigabyte of arena that never has to be grown.

Correctness is the whole risk, so these tests are about **equality with the
implementation it replaced**, not about the saving. Two properties carry it:
every row is present with its fields intact, and the rows are in file order --
batching is the one way a loader can silently reorder or drop a tail, and
`Index.embeddings` is row-aligned to `Index.chunks` (invariant `I1`), so a
reordering here would mispair every vector in the index without raising
anything.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from rag_lab.io import artifact_store as store_mod
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.schema import Chunk, Index


def _index(n: int) -> Index:
    """`n` chunks whose fields are all distinguishable per row.

    `chunk_index` deliberately runs BACKWARDS: file order is what is being
    tested, and a fixture numbered 0..n-1 would let a loader that sorted (or one
    whose batches came back in any order at all) pass by coincidence.
    """
    chunks = [
        Chunk(
            chunk_id=f"c-{i:04d}",
            resolution_id=f"2568/ครั้งที่ {i % 7}/เรื่อง ที่ {i}",
            text=f"ข้อความไทย บรรทัดที่ {i} — ตรวจสอบ unicode",
            chunk_index=n - i,
            page=(i % 13) + 1,
            metadata={"year": 2560 + (i % 9), "row": i, "tags": [f"t{i}"]},
        )
        for i in range(n)
    ]
    return Index(
        chunks=chunks,
        embeddings=np.arange(max(n, 1) * 3, dtype=np.float32).reshape(max(n, 1), 3)[:n],
        meta={"combo_id": "streaming-fixture", "n": n},
    )


def _rows(chunks):
    return [
        (c.chunk_id, c.resolution_id, c.text, c.chunk_index, c.page, c.metadata)
        for c in chunks
    ]


def _whole_table(path):
    """The implementation this replaced, kept here as the reference arm."""
    cols = pq.read_table(path).to_pydict()
    return [
        (
            cols["chunk_id"][i],
            cols["resolution_id"][i],
            cols["text"][i],
            int(cols["chunk_index"][i]),
            int(cols["page"][i]),
            json.loads(cols["metadata"][i]),
        )
        for i in range(len(cols["chunk_id"]))
    ]


# ------------------------------------------------------- equality with legacy
@pytest.mark.parametrize("n", [0, 1, 6, 7, 8, 20])
def test_streaming_equals_the_whole_table_read(tmp_path, monkeypatch, n):
    """Batch size 7 against 0..20 rows, so the parametrisation straddles the
    boundary in both directions -- an off-by-one on the last partial batch is
    exactly the bug a single round-number fixture would miss."""
    monkeypatch.setattr(store_mod, "_BATCH_ROWS", 7)
    d = tmp_path / f"idx{n}"
    ArtifactStore().save(_index(n), d)

    loaded = ArtifactStore().load(d, with_embeddings=False)
    assert _rows(loaded.chunks) == _whole_table(d / "chunks.parquet")
    assert len(loaded.chunks) == n


def test_the_fixture_really_spans_several_batches(tmp_path, monkeypatch):
    """The tests above are only evidence while the mechanism is live: at a batch
    size no smaller than the file every one of them would pass against a loader
    that never batched at all."""
    monkeypatch.setattr(store_mod, "_BATCH_ROWS", 7)
    d = tmp_path / "idx"
    ArtifactStore().save(_index(20), d)
    n_batches = sum(
        1 for _ in pq.ParquetFile(d / "chunks.parquet").iter_batches(batch_size=7)
    )
    assert n_batches > 1, "batching is inert here, so the equality tests prove nothing"


def test_file_order_is_preserved_across_batches(tmp_path, monkeypatch):
    """`Index.embeddings` is row-aligned to `Index.chunks`, so a reordering here
    mispairs every vector in the index and raises nothing."""
    monkeypatch.setattr(store_mod, "_BATCH_ROWS", 3)
    d = tmp_path / "idx"
    saved = _index(20)
    ArtifactStore().save(saved, d)

    loaded = ArtifactStore().load(d, with_embeddings=False)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in saved.chunks]
    # Descending by construction: a sort would have to invert it to pass.
    assert [c.chunk_index for c in loaded.chunks] == list(range(20, 0, -1))


def test_a_round_trip_preserves_every_field_and_its_type(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "_BATCH_ROWS", 4)
    d = tmp_path / "idx"
    saved = _index(11)
    ArtifactStore().save(saved, d)

    loaded = ArtifactStore().load(d)
    for got, want in zip(loaded.chunks, saved.chunks):
        assert got.chunk_id == want.chunk_id
        assert got.resolution_id == want.resolution_id
        assert got.text == want.text  # Thai text, em dash, round-tripped
        assert got.metadata == want.metadata
        assert isinstance(got.chunk_index, int)
        assert isinstance(got.page, int)
        assert isinstance(got.metadata, dict)
    assert np.array_equal(loaded.embeddings, saved.embeddings)


# ---------------------------------------------------------------- the mechanism
def test_load_never_materialises_the_whole_table(tmp_path, monkeypatch):
    """The saving IS not calling `read_table`. Nothing about the rows would
    change if a future edit reverted to it, so the equality tests above cannot
    detect that -- this one can."""
    d = tmp_path / "idx"
    ArtifactStore().save(_index(9), d)

    def _refuse(*a, **k):  # pragma: no cover - the assertion is that it is unused
        raise AssertionError("load() read the whole table instead of streaming")

    monkeypatch.setattr(store_mod.pq, "read_table", _refuse)
    assert len(ArtifactStore().load(d, with_embeddings=False).chunks) == 9


def test_the_shipped_batch_size_is_smaller_than_a_real_index(tmp_path):
    """A default of pyarrow's own 65,536 would put every shipped index (57k-75k
    chunks) in one or two batches and give back barely a tenth of the saving --
    the measured curve in `_BATCH_ROWS` is what this constant is for."""
    assert store_mod._BATCH_ROWS <= 4096
