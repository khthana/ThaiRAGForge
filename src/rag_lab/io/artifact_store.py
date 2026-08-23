"""Persist / load an Index artifact.

Layout under one directory (per ADR-0001, this is the serialized index output):
- ``chunks.parquet``   — chunk rows (metadata JSON-encoded per row)
- ``embeddings.npy``   — the (n, dim) float matrix, aligned to chunk order
- ``meta.json``        — how the index was built (chunker params, embedder id)
- ``_complete.json``   — the writer declaring those files to be one build

``manifest.json`` sits in the same directory but is written by the runner
(``manifest.py``), not here. ``load`` reads it when present, purely to stamp the
returned Index with where it came from — see ``Index.provenance``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rag_lab.schema import Chunk, Index

_CHUNKS = "chunks.parquet"
_EMBEDDINGS = "embeddings.npy"
_META = "meta.json"
_LEXICAL = "lexical.json"
_MANIFEST = "manifest.json"
_SEAL = "_complete.json"

#: The files whose (mtime_ns, size) make up an index's identity on disk. ONE
#: copy of that list: `index_cache` stats exactly these, and `seal` records
#: exactly these, so the two can never drift into checking different things.
ARTIFACT_FILES = (_CHUNKS, _EMBEDDINGS, _META, _LEXICAL)


def artifact_stamp(directory: str | Path) -> list:
    """What the index's artifacts look like on disk right now.

    `[mtime_ns, size]` per file in `ARTIFACT_FILES` order, `None` for one that
    does not exist. Both halves matter: a same-second rewrite is common and
    size catches the truncation a coarse mtime would miss.

    **The bound, measured rather than assumed.** An earlier version of this
    docstring said a rewrite landing on the same nanosecond was "absurd". On
    Windows it is not: two `np.save` calls microseconds apart can produce an
    identical `st_mtime_ns`, which made a test that rewrote a same-shaped array
    fail 3 runs in 5. So the honest statement is that this stamp cannot see a
    rewrite that changes NEITHER size NOR mtime, and there is no third signal
    short of hashing every artifact on each cache hit -- the cost the stamp
    exists to avoid. That is acceptable because a real `save` takes seconds and
    does not reproduce a byte-identical size at an identical timestamp; it is
    pinned in the undetected direction by
    `tests/tools/test_seal_index_dirs.py::test_a_same_size_rewrite_at_an_unchanged_mtime_is_NOT_detectable`
    so the boundary is written down instead of resurfacing as a flake.

    A list rather than a tuple because it round-trips through JSON in the seal.
    """
    d = Path(directory)
    out: list = []
    for name in ARTIFACT_FILES:
        try:
            st = (d / name).stat()
            out.append([st.st_mtime_ns, st.st_size])
        except OSError:
            out.append(None)
    return out


def seal(directory: str | Path) -> list:
    """Declare the four artifacts in `directory` to be ONE build, and return
    the stamp recorded.

    **Why a writer-side seal exists at all.** `chunks.parquet` and
    `embeddings.npy` are row-aligned (`I1`) and are written one after the
    other, so between the two writes the directory is *stably inconsistent*:
    new chunks beside the previous build's vectors, with nothing changing.
    A reader stamping before and after its own read cannot see that -- the
    stamps agree, because the write is not overlapping the read, it happened
    just before it -- and the pairing is undetectable downstream (same row
    count, same dtype, wrong rows). MEASURED, not argued: reader threads
    hammering `load_index_cached` while a writer left a 150 ms gap between the
    two halves served a mixed Index on the majority of reads
    (`data/results/serving_concurrency.md` section 6), because one mixed read
    is then cached and handed out until the next write moves the stamp.

    So the writer states when it is finished, and the reader checks. A
    directory whose seal does not match its artifacts is mid-write or was
    edited out of band; either way it is not the build the writer declared.

    **Anything that rewrites an artifact in place must re-seal**, or the cache
    will refuse the directory -- `relabel_index_resolution_ids.py` is the one
    such writer in this repo today. `tools/seal_index_dirs.py` seals the
    indices that predate this.
    """
    d = Path(directory)
    stamp = artifact_stamp(d)
    (d / _SEAL).write_text(
        json.dumps({"artifacts": list(ARTIFACT_FILES), "stamp": stamp}),
        encoding="utf-8",
    )
    return stamp


def read_seal(directory: str | Path) -> list | None:
    """The stamp a writer last declared complete, or None for an UNSEALED
    directory (one written before seals existed).

    None is deliberately not "fine": it means the mixed-read window above
    cannot be detected for this directory, only the narrower overlapping-write
    one. `index_cache_info()` reports it per entry so the gap stays visible
    rather than being assumed closed.
    """
    try:
        data = json.loads((Path(directory) / _SEAL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stamp = data.get("stamp")
    return stamp if isinstance(stamp, list) else None


#: Rows per parquet batch. CHOSEN ON THE MEASURED CURVE, not on pyarrow's own
#: default of 65,536 -- which for a 57k-chunk index is the whole file in one
#: batch and gives back barely a tenth of the saving. Held RSS reading the
#: shipped `person` index, one process per point so no arena is inherited, and
#: the sweep run twice in opposite orders because a cold page cache would
#: otherwise be charged to whichever reader went first: 64 rows -> 176 MB,
#: 256 -> 184, 1,024 -> 195, 2,048 -> 208-215, 8,192 -> 271-275,
#: 65,536 -> 313-322, whole table -> 360. The curve is flat below ~1,024 (256
#: buys 11 MB more and 64 only 19), so this is the knee, not the minimum.
_BATCH_ROWS = 1_024


def _read_chunks(path: Path) -> list[Chunk]:
    """Every Chunk in a chunks.parquet, streamed a batch at a time.

    The obvious `pq.read_table(path).to_pydict()` costs **280 MB of the 581 MB**
    a loaded 57k-chunk index holds (`data/results/serving_cache_memory.md` 1b):
    182 MB of arrow buffers plus 98 MB of whole-column Python lists, of which
    deleting both returns only 2 MB because the rest stays in the allocator's
    arenas. That is transient work, not live data -- the Chunk objects it
    produces are 80 MB -- so it was pure waste held for the lifetime of a
    serving process, and the largest single lever in that report.

    Streaming holds one batch of columns at a time instead of every column of
    every row. Measured on that index, one child process per arm so no arena is
    inherited (`1c`): **379 MB -> 244 MB held**, at **no cost in time** (596 ms
    against 563, inside the run-to-run spread -- building 57k pydantic `Chunk`s
    dominates either way). End to end the four routed indices resident in one
    serving process hold **3,135 MB** (`data/results/serving_cache_memory.md`),
    less than 4x the per-index saving because the parent reuses arenas across
    loads -- so the per-index figure does not multiply, and the resident total is
    the one to quote.

    Two things to know before touching this. The batch size is on a measured
    knee, not a guess -- see `_BATCH_ROWS`. And the rows are byte-identical in
    file order: parquet preserves row order and `iter_batches` yields batches in
    file order, which matters more than it looks, because `Index.embeddings` is
    row-aligned to `Index.chunks` (invariant `I1`) so a reordering here would
    mispair every vector in the index without raising anything.
    `tests/io/test_artifact_store_streaming.py` pins both against the
    whole-table read rather than trusting the documentation.
    """
    chunks: list[Chunk] = []
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=_BATCH_ROWS):
        cols = batch.to_pydict()
        for i in range(len(cols["chunk_id"])):
            chunks.append(
                Chunk(
                    chunk_id=cols["chunk_id"][i],
                    resolution_id=cols["resolution_id"][i],
                    text=cols["text"][i],
                    chunk_index=int(cols["chunk_index"][i]),
                    page=int(cols["page"][i]),
                    metadata=json.loads(cols["metadata"][i]),
                )
            )
    return chunks


class ArtifactStore:
    def save(self, index: Index, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        table = pa.table(
            {
                "chunk_id": [c.chunk_id for c in index.chunks],
                "resolution_id": [c.resolution_id for c in index.chunks],
                "text": [c.text for c in index.chunks],
                "chunk_index": [c.chunk_index for c in index.chunks],
                "page": [c.page for c in index.chunks],
                "metadata": [
                    json.dumps(c.metadata, ensure_ascii=False) for c in index.chunks
                ],
            }
        )
        pq.write_table(table, d / _CHUNKS)
        np.save(d / _EMBEDDINGS, index.embeddings)
        (d / _META).write_text(
            json.dumps(index.meta, ensure_ascii=False), encoding="utf-8"
        )
        if index.lexical is not None:
            (d / _LEXICAL).write_text(
                json.dumps(index.lexical, ensure_ascii=False), encoding="utf-8"
            )
        # LAST, and never removed first: the stale seal left standing during
        # the writes above is exactly what tells a concurrent reader that this
        # directory is mid-build. Clearing it up front would present the
        # half-written state as merely UNSEALED, i.e. as legacy, which is the
        # one classification that does not refuse.
        seal(d)

    def load(self, directory: str | Path, *, with_embeddings: bool = True) -> Index:
        """`with_embeddings=False` returns an Index whose `embeddings` is an empty
        (0, 0) array, for a retriever that serves the vectors from elsewhere
        (`BaseRetriever.reads_index_rows is False`). It is a real saving, not a
        micro-optimisation -- `embeddings.npy` is ~234MB for a 57k x 1024
        collection, and avoiding it is the point of an engine-served path.

        Every other field is loaded normally, so the Index still identifies
        itself (`provenance`) and still carries its rows. Callers that only
        inspect metadata may use it too; `Index.select` already tolerates an
        empty matrix, so the filter path does not crash on one -- but it would
        silently fail to narrow the engine, which is why query_service refuses
        that combination outright rather than relying on this."""
        d = Path(directory)
        chunks = _read_chunks(d / _CHUNKS)
        embeddings = (
            np.load(d / _EMBEDDINGS) if with_embeddings else np.zeros((0, 0), dtype=np.float32)
        )
        meta = json.loads((d / _META).read_text(encoding="utf-8"))
        lexical_path = d / _LEXICAL
        lexical = (
            json.loads(lexical_path.read_text(encoding="utf-8"))
            if lexical_path.exists()
            else None
        )
        # Stamp the Index with where it was loaded from, so a result computed
        # against it can name the index rather than only the combo id (which does
        # not identify one -- BuildCombo.id omits the corpus). Absent for a build
        # cache directory, which has no manifest; None then propagates and the
        # result simply carries no provenance, as every pre-2026-08-09 result does.
        manifest_path = d / _MANIFEST
        provenance = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            provenance = {
                "index_dir": str(d),
                "docset_hash": manifest.get("docset_hash"),
            }
        return Index(
            chunks=chunks,
            embeddings=embeddings,
            meta=meta,
            lexical=lexical,
            provenance=provenance,
        )
