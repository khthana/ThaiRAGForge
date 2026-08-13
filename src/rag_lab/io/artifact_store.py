"""Persist / load an Index artifact.

Layout under one directory (per ADR-0001, this is the serialized index output):
- ``chunks.parquet``   — chunk rows (metadata JSON-encoded per row)
- ``embeddings.npy``   — the (n, dim) float matrix, aligned to chunk order
- ``meta.json``        — how the index was built (chunker params, embedder id)

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
        cols = pq.read_table(d / _CHUNKS).to_pydict()
        chunks = [
            Chunk(
                chunk_id=cols["chunk_id"][i],
                resolution_id=cols["resolution_id"][i],
                text=cols["text"][i],
                chunk_index=int(cols["chunk_index"][i]),
                page=int(cols["page"][i]),
                metadata=json.loads(cols["metadata"][i]),
            )
            for i in range(len(cols["chunk_id"]))
        ]
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
