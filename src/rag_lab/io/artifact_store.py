"""Persist / load an Index artifact.

Layout under one directory (per ADR-0001, this is the serialized index output):
- ``chunks.parquet``   — chunk rows (metadata JSON-encoded per row)
- ``embeddings.npy``   — the (n, dim) float matrix, aligned to chunk order
- ``meta.json``        — how the index was built (chunker params, embedder id)
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

    def load(self, directory: str | Path) -> Index:
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
        embeddings = np.load(d / _EMBEDDINGS)
        meta = json.loads((d / _META).read_text(encoding="utf-8"))
        lexical_path = d / _LEXICAL
        lexical = (
            json.loads(lexical_path.read_text(encoding="utf-8"))
            if lexical_path.exists()
            else None
        )
        return Index(chunks=chunks, embeddings=embeddings, meta=meta, lexical=lexical)
