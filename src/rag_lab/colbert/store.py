"""Persist / load a ColBERT artifact, and check the invariant it can violate.

`ArtifactStore` writes an `(n, dim)` matrix whose rows *are* the chunks, so I1
("embeddings rows == chunk rows == lexical rows") is the whole alignment story
there. Late interaction has no such row correspondence: the artifact is one
packed `(total_tokens, dim)` matrix plus a length per chunk, and the mapping
chunk -> rows exists only as a cumulative sum. Two consequences drive this file.

1. **The failure mode is silent.** `np.maximum.reduceat` over a wrong offset
   vector returns finite, plausible, well-ordered scores; it just attributes one
   document's best match to another. Nothing downstream can notice, which is the
   same shape as every silent-corruption bug this project has found by accident
   (see CLAUDE.md's invariant-audit bullet). So the invariant is checked, never
   assumed -- `encode_documents` already refuses to return a mismatched pair, but
   an artifact on disk was written by *some* run, not by this one.

2. **Shape agreement is not alignment.** `len(lengths) == len(chunks)` holds
   between any two builds of the same corpus, including a ColBERT artifact built
   before a re-OCR and a `chunks.parquet` written after it -- the
   two-artifacts-from-different-days shape. So the artifact stores its own
   `chunk_id` sequence and `L1b` compares it element-wise against the index it is
   about to be used with. That is the check with teeth; `L1a` only catches a
   truncated write.

Layout under one directory, alongside (not inside) the dense index:
- ``colbert_vecs.npy``       -- (total_tokens, dim) fp16, chunk-major
- ``colbert_lengths.npy``    -- (n_chunks,) int64, token count per chunk
- ``colbert_chunk_ids.parquet`` -- the chunk_id of each length, in order
- ``colbert_meta.json``      -- model, caps, dim, and the docset it was built on
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_VECS = "colbert_vecs.npy"
_LENGTHS = "colbert_lengths.npy"
_CHUNK_IDS = "colbert_chunk_ids.parquet"
_META = "colbert_meta.json"

# L5 reads vectors, and one chunker's artifact is ~1.8 GB; I4 samples embeddings
# for the same reason. Sampled, and the detail line says so.
_NORM_SAMPLE = 4096
_NORM_TOL = 2e-2  # fp16 storage of a unit vector, not an algorithmic tolerance


@dataclass
class ColbertArtifact:
    chunk_ids: list[str]
    vecs: np.ndarray            # (total_tokens, dim), possibly memory-mapped
    lengths: np.ndarray         # (n_chunks,)
    meta: dict = field(default_factory=dict)
    directory: str | None = None


class ColbertStore:
    def save(
        self,
        directory: str | Path,
        chunk_ids: list[str],
        vecs: np.ndarray,
        lengths: np.ndarray,
        meta: dict,
    ) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / _VECS, vecs)
        np.save(d / _LENGTHS, np.asarray(lengths, dtype=np.int64))
        pq.write_table(pa.table({"chunk_id": list(chunk_ids)}), d / _CHUNK_IDS)
        (d / _META).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def load(self, directory: str | Path, *, mmap: bool = True) -> ColbertArtifact:
        """`mmap` by default: an audit reads `lengths` and a sample of `vecs`, and
        paging in gigabytes to do that would make the check too expensive to run."""
        d = Path(directory)
        return ColbertArtifact(
            chunk_ids=list(pq.read_table(d / _CHUNK_IDS).column("chunk_id").to_pylist()),
            vecs=np.load(d / _VECS, mmap_mode="r" if mmap else None),
            lengths=np.load(d / _LENGTHS),
            meta=json.loads((d / _META).read_text(encoding="utf-8")),
            directory=str(d),
        )


def verify_alignment(
    art: ColbertArtifact,
    index_chunk_ids: list[str] | None = None,
    *,
    doc_maxlen: int | None = None,
    sample: int = _NORM_SAMPLE,
    seed: int = 0,
) -> list[tuple[str, bool, str]]:
    """The I1 analogue: `(name, ok, detail)` per check, in reporting order.

    `index_chunk_ids` is optional only so the artifact can be checked on its own
    (during a build, before an index is at hand); **L1b is the check that matters**
    and it reports UNCHECKED rather than PASS when they are not supplied, because
    a check that silently passes for lack of an input is worse than no check
    -- see CLAUDE.md on `0` being ambiguous between clean and not-examined.
    """
    out: list[tuple[str, bool, str]] = []
    lengths = np.asarray(art.lengths)
    n_ids, n_len = len(art.chunk_ids), len(lengths)
    total = int(lengths.sum()) if n_len else 0

    out.append((
        "L1a one length per chunk_id in the artifact",
        n_ids == n_len,
        f"{n_len} lengths for {n_ids} chunk_ids",
    ))

    if index_chunk_ids is None:
        out.append((
            "L1b artifact chunk_ids match the index, in order",
            False,
            "UNCHECKED -- no index chunk_ids supplied",
        ))
    else:
        if len(index_chunk_ids) != n_ids:
            detail = f"{n_ids} in artifact vs {len(index_chunk_ids)} in index"
            ok = False
        else:
            bad = [i for i, (a, b) in enumerate(zip(art.chunk_ids, index_chunk_ids)) if a != b]
            ok = not bad
            detail = "identical" if ok else f"{len(bad)} differ, first at row {bad[0]}"
        out.append(("L1b artifact chunk_ids match the index, in order", ok, detail))

    out.append((
        "L2 packing is exact (sum of lengths == packed rows)",
        int(art.vecs.shape[0]) == total,
        f"{int(art.vecs.shape[0])} rows for {total} claimed tokens",
    ))

    # A zero-length document makes `reduceat` read the *next* segment's first
    # row, i.e. silently score one document with another's best match.
    n_zero = int((lengths <= 0).sum()) if n_len else 0
    out.append((
        "L3 no zero-length document (reduceat would borrow the next one's row)",
        n_zero == 0,
        f"{n_zero} of {n_len}",
    ))

    if doc_maxlen is None:
        doc_maxlen = art.meta.get("doc_maxlen")
    if doc_maxlen is None:
        out.append(("L4 no chunk exceeds doc_maxlen", False, "UNCHECKED -- no doc_maxlen recorded"))
    else:
        over = int((lengths > int(doc_maxlen)).sum()) if n_len else 0
        longest = int(lengths.max()) if n_len else 0
        out.append((
            f"L4 no chunk exceeds doc_maxlen={doc_maxlen}",
            over == 0,
            f"{over} over, longest {longest}",
        ))

    # MaxSim terms are cosines only because both sides are L2-normalised. An
    # un-normalised artifact scores plausibly and ranks by magnitude -- it is the
    # `unnormalised` control the qualification gate rejects, arriving on disk.
    if art.vecs.shape[0] == 0:
        out.append(("L5 vectors finite and unit-norm (sampled)", False, "UNCHECKED -- no vectors"))
    else:
        rng = np.random.default_rng(seed)
        k = min(sample, int(art.vecs.shape[0]))
        rows = rng.choice(int(art.vecs.shape[0]), size=k, replace=False)
        v = np.asarray(art.vecs[np.sort(rows)], dtype=np.float32)
        norms = np.linalg.norm(v, axis=1)
        n_bad = int((~np.isfinite(v).all(axis=1)).sum() + (np.abs(norms - 1.0) > _NORM_TOL).sum())
        out.append((
            "L5 vectors finite and unit-norm (sampled)",
            n_bad == 0,
            f"{n_bad} bad of {k} sampled; |norm-1| max {float(np.abs(norms - 1.0).max()):.2e}",
        ))

    meta_dim = art.meta.get("dim")
    dim = int(art.vecs.shape[1]) if art.vecs.ndim == 2 else -1
    out.append((
        "L6 vector width matches the recorded dim",
        meta_dim is not None and int(meta_dim) == dim,
        f"vecs dim {dim} vs meta {meta_dim}",
    ))
    return out
