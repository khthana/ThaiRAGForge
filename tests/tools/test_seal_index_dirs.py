"""`tools/seal_index_dirs.py` — the four-way classification, both directions.

`ArtifactStore.seal` records the four artifacts' stamps so a reader can tell a
finished build from a half-written one (`index_cache._settle`). Every index on
disk predates it, so this tool seals them; `audit_pipeline_invariants.py`'s I7
applies the same rule fleet-wide.

The verdict that carries the risk is **too-new**: sealing a directory does not
verify that its artifacts came from one build, it only records what is there.
Sealing something still being written would bless exactly the mixed pairing the
seal exists to catch, so a recently-touched directory must be refused rather
than sealed — and that is pinned here rather than trusted to a comment.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from rag_lab.io.artifact_store import ArtifactStore, seal
from rag_lab.schema import Chunk, Index

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "seal_index_dirs", REPO / "tools" / "seal_index_dirs.py"
)
sid = importlib.util.module_from_spec(_spec)
sys.modules["seal_index_dirs"] = sid
_spec.loader.exec_module(sid)


def _index(tmp_path: Path, name: str, tag: str = "a") -> Path:
    n = 4
    idx = Index(
        chunks=[
            Chunk(
                chunk_id=f"{name}-{i}",
                resolution_id=f"2568/1/เรื่อง {i}",
                text=f"ข้อความ {tag} {i}",
                chunk_index=i,
            )
            for i in range(n)
        ],
        embeddings=np.zeros((n, 3), dtype=np.float32),
        meta={"combo_id": name},
    )
    d = tmp_path / name
    ArtifactStore().save(idx, d)
    return d


def test_a_freshly_saved_index_is_sealed(tmp_path):
    d = _index(tmp_path, "one")
    assert sid.classify(d, 0.0) == ("sealed", "matches")


def test_a_directory_with_no_seal_is_unsealed(tmp_path):
    d = _index(tmp_path, "one")
    (d / "_complete.json").unlink()
    assert sid.classify(d, 0.0)[0] == "unsealed"


def test_an_artifact_changed_after_the_seal_is_stale(tmp_path):
    d = _index(tmp_path, "one")
    np.save(d / "embeddings.npy", np.ones((4, 3), dtype=np.float32))
    assert sid.classify(d, 0.0)[0] == "stale"


def test_a_recently_touched_directory_is_refused_not_sealed(tmp_path):
    """The load-bearing one: a directory still being written must never be
    sealed, or the tool blesses a mixed pairing instead of catching it."""
    d = _index(tmp_path, "one")
    (d / "_complete.json").unlink()
    verdict, detail = sid.classify(d, min_age_s=3600.0)
    assert verdict == "too-new"
    assert "3600" in detail


def test_sealing_makes_it_serveable_again(tmp_path):
    """The stale case is what an in-place writer leaves behind; re-sealing is
    the one-line obligation that clears it."""
    from rag_lab.io.index_cache import clear_index_cache, load_index_cached

    d = _index(tmp_path, "one")
    np.save(d / "embeddings.npy", np.ones((4, 3), dtype=np.float32))
    clear_index_cache()
    with pytest.raises(RuntimeError):
        load_index_cached(d)

    seal(d)
    assert sid.classify(d, 0.0)[0] == "sealed"
    assert float(load_index_cached(d).embeddings.sum()) == 12.0
    clear_index_cache()


def test_index_dirs_finds_them_at_any_depth(tmp_path):
    _index(tmp_path / "family_a", "one")
    _index(tmp_path / "family_b", "two")
    found = sid.index_dirs([str(tmp_path)])
    assert {p.name for p in found} == {"one", "two"}
