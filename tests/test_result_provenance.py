"""A persisted result must name the index that produced it, not just the combo.

`BuildCombo.id` hashes loader x chunker x embedder and *not* the corpus, so the
same combo id is the directory name of several different indices -- a 10-file
smoke fixture and the 2,854-file corpus among them. Before 2026-08-09 a result
file recorded only that id, which is why the stale BM25/hybrid cache incident
could not be seen in the data: nothing in a result said which index answered it.

These pin the chain that closes it -- manifest -> Index.provenance ->
RetrievalResult -- including the two ways it could quietly stop working: the
derived field leaking back into the artifact on save, and the new fields making
the ~24k pre-existing result files unreadable.
"""
from __future__ import annotations

import json

import numpy as np

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.io.artifact_store import ArtifactStore
from rag_lab.pipeline import build_index, retrieve
from rag_lab.retrievers import DenseRetriever
from rag_lab.schema import Chunk, Index, Resolution, RetrievalResult

from tests.fakes import BagOfWordsEmbedder


def _built(texts: list[str]):
    resolutions = [
        Resolution(resolution_id=f"r{i}", source_path=f"r{i}.md", raw_text=t)
        for i, t in enumerate(texts)
    ]
    embedder = BagOfWordsEmbedder()
    index = build_index(
        resolutions, FixedSizeChunker(chunk_size=100, chunk_overlap=0), embedder
    )
    return index, embedder


def _save(index, directory, docset_hash: str) -> None:
    ArtifactStore().save(index, directory)
    (directory / "manifest.json").write_text(
        json.dumps({"combo_id": "plain__fixed_size__bow__deadbeef",
                    "docset_hash": docset_hash}),
        encoding="utf-8",
    )


def test_load_stamps_the_index_with_where_it_came_from(tmp_path):
    index, _ = _built(["เอกสาร หนึ่ง"])
    _save(index, tmp_path, "abc123")

    loaded = ArtifactStore().load(tmp_path)

    assert loaded.provenance == {"index_dir": str(tmp_path), "docset_hash": "abc123"}


def test_load_without_a_manifest_leaves_provenance_none(tmp_path):
    """A build-cache directory has no manifest. Absent provenance must stay
    absent rather than being invented -- an unattributable result is a finding,
    a wrongly-attributed one is the bug this whole chain exists to prevent."""
    index, _ = _built(["เอกสาร หนึ่ง"])
    ArtifactStore().save(index, tmp_path)

    assert ArtifactStore().load(tmp_path).provenance is None


def test_provenance_never_round_trips_into_the_artifact(tmp_path):
    """It is derived at load time, so saving a loaded Index must not write it
    back into meta.json -- otherwise a copied index dir would keep claiming the
    path it was copied from."""
    index, _ = _built(["เอกสาร หนึ่ง"])
    _save(index, tmp_path, "abc123")
    loaded = ArtifactStore().load(tmp_path)

    elsewhere = tmp_path / "copy"
    ArtifactStore().save(loaded, elsewhere)

    assert json.loads((elsewhere / "meta.json").read_text(encoding="utf-8")) == index.meta
    assert ArtifactStore().load(elsewhere).provenance is None


def test_select_keeps_provenance():
    """A filtered view (MetadataFilter/EntityFilter) is still the same build, so
    a narrowed query's result must still name the index it was narrowed from."""
    index = Index(
        chunks=[Chunk(chunk_id="c0", resolution_id="r", text="t", chunk_index=0)],
        embeddings=np.zeros((1, 2)),
        meta={},
        provenance={"index_dir": "data/index/x/y", "docset_hash": "abc123"},
    )

    assert index.select([0]).provenance == index.provenance


def test_retrieve_records_the_index_on_the_result(tmp_path):
    index, embedder = _built(["เอกสาร หนึ่ง เรื่อง ทดสอบ", "เอกสาร สอง"])
    _save(index, tmp_path, "abc123")
    loaded = ArtifactStore().load(tmp_path)

    result = retrieve("ทดสอบ", loaded, embedder, DenseRetriever(), k=1)

    assert result.index_dir == str(tmp_path)
    assert result.docset_hash == "abc123"


def test_retrieve_leaves_provenance_none_for_an_unsaved_index():
    index, embedder = _built(["เอกสาร หนึ่ง เรื่อง ทดสอบ"])

    result = retrieve("ทดสอบ", index, embedder, DenseRetriever(), k=1)

    assert result.index_dir is None
    assert result.docset_hash is None


def test_two_indices_sharing_a_combo_id_are_told_apart_by_the_result(tmp_path):
    """The defect itself, in miniature: identical loader/chunker/embedder over
    two different corpora produce the *same* BuildCombo.id, so the combo id
    cannot say which index answered -- but the result now can."""
    full, embedder = _built(["เอกสาร หนึ่ง เรื่อง ทดสอบ", "เอกสาร สอง เรื่อง ทดสอบ"])
    smoke, _ = _built(["เอกสาร หนึ่ง เรื่อง ทดสอบ"])
    full_dir, smoke_dir = tmp_path / "full", tmp_path / "smoke"
    _save(full, full_dir, "full-hash")
    _save(smoke, smoke_dir, "smoke-hash")

    store = ArtifactStore()
    combo_ids, results = set(), []
    for d in (full_dir, smoke_dir):
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        combo_ids.add(manifest["combo_id"])
        results.append(
            retrieve("ทดสอบ", store.load(d), embedder, DenseRetriever(), k=1,
                     combination_id=f"{manifest['combo_id']}__dense")
        )

    # indistinguishable by combo id -- and therefore by result *filename*, which
    # results.py builds from combination_id + a hash of the query
    assert len(combo_ids) == 1
    assert len({r.combination_id for r in results}) == 1
    # but distinguishable by what the result records about its own origin
    assert {r.index_dir for r in results} == {str(full_dir), str(smoke_dir)}
    assert {r.docset_hash for r in results} == {"full-hash", "smoke-hash"}


def test_a_result_written_before_these_fields_still_loads():
    """~24k persisted results predate index_dir/docset_hash. They must keep
    validating, or every eval script that replays them breaks at once."""
    legacy = json.dumps({
        "query": "ทดสอบ",
        "combination_id": "plain__fixed_size__local__ceea7536__hybrid",
        "results": [{"chunk_id": "c0", "resolution_id": "r0", "page": 1,
                     "score": 1.0, "rank": 1, "text": "t"}],
        "top_k": 10,
        "retriever": "hybrid",
        "reranker": None,
    })

    result = RetrievalResult.model_validate_json(legacy)

    assert result.index_dir is None
    assert result.docset_hash is None
