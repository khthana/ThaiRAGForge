"""Cycle F — build_index copies resolution metadata onto each chunk, so the
MetadataFilter can filter on year/session (and, once #6 lands, faculty)."""
from __future__ import annotations

from rag_lab.chunkers import FixedSizeChunker
from rag_lab.embedders import HashingEmbedder
from rag_lab.pipeline import build_index
from rag_lab.retrievers.filters import MetadataFilter
from rag_lab.schema import Resolution


def test_build_index_propagates_resolution_metadata_to_chunks():
    resolutions = [
        Resolution(
            resolution_id="r1",
            source_path="r1.md",
            raw_text="ก ข ค ง จ",
            year="2569",
            session="3",
            metadata={"faculty": "วิศวกรรม"},
        ),
        Resolution(
            resolution_id="r2",
            source_path="r2.md",
            raw_text="a b c",
            year="2568",
            session="1",
        ),
    ]
    index = build_index(resolutions, FixedSizeChunker(chunk_size=100), HashingEmbedder())

    r1_chunks = [c for c in index.chunks if c.resolution_id == "r1"]
    assert all(c.metadata["year"] == "2569" for c in r1_chunks)
    assert all(c.metadata["session"] == "3" for c in r1_chunks)
    assert all(c.metadata["faculty"] == "วิศวกรรม" for c in r1_chunks)

    # and the filter can now narrow the built index by that metadata
    only_2569 = MetadataFilter({"year": "2569"}).apply(index)
    assert {c.resolution_id for c in only_2569.chunks} == {"r1"}
