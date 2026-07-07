from __future__ import annotations

from rag_lab.loaders.base import BaseLoader
from rag_lab.loaders.common import (
    make_resolution_id,
    parse_path,
    read_text,
    strip_mapping_tables,
)
from rag_lab.registries import loader_registry
from rag_lab.schema import Resolution


@loader_registry.register("plain")
class PlainLoader(BaseLoader):
    """Reads an OCR'd Markdown resolution and derives basic metadata from its
    corpus path. Page markers are left in the text for the chunker; no OCR
    cleaning or enrichment beyond stripping Curriculum/SKILL Mapping tables
    (checkbox grids with no retrieval value -- see strip_mapping_tables) --
    this is the raw baseline otherwise."""

    def load(self, path: str) -> Resolution:
        raw_text = strip_mapping_tables(read_text(path))
        year, session, title = parse_path(path)
        return Resolution(
            resolution_id=make_resolution_id(path, year, session, title),
            source_path=str(path),
            raw_text=raw_text,
            year=year,
            session=session,
            title=title,
        )
