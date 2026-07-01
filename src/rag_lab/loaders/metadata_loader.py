from __future__ import annotations

from rag_lab.loaders.base import BaseLoader
from rag_lab.loaders.common import (
    make_resolution_id,
    parse_path,
    read_source_url,
    read_text,
    strip_document_header,
)
from rag_lab.registries import loader_registry
from rag_lab.schema import Resolution


@loader_registry.register("metadata")
class MetadataLoader(BaseLoader):
    """Cleans the OCR `# Document:` header, attaches the source PDF URL (from the
    sibling `_LINK.txt`), and mirrors structural fields into `metadata` so they
    propagate onto chunks for filtering."""

    def load(self, path: str) -> Resolution:
        text = strip_document_header(read_text(path))
        year, session, title = parse_path(path)
        source_url = read_source_url(path)
        metadata = {
            "year": year,
            "session": session,
            "title": title,
            "source_url": source_url,
        }
        return Resolution(
            resolution_id=make_resolution_id(path, year, session, title),
            source_path=str(path),
            raw_text=text,
            year=year,
            session=session,
            title=title,
            source_url=source_url,
            metadata=metadata,
        )
