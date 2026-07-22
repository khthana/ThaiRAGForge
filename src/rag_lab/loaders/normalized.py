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
from rag_lab.text_normalize import normalize_thai_text


@loader_registry.register("normalized")
class NormalizedLoader(BaseLoader):
    """Identical to PlainLoader, plus one extra step: normalize_thai_text()
    applied after strip_mapping_tables, before chunking. Exists to isolate
    RQ3's normalization ablation (docs/research-framework-gap-analysis.md) --
    see rag_lab.text_normalize for exactly what "normalize" means here, and
    why the eval script must apply the same function to queries too."""

    def load(self, path: str) -> Resolution:
        raw_text = normalize_thai_text(strip_mapping_tables(read_text(path)))
        year, session, title = parse_path(path)
        return Resolution(
            resolution_id=make_resolution_id(path, year, session, title),
            source_path=str(path),
            raw_text=raw_text,
            year=year,
            session=session,
            title=title,
        )
