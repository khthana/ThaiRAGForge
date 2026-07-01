from __future__ import annotations

import re
from pathlib import Path

from rag_lab.loaders.base import BaseLoader
from rag_lab.registries import loader_registry
from rag_lab.schema import Resolution


@loader_registry.register("plain")
class PlainLoader(BaseLoader):
    """Reads an OCR'd Markdown resolution verbatim and derives basic metadata
    from its corpus path (<year>/ครั้งที่ N/<title>.md). Page markers are left
    in the text for the chunker to segment on."""

    def load(self, path: str) -> Resolution:
        p = Path(path)
        raw_text = p.read_text(encoding="utf-8-sig")

        title = p.stem
        session_match = re.search(r"(\d+)", p.parent.name)
        session = session_match.group(1) if session_match else None
        grandparent = p.parent.parent.name
        year = grandparent if re.fullmatch(r"\d{4}", grandparent) else None

        resolution_id = (
            f"{year}/{session}/{title}" if year and session else str(p.as_posix())
        )
        return Resolution(
            resolution_id=resolution_id,
            source_path=str(p),
            raw_text=raw_text,
            year=year,
            session=session,
            title=title,
        )
