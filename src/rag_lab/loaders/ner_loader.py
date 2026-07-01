from __future__ import annotations

from typing import Any

from rag_lab.loaders.base import BaseLoader
from rag_lab.loaders.common import (
    make_resolution_id,
    parse_path,
    read_text,
    strip_document_header,
)
from rag_lab.registries import loader_registry
from rag_lab.schema import Resolution

_ner_tagger = None


def _tagger(engine: str):
    """Load the pythainlp NER tagger lazily (and once) — the model is heavy."""
    global _ner_tagger
    if _ner_tagger is None:
        from pythainlp.tag import NER

        _ner_tagger = NER(engine)
    return _ner_tagger


def _group_entities(tagged: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Collapse IOB word tags (B-PERSON, I-PERSON, O, ...) into entity spans."""
    entities: list[dict[str, str]] = []
    text = ""
    tag: str | None = None

    def flush() -> None:
        nonlocal text, tag
        if tag is not None and text.strip():
            entities.append({"text": text.strip(), "tag": tag})
        text, tag = "", None

    for word, label in tagged:
        if label.startswith("B-"):
            flush()
            tag = label[2:]
            text = word
        elif label.startswith("I-") and tag == label[2:]:
            text += word
        else:
            flush()
    flush()
    return entities


@loader_registry.register("ner")
class NERLoader(BaseLoader):
    """Runs Thai NER (pythainlp) and stores extracted entities in
    metadata['entities'] for entity-aware filtering/analysis."""

    def __init__(self, engine: str = "thainer") -> None:
        self.engine = engine

    def load(self, path: str) -> Resolution:
        text = strip_document_header(read_text(path))
        year, session, title = parse_path(path)
        entities = _group_entities(_tagger(self.engine).tag(text))
        metadata: dict[str, Any] = {
            "year": year,
            "session": session,
            "title": title,
            "entities": entities,
        }
        return Resolution(
            resolution_id=make_resolution_id(path, year, session, title),
            source_path=str(path),
            raw_text=text,
            year=year,
            session=session,
            title=title,
            metadata=metadata,
        )
