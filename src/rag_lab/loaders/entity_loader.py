"""Composite loader for the entity-lookup retrieval mode
(docs/research-framework-gap-analysis.md, entity-lookup next plan): tags
each Resolution with people, programs, AND courses in one pass, plus the
full structural field set MetadataLoader already provides (year, session,
title, source_url).

PersonLoader/ProgramLoader/CourseLoader stay independently useful and
unchanged -- this is a new, parallel loader, not a replacement. An index
meant to back entity_lookup / query_indices(entity_boost=True) must be
built with this loader specifically: those two features check
metadata['people']/['programs']/['courses'] on every chunk, and a missing
key is indistinguishable from a genuine empty match (see
retrievers/entity_lookup.py and retrievers/filters.py's EntityFilter).
"""
from __future__ import annotations

from typing import Any

from rag_lab.loaders.base import BaseLoader
from rag_lab.loaders.common import (
    make_resolution_id,
    parse_path,
    read_source_url,
    read_text,
    strip_document_header,
    strip_mapping_tables,
)
from rag_lab.loaders.course_loader import match_courses
from rag_lab.loaders.person_loader import match_people
from rag_lab.loaders.program_loader import match_programs
from rag_lab.registries import loader_registry
from rag_lab.schema import Resolution


@loader_registry.register("entity_tags")
class EntityTagLoader(BaseLoader):
    """Tags each Resolution with metadata['people']/['programs']/['courses']
    in one pass, plus year/session/title/source_url (MetadataLoader's field
    set) so an entity-tagged index isn't metadata-poorer than any other
    built index."""

    def load(self, path: str) -> Resolution:
        text = strip_mapping_tables(strip_document_header(read_text(path)))
        year, session, title = parse_path(path)
        source_url = read_source_url(path)
        metadata: dict[str, Any] = {
            "year": year,
            "session": session,
            "title": title,
            "source_url": source_url,
            "people": match_people(text),
            "programs": match_programs(text),
            "courses": match_courses(text),
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
