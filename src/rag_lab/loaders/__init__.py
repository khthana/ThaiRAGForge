from rag_lab.loaders.base import BaseLoader
from rag_lab.loaders.faculty_loader import FacultyLoader
from rag_lab.loaders.metadata_loader import MetadataLoader
from rag_lab.loaders.ner_loader import NERLoader
from rag_lab.loaders.person_loader import PersonLoader
from rag_lab.loaders.plain_loader import PlainLoader
from rag_lab.loaders.program_loader import ProgramLoader

__all__ = [
    "BaseLoader",
    "PlainLoader",
    "MetadataLoader",
    "NERLoader",
    "FacultyLoader",
    "ProgramLoader",
    "PersonLoader",
]
