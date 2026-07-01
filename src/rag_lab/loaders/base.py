from __future__ import annotations

from abc import ABC, abstractmethod

from rag_lab.schema import Resolution


class BaseLoader(ABC):
    """Reads a source document into a Resolution."""

    @abstractmethod
    def load(self, path: str) -> Resolution:
        ...
