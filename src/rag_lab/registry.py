"""A tiny decorator registry, one instance per pluggable stage.

Lets a new Loader/Chunker/Embedder/Retriever register itself by name so config
and the runner can select strategies by string without being edited (ADR/PRD:
Open/Closed — add a file + register, never touch the runner).
"""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self.kind} strategy already registered: {name!r}")
            self._items[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} strategy: {name!r}; known: {sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return list(self._items)
