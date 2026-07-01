"""Shared OCR page-segmentation for all chunkers.

`## Page N` marker lines are hard boundaries: every chunker segments on them so a
chunk never spans two pages. Text before the first marker belongs to page 1.
"""
from __future__ import annotations

import re

_PAGE_MARKER = re.compile(r"^\s*##\s*Page\s+(\d+)\s*$")


def segment_by_page(text: str) -> list[tuple[int, str]]:
    segments: list[tuple[int, str]] = []
    current_page = 1
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            segments.append((current_page, body))

    for line in text.splitlines():
        marker = _PAGE_MARKER.match(line)
        if marker:
            flush()
            current_page = int(marker.group(1))
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return segments
