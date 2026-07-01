"""Shared loader helpers: read the file and derive fields from the corpus path.

Corpus layout: <year พ.ศ.>/ครั้งที่ N/<เรื่อง>.md
"""
from __future__ import annotations

import re
from pathlib import Path

_DOCUMENT_HEADER = re.compile(r"^\s*#\s*Document:.*$", re.MULTILINE)


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8-sig")


def parse_path(path: str) -> tuple[str | None, str | None, str]:
    """Return (year, session, title) derived from the path/filename."""
    p = Path(path)
    title = p.stem
    session_match = re.search(r"(\d+)", p.parent.name)
    session = session_match.group(1) if session_match else None
    grandparent = p.parent.parent.name
    year = grandparent if re.fullmatch(r"\d{4}", grandparent) else None
    return year, session, title


def make_resolution_id(path: str, year: str | None, session: str | None, title: str) -> str:
    if year and session:
        return f"{year}/{session}/{title}"
    return str(Path(path).as_posix())


def strip_document_header(text: str) -> str:
    """Remove the OCR `# Document: <name>.pdf` header line(s); keep everything
    else (including `## Page N` markers the chunkers rely on)."""
    return _DOCUMENT_HEADER.sub("", text, count=1).lstrip("\n")


def read_source_url(path: str) -> str | None:
    """Return the URL from a sibling `<stem>_LINK.txt`, if present (provenance)."""
    p = Path(path)
    link = p.with_name(f"{p.stem}_LINK.txt")
    if link.exists():
        return link.read_text(encoding="utf-8-sig").strip()
    return None
