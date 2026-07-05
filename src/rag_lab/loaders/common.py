"""Shared loader helpers: read the file and derive fields from the corpus path.

Corpus layout: <year พ.ศ.>/ครั้งที่ N/<เรื่อง>.md — special sessions (วาระพิเศษ)
use an `s` suffix on the session number: ครั้งที่ Ns.

Each meeting folder may carry a `meeting_manifest.json`: the metadata source of
truth mapping each .md file to its full resolution title and source URL.
Filenames are truncated to ~100 chars by the download tooling, so the manifest
title (recovered from the agenda capture) wins over the filename; files absent
from the manifest fall back to filename-derived metadata.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_DOCUMENT_HEADER = re.compile(r"^\s*#\s*Document:.*$", re.MULTILINE)
_MANIFEST_NAME = "meeting_manifest.json"


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8-sig")


@lru_cache(maxsize=None)
def _meeting_manifest(dir_path: str) -> dict[str, dict]:
    """filename -> manifest entry for the folder's meeting_manifest.json.

    Cached per directory for the lifetime of the process; a missing or invalid
    manifest degrades to an empty mapping (filename-derived metadata)."""
    try:
        entries = json.loads((Path(dir_path) / _MANIFEST_NAME).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {e["file"]: e for e in entries if isinstance(e, dict) and e.get("file")}


def parse_path(path: str) -> tuple[str | None, str | None, str]:
    """Return (year, session, title) derived from the path/filename.

    Session keeps the special-meeting suffix (e.g. "3s") so a วาระพิเศษ never
    collides with the regular session of the same number. Title comes from the
    meeting manifest when the file is listed there."""
    p = Path(path)
    entry = _meeting_manifest(str(p.parent)).get(p.name)
    title = (entry or {}).get("title") or p.stem
    session_match = re.search(r"(\d+s?)", p.parent.name)
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
    """Return the resolution's source URL (provenance): the meeting-manifest
    entry when present, else the sibling `<stem>_LINK.txt`."""
    p = Path(path)
    entry = _meeting_manifest(str(p.parent)).get(p.name)
    if entry and entry.get("url"):
        return entry["url"]
    link = p.with_name(f"{p.stem}_LINK.txt")
    if link.exists():
        return link.read_text(encoding="utf-8-sig").strip()
    return None
