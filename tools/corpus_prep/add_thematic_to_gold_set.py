# -*- coding: utf-8 -*-
"""Append every verified theme from llm_thematic_bootstrap.py's output into
config/eval/gold_query_set.yaml as entity_type: thematic entries.

Ground truth (relevant_resolution_ids) is already deterministically verified
by _verify_span in llm_thematic_bootstrap.py -- this script does no further
judgment, just reshapes academic_resolutions/llm_thematic_scan/*.json into the
QuerySetEntry-compatible yaml shape used by the existing program/person/
faculty_adjunct_aggregate entries (load_gold_query_set only reads `query` and
`relevant_resolution_ids`; `entity_type`/`theme`/`session`/`event_count` are
kept for human traceability, same convention as the existing entries' unused
`entity_type`/`entity` fields).

Run with:
    .venv/Scripts/python.exe tools/corpus_prep/add_thematic_to_gold_set.py
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SCAN_DIR = REPO / "academic_resolutions" / "llm_thematic_scan"
GOLD_PATH = REPO / "config" / "eval" / "gold_query_set.yaml"


def main() -> None:
    existing = yaml.safe_load(GOLD_PATH.read_text(encoding="utf-8"))
    existing = [e for e in existing if e.get("entity_type") != "thematic"]

    new_entries = []
    for f in sorted(SCAN_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        for theme in data.get("themes", []):
            new_entries.append({
                "query": theme["query"],
                "relevant_resolution_ids": theme["relevant_resolution_ids"],
                "entity_type": "thematic",
                "entity": theme["theme"],
                "session": data["session"],
                "event_count": theme["event_count"],
            })

    combined = existing + new_entries
    GOLD_PATH.write_text(
        yaml.safe_dump(combined, allow_unicode=True, sort_keys=False, width=1000),
        encoding="utf-8",
    )
    print(f"existing (non-thematic) entries kept: {len(existing)}")
    print(f"thematic entries added: {len(new_entries)}")
    print(f"total entries now in {GOLD_PATH}: {len(combined)}")


if __name__ == "__main__":
    main()
