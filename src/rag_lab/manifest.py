"""Per-combo run manifest for reproducibility.

Captures the resolved combo (including loader identity — the #2 cache key omits
it, so the manifest is the unambiguous record; revisit the key at #6), plus a
content hash of the document set, the git commit, timestamp, and seed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_lab.combos import BuildCombo
from rag_lab.config import ExperimentConfig
from rag_lab.schema import Resolution


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _docset_hash(resolutions: list[Resolution]) -> str:
    payload = json.dumps(
        [[r.resolution_id, r.raw_text] for r in resolutions],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_manifest(
    config: ExperimentConfig, combo: BuildCombo, resolutions: list[Resolution]
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "combo_id": combo.id,
        "combo": {
            "loader": combo.loader.model_dump(),
            "chunker": combo.chunker.model_dump(),
            "embedder": combo.embedder.model_dump(),
        },
        "run_mode": config.run_mode,
        "seed": config.seed,
        "n_resolutions": len(resolutions),
        "docset_hash": _docset_hash(resolutions),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(directory: str | Path, manifest: dict[str, Any]) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
