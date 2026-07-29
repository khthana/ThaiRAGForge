"""Watch an index output dir and log each combo's completion with a real
wall-clock timestamp, by polling for new/updated manifest.json files.

Exists because `rag_lab.runner.run_experiment`'s own progress output (tqdm
bars) has no absolute timestamps -- a batch counter like "9/9353" alone
can't tell you whether a run just started or has been going for hours,
which caused a real false-alarm/wasted-recompute incident on 2026-07-28
(see docs/llm-ocr-scan-log.md and the memory note
feedback_dont_extrapolate_gpu_eta_from_first_batches). Deliberately kept
outside the runner (never edit the runner -- CLAUDE.md, Open/Closed): this
just watches its output directory from outside, so it works with any
experiment run without touching build code.

Run alongside a background build with:
    .venv/Scripts/python.exe tools/eval/watch_build_progress.py \\
        data/index/chunker_compare_full --interval 15
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path


def _embedder_label(manifest_path: Path) -> str:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embedder = manifest["combo"]["embedder"]
        chunker = manifest["combo"]["chunker"]["type"]
        model_name = embedder.get("params", {}).get("model_name", "")
        return f"{chunker}/{embedder['type']}" + (f" ({model_name})" if model_name else "")
    except Exception as exc:  # noqa: BLE001 -- best-effort label, never fatal
        return f"<unreadable manifest: {exc}>"


def _snapshot(index_dir: Path) -> dict[str, float]:
    if not index_dir.exists():
        return {}
    return {
        d.name: (d / "manifest.json").stat().st_mtime
        for d in index_dir.iterdir()
        if (d / "manifest.json").exists()
    }


def watch(index_dir: Path, interval: float) -> None:
    def log(msg: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    # Baseline: whatever already exists at watch-start is NOT a new
    # completion -- only report combos that finish (or re-finish) after this.
    seen_mtimes = _snapshot(index_dir)
    log(f"watching {index_dir} every {interval}s -- {len(seen_mtimes)} combo(s) already present at baseline (Ctrl+C to stop)")
    while True:
        if index_dir.exists():
            for d in sorted(index_dir.iterdir()):
                manifest_path = d / "manifest.json"
                if not manifest_path.exists():
                    continue
                mtime = manifest_path.stat().st_mtime
                prev = seen_mtimes.get(d.name)
                if prev is None:
                    seen_mtimes[d.name] = mtime
                    log(f"COMPLETED: {d.name} -- {_embedder_label(manifest_path)}")
                elif mtime > prev:
                    seen_mtimes[d.name] = mtime
                    log(f"RE-COMPLETED (overwritten): {d.name} -- {_embedder_label(manifest_path)}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    watch(args.index_dir, args.interval)
