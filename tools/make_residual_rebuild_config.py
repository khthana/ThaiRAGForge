"""Turn a partially-finished rebuild into a resume, at PER-COMBO granularity.

`runner.py` has no resume logic, but it is durable per combo: it does
`store.save(index, combo_dir)` then `write_manifest(...)` inside the loop, so a
combo that finished is on disk and a combo that did not is untouched. What the
batch driver lacks is granularity -- its `.DONE` markers cover a whole 9-combo
step, so a stop 8 combos in would redo all 9.

This script re-derives the still-stale set by I6's own rule and emits ONE
single-combo config per remaining combo, plus a driver whose `.DONE` markers are
per combo. Nothing is estimated: a combo is emitted iff it is not on disk, or is
on disk older than the corpus's last edit.

Splitting a config costs no extra GPU time. `run_experiment` re-runs
`loader.load(...)` and `build_index(...)` INSIDE the per-combo loop (runner.py
:65-78), so nothing is shared between combos to lose -- only ~15 s of process +
CUDA startup per invocation.

Nor does it change the artifact. `build_manifest` records experiment_name,
combo, run_mode, seed, n_resolutions and docset_hash; every one of those is
copied verbatim from the source config, so an index built from a residual config
is byte-comparable with one built from the original. `--verify` re-reads each
emitted YAML FROM DISK and requires it to enumerate exactly the intended
`BuildCombo.id` -- the id is a sha256 over the three `model_dump()`s, so a
single reordered key would silently mint a new index and leave the stale one
standing.

HOW THE RUN IS STOPPED, and why that half is recorded here. A long rebuild is
stopped by `data/logs/stop_rebuild_at_<date>.ps1`, which is GITIGNORED
(`.gitignore:28` covers `data/logs/`) -- so the stop rule has no tracked home of
its own and would be lost the moment the log directory is cleaned. It is:

  * Never kill on the clock. `store.save()` writes chunks.parquet ->
    embeddings.npy -> meta.json and only THEN does `write_manifest()` run, so a
    kill inside that window leaves a truncated embeddings.npy beside a manifest
    still carrying the OLD timestamp. This script would correctly re-emit that
    combo, but a corrupt index would sit on disk meanwhile and I1 (row
    alignment) would fail against it.
  * So the watchdog waits for the deadline, then for the next manifest.json
    write anywhere under the index roots -- that write is a combo's LAST action,
    and the next combo needs minutes of loading + chunking before it touches
    disk. A wide, safe window.
  * The driver and the worker are two processes, and the worker is absent for a
    second or two BETWEEN steps. "Is the chain finished?" must therefore be
    keyed on both being gone; a deadline-passed poll that finds the driver alive
    with no worker has found something better than a combo boundary -- a whole
    step boundary -- and should stop the driver there.
  * Kill the worker LEAF FIRST: the venv launcher re-execs, so `rag_lab.cli`
    matches a ~4 MB stub and its ~3 GB worker child, and killing the stub alone
    ends the driver's `-Wait` while the worker keeps building and keeps writing.
  * Match the driver by its `-File <script>` invocation, not by a bare substring
    of the script name: any shell that merely MENTIONS the script carries the
    name in its command line too, and a self-match both fakes a live driver and
    aims the kill at the wrong process.

Usage (read-only by default -- writes nothing unless --write):
    python tools/make_residual_rebuild_config.py
    python tools/make_residual_rebuild_config.py --write
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "eval"))

import audit_pipeline_invariants as A  # noqa: E402
from rag_lab.combos import enumerate_build_combos  # noqa: E402
from rag_lab.config import ExperimentConfig  # noqa: E402

# The 7 configs the 2026-08-14 chain builds from, in the order it builds them.
# Verified before that chain launched to enumerate exactly the 40 I6-stale
# combos, 0 uncovered / 0 extra (see run_rebuild_chain_2026_08_14.ps1).
SOURCE_CONFIGS = [
    "config/experiments/_history/chunker_compare_full_rebuild_2026_07_28_1_semantic.yaml",
    "config/experiments/_history/chunker_compare_full_rebuild_2026_07_28_2_recursive.yaml",
    "config/experiments/_history/chunker_compare_full_rebuild_2026_07_28_3_sentence.yaml",
    "config/experiments/_history/chunker_compare_full_rebuild_2026_07_28_4_fixed_size.yaml",
    "config/experiments/rq3_segmentation_ablation.yaml",
    "config/experiments/rq3_chunksize_sweep.yaml",
    "config/experiments/rq3_normalize_ablation.yaml",
]

# Downstream steps that must run once every index above is current. Kept here so
# the resume driver is the whole remaining job, not just the builds.
EVAL_STEPS = [
    "tools/eval/rq3_segmentation_significance_test.py",
    "tools/eval/rq3_chunksize_sweep_report.py",
    "tools/eval/rq3_normalize_significance_test.py",
]

OUT_CONFIG_DIR = REPO / "config" / "experiments" / "_resume" / "rebuild_2026_08_14"
OUT_DRIVER = REPO / "data" / "logs" / "run_rebuild_resume_2026_08_14.ps1"
TAG = "resume_2026_08_14"


def newest_corpus_mtime() -> tuple[float, Path]:
    """I6's own rule: the corpus's last edit is over *.md AND the manifests,
    because a resolution_id is built from the manifest title (ADR-0003)."""
    files = [*A.iter_corpus_files(A.CORPUS), *A.CORPUS.rglob("meeting_manifest.json")]
    newest = max(files, key=lambda p: p.stat().st_mtime)
    return newest.stat().st_mtime, newest


def built_at(combo_dir: Path) -> float | None:
    """None = no index here at all. Otherwise the manifest timestamp, raised by
    any recorded relabel (a title repair moves ids without touching chunk text,
    so it brings an index current without a rebuild)."""
    mpath = combo_dir / "manifest.json"
    if not mpath.exists():
        return None
    man = json.loads(mpath.read_text(encoding="utf-8"))
    built = datetime.fromisoformat(man["timestamp"]).timestamp()
    for marker in ("relabeled_mispairings", "relabeled"):
        at = (man.get(marker) or {}).get("at")
        if at:
            built = max(built, datetime.fromisoformat(at).timestamp())
    return built


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="emit the configs + driver")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="emit somewhere other than the repo (rehearsal; the driver still "
             "points at the emitted configs, so a rehearsal driver is not runnable)",
    )
    args = ap.parse_args()

    out_config_dir = Path(args.out_dir) / "configs" if args.out_dir else OUT_CONFIG_DIR
    out_driver = Path(args.out_dir) / OUT_DRIVER.name if args.out_dir else OUT_DRIVER

    newest, newest_path = newest_corpus_mtime()
    print(f"corpus newest edit : {datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}")
    print(f"  from             : {newest_path}\n")

    todo: list[dict] = []
    done_count = 0

    for cfg_path in SOURCE_CONFIGS:
        p = REPO / cfg_path
        if not p.exists():
            print(f"  MISSING CONFIG: {cfg_path}")
            return 1
        cfg = ExperimentConfig.from_yaml(p)
        out_root = REPO / cfg.output_dir
        pending = []
        for combo in enumerate_build_combos(cfg):
            built = built_at(out_root / combo.id)
            if built is not None and built >= newest:
                done_count += 1
                continue
            pending.append(
                {
                    "source": cfg_path,
                    "cfg": cfg,
                    "combo": combo,
                    "root": Path(cfg.output_dir).name,
                    "why": "missing" if built is None else "stale",
                }
            )
        todo.extend(pending)
        print(f"  {len(pending):2d} still to build of "
              f"{len(enumerate_build_combos(cfg)):2d}   {Path(cfg_path).stem}")

    print(f"\nalready current: {done_count}    remaining: {len(todo)}")
    if not todo:
        print("\nNothing to resume -- every combo is current. Run the eval steps only.")

    if not args.write:
        print("\n(read-only; pass --write to emit configs + driver)")
        return 0

    out_config_dir.mkdir(parents=True, exist_ok=True)
    emitted: list[tuple[str, str, str]] = []  # (rel_cfg_path, root, combo_id)

    for i, item in enumerate(todo, start=1):
        src: ExperimentConfig = item["cfg"]
        combo = item["combo"]
        # Everything build_manifest records is carried over verbatim; only the
        # three lists are narrowed to the single combo.
        one = src.model_copy(
            update={
                "loaders": [combo.loader],
                "chunkers": [combo.chunker],
                "embedders": [combo.embedder],
            },
            deep=True,
        )
        name = f"{i:02d}_{item['root']}__{combo.id}.yaml"
        path = out_config_dir / name
        one.to_yaml(path)

        # Verify the ARTIFACT ON DISK, not the object in memory.
        back = ExperimentConfig.from_yaml(path)
        got = enumerate_build_combos(back)
        assert len(got) == 1, f"{name}: enumerates {len(got)} combos, expected 1"
        assert got[0].id == combo.id, f"{name}: id {got[0].id} != {combo.id}"
        assert Path(back.output_dir).name == item["root"], f"{name}: output_dir moved"
        assert back.experiment_name == src.experiment_name, f"{name}: experiment_name moved"
        assert back.seed == src.seed and back.run_mode == src.run_mode, f"{name}: seed/run_mode moved"
        assert back.corpus.model_dump() == src.corpus.model_dump(), f"{name}: corpus moved"
        emitted.append((f"config/experiments/_resume/rebuild_2026_08_14/{name}",
                        item["root"], combo.id))

    print(f"\nwrote {len(emitted)} single-combo configs -> {out_config_dir}")
    print("verified: each re-reads from disk and enumerates exactly its intended combo id")

    out_driver.write_text(render_driver(emitted), encoding="utf-8")
    print(f"wrote driver -> {out_driver}")
    if args.out_dir:
        print("\n(rehearsal: --out-dir was set, so the driver's config paths do NOT "
              "point at these files -- do not launch it)")
    else:
        print("\nlaunch with:  powershell -ExecutionPolicy Bypass -File "
              f"{out_driver.relative_to(REPO).as_posix()}")
    return 0


def render_driver(emitted: list[tuple[str, str, str]]) -> str:
    """One .DONE marker per COMBO, so a further stop costs at most one combo."""
    lines = [
        "# Resume of the 2026-08-14 I6 rebuild -- generated by",
        "# tools/make_residual_rebuild_config.py --write. Do not hand-edit; re-generate.",
        "#",
        "# One combo per step, one .DONE per combo: a stop now costs at most the",
        "# combo in flight. Stopping mid-combo is safe (runner.py writes the",
        "# manifest last), but prefer stop_rebuild_at_*.ps1, which waits for a",
        "# manifest write before killing so no half-written index dir is left.",
        "#",
        "# `rag_lab.cli run` exits 0 even when a combo fails (runner.py isolates",
        "# the exception and records status=\"error\"), so every build step also",
        "# greps its log for [error] and requires a complete Done: N/N line.",
        "",
        'Set-Location "C:\\Users\\Terry\\Desktop\\Code\\RAG"',
        '$env:PYTHONPATH = "C:\\Users\\Terry\\Desktop\\Code\\RAG\\src"',
        '$env:PYTHONIOENCODING = "utf-8"',
        '$py = "C:\\Users\\Terry\\Desktop\\Code\\RAG\\.venv\\Scripts\\python.exe"',
        '$logDir = "C:\\Users\\Terry\\Desktop\\Code\\RAG\\data\\logs"',
        f'$marker = "$logDir\\{TAG}_STATUS.txt"',
        "",
        "function Note($msg) {",
        '    "$msg $(Get-Date -Format o)" | Out-File -FilePath $marker -Append -Encoding utf8',
        "}",
        "",
        'if (Test-Path $marker) { Note "restarted" } else { "started $(Get-Date -Format o)" | Out-File -FilePath $marker -Encoding utf8 }',
        "",
        "$steps = @(",
    ]
    for i, (cfg, root, cid) in enumerate(emitted, start=1):
        lines.append(
            f'    @{{ name = "b{i:02d}_{root}_{cid}"; kind = "build"; cfg = "{cfg}" }}'
        )
    for j, script in enumerate(EVAL_STEPS, start=1):
        lines.append(
            f'    @{{ name = "e{j:02d}_{Path(script).stem}"; kind = "script"; cfg = "{script}" }}'
        )
    lines.append(")")
    lines.extend(
        [
            "",
            "foreach ($step in $steps) {",
            "    $stepName = $step.name",
            f'    $outLog = "$logDir\\{TAG}_$stepName.log"',
            f'    $errLog = "$logDir\\{TAG}_$stepName.log.err"',
            f'    $doneMarker = "$logDir\\{TAG}_$stepName.DONE"',
            "",
            '    if (Test-Path $doneMarker) { Note "skipping $stepName (already done)"; continue }',
            '    Note "running $stepName"',
            "",
            '    if ($step.kind -eq "build") {',
            '        $argList = @("-m", "rag_lab.cli", "run", "--config", $step.cfg)',
            "    } else {",
            "        $argList = @($step.cfg)",
            "    }",
            "",
            "    $proc = Start-Process -FilePath $py -ArgumentList $argList `",
            "        -RedirectStandardOutput $outLog -RedirectStandardError $errLog -NoNewWindow -Wait -PassThru",
            "    $exitCode = $proc.ExitCode",
            "",
            "    if ($exitCode -ne 0) {",
            '        Note "FAILED $stepName exit=$exitCode"',
            "        exit $exitCode",
            "    }",
            "",
            '    if ($step.kind -eq "build") {',
            "        $errLines = @(Select-String -Path $outLog -Pattern '^\\[error\\]')",
            "        if ($errLines.Count -gt 0) {",
            '            Note "FAILED $stepName : $($errLines.Count) combo(s) reported [error]"',
            "            exit 1",
            "        }",
            "        $doneLine = @(Select-String -Path $outLog -Pattern '^Done: (\\d+)/(\\d+) combos built')",
            "        if ($doneLine.Count -ne 1) {",
            '            Note "FAILED $stepName : no single Done summary line"',
            "            exit 1",
            "        }",
            "        $m = $doneLine[0].Matches[0]",
            "        if ($m.Groups[1].Value -ne $m.Groups[2].Value) {",
            '            Note "FAILED $stepName : only $($m.Groups[1].Value)/$($m.Groups[2].Value) built"',
            "            exit 1",
            "        }",
            "    }",
            "",
            '    Note "completed $stepName"',
            '    "done" | Out-File -FilePath $doneMarker -Encoding utf8',
            "}",
            "",
            'Note "RESUME_CHAIN_DONE exit=0"',
            "exit 0",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
