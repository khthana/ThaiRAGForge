"""What the SHIPPED served path does when something is wrong.

Every other measurement in this project asks how good the answer is. This one
asks what happens when there is no answer to give -- the engine is down, a
collection was never ingested, an index directory is missing, a rebuild landed
and nobody re-ingested. It exists because the deployment is real
(`project_real_deployment_intent`), and because an operator reading
`[WinError 10061] No connection could be made because the target machine
actively refused it` learns nothing about which of five components is down.

**The finding that motivated it, and it is the worst class this project has.**
`QdrantHybridRetriever` resolves its collection from the Index it is handed, but
`_to_ranked` builds every result from the ENGINE's stored payload -- the Index
supplies only a name. So a collection left behind by an index rebuild does not
fail, it *answers*. Measured here with a scratch index and a scratch collection:
after a rebuild without a re-ingest, one `IndexInfo` and one query return the
CURRENT build in-process and the PREVIOUS build through the served retriever,
with no error on either side. That is the shape this codebase keeps getting hurt
by -- two artifacts produced at different times, nothing crashes, a number is
just wrong -- and until 2026-08-23 the file path had a seal against it
(`index_cache._settle`) while the engine path had nothing.

**How a verdict is decided, and why it is mechanical rather than my reading.**
Each mode gets one of three:

* `SILENT`   -- returned results, and they came from the wrong build. The only
               mode that can earn this is `collection_stale`, because it is the
               only one where a wrong answer is distinguishable from a right one
               (the two builds are stamped). A `SILENT` verdict fails the run.
* `OPAQUE`   -- raised, but the message names neither the artifact that is wrong
               nor anything to do about it. Judged by asking whether the message
               contains the failing artifact's name AND a remedy token, not by
               reading it.
* `ACTIONABLE` -- raised, and the message names both.

**Lessons, each of which cost a wrong first attempt.**

* **A failure mode has to be constructed at the layer it lives in, and this was
  got wrong twice.** The first attempt at "never ingested" never reached the
  engine at all -- `_arms_for` checks the vocabulary sidecar before it builds a
  client, so the mode it exercised was `engine_down`. The second attempt dropped
  the collection first, and since the staleness guard runs ahead of `_arms_for`
  it measured a 404 three times while believing it had three modes. Modes that
  share a prefix of the call path must be ordered, and each one has to defeat
  the guards that come before it -- here, by putting the directory back in sync
  with the collection so the layer underneath is reachable at all.
* **The healthy control is not a formality.** It is what says the guard added on
  2026-08-23 does not refuse the four collections that are actually deployed --
  a check that refuses everything would pass every failure-mode test in here.
* **Nothing falls back.** It is tempting to have the served retriever drop to
  the in-process one when the engine is unreachable. It must not: the two paths
  are different retrievers over different copies of the rows, so a silent switch
  is a different answer, not a degraded one -- the same reason `resolve_index`
  refuses an ambiguous route rather than picking one.

Safety: this script creates and deletes ONLY collections whose name starts with
`_PROBE_PREFIX`, and a scratch index under %TEMP%. It never writes to
`data/index/`, and it refuses to delete a collection not carrying that prefix.

Generated report: `data/results/serving_failure_modes.md`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.io.artifact_store import ArtifactStore  # noqa: E402
from rag_lab.factory import clear_retriever_cache  # noqa: E402
from rag_lab.io.index_cache import clear_index_cache  # noqa: E402
from rag_lab.query_service import (  # noqa: E402
    IndexInfo,
    discover_indices,
    route_query,
)
from rag_lab.schema import Chunk, Index  # noqa: E402

INDEX_ROOT = REPO / "data/index/chunker_compare_full"
REPORT = REPO / "data/results/serving_failure_modes.md"
RAW = REPO / "data/results/serving_failure_modes_raw.json"
URL = "http://127.0.0.1:6333"
DEAD_URL = "http://127.0.0.1:6399"
VOCAB_ROOT = REPO / "data/qdrant"

_PROBE_PREFIX = "probe_fm_"
SCRATCH = Path(os.environ.get("TEMP", "/tmp")) / "rag_lab_failure_modes"
SCRATCH_NAME = f"{_PROBE_PREFIX}stale__sentence__local__aaaaaaaa"
N_ROWS, DIM = 200, 1024

#: A person query: the route whose target is the collection the pilot ingested.
QUERY = "อาจารย์ ดร.กลกรณ์ วงศ์ภาคิกะเสรี"

#: A message is ACTIONABLE only if it names a remedy. Checked against the
#: message text rather than judged by eye, so a future edit that drops the
#: remedy turns the verdict, not just the prose.
_REMEDY_TOKENS = ("re-ingest", "qdrant_pilot_ingest", "docker start", "rebuild",
                  "Pass ", "Run ", "Use a row-reading", "retry")


def _spec(url: str = URL, **kw) -> StrategySpec:
    return StrategySpec(type="qdrant_hybrid",
                        params={"url": url, "fetch_depth": 200, **kw})


def _hybrid() -> StrategySpec:
    return StrategySpec(type="hybrid", params={"fetch_depth": 200})


# --------------------------------------------------------------------------- #
# the scratch index: two builds of one directory, distinguishable in the payload
# --------------------------------------------------------------------------- #
_RNG = np.random.default_rng(0)
_VECS = _RNG.standard_normal((N_ROWS, DIM)).astype(np.float32)
_VECS /= np.linalg.norm(_VECS, axis=1, keepdims=True)


def build_scratch(directory: Path, tag: str) -> None:
    """Write build `tag` of the scratch index.

    The two builds share their vectors and differ in `chunk_id` and text, so a
    result can be attributed to a build without depending on which one ranks
    higher -- the point is *which rows answered*, never their order.
    """
    chunks = [
        Chunk(
            chunk_id=f"{tag}::{i}",
            resolution_id=f"2568/1/{tag} {i}",
            text=f"เอกสารชุด {tag} ลำดับ {i} อาจารย์ ดร.กลกรณ์ วงศ์ภาคิกะเสรี",
            chunk_index=i,
            page=1,
            metadata={"year": 2568},
        )
        for i in range(N_ROWS)
    ]
    ArtifactStore().save(
        Index(chunks=chunks, embeddings=_VECS.copy(), meta={"combo_id": directory.name},
              lexical=[c.text.split() for c in chunks]),
        directory,
    )
    # query_indices reads manifest.json, which ArtifactStore.save does not write
    # (the build pipeline does). docset_hash differs per build, as a real rebuild
    # of a re-OCR'd corpus would -- but the guard must not NEED it, so nothing
    # below reads it.
    (directory / "manifest.json").write_text(
        json.dumps({
            "experiment_name": "serving_failure_modes",
            "combo_id": directory.name,
            "combo": {
                "loader": {"type": "plain", "params": {}},
                "chunker": {"type": "sentence", "params": {"chunk_size": 512}},
                "embedder": {"type": "local", "params": {"model_name": "BAAI/bge-m3"}},
            },
            "run_mode": "probe", "seed": 0, "n_resolutions": N_ROWS,
            "docset_hash": f"probe_{tag}", "git_commit": "probe",
            "timestamp": "2026-08-23T00:00:00",
        }),
        encoding="utf-8",
    )


def scratch_info(directory: Path) -> IndexInfo:
    return IndexInfo(
        combo_id=directory.name,
        dir=str(directory),
        loader=StrategySpec(type="plain"),
        chunker=StrategySpec(type="sentence"),
        embedder=StrategySpec(type="local", params={"model_name": "BAAI/bge-m3"}),
    )


def ingest(directory: Path) -> tuple[bool, str]:
    r = subprocess.run(
        [sys.executable, str(REPO / "tools/eval/qdrant_pilot_ingest.py"),
         "--index", str(directory), "--url", URL],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(REPO),
    )
    return r.returncode == 0, (r.stderr or r.stdout or "")[-400:]


def drop_collection(name: str) -> None:
    """Delete a collection -- and ONLY a probe one.

    The guard is not paranoia: the four routed collections cost a real ingest to
    rebuild, and a probe that could name one of them by accident is one edit
    away from deleting it.
    """
    if not name.startswith(_PROBE_PREFIX):
        raise ValueError(f"refusing to drop non-probe collection {name!r}")
    from qdrant_client import QdrantClient

    client = QdrantClient(url=URL)
    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# running one mode
# --------------------------------------------------------------------------- #
def run_mode(name: str, fn, *, expect_results: bool, artifact: str) -> dict:
    """Call the shipped path once and classify what came back."""
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 -- classifying is the job
        msg = str(exc).replace("\n", " ")
        names_artifact = bool(artifact) and artifact.lower() in msg.lower()
        has_remedy = any(t.lower() in msg.lower() for t in _REMEDY_TOKENS)
        return {
            "mode": name,
            "outcome": "raised",
            "exception": type(exc).__name__,
            "seconds": round(time.perf_counter() - t0, 2),
            "message": msg[:300],
            "names_artifact": names_artifact,
            "has_remedy": has_remedy,
            "verdict": "ACTIONABLE" if (names_artifact and has_remedy) else "OPAQUE",
        }
    builds = sorted({rc.chunk_id.split("::")[0] for rc in result.results
                     if "::" in rc.chunk_id})
    wrong_build = builds == ["A"]
    return {
        "mode": name,
        "outcome": "returned",
        "exception": None,
        "seconds": round(time.perf_counter() - t0, 2),
        "n_results": len(result.results),
        "builds": builds,
        "message": "",
        "names_artifact": False,
        "has_remedy": False,
        "verdict": "SILENT" if wrong_build else ("OK" if expect_results else "SILENT"),
    }


def probe_all(skip_engine: bool) -> dict:
    indices = discover_indices(INDEX_ROOT)
    modes: list[dict] = []

    # ---- modes that need no engine ----
    modes.append(run_mode(
        "index_dir_missing",
        lambda: route_query(
            QUERY,
            [dataclasses.replace(i, dir=str(Path(i.dir).parent / "no_such_index"))
             for i in indices],
            _hybrid(), 10),
        expect_results=False, artifact="no_such_index"))

    modes.append(run_mode(
        "route_target_not_built",
        lambda: route_query(QUERY, [], _hybrid(), 10),
        expect_results=False, artifact="sentence"))

    modes.append(run_mode(
        "in_process_control",
        lambda: route_query(QUERY, indices, _hybrid(), 10),
        expect_results=True, artifact=""))

    if skip_engine:
        return {"modes": modes, "engine": "skipped"}

    # ---- engine modes ----
    modes.append(run_mode(
        "engine_down",
        lambda: route_query(QUERY, indices, _spec(url=DEAD_URL), 10),
        expect_results=False, artifact="6399"))

    modes.append(run_mode(
        "served_control",
        lambda: route_query(QUERY, indices, _spec(), 10),
        expect_results=True, artifact=""))

    # ---- the stale collection, built for real ----
    d = SCRATCH / SCRATCH_NAME
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True)
    d.mkdir(parents=True)
    drop_collection(SCRATCH_NAME)
    build_scratch(d, "A")
    ok, tail = ingest(d)
    if not ok:
        raise SystemExit(f"scratch ingest failed:\n{tail}")

    build_scratch(d, "B")          # the rebuild nobody re-ingested
    clear_index_cache()
    info = scratch_info(d)

    modes.append(run_mode(
        "collection_stale (guard OFF -- what shipped before 2026-08-23)",
        lambda: route_query(QUERY, [info], _spec(verify_collection=False), 5),
        expect_results=False, artifact=SCRATCH_NAME))

    modes.append(run_mode(
        "collection_stale (guard ON -- what ships now)",
        lambda: route_query(QUERY, [info], _spec(), 5),
        expect_results=False, artifact=SCRATCH_NAME))

    # ---- the vocabulary sidecar, which is reachable ONLY when the collection
    # agrees with the index. `_verify` runs ahead of `_arms_for`, so putting the
    # directory back in sync with the collection is what exposes the layer
    # underneath -- the first version of this probe dropped the collection first
    # and measured a 404 three times while believing it had three modes.
    build_scratch(d, "A")
    clear_index_cache()
    sidecar = VOCAB_ROOT / SCRATCH_NAME
    kept = SCRATCH / "sidecar_backup"
    shutil.copytree(sidecar, kept)
    shutil.rmtree(sidecar, ignore_errors=True)
    modes.append(run_mode(
        "vocabulary_sidecar_missing (collection agrees with the index)",
        lambda: route_query(QUERY, [info], _spec(), 5),
        expect_results=False, artifact="vocab.json"))

    # ---- the collection gone: never ingested, or dropped ----
    # The sidecar goes back, and the retriever cache is cleared, or this mode
    # measures the previous one twice. `_verified` is per retriever INSTANCE and
    # the serving layer caches instances, so a collection verified once is not
    # re-checked -- by design, for cost, and the first run of this probe read
    # that property back as a mode it had not written.
    shutil.copytree(kept, sidecar, dirs_exist_ok=True)
    drop_collection(SCRATCH_NAME)
    clear_index_cache()
    clear_retriever_cache()
    modes.append(run_mode(
        "collection_absent (never ingested, or dropped)",
        lambda: route_query(QUERY, [info], _spec(), 5),
        expect_results=False, artifact=SCRATCH_NAME))

    shutil.rmtree(SCRATCH, ignore_errors=True)
    return {"modes": modes, "engine": URL}


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def render(data: dict) -> tuple[str, list[tuple[str, bool, str]]]:
    modes = data["modes"]
    by = {m["mode"]: m for m in modes}
    L: list[str] = []
    L.append("# Serving failure modes: what the shipped path does when it cannot answer")
    L.append("")
    L.append("Generated by `tools/eval/serving_failure_modes.py`.")
    L.append("")
    L.append(
        "Every mode below is driven through the **shipped** `route_query`, not through "
        "a retriever assembled by hand. A mode is `ACTIONABLE` only if its message "
        "names the artifact that is wrong **and** a remedy — checked against the text, "
        "not judged by eye. `SILENT` means results came back from the wrong build, "
        "which is the only outcome here that a caller cannot detect."
    )
    L.append("")
    L.append("| mode | outcome | exception | to fail | names artifact | remedy | verdict |")
    L.append("| --- | --- | --- | ---: | :---: | :---: | --- |")
    for m in modes:
        out = m["outcome"]
        if out == "returned":
            out = f"returned {m.get('n_results', 0)}"
            if m.get("builds"):
                out += f" (build {'/'.join(m['builds'])})"
        L.append(
            f"| `{m['mode']}` | {out} | {m['exception'] or '—'} | {m['seconds']:.2f}s | "
            f"{'yes' if m['names_artifact'] else '—'} | "
            f"{'yes' if m['has_remedy'] else '—'} | "
            f"{'**' + m['verdict'] + '**' if m['verdict'] in ('SILENT', 'OPAQUE') else m['verdict']} |"
        )
    L.append("")

    off = by.get("collection_stale (guard OFF -- what shipped before 2026-08-23)")
    on = by.get("collection_stale (guard ON -- what ships now)")
    if off and on:
        L.append("## The mode that motivated this report")
        L.append("")
        L.append(
            "A collection is a copy of an `Index`'s rows, so **any** index rebuild "
            "stales it — and `_to_ranked` builds every result from the engine's stored "
            "**payload**, with the Index supplying only the collection name. So a "
            "collection nobody re-ingested does not fail, it *answers*."
        )
        L.append("")
        L.append(
            f"With the guard off — what shipped until 2026-08-23 — one `IndexInfo` and "
            f"one query returned **{off.get('n_results', 0)} results from build "
            f"{'/'.join(off.get('builds') or ['?'])}** through the served retriever "
            f"while the in-process path returned the current build, **no error on "
            f"either side**. With the guard on the same call raises in "
            f"{on['seconds']:.2f}s, naming the collection, the row that disagrees and "
            f"the command that repairs it."
        )
        L.append("")
        L.append(
            "**The guard is the engine-side counterpart of the index seal.** "
            "`index_cache._settle` refuses a directory whose artifacts disagree with "
            "the build its writer sealed; nothing made the equivalent claim about a "
            "collection. Two signals: the row **count** (one call, does most of the "
            "work) and a **sample of rows compared by identity** (point id == row index "
            "at ingest, so row *i*'s `chunk_id` must match) — the second exists because "
            "the first cannot see a rebuild that preserves the count, which is exactly "
            "what a re-OCR that moves text without moving chunk boundaries produces."
        )
        L.append("")
        L.append(
            "It runs **once per collection per retriever instance**, and the serving "
            "layer caches retrievers — so once per process, not per query."
        )
        L.append("")

    L.append("## Nothing falls back, and that is deliberate")
    L.append("")
    L.append(
        "It is tempting to drop to the in-process retrievers when the engine is "
        "unreachable. That would be wrong: the two paths are different retrievers over "
        "different copies of the rows, so a silent switch is a **different answer, not "
        "a degraded one** — the same reason `resolve_index` refuses an ambiguous route "
        "rather than picking one. Every engine failure above is loud."
    )
    L.append("")

    L.append("## What is NOT established")
    L.append("")
    L.append(
        "- **No timeout mode.** A refused connection is not a slow one; nothing here "
        "measures what a hung engine does to a served request.\n"
        "- **No GPU failure mode.** An embedder that OOMs mid-serve is not probed, "
        "because forcing it safely on the card the eval scripts share is not worth the "
        "risk of leaving it wedged.\n"
        "- **One process, no network hop** — the same limit every serving measurement "
        "here carries.\n"
        "- **The guard proves the collection matches the index it is handed.** It says "
        "nothing about whether that index is the one you meant to serve; that is "
        "`route_targets` and `resolve_index`, checked in `qdrant_routed_check.py`.\n"
        "- **It is a FIRST-USE check, not a per-query one.** `_verified` is per "
        "retriever instance and the serving layer caches instances, so a collection "
        "re-ingested — or dropped — *after* a process verified it is not re-checked "
        "until that process restarts. That is the cost trade, stated rather than "
        "hidden: the alternative is two round trips on every query."
    )
    L.append("")

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    silent = [m["mode"] for m in modes if m["verdict"] == "SILENT"
              and "guard OFF" not in m["mode"]]
    opaque = [m["mode"] for m in modes if m["verdict"] == "OPAQUE"]
    controls = [m for m in modes if m["mode"].endswith("control")]

    add("F1 no shipped mode answers from the wrong build",
        not silent,
        f"{len(silent)} silent of {len(modes)} modes "
        f"(the guard-OFF arm is excluded on purpose: it IS the defect, and a run where "
        f"it stopped being silent would mean the probe no longer reproduces it)")
    add("F2 the guard-OFF arm still reproduces the defect",
        off is None or off["verdict"] == "SILENT",
        "the negative control returned the previous build; without it F1 is a claim "
        "about a race that was never run"
        if off else "engine skipped")
    add("F3 every failure names its artifact and a remedy",
        not opaque,
        f"{len(opaque)} opaque of {len([m for m in modes if m['outcome'] == 'raised'])} "
        f"raising modes" + (f": {', '.join(opaque)}" if opaque else ""))
    add("F4 the controls still serve (the guard refuses nothing deployed)",
        bool(controls) and all(c["outcome"] == "returned" and c.get("n_results", 0) > 0
                               for c in controls),
        "; ".join(f"{c['mode']} {c.get('n_results', 0)} results" for c in controls)
        or "no control ran")

    L.append("## Self-checks")
    L.append("")
    L.append("| check | verdict | detail |")
    L.append("| --- | --- | --- |")
    for name, ok, detail in checks:
        L.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    L.append("")
    return "\n".join(L) + "\n", checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true", help="re-render from the raw cache")
    ap.add_argument("--skip-engine", action="store_true",
                    help="probe only the modes that need no Qdrant server")
    args = ap.parse_args()

    if args.render:
        data = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        data = probe_all(args.skip_engine)
        RAW.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    text, checks = render(data)
    REPORT.write_text(text, encoding="utf-8")
    print(f"wrote {REPORT}")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
