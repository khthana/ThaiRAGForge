"""What the two serving caches actually hold: host RAM and VRAM.

Generated report: data/results/serving_cache_memory.md

`serving_cost_profile.md` measured what the caches SAVE (a warm served query is
422 ms against 11,980) and explicitly listed footprint as not established. This
is that measurement. Both caches trade memory for latency and neither trade was
priced:

  index cache     4 routed `Index` objects  -> host RAM (embeddings + chunks +
                                               the BM25Okapi memoised on them)
  embedder cache  2 sentence-transformers   -> VRAM on a 12 GB card

**Two questions that look like one and are not.** "Is the memory returned?" is an
allocator question, and on Windows a large numpy buffer is freed straight back to
the OS while small objects sit in Python's arenas — so RSS is the *operational*
number but a poor leak test. "Is the object freed?" is answered exactly, by
holding a `weakref` to every cached Index and requiring it to be dead after a
clear. `C3` is therefore the real leak check and `C4` merely reports what the OS
took back.

**The instrument is calibrated in-process before it is trusted** (`C1`): a
200 MB array must register within 10%, because a working-set reader that
silently returns a stale or wrong figure would make every number here plausible
and wrong. `GetProcessMemoryInfo` needs explicit `restype`/`argtypes` or the
pseudo-handle from `GetCurrentProcess` is truncated and the call fails with
ERROR_INVALID_HANDLE — which is how this was written the first time.

Windows-only (psapi/kernel32 via ctypes, no new dependency); it exits 0 with a
skip notice elsewhere rather than pretending to measure.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import weakref
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from rag_lab.config import StrategySpec  # noqa: E402
from rag_lab.factory import (  # noqa: E402
    build_embedder_cached,
    clear_embedder_cache,
)
from rag_lab.io.index_cache import (  # noqa: E402
    clear_index_cache,
    index_cache_info,
    load_index_cached,
)
from rag_lab.query_service import _read_manifest, discover_indices, resolve_index  # noqa: E402
from rag_lab.retrievers.bm25 import BM25Retriever  # noqa: E402
from rag_lab.router import route_targets  # noqa: E402

INDEX_ROOT = REPO / "data/index/chunker_compare_full"
REPORT = REPO / "data/results/serving_cache_memory.md"
RAW = REPO / "data/results/serving_cache_memory_raw.json"
MB = 1024 * 1024

# --------------------------------------------------------------- RSS reader
if sys.platform == "win32":
    import ctypes.wintypes as wt

    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class _MEMSTATUS(ctypes.Structure):
        _fields_ = [
            ("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    # Explicit signatures are load-bearing: without them ctypes takes the default
    # c_int restype and truncates the 64-bit pseudo-handle, and the call fails
    # with ERROR_INVALID_HANDLE.
    _k32.GetCurrentProcess.restype = wt.HANDLE
    _k32.GetCurrentProcess.argtypes = []
    _psapi.GetProcessMemoryInfo.restype = wt.BOOL
    _psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
    _k32.GlobalMemoryStatusEx.restype = wt.BOOL
    _k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMSTATUS)]

    def rss() -> int:
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if not _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError(ctypes.get_last_error())
        return c.WorkingSetSize

    def total_ram() -> int:
        m = _MEMSTATUS()
        m.dwLength = ctypes.sizeof(_MEMSTATUS)
        if not _k32.GlobalMemoryStatusEx(ctypes.byref(m)):
            raise OSError(ctypes.get_last_error())
        return m.ullTotalPhys
else:  # pragma: no cover - the rig is Windows
    def rss() -> int:
        raise NotImplementedError

    def total_ram() -> int:
        raise NotImplementedError


def settled_rss() -> int:
    """RSS after a full collection, so a delta measures what is still HELD
    rather than what has not been swept yet."""
    gc.collect()
    return rss()


def vram() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated())
    except Exception:
        return None


# ------------------------------------------------------------- measurement
def calibrate() -> dict:
    """C1: does the reader track a known allocation? `ones`, not `zeros` -- a
    zero page can be lazily backed and would under-report."""
    import numpy as np

    before = settled_rss()
    n_bytes = 200 * MB
    a = np.ones(n_bytes // 4, dtype=np.float32)
    after = settled_rss()
    delta = after - before
    del a
    freed = after - settled_rss()
    return {"expected": n_bytes, "observed": delta, "freed_on_del": freed}


def peak_rss() -> int:
    """The process's peak working set. `ArtifactStore.load` materialises every
    parquet column as Python lists before building Chunks, so the transient peak
    can exceed what is finally held -- and a deployment sizes for the peak."""
    c = _PMC()
    c.cb = ctypes.sizeof(_PMC)
    if not _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError(ctypes.get_last_error())
    return c.PeakWorkingSetSize


def decompose_load(directory) -> dict:
    """Where one index's RSS actually goes.

    940 MB held for a 223 MB embedding matrix is not self-explanatory, and an
    unexplained number is not a measurement. This walks `ArtifactStore.load`'s
    own steps so the chunk-object cost, the transient column materialisation and
    the matrix are separated rather than summed.
    """
    import json as _json

    import numpy as np
    import pyarrow.parquet as pq

    from rag_lab.schema import Chunk

    d = Path(directory)
    base = settled_rss()
    table = pq.read_table(d / "chunks.parquet")
    after_table = settled_rss()
    cols = table.to_pydict()
    after_cols = settled_rss()
    chunks = [
        Chunk(
            chunk_id=cols["chunk_id"][i],
            resolution_id=cols["resolution_id"][i],
            text=cols["text"][i],
            chunk_index=int(cols["chunk_index"][i]),
            page=int(cols["page"][i]),
            metadata=_json.loads(cols["metadata"][i]),
        )
        for i in range(len(cols["chunk_id"]))
    ]
    after_chunks = settled_rss()
    del cols, table
    after_free = settled_rss()
    emb = np.load(d / "embeddings.npy")
    after_emb = settled_rss()
    out = {
        "dir": d.name,
        "n_chunks": len(chunks),
        "arrow_table": after_table - base,
        "to_pydict": after_cols - after_table,
        "chunk_objects": after_chunks - after_cols,
        "freed_columns": after_free - after_chunks,
        "embeddings": after_emb - after_free,
        "held": after_emb - base,
        "embeddings_bytes": int(emb.nbytes),
    }
    del chunks, emb
    return out


def measure(indices) -> dict:
    import numpy as np

    routed = []
    seen: set[str] = set()
    for route, target in route_targets("hybrid").items():
        info = resolve_index(target, indices)
        if info.dir in seen:
            continue  # faculty and unmatched share one index
        seen.add(info.dir)
        routed.append((route, info))

    breakdown = decompose_load(routed[0][1].dir)

    baseline = settled_rss()
    per_index = []
    refs = []
    for route, info in routed:
        before = settled_rss()
        idx = load_index_cached(info.dir)
        after_load = settled_rss()
        scorer = BM25Retriever._scorer(idx)
        after_scorer = settled_rss()

        text_bytes = sum(len(c.text.encode("utf-8")) for c in idx.chunks)
        tokens = sum(len(t) for t in (idx.lexical or []))
        per_index.append({
            "route": route,
            "dir": Path(info.dir).name,
            "n_chunks": len(idx.chunks),
            "embeddings_shape": list(idx.embeddings.shape),
            "embeddings_bytes": int(idx.embeddings.nbytes),
            "text_bytes": text_bytes,
            "lexical_tokens": tokens,
            "rss_load": after_load - before,
            "rss_scorer": after_scorer - after_load,
            "rss_total": after_scorer - before,
        })
        refs.append(weakref.ref(idx))
        del idx, scorer

    held = settled_rss() - baseline
    info_after = index_cache_info()

    # C3: the sound leak test. RSS is an allocator question; this is not.
    clear_index_cache()
    gc.collect()
    alive = sum(1 for r in refs if r() is not None)
    returned = held - (settled_rss() - baseline)

    # ---- VRAM, held by the embedder cache
    clear_embedder_cache()
    gc.collect()
    v0 = vram()
    emb_specs, seen_spec = [], set()
    for _, info in routed:
        spec = StrategySpec.model_validate(_read_manifest(info.dir)["combo"]["embedder"])
        key = json.dumps({"t": spec.type, "p": spec.params}, sort_keys=True)
        if key in seen_spec:
            continue
        seen_spec.add(key)
        emb_specs.append(spec)

    per_model = []
    for spec in emb_specs:
        b = vram()
        e = build_embedder_cached(spec)
        e.embed(["วัดหน่วยความจำ"])  # force the lazy load
        per_model.append({
            "type": spec.type,
            "model": spec.params.get("model_name", spec.type),
            "vram_bytes": (vram() or 0) - (b or 0),
        })
    v_held = (vram() or 0) - (v0 or 0)
    clear_embedder_cache()
    gc.collect()
    v_returned = v_held - ((vram() or 0) - (v0 or 0))

    return {
        "load_breakdown": breakdown,
        "peak_rss": peak_rss(),
        "baseline_rss": baseline,
        "per_index": per_index,
        "held_rss": held,
        "cache_entries": info_after["size"],
        "cache_max": info_after["max_size"],
        "weakrefs_alive_after_clear": alive,
        "rss_returned_on_clear": returned,
        "total_ram": total_ram(),
        "vram_per_model": per_model,
        "vram_held": v_held,
        "vram_returned_on_clear": v_returned,
        "vram_available": v0 is not None,
    }


def render(data: dict) -> tuple[str, list[tuple[str, bool, str]]]:
    c, m = data["calibration"], data["measure"]
    exact = sum(i["embeddings_bytes"] for i in m["per_index"])
    cal_err = abs(c["observed"] - c["expected"]) / c["expected"]

    checks = [
        ("C1 the RSS reader tracks a known 200 MB allocation",
         cal_err < 0.10,
         f"observed {c['observed'] / MB:.1f} MB for {c['expected'] / MB:.0f} MB "
         f"({cal_err * 100:.1f}% off)"),
        ("C2 each index holds at least its embedding matrix",
         all(i["rss_total"] >= i["embeddings_bytes"] for i in m["per_index"]),
         "; ".join(f"{i['route']} {i['rss_total'] / MB:.0f} vs "
                   f"{i['embeddings_bytes'] / MB:.0f} MB" for i in m["per_index"])),
        ("C3 clearing the cache frees every Index object (weakref, not RSS)",
         m["weakrefs_alive_after_clear"] == 0,
         f"{m['weakrefs_alive_after_clear']} of {len(m['per_index'])} still alive"),
        ("C4 the cache held the routed index count while measuring",
         m["cache_entries"] == len(m["per_index"]),
         f"{m['cache_entries']} of max {m['cache_max']}"),
        ("C5 VRAM is returned when the embedder cache is cleared",
         (not m["vram_available"]) or m["vram_returned_on_clear"] >= 0.9 * m["vram_held"],
         "no CUDA" if not m["vram_available"] else
         f"returned {m['vram_returned_on_clear'] / MB:.0f} of "
         f"{m['vram_held'] / MB:.0f} MB"),
    ]

    L = ["# What the serving caches hold — host RAM and VRAM", ""]
    L.append(f"Generated by `tools/eval/serving_cache_memory.py` on "
             f"{datetime.fromtimestamp(data['ts']):%Y-%m-%d %H:%M}.")
    L.append("")
    L.append("`serving_cost_profile.md` measured what these caches save; it listed "
             "footprint as not established. This is that measurement.")
    L.append("")
    L.append("## 1. Host RAM — the index cache")
    L.append("")
    L.append("| route | index | chunks | embeddings | text | RSS: load | + BM25 | total |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i in m["per_index"]:
        L.append(
            f"| {i['route']} | `{i['dir']}` | {i['n_chunks']:,} | "
            f"{i['embeddings_bytes'] / MB:,.0f} MB | {i['text_bytes'] / MB:,.0f} MB | "
            f"{i['rss_load'] / MB:,.0f} MB | {i['rss_scorer'] / MB:,.0f} MB | "
            f"**{i['rss_total'] / MB:,.0f} MB** |"
        )
    L.append("")
    L.append(f"- **held at the default size {m['cache_max']}: "
             f"{m['held_rss'] / MB:,.0f} MB** "
             f"({m['held_rss'] / m['total_ram'] * 100:.1f}% of this machine's "
             f"{m['total_ram'] / (1024 ** 3):.0f} GB)")
    L.append(f"- of which **{exact / MB:,.0f} MB is the embedding matrices**, an exact "
             f"figure (`ndarray.nbytes`) rather than a measured one — the remainder is "
             f"chunk objects and the BM25 structures")
    L.append(f"- the BM25 scorers alone add "
             f"**{sum(i['rss_scorer'] for i in m['per_index']) / MB:,.0f} MB**, and they "
             f"are what the 921 ms rebuild buys back")
    L.append(f"- on clear the OS took back **{m['rss_returned_on_clear'] / MB:,.0f} MB** "
             f"of {m['held_rss'] / MB:,.0f}")
    L.append("")
    b = m.get("load_breakdown")
    if b:
        L.append(f"### 1b. Where one index's RSS goes — `{b['dir']}`, "
                 f"{b['n_chunks']:,} chunks")
        L.append("")
        L.append("940 MB held for a 223 MB matrix is not self-explanatory, so "
                 "`ArtifactStore.load`'s own steps are walked separately rather than "
                 "summed.")
        L.append("")
        L.append("| step | RSS |")
        L.append("| --- | ---: |")
        for k, label in [
            ("arrow_table", "`pq.read_table` (arrow buffers)"),
            ("to_pydict", "`.to_pydict()` — every column as Python lists"),
            ("chunk_objects", "building the `Chunk` objects"),
            ("freed_columns", "freeing the arrow table + column lists"),
            ("embeddings", "`np.load(embeddings.npy)`"),
        ]:
            L.append(f"| {label} | {b[k] / MB:+,.0f} MB |")
        L.append(f"| **held afterwards** | **{b['held'] / MB:,.0f} MB** |")
        L.append("")
        # DERIVED, not asserted. A first draft of this bullet claimed the chunk
        # objects were "the larger half" and the run said 80 MB against 223 MB
        # of vectors -- a verdict word typed beside numbers that contradict it,
        # which is the exact rot the D-family exists to catch.
        transient = b["arrow_table"] + b["to_pydict"]
        live = b["chunk_objects"] + b["embeddings"]
        bigger, smaller = (("chunk objects", b["chunk_objects"]),
                           ("embedding matrix", b["embeddings"]))
        if smaller[1] > bigger[1]:
            bigger, smaller = smaller, bigger
        L.append(f"- **{transient / MB:,.0f} MB of the {b['held'] / MB:,.0f} MB held is "
                 f"the transient parquet read**, not live data: `pq.read_table` plus "
                 f"`.to_pydict()` allocate it and deleting both returns only "
                 f"**{-b['freed_columns'] / MB:,.0f} MB** — the rest stays in the "
                 f"allocator's arenas. That is the single largest lever here, and it is "
                 f"a property of `ArtifactStore.load`, not of the cache.")
        L.append(f"- of what IS live, the **{bigger[0]}** is larger "
                 f"({bigger[1] / MB:,.0f} MB against {smaller[1] / MB:,.0f} MB); the "
                 f"chunk objects work out at roughly "
                 f"{b['chunk_objects'] / b['n_chunks']:,.0f} bytes per chunk for a "
                 f"pydantic model holding a string and a metadata dict")
        L.append(f"- process peak working set during this run: "
                 f"**{m['peak_rss'] / MB:,.0f} MB** — a deployment sizes for the peak, "
                 f"not the steady state")
        L.append("")
    L.append("## 2. VRAM — the embedder cache")
    L.append("")
    if m["vram_available"]:
        L.append("| embedder | model | VRAM |")
        L.append("| --- | --- | ---: |")
        for e in m["vram_per_model"]:
            L.append(f"| `{e['type']}` | {e['model']} | {e['vram_bytes'] / MB:,.0f} MB |")
        L.append("")
        L.append(f"- **held at the default size 2: {m['vram_held'] / MB:,.0f} MB** on a "
                 f"12 GB card, returned on clear: {m['vram_returned_on_clear'] / MB:,.0f} MB")
    else:
        L.append("No CUDA device visible; VRAM not measured.")
    L.append("")
    L.append("## 3. Checks")
    L.append("")
    L.append("| check | verdict | detail |")
    L.append("| --- | --- | --- |")
    for name, ok, detail in checks:
        L.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    L.append("")
    L.append("## 4. How to read this")
    L.append("")
    L.append("- **RSS is the operational number, `C3` is the leak test.** Whether the OS "
             "takes memory back is an allocator question — a large numpy buffer is "
             "released directly while small objects stay in Python's arenas — so a "
             "shortfall in the returned figure is not evidence of a leak. Whether the "
             "objects are *freed* is answered exactly, by weakref.")
    L.append("- **These are single-process figures.** Streamlit serves from one process, "
             "so they transfer; a multi-worker deployment pays them **per worker**, and "
             "that is the number to plan with.")
    L.append("- **Neither cache is sized for a sweep.** Both are bounded at the shipped "
             "route count and the eval path is uncached, so a 36-combo script keeps its "
             "current profile.")
    L.append("")
    return "\n".join(L) + "\n", checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="re-render from the raw JSON")
    args = ap.parse_args()

    if sys.platform != "win32" and not args.render:
        print("This measurement uses the Windows working-set API; skipping.")
        sys.exit(0)

    if args.render:
        data = json.loads(RAW.read_text(encoding="utf-8"))
    else:
        import time

        indices = discover_indices(INDEX_ROOT)
        data = {"ts": time.time(), "calibration": calibrate(), "measure": measure(indices)}
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    text, checks = render(data)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    failed = [c for c in checks if not c[1]]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks pass -> {REPORT}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
