"""Pre-filling the serving caches, before a user asks.

The caches take a warm routed query from 12,329 ms to 447 ms
(`data/results/serving_cost_profile.md`) -- but only for the SECOND caller on
each route. A fresh process has four such first callers (four routed index
directories, two embedders), so a deployment that never warms pays ~12 s four
times, in front of real users.

The tests that matter here are not "did it run" but the three things that make a
warm-up real rather than decorative: it must force the embedder's LAZY weight
load, it must warm the BM25 scorer that rides on the Index, and it must fill the
same cache keys the serving path will ask for.
"""
from __future__ import annotations

import pytest

from rag_lab.config import ExperimentConfig, StrategySpec
from rag_lab.factory import clear_embedder_cache
from rag_lab.io.index_cache import clear_index_cache, index_cache_info, load_index_cached
import rag_lab.query_service as qs
from rag_lab.query_service import discover_indices, warm_serving_caches
from rag_lab.router import RouteTarget
from rag_lab.runner import run_experiment


@pytest.fixture(autouse=True)
def _clean():
    clear_index_cache()
    clear_embedder_cache()
    yield
    clear_index_cache()
    clear_embedder_cache()


def _indices(tmp_path):
    corpus = tmp_path / "corpus" / "2569" / "ครั้งที่ 1"
    corpus.mkdir(parents=True)
    (corpus / "เรื่อง ค่าธรรมเนียม.md").write_text(
        "## Page 1\nค่าธรรมเนียม การศึกษา ภาคเรียน", encoding="utf-8"
    )
    (corpus / "เรื่อง หลักสูตร.md").write_text(
        "## Page 1\nหลักสูตร วิศวกรรม คอมพิวเตอร์", encoding="utf-8"
    )
    out = tmp_path / "out"
    run_experiment(
        ExperimentConfig(
            experiment_name="e",
            corpus={"input_dir": (tmp_path / "corpus").as_posix()},
            output_dir=out.as_posix(),
            loaders=[StrategySpec(type="plain")],
            chunkers=[
                StrategySpec(type="fixed_size", params={"chunk_size": 100}),
                StrategySpec(type="recursive", params={"chunk_size": 100}),
            ],
            embedders=[StrategySpec(type="hashing")],
        )
    )
    return discover_indices(out)


_FIXED = RouteTarget(chunker_type="fixed_size", embedder_type="hashing")
_RECURSIVE = RouteTarget(chunker_type="recursive", embedder_type="hashing")
_TWO_ROUTES = {"person": _RECURSIVE, "program": _FIXED, "unmatched": _FIXED}


def test_it_fills_the_index_cache_with_what_serving_will_ask_for(tmp_path):
    infos = _indices(tmp_path)
    report = warm_serving_caches(infos, "dense", route_combo=_TWO_ROUTES)

    assert not report["failures"]
    assert index_cache_info()["size"] == 2
    for entry in report["warmed"]:
        # The same key, so the first served query is a HIT, not a second load.
        assert load_index_cached(entry["dir"]) is load_index_cached(entry["dir"])


def test_two_routes_sharing_one_index_are_warmed_once(tmp_path):
    """`faculty` and `unmatched` share a directory in the shipped map, so
    looping over routes would load it twice and report a cost that never
    happened."""
    infos = _indices(tmp_path)
    report = warm_serving_caches(infos, "dense", route_combo=_TWO_ROUTES)

    dirs = [e["dir"] for e in report["warmed"]]
    assert len(dirs) == len(set(dirs)) == 2, "program and unmatched share one target"


def test_it_forces_the_lazy_weight_load(tmp_path, monkeypatch):
    """THE check. `LocalSTEmbedder._load()` runs inside the first `embed()`, so
    constructing an embedder costs 0.0 ms and warms nothing -- a warm-up that
    only builds one would report success and leave the 9.3 s where it was."""
    infos = _indices(tmp_path)
    calls: list[list[str]] = []

    from rag_lab import query_service as qs

    real = qs.build_embedder_cached

    def recording(spec):
        emb = real(spec)
        original = emb.embed

        def embed(texts):
            calls.append(list(texts))
            return original(texts)

        emb.embed = embed  # type: ignore[method-assign]
        return emb

    monkeypatch.setattr(qs, "build_embedder_cached", recording)
    warm_serving_caches(infos, "dense", route_combo=_TWO_ROUTES, probe="โพรบ")

    assert calls, "no embed() call -- the weights were never pulled"
    assert all(c == ["โพรบ"] for c in calls)


def test_it_warms_the_bm25_scorer_that_rides_on_the_index(tmp_path):
    """~1.0 s of the first hybrid query. The memo lives on the Index object, so
    a warm-up that loads the rows without it delivers less than half."""
    infos = _indices(tmp_path)
    warm_serving_caches(infos, "hybrid", route_combo=_TWO_ROUTES)

    for entry in index_cache_info()["entries"]:
        assert entry["has_bm25_scorer"], f"{entry['dir']} has no memoised scorer"


def test_a_dense_deployment_does_not_pay_for_the_scorer(tmp_path):
    """Whether to warm the lexical arm is DERIVED from the retriever, not a
    separate flag: ~1.0 s per index for a structure a dense deployment never
    scores with, and a caller able to ask for `hybrid` + "no scorer" would be
    asking for a state that cannot serve."""
    infos = _indices(tmp_path)
    warm_serving_caches(infos, "dense", route_combo=_TWO_ROUTES)

    assert not any(e["has_bm25_scorer"] for e in index_cache_info()["entries"])


def test_with_rows_false_warms_the_engine_shape(tmp_path):
    """`with_embeddings` is part of the cache key: an engine-served retriever
    loads without the matrix, so warming the full variant fills an entry the
    serving path never asks for AND wastes the ~234MB the flag exists to save."""
    infos = _indices(tmp_path)
    warm_serving_caches(infos, "hybrid", route_combo=_TWO_ROUTES, with_rows=False)

    entries = index_cache_info()["entries"]
    assert entries and all(e["with_embeddings"] is False for e in entries)
    assert not any(e["has_bm25_scorer"] for e in entries), "no rows means no lexical arm"


def test_a_missing_target_is_reported_not_raised(tmp_path):
    """A warm-up is an optimisation. A deployment that refuses to start because
    one route's index is missing is worse than one that serves the other three."""
    infos = _indices(tmp_path)
    combo = dict(_TWO_ROUTES)
    combo["ghost"] = RouteTarget(chunker_type="semantic", embedder_type="local")

    report = warm_serving_caches(infos, "dense", route_combo=combo)

    assert len(report["failures"]) == 1
    assert report["failures"][0]["route"] == "ghost"
    assert len(report["warmed"]) == 2, "the reachable routes were still warmed"


def test_the_report_carries_a_cost_per_index(tmp_path):
    infos = _indices(tmp_path)
    report = warm_serving_caches(infos, "hybrid", route_combo=_TWO_ROUTES)

    assert report["total_ms"] > 0
    for e in report["warmed"]:
        assert e["n_chunks"] > 0
        assert e["embedder_ms"] >= 0 and e["index_ms"] >= 0
        assert e["total_ms"] >= e["index_ms"]


def test_the_probe_retrieval_is_what_makes_it_actually_warm(tmp_path):
    """Everything resident is still not warm. With all four indices and both
    embedders loaded, the first real query measured 1,246 ms against ~450 for
    the ones after it; one throwaway retrieval takes it to 468 (2026-08-21).
    The residue is process-global CUDA/BLAS init, not per-index -- one probe
    fixed all four routes -- so this does one, and reports its cost."""
    infos = _indices(tmp_path)
    report = warm_serving_caches(infos, "hybrid", route_combo=_TWO_ROUTES)
    assert report["probe_ms"] is not None and report["probe_ms"] >= 0

    off = warm_serving_caches(
        infos, "hybrid", route_combo=_TWO_ROUTES, probe_retrieval=False
    )
    assert off["probe_ms"] is None


def test_a_probe_that_cannot_run_is_reported_not_raised(tmp_path):
    """`qdrant_hybrid` needs a url it is not given here. A warm-up must not take
    the app down over its own optimisation."""
    infos = _indices(tmp_path)
    report = warm_serving_caches(
        infos, "qdrant_hybrid", route_combo=_TWO_ROUTES, with_rows=True
    )
    assert report["probe_ms"] is None
    assert any(f.get("route") == "(probe retrieval)" for f in report["failures"])


def test_the_probe_uses_the_params_the_deployment_serves(tmp_path, monkeypatch):
    """A probe left at the class defaults warms a DIFFERENT code path from the
    one the user's query takes: `hybrid` at `fetch_depth=None` fuses over the
    whole corpus, measured at 2,052 ms against the shipped F=200's 1,093
    (2026-08-21). It is the startup budget paying for a path nothing serves."""
    seen = []
    real = qs.build_retriever_cached

    def recording(spec):
        seen.append(spec)
        return real(spec)

    monkeypatch.setattr(qs, "build_retriever_cached", recording)
    warm_serving_caches(
        _indices(tmp_path),
        "hybrid",
        route_combo=_TWO_ROUTES,
        retriever_params={"fetch_depth": 200},
    )

    assert seen and all(s.params.get("fetch_depth") == 200 for s in seen)


def test_the_engine_shape_gets_a_probe_too(tmp_path):
    """The probe was gated on `with_rows` until 2026-08-21, so an ENGINE-only
    process got none -- and the probe's job is process-global CUDA/BLAS
    initialisation, which has nothing to do with which rows are resident.
    Measured cost of that gating: the engine topology's first real query came
    back at 657 ms against a ~160 ms steady state, while the row-reading arm
    beside it had been probed (`data/results/serving_concurrency.md`).

    There is no Qdrant server here, so the probe cannot succeed -- what is
    pinned is that it is ATTEMPTED, i.e. reported as a failure rather than
    skipped in silence."""
    infos = _indices(tmp_path)
    report = warm_serving_caches(
        infos, "qdrant_hybrid", route_combo=_TWO_ROUTES, with_rows=False
    )
    assert any(f.get("route") == "(probe retrieval)" for f in report["failures"]), (
        "the engine shape skipped its probe entirely"
    )
    assert report["probe_ms"] is None
    # and it is still the engine cache shape, not the row-reading one
    assert all(e["with_embeddings"] is False for e in index_cache_info()["entries"])


def test_probe_retrieval_false_still_skips_it_for_the_engine_shape(tmp_path):
    """The escape hatch must survive the ungating, or the only way to warm
    without a probe is gone."""
    infos = _indices(tmp_path)
    report = warm_serving_caches(
        infos, "qdrant_hybrid", route_combo=_TWO_ROUTES,
        with_rows=False, probe_retrieval=False,
    )
    assert not any(f.get("route") == "(probe retrieval)" for f in report["failures"])
    assert report["probe_ms"] is None
