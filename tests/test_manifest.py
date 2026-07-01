"""Cycle 6 — the manifest records the resolved combo (incl loader) + provenance."""
from __future__ import annotations

from rag_lab.combos import enumerate_build_combos
from rag_lab.config import CorpusSpec, ExperimentConfig, StrategySpec
from rag_lab.manifest import build_manifest
from rag_lab.schema import Resolution


def _config():
    return ExperimentConfig(
        experiment_name="exp",
        corpus=CorpusSpec(input_dir="x"),
        output_dir="o",
        loaders=[StrategySpec(type="plain")],
        chunkers=[StrategySpec(type="fixed_size", params={"chunk_size": 100})],
        embedders=[StrategySpec(type="hashing")],
    )


def test_manifest_records_resolved_combo_and_provenance():
    config = _config()
    combo = enumerate_build_combos(config)[0]
    resolutions = [Resolution(resolution_id="r1", source_path="r1.md", raw_text="hi")]

    manifest = build_manifest(config, combo, resolutions)

    assert manifest["experiment_name"] == "exp"
    assert manifest["combo"]["loader"]["type"] == "plain"  # loader identity captured
    assert manifest["combo"]["chunker"]["params"]["chunk_size"] == 100
    assert manifest["seed"] == 42
    assert manifest["docset_hash"]
    # provenance keys present (values are environment-dependent)
    assert "git_commit" in manifest
    assert "timestamp" in manifest


def test_docset_hash_changes_with_content():
    config = _config()
    combo = enumerate_build_combos(config)[0]
    a = build_manifest(config, combo, [Resolution(resolution_id="r", source_path="r.md", raw_text="A")])
    b = build_manifest(config, combo, [Resolution(resolution_id="r", source_path="r.md", raw_text="B")])
    assert a["docset_hash"] != b["docset_hash"]
