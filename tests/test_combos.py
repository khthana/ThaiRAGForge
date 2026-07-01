"""Cycles 3–4 — build-combo enumeration (cartesian product / paired zip)."""
from __future__ import annotations

import pytest

from rag_lab.combos import enumerate_build_combos
from rag_lab.config import CorpusSpec, ExperimentConfig, StrategySpec


def _cfg(run_mode, n_loaders, n_chunkers, n_embedders):
    return ExperimentConfig(
        experiment_name="e",
        corpus=CorpusSpec(input_dir="x"),
        output_dir="o",
        loaders=[StrategySpec(type="plain") for _ in range(n_loaders)],
        chunkers=[
            StrategySpec(type="fixed_size", params={"chunk_size": 10 * (i + 1)})
            for i in range(n_chunkers)
        ],
        embedders=[
            StrategySpec(type="hashing", params={"dim": 128 * (i + 1)})
            for i in range(n_embedders)
        ],
        run_mode=run_mode,
    )


def test_cartesian_is_the_product():
    combos = enumerate_build_combos(_cfg("cartesian", 1, 2, 2))
    assert len(combos) == 4
    assert len({c.id for c in combos}) == 4  # each combo is distinct


def test_paired_zips_equal_length_lists():
    combos = enumerate_build_combos(_cfg("paired", 2, 2, 2))
    assert len(combos) == 2  # zipped, not 8


def test_paired_requires_equal_lengths():
    with pytest.raises(ValueError):
        enumerate_build_combos(_cfg("paired", 1, 2, 2))
